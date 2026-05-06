const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const json_grammar = @import("json_grammar.zig");
const json_schema = @import("json_schema.zig");
const token_mask = @import("token_mask.zig");
const io_util = @import("io_util.zig");
const pld_index = @import("pld_index.zig");
const drafter_mod = @import("drafter.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const SSMCacheEntrySnapshot = transformer_mod.SSMCacheEntrySnapshot;
const ssmSnapshot = transformer_mod.ssmSnapshot;
const ssmSnapshotDeinit = transformer_mod.ssmSnapshotDeinit;
const ssmRestore = transformer_mod.ssmRestore;
const DrafterModel = drafter_mod.DrafterModel;

/// Grammar-constrained sampling state. The caller owns `grammar`, `token_bytes`,
/// and `mask_buf`; the generator only reads them. `mask_buf.len` must equal
/// `token_bytes.bytes.len` (the tokenizer's vocab size).
pub const Constraint = struct {
    grammar: *json_grammar.Grammar,
    token_bytes: *const token_mask.TokenBytes,
    mask_buf: []bool,
};

/// RAII bundle for grammar-constrained sampling. Owns the parsed schema,
/// grammar state machine, and per-step mask buffer. The embedded `Constraint`
/// holds pointers/slices into the surrounding struct, so this struct must NOT
/// be moved after `initFromValue`. Construct via `var sc: SchemaConstraint =
/// undefined; try sc.initFromValue(...);` and pass `&sc.constraint` to
/// `SamplingParams`.
pub const SchemaConstraint = struct {
    schema: json_schema.Schema,
    grammar: json_grammar.Grammar,
    mask_buf: []bool,
    constraint: Constraint,
    allocator: std.mem.Allocator,

    /// Initialize in-place from a JSON schema value. On failure, any partial
    /// allocations made during this call are freed and the struct is left
    /// undefined (do not call `deinit`).
    pub fn initFromValue(
        self: *SchemaConstraint,
        allocator: std.mem.Allocator,
        schema_value: std.json.Value,
        token_bytes: *const token_mask.TokenBytes,
    ) !void {
        self.allocator = allocator;
        self.schema = try json_schema.parse(allocator, schema_value);
        errdefer self.schema.deinit();

        self.grammar = try json_grammar.Grammar.init(allocator, &self.schema);
        errdefer self.grammar.deinit();

        self.mask_buf = try allocator.alloc(bool, token_bytes.bytes.len);
        errdefer allocator.free(self.mask_buf);

        self.constraint = .{
            .grammar = &self.grammar,
            .token_bytes = token_bytes,
            .mask_buf = self.mask_buf,
        };
    }

    pub fn deinit(self: *SchemaConstraint) void {
        self.allocator.free(self.mask_buf);
        self.grammar.deinit();
        self.schema.deinit();
    }
};

/// Per-token logprob info (OpenAI format).
pub const TokenLogprob = struct {
    token_id: u32,
    logprob: f32,
};

/// Logprob result for a single generated token.
pub const LogprobResult = struct {
    token_logprob: f32, // logprob of the chosen token
    top_logprobs: []TokenLogprob, // top N alternatives (caller must free)
};

/// Sampling parameters for token generation.
pub const SamplingParams = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0, // 0 = disabled
    repeat_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0, // 0.0 = disabled
    seed: ?u64 = null,
    /// When non-null, generation is constrained to outputs that satisfy the
    /// grammar at byte level. Forces a synchronous sampling path (no lazy
    /// pipeline) since grammar advancement requires the realized token id.
    constraint: ?*Constraint = null,
};

/// Generation result (for non-streaming use).
pub const GenerationResult = struct {
    text: []u8,
    token_ids: []u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    prefill_tps: f64,
    decode_tps: f64,
    logprobs: ?[]LogprobResult = null, // per-token logprobs (caller must free)
};

/// Step-based generator. Call `init` to prefill, then `next` per token.
/// Uses a fully-lazy async pipeline matching mlx-lm: sample + next forward are
/// built as a single lazy computation graph, async_eval'd together. The GPU
/// never idles between token generation steps.
pub const Generator = struct {
    xfm: *Transformer,
    tok: *const Tokenizer,
    next_token_id: u32,
    step: u32,
    max_tokens: u32,
    sampling: SamplingParams,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    done: bool,
    eos_token_ids: []const u32,
    generated_ids: std.ArrayList(u32),
    consecutive_pad: u32 = 0, // count of consecutive token-0 (pad) generations
    timeout_ns: u64, // 0 = no timeout
    timer: io_util.Stopwatch,
    logprobs_n: u32 = 0, // 0 = disabled, >0 = number of top_logprobs to return
    last_logprob: ?LogprobResult = null, // logprob result for the most recently returned token
    // Async pipeline state: pre-computed forward pass logits for next decode step
    pending_logits: mlx.mlx_array = .{},
    has_pending_logits: bool = false,
    // Deferred token: lazy array from async pipeline, eval'd at start of next iteration
    pending_token: mlx.mlx_array = .{},
    has_pending_token: bool = false,

    // ── MTP (Multi-Token Prediction) state ──
    // Set by handler when `enable_mtp` is true on the request and the model
    // has an MTP head. When set, callers use `nextMtp` instead of `next`.
    mtp_enabled: bool = false,
    // Pre-final-norm hidden state at the last produced token's position. Owned
    // by the Generator (freed in `deinit`). Captured by `forwardCaptureHidden`
    // during prefill final-token forward + every nextMtp verify forward.
    mtp_last_hidden: mlx.mlx_array = .{},
    has_mtp_last_hidden: bool = false,
    // Statistics for benchmarking / debug logs.
    mtp_attempted: u64 = 0,
    mtp_accepted: u64 = 0,
    /// PRNG for the MTP / PLD stochastic-verify accept test (probability-ratio
    /// requires a uniform draw per draft step). Seeded from `sampling.seed`
    /// when set, otherwise from system time at init.
    prng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(0),

    // ── PLD (Prompt Lookup Decoding) state ──
    // Owned copy of the input prompt ids — needed because PLD's n-gram lookup
    // table is `prompt + generated_ids`, and the caller-supplied `prompt_ids`
    // slice is freed after `init` returns. `generated_ids` (above) tracks
    // post-prefill tokens; `prompt_ids_owned` is the immutable prefix.
    prompt_ids_owned: []u32 = &.{},
    /// Allocator that owns `prompt_ids_owned`. Stored so `deinit` can free it
    /// without requiring callers to thread the allocator a second time. (Other
    /// owned slices are freed via the `allocator` argument to `deinit` for
    /// historical reasons; this one is set during `initWithOptions`.)
    prompt_ids_alloc: ?std.mem.Allocator = null,
    /// Stats for PLD benchmark logging. `pld_attempted` counts every step
    /// where lookup found a candidate (so a verify forward ran);
    /// `pld_accepted_tokens` is the cumulative number of *drafted* tokens
    /// (not including the always-accepted t1) that were successfully verified.
    pld_attempted: u64 = 0,
    pld_accepted_tokens: u64 = 0,

    // ── Gemma 4 assistant drafter state ──
    // External drafter model (cross-attends into target's KV). When
    // `drafter != null`, callers use `nextDrafter` instead of `next`. The
    // drafter is owned by the server (loaded once at startup); the Generator
    // only holds a non-owning pointer.
    drafter: ?*DrafterModel = null,
    /// Number of tokens proposed per round (= drafter forwards + 1 verify token).
    /// Defaults to 4 (3 drafter steps + 1 t1 prepend → length-4 verify).
    drafter_block_size: u32 = 4,
    /// Stats: count of nextDrafter calls that ran a verify forward.
    drafter_attempted: u64 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    drafter_accepted_tokens: u64 = 0,

    /// Prefill the prompt and prepare for token-by-token generation.
    /// Backwards-compatible — prefer `initWithOptions` for new callers.
    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling: SamplingParams,
        eos_token_ids: []const u32,
    ) !Generator {
        return initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{});
    }

    pub const InitOptions = struct {
        /// Enable MTP draft+verify in the decode loop. Requires `xfm.mtp_layers != null`.
        /// When true, init's final-token prefill forward captures the pre-final-norm
        /// hidden state into `Generator.mtp_last_hidden` so the first `nextMtp` call
        /// has a starting point.
        mtp_enabled: bool = false,
        /// No-op — kept for source compatibility. PLD now reuses the same lazy
        /// pre-forward init path as the regular generator (cache lands at
        /// `prompt_len + 1` with t1 in cache and `pending_logits = forward(t1)`).
        /// `nextPld` consumes from that pending state on every step, matching
        /// `Generator.next`'s overlap on the cold path.
        pld_enabled: bool = false,
        /// Enable Gemma 4 assistant drafter. When set, `drafter` must be
        /// non-null and already `bind()`-ed to `xfm`. Init's prefill final-token
        /// forward captures the post-final-norm hidden state into
        /// `Generator.mtp_last_hidden` (reused for the drafter's first-step
        /// h_prev — same buffer, different semantics — see comment in
        /// `nextDrafter`). Same lazy-pre-forward skip semantics as MTP/PLD.
        drafter_enabled: bool = false,
        /// Non-owning pointer to the loaded drafter (must be non-null when
        /// `drafter_enabled` is true).
        drafter: ?*DrafterModel = null,
        /// Number of tokens per draft round. Default 4 (3 drafter steps +
        /// 1 t1 prepend → length-4 verify forward).
        drafter_block_size: u32 = 4,
    };

    pub fn initWithOptions(
        io: std.Io,
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling: SamplingParams,
        eos_token_ids: []const u32,
        options: InitOptions,
    ) !Generator {
        const s = xfm.s;

        const ids_i32 = try allocator.alloc(i32, prompt_ids.len);
        defer allocator.free(ids_i32);
        for (prompt_ids, 0..) |id, i| {
            ids_i32[i] = @intCast(id);
        }

        // Clone the prompt for the lifetime of the Generator. PLD's n-gram
        // lookup needs `prompt + generated`, and `prompt_ids` (caller-owned)
        // can be freed before `nextPld` runs. Allocated up front so init's
        // errdefer paths don't have to track partial state.
        const prompt_owned = try allocator.dupe(u32, prompt_ids);
        errdefer allocator.free(prompt_owned);

        // Split prefill: process first N-1 tokens (cache-only, skip lm_head eval),
        // then the last token (produces logits for sampling). This mirrors mlx-lm's
        // generate_step which avoids the expensive lm_head projection over the full
        // sequence length. For vocab_size=262144, skipping lm_head on N-1 tokens
        // avoids a [N-1, hidden] @ [hidden, 262144] matmul.
        //
        // Chunked prefill: large prompts are processed in PREFILL_CHUNK-sized pieces
        // to bound peak activation memory. Each chunk fills KV cache entries for its
        // positions, gets eval'd, and intermediates are freed before the next chunk.
        // Without chunking, Gemma-4 MoE's 2 MLPs × 4 stacked layers can spike to
        // ~20 GB of activations alone on a 50k-token prompt, causing Metal OOM.
        // Vision requests skip chunking since image token positions must be visible
        // in a single forward pass for spliceVisionEmbeddings to work correctly.
        const PREFILL_CHUNK: usize = 8192;
        if (prompt_ids.len > 1) {
            const prefix_len = prompt_ids.len - 1;
            const has_vision = xfm.vision_embeddings != null;
            const chunk_size = if (has_vision) prefix_len else PREFILL_CHUNK;

            var pos: usize = 0;
            while (pos < prefix_len) {
                const end = @min(pos + chunk_size, prefix_len);
                const chunk_len: c_int = @intCast(end - pos);
                const chunk_shape = [_]c_int{ 1, chunk_len };
                const chunk_input = mlx.mlx_array_new_data(@ptrCast(&ids_i32[pos]), &chunk_shape, 2, .int32);
                defer _ = mlx.mlx_array_free(chunk_input);

                const chunk_logits = try xfm.forward(chunk_input);
                _ = mlx.mlx_array_free(chunk_logits);

                // Eval KV cache — materializes this chunk's K/V, frees activation graph
                {
                    const eval_vec = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(eval_vec);
                    for (xfm.cache.entries) |*entry| {
                        if (!entry.initialized) continue;
                        _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                        _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
                    }
                    _ = mlx.mlx_eval(eval_vec);
                }
                _ = mlx.mlx_clear_cache();

                pos = end;
            }
        }

        // Process last token (or single token for len=1) — this applies lm_head
        // on just 1 token, producing the logits we need for sampling.
        const last_shape = [_]c_int{ 1, 1 };
        const last_idx = prompt_ids.len - 1;
        const last_input = mlx.mlx_array_new_data(&ids_i32[last_idx], &last_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(last_input);

        // MTP enabled at init: capture the pre-final-norm hidden state from
        // this prefill forward so the first `nextMtp` call has h_N to draft
        // from. Also disables the lazy pre-forward of the first sampled token
        // below — MTP semantics require the cache to be at exactly `prompt_len`
        // (last prompt token forwarded, first sampled token NOT yet forwarded)
        // before nextMtp's verify forward runs over `[first_sample, draft]`.
        //
        // The drafter (Gemma 4 assistant) uses the same captured hidden as its
        // first-step h_prev — same buffer, same `forwardCaptureHidden` path,
        // different downstream semantics. Reuse the existing `mtp_last_hidden`
        // slot rather than introducing a parallel field.
        const mtp_active = options.mtp_enabled and xfm.mtp_layers != null;
        const drafter_active = options.drafter_enabled and options.drafter != null;
        const need_capture = mtp_active or drafter_active;
        var captured_hidden: mlx.mlx_array = mlx.mlx_array_new();
        var has_captured_hidden = false;
        const logits = if (need_capture) blk: {
            has_captured_hidden = true;
            break :blk try xfm.forwardCaptureHidden(last_input, &captured_hidden);
        } else try xfm.forward(last_input);
        errdefer if (has_captured_hidden) {
            _ = mlx.mlx_array_free(captured_hidden);
        };

        // Constrained generation skips the lazy first-sample fast path: we cannot
        // sample the first token until we have applied the grammar mask, and we
        // cannot pipeline because grammar advancement depends on the realized id.
        if (sampling.constraint != null) {
            var gen = Generator{
                .xfm = xfm,
                .tok = tok,
                .next_token_id = 0,
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = sampling,
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .mtp_enabled = false, // grammar-constrained + MTP is not supported in v1
                .mtp_last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
                .has_mtp_last_hidden = has_captured_hidden,
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
            };
            gen.pending_logits = logits;
            gen.has_pending_logits = true;
            return gen;
        }

        // MTP / drafter path: sample synchronously and DO NOT pre-forward
        // the sampled token. The first nextMtp / nextDrafter call needs the
        // cache at exactly prompt_len (last prompt token forwarded; first
        // sampled token deferred). The lazy pre-forward path below would
        // over-advance the cache and corrupt every verify forward.
        //
        // PLD does NOT take this branch — it now uses the same lazy pre-forward
        // path as `Generator.next`, with cache landing at `prompt_len + 1`
        // (first sampled token IN cache). `nextPld` consumes the pending
        // pipeline on every step so the cold path stays fully overlapped.
        if (mtp_active or drafter_active) {
            const sample_lazy = sampleTokenLazy(logits, sampling, s);
            _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(sample_lazy));
            var first_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&first_val, sample_lazy));
            _ = mlx.mlx_array_free(sample_lazy);

            const gen = Generator{
                .xfm = xfm,
                .tok = tok,
                .next_token_id = @intCast(first_val),
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = sampling,
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .mtp_enabled = mtp_active,
                .mtp_last_hidden = if (need_capture) captured_hidden else mlx.mlx_array_new(),
                .has_mtp_last_hidden = need_capture,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
                .drafter = if (drafter_active) options.drafter else null,
                .drafter_block_size = options.drafter_block_size,
            };
            // pending_logits/pending_token left empty — the lazy pipeline is
            // skipped under MTP / PLD / drafter. The speculative `next*` paths
            // drive every subsequent step with predictable cache offset.
            return gen;
        }

        // Non-MTP: sample first token lazily, then build the next forward pass
        const lazy_token = sampleTokenLazy(logits, sampling, s);
        _ = mlx.mlx_array_free(logits);

        const next_logits = try lazyForward(xfm, lazy_token);

        // Async-eval the decode pipeline (single-token graphs, much smaller)
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lazy_token);
            _ = mlx.mlx_vector_array_append_value(eval_vec, next_logits);
            _ = mlx.mlx_async_eval(eval_vec);
        }

        // Sync to get the first token value
        try mlx.check(mlx.mlx_array_eval(lazy_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
        _ = mlx.mlx_array_free(lazy_token);

        var gen = Generator{
            .xfm = xfm,
            .tok = tok,
            .next_token_id = @intCast(val),
            .step = 0,
            .max_tokens = max_tokens,
            .sampling = sampling,
            .prompt_tokens = @intCast(prompt_ids.len),
            .completion_tokens = 0,
            .finish_reason = "length",
            .done = false,
            .eos_token_ids = eos_token_ids,
            .generated_ids = std.ArrayList(u32).empty,
            .timeout_ns = 0,
            .timer = io_util.Stopwatch.init(io),
            .mtp_enabled = false,
            .mtp_last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
            .has_mtp_last_hidden = has_captured_hidden,
            .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
            .prompt_ids_owned = prompt_owned,
            .prompt_ids_alloc = allocator,
        };

        gen.pending_logits = next_logits;
        gen.has_pending_logits = true;

        return gen;
    }

    pub fn deinit(self: *Generator, allocator: std.mem.Allocator) void {
        if (self.last_logprob) |*lp| {
            allocator.free(lp.top_logprobs);
        }
        if (self.has_pending_logits) {
            _ = mlx.mlx_array_free(self.pending_logits);
            self.has_pending_logits = false;
        }
        if (self.has_pending_token) {
            _ = mlx.mlx_array_free(self.pending_token);
            self.has_pending_token = false;
        }
        if (self.has_mtp_last_hidden) {
            _ = mlx.mlx_array_free(self.mtp_last_hidden);
            self.has_mtp_last_hidden = false;
        }
        if (self.prompt_ids_alloc) |a| {
            a.free(self.prompt_ids_owned);
            self.prompt_ids_owned = &.{};
            self.prompt_ids_alloc = null;
        }
        self.generated_ids.deinit(allocator);
    }

    /// Resolve the deferred pending token: eval the lazy array and extract the u32 value.
    /// This is called at the START of each iteration, giving the GPU maximum time
    /// to compute since the async_eval at the END of the previous iteration.
    fn resolvePendingToken(self: *Generator) !void {
        if (!self.has_pending_token) return;
        try mlx.check(mlx.mlx_array_eval(self.pending_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, self.pending_token));
        _ = mlx.mlx_array_free(self.pending_token);
        self.has_pending_token = false;
        self.next_token_id = @intCast(val);
    }

    /// Result of one `nextMtp` step. Yields up to 2 token IDs (1 if the draft
    /// was rejected, 2 on accept). `accepted_draft` lets the caller account for
    /// acceptance-rate stats.
    pub const MtpStepResult = struct {
        first: u32,
        second: ?u32 = null,
        accepted_draft: bool,
    };

    /// MTP draft+verify decode step. Returns 1 or 2 tokens per call.
    /// Algorithm:
    ///   1. Draft: `mtpForward(mtp_last_hidden, next_token_id)` → sample `draft_id`.
    ///   2. Snapshot KV (and SSM for hybrid models) at offset N+1.
    ///   3. Verify: main `forward([next_token_id, draft_id])` (length-2) →
    ///      logits at both positions + new pre-final-norm hidden state at pos 1.
    ///   4. Accept iff `argmax(verify_logits[..,1,..]) == draft_id` (greedy verify).
    ///   5. Reject path: restore snapshot, re-forward `[next_token_id]` alone
    ///      with hidden capture, sample fallback from those logits.
    ///
    /// Caller must check `self.done` and stop conditions BETWEEN tokens (the
    /// pair returned here may include an EOS as the second; the caller is
    /// expected to honor it).
    pub fn nextMtp(self: *Generator, allocator: std.mem.Allocator) !?MtpStepResult {
        _ = allocator;
        if (self.done) return null;
        std.debug.assert(self.mtp_enabled);
        std.debug.assert(self.has_mtp_last_hidden);
        std.debug.assert(self.xfm.mtp_layers != null);

        const xfm = self.xfm;
        const s = xfm.s;
        const t1: u32 = self.next_token_id; // already-decided token at position N+1

        // ── Phase 1: Draft ──
        try xfm.resetMtpCache();
        const t1_i32: i32 = @intCast(t1);
        const t1_shape = [_]c_int{ 1, 1 };
        const t1_input = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(t1_input);

        const draft_logits = try xfm.mtpForward(self.mtp_last_hidden, t1_input, 0);
        defer _ = mlx.mlx_array_free(draft_logits);

        // Sample draft via the existing lazy sampler, then realize.
        const draft_lazy = sampleTokenLazy(draft_logits, self.sampling, s);
        try mlx.check(mlx.mlx_array_eval(draft_lazy));
        var draft_val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&draft_val, draft_lazy));
        _ = mlx.mlx_array_free(draft_lazy);
        const t2_draft: u32 = @intCast(draft_val);

        // ── Phase 2: Snapshot KV + SSM ──
        var kv_snap = try xfm.cache.snapshot();
        defer kv_snap.deinit();
        var ssm_snaps: ?[]SSMCacheEntrySnapshot = null;
        defer if (ssm_snaps) |snaps| {
            for (snaps) |*sn| ssmSnapshotDeinit(sn);
            xfm.allocator.free(snaps);
        };
        if (xfm.ssm_entries) |entries| {
            const out = try xfm.allocator.alloc(SSMCacheEntrySnapshot, entries.len);
            for (entries, 0..) |*entry, i| out[i] = ssmSnapshot(entry);
            ssm_snaps = out;
        }
        const moe_seq_offset_snap = xfm.moe_seq_offset;

        // ── Phase 3: Verify (length-2 forward) ──
        const pair_i32 = [_]i32{ t1_i32, draft_val };
        const pair_shape = [_]c_int{ 1, 2 };
        const pair_input = mlx.mlx_array_new_data(&pair_i32, &pair_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(pair_input);

        var new_hidden = mlx.mlx_array_new();
        // `forwardCaptureHidden` captures the hidden state at the LAST position
        // of the input — i.e., at position 1 (= prediction site for t̂_{N+2}).
        const verify_logits = try xfm.forwardCaptureHidden(pair_input, &new_hidden);
        // Lifetime: verify_logits is freed below; new_hidden is moved into
        // self.mtp_last_hidden on accept, freed otherwise.

        self.mtp_attempted += 1;

        // ── Phase 4: Decide accept/reject ──
        // Two paths: greedy (temp ≤ 0.01) compares argmax of verify_logits[0]
        // with the drafted token; stochastic (temp > 0.01) uses the
        // probability-ratio test from Leviathan et al. speculative decoding —
        // accept_prob = min(1, p[draft] / q[draft]) where p is the main
        // model's distribution and q is the draft distribution. On reject,
        // sample a corrected token from `max(p - q, 0)` (residual
        // distribution) so the marginal output distribution stays p.
        //
        // Stochastic verify ALWAYS yields > greedy acceptance for the same
        // model: drafts with high p/q ratio accept reliably even when the
        // argmax differs.
        const vl_shape = mlx.getShape(verify_logits);
        const slice_strides = [_]c_int{ 1, 1, 1 };
        const slice0_start = [_]c_int{ 0, 0, 0 };
        const slice0_stop = [_]c_int{ vl_shape[0], 1, vl_shape[2] };
        var pos0_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos0_logits);
        try mlx.check(mlx.mlx_slice(&pos0_logits, verify_logits, &slice0_start, 3, &slice0_stop, 3, &slice_strides, 3, s));

        const slice1_start = [_]c_int{ 0, 1, 0 };
        const slice1_stop = [_]c_int{ vl_shape[0], 2, vl_shape[2] };
        var pos1_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pos1_logits);
        try mlx.check(mlx.mlx_slice(&pos1_logits, verify_logits, &slice1_start, 3, &slice1_stop, 3, &slice_strides, 3, s));
        _ = mlx.mlx_array_free(verify_logits);

        const stochastic = self.sampling.temperature > 0.01;
        var accept: bool = false;
        var t2_fallback: u32 = 0; // populated on reject

        if (stochastic) {
            // Build target/draft probability distributions over the vocab.
            const target_probs = try mtpProbsAtLastPos(pos0_logits, self.sampling, s);
            defer _ = mlx.mlx_array_free(target_probs);
            const draft_probs_for_verify = try mtpProbsAtLastPos(draft_logits, self.sampling, s);
            defer _ = mlx.mlx_array_free(draft_probs_for_verify);

            const p_draft = try mtpProbAt(target_probs, t2_draft, s);
            const q_draft = try mtpProbAt(draft_probs_for_verify, t2_draft, s);
            const accept_prob: f32 = if (q_draft > 0) @min(1.0, p_draft / q_draft) else 0.0;
            const u: f32 = self.prng.random().float(f32);
            accept = (u < accept_prob);

            if (!accept) {
                t2_fallback = try mtpSampleResidual(target_probs, draft_probs_for_verify, s);
            }
        } else {
            // Greedy verify: compare argmax of pos0_logits with draft.
            var argmax_arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(argmax_arr);
            try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, pos0_logits, 2, false, s));
            try mlx.check(mlx.mlx_array_eval(argmax_arr));
            var argmax_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&argmax_val, argmax_arr));
            accept = (argmax_val == draft_val);
            if (!accept) {
                const fallback_lazy = sampleTokenLazy(pos0_logits, self.sampling, s);
                try mlx.check(mlx.mlx_array_eval(fallback_lazy));
                var fallback_val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&fallback_val, fallback_lazy));
                _ = mlx.mlx_array_free(fallback_lazy);
                t2_fallback = @intCast(fallback_val);
            }
        }

        if (accept) {
            // ── Accept path: cache already advanced to N+2 ──
            // Sample t_{N+3} from pos1_logits (becomes next iteration's t1).
            const next_lazy = sampleTokenLazy(pos1_logits, self.sampling, s);
            try mlx.check(mlx.mlx_array_eval(next_lazy));
            var next_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&next_val, next_lazy));
            _ = mlx.mlx_array_free(next_lazy);
            const t3_next: u32 = @intCast(next_val);

            // Replace mtp_last_hidden with new (position-1 hidden = h_{N+2}).
            if (self.has_mtp_last_hidden) _ = mlx.mlx_array_free(self.mtp_last_hidden);
            self.mtp_last_hidden = new_hidden;
            self.has_mtp_last_hidden = true;
            self.mtp_accepted += 1;
            self.next_token_id = t3_next;
            self.step += 2;
            self.completion_tokens += 2;
            return MtpStepResult{ .first = t1, .second = t2_draft, .accepted_draft = true };
        }

        // ── Reject path ──
        _ = mlx.mlx_array_free(new_hidden);

        // Roll back the cache to pre-verify state.
        try xfm.cache.restore(&kv_snap);
        if (ssm_snaps) |snaps| {
            for (xfm.ssm_entries.?, snaps) |*entry, *sn| try ssmRestore(entry, sn);
        }
        xfm.moe_seq_offset = moe_seq_offset_snap;

        // Re-forward [t1] alone to advance cache by 1 AND capture hidden_state at N+1
        // for the next nextMtp call.
        var new_hidden_after_t1 = mlx.mlx_array_new();
        const t1_logits = try xfm.forwardCaptureHidden(t1_input, &new_hidden_after_t1);
        _ = mlx.mlx_array_free(t1_logits);

        if (self.has_mtp_last_hidden) _ = mlx.mlx_array_free(self.mtp_last_hidden);
        self.mtp_last_hidden = new_hidden_after_t1;
        self.has_mtp_last_hidden = true;
        self.next_token_id = t2_fallback;
        self.step += 1;
        self.completion_tokens += 1;
        return MtpStepResult{ .first = t1, .second = null, .accepted_draft = false };
    }

    /// Result of one `nextPld` step. Yields 1..=(1+max_draft_len) tokens.
    /// Caller owns `tokens` (must `allocator.free` it).
    pub const PldStepResult = struct {
        /// Tokens to emit this step (always at least the already-decided t1).
        /// On a full-accept, contains [t1, ...all_drafts]. On partial accept j,
        /// contains [t1, draft[0..j]] (the corrected fallback is stored as the
        /// generator's pending `next_token_id`, NOT included here — same
        /// "pending becomes next-step's first" convention as `nextMtp`).
        tokens: []const u32,
        /// Number of *drafted* tokens accepted (not counting t1). 0..=draft_len.
        accepted_tokens: u32,
        /// Whether n-gram lookup found a candidate this step. False means PLD
        /// degraded to a single regular forward (no speculative work done).
        used_lookup: bool,
    };

    /// PLD draft+verify decode step. The draft comes from an n-gram lookup
    /// over `prompt_ids_owned ++ generated_ids`, NOT a model call — that's
    /// what makes PLD model-agnostic and cheap.
    ///
    /// `key_len` is the n-gram size used for matching (default 3). `draft_len`
    /// is the maximum number of speculative tokens to verify per step (default
    /// 5). Both are clamped to safe upper bounds internally.
    ///
    /// Returns `null` only when generation is already done. When no n-gram
    /// match exists (cold start, novel output), falls back to the regular
    /// `next()` path and returns a single-token result with `used_lookup=false`.
    pub fn nextPld(
        self: *Generator,
        allocator: std.mem.Allocator,
        draft_len: u32,
        key_len: u32,
    ) !?PldStepResult {
        if (self.done) return null;
        std.debug.assert(self.sampling.constraint == null); // PLD + grammar not supported
        std.debug.assert(self.logprobs_n == 0); // PLD + logprobs not supported

        const xfm = self.xfm;
        const s = xfm.s;


        // ── INVARIANT going INTO this call (matches `Generator.next`) ──
        //   cache.step = prompt_len + tokens_emitted + 1
        //   t1 = next_token_id (= "this step's emit"); already in cache from
        //   the previous step's lazy pre-forward.
        //   pending_logits = forward(t1) (cached, GPU done after async_eval).
        //   pending_token = lazy sampleTokenLazy(forward(t1)) — the lookahead
        //   candidate for "what comes after t1". (Empty on first call.)
        //
        // The cold path mirrors `Generator.next.next()`: we BUILD the next
        // step's lazy graph BEFORE resolving the previous step's pending_token,
        // so the GPU starts the next forward as soon as possible and the
        // resolve becomes a near-instant dependency-already-computed wait.
        // That's the property that recovers the novel-output regression.
        const t1: u32 = self.next_token_id;

        // Cap draft_len so the verify forward stays a small fixed cost.
        const max_draft: u32 = @min(draft_len, 15);
        const klen: u32 = @max(@as(u32, 1), key_len);

        // ── Phase 1: Lookup ──
        // committed = prompt + generated_ids + [t1]. Key = trailing klen
        // tokens (ends at t1). The lookup returns candidates for "what comes
        // after t1" — same key shape as the previous PLD implementation, so
        // n-gram match behavior is preserved.
        const prompt = self.prompt_ids_owned;
        const generated = self.generated_ids.items;
        const total_len = prompt.len + generated.len + 1;

        var committed = try allocator.alloc(u32, total_len);
        defer allocator.free(committed);
        @memcpy(committed[0..prompt.len], prompt);
        @memcpy(committed[prompt.len .. prompt.len + generated.len], generated);
        committed[total_len - 1] = t1;

        var draft_slice: ?[]const u32 = null;
        if (klen <= total_len - 1) {
            const key_start = total_len - klen;
            const key = committed[key_start..total_len];
            const lookup = pld_index.PldLookup{ .committed = committed, .key_len = klen };
            draft_slice = lookup.findMatch(key, max_draft);
        }
        if (draft_slice) |d| {
            if (d.len == 0) draft_slice = null;
        }

        const stochastic = self.sampling.temperature > 0.01;

        // ── Phase 2: NO match — fast cold path matching `Generator.next` ──
        // No need to look at draft[0] / lookahead, so we can build the next
        // step's lazy graph FIRST, async_eval, THEN resolve pending_token —
        // exactly the order that lets the GPU keep working while CPU handles
        // the just-returned token.
        if (draft_slice == null and self.has_pending_logits and self.has_pending_token and self.step + 1 < self.max_tokens) {
            const step_logits = self.pending_logits;
            self.has_pending_logits = false;

            // lazy_token = lookahead candidate (= what to commit as new t1).
            // It's `self.pending_token` already realized lazy from prev step
            // (the GPU finished it during async_eval of THIS step's prior
            // setup, so its evaluation will be instant). We DON'T re-sample
            // here — pending_token already IS the lazy sample.
            const lazy_lookahead = self.pending_token;
            self.has_pending_token = false;
            // Free the OLD pending_logits (= forward(t1)); we extracted what we
            // needed above. step_logits owned by us now.
            _ = mlx.mlx_array_free(step_logits);

            // Build NEXT-STEP graph using the lazy lookahead as input. This
            // forwards lookahead's value once it's realized — but lazyForward
            // doesn't need a realized int, just the lazy mlx_array.
            if (lazyForward(xfm, lazy_lookahead)) |next_logits| {
                // Sample the new pending_token lazily from next_logits (=
                // forward(lookahead) = predict "after-lookahead").
                const lazy_token = sampleTokenLazy(next_logits, self.sampling, s);

                // async_eval BOTH the lazy lookahead realization AND the
                // next-step's forward + sample as one graph. The GPU now
                // computes lookahead as a *dependency* of next_logits, so by
                // the time we resolve below, both are done.
                const arr = [_]mlx.mlx_array{ lazy_lookahead, lazy_token, next_logits };
                const vec = mlx.mlx_vector_array_new_data(&arr, 3);
                _ = mlx.mlx_async_eval(vec);
                _ = mlx.mlx_vector_array_free(vec);

                // Resolve lookahead — INSTANT (GPU computed it as dependency).
                try mlx.check(mlx.mlx_array_eval(lazy_lookahead));
                var lv: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&lv, lazy_lookahead));
                _ = mlx.mlx_array_free(lazy_lookahead);
                const new_t1: u32 = @intCast(lv);

                // Commit t1.
                try self.generated_ids.append(allocator, t1);
                self.completion_tokens += 1;
                self.step += 1;
                if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

                self.pending_token = lazy_token;
                self.has_pending_token = true;
                self.pending_logits = next_logits;
                self.has_pending_logits = true;
                self.next_token_id = new_t1;

                const tokens = try allocator.alloc(u32, 1);
                tokens[0] = t1;
                return PldStepResult{
                    .tokens = tokens,
                    .accepted_tokens = 0,
                    .used_lookup = false,
                };
            } else |_| {
                // lazyForward failed — fall through to slow path.
                try mlx.check(mlx.mlx_array_eval(lazy_lookahead));
                var lv: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&lv, lazy_lookahead));
                _ = mlx.mlx_array_free(lazy_lookahead);
                const new_t1: u32 = @intCast(lv);
                try self.generated_ids.append(allocator, t1);
                self.completion_tokens += 1;
                self.step += 1;
                self.next_token_id = new_t1;
                const tokens = try allocator.alloc(u32, 1);
                tokens[0] = t1;
                return PldStepResult{ .tokens = tokens, .accepted_tokens = 0, .used_lookup = false };
            }
        }

        // ── Slow path: realize lookahead before deciding cold vs verify ──
        // We need the int value of lookahead either to compare against draft[0]
        // (greedy) or to pick the residual sample (stochastic). On first call
        // there's no pending_token; sample synchronously instead.
        var lookahead: u32 = 0;
        const has_lookahead = self.has_pending_token;
        if (has_lookahead) {
            try mlx.check(mlx.mlx_array_eval(self.pending_token));
            var lv: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&lv, self.pending_token));
            lookahead = @intCast(lv);
            _ = mlx.mlx_array_free(self.pending_token);
            self.has_pending_token = false;
        } else if (self.has_pending_logits) {
            // First call after init.
            const la_lazy = sampleTokenLazy(self.pending_logits, self.sampling, s);
            try mlx.check(mlx.mlx_array_eval(la_lazy));
            var lv: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&lv, la_lazy));
            lookahead = @intCast(lv);
            _ = mlx.mlx_array_free(la_lazy);
        }

        // First-position acceptance test (only when we have a draft).
        var first_accept: bool = false;
        if (draft_slice) |draft_first| {
            if (stochastic) {
                std.debug.assert(self.has_pending_logits);
                const target_p = try mtpProbsAtLastPos(self.pending_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(target_p);
                const p_draft = try mtpProbAt(target_p, draft_first[0], s);
                const accept_prob: f32 = @min(1.0, p_draft);
                const u: f32 = self.prng.random().float(f32);
                first_accept = u < accept_prob;
            } else {
                first_accept = (lookahead == draft_first[0]);
            }
        }

        // Cold path (no match, first-position miss, OR last token before
        // max_tokens — we don't pre-launch the lazy pipeline on the very last
        // step). Emit t1, set up next-step pending if more steps remain.
        if (draft_slice == null or !first_accept) {
            var new_t1: u32 = lookahead;
            if (stochastic and draft_slice != null and !first_accept) {
                std.debug.assert(self.has_pending_logits);
                const draft_first = draft_slice.?;
                const vocab_size = mlx.getShape(self.pending_logits)[2];
                const probs = try mtpProbsAtLastPos(self.pending_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                const onehot = try pldOneHotRow(draft_first[0], vocab_size, s);
                defer _ = mlx.mlx_array_free(onehot);
                new_t1 = try mtpSampleResidual(probs, onehot, s);
            }

            if (self.has_pending_logits) {
                _ = mlx.mlx_array_free(self.pending_logits);
                self.has_pending_logits = false;
            }

            try self.generated_ids.append(allocator, t1);
            self.completion_tokens += 1;
            self.step += 1;
            if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

            if (self.step < self.max_tokens) {
                const new_t1_i32: i32 = @intCast(new_t1);
                const tok_shape = [_]c_int{ 1, 1 };
                const tok_input = mlx.mlx_array_new_data(&new_t1_i32, &tok_shape, 2, .int32);
                defer _ = mlx.mlx_array_free(tok_input);

                const next_logits = try xfm.forward(tok_input); // cache.step += 1
                const lazy_token = sampleTokenLazy(next_logits, self.sampling, s);

                const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
                const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                _ = mlx.mlx_async_eval(vec);
                _ = mlx.mlx_vector_array_free(vec);

                self.pending_token = lazy_token;
                self.has_pending_token = true;
                self.pending_logits = next_logits;
                self.has_pending_logits = true;
            }

            self.next_token_id = new_t1;

            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = t1;
            return PldStepResult{
                .tokens = tokens,
                .accepted_tokens = 0,
                .used_lookup = (draft_slice != null),
            };
        }

        // ── Phase 4: First-position accepted — run verify forward ──
        // Free pending_logits now (we already extracted what we needed). The
        // verify forward will produce per-position logits we'll use instead.
        if (self.has_pending_logits) {
            _ = mlx.mlx_array_free(self.pending_logits);
            self.has_pending_logits = false;
        }

        const draft = draft_slice.?;
        const m: u32 = @intCast(draft.len);

        // Snapshot KV + per-layer SSM + moe_seq_offset for partial-accept
        // rollback. Cache enters at cache.step = prompt_len + TE + 1.
        var kv_snap = try xfm.cache.snapshot();
        defer kv_snap.deinit();
        var ssm_snaps: ?[]SSMCacheEntrySnapshot = null;
        defer if (ssm_snaps) |snaps| {
            for (snaps) |*sn| ssmSnapshotDeinit(sn);
            xfm.allocator.free(snaps);
        };
        if (xfm.ssm_entries) |entries| {
            const out = try xfm.allocator.alloc(SSMCacheEntrySnapshot, entries.len);
            for (entries, 0..) |*entry, i| out[i] = ssmSnapshot(entry);
            ssm_snaps = out;
        }
        const moe_seq_offset_snap = xfm.moe_seq_offset;

        // Verify forward `[draft[0..m-1]]` of length m. cache.step at start =
        // prompt_len + TE + 1 (t1 in cache). After: + m. draft[i] sits at
        // slot prompt_len + TE + 1 + i — i.e., at position i+1 *past t1*,
        // which is exactly the slot a candidate for "the i-th token after t1"
        // should occupy.
        //
        // verify_logits[i] (i=0..m-1) = predicts the position right after
        // draft[i] = candidate for "the (i+1)-th token after t1." We compare
        // verify_logits[i] vs draft[i+1] for i=0..m-2 (m-1 comparisons),
        // verifying drafts beyond the first. The first draft was already
        // verified by `lookahead == draft[0]` (greedy) / accept test
        // (stochastic) using pending_logits.
        const seq_len: c_int = @intCast(m);
        const verify_input_buf = try allocator.alloc(i32, m);
        defer allocator.free(verify_input_buf);
        for (draft, 0..) |d, i| verify_input_buf[i] = @intCast(d);
        const verify_shape = [_]c_int{ 1, seq_len };
        const verify_input = mlx.mlx_array_new_data(verify_input_buf.ptr, &verify_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(verify_input);

        const verify_logits = try xfm.forward(verify_input);
        // verify_logits shape [1, m, V]. Sliced and freed below.
        self.pld_attempted += 1;

        const vl_shape = mlx.getShape(verify_logits);
        const slice_strides = [_]c_int{ 1, 1, 1 };

        // Slice per-position logits up front so we can sample the correction
        // from the original verify forward (cache state aligned) without
        // re-running forward, and re-use them for both stochastic accept tests
        // and the correction sample.
        const per_pos_logits = try allocator.alloc(mlx.mlx_array, m);
        defer {
            for (per_pos_logits) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(per_pos_logits);
        }
        for (per_pos_logits, 0..) |*slot, idx| {
            slot.* = mlx.mlx_array_new();
            const start = [_]c_int{ 0, @intCast(idx), 0 };
            const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
            try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
        }
        _ = mlx.mlx_array_free(verify_logits);

        // Walk drafts beyond the first. `accepted_beyond_first` counts
        // matches; total drafts accepted = 1 + accepted_beyond_first.
        var accepted_beyond_first: u32 = 0;
        if (m >= 2) {
            if (stochastic) {
                var i: u32 = 0;
                while (i < m - 1) : (i += 1) {
                    const target_p = try mtpProbsAtLastPos(per_pos_logits[i], self.sampling, s);
                    defer _ = mlx.mlx_array_free(target_p);
                    const p_draft = try mtpProbAt(target_p, draft[i + 1], s);
                    const accept_prob: f32 = @min(1.0, p_draft);
                    const u: f32 = self.prng.random().float(f32);
                    if (u >= accept_prob) break;
                    accepted_beyond_first += 1;
                }
            } else {
                var i: u32 = 0;
                while (i < m - 1) : (i += 1) {
                    var argmax_arr = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(argmax_arr);
                    try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, per_pos_logits[i], 2, false, s));
                    try mlx.check(mlx.mlx_array_eval(argmax_arr));
                    var argmax_val: i32 = 0;
                    try mlx.check(mlx.mlx_array_item_int32(&argmax_val, argmax_arr));
                    if (@as(u32, @intCast(argmax_val)) != draft[i + 1]) break;
                    accepted_beyond_first += 1;
                }
            }
        }

        // accepted_beyond_first ∈ [0, m-1]. Total drafts accepted = 1 +
        // accepted_beyond_first ∈ [1, m].
        const accepted_drafts: u32 = 1 + accepted_beyond_first;
        const full_accept = accepted_drafts == m;

        // The new pending t1' lives at slot = draft[accepted_drafts - 1] +
        // 1 in (post-t1) terms. Its source is per_pos_logits[accepted_drafts
        // - 1] (= predicts what comes after the last accepted draft). For a
        // partial accept, the rejected slot is accepted_drafts (=
        // accepted_beyond_first + 1), and we sample from residual against
        // draft[accepted_drafts] for stochastic.
        const correction_logits = per_pos_logits[accepted_drafts - 1];
        const new_t1: u32 = blk: {
            if (stochastic) {
                const probs = try mtpProbsAtLastPos(correction_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                if (!full_accept) {
                    // The rejected draft is draft[accepted_drafts] (the loop
                    // checked draft[accepted_beyond_first + 1] = draft[accepted_drafts]).
                    const rejected_idx = accepted_drafts;
                    const onehot = try pldOneHotRow(draft[rejected_idx], vl_shape[2], s);
                    defer _ = mlx.mlx_array_free(onehot);
                    break :blk try mtpSampleResidual(probs, onehot, s);
                } else {
                    break :blk try mtpSampleFromProbs(probs, s);
                }
            } else {
                const lazy = sampleTokenLazy(correction_logits, self.sampling, s);
                try mlx.check(mlx.mlx_array_eval(lazy));
                var v: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
                _ = mlx.mlx_array_free(lazy);
                break :blk @intCast(v);
            }
        };

        // ── Phase 5: Cache rollback on partial accept ──
        // After verify (length m), cache.step = prompt_len + TE + 1 + m. On
        // full accept this is exactly prompt_len + TE_new (no rollback).
        // On partial accept j (= accepted_beyond_first), we accepted only
        // accepted_drafts = j+1 drafts, so cache must land at prompt_len +
        // TE + 1 + (accepted_drafts) = prompt_len + TE_new. Roll back and
        // re-forward the accepted prefix.
        if (!full_accept) {
            try xfm.cache.restore(&kv_snap);
            if (ssm_snaps) |snaps| {
                for (xfm.ssm_entries.?, snaps) |*entry, *sn| try ssmRestore(entry, sn);
            }
            xfm.moe_seq_offset = moe_seq_offset_snap;

            const re_seq_len: c_int = @intCast(accepted_drafts);
            const re_input_buf = try allocator.alloc(i32, accepted_drafts);
            defer allocator.free(re_input_buf);
            for (draft[0..accepted_drafts], 0..) |d, i| re_input_buf[i] = @intCast(d);
            const re_shape = [_]c_int{ 1, re_seq_len };
            const re_input = mlx.mlx_array_new_data(re_input_buf.ptr, &re_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(re_input);
            const re_logits = try xfm.forward(re_input);
            _ = mlx.mlx_array_free(re_logits);
        }

        // ── Phase 6: Commit emitted tokens, set up next-step lazy pipeline ──
        // Tokens emitted: [t1, draft[0..accepted_drafts]] = 1 + accepted_drafts.
        const num_emit: u32 = 1 + accepted_drafts;
        const tokens = try allocator.alloc(u32, num_emit);
        tokens[0] = t1;
        for (draft[0..accepted_drafts], 0..) |d, i| tokens[1 + i] = d;

        try self.generated_ids.append(allocator, t1);
        for (draft[0..accepted_drafts]) |d| try self.generated_ids.append(allocator, d);

        self.pld_accepted_tokens += accepted_drafts;
        self.completion_tokens += num_emit;
        self.step += num_emit;
        if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

        // Set up next-step lazy pipeline so the regular Generator.next
        // invariant holds for the next nextPld call: pending_logits =
        // forward(new_t1), pending_token = sampleTokenLazy(...). cache.step
        // advances by 1 (= prompt_len + TE_new + 1).
        if (self.step < self.max_tokens) {
            const new_t1_i32: i32 = @intCast(new_t1);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&new_t1_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);

            const next_logits = try xfm.forward(tok_input); // cache.step += 1
            const lazy_token = sampleTokenLazy(next_logits, self.sampling, s);

            const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);

            self.pending_token = lazy_token;
            self.has_pending_token = true;
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        }

        self.next_token_id = new_t1;

        return PldStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted_drafts,
            .used_lookup = true,
        };
    }

    /// Result of one `nextDrafter` step. Same shape as PLD's result so the
    /// outer wrapper can share token-emit / EOS-check logic.
    pub const DrafterStepResult = struct {
        /// Tokens to emit this step. On a full accept this is
        /// `[t1, ...all_drafts]` (length `block_size`); on partial accept j
        /// it is `[t1, draft[0..j]]` (length `1+j`). The corrected fallback
        /// becomes `next_token_id` for the next call.
        tokens: []const u32,
        /// Number of *drafted* tokens accepted (excludes always-accepted t1).
        accepted_tokens: u32,
    };

    /// Drafter-assisted decode step. Mirrors `nextPld` but the draft comes
    /// from `block_size - 1` autoregressive forwards through the Gemma 4
    /// assistant drafter (cross-attending into target's KV) instead of an
    /// n-gram lookup. Verify is identical: target forward over
    /// `[t1, draft0..draft_{m-1}]` with greedy / stochastic accept.
    ///
    /// Algorithm:
    ///   1. Run `block_size - 1` drafter steps. Each step's input is
    ///      `concat(target.embed(prev_tok)*scale, h_prev)`. `prev_tok` starts
    ///      at `next_token_id` (= t1); after step i it's the just-sampled
    ///      `draft[i]`. `h_prev` starts at `mtp_last_hidden` (captured at
    ///      prefill or the previous accept's verify-forward); after step i
    ///      it's the drafter's own `post_proj` output.
    ///      All drafter forwards in one round share `rope_offset =
    ///      target.cache.step` (per upstream `set_shared_kv`).
    ///   2. Snapshot KV + SSM, run target verify forward over
    ///      `[t1, draft0..draft_{m-1}]` length `block_size` with
    ///      `forwardCaptureHidden` so we get the new `h_prev` at position m.
    ///   3. Walk argmax(verify_logits[i]) vs draft[i] for i in 0..m-1.
    ///      Greedy: equal → accept. Stochastic: standard speculative-decoding
    ///      ratio test using `mtpProbAt(target_p, draft[i])` (the drafter's
    ///      masked-LM-head produces probabilistic logits, so we treat its
    ///      sampled draft as a one-hot proposal — same simplification PLD
    ///      uses).
    ///   4. Full accept (j == m): emit drafts, sample new pending from
    ///      verify_logits[m-1] (the target's prediction one position past the
    ///      last accepted draft — already computed during verify), update
    ///      `mtp_last_hidden` to the captured post-final-norm hidden.
    ///   5. Partial accept (j < m): roll back KV+SSM, re-forward
    ///      `[t1, draft[0..j-1]]` length `j+1` (with hidden capture) so
    ///      cache lands at exactly `+j+1`. Sample correction from the
    ///      *original* verify_logits[j] (the model's prediction at the
    ///      rejected position).
    pub fn nextDrafter(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        if (self.done) return null;
        std.debug.assert(self.drafter != null);
        std.debug.assert(self.has_mtp_last_hidden); // captured at init or last accept
        std.debug.assert(self.sampling.constraint == null); // grammar + drafter unsupported
        std.debug.assert(self.logprobs_n == 0); // logprobs + drafter unsupported

        const xfm = self.xfm;
        const s = xfm.s;
        const drafter = self.drafter.?;
        const m: u32 = @max(@as(u32, 1), self.drafter_block_size - 1);
        const t1: u32 = self.next_token_id; // already-decided token at position cache.step

        // RoPE offset: position the drafter's queries rotate by. Per upstream
        // `set_shared_kv`, this is `target.cache.step` and stays constant
        // across all `m` drafter steps in this round.
        const rope_offset: c_int = @intCast(xfm.cache.step);

        // ── Phase 1: draft `m` tokens autoregressively ──
        var drafts = try allocator.alloc(u32, m);
        errdefer allocator.free(drafts);

        var prev_tok = t1;
        // h_prev rolls forward through the drafter. Starts at the captured
        // target hidden; subsequent steps use the drafter's post_proj output.
        // We do NOT free `mtp_last_hidden` during drafting — it stays valid
        // until the verify path either replaces it (accept) or restores via
        // re-forward (reject). Drafter step's h_prev_next is owned per step.
        var h_prev_owner: ?mlx.mlx_array = null;
        defer if (h_prev_owner) |h| {
            _ = mlx.mlx_array_free(h);
        };

        var i: u32 = 0;
        while (i < m) : (i += 1) {
            const h_prev_arg: mlx.mlx_array = if (h_prev_owner) |h| h else self.mtp_last_hidden;
            const step_out = try drafter_mod.step(drafter, xfm, prev_tok, h_prev_arg, rope_offset);
            // Sample the drafted token.
            const draft_lazy = sampleTokenLazy(step_out.logits, self.sampling, s);
            _ = mlx.mlx_array_free(step_out.logits);
            try mlx.check(mlx.mlx_array_eval(draft_lazy));
            var draft_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&draft_val, draft_lazy));
            _ = mlx.mlx_array_free(draft_lazy);
            drafts[i] = @intCast(draft_val);

            // Roll h_prev forward: free previous owner, take ownership of new.
            if (h_prev_owner) |h_old| {
                _ = mlx.mlx_array_free(h_old);
            }
            h_prev_owner = step_out.h_prev_next;
            prev_tok = drafts[i];
        }

        // ── Phase 2: snapshot KV + SSM ──
        var kv_snap = try xfm.cache.snapshot();
        defer kv_snap.deinit();
        var ssm_snaps: ?[]SSMCacheEntrySnapshot = null;
        defer if (ssm_snaps) |snaps| {
            for (snaps) |*sn| ssmSnapshotDeinit(sn);
            xfm.allocator.free(snaps);
        };
        if (xfm.ssm_entries) |entries| {
            const out = try xfm.allocator.alloc(SSMCacheEntrySnapshot, entries.len);
            for (entries, 0..) |*entry, idx| out[idx] = ssmSnapshot(entry);
            ssm_snaps = out;
        }
        const moe_seq_offset_snap = xfm.moe_seq_offset;

        // ── Phase 3: verify forward [t1, draft0..draft_{m-1}] length 1+m ──
        const seq_len: c_int = @intCast(1 + m);
        const verify_input_buf = try allocator.alloc(i32, 1 + m);
        defer allocator.free(verify_input_buf);
        verify_input_buf[0] = @intCast(t1);
        for (drafts, 0..) |d, idx| verify_input_buf[1 + idx] = @intCast(d);
        const verify_shape = [_]c_int{ 1, seq_len };
        const verify_input = mlx.mlx_array_new_data(verify_input_buf.ptr, &verify_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(verify_input);

        var new_hidden = mlx.mlx_array_new();
        // Captures the post-final-norm hidden at the LAST input position
        // (= position m, predicting the bonus token if all drafts accept).
        const verify_logits = try xfm.forwardCaptureHidden(verify_input, &new_hidden);
        // verify_logits shape: [1, 1+m, V]
        self.drafter_attempted += 1;

        // Slice per-position logits up front so we can reuse them for the
        // partial-accept correction sample below without re-forwarding.
        const vl_shape = mlx.getShape(verify_logits);
        const slice_strides = [_]c_int{ 1, 1, 1 };
        const per_pos_logits = try allocator.alloc(mlx.mlx_array, 1 + m);
        defer {
            for (per_pos_logits) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(per_pos_logits);
        }
        for (per_pos_logits, 0..) |*slot, idx| {
            slot.* = mlx.mlx_array_new();
            const start = [_]c_int{ 0, @intCast(idx), 0 };
            const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
            try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
        }
        _ = mlx.mlx_array_free(verify_logits);

        // ── Phase 4: decide longest accepted prefix ──
        const stochastic = self.sampling.temperature > 0.01;
        var accepted: u32 = 0;
        if (stochastic) {
            // Same simplification as PLD: treat the drafted token as a
            // one-hot proposal (q[draft[i]] = 1), so the speculative-decoding
            // accept ratio reduces to `min(1, target_p[draft[i]])`. The
            // drafter's actual probability for draft[i] is unavailable here
            // without re-running the masked LM head softmax, and the
            // simplification preserves the marginal output distribution.
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const target_p = try mtpProbsAtLastPos(per_pos_logits[k], self.sampling, s);
                defer _ = mlx.mlx_array_free(target_p);
                const p_draft = try mtpProbAt(target_p, drafts[k], s);
                const accept_prob: f32 = @min(1.0, p_draft);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
        } else {
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                var argmax_arr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(argmax_arr);
                try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, per_pos_logits[k], 2, false, s));
                try mlx.check(mlx.mlx_array_eval(argmax_arr));
                var argmax_val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&argmax_val, argmax_arr));
                if (@as(u32, @intCast(argmax_val)) != drafts[k]) break;
                accepted += 1;
            }
        }

        // Sample the next pending token from per_pos_logits[accepted]:
        //   - full accept (accepted == m): logits[m] is the target's prediction
        //     one past the last draft (the bonus token). Sample directly.
        //   - partial accept: logits[accepted] is the target's prediction at
        //     the rejected position. For stochastic, sample from the residual
        //     `max(p − one_hot(draft[accepted]), 0)` to preserve the marginal
        //     distribution conditional on "not draft[accepted]" (Leviathan
        //     et al). Greedy: just argmax (the rejected position's argmax is
        //     the model's true next token).
        const correction_logits = per_pos_logits[accepted];
        const next_pending: u32 = blk: {
            if (stochastic) {
                const probs = try mtpProbsAtLastPos(correction_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                if (accepted < m) {
                    const onehot = try pldOneHotRow(drafts[accepted], vl_shape[2], s);
                    defer _ = mlx.mlx_array_free(onehot);
                    break :blk try mtpSampleResidual(probs, onehot, s);
                } else {
                    break :blk try mtpSampleFromProbs(probs, s);
                }
            } else {
                const lazy = sampleTokenLazy(correction_logits, self.sampling, s);
                try mlx.check(mlx.mlx_array_eval(lazy));
                var v: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
                _ = mlx.mlx_array_free(lazy);
                break :blk @intCast(v);
            }
        };

        // ── Phase 5: commit / rollback ──
        if (accepted == m) {
            // Full accept: cache at +1+m. Emit [t1, ...drafts]. Pending = next_pending.
            // The captured `new_hidden` is the post-final-norm hidden at
            // position m — the last accepted draft's position. That's the
            // h_prev for the NEXT round (drafting from t = next_pending; the
            // hidden corresponds to draft[m-1], which is what next_pending
            // follows). This matches the convention `nextMtp` uses.
            const tokens = try allocator.alloc(u32, 1 + m);
            tokens[0] = t1;
            for (drafts, 0..) |d, idx| tokens[1 + idx] = d;

            try self.generated_ids.append(allocator, t1);
            for (drafts) |d| try self.generated_ids.append(allocator, d);

            if (self.has_mtp_last_hidden) _ = mlx.mlx_array_free(self.mtp_last_hidden);
            self.mtp_last_hidden = new_hidden;
            self.has_mtp_last_hidden = true;

            self.drafter_accepted_tokens += m;
            self.next_token_id = next_pending;
            self.step += 1 + m;
            self.completion_tokens += 1 + m;

            // drafts buffer transferred into tokens copy; free original.
            allocator.free(drafts);
            return DrafterStepResult{
                .tokens = tokens,
                .accepted_tokens = m,
            };
        }

        // Partial accept (accepted < m). Cache over-advanced by (m - accepted).
        // The captured new_hidden is for position m (which we're rolling back
        // past) — discard it. Roll back KV+SSM, then re-forward
        // [t1, drafts[0..accepted]] length 1+accepted with hidden capture so
        // mtp_last_hidden lands at the position immediately past the last
        // accepted draft (where next_pending will live).
        _ = mlx.mlx_array_free(new_hidden);

        try xfm.cache.restore(&kv_snap);
        if (ssm_snaps) |snaps| {
            for (xfm.ssm_entries.?, snaps) |*entry, *sn| try ssmRestore(entry, sn);
        }
        xfm.moe_seq_offset = moe_seq_offset_snap;

        const re_seq_len: c_int = @intCast(1 + accepted);
        const re_input_buf = try allocator.alloc(i32, 1 + accepted);
        defer allocator.free(re_input_buf);
        re_input_buf[0] = @intCast(t1);
        for (drafts[0..accepted], 0..) |d, idx| re_input_buf[1 + idx] = @intCast(d);
        const re_shape = [_]c_int{ 1, re_seq_len };
        const re_input = mlx.mlx_array_new_data(re_input_buf.ptr, &re_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(re_input);

        var re_new_hidden = mlx.mlx_array_new();
        const re_logits = try xfm.forwardCaptureHidden(re_input, &re_new_hidden);
        _ = mlx.mlx_array_free(re_logits);

        const tokens = try allocator.alloc(u32, 1 + accepted);
        tokens[0] = t1;
        for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;

        try self.generated_ids.append(allocator, t1);
        for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);

        if (self.has_mtp_last_hidden) _ = mlx.mlx_array_free(self.mtp_last_hidden);
        self.mtp_last_hidden = re_new_hidden;
        self.has_mtp_last_hidden = true;

        self.drafter_accepted_tokens += accepted;
        self.next_token_id = next_pending;
        self.step += 1 + accepted;
        self.completion_tokens += 1 + accepted;

        allocator.free(drafts);
        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
        };
    }

    /// Returns the next token ID, or null when generation is finished.
    ///
    /// Pipeline architecture (matches mlx-lm's generator pattern):
    ///
    ///   The KEY to effective pipelining is the ORDER of operations:
    ///   1. Build next step's lazy graph (depends on pending lazy token)
    ///   2. async_eval the next graph — GPU computes pending token as a DEPENDENCY,
    ///      then continues with the forward pass
    ///   3. eval(pending_token) — returns INSTANTLY since GPU already computed it
    ///   4. Return the token while GPU continues computing the next forward pass
    ///
    ///   This mirrors mlx-lm's: _step(y) → async_eval(next_y) → yield y.item()
    ///   where y.item() is instant because async_eval forced y's computation.
    pub fn next(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (self.done) return null;
        if (self.sampling.constraint != null) return self.nextConstrained(allocator);

        // ── Phase 1: Build and submit the NEXT step FIRST ──
        // This forces the GPU to compute the pending token as a dependency,
        // so when we eval it in Phase 2, it's already ready.
        if (self.has_pending_logits and self.logprobs_n == 0 and self.step + 1 < self.max_tokens) {
            const step_logits = self.pending_logits;
            self.has_pending_logits = false;

            const lazy_token = sampleTokenLazy(step_logits, self.sampling, self.xfm.s);
            _ = mlx.mlx_array_free(step_logits);

            if (lazyForward(self.xfm, lazy_token)) |next_logits| {
                const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
                const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                _ = mlx.mlx_async_eval(vec);
                _ = mlx.mlx_vector_array_free(vec);

                // NOW resolve the pending token — GPU already computed it as a
                // dependency of the graph we just submitted. Should be instant.
                try self.resolvePendingToken();

                // Check stop conditions on the resolved token
                if (try self.checkStop()) return null;

                const token = self.next_token_id;
                self.completion_tokens += 1;
                self.step += 1;
                try self.generated_ids.append(allocator, token);

                if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

                // Store new pending state
                self.pending_token = lazy_token;
                self.has_pending_token = true;
                self.pending_logits = next_logits;
                self.has_pending_logits = true;

                return token;
            } else |_| {
                // lazyForward failed — fall through to slow path
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                self.next_token_id = @intCast(val);
                self.has_pending_token = false;
            }
        }

        // ── Phase 2: Slow path (first token, last token, logprobs, or pipeline miss) ──
        try self.resolvePendingToken();

        if (try self.checkStop()) return null;

        const token = self.next_token_id;
        self.completion_tokens += 1;
        self.step += 1;
        try self.generated_ids.append(allocator, token);

        if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

        const step_logits = if (self.has_pending_logits) blk: {
            const logits = self.pending_logits;
            self.has_pending_logits = false;
            break :blk logits;
        } else blk: {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            break :blk try self.xfm.forward(tok_input);
        };

        // Logprobs: fully synchronous
        if (self.logprobs_n > 0) {
            defer _ = mlx.mlx_array_free(step_logits);
            const result = try sampleToken(allocator, step_logits, self.sampling, self.generated_ids.items, self.logprobs_n, self.xfm.s);
            self.next_token_id = result.token_id;
            if (self.last_logprob) |*lp| allocator.free(lp.top_logprobs);
            self.last_logprob = result.logprob_result;
            if (self.step < self.max_tokens) self.startAsyncForward(result.token_id);
            return token;
        }

        // Last token or pipeline bootstrap
        const lazy_token = sampleTokenLazy(step_logits, self.sampling, self.xfm.s);
        _ = mlx.mlx_array_free(step_logits);

        if (self.step < self.max_tokens) {
            const next_logits = lazyForward(self.xfm, lazy_token) catch {
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                self.next_token_id = @intCast(val);
                return token;
            };

            const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);

            self.pending_token = lazy_token;
            self.has_pending_token = true;
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            try mlx.check(mlx.mlx_array_eval(lazy_token));
            var val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
            _ = mlx.mlx_array_free(lazy_token);
            self.next_token_id = @intCast(val);
        }

        return token;
    }

    /// Synchronous, grammar-constrained sampling step. Used whenever
    /// `sampling.constraint` is non-null. Builds a token mask from the grammar's
    /// current state, applies it to the pending logits, samples, advances the
    /// grammar by the sampled token's bytes, and pre-launches the next forward
    /// pass to overlap with the next mask build.
    fn nextConstrained(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (!self.has_pending_logits) {
            self.done = true;
            return null;
        }
        if (self.timeout_ns > 0 and self.timer.read() >= self.timeout_ns) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }

        const constraint = self.sampling.constraint.?;
        const s = self.xfm.s;

        _ = try token_mask.buildMask(constraint.grammar, constraint.token_bytes, constraint.mask_buf);

        // Also allow every stop-id the generator recognises once the grammar is
        // complete. `token_mask.buildMask` only knows about `tokenizer.eos_id`,
        // but models often have additional stop tokens (e.g. `<|im_end|>` for
        // Qwen, `<end_of_turn>` for Gemma 4) registered via the config — without
        // this, the model can never stop.
        if (constraint.grammar.isComplete()) {
            for (self.eos_token_ids) |eos_id| {
                if (eos_id < constraint.mask_buf.len) constraint.mask_buf[eos_id] = true;
            }
        }

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        defer _ = mlx.mlx_array_free(step_logits);

        var masked_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked_logits);
        try applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, s);

        // Synchronous sample: we need the realized token id to advance the grammar.
        const lazy = sampleTokenLazy(masked_logits, self.sampling, s);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        _ = mlx.mlx_array_free(lazy);
        const token: u32 = @intCast(val);
        self.next_token_id = token;

        // Stop on EOS — do not advance grammar or include in output.
        for (self.eos_token_ids) |eos_id| {
            if (token == eos_id) {
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        }
        if (token == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        } else {
            self.consecutive_pad = 0;
        }

        // Advance the grammar by the sampled token's byte sequence. The mask
        // guarantees every byte is accepted (or the token has no byte form, e.g. a
        // special tag) — so a rejection here means a bug we want to surface.
        if (token < constraint.token_bytes.bytes.len) {
            if (constraint.token_bytes.bytes[token]) |bytes| {
                for (bytes) |b| {
                    const ok = try constraint.grammar.acceptByte(b);
                    if (!ok) {
                        log.warn("[grammar] sampled token {d} produced byte 0x{x} that was rejected — disabling further mask enforcement\n", .{ token, b });
                        constraint.grammar.dead = true;
                        break;
                    }
                }
            }
        }

        self.completion_tokens += 1;
        self.step += 1;
        try self.generated_ids.append(allocator, token);
        if (self.step % 256 == 0) _ = mlx.mlx_clear_cache();

        if (self.step < self.max_tokens) {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const next_logits = try self.xfm.forward(tok_input);
            const arr = [_]mlx.mlx_array{next_logits};
            const vec = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            self.done = true;
            self.finish_reason = "length";
        }

        return token;
    }

    /// Check all stop conditions. Returns true if generation should stop.
    fn checkStop(self: *Generator) !bool {
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        if (self.timeout_ns > 0 and self.timer.read() >= self.timeout_ns) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        for (self.eos_token_ids) |eos_id| {
            if (self.next_token_id == eos_id) {
                self.done = true;
                self.finish_reason = "stop";
                return true;
            }
        }
        if (self.next_token_id == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                self.done = true;
                self.finish_reason = "stop";
                return true;
            }
        } else {
            self.consecutive_pad = 0;
        }
        return false;
    }

    /// Legacy sync forward for logprobs path.
    fn startAsyncForward(self: *Generator, token_id: u32) void {
        const tok_i32: i32 = @intCast(token_id);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);

        const logits = self.xfm.forward(tok_input) catch return;
        const arr = [_]mlx.mlx_array{logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);

        self.pending_logits = logits;
        self.has_pending_logits = true;
    }
};

/// Build forward pass from a lazy sampled token array.
/// Reshapes [1] -> [1, 1] and calls transformer forward. All lazy (no eval).
fn lazyForward(xfm: *Transformer, lazy_token: mlx.mlx_array) !mlx.mlx_array {
    const tok_shape = [_]c_int{ 1, 1 };
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, lazy_token, &tok_shape, 2, xfm.s));
    return try xfm.forward(reshaped);
}

/// Sample a token lazily from logits — returns a lazy MLX array (no eval).
/// Handles temperature scaling, top-k, and top-p, but defers materialization.
/// The returned array has shape [1] with the sampled token ID.
/// Caller must free the returned array.
/// Compute the probability distribution over the vocabulary at the LAST
/// position of `logits_3d` (shape `[B, S, V]`), with the SAME temperature +
/// top-k + top-p masking the sampler would apply. Both `target_p` and `draft_q`
/// in the stochastic-verify accept test must be computed via this function so
/// the ratio `p[draft] / q[draft]` is well-defined over the kept support.
/// Caller owns the returned array; shape `[B, V]`.
fn mtpProbsAtLastPos(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(logits_3d);
    const seq_len = shape[1];
    var current = mlx.mlx_array_new();
    if (seq_len == 1) {
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, logits_3d, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ shape[0], seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits_3d, &start, 3, &stop, 3, &strides, 3, s));
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s));
    }

    // Apply temperature → top-k → top-p (same order as `sampleTokenLazy`).
    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0) {
        var masked = mlx.mlx_array_new();
        applyTopK(&masked, current, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }
    if (sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        applyTopP(&masked, current, sampling.top_p, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    // Softmax: tokens at -inf become 0, kept tokens renormalize to sum=1.
    var probs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    return probs;
}

/// Read `probs[0, token_id]` as f32. Forces realization with a single eval.
fn mtpProbAt(probs: mlx.mlx_array, token_id: u32, s: mlx.mlx_stream) !f32 {
    const idx_val: i32 = @intCast(token_id);
    const idx_shape = [_]c_int{1};
    const idx_arr = mlx.mlx_array_new_data(&idx_val, &idx_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(idx_arr);

    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, probs, idx_arr, -1, s));

    // Cast to f32 so item_float32 is exact regardless of source dtype (bf16 etc.).
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, taken, .float32, s));
    try mlx.check(mlx.mlx_array_eval(as_f32));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, as_f32));
    return v;
}

/// Sample one token from probability distribution `probs` (shape `[B, V]`).
/// Returns a u32 token id (caller can append directly).
fn mtpSampleFromProbs(probs: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    // mlx_random_categorical takes logits and applies softmax. Feed log(probs)
    // so the categorical's softmax recovers the original distribution.
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_log(&log_probs, probs, s));

    const null_key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(null_key);
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, log_probs, -1, null_key, s));
    try mlx.check(mlx.mlx_array_eval(sampled));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, sampled));
    return @intCast(v);
}

/// Build a one-hot float32 row vector of shape `[1, vocab]` with 1.0 at
/// `index` and 0.0 elsewhere. Used by PLD's stochastic-verify reject path,
/// which models the draft (an n-gram lookup, not a probabilistic model) as a
/// degenerate one-hot distribution. Caller owns the returned array.
fn pldOneHotRow(index: u32, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var indices = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices);
    try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));

    const target_val: i32 = @intCast(index);
    const tgt_shape = [_]c_int{1};
    const target_idx = mlx.mlx_array_new_data(&target_val, &tgt_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(target_idx);

    var mask_bool = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_bool);
    try mlx.check(mlx.mlx_equal(&mask_bool, indices, target_idx, s));

    var mask_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_f32);
    try mlx.check(mlx.mlx_astype(&mask_f32, mask_bool, .float32, s));

    const out_shape = [_]c_int{ 1, vocab };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, mask_f32, &out_shape, 2, s));
    return out;
}

/// Sample from the residual distribution `residual = max(target - draft, 0)`,
/// renormalized. Used on stochastic-verify reject so the corrected token
/// preserves the target distribution (per Leviathan et al. speculative
/// decoding paper).
fn mtpSampleResidual(target_probs: mlx.mlx_array, draft_probs: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, target_probs, draft_probs, s));

    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var residual = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(residual);
    try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));

    return mtpSampleFromProbs(residual, s);
}

fn sampleTokenLazy(logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) mlx.mlx_array {
    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    // `current` is the single owned intermediate — freed before each reassignment.
    var current = mlx.mlx_array_new();

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, logits, &sq_shape, 2, s);
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        _ = mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s);

        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s);
    }

    // Greedy: argmax (no temperature)
    if (sampling.temperature < 0.01) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, current, -1, false, s);
        _ = mlx.mlx_array_free(current);
        return result;
    }

    // Scale by 1/temperature
    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        var next = mlx.mlx_array_new();
        _ = mlx.mlx_divide(&next, current, temp_arr, s);
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Apply top-k filtering (lazy)
    if (sampling.top_k > 0) {
        var next = mlx.mlx_array_new();
        applyTopK(&next, current, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Apply top-p filtering (lazy)
    if (sampling.top_p < 1.0) {
        var next = mlx.mlx_array_new();
        applyTopP(&next, current, sampling.top_p, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Sample from categorical distribution (lazy — no eval!)
    var sampled = mlx.mlx_array_new();
    const null_key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(null_key);
    _ = mlx.mlx_random_categorical(&sampled, current, -1, null_key, s);
    _ = mlx.mlx_array_free(current);

    return sampled; // Shape [1], lazy
}

/// Convenience: generate all tokens at once (non-streaming).
pub fn generate(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    logprobs_n: u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.init(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids);
    gen.timeout_ns = timeout_ns;
    gen.logprobs_n = logprobs_n;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    var logprob_results = std.ArrayList(LogprobResult).empty;
    defer {
        if (logprobs_n == 0) {
            for (logprob_results.items) |*lp| allocator.free(lp.top_logprobs);
            logprob_results.deinit(allocator);
        }
    }

    timer.reset();
    while (try gen.next(allocator)) |token_id| {
        try output_ids.append(allocator, token_id);
        if (logprobs_n > 0) {
            if (gen.last_logprob) |lp| {
                try logprob_results.append(allocator, lp);
                gen.last_logprob = null; // Transfer ownership
            }
        }
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    log.debug("Decode: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        decode_ns / std.time.ns_per_ms,
        num_decoded,
        decode_tps,
    });

    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = if (logprobs_n > 0) try logprob_results.toOwnedSlice(allocator) else null,
    };
}

/// MTP-enabled non-streaming variant of `generate`. Returns the same
/// `GenerationResult` shape so callers can dispatch on `enable_mtp` without
/// code-path forking. Logprobs are unsupported under MTP in v1.
pub fn generateMtp(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .mtp_enabled = true });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (MTP): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    // initWithOptions(.{ .mtp_enabled = true }) seeded `next_token_id` with the
    // prefill's first sampled token but skipped the lazy pre-forward. Each
    // nextMtp call consumes `next_token_id` as t1, runs draft+verify, and emits
    // 1 or 2 tokens via MtpStepResult. We loop until done / EOS / max_tokens.
    while (!gen.done and gen.completion_tokens < gen.max_tokens) {
        const result = (try gen.nextMtp(allocator)) orelse break;
        // Match `generate`'s convention: stop tokens are NOT included in
        // output_ids. `gen.next` enforces this by returning null on EOS
        // before any append; the speculative path has to check explicitly.
        if (isEosId(result.first, eos_token_ids)) {
            gen.done = true;
            gen.finish_reason = "stop";
            break;
        }
        try output_ids.append(allocator, result.first);
        if (result.second) |second| {
            if (isEosId(second, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break;
            }
            try output_ids.append(allocator, second);
        }
        if (gen.completion_tokens >= gen.max_tokens) {
            gen.finish_reason = "length";
            gen.done = true;
            break;
        }
    }

    return finishMtpResult(&gen, &output_ids, allocator, prompt_ids.len, prefill_tps, timer, tok);
}

/// PLD-enabled non-streaming variant of `generate`. Model-agnostic — works on
/// every supported architecture, no MTP weights required. Logprobs and
/// constrained sampling are unsupported (asserted out by `nextPld`).
///
/// `draft_len` and `key_len` come from server config (`--pld-draft-len` /
/// `--pld-key-len`); typical values are 5 and 3 respectively.
pub fn generatePld(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    draft_len: u32,
    key_len: u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{});
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (PLD): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    // Decode loop. Each `nextPld` returns 1..=(1+draft_len) tokens. Stop on
    // EOS / max_tokens / timeout. We check stop conditions on every emitted
    // token (drafts can include EOS just like regular sampling) so the early
    // exit is correct.
    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextPld(allocator, draft_len, key_len)) orelse break;
        defer allocator.free(result.tokens);
        // Match `generate`'s convention: stop tokens are NOT included in
        // output_ids. Check before appending — the speculative path has to do
        // this explicitly because `nextPld` emits multiple tokens at once and
        // can't return-null mid-batch like the single-token `next` does.
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    return finishPldResult(&gen, &output_ids, allocator, prefill_tps, timer, tok);
}

/// Drafter-enabled non-streaming variant of `generate`. Mirrors
/// `generatePld` (multi-token-per-step emit pattern) but the draft comes from
/// a Gemma 4 assistant drafter cross-attending into the target's KV cache
/// instead of an n-gram lookup.
///
/// `drafter` must already be `bind()`-ed to `xfm`. `block_size` is the
/// per-round token budget (drafter forwards = block_size - 1; verify forward
/// length = block_size).
pub fn generateDrafter(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    drafter: *DrafterModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    block_size: u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{
        .drafter_enabled = true,
        .drafter = drafter,
        .drafter_block_size = block_size,
    });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (drafter): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextDrafter(allocator)) orelse break;
        defer allocator.free(result.tokens);
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    return finishDrafterResult(&gen, &output_ids, allocator, prefill_tps, timer, tok);
}

fn finishDrafterResult(
    gen: *Generator,
    output_ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
    prefill_tps: f64,
    timer: io_util.Stopwatch,
    tok: *const Tokenizer,
) !GenerationResult {
    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.drafter_attempted > 0) {
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.drafter_accepted_tokens)) / @as(f64, @floatFromInt(gen.drafter_attempted));
        log.info("Decode (drafter): {d}ms ({d} tokens, {d:.3} tok/s; drafter accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.drafter_accepted_tokens,
            gen.drafter_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (drafter): {d}ms ({d} tokens, {d:.3} tok/s; no draft attempts)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

fn finishPldResult(
    gen: *Generator,
    output_ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
    prefill_tps: f64,
    timer: io_util.Stopwatch,
    tok: *const Tokenizer,
) !GenerationResult {
    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.pld_attempted > 0) {
        // "Tokens saved" = accepted_tokens (drafts that landed) + 0 from
        // verify forwards that ran. Acceptance ratio is per-position, so we
        // compute average tokens accepted per attempt for visibility.
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.pld_accepted_tokens)) / @as(f64, @floatFromInt(gen.pld_attempted));
        log.info("Decode (PLD): {d}ms ({d} tokens, {d:.3} tok/s; pld accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.pld_accepted_tokens,
            gen.pld_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (PLD): {d}ms ({d} tokens, {d:.3} tok/s; no n-gram matches found)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

pub fn isEosId(id: u32, eos: []const u32) bool {
    for (eos) |e| if (id == e) return true;
    return false;
}

fn finishMtpResult(
    gen: *Generator,
    output_ids: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
    prompt_len: usize,
    prefill_tps: f64,
    timer: io_util.Stopwatch,
    tok: *const Tokenizer,
) !GenerationResult {
    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.mtp_attempted > 0) {
        const acc_pct = @as(f64, @floatFromInt(gen.mtp_accepted)) * 100.0 / @as(f64, @floatFromInt(gen.mtp_attempted));
        log.info("Decode (MTP): {d}ms ({d} tokens, {d:.3} tok/s; mtp accept={d}/{d} = {d:.1}%)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.mtp_accepted,
            gen.mtp_attempted,
            acc_pct,
        });
    }
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    _ = prompt_len;
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

/// Compute mean-pooled, L2-normalized embedding from token IDs.
/// Returns a float32 array of shape [hidden_size]. Caller must free the returned slice.
pub fn computeEmbedding(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    token_ids: []const u32,
) ![]f32 {
    const s = xfm.s;
    const prompt_len: c_int = @intCast(token_ids.len);
    const shape = [_]c_int{ 1, prompt_len };

    const ids_i32 = try allocator.alloc(i32, token_ids.len);
    defer allocator.free(ids_i32);
    for (token_ids, 0..) |id, i| {
        ids_i32[i] = @intCast(id);
    }

    const input = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(input);

    // Get hidden states: [1, seq_len, hidden_size]
    const hidden = try xfm.forwardEmbedding(input);
    defer _ = mlx.mlx_array_free(hidden);

    // Mean pool over sequence dimension (axis=1): [1, hidden_size]
    var pooled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_mean_axis(&pooled, hidden, 1, false, s));

    // L2 normalize: pooled / sqrt(sum(pooled^2))
    var squared = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(squared);
    try mlx.check(mlx.mlx_multiply(&squared, pooled, pooled, s));

    var sum_sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum_sq);
    try mlx.check(mlx.mlx_sum_axis(&sum_sq, squared, -1, true, s));

    var norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm);
    try mlx.check(mlx.mlx_sqrt(&norm, sum_sq, s));

    // Add epsilon to avoid division by zero
    const eps = mlx.mlx_array_new_float(1e-12);
    defer _ = mlx.mlx_array_free(eps);
    var norm_safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm_safe);
    try mlx.check(mlx.mlx_maximum(&norm_safe, norm, eps, s));

    var normalized = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normalized);
    try mlx.check(mlx.mlx_divide(&normalized, pooled, norm_safe, s));

    // Eval and extract float data
    try mlx.check(mlx.mlx_array_eval(normalized));
    const n_shape = mlx.getShape(normalized);
    const dim: usize = @intCast(n_shape[n_shape.len - 1]);
    const data_ptr = mlx.mlx_array_data_float32(normalized) orelse return error.MlxError;

    const result = try allocator.alloc(f32, dim);
    @memcpy(result, data_ptr[0..dim]);
    return result;
}

const SampleResult = struct {
    token_id: u32,
    logprob_result: ?LogprobResult = null,
};

/// Sample a token from the last position's logits.
/// temperature <= 0.01: greedy argmax. Otherwise: scale logits, apply top_p, and sample.
/// If logprobs_n > 0, also computes logprobs for the sampled token and top N alternatives.
fn sampleToken(allocator: std.mem.Allocator, logits: mlx.mlx_array, sampling: SamplingParams, generated_ids: ?[]const u32, logprobs_n: u32, s: mlx.mlx_stream) !SampleResult {
    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    var last_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last_logits);

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, logits, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));

        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, sliced, &sq_shape, 2, s));
    }

    // Track current working logits (avoid copies when no transform needed)
    var current = last_logits;

    // Apply repeat penalty to already-generated tokens
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    const needs_penalty = (sampling.repeat_penalty != 1.0 or sampling.presence_penalty != 0.0);
    if (needs_penalty) {
        if (generated_ids) |ids| {
            if (ids.len > 0) {
                try applyRepeatPenalty(&penalized, current, ids, sampling.repeat_penalty, sampling.presence_penalty, s);
                current = penalized;
                penalized_owned = true;
            }
        }
    }

    // Greedy if temperature is ~0
    if (sampling.temperature < 0.01) {
        const token_id = try argmax(current, s);
        var logprob_result: ?LogprobResult = null;
        if (logprobs_n > 0) {
            logprob_result = try computeLogprobs(allocator, current, token_id, logprobs_n, s);
        }
        return .{ .token_id = token_id, .logprob_result = logprob_result };
    }

    // Scale logits by 1/temperature
    var scaled = mlx.mlx_array_new();
    var scaled_owned = false;
    defer if (scaled_owned) {
        _ = mlx.mlx_array_free(scaled);
    };

    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        try mlx.check(mlx.mlx_divide(&scaled, current, temp_arr, s));
        current = scaled;
        scaled_owned = true;
    }

    // For logprobs, remember the logits after temp scaling but before filtering
    const logprobs_logits = current;

    // Apply top-k filtering
    var after_topk = mlx.mlx_array_new();
    var topk_owned = false;
    defer if (topk_owned) {
        _ = mlx.mlx_array_free(after_topk);
    };

    if (sampling.top_k > 0) {
        try applyTopK(&after_topk, current, sampling.top_k, s);
        current = after_topk;
        topk_owned = true;
    }

    // Apply top-p (nucleus) sampling
    var after_topp = mlx.mlx_array_new();
    var topp_owned = false;
    defer if (topp_owned) {
        _ = mlx.mlx_array_free(after_topp);
    };

    if (sampling.top_p < 1.0) {
        try applyTopP(&after_topp, current, sampling.top_p, s);
        current = after_topp;
        topp_owned = true;
    }

    // Sample from categorical distribution
    var sampled = mlx.mlx_array_new();

    if (sampling.seed) |seed| {
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, key, s));
    } else {
        const null_key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(null_key);
        try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, null_key, s));
    }

    // Eval and extract
    try mlx.check(mlx.mlx_array_eval(sampled));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, sampled));

    const token_id: u32 = @intCast(val);

    // Compute logprobs after sampling (we now know the token_id)
    var logprob_result: ?LogprobResult = null;
    if (logprobs_n > 0) {
        logprob_result = try computeLogprobs(allocator, logprobs_logits, token_id, logprobs_n, s);
    }

    _ = mlx.mlx_array_free(sampled);
    return .{ .token_id = token_id, .logprob_result = logprob_result };
}

/// Compute log-probabilities from logits. Returns the logprob of the chosen token
/// and the top N alternatives with their token IDs and logprobs.
fn computeLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen_token: u32, top_n: u32, s: mlx.mlx_stream) !LogprobResult {
    // Compute log_softmax = log(softmax(logits)) on GPU
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, logits, -1, true, s));

    var log_probs_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs_raw);
    try mlx.check(mlx.mlx_log(&log_probs_raw, probs, s));

    // Cast to float32 for CPU readback (model may produce float16 logits)
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_astype(&log_probs, log_probs_raw, .float32, s));

    // Get top-k logprobs using mlx_topk (returns top values in descending order)
    const k: c_int = @intCast(@min(top_n + 1, 20)); // +1 to ensure chosen token is included
    var topk_vals_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_vals_raw);
    try mlx.check(mlx.mlx_topk(&topk_vals_raw, log_probs, k, s));

    var topk_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_vals);
    try mlx.check(mlx.mlx_astype(&topk_vals, topk_vals_raw, .float32, s));

    // Eval both arrays to CPU
    try mlx.check(mlx.mlx_array_eval(log_probs));
    try mlx.check(mlx.mlx_array_eval(topk_vals));

    // Read the logprob of the chosen token from the full array
    const lp_shape = mlx.getShape(log_probs);
    const vocab_size: usize = @intCast(lp_shape[lp_shape.len - 1]);
    const lp_data = mlx.mlx_array_data_float32(log_probs);
    const chosen_logprob: f32 = if (lp_data) |ptr|
        (if (chosen_token < vocab_size) ptr[chosen_token] else -100.0)
    else
        -100.0;

    // Read top-k values and find their token IDs by scanning the full logprobs
    const topk_data = mlx.mlx_array_data_float32(topk_vals);
    const actual_k: usize = @intCast(k);

    var top_logprobs = try allocator.alloc(TokenLogprob, @min(top_n, @as(u32, @intCast(actual_k))));
    var filled: usize = 0;

    if (topk_data) |tk_ptr| {
        if (lp_data) |full_ptr| {
            // Track used token IDs to avoid duplicates
            var used = std.AutoHashMap(u32, void).init(std.heap.page_allocator);
            defer used.deinit();

            // For each top-k value, find the matching token ID in the full array
            for (0..actual_k) |i| {
                if (filled >= top_n) break;
                const target_val = tk_ptr[i];
                // Find token ID with this logprob value (skip already-used IDs)
                for (0..vocab_size) |tid| {
                    if (full_ptr[tid] == target_val and !used.contains(@intCast(tid))) {
                        const tid_u32: u32 = @intCast(tid);
                        top_logprobs[filled] = .{
                            .token_id = tid_u32,
                            .logprob = target_val,
                        };
                        used.put(tid_u32, {}) catch {};
                        filled += 1;
                        break;
                    }
                }
            }
        }
    }

    // Shrink if we didn't fill all slots
    if (filled < top_logprobs.len) {
        top_logprobs = allocator.realloc(top_logprobs, filled) catch top_logprobs;
    }

    return .{
        .token_logprob = chosen_logprob,
        .top_logprobs = top_logprobs,
    };
}

/// Apply a grammar token mask to logits. `mask[i]==true` keeps `logits[i]`,
/// `false` replaces it with `-inf`. The mask is broadcast over leading dims so
/// `logits` can be either `[1, vocab]` or `[1, 1, vocab]`.
fn applyGrammarMask(allocator: std.mem.Allocator, res: *mlx.mlx_array, logits: mlx.mlx_array, mask: []const bool, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);
    const logit_mask = try maskForLogitVocab(allocator, mask, vocab_size);
    defer logit_mask.deinit(allocator);

    // Zig's `bool` is one byte and matches MLX's `.bool_` storage exactly.
    const arr_shape = [_]c_int{@intCast(vocab_size)};
    const mask_arr = mlx.mlx_array_new_data(@ptrCast(logit_mask.slice.ptr), &arr_shape, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);

    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);

    try mlx.check(mlx.mlx_where(res, mask_arr, logits, neg_inf, s));
}

const LogitMaskView = struct {
    slice: []const bool,
    owned: ?[]bool = null,

    fn deinit(self: LogitMaskView, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
    }
};

fn maskForLogitVocab(allocator: std.mem.Allocator, mask: []const bool, vocab_size: usize) !LogitMaskView {
    if (mask.len == vocab_size) return .{ .slice = mask };

    var adjusted = try allocator.alloc(bool, vocab_size);
    @memset(adjusted, false);
    const copy_len = @min(mask.len, vocab_size);
    @memcpy(adjusted[0..copy_len], mask[0..copy_len]);
    return .{ .slice = adjusted, .owned = adjusted };
}

/// Apply top-k filtering: keep only the top k logits, set the rest to -inf.
fn applyTopK(res: *mlx.mlx_array, logits: mlx.mlx_array, k: u32, s: mlx.mlx_stream) !void {
    // Get the top-k values (returned in descending order)
    var topk_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_vals);
    try mlx.check(mlx.mlx_topk(&topk_vals, logits, @intCast(k), s));

    // Get the minimum of the top-k values (the k-th largest) as cutoff
    var cutoff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cutoff);
    try mlx.check(mlx.mlx_min_axis(&cutoff, topk_vals, -1, true, s));

    // Mask: logits >= cutoff
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_greater_equal(&mask, logits, cutoff, s));

    // Replace masked-out logits with -inf
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask, logits, neg_inf, s));
}

/// Apply top-p (nucleus) sampling: mask logits outside the top-p probability mass.
/// Works on the original (unsorted) logits by computing which tokens to keep.
fn applyTopP(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, s: mlx.mlx_stream) !void {
    // Sort logits ascending to get sorted probabilities
    var sorted_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_logits);
    try mlx.check(mlx.mlx_sort_axis(&sorted_logits, logits, -1, s));

    // Softmax of sorted logits (ascending order: smallest probs first)
    var sorted_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_probs);
    try mlx.check(mlx.mlx_softmax_axis(&sorted_probs, sorted_logits, -1, true, s));

    // Cumulative sum from smallest to largest
    var cumsum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cumsum);
    try mlx.check(mlx.mlx_cumsum(&cumsum, sorted_probs, -1, false, true, s));

    // Find the cutoff: tokens where cumsum <= (1 - top_p) are outside the nucleus
    const threshold = mlx.mlx_array_new_float(1.0 - top_p);
    defer _ = mlx.mlx_array_free(threshold);

    var outside_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(outside_mask);
    try mlx.check(mlx.mlx_less_equal(&outside_mask, cumsum, threshold, s));

    // Set outside-nucleus logits to -inf in sorted space
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);

    // where(outside_mask, -inf, sorted_logits) — mask out the low-prob tokens
    try mlx.check(mlx.mlx_where(res, outside_mask, neg_inf, sorted_logits, s));

    // Note: categorical sampling doesn't care about token ordering,
    // but the sampled index will be in sorted space. We need to unsort.
    // Since categorical returns an index into the logits array, and we want
    // the original vocab index, we need to work in original space instead.

    // Better approach: find the minimum logit value that's in the nucleus,
    // then mask original logits below that threshold.
    _ = mlx.mlx_array_free(res.*);
    res.* = mlx.mlx_array_new();

    // The cutoff logit is the smallest logit still in the nucleus.
    // In sorted (ascending) order, tokens with cumsum > (1-top_p) are in nucleus.
    // The first such token's logit value is our threshold.
    // We can achieve this by: where(cumsum > 1-top_p, sorted_logits, +inf) then take min
    var in_nucleus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(in_nucleus);
    try mlx.check(mlx.mlx_greater(&in_nucleus, cumsum, threshold, s));

    const pos_inf = mlx.mlx_array_new_float(std.math.inf(f32));
    defer _ = mlx.mlx_array_free(pos_inf);

    var nucleus_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nucleus_logits);
    try mlx.check(mlx.mlx_where(&nucleus_logits, in_nucleus, sorted_logits, pos_inf, s));

    // Min value = the cutoff
    var min_val = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(min_val);
    try mlx.check(mlx.mlx_min_axis(&min_val, nucleus_logits, -1, true, s));

    // Mask original logits: keep if >= cutoff, else -inf
    var keep_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep_mask);
    try mlx.check(mlx.mlx_greater_equal(&keep_mask, logits, min_val, s));

    try mlx.check(mlx.mlx_where(res, keep_mask, logits, neg_inf, s));
}

/// Apply repeat penalty to already-generated tokens.
/// Uses pure MLX GPU ops — no CPU readback, preserves lazy evaluation graph.
fn applyRepeatPenalty(res: *mlx.mlx_array, logits: mlx.mlx_array, generated_ids: []const u32, repeat_penalty: f32, presence_penalty: f32, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);

    // Collect unique token ids
    var seen_set = std.AutoHashMap(u32, void).init(std.heap.page_allocator);
    defer seen_set.deinit();
    for (generated_ids) |id| {
        if (id < vocab_size) {
            seen_set.put(id, {}) catch continue;
        }
    }

    if (seen_set.count() == 0) return;

    // Build boolean mask: true at positions of seen tokens
    const mask_data = try std.heap.page_allocator.alloc(u8, vocab_size);
    defer std.heap.page_allocator.free(mask_data);
    @memset(mask_data, 0);

    var it = seen_set.keyIterator();
    while (it.next()) |id_ptr| {
        mask_data[id_ptr.*] = 1;
    }

    const arr_shape = [_]c_int{ 1, @intCast(vocab_size) };
    const mask_arr = mlx.mlx_array_new_data(mask_data.ptr, &arr_shape, 2, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);

    var current = logits;

    // Repeat penalty: multiply seen tokens by 1/penalty (positive) or penalty (negative)
    // This is equivalent to: where(mask & logits > 0, logits / penalty, where(mask, logits * penalty, logits))
    // Simplified: where(mask, where(logits > 0, logits / penalty, logits * penalty), logits)
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    if (repeat_penalty != 1.0) {
        const rp = mlx.mlx_array_new_float(repeat_penalty);
        defer _ = mlx.mlx_array_free(rp);
        const inv_rp = mlx.mlx_array_new_float(1.0 / repeat_penalty);
        defer _ = mlx.mlx_array_free(inv_rp);
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);

        // positive_mask = logits > 0
        var positive_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(positive_mask);
        try mlx.check(mlx.mlx_greater(&positive_mask, current, zero, s));

        // penalized_positive = logits * (1/penalty)
        var pen_pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_pos);
        try mlx.check(mlx.mlx_multiply(&pen_pos, current, inv_rp, s));

        // penalized_negative = logits * penalty
        var pen_neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_neg);
        try mlx.check(mlx.mlx_multiply(&pen_neg, current, rp, s));

        // sign_selected = where(positive, logits/penalty, logits*penalty)
        var sign_selected = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sign_selected);
        try mlx.check(mlx.mlx_where(&sign_selected, positive_mask, pen_pos, pen_neg, s));

        // result = where(mask, sign_selected, logits)
        try mlx.check(mlx.mlx_where(&penalized, mask_arr, sign_selected, current, s));
        current = penalized;
        penalized_owned = true;
    }

    // Presence penalty: subtract from seen tokens
    if (presence_penalty != 0.0) {
        const pp = mlx.mlx_array_new_float(presence_penalty);
        defer _ = mlx.mlx_array_free(pp);

        // Cast mask to float for arithmetic
        var mask_float = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mask_float);
        try mlx.check(mlx.mlx_astype(&mask_float, mask_arr, .float16, s));

        // subtract = mask * presence_penalty
        var subtract = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(subtract);
        try mlx.check(mlx.mlx_multiply(&subtract, mask_float, pp, s));

        // result = current - subtract
        try mlx.check(mlx.mlx_subtract(res, current, subtract, s));
    } else {
        try mlx.check(mlx.mlx_copy(res, current, s));
    }
}

/// Greedy argmax over the last axis.
fn argmax(last_logits: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var argmax_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(argmax_arr);
    try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, last_logits, -1, false, s));

    try mlx.check(mlx.mlx_array_eval(argmax_arr));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, argmax_arr));

    return @intCast(val);
}

// ── Tests ──

const testing = std.testing;

test "SamplingParams defaults" {
    const params = SamplingParams{};
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 0), params.top_k);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.repeat_penalty, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), params.presence_penalty, 0.001);
    try testing.expect(params.seed == null);
}

test "SamplingParams custom values" {
    const params = SamplingParams{
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
        .repeat_penalty = 1.1,
        .presence_penalty = 0.5,
        .seed = 42,
    };
    try testing.expectApproxEqAbs(@as(f32, 0.7), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.9), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 40), params.top_k);
    try testing.expectEqual(@as(u64, 42), params.seed.?);
}

test "GenerationResult fields" {
    // Just verifying the struct shape compiles correctly with all fields
    const result = GenerationResult{
        .text = @constCast("hello"),
        .token_ids = @constCast(&[_]u32{ 1, 2, 3 }),
        .prompt_tokens = 10,
        .completion_tokens = 3,
        .finish_reason = "stop",
        .prefill_tps = 100.0,
        .decode_tps = 35.0,
        .logprobs = null,
    };
    try testing.expectEqual(@as(u32, 10), result.prompt_tokens);
    try testing.expectEqual(@as(u32, 3), result.completion_tokens);
    try testing.expectEqualStrings("stop", result.finish_reason);
    try testing.expect(result.logprobs == null);
}

test "maskForLogitVocab pads and truncates to logits size" {
    const short_mask = [_]bool{ true, false, true };
    const padded = try maskForLogitVocab(testing.allocator, &short_mask, 5);
    defer padded.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 5), padded.slice.len);
    try testing.expect(padded.slice[0]);
    try testing.expect(!padded.slice[1]);
    try testing.expect(padded.slice[2]);
    try testing.expect(!padded.slice[3]);
    try testing.expect(!padded.slice[4]);

    const long_mask = [_]bool{ false, true, true, true };
    const truncated = try maskForLogitVocab(testing.allocator, &long_mask, 2);
    defer truncated.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), truncated.slice.len);
    try testing.expect(!truncated.slice[0]);
    try testing.expect(truncated.slice[1]);
}

test "argmax selects highest value" {
    // Create a simple logits array [1, 5] with values [0.1, 0.5, 0.9, 0.2, 0.3]
    const data = [_]f32{ 0.1, 0.5, 0.9, 0.2, 0.3 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 2), result); // index 2 has value 0.9
}

test "argmax with negative values" {
    const data = [_]f32{ -5.0, -1.0, -3.0, -0.5, -2.0 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 3), result); // index 3 has value -0.5 (highest)
}

test "applyRepeatPenalty reduces seen token logits" {
    const s = mlx.gpuStream();
    // logits: [1.0, 2.0, 3.0, 4.0, 5.0]
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const shape = [_]c_int{ 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize tokens at indices 1 and 3
    const generated = [_]u32{ 1, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 0.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: untouched → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[1], 0.01);
    // Index 2: untouched → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
    // Index 4: untouched → 5.0
    try testing.expectApproxEqAbs(@as(f32, 5.0), res_data[4], 0.01);
}

test "applyRepeatPenalty with negative logits" {
    const s = mlx.gpuStream();
    // Mix of positive and negative logits
    const data = [_]f32{ -2.0, 3.0, -1.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize all tokens
    const generated = [_]u32{ 0, 1, 2, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 0.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: negative, multiplied by 2.0 → -4.0
    try testing.expectApproxEqAbs(@as(f32, -4.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.5
    try testing.expectApproxEqAbs(@as(f32, 1.5), res_data[1], 0.01);
    // Index 2: negative, multiplied by 2.0 → -2.0
    try testing.expectApproxEqAbs(@as(f32, -2.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
}

test "applyRepeatPenalty presence penalty" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 2 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 1.0, 0.5, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: seen, presence penalty subtracted → 1.0 - 0.5 = 0.5
    try testing.expectApproxEqAbs(@as(f32, 0.5), res_data[0], 0.01);
    // Index 1: unseen → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[1], 0.01);
    // Index 2: seen → 3.0 - 0.5 = 2.5
    try testing.expectApproxEqAbs(@as(f32, 2.5), res_data[2], 0.01);
    // Index 3: unseen → 4.0
    try testing.expectApproxEqAbs(@as(f32, 4.0), res_data[3], 0.01);
}

test "applyRepeatPenalty combined penalties" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 2.0, -1.0, 3.0 };
    const shape = [_]c_int{ 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 1 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 1.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: positive, divide by 2.0 = 1.0, then - 1.0 = 0.0
    try testing.expectApproxEqAbs(@as(f32, 0.0), res_data[0], 0.01);
    // Index 1: negative, multiply by 2.0 = -2.0, then - 1.0 = -3.0
    try testing.expectApproxEqAbs(@as(f32, -3.0), res_data[1], 0.01);
    // Index 2: unseen → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
}

test "sampleToken greedy selects argmax" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // Create logits [1, 1, 5] — 5 vocab entries, token at index 3 has highest
    const data = [_]f32{ 1.0, 0.5, 0.1, 5.0, 0.2 };
    const logits_shape = [_]c_int{ 1, 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    try testing.expectEqual(@as(u32, 3), result.token_id);
}

test "sampleToken with temperature produces valid token" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    const data = [_]f32{ 1.0, 2.0, 3.0 };
    const logits_shape = [_]c_int{ 1, 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.5 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    // Token should be in valid range
    try testing.expect(result.token_id < 3);
}

test "sampleToken from prefill logits (seq_len > 1)" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // [1, 3, 4] — 3 positions, 4 vocab, should take last position
    const data = [_]f32{
        0.1, 0.2, 0.3, 0.4, // pos 0
        0.5, 0.6, 0.7, 0.8, // pos 1
        9.0, 0.1, 0.1, 0.1, // pos 2 — token 0 is clearly highest
    };
    const logits_shape = [_]c_int{ 1, 3, 4 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, s);
    try testing.expectEqual(@as(u32, 0), result.token_id); // pos 2, index 0 = 9.0
}
