//! Gemma 4 assistant drafter — speculative-decoding head.
//!
//! Mirrors upstream Python at
//! `mlx-vlm/speculative/drafters/gemma4_assistant/{gemma4_assistant,masked_embedder,masks}.py`.
//!
//! Mechanism:
//!   1. Drafter input per step: `concat([target.embed(prev_tok) * sqrt(target.hidden), h_prev], -1)`
//!      — shape `[1, 1, 2*backbone_hidden]`.
//!   2. `pre_projection` projects 5120 → drafter hidden 256.
//!   3. 4 cross-attention DecoderLayers (`kv_shared_only=True` — no K/V projections,
//!      no own KV cache; each layer reads layer-type-matching K/V from target).
//!   4. `model.norm` + `post_projection` → 2560 (next-step h_prev).
//!   5. Masked LM head (centroid-routed top-K sparse logits) → token logits.
//!
//! Position is constant across all `block_size - 1` drafter steps in a round.
//! Q rotates with `offset = target.cache.step + 1`; K is already RoPE'd by target.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");

const Weights = model_mod.Weights;
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;

pub const LayerType = enum {
    sliding_attention,
    full_attention,

    pub fn fromString(s: []const u8) !LayerType {
        if (std.mem.eql(u8, s, "sliding_attention")) return .sliding_attention;
        if (std.mem.eql(u8, s, "full_attention")) return .full_attention;
        return error.UnknownLayerType;
    }
};

pub const RopeKind = enum {
    /// Standard RoPE — fixed `theta`, full rotation across head_dim.
    default,
    /// Proportional RoPE — per-frequency scaling table; full layers in
    /// gemma-4-E4B drafter use this (partial_rotary_factor=0.25).
    proportional,
};

pub const DrafterRopeConfig = struct {
    kind: RopeKind = .default,
    theta: f32 = 10000.0,
    /// Used only when kind == .proportional.
    partial_rotary_factor: f32 = 0.25,
};

pub const DrafterConfig = struct {
    backbone_hidden_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    head_dim: u32,
    /// 0 → same as `head_dim`. 512 for E4B drafter (full layer).
    global_head_dim: u32 = 0,
    intermediate_size: u32,
    sliding_window: u32,
    layer_types: []LayerType, // owned
    rope_sliding: DrafterRopeConfig,
    rope_full: DrafterRopeConfig,
    rms_norm_eps: f32,
    vocab_size: u32,
    tie_word_embeddings: bool,
    use_ordered_embeddings: bool,
    num_centroids: u32,
    centroid_top_k: u32,
    block_size: u32 = 4,

    pub fn deinit(self: *DrafterConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.layer_types);
    }

    pub fn layerHeadDim(self: *const DrafterConfig, layer_idx: u32) u32 {
        if (layer_idx >= self.layer_types.len) return self.head_dim;
        if (self.layer_types[layer_idx] == .full_attention and self.global_head_dim > 0) {
            return self.global_head_dim;
        }
        return self.head_dim;
    }
};

/// Per-layer drafter weights. Cross-attention only — no K/V projections.
/// Linear weights are stored pre-transposed as `[in, out]` so `step()` can
/// use plain `mlx_matmul(out, x, w_t)` instead of paying a transpose per call.
pub const DrafterLayer = struct {
    layer_type: LayerType,
    head_dim: u32,
    n_heads: u32,

    // RMS norms
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    pre_ff_norm: mlx.mlx_array,
    post_ff_norm: mlx.mlx_array,

    // Q-only attention (q_w pre-transposed)
    q_w_t: mlx.mlx_array, // [hidden, n_heads*head_dim]
    q_norm: mlx.mlx_array,
    o_w_t: mlx.mlx_array, // [n_heads*head_dim, hidden]

    // MLP (pre-transposed)
    gate_w_t: mlx.mlx_array, // [hidden, intermediate]
    up_w_t: mlx.mlx_array, // [hidden, intermediate]
    down_w_t: mlx.mlx_array, // [intermediate, hidden]

    layer_scalar: mlx.mlx_array, // [1]
};

/// Centroid-routed sparse LM head. Mirrors `MaskedEmbedder` in
/// `masked_embedder.py`. Scores num_centroids clusters, picks top-K,
/// materializes K * (vocab/num_centroids) token logits densely, scatters
/// back into a dense `[B, L, vocab]` tensor with `min(selected_logits) − 1`
/// at non-selected positions.
pub const MaskedEmbedding = struct {
    /// `centroids.weight` pre-transposed → `[hidden, num_centroids]`.
    centroids_w_t: mlx.mlx_array,
    /// `token_ordering` reshaped to `[num_centroids, vocab_per_centroid]` (i32).
    ordering_2d: mlx.mlx_array,
    num_centroids: u32,
    top_k: u32,
    vocab_size: u32,
    vocab_per_centroid: u32,
};

pub const DrafterModel = struct {
    config: DrafterConfig,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,

    // bf16 weights
    embed_w: mlx.mlx_array, // [vocab, hidden]
    norm_w: mlx.mlx_array, // [hidden]
    pre_proj_w_t: mlx.mlx_array, // [2*backbone, hidden]
    post_proj_w_t: mlx.mlx_array, // [hidden, backbone]

    layers: []DrafterLayer,
    masked: ?MaskedEmbedding,

    /// Reference to target's `embed_tokens` — refcount-shared. Set in `bind`.
    target_embed_w: mlx.mlx_array,
    target_embed_scale: f32,

    /// Proportional-RoPE freqs for layers that use kind=proportional.
    rope_freqs_full: ?mlx.mlx_array = null,

    /// Map: layer_type → target KV-cache layer index. Filled in `bind`.
    target_kv_layer_for_type: [2]?u32 = .{ null, null },

    pub fn deinit(self: *DrafterModel) void {
        const allocator = self.allocator;
        _ = mlx.mlx_array_free(self.embed_w);
        _ = mlx.mlx_array_free(self.norm_w);
        _ = mlx.mlx_array_free(self.pre_proj_w_t);
        _ = mlx.mlx_array_free(self.post_proj_w_t);

        for (self.layers) |*lw| {
            _ = mlx.mlx_array_free(lw.input_norm);
            _ = mlx.mlx_array_free(lw.post_attn_norm);
            _ = mlx.mlx_array_free(lw.pre_ff_norm);
            _ = mlx.mlx_array_free(lw.post_ff_norm);
            _ = mlx.mlx_array_free(lw.q_w_t);
            _ = mlx.mlx_array_free(lw.q_norm);
            _ = mlx.mlx_array_free(lw.o_w_t);
            _ = mlx.mlx_array_free(lw.gate_w_t);
            _ = mlx.mlx_array_free(lw.up_w_t);
            _ = mlx.mlx_array_free(lw.down_w_t);
            _ = mlx.mlx_array_free(lw.layer_scalar);
        }
        allocator.free(self.layers);

        if (self.masked) |*m| {
            _ = mlx.mlx_array_free(m.centroids_w_t);
            _ = mlx.mlx_array_free(m.ordering_2d);
        }
        if (self.rope_freqs_full) |f| _ = mlx.mlx_array_free(f);

        self.config.deinit(allocator);
    }

    /// Bind the drafter to a target Transformer. Captures `embed_tokens` (via
    /// refcount) and the `embed_scale`, computes per-layer-type KV mapping.
    pub fn bind(self: *DrafterModel, target: *Transformer) !void {
        if (self.config.backbone_hidden_size != target.config.hidden_size) {
            log.err("[drafter] config mismatch: backbone_hidden_size={d}, target.hidden_size={d}\n", .{
                self.config.backbone_hidden_size, target.config.hidden_size,
            });
            return error.DrafterTargetMismatch;
        }

        // Refcount-share the target's embed_w. The target's emb_w is
        // dequantized to bf16 lazily by the target's own `embedding`; for the
        // drafter input concat we want the same already-dequantized bf16
        // values. The drafter therefore re-runs the target's embedding lookup
        // path through its own copy of `embed_w` (== target's embed weight).
        // Actually simpler: just take the target's emb_w via `mlx_array_set`
        // — same buffer, two refcounts. The drafter dequantizes itself in
        // `embedTargetToken`.
        _ = mlx.mlx_array_free(self.target_embed_w);
        var owned = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&owned, target.emb_w));
        self.target_embed_w = owned;
        self.target_embed_scale = if (target.emb_scale != null)
            @sqrt(@as(f32, @floatFromInt(target.config.hidden_size)))
        else
            1.0;

        // Per-layer-type KV mapping. We want the *last* target layer of each
        // type within the kv-source region (layers 0..N-num_kv_shared, where
        // num_kv_shared > 0 means the last K layers alias earlier ones).
        const tcfg = &target.config;
        const num_shared = tcfg.num_kv_shared_layers;
        const N = tcfg.num_hidden_layers;
        const kv_source_end: u32 = if (num_shared > 0) N - num_shared else N;
        var saw_sliding: ?u32 = null;
        var saw_full: ?u32 = null;
        var li: u32 = 0;
        while (li < kv_source_end) : (li += 1) {
            if (tcfg.isGlobalLayer(li)) saw_full = li else saw_sliding = li;
        }
        self.target_kv_layer_for_type[@intFromEnum(LayerType.sliding_attention)] = saw_sliding;
        self.target_kv_layer_for_type[@intFromEnum(LayerType.full_attention)] = saw_full;

        for (self.config.layer_types) |lt| {
            if (self.target_kv_layer_for_type[@intFromEnum(lt)] == null) {
                log.err("[drafter] no target layer of type {s} found (target has {d} hidden layers, {d} kv-shared)\n", .{
                    @tagName(lt), N, num_shared,
                });
                return error.DrafterTargetMismatch;
            }
        }
    }
};

// ── Configuration parsing ──

pub fn parseConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !DrafterConfig {
    const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(path);

    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const content = try reader_state.interface.allocRemaining(allocator, .limited(1 << 20));
    defer allocator.free(content);

    return parseConfigFromJson(allocator, content);
}

pub fn parseConfigFromJson(allocator: std.mem.Allocator, content: []const u8) !DrafterConfig {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const model_type = if (root.get("model_type")) |v| v.string else "";
    if (!std.mem.eql(u8, model_type, "gemma4_assistant")) {
        log.err("[drafter] unsupported drafter model_type='{s}' (expected 'gemma4_assistant')\n", .{model_type});
        return error.UnsupportedDrafterArch;
    }

    const backbone = if (root.get("backbone_hidden_size")) |v| @as(u32, @intCast(v.integer)) else 0;
    if (backbone == 0) return error.MissingBackboneHiddenSize;

    const num_centroids: u32 = if (root.get("num_centroids")) |v| @intCast(v.integer) else 0;
    const top_k: u32 = if (root.get("centroid_intermediate_top_k")) |v| @intCast(v.integer) else 0;
    const use_ordered: bool = if (root.get("use_ordered_embeddings")) |v| v.bool else false;
    const tie: bool = if (root.get("tie_word_embeddings")) |v| v.bool else true;

    const tc_val = root.get("text_config") orelse return error.MissingTextConfig;
    const tc = tc_val.object;

    const hidden_size: u32 = if (tc.get("hidden_size")) |v| @intCast(v.integer) else 0;
    const num_layers: u32 = if (tc.get("num_hidden_layers")) |v| @intCast(v.integer) else 0;
    const n_heads: u32 = if (tc.get("num_attention_heads")) |v| @intCast(v.integer) else 0;
    const head_dim: u32 = if (tc.get("head_dim")) |v| @intCast(v.integer) else 0;
    const global_head_dim: u32 = if (tc.get("global_head_dim")) |v| @intCast(v.integer) else 0;
    const intermediate: u32 = if (tc.get("intermediate_size")) |v| @intCast(v.integer) else 0;
    const sliding: u32 = if (tc.get("sliding_window")) |v| @intCast(v.integer) else 0;
    const vocab_size: u32 = if (tc.get("vocab_size")) |v| @intCast(v.integer) else 0;
    const eps_val = tc.get("rms_norm_eps") orelse return error.MissingRmsNormEps;
    const eps: f32 = jsonFloat(eps_val);
    if (hidden_size == 0 or num_layers == 0 or n_heads == 0 or head_dim == 0 or vocab_size == 0)
        return error.IncompleteTextConfig;

    const lt_val = tc.get("layer_types") orelse return error.MissingLayerTypes;
    const lt_arr = switch (lt_val) {
        .array => |a| a,
        else => return error.InvalidLayerTypes,
    };
    if (lt_arr.items.len != num_layers) {
        log.err("[drafter] layer_types length {d} != num_hidden_layers {d}\n", .{ lt_arr.items.len, num_layers });
        return error.LayerTypesLengthMismatch;
    }
    const layer_types = try allocator.alloc(LayerType, num_layers);
    errdefer allocator.free(layer_types);
    for (lt_arr.items, 0..) |elem, i| {
        const name = switch (elem) {
            .string => |s| s,
            else => return error.InvalidLayerTypeEntry,
        };
        layer_types[i] = try LayerType.fromString(name);
    }

    var rope_sliding: DrafterRopeConfig = .{ .kind = .default, .theta = 10000.0 };
    var rope_full: DrafterRopeConfig = .{ .kind = .default, .theta = 10000.0 };
    if (tc.get("rope_parameters")) |rp_val| {
        if (rp_val == .object) {
            if (rp_val.object.get("sliding_attention")) |slid| if (slid == .object) {
                if (slid.object.get("rope_theta")) |t| rope_sliding.theta = jsonFloat(t);
            };
            if (rp_val.object.get("full_attention")) |full| if (full == .object) {
                if (full.object.get("rope_theta")) |t| rope_full.theta = jsonFloat(t);
                if (full.object.get("rope_type")) |rt| if (rt == .string) {
                    if (std.mem.eql(u8, rt.string, "proportional")) rope_full.kind = .proportional;
                };
                if (full.object.get("partial_rotary_factor")) |pf| {
                    rope_full.partial_rotary_factor = jsonFloat(pf);
                }
            };
        }
    }

    return DrafterConfig{
        .backbone_hidden_size = backbone,
        .hidden_size = hidden_size,
        .num_hidden_layers = num_layers,
        .num_attention_heads = n_heads,
        .head_dim = head_dim,
        .global_head_dim = global_head_dim,
        .intermediate_size = intermediate,
        .sliding_window = sliding,
        .layer_types = layer_types,
        .rope_sliding = rope_sliding,
        .rope_full = rope_full,
        .rms_norm_eps = eps,
        .vocab_size = vocab_size,
        .tie_word_embeddings = tie,
        .use_ordered_embeddings = use_ordered,
        .num_centroids = num_centroids,
        .centroid_top_k = top_k,
        .block_size = 4,
    };
}

fn jsonFloat(v: std.json.Value) f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => 0.0,
    };
}

// ── Helpers ──

/// Take ownership of a weight by refcount-share. The original is freed when
/// `Weights.deinit` runs; the returned array survives independently.
fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[drafter] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingDrafterWeight;
    };
    var owned = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&owned, arr));
    return owned;
}

/// Pre-transpose a 2-D weight stored as `[out, in]` to `[in, out]` so
/// `mlx_matmul(y, x, w_t)` computes `y = x @ w.T`. Frees the input array.
fn ownAndTranspose2D(w: *const Weights, key: []const u8, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = try ownWeight(w, key);
    defer _ = mlx.mlx_array_free(raw);
    var transposed = mlx.mlx_array_new();
    const perm = [_]c_int{ 1, 0 };
    try mlx.check(mlx.mlx_transpose_axes(&transposed, raw, &perm, 2, s));
    return transposed;
}

fn freeAll(arrs: []const mlx.mlx_array) void {
    for (arrs) |a| _ = mlx.mlx_array_free(a);
}

// ── Weight loading ──

/// Load a drafter from `model_dir`. After loading, call `bind(target)`.
pub fn loadDrafter(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    model_dir: []const u8,
) !DrafterModel {
    var cfg = try parseConfig(io, allocator, model_dir);
    errdefer cfg.deinit(allocator);

    var weights = try model_mod.loadWeights(io, allocator, model_dir);
    defer weights.deinit();

    // Globals — embed/norm don't need transposing (used in matmul-as-take and rmsnorm).
    const embed_w = try ownWeight(&weights, "model.embed_tokens.weight");
    errdefer _ = mlx.mlx_array_free(embed_w);
    const norm_w = try ownWeight(&weights, "model.norm.weight");
    errdefer _ = mlx.mlx_array_free(norm_w);
    const pre_proj_w_t = try ownAndTranspose2D(&weights, "pre_projection.weight", s);
    errdefer _ = mlx.mlx_array_free(pre_proj_w_t);
    const post_proj_w_t = try ownAndTranspose2D(&weights, "post_projection.weight", s);
    errdefer _ = mlx.mlx_array_free(post_proj_w_t);

    const layers = try allocator.alloc(DrafterLayer, cfg.num_hidden_layers);
    errdefer allocator.free(layers);
    var layers_inited: u32 = 0;
    errdefer {
        var i: u32 = 0;
        while (i < layers_inited) : (i += 1) {
            const lw = &layers[i];
            _ = mlx.mlx_array_free(lw.input_norm);
            _ = mlx.mlx_array_free(lw.post_attn_norm);
            _ = mlx.mlx_array_free(lw.pre_ff_norm);
            _ = mlx.mlx_array_free(lw.post_ff_norm);
            _ = mlx.mlx_array_free(lw.q_w_t);
            _ = mlx.mlx_array_free(lw.q_norm);
            _ = mlx.mlx_array_free(lw.o_w_t);
            _ = mlx.mlx_array_free(lw.gate_w_t);
            _ = mlx.mlx_array_free(lw.up_w_t);
            _ = mlx.mlx_array_free(lw.down_w_t);
            _ = mlx.mlx_array_free(lw.layer_scalar);
        }
    }

    var key_buf: [256]u8 = undefined;
    var li: u32 = 0;
    while (li < cfg.num_hidden_layers) : (li += 1) {
        const lt = cfg.layer_types[li];
        const layer_hd = cfg.layerHeadDim(li);

        layers[li] = .{
            .layer_type = lt,
            .head_dim = layer_hd,
            .n_heads = cfg.num_attention_heads,
            .input_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.input_layernorm.weight", .{li})),
            .post_attn_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.post_attention_layernorm.weight", .{li})),
            .pre_ff_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.pre_feedforward_layernorm.weight", .{li})),
            .post_ff_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.post_feedforward_layernorm.weight", .{li})),
            .q_w_t = try ownAndTranspose2D(&weights, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.self_attn.q_proj.weight", .{li}), s),
            .q_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.self_attn.q_norm.weight", .{li})),
            .o_w_t = try ownAndTranspose2D(&weights, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.self_attn.o_proj.weight", .{li}), s),
            .gate_w_t = try ownAndTranspose2D(&weights, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.gate_proj.weight", .{li}), s),
            .up_w_t = try ownAndTranspose2D(&weights, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.up_proj.weight", .{li}), s),
            .down_w_t = try ownAndTranspose2D(&weights, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.down_proj.weight", .{li}), s),
            .layer_scalar = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.layer_scalar", .{li})),
        };
        layers_inited += 1;
    }

    // Masked LM head
    var masked: ?MaskedEmbedding = null;
    errdefer if (masked) |*m| {
        _ = mlx.mlx_array_free(m.centroids_w_t);
        _ = mlx.mlx_array_free(m.ordering_2d);
    };
    if (cfg.use_ordered_embeddings) {
        if (cfg.num_centroids == 0 or cfg.vocab_size % cfg.num_centroids != 0) {
            log.err("[drafter] invalid centroid config: num_centroids={d}, vocab_size={d}\n", .{ cfg.num_centroids, cfg.vocab_size });
            return error.InvalidCentroidConfig;
        }
        const centroids_w_t = try ownAndTranspose2D(&weights, "masked_embedding.centroids.weight", s);
        errdefer _ = mlx.mlx_array_free(centroids_w_t);

        const ordering_raw = try ownWeight(&weights, "masked_embedding.token_ordering");
        defer _ = mlx.mlx_array_free(ordering_raw);

        var ordering_i32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ordering_i32);
        try mlx.check(mlx.mlx_astype(&ordering_i32, ordering_raw, .int32, s));

        const vpc = cfg.vocab_size / cfg.num_centroids;
        const reshape_shape = [_]c_int{ @intCast(cfg.num_centroids), @intCast(vpc) };
        var ordering_2d = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&ordering_2d, ordering_i32, &reshape_shape, 2, s));

        masked = .{
            .centroids_w_t = centroids_w_t,
            .ordering_2d = ordering_2d,
            .num_centroids = cfg.num_centroids,
            .top_k = cfg.centroid_top_k,
            .vocab_size = cfg.vocab_size,
            .vocab_per_centroid = vpc,
        };
    }

    // Proportional-RoPE freqs for the full layer (matches transformer.zig:1072-1126).
    var rope_freqs_full: ?mlx.mlx_array = null;
    errdefer if (rope_freqs_full) |f| {
        _ = mlx.mlx_array_free(f);
    };
    if (cfg.rope_full.kind == .proportional) {
        const ghd = if (cfg.global_head_dim > 0) cfg.global_head_dim else cfg.head_dim;
        const rotated_dims: u32 = @intFromFloat(@as(f32, @floatFromInt(ghd)) * cfg.rope_full.partial_rotary_factor);
        const n_rotated: u32 = rotated_dims / 2;
        const n_pad: u32 = (ghd - rotated_dims) / 2;

        var arange_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(arange_arr);
        try mlx.check(mlx.mlx_arange(&arange_arr, 0, @floatFromInt(rotated_dims), 2, .float32, s));

        const ghd_scalar = mlx.mlx_array_new_float(@floatFromInt(ghd));
        defer _ = mlx.mlx_array_free(ghd_scalar);
        var exponents = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(exponents);
        try mlx.check(mlx.mlx_divide(&exponents, arange_arr, ghd_scalar, s));

        const base_scalar = mlx.mlx_array_new_float(cfg.rope_full.theta);
        defer _ = mlx.mlx_array_free(base_scalar);
        var base_pow = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(base_pow);
        try mlx.check(mlx.mlx_power(&base_pow, base_scalar, exponents, s));

        var freqs_arr = mlx.mlx_array_new();
        if (n_pad > 0) {
            const pad_shape = [_]c_int{@intCast(n_pad)};
            const inf_val = mlx.mlx_array_new_float(std.math.inf(f32));
            defer _ = mlx.mlx_array_free(inf_val);
            var inf_arr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(inf_arr);
            try mlx.check(mlx.mlx_full(&inf_arr, &pad_shape, 1, inf_val, .float32, s));
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, base_pow);
            _ = mlx.mlx_vector_array_append_value(vec, inf_arr);
            try mlx.check(mlx.mlx_concatenate_axis(&freqs_arr, vec, 0, s));
        } else {
            const total_shape = [_]c_int{@intCast(n_rotated + n_pad)};
            try mlx.check(mlx.mlx_reshape(&freqs_arr, base_pow, &total_shape, 1, s));
        }
        rope_freqs_full = freqs_arr;
    }

    // Eagerly evaluate all weights so they're materialized on the GPU.
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        _ = mlx.mlx_vector_array_append_value(eval_vec, embed_w);
        _ = mlx.mlx_vector_array_append_value(eval_vec, norm_w);
        _ = mlx.mlx_vector_array_append_value(eval_vec, pre_proj_w_t);
        _ = mlx.mlx_vector_array_append_value(eval_vec, post_proj_w_t);
        for (layers) |lw| {
            _ = mlx.mlx_vector_array_append_value(eval_vec, lw.input_norm);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lw.post_attn_norm);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lw.pre_ff_norm);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lw.post_ff_norm);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lw.q_w_t);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lw.q_norm);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lw.o_w_t);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lw.gate_w_t);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lw.up_w_t);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lw.down_w_t);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lw.layer_scalar);
        }
        if (masked) |m| {
            _ = mlx.mlx_vector_array_append_value(eval_vec, m.centroids_w_t);
            _ = mlx.mlx_vector_array_append_value(eval_vec, m.ordering_2d);
        }
        if (rope_freqs_full) |f| _ = mlx.mlx_vector_array_append_value(eval_vec, f);
        _ = mlx.mlx_eval(eval_vec);
    }

    log.info("[drafter] loaded {d} layers, vocab={d}, hidden={d} → backbone {d}\n", .{
        cfg.num_hidden_layers, cfg.vocab_size, cfg.hidden_size, cfg.backbone_hidden_size,
    });

    return DrafterModel{
        .config = cfg,
        .allocator = allocator,
        .s = s,
        .embed_w = embed_w,
        .norm_w = norm_w,
        .pre_proj_w_t = pre_proj_w_t,
        .post_proj_w_t = post_proj_w_t,
        .layers = layers,
        .masked = masked,
        .target_embed_w = mlx.mlx_array_new(),
        .target_embed_scale = 1.0,
        .rope_freqs_full = rope_freqs_full,
    };
}

// ── Forward step ──

pub const DrafterStepOut = struct {
    /// Logits over the full vocab — `[1, 1, vocab_size]`. Caller frees.
    logits: mlx.mlx_array,
    /// Next-step h_prev — `[1, 1, backbone_hidden_size]`. Caller frees.
    h_prev_next: mlx.mlx_array,
};

/// Lookup the embedding for a single token id from a quantized weight.
/// Mirrors `Transformer.embedding` but produces a `[1, 1, hidden]` bf16 array
/// for a single token (the drafter only ever embeds one token per step).
fn embedSingleToken(
    embed_w: mlx.mlx_array,
    quant_group_size: u32,
    quant_bits: u32,
    token_id: u32,
    target_hidden_size: u32,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const id_i32: i32 = @intCast(token_id);
    const id_shape = [_]c_int{1};
    const id_arr = mlx.mlx_array_new_data(&id_i32, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(id_arr);

    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, embed_w, id_arr, 0, s));

    // Reshape to [1, 1, H]. If embed_w was bf16 (drafter checkpoint), `taken`
    // is already `[1, H]` bf16. If embed_w is quantized (target's emb_w), we
    // can't just reshape — we need to dequantize. The path used here is for
    // the drafter's *own* embed_w, which is bf16, so the reshape suffices.
    _ = quant_group_size;
    _ = quant_bits;

    var out = mlx.mlx_array_new();
    const out_shape = [_]c_int{ 1, 1, @intCast(target_hidden_size) };
    try mlx.check(mlx.mlx_reshape(&out, taken, &out_shape, 3, s));
    return out;
}

/// Embed a single token using the *target's* possibly-quantized embed_w.
/// Returns shape `[1, 1, target.hidden_size]` in bf16.
fn embedTargetToken(
    self: *const DrafterModel,
    target: *Transformer,
    token_id: u32,
) !mlx.mlx_array {
    // Mirror Transformer.embedding (src/transformer.zig:1808): take_axis on each
    // of (w, s, b) by token_id, then dequantize to bf16. The target's emb_w
    // is quantized; emb_s / emb_b are its scales and biases.
    const id_i32: i32 = @intCast(token_id);
    const id_shape = [_]c_int{1};
    const id_arr = mlx.mlx_array_new_data(&id_i32, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(id_arr);

    var tw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tw);
    try mlx.check(mlx.mlx_take_axis(&tw, target.emb_w, id_arr, 0, self.s));

    var ts = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ts);
    try mlx.check(mlx.mlx_take_axis(&ts, target.emb_s, id_arr, 0, self.s));

    var tb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tb);
    try mlx.check(mlx.mlx_take_axis(&tb, target.emb_b, id_arr, 0, self.s));

    var dequant = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dequant);
    try mlx.check(mlx.mlx_dequantize(
        &dequant,
        tw,
        ts,
        tb,
        mlx.mlx_optional_int.some(@intCast(target.config.quant_group_size)),
        mlx.mlx_optional_int.some(@intCast(target.config.quant_bits)),
        "affine",
        .{}, // global_scale
        .{ .value = .bfloat16, .has_value = true },
        self.s,
    ));

    var emb = mlx.mlx_array_new();
    const out_shape = [_]c_int{ 1, 1, @intCast(target.config.hidden_size) };
    try mlx.check(mlx.mlx_reshape(&emb, dequant, &out_shape, 3, self.s));

    // Apply embed_scale (Gemma 4: sqrt(target.hidden_size)).
    if (self.target_embed_scale != 1.0) {
        const sc = mlx.mlx_array_new_float(self.target_embed_scale);
        defer _ = mlx.mlx_array_free(sc);
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, emb, sc, self.s));
        _ = mlx.mlx_array_free(emb);
        return scaled;
    }
    return emb;
}

inline fn rmsNormFn(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, s));
    return out;
}

inline fn matmulFn(x: mlx.mlx_array, w_t: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&out, x, w_t, s));
    return out;
}

/// Approximate GeGLU (matches `geglu` in upstream gemma4 language.py):
/// `gelu_approx(gate) * up`.
fn geglu(gate: mlx.mlx_array, up: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    // mlx_lm uses `gelu_approx(g) * x`. The `gelu_approx` op exists in mlx-c
    // as `mlx_gelu_approx` (commonly), but our FFI may not expose it — fall
    // back to `0.5 * gate * (1 + tanh(sqrt(2/pi) * (gate + 0.044715 * gate^3)))`.
    // Simpler exact-equivalent path: use `mlx_gelu_fast_approx` if present.
    // For robust correctness we compute manually with mlx ops.
    const c0 = mlx.mlx_array_new_float(0.7978845608028654); // sqrt(2/pi)
    defer _ = mlx.mlx_array_free(c0);
    const c1 = mlx.mlx_array_new_float(0.044715);
    defer _ = mlx.mlx_array_free(c1);
    const half = mlx.mlx_array_new_float(0.5);
    defer _ = mlx.mlx_array_free(half);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);

    var g3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g3);
    var g_sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_sq);
    try mlx.check(mlx.mlx_multiply(&g_sq, gate, gate, s));
    try mlx.check(mlx.mlx_multiply(&g3, g_sq, gate, s));

    var c1_g3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c1_g3);
    try mlx.check(mlx.mlx_multiply(&c1_g3, g3, c1, s));

    var inner = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inner);
    try mlx.check(mlx.mlx_add(&inner, gate, c1_g3, s));

    var inner_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inner_scaled);
    try mlx.check(mlx.mlx_multiply(&inner_scaled, inner, c0, s));

    var tanh_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tanh_out);
    try mlx.check(mlx.mlx_tanh(&tanh_out, inner_scaled, s));

    var one_plus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one_plus);
    try mlx.check(mlx.mlx_add(&one_plus, tanh_out, one, s));

    var half_g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(half_g);
    try mlx.check(mlx.mlx_multiply(&half_g, gate, half, s));

    var gelu_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gelu_out);
    try mlx.check(mlx.mlx_multiply(&gelu_out, half_g, one_plus, s));

    var prod = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&prod, gelu_out, up, s));
    return prod;
}

/// Build a bidirectional sliding-window mask of shape `[1, 1, query_len, kv_len]`.
/// For drafter cross-attn we have query_len=1 and the query position lives at
/// `query_offset`, so this reduces to `bias[0, k] = 0 if (q-k > -W) and (q-k < W)
/// and (k <= query_offset)` else `-inf`.
/// Returns null when no masking is needed (entire KV fits in the window).
fn buildSwaMask(
    query_offset: c_int,
    kv_len: c_int,
    window: c_int,
    s: mlx.mlx_stream,
) !?mlx.mlx_array {
    // Skip when KV fits entirely inside the bidirectional window for the query.
    if (kv_len <= window and query_offset + 1 <= kv_len + window) return null;

    // q_idx = query_offset (scalar); k_idx = arange(kv_len)
    var k_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_idx);
    try mlx.check(mlx.mlx_arange(&k_idx, 0, @floatFromInt(kv_len), 1, .int32, s));

    const q_scalar = mlx.mlx_array_new_int(query_offset);
    defer _ = mlx.mlx_array_free(q_scalar);

    var dist = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dist);
    try mlx.check(mlx.mlx_subtract(&dist, q_scalar, k_idx, s));

    const neg_w = mlx.mlx_array_new_int(-window);
    defer _ = mlx.mlx_array_free(neg_w);
    const w_scalar = mlx.mlx_array_new_int(window);
    defer _ = mlx.mlx_array_free(w_scalar);
    const off_scalar = mlx.mlx_array_new_int(query_offset);
    defer _ = mlx.mlx_array_free(off_scalar);

    var dist_gt_negw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dist_gt_negw);
    try mlx.check(mlx.mlx_greater(&dist_gt_negw, dist, neg_w, s));

    var dist_lt_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dist_lt_w);
    try mlx.check(mlx.mlx_less(&dist_lt_w, dist, w_scalar, s));

    var k_le_off = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_le_off);
    try mlx.check(mlx.mlx_less_equal(&k_le_off, k_idx, off_scalar, s));

    var inside = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inside);
    try mlx.check(mlx.mlx_logical_and(&inside, dist_gt_negw, dist_lt_w, s));
    var inside2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inside2);
    try mlx.check(mlx.mlx_logical_and(&inside2, inside, k_le_off, s));

    const zero_v = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero_v);
    const ninf_v = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(ninf_v);

    var bias = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bias);
    try mlx.check(mlx.mlx_where(&bias, inside2, zero_v, ninf_v, s));

    var bias_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bias_bf);
    try mlx.check(mlx.mlx_astype(&bias_bf, bias, .bfloat16, s));

    var out = mlx.mlx_array_new();
    const reshape = [_]c_int{ 1, 1, 1, kv_len };
    try mlx.check(mlx.mlx_reshape(&out, bias_bf, &reshape, 4, s));
    return out;
}

/// Build a full-attention mask `[1, 1, 1, kv_len]` allowing positions `0..=query_offset`.
/// Returns null when query_offset+1 == kv_len (no masking needed).
fn buildFullMask(
    query_offset: c_int,
    kv_len: c_int,
    s: mlx.mlx_stream,
) !?mlx.mlx_array {
    if (query_offset + 1 >= kv_len) return null;

    var k_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_idx);
    try mlx.check(mlx.mlx_arange(&k_idx, 0, @floatFromInt(kv_len), 1, .int32, s));

    const off_scalar = mlx.mlx_array_new_int(query_offset);
    defer _ = mlx.mlx_array_free(off_scalar);

    var inside = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inside);
    try mlx.check(mlx.mlx_less_equal(&inside, k_idx, off_scalar, s));

    const zero_v = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero_v);
    const ninf_v = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(ninf_v);

    var bias = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bias);
    try mlx.check(mlx.mlx_where(&bias, inside, zero_v, ninf_v, s));

    var bias_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bias_bf);
    try mlx.check(mlx.mlx_astype(&bias_bf, bias, .bfloat16, s));

    var out = mlx.mlx_array_new();
    const reshape = [_]c_int{ 1, 1, 1, kv_len };
    try mlx.check(mlx.mlx_reshape(&out, bias_bf, &reshape, 4, s));
    return out;
}

/// Cross-attention forward through one drafter layer. Reads K/V from
/// `target.cache.entries[kv_layer_idx]` (refcount-shared via key_view/value_view).
fn layerForward(
    self: *const DrafterModel,
    layer: *const DrafterLayer,
    h: mlx.mlx_array,
    target: *Transformer,
    rope_offset: c_int,
) !mlx.mlx_array {
    const eps = self.config.rms_norm_eps;
    const s = self.s;
    const n_heads: c_int = @intCast(layer.n_heads);
    const head_dim: c_int = @intCast(layer.head_dim);

    // ── Pre-attn norm ──
    const residual = h; // borrowed; caller must NOT free until we return
    const x_normed = try rmsNormFn(h, layer.input_norm, eps, s);
    defer _ = mlx.mlx_array_free(x_normed);

    // ── Q projection + Q norm + RoPE ──
    const q_proj = try matmulFn(x_normed, layer.q_w_t, s);
    defer _ = mlx.mlx_array_free(q_proj);

    const q_shape = [_]c_int{ 1, 1, n_heads, head_dim };
    var q_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_r);
    try mlx.check(mlx.mlx_reshape(&q_r, q_proj, &q_shape, 4, s));

    const q_normed = try rmsNormFn(q_r, layer.q_norm, eps, s);
    defer _ = mlx.mlx_array_free(q_normed);

    // [B, S, H, D] → [B, H, S, D]
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var q_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_t);
    try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, s));

    // RoPE
    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    if (layer.layer_type == .full_attention and self.rope_freqs_full != null) {
        // Proportional RoPE: freqs is the precomputed table; base is unused.
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, head_dim, false, .{}, 1.0, rope_offset, self.rope_freqs_full.?, s));
    } else {
        const theta = if (layer.layer_type == .full_attention) self.config.rope_full.theta else self.config.rope_sliding.theta;
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, head_dim, false, mlx.mlx_optional_float.some(theta), 1.0, rope_offset, .{ .ctx = null }, s));
    }

    // ── Read shared K/V from target ──
    const kv_layer_idx = self.target_kv_layer_for_type[@intFromEnum(layer.layer_type)] orelse
        return error.DrafterTargetMismatch;
    const tgt_entry = &target.cache.entries[kv_layer_idx];
    if (!tgt_entry.initialized) return error.TargetCacheUninitialized;

    const full_k = tgt_entry.key_view;
    const full_v = tgt_entry.value_view;

    // Build mask
    const k_shape = mlx.getShape(full_k);
    const kv_len = k_shape[2];
    var mask_owned: ?mlx.mlx_array = null;
    defer if (mask_owned) |m| {
        _ = mlx.mlx_array_free(m);
    };
    mask_owned = if (layer.layer_type == .sliding_attention)
        try buildSwaMask(rope_offset, kv_len, @intCast(self.config.sliding_window), s)
    else
        try buildFullMask(rope_offset, kv_len, s);

    // SDPA
    const scale: f32 = 1.0; // Gemma 4 uses scale=1.0 (q_norm absorbs it)
    var attn_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_out);
    if (mask_owned) |mask| {
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, scale, "array", mask, .{ .ctx = null }, s));
    } else {
        const none_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none_mask);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, scale, "", none_mask, .{ .ctx = null }, s));
    }

    // [B, H, S, D] → [B, S, H*D]
    const perm_back = [_]c_int{ 0, 2, 1, 3 };
    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, s));
    const flat_shape = [_]c_int{ 1, 1, n_heads * head_dim };
    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, s));

    // O projection
    const o_proj = try matmulFn(attn_flat, layer.o_w_t, s);
    defer _ = mlx.mlx_array_free(o_proj);

    // post_attn_norm + residual add
    const o_normed = try rmsNormFn(o_proj, layer.post_attn_norm, eps, s);
    defer _ = mlx.mlx_array_free(o_normed);
    var h_attn = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&h_attn, residual, o_normed, s));

    // ── MLP ──
    const ff_residual = h_attn;
    const ff_in = try rmsNormFn(h_attn, layer.pre_ff_norm, eps, s);
    defer _ = mlx.mlx_array_free(ff_in);

    const gate = try matmulFn(ff_in, layer.gate_w_t, s);
    defer _ = mlx.mlx_array_free(gate);
    const up = try matmulFn(ff_in, layer.up_w_t, s);
    defer _ = mlx.mlx_array_free(up);

    const act = try geglu(gate, up, s);
    defer _ = mlx.mlx_array_free(act);
    const down = try matmulFn(act, layer.down_w_t, s);
    defer _ = mlx.mlx_array_free(down);

    const ff_normed = try rmsNormFn(down, layer.post_ff_norm, eps, s);
    defer _ = mlx.mlx_array_free(ff_normed);

    var h_ff = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&h_ff, ff_residual, ff_normed, s));
    _ = mlx.mlx_array_free(ff_residual);

    // Layer scalar (broadcast multiply).
    var h_scaled = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&h_scaled, h_ff, layer.layer_scalar, s));
    _ = mlx.mlx_array_free(h_ff);

    return h_scaled;
}

/// Compute logits via the masked LM head.
/// Returns a `[1, 1, vocab]` bf16 array with `min(selected) - 1` everywhere
/// except the K * vocab_per_centroid materialized positions.
fn maskedLmHead(
    self: *const DrafterModel,
    h: mlx.mlx_array,
) !mlx.mlx_array {
    const m = self.masked.?;
    const s = self.s;
    const top_k: c_int = @intCast(m.top_k);
    const vpc: c_int = @intCast(m.vocab_per_centroid);
    const n_selected: c_int = top_k * vpc;

    // h: [1, 1, hidden]
    // centroid_logits = h @ centroids.T: [1, 1, num_centroids]
    const centroid_logits = try matmulFn(h, m.centroids_w_t, s);
    defer _ = mlx.mlx_array_free(centroid_logits);

    // top-K indices via argpartition along last axis.
    var topk_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_idx);
    try mlx.check(mlx.mlx_argpartition_axis(&topk_idx, centroid_logits, -top_k, -1, s));

    // Slice last K → [1, 1, top_k]
    const ci_shape = mlx.getShape(centroid_logits);
    const start = [_]c_int{ 0, 0, ci_shape[2] - top_k };
    const stop = [_]c_int{ ci_shape[0], ci_shape[1], ci_shape[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    var topk_idx_sl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(topk_idx_sl);
    try mlx.check(mlx.mlx_slice(&topk_idx_sl, topk_idx, &start, 3, &stop, 3, &strides, 3, s));

    // selected_canonical[b,l,k,j] = ordering_2d[topk_idx_sl[b,l,k], j]
    // — gather rows. Use take_axis on axis 0 with flattened indices.
    // Flatten topk_idx_sl: [1*1*K] = [K]
    const flat_k_shape = [_]c_int{top_k};
    var flat_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_idx);
    try mlx.check(mlx.mlx_reshape(&flat_idx, topk_idx_sl, &flat_k_shape, 1, s));

    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_take_axis(&rows, m.ordering_2d, flat_idx, 0, s));
    // rows: [K, vpc]

    // selected_canonical: [1, 1, K*vpc]
    const sc_shape = [_]c_int{ 1, 1, n_selected };
    var selected_canonical = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(selected_canonical);
    try mlx.check(mlx.mlx_reshape(&selected_canonical, rows, &sc_shape, 3, s));

    // selected_emb: gather `embed_w` at selected_canonical (flat).
    const flat_n = [_]c_int{n_selected};
    var sc_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_flat);
    try mlx.check(mlx.mlx_reshape(&sc_flat, selected_canonical, &flat_n, 1, s));

    var sel_emb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_emb);
    try mlx.check(mlx.mlx_take_axis(&sel_emb, self.embed_w, sc_flat, 0, s));
    // sel_emb: [K*vpc, hidden]

    // Reshape to [1, 1, K*vpc, hidden] then transpose last two to multiply with h.
    // selected_logits = (h[..., None, :] @ sel_emb.T).squeeze(-2)
    // Equivalent: h_flat [1, hidden] @ sel_emb.T [hidden, K*vpc] → [1, K*vpc].
    const h_shape = mlx.getShape(h);
    const hidden = h_shape[2];

    // h reshape to [1, hidden]
    const h2_shape = [_]c_int{ 1, hidden };
    var h_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h_flat);
    try mlx.check(mlx.mlx_reshape(&h_flat, h, &h2_shape, 2, s));

    // sel_emb.T → [hidden, K*vpc]
    const perm2 = [_]c_int{ 1, 0 };
    var sel_emb_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_emb_t);
    try mlx.check(mlx.mlx_transpose_axes(&sel_emb_t, sel_emb, &perm2, 2, s));

    var sel_logits_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_logits_2d);
    try mlx.check(mlx.mlx_matmul(&sel_logits_2d, h_flat, sel_emb_t, s));
    // [1, K*vpc] → [1, 1, K*vpc]
    const sl3_shape = [_]c_int{ 1, 1, n_selected };
    var sel_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_logits);
    try mlx.check(mlx.mlx_reshape(&sel_logits, sel_logits_2d, &sl3_shape, 3, s));

    // mask_value = min(sel_logits) - 1
    var min_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(min_arr);
    try mlx.check(mlx.mlx_min_axis(&min_arr, sel_logits, -1, true, s));
    const one_v = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one_v);
    var mask_val = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_val);
    try mlx.check(mlx.mlx_subtract(&mask_val, min_arr, one_v, s));
    // mask_val shape: [1, 1, 1] — `mlx_full` needs a scalar broadcast. Use
    // multiply-with-ones to broadcast to [1, 1, vocab].
    const ones_shape = [_]c_int{ 1, 1, @intCast(m.vocab_size) };
    var ones_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones_arr);
    const one_f = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one_f);
    try mlx.check(mlx.mlx_full(&ones_arr, &ones_shape, 3, one_f, .bfloat16, s));

    var mask_val_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_val_bf);
    try mlx.check(mlx.mlx_astype(&mask_val_bf, mask_val, .bfloat16, s));

    var bg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bg);
    try mlx.check(mlx.mlx_multiply(&bg, ones_arr, mask_val_bf, s));

    // sel_logits is bf16 already (drafter weights are bf16), no cast needed.
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_put_along_axis(&out, bg, selected_canonical, sel_logits, -1, s));
    return out;
}

/// One drafter forward pass.
///
/// Inputs:
///   `prev_token_id` — token we're drafting from (initial t1, then last sampled draft).
///   `target_hidden` — `[1, 1, backbone_hidden]` from target (post-final-norm
///       captured via `forwardCaptureHidden`).
///   `target` — pointer to target Transformer (we cross-attend its KV cache).
///   `rope_offset` — Q rotation position. Caller passes `target.cache.step`
///       on the first draft of a round; subsequent drafts in the same round
///       use the SAME offset (per upstream `set_shared_kv`).
///
/// Output: `DrafterStepOut` — caller frees both arrays.
pub fn step(
    self: *DrafterModel,
    target: *Transformer,
    prev_token_id: u32,
    target_hidden: mlx.mlx_array,
    rope_offset: c_int,
) !DrafterStepOut {
    const s = self.s;

    // ── Build input: concat([target.embed(prev_tok)*scale, target_hidden], -1) ──
    const tok_embed = try embedTargetToken(self, target, prev_token_id);
    defer _ = mlx.mlx_array_free(tok_embed);

    // Concat along axis=-1 → [1, 1, 2*backbone]
    var inputs_embeds = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inputs_embeds);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, tok_embed);
        _ = mlx.mlx_vector_array_append_value(vec, target_hidden);
        try mlx.check(mlx.mlx_concatenate_axis(&inputs_embeds, vec, 2, s));
    }

    // ── pre_projection: 2*backbone → drafter hidden ──
    var h = try matmulFn(inputs_embeds, self.pre_proj_w_t, s);

    // ── Forward through drafter layers (cross-attn into target's KV) ──
    for (self.layers) |*lw| {
        const new_h = try layerForward(self, lw, h, target, rope_offset);
        _ = mlx.mlx_array_free(h);
        h = new_h;
    }

    // ── Final norm + post_projection (next-step h_prev) ──
    const h_normed = try rmsNormFn(h, self.norm_w, self.config.rms_norm_eps, s);
    _ = mlx.mlx_array_free(h);
    const h_post = try matmulFn(h_normed, self.post_proj_w_t, s);

    // ── LM head ──
    const logits = if (self.masked != null)
        try maskedLmHead(self, h_normed)
    else if (self.config.tie_word_embeddings) blk: {
        // Tied: logits = h_normed @ embed_w.T. embed_w is [vocab, hidden] so
        // we need its transpose — do it once here (slow path; the masked path
        // is the production case for E4B).
        const perm = [_]c_int{ 1, 0 };
        var ew_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ew_t);
        try mlx.check(mlx.mlx_transpose_axes(&ew_t, self.embed_w, &perm, 2, s));
        break :blk try matmulFn(h_normed, ew_t, s);
    } else return error.DrafterMissingLmHead;

    _ = mlx.mlx_array_free(h_normed);
    return .{ .logits = logits, .h_prev_next = h_post };
}

// ── Tests ──

const testing = std.testing;

test "DrafterConfig parses gemma-4-E4B drafter config.json shape" {
    const json =
        \\{
        \\  "model_type": "gemma4_assistant",
        \\  "backbone_hidden_size": 2560,
        \\  "num_centroids": 2048,
        \\  "centroid_intermediate_top_k": 32,
        \\  "use_ordered_embeddings": true,
        \\  "tie_word_embeddings": true,
        \\  "text_config": {
        \\    "hidden_size": 256,
        \\    "intermediate_size": 2048,
        \\    "num_hidden_layers": 4,
        \\    "num_attention_heads": 4,
        \\    "head_dim": 256,
        \\    "global_head_dim": 512,
        \\    "sliding_window": 512,
        \\    "vocab_size": 262144,
        \\    "rms_norm_eps": 1e-6,
        \\    "layer_types": ["sliding_attention","sliding_attention","sliding_attention","full_attention"],
        \\    "rope_parameters": {
        \\      "sliding_attention": {"rope_theta": 10000.0},
        \\      "full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25}
        \\    }
        \\  }
        \\}
    ;
    var cfg = try parseConfigFromJson(testing.allocator, json);
    defer cfg.deinit(testing.allocator);

    try testing.expectEqual(@as(u32, 2560), cfg.backbone_hidden_size);
    try testing.expectEqual(@as(u32, 256), cfg.hidden_size);
    try testing.expectEqual(@as(u32, 4), cfg.num_hidden_layers);
    try testing.expectEqual(@as(u32, 4), cfg.num_attention_heads);
    try testing.expectEqual(@as(u32, 256), cfg.head_dim);
    try testing.expectEqual(@as(u32, 512), cfg.global_head_dim);
    try testing.expectEqual(@as(u32, 2048), cfg.intermediate_size);
    try testing.expectEqual(@as(u32, 512), cfg.sliding_window);
    try testing.expectEqual(@as(u32, 262144), cfg.vocab_size);
    try testing.expect(cfg.tie_word_embeddings);
    try testing.expect(cfg.use_ordered_embeddings);
    try testing.expectEqual(@as(u32, 2048), cfg.num_centroids);
    try testing.expectEqual(@as(u32, 32), cfg.centroid_top_k);
    try testing.expectEqual(@as(usize, 4), cfg.layer_types.len);
    try testing.expectEqual(LayerType.sliding_attention, cfg.layer_types[0]);
    try testing.expectEqual(LayerType.full_attention, cfg.layer_types[3]);
    try testing.expectEqual(RopeKind.default, cfg.rope_sliding.kind);
    try testing.expectEqual(@as(f32, 10000.0), cfg.rope_sliding.theta);
    try testing.expectEqual(RopeKind.proportional, cfg.rope_full.kind);
    try testing.expectEqual(@as(f32, 1000000.0), cfg.rope_full.theta);
    try testing.expectEqual(@as(f32, 0.25), cfg.rope_full.partial_rotary_factor);
}

test "DrafterConfig rejects non-gemma4_assistant model_type" {
    const json =
        \\{"model_type": "qwen3_5_moe", "backbone_hidden_size": 2560, "text_config": {}}
    ;
    const err = parseConfigFromJson(testing.allocator, json);
    try testing.expectError(error.UnsupportedDrafterArch, err);
}

test "DrafterConfig.layerHeadDim respects per-layer-type" {
    var layer_types = [_]LayerType{ .sliding_attention, .full_attention };
    const cfg = DrafterConfig{
        .backbone_hidden_size = 2560,
        .hidden_size = 256,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .head_dim = 256,
        .global_head_dim = 512,
        .intermediate_size = 2048,
        .sliding_window = 512,
        .layer_types = layer_types[0..],
        .rope_sliding = .{},
        .rope_full = .{ .kind = .proportional, .theta = 1000000.0 },
        .rms_norm_eps = 1e-6,
        .vocab_size = 262144,
        .tie_word_embeddings = true,
        .use_ordered_embeddings = true,
        .num_centroids = 2048,
        .centroid_top_k = 32,
    };
    try testing.expectEqual(@as(u32, 256), cfg.layerHeadDim(0));
    try testing.expectEqual(@as(u32, 512), cfg.layerHeadDim(1));
}

test "LayerType.fromString" {
    try testing.expectEqual(LayerType.sliding_attention, try LayerType.fromString("sliding_attention"));
    try testing.expectEqual(LayerType.full_attention, try LayerType.fromString("full_attention"));
    try testing.expectError(error.UnknownLayerType, LayerType.fromString("bogus"));
}
