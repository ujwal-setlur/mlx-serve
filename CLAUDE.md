# mlx-serve – project context for AI

Native Zig server that runs MLX-format LMs on Apple Silicon and exposes OpenAI-compatible and Anthropic-compatible HTTP APIs. No Python.

## Stack

- **Zig** 0.15+
- **mlx-c** (Apple) via Homebrew; FFI in `src/mlx.zig`.
- **Jinja engine** (lib/jinja_cpp): llama.cpp's C++17 Jinja2 implementation with nlohmann/json. Pre-compiled as `libjinja.a` (rebuild: see comment in `build.zig`).
- **stb_image** (lib/stb_image.h): JPEG/PNG decoding for vision pipeline
- **libwebp** via Homebrew: WebP image decoding for vision pipeline
- **safetensors** for weights; BPE tokenizers (SentencePiece / byte-level)

## Layout

| Path | Role |
|------|------|
| `src/main.zig` | Entry, CLI (`--model`, `--serve`, `--host`, `--port`, `--prompt`, `--max-tokens`, `--temp`, `--ctx-size`, `--timeout`, `--reasoning-budget`, `--no-vision`, `--mtp`, `--pld`, `--pld-draft-len`, `--pld-key-len`, `--drafter`, `--draft-block-size`, `--log-level`, `--version`, `--help`) |
| `src/mlx.zig` | mlx-c FFI |
| `src/model.zig` | Config + safetensors loading; see **Supported Architectures** below |
| `src/tokenizer.zig` | BPE tokenizer |
| `src/transformer.zig` | Forward pass (embedding, attention, MLP, MoE, GatedDeltaNet); architecture dispatch |
| `src/generate.zig` | Autoregressive generation, sampling (temperature, top-k, top-p, repeat penalty, presence penalty, logprobs) |
| `src/chat.zig` | Chat template formatting (ChatML, Gemma turns, Llama-3, Jinja2 via llama.cpp engine); thinking/reasoning tags; tool call parsing |
| `src/vision.zig` | Vision encoder (Gemma 4 SigLIP): patch embedding, 2D RoPE, clipped linears, position pooling, embedding projection |
| `src/server.zig` | HTTP server: `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses`, `/v1/responses/compact`, plus a WebSocket transport on `/v1/responses` (OpenAI Chat + Responses + Anthropic Messages, stream + non-stream, tool calling, KV cache, vision) |
| `src/responses.zig` | OpenAI Responses API: input-item parser (incl. `compaction` items), tool-shape translation, output-item builders, in-memory `ResponseStore`, `encodeCompactionBlob` (HTTP/streaming live in `server.zig`) |
| `src/ws.zig` | RFC 6455 WebSocket framing + handshake (server-side only). Generic over a `Conn`-shaped type so it stays test-friendly without depending on `server.zig`. |
| `src/pld_index.zig` | Prompt Lookup Decoding (PLD) n-gram index. Pure-data `PldLookup.findMatch` — given a key (last `key_len` tokens) and committed stream, returns up to `max_draft` tokens from the latest prior occurrence. Tests at the bottom of the file. |
| `src/drafter.zig` | Gemma 4 assistant drafter (cross-attention spec-decode). `DrafterModel.step` runs one drafter forward; `bind(target)` resolves the layer-type → target K/V mapping; `MaskedEmbedding` is the centroid-routed sparse LM head. Loaded with `--drafter <dir>`. |
| `src/status.zig` | TUI status bar (CPU, memory, GPU metrics) |
| `src/log.zig` | Leveled logging (error, warn, info, debug) |
| `build.zig` | Zig build; links mlx-c, libjinja.a, libwebp, stb_image |

### MLX Core (Swift macOS app)

| Path | Role |
|------|------|
| `app/Package.swift` | Swift package; `MLXCore` executable + `MLXCoreTests` test target |
| `app/Sources/MLXServe/MLXServeApp.swift` | App entry, menu bar + Chat/Browser windows |
| `app/Sources/MLXServe/AppState.swift` | Global state, chat session management, persistence |
| `app/Sources/MLXServe/Models/ChatModels.swift` | `ChatMessage`, `ChatImage`, `SerializedToolCall`, `ChatSession` |
| `app/Sources/MLXServe/Models/AgentModels.swift` | `AgentToolKind`, `AgentPlan`, `StepResult` |
| `app/Sources/MLXServe/Services/APIClient.swift` | HTTP + SSE streaming client for mlx-serve |
| `app/Sources/MLXServe/Services/AgentPrompt.swift` | System prompt, tool definitions (10 tools), `SkillManager` (prompt-based skills from `~/.mlx-serve/skills/`) |
| `app/Sources/MLXServe/Services/AgentEngine.swift` | Shared agent logic: history building, tool execution, repetition tracking, token estimation, overflow management |
| `app/Sources/MLXServe/Services/ToolExecutor.swift` | Tool handlers: shell, cwd, readFile, writeFile, editFile, searchFiles, listFiles, browse, webSearch, saveMemory |
| `app/Sources/MLXServe/Services/ImagePreprocessor.swift` | Image preprocessing for vision encoder (resize, float32 CHW conversion) |
| `app/Sources/MLXServe/Services/BrowserManager.swift` | WKWebView (headless, created eagerly for background browsing) |
| `app/Sources/MLXServe/Services/ServerManager.swift` | mlx-serve process lifecycle, stderr capture (`serverLog`), auto-start |
| `app/Sources/MLXServe/Services/TestServer.swift` | Embedded HTTP server (port 8090) for test automation — uses AgentEngine for shared logic |
| `app/Sources/MLXServe/Services/AgentMemory.swift` | Agent context memory (recent dirs, commands) |
| `app/Sources/MLXServe/Views/ChatView.swift` | Chat UI + `runAgentLoop()` + image attachment + context monitor |
| `app/Sources/MLXServe/Views/StatusMenuView.swift` | Menu bar UI, server log viewer, Claude Code launcher |
| `app/Sources/MLXServe/Views/BrowserView.swift` | Browser window (uses shared WKWebView) |

## Testing

- Always add tests, for anything you do, and update them as needed
- Unit tests are fine, but also add integration tests with real models, these are the real tests
- Make sure tests account for all the suported model architecture types, not just one.
- After a big feature, always test by building mlx-serve and mlx core.app, then run the .app bundle with TestServer.swift enable and test agentic harness
- `zig build test` — unit tests (chat, server, generate, model, log, tokenizer)
- `cd app && swift test` — Swift unit tests (agent harness, SSE parsing, serialization, history)
- `./tests/integration_test.sh [model_dir] [port]` — 36 end-to-end API tests (needs a model)
- `./tests/test_tool_response.sh [port]` — tool calling round-trip tests (needs running server)
- `./tests/test_kv_cache_poison.sh [port]` — KV cache poisoning regression test (needs running server)
- `./tests/test_anthropic_api.sh [port]` — Anthropic Messages API integration tests (needs running server)
- `MTP_TEST_MODEL=<dir> ./tests/test_mtp_equivalence.sh [port]` — MTP byte-equivalence test (auto-skips when env var unset OR safetensors lack `*.mtp.*` weights — common for MLX-converted Qwen3.5)
- `PLD_TEST_MODEL=<dir> ./tests/test_pld_equivalence.sh [port]` — PLD byte-equivalence test (defaults to `~/.mlx-serve/models/Qwen3.5-4B-MLX-4bit`; PLD is model-agnostic, any MLX checkpoint works). Verified on Qwen3.5-4B, Gemma-4-E4B, LFM2.5-350M (hybrid SSM).
- `./tests/test_streaming_pld.sh [port]` — verifies streaming PLD output is byte-identical to non-streaming PLD AND to regular streaming (B's contribution)
- `./tests/test_streaming_mtp.sh [port]` — same for MTP, against Qwen3.5-4B-MTPLX-Speed (auto-skips if MTP weights absent)
- `./tests/test_drafter_equivalence.sh [port]` — Gemma 4 drafter byte-equivalence test (paired `--drafter`/no-drafter on `gemma-4-e4b-it-4bit` + `gemma-4-E4B-it-assistant-bf16`)
- `./tests/bench_spec.sh [runs]` — focused spec-decode benchmark (none vs PLD vs drafter × heavy-echo + creative × Qwen/Gemma/LFM). Run 1 is warmup. Drives default-on flip decisions.
- Always run `zig build test` and `swift test` before submitting changes
- Add tests for new pure logic functions in the same source file (Zig convention)
- Shell integration tests go in `tests/` and need a running server with a loaded model

## Building

- **Full app bundle**: `cd app && SKIP_NOTARIZE=1 bash build.sh` — builds Zig + Swift, assembles `.app`, signs (requires `APPLE_DEVELOPER_ID` and `APPLE_TEAM_ID` env vars). Bundles libwebp + libsharpyuv for vision support.
- Zig server only: `zig build -Doptimize=ReleaseFast` (requires `brew install webp` for vision pipeline)
- Swift app only: `cd app && swift build -c release`
- For tests: `zig build test` (Zig) and `cd app && swift test` (Swift)
- **Rebuild Jinja library** (after changing `lib/jinja_cpp/*.cpp`): `cd lib/jinja_cpp && for f in jinja_wrapper caps lexer parser runtime jinja_string value; do clang++ -std=c++17 -O2 -DNDEBUG -I . -c $f.cpp -o obj/$f.o; done && ar rcs libjinja.a obj/*.o`

## Versioning & Releases

**Scheme**: CalVer `YY.M.N` — e.g., `v26.4.25` means 2026, April, 25th release that month.
- `YY.M` comes from the build date
- `N` is auto-incremented from the last GitHub release for that `YY.M` prefix
- `build.sh` computes the version automatically via `gh release list`

**Version sources** (all set by `build.sh`):
- `app/Info.plist` → `CFBundleVersion` + `CFBundleShortVersionString`
- Zig binary → passed via `-Dversion` build option (consumed as `build_options.version` in `main.zig`)
- Git tag → created manually with `gh release create v{version}`

**Release process**:
1. Update `CHANGELOG.md` with a new entry — use the NEXT version, not the current latest release. Run `gh release list --limit 1` to check what's already released.
2. Commit and push changes
3. Run `cd app && SKIP_NOTARIZE=1 bash build.sh` — this computes the version, builds everything, and prints the `gh release create` command at the end
4. Run the printed `gh release create` command

**Important**: Never write a CHANGELOG entry using a version that already exists as a GitHub release. Always check `gh release list` first.

## Benchmarking

Run `./bench.sh` after every major feature or optimization change. Results go in `BenchmarkLog.md`.
- `./bench.sh` — full suite: mlx-serve + mlx-lm reference, all models
- `./bench.sh --model gemma` — single model
- `./bench.sh --no-mlx-lm` — skip Python reference
- `./bench.sh --runs 5` — more runs for tighter averages

## Conventions

- Prefer minimal, DRY Zig; avoid unnecessary abstraction.
- Chat templates live in model dirs; llama.cpp's Jinja engine renders them (with fallback formatting).
- Server supports concurrent health checks via threaded connections, single-slot generation.
- KV cache reuse across requests via prompt prefix matching; invalidated after tool-calling requests and pad-only generations.
- Tests go at the bottom of each source file (Zig convention).
- Jinja static library must be rebuilt with system clang++ after changing `lib/jinja_cpp/*.cpp` (see build command in `build.zig`).

## Supported Architectures

Model support is determined by `model_type` in the model's `config.json`. The server dispatches to architecture-specific code paths in `model.zig` (config parsing, weight prefix) and `transformer.zig` (forward pass).

### Working

| `model_type` | Family | Weight prefix | Vision | MoE | Notes |
|---|---|---|---|---|---|
| `gemma4`, `gemma4_text` | Gemma 4 | `language_model.model` | SigLIP | -- | Full support incl. vision, clipped linears, PLE |
| `gemma3` | Gemma 3 | `language_model.model` | -- | -- | |
| `qwen3` | Qwen 3 | `model` | -- | -- | QK norm enabled |
| `qwen3_5`, `qwen3_5_moe`, `qwen3_5_moe_text` | Qwen 3.5 / 3.6 | `language_model.model` | -- | Optional | GatedDeltaNet + MoE/dense MLP, shared expert routing |
| `qwen3_next` | Qwen 3-next | `model` | -- | Optional | DeltaNet (GatedDeltaNet layers) |
| `nemotron_h` | NVIDIA Nemotron-H | `backbone` | -- | -- | Hybrid transformer + Mamba2 SSM, per-timestep recurrence |
| `lfm2` | Liquid LFM2.5 | `model` | -- | -- | Hybrid gated conv + full attention, state-space recurrence |
| `llama` | Llama | `model` | -- | -- | |
| `mistral` | Mistral | `model` | -- | -- | |

### Not Yet Supported (TODO)

| `model_type` | Family | Blocked by | Effort |
|---|---|---|---|
| `lfm2-vl` | Liquid LFM2.5-VL | Needs vision encoder integration | Medium |
| `phi`, `phi3` | Microsoft Phi | Different attention/MLP layout, different weight names | Medium |
| `command-r` | Cohere Command R | Different architecture | Medium |

Models with `vision_config` in config.json but no vision weights (e.g., text-only quantized Qwen 3.5) are handled gracefully — the vision encoder init detects missing weights early and disables vision. The Swift app flags unsupported architectures in the Model Browser via `supportedModelTypes` in `HFModels.swift`.

## OpenAI Responses API

The server exposes `POST /v1/responses` (plus `GET`/`DELETE /v1/responses/{id}`) — OpenAI's stateful Responses API. Pure data handling (input parsing, output-item builders, in-memory store) lives in `src/responses.zig`; HTTP and generation orchestration in `src/server.zig`.

### Envelope shape (`buildResponsesEnvelope` + `ResponseEcho`)
The response body must echo most request configuration to satisfy OpenAI's strict ResponseResource schema. Every response includes: `tools`, `tool_choice`, `text`, `reasoning`, `usage` (with `input_tokens_details.cached_tokens` + `output_tokens_details.reasoning_tokens`), `truncation`, `parallel_tool_calls`, `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `top_logprobs`, `max_output_tokens`, `max_tool_calls`, `background`, `service_tier`, `metadata`, `safety_identifier`, `prompt_cache_key`, `instructions`, `error`, `completed_at`. Renderers `renderResponsesToolsEcho`/`renderResponsesToolChoiceEcho`/`renderResponsesTextEcho`/`renderResponsesReasoningEcho`/`renderResponsesMetadataEcho` reshape the request JSON into the exact schema-conformant form (e.g., flat `{type, name, description, parameters, strict}` for tools — not the nested chat-completions form).

### Streaming SSE
Events are: `response.created`, `response.in_progress`, `response.output_item.added` (per item), per-type deltas (`response.reasoning_summary_text.delta`, `response.output_text.delta`, `response.function_call_arguments.delta`), per-type `.done`, `response.output_item.done`, `response.completed`. **Every event must carry a `sequence_number` field** (incrementing integer). `sendResponsesEvent` injects it before send; the POST handler keeps a per-request `seq_num` counter that's threaded through every emit helper (`emitResponses*`).

### Stateful chains
`ResponseStore` (capacity `RESPONSE_STORE_CAP`) keeps prior responses keyed by id. When a request supplies `previous_response_id`, history is replayed; if the id is missing → 404. `parseInput` accepts both string and content-block array shapes, and `inputContainsFunctionCallOutput` triggers final-answer mode (tools disabled) when the user is supplying tool outputs for a structured-output turn.

### Compatibility quirks
- The compliance suite at `experiments/openresponses` (run via `bun run test:compliance --base-url http://host:port/v1 --api-key X --model mlx-serve`) validates against the strict ResponseResource schema and the per-event streaming union — currently passes 17/17.
- `top_level response_format` is accepted as an alias for `text.format` (some clients reuse their chat-completions adapter).

### Compaction (`POST /v1/responses/compact`)

Pure data transformation — no LLM call, no inference slot. The server reuses `responses_mod.parseInput` to materialize the resolved message history (including any `previous_response_id` lookup) and synthesizes an opaque `encrypted_content` blob: base64 over `{"v":1,"msgs":[{"role":..., "content":...}, ...]}`. Feeding the returned `compaction` item back into `response.create` as an `input` element (handled by `appendCompactionInputItem` in `responses.zig`) reconstitutes the messages — exercising the round-trip without an LLM call. `model` is required (422 on missing). Tool calls and images are dropped when encoding (the blob is text-only).

### WebSocket transport (`ws[s]://host/v1/responses`)

Same endpoint, opt-in via the standard `Upgrade: websocket` handshake. Each text frame is a `response.create`-shaped JSON message; the server bridges the per-frame turn through `handleResponses` and emits each SSE event as a single WS text frame.

- **No `[DONE]` on success.** `response.completed` (or `.failed`/`.incomplete`) is the per-response terminator, and the compliance suite advances turns the moment it sees one. A trailing `[DONE]` would land in the next turn's bucket and break chained sessions. `[DONE]` is reserved for error fallbacks where no terminal event is sent.
- **Sequence numbers reset per response**, not per connection. `seq_num` lives inside `handleResponses`, fresh each call.
- **Per-connection store-false cache.** `WsLocalCache` holds responses requested with `store: false` for the lifetime of the WS connection. After each turn, if the user requested `store: false`, the freshly-stored response is moved from the global `ResponseStore` into the connection-local cache; on connection close, all entries are freed. Cross-connection lookups of those ids correctly return `previous_response_not_found`.
- **Cache eviction on failed continuation.** A failed continuation (status != "completed", or invalid `function_call_output`) evicts the chain root from the local cache.
- **Bridge mechanism.** A `WsBridge` value (function pointer + opaque impl) is attached to `Conn.ws_mode` for the duration of a turn. `sendResponse` and `sendAnthropicEvent` branch on `ws_mode` so SSE bytes never hit the wire when bridging — instead the JSON payload becomes a single WS text frame. The SSE-headers write at the top of `handleResponses` is similarly guarded.

## Anthropic Messages API

The server exposes `POST /v1/messages` for Anthropic API compatibility, enabling Claude Code and other Anthropic SDK clients to use local models.

### Request/Response mapping
- **System prompt**: Anthropic puts `system` at top level → converted to internal system message
- **Content blocks**: Anthropic messages use typed content blocks (`text`, `tool_use`, `tool_result`, `thinking`) → converted to internal `Message` structs
- **Tools**: Anthropic `input_schema` → converted to OpenAI `parameters` format for chat template compatibility
- **Tool results**: Anthropic `tool_result` in user messages → internal `role: "tool"` messages
- **Thinking**: `thinking` config parsed → maps to `enable_thinking` + `reasoning_budget`; thinking blocks emitted with fake `signature` field
- **Stop reasons**: `stop` → `end_turn`, `length` → `max_tokens`, `tool_calls` → `tool_use`

### Streaming format
Anthropic SSE uses named events: `message_start`, `content_block_start`, `content_block_delta` (with `text_delta`, `thinking_delta`, `signature_delta`, `input_json_delta`), `content_block_stop`, `message_delta`, `message_stop`. Each content block has an explicit start/stop lifecycle with an index.

### Claude Code integration
The MLX Core app has a "Launch Claude Code" button (visible when server is running) that opens Terminal with the `claude` CLI configured to use the local server:
- `ANTHROPIC_BASE_URL` → local server URL
- `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` → dummy values (local server, no auth)
- `ANTHROPIC_DEFAULT_*_MODEL` → `mlx-serve` (routes all model tiers through local)
- `CLAUDE_CODE_SUBAGENT_MODEL` → `mlx-serve`

## Tool Calling Architecture

### Server side (Zig)
- **Tool call detection**: When `tools` param is present, server buffers tokens and checks for tool call patterns. If thinking is enabled, thinking tokens are buffered separately and not flushed as content. After generation, `chat.parseToolCalls()` checks for patterns (`<tool_call>`, Hermes XML, Gemma 4 `<|tool_call>`, raw JSON). Gemma 4 double-brace args (`{{"key":"value"}}`) are unwrapped before JSON parsing.
- **Message serialization** (`chat.serializeMessagesJson`): Converts `Message` structs to JSON for Jinja templates. `role: "tool"` messages are passed natively (no transformation) — Gemma 4 templates handle them directly as `<|turn>tool`. Tool call `arguments` are serialized as JSON strings (not objects) so templates render them correctly.
- **Streaming SSE**: Tool call arguments are sent in a single delta (name + id + full args) to prevent client-side accumulation bugs. Thinking content (`<|channel>thought`) is detected during streaming and buffered until the closing tag, then emitted as `reasoning_content`.
- **Fallback formatter** (`chat.fallbackFormatChat`): Used when Jinja fails. Handles ChatML (`<tool_call>/<tool_response>`), Llama (`ipython` role), Gemma (`Tool result:` in user turn).
- **KV cache**: `reuseKVCache()` compares token-by-token prefix. Cache is automatically invalidated after tool-calling requests (generated tool-call tokens corrupt the cache for the next request) and after pad-only generations. Sliding window layers keep full buffers (no trimming) — views return the last `sw` entries during decode, all entries during prefill.

### Client side (Swift)
- **Agent loop** (`ChatView.runAgentLoop`): Up to 150 iterations. Calls model with tools → parses tool calls → executes locally → feeds results back → repeats until model responds without tool calls. Adds synthetic user nudge after tool results for models that need it.
- **History builder** (`ChatView.buildAgentHistory`): Converts `ChatMessage` array to OpenAI API format. Filters out error messages, pad-only content, and agent summaries. Truncates assistant messages at 500 chars. Budget-aware: walks backward from newest message, fitting history within available context tokens (context_length - max_tokens - system_prompt). Pins first user message + first assistant response. Auto-compacts tool results when context is tight.
- **SSE parsing** (`APIClient.performStream`): Accumulates streamed tool call deltas. Server sends full arguments in one delta. Emits `.toolCalls` event on `finish_reason: "tool_calls"`. Fallback emission if stream drops without finish_reason.
- **Tool call storage**: `SerializedToolCall` (id, name, arguments as JSON string) stored on `ChatMessage.toolCalls`. Persisted via Codable for history replay. Backwards-compatible with old history files (field is optional).
- **Error recovery**: Tool execution errors include what args were sent and ask the model to retry, enabling self-correction in the agent loop.

## Prompt-based Skills

Users can teach the agent new capabilities by dropping `.md` files in `~/.mlx-serve/skills/`. Each file has YAML frontmatter:

```markdown
---
name: deploy
description: Deploy the project to production
trigger: deploy, release, push to prod, ship it
---
When asked to deploy:
1. Run `git push origin main`
2. Check CI with `gh run list --limit 1`
```

- `trigger` — comma-separated keywords; if the user's message contains ANY keyword (case-insensitive substring), the skill body is injected into the system prompt
- A short skill index (name + description) is always included so the model knows what's available
- `SkillManager` in `AgentPrompt.swift` scans the directory on each agent loop iteration (re-scans when dir modification date changes)
- Skills are injected in `ChatView.runAgentLoop()` between the base system prompt and agent memory context
- UI: folder icon button in menu bar and chat toolbar opens `~/.mlx-serve/` in Finder

## Resumable Downloads

Large model downloads (e.g., 26B at ~15 GB) use streaming writes to `.partial` files:

- `DownloadManager` uses `URLSessionDataTask` with `StreamingDelegate` — bytes are written to `<file>.partial` as they arrive
- If a download is interrupted (network drop, app crash), the `.partial` file survives on disk
- On retry/resume, sends `Range: bytes=<existingSize>-` header; if server returns 206, only the remainder is downloaded; if 200, truncates and restarts
- Automatic retry: 3 attempts per file with 2s/4s backoff; status text shows "Connection lost, retrying..."
- Cancellation preserves `.partial` files for future resume
- UI shows "Resume" instead of "Download"/"Retry" when `.partial` files exist (`hasPartialDownload()`)
- Already-completed files are skipped (size check against HuggingFace metadata)

## Debugging

### Server logs
- Start server with `--log-level debug` for verbose output (Jinja errors, cache hits, token counts)
- The MLX Core app starts the server as a subprocess; stderr is captured in `ServerManager.serverLog` (64KB rolling buffer). View it via the log button (text-align icon) next to Start/Stop in the menu bar.
- To see logs from a manually-started server: `./zig-out/bin/mlx-serve --model <path> --serve --port 8080 --log-level debug 2>&1`
- Key log patterns:
  - `jinja error: ..., using fallback` — Jinja template failed, check template compatibility
  - `[cache] reusing N/M tokens` — KV cache hit; if N is close to M, most of prompt is cached
  - `[cache] invalidated` — cache was reset (tools config changed, etc.)
  - `<- N+M tokens (Xms) [reason]` — N prompt tokens, M completion tokens, finish reason
  - `tool_msgs=N` — count of `role: "tool"` messages in the request

### Swift app logs
- `print()` in the Swift app goes to stdout, not visible when launched via `open`. To see it: run the binary directly from terminal, or write to a file.
- The app dumps every agent loop request to `~/.mlx-serve/last-agent-request.json` (debug aid). Replay with: `curl -sf http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d @~/.mlx-serve/last-agent-request.json`
- Chat history is persisted at `~/.mlx-serve/chat-history.json`

### Reproducing issues
- To test tool calling without the app: use `curl` with `stream: false` first (simpler to inspect), then `stream: true` (matches app behavior).
- To test the Jinja template offline: `pip3 install jinja2`, then render with Python using the model's `chat_template.jinja` file and the dumped request JSON.
- To test KV cache effects: restart the server fresh between tests (`pkill -f mlx-serve`). A single bad request can poison the cache for all subsequent requests.

## Gotchas

### KV cache after tool calls
After a tool-calling request, the KV cache is automatically invalidated. The generated tool-call tokens are in the cache but not in `cached_prompt_ids`, so reusing the cache for the next request (which includes tool results) would corrupt attention. Similarly, pad-only generations trigger cache invalidation.

### Sliding window KV cache
Models with sliding window attention (e.g., Gemma 4 E4B with 512-token window) keep the full KV buffer — no trimming. During prefill, all entries are returned so Q and K dimensions match. During decode, views return only the last `sw` entries. The sliding window mask handles attention scope. This matches mlx-lm's `RotatingKVCache` behavior.

### Gemma 4 tool calling format
Gemma 4 templates handle `role: "tool"` natively (producing `<|turn>tool`). No transformation is needed — the server passes tool messages through as-is. The `tool_responses` field is NOT added (it causes duplicate content in rendered prompts). Tool call arguments are serialized as JSON strings so the template renders them verbatim.

### Streaming with tools and thinking
When `tools` are present, the server buffers tokens to detect tool call patterns. If thinking is also enabled, `<|channel>thought` tokens are detected and kept buffered (not flushed as content) until the closing `<channel|>` tag. After generation, thinking content is split from visible content and emitted as `reasoning_content`. Channel tags (`<|channel>`, `<channel|>`) are stripped from visible content.

### SSM/GatedDeltaNet state initialization
`conv1dWithCache` sets `ssm.initialized = true` after updating the conv state, but BEFORE the SSM recurrence state is created. Code that initializes SSM state must check `ssm.ssm_state.ctx == null` (not `!ssm.initialized`). Both `mamba2Mixer` and `gatedDeltaNet` use this pattern.

### Parameter-free RMS norm (mlx-c)
mlx-c requires a non-empty weight array for `mlx_fast_rms_norm`. Passing a null/empty array (`.{ .ctx = null }`) crashes. For parameter-free normalization, pass `ones([dim], bfloat16)` as the weight. This affects GatedDeltaNet Q/K normalization and Mamba2 group norm.

### Nemotron-H time_step_limit
Python's `ModelArgs.__post_init__` defaults `time_step_limit` to `(0.0, inf)` — effectively no dt clipping. The config.json fields `time_step_min`/`time_step_max` exist but are NOT used for SSM clipping by Python. Our defaults match Python: `(0.0, inf)`. Only the `time_step_limit` JSON array (if present) overrides these.

### MTP (Multi-Token Prediction) speculative decoding
Qwen3.5/3.6/Qwen3-Next ship a native MTP head — a single transformer block trained to predict token N+2 from `(hidden_state_N, embed(token_N+1))`. Enabled per-server via `--mtp` and per-request via the `enable_mtp` JSON field. Only active when `config.has_mtp` is true (parsed from `mtp_num_hidden_layers` or `num_nextn_predict_layers`).

**Streaming support**: MTP works under `stream: true` via `StreamingTokenStream` (server.zig:2455) — the adapter wraps `Generator` and yields one token per `next()` call regardless of whether the underlying step is regular (`gen.next`), MTP (`gen.nextMtp`, 1–2 tokens/step) or PLD (`gen.nextPld`, up to `1+draft_len` tokens/step). Each per-token state machine in `handleStreamingGeneration` / `handleAnthropicStreaming` stays untouched; only the loop's `gen.next(allocator)` call site changed to `ts.next(allocator)`. Dispatch via `pickStreamMode` (server.zig:2561), priority `mtp > pld > regular`. EOS-in-batch behavior matches the non-streaming `generateMtp`/`generatePld`: EOS tokens are not yielded, the adapter just terminates. Auto-disabled when `tools` are present (tool-call buffering is incompatible with multi-token verify) and when `logprobs` is requested.

Algorithm (in `Generator.nextMtp`): draft via `mtpForward(last_hidden, last_token)` → sample → snapshot KV+SSM → verify via main `forwardCaptureHidden([last_token, draft_id])` length-2 → greedy compare argmax at position 1 → accept (advance step+=2, save new hidden) or reject (restore caches, re-forward `[last_token]` alone, sample fallback from position 0). Snapshot/restore via `KVCache.snapshot/restore` and `ssmSnapshot/Restore` ensures hybrid (GatedDeltaNet) models roll back correctly.

**Critical**: most MLX-converted Qwen3.5/3.6 checkpoints declare `mtp_num_hidden_layers: 1` in their config but **strip the MTP weights** during conversion (no `*.mtp.*` tensors in safetensors). Without those weights, `--mtp` requests will fail at first `mtpForward` with `MISSING WEIGHT: ...mtp.0.eh_proj.weight`. Verify with `safe_open` before benchmarking. The `tests/test_mtp_equivalence.sh` script auto-detects this and skips.

### PLD (Prompt Lookup Decoding) speculative decoding
Model-agnostic speculative decoding via n-gram match in `prompt + generated_tokens`. No model weights required — works on every supported architecture. Enabled per-server via `--pld` (with `--pld-draft-len <n>` default 5 and `--pld-key-len <n>` default 3) and per-request via the `enable_pld` JSON field. The pure n-gram lookup lives in `src/pld_index.zig` (`PldLookup.findMatch`); the draft+verify orchestration is in `Generator.nextPld` / `generatePld` in `src/generate.zig`.

**Lazy decode pipeline (post-A rewrite)**: PLD's cold path now consumes the same pre-launched `pending_logits` that `Generator.next` does — when no n-gram match is found (or the first-position acceptance test fails), `nextPld` emits the lazy `pending_sample` and schedules the next `forward(pending_sample)` async, identical to `Generator.next`. No verify forward, no snapshot. This is the change that eliminated the previous ~10% novel-output regression. PLD's hot path (first-position match) snapshots KV+SSM, forwards `[draft[0..m-1]]` length `m` (not `1+m`), and walks `verify_logits[i]` vs `draft[i+1]` — same logic as before, just consuming the lazy pipeline.

**Auto-disable rules** (mirror MTP): off when `tools` are present, off when `logprobs > 0`, off when grammar-constrained sampling is active. **Streaming is now supported** via `StreamingTokenStream` (same adapter as MTP — see MTP section). PLD works on hybrid SSM architectures (LFM2.5, Nemotron-H) — see "PLD on hybrid SSM models" below for the snapshot/restore null-state caveat. **Drafter+MTP+PLD priority**: drafter > MTP > PLD > regular. When multiple are enabled, the highest-priority active mode wins; the others' enable flags are silently zeroed before dispatch to prevent log spam.

**Default-on caveat**: PLD is **not** flipped on by default. Bench shows mixed results — strong wins on Gemma-4-E4B heavy-echo (1.50×) but regressions on Qwen3.5-4B heavy-echo (0.97×) and LFM2.5-350M heavy-echo (0.91×). On creative/novel content all models are at parity (≥0.99×). Until the per-step overhead is reduced (e.g., fusing the verify forward and post-step `forward(correction)` into a single length-(m+1) forward), PLD stays opt-in via `--pld` per-request `enable_pld:true`.

**`prompt_ids_owned`**: `Generator.initWithOptions` clones the input `prompt_ids` into `prompt_ids_owned` (freed in `deinit`) so PLD's lookup table sees the full context. The caller-supplied slice is freed before `nextPld` runs, so we cannot reference it. The owned copy is also visible to non-PLD generators but unused there.

**Partial-accept re-forward**: when verify accepts `j < m` drafts, the cache is over-advanced by `m - j`. `nextPld` rolls back via `KVCache.snapshot/restore` + `ssmSnapshot/Restore`, then re-forwards `[t1, draft[0..j]]` (length `1+j`) to land the cache at `+1+j`. The pending correction is sampled from the *original* verify_logits[j] (not the re-forward) — that's the model's choice for the position the partial draft missed. This costs ~1 extra forward per partial-reject step, balanced against `1+j` accepted tokens for free.

**Stochastic verify** treats the draft as a one-hot distribution (since it came from n-gram lookup, not a probabilistic model): `accept_prob = min(1, target_p[draft[i]])`. On reject, sample from residual `max(target_p − one_hot(draft[i]), 0)` renormalized — equivalent to "sample from target distribution conditional on not draft[i]" which preserves the marginal distribution per Leviathan et al. The one-hot is built via `pldOneHotRow` (arange + equal + cast) — no scatter required.

**Equivalence test**: `./tests/test_pld_equivalence.sh [port]` (defaults to `~/.mlx-serve/models/Qwen3.5-4B-MLX-4bit`; override with `PLD_TEST_MODEL=<dir>`). Greedy temp=0 output must be byte-identical with vs without `--pld`. Skips cleanly when no model is available. Verified on Qwen3.5-4B, Gemma-4-E4B, and LFM2.5-350M.

### Gemma 4 assistant drafter speculative decoding
Google ships small 4-layer drafter checkpoints alongside Gemma 4 (`gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant-bf16`). Hidden size 256, 4 attention heads, **no K/V projections** — the drafter cross-attends into the **target's** K/V cache via a layer-type mapping (drafter sliding layer reads target's last sliding layer's K/V; drafter full layer reads target's last full layer's K/V). Loaded explicitly via `--drafter <dir>` (and `--draft-block-size <n>` default 4); per-request `enable_drafter` JSON field defaults true when a drafter is loaded.

Drafter input per step: `concat([target.embed(prev_tok) * sqrt(target.hidden), h_prev], -1)` projected from `[1,1,2*backbone_hidden]` → drafter hidden 256. The drafter is autoregressive within the round (`block_size − 1` forwards = 3 drafts per round at default), each step's hidden feeding the next; position is constant across all drafts in a round (RoPE offset = `target.cache.step + 1`). Verify mirrors MTP/PLD: target forwards `[t1, draft[0..K-1]]` length `1+K`, argmax compare per position, partial-accept rollback via `KVCache.snapshot/restore`. Sparse `MaskedEmbedding` LM head: ~2048 centroids, top-32 → ~4096 token logits materialized of 262144 total.

Validation at load: pair-check rejects mismatched drafter+target (`error.UnsupportedDrafterArch` if `model_type != "gemma4_assistant"`, `error.DrafterTargetMismatch` if `backbone_hidden_size != target.hidden_size` or any drafter `layer_type` is absent from target's `layer_types[:N - num_kv_shared_layers]`). All drafter linear weights are pre-transposed at load (matches MTPLX's `eh_proj_w_t` pattern) so `step()` uses plain `mlx_matmul` — no per-step transpose cost.

**Critical fix**: `forwardCaptureHidden` previously left the captured array empty for any model going through `forwardStandard` (= all non-MoE Gemma 3/4 + Llama + Mistral + Qwen 3 etc.). Workstream C added an explicit `mtp_capture_hidden` slice-and-set block at `transformer.zig:2841-2856` that fixes both the drafter's first-step h_prev and MTP on non-MoE checkpoints.

**Streaming**: drafter dispatch is **non-streaming-only** in v1 (TODO: extend `StreamMode` enum + `StreamingTokenStream` adapter to include `.drafter`). Streaming requests with `enable_drafter:true` log `drafter=disabled (streaming; non-stream supports it)` and fall through to regular streaming. Auto-disabled with `tools`, `logprobs > 0`, grammar-constrained sampling, and on hybrid SSM architectures (the multi-token verify forward isn't yet wired through the SSM/conv recurrence path; same restriction MTP has on hybrid Qwen3-Next).

**Equivalence test**: `tests/test_drafter_equivalence.sh` — byte-identical greedy temp=0 output with `--drafter` vs without on `~/.mlx-serve/models/gemma-4-e4b-it-4bit` paired with `~/.mlx-serve/models/gemma-4-E4B-it-assistant-bf16`. Verified at PASS with 36/12 attempts (3.0/3 max acceptance rate) on echo-heavy prompt, 67 tok/s decode (vs 33.5 baseline = 2.0×).

**Default-on caveat**: drafter is **not** flipped on by default. Bench shows extreme bimodal behavior — Gemma-4-E4B heavy-echo gets 1.98× speedup, but creative/novel content runs at **0.555× (45% slowdown)** because draft acceptance collapses on out-of-distribution content and verify-then-fallback overhead dominates. The agent harness mixes echo + novel content per-turn, so a default-on flip would slow down half of all turns. Stays opt-in via `--drafter` until a runtime heuristic (e.g., n-gram match score on the prefix) gates per-request enablement.

### PLD on hybrid SSM models (snapshot null-state guard)
PLD requires snapshotting the per-layer SSM cache before the multi-token verify forward, so partial-accept can roll back. On hybrid models the `SSMCacheEntry` has two independent slots (`conv_state`, `ssm_state`) populated by *different* layer types: LFM2's `gated_conv` writes only `conv_state` (sets `initialized=true` for cache reuse) and never touches `ssm_state`. Calling `mlx_array_set` with a null source aborts the process via mlx-c's default error handler (`printf("MLX error: expected a non-empty mlx_array") + exit(-1)`), so the snapshot/restore code in `transformer.zig` (`ssmSnapshot`, `ssmRestore`, plus the parallel `PrefillCache` save/restore paths) checks each field's `.ctx != null` independently — the `initialized` flag alone is not sufficient. This was the previous "off on hybrid SSM" auto-disable; lifted once the per-field guard landed.

### mlx-c API changes
mlx-c 0.6.0 added a `global_scale` parameter (may be null) to `mlx_dequantize` between `mode` and `dtype`. The FFI declaration in `mlx.zig` must match the installed header. When upgrading mlx-c, diff the headers in `/opt/homebrew/include/mlx/c/ops.h` against the `extern "c"` declarations in `src/mlx.zig`.

### Two binaries in the app bundle
The MLX Core `.app` bundle contains TWO binaries: `MLXCore` (Swift UI) and `mlx-serve` (Zig server). Both must be updated when making changes. The Swift app starts the Zig server as a child process. Forgetting to copy one binary after a rebuild is a common source of "it still doesn't work."

### WebSearch and Browse
The `webSearch` tool navigates to DuckDuckGo HTML search and extracts structured results (titles, URLs, snippets) via JavaScript. The `browse` tool's `readText` action navigates to the URL first, then extracts text — this ensures each browse returns the correct page content (not the previous page's).

### WKWebView requires main thread
`BrowserManager` is `@MainActor`. All WKWebView operations (navigate, readText, evaluateJS) must happen on the main thread. The WKWebView is created eagerly at app launch so tools work without the Browser window being open.

### Swift JSONSerialization quirks
- `[String: Any]` dictionaries serialize with non-deterministic key order
- Empty string `""` stays as `""` in JSON (not `null`); the server treats both as empty
- `Double` values like `0.7` serialize as `0.69999999999999996` (floating point); this is fine
- `arguments` in tool_calls must be a JSON String (e.g., `"{\"command\":\"ls\"}"`) not a nested dict; the server checks `if (v == .string)` to extract it
