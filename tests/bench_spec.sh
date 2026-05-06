#!/bin/bash
# Focused spec-decode benchmark: measures decode tok/s for (none/PLD/drafter) ×
# (heavy-echo / creative-novel) on the models we ship support for. Drives the
# default-on flip decision for `--pld` and `--drafter`.
#
# Usage: ./tests/bench_spec.sh [runs]   (default 5 runs per cell, run 1 is warmup)
#
# Output: pipe-separated rows to stdout; per-run debug to stderr.
set -uo pipefail

RUNS="${1:-5}"
BIN="./zig-out/bin/mlx-serve"
PORT=8091
MODELS_DIR="$HOME/.mlx-serve/models"

HEAVY_ECHO='Repeat the following Python code verbatim, but rename the function `compute_total` to `total`:
```python
def compute_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    if total > 100:
        total *= 0.9
    return total
```
Output ONLY the renamed code, nothing else.'

CREATIVE='Write a 30-line poem about a lighthouse keeper at the end of the world. Use vivid imagery.'

start_server() {
    local model="$1" extra="$2"
    pkill -f "mlx-serve" 2>/dev/null; sleep 0.5
    eval "$BIN --model '$model' --serve --port $PORT --log-level info $extra" >/tmp/bench-srv.log 2>&1 &
    SRV=$!
    for i in $(seq 1 60); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
        sleep 0.5
    done
    echo "  ERROR: server failed to start" >&2; return 1
}

stop_server() { kill ${SRV:-0} 2>/dev/null; wait ${SRV:-0} 2>/dev/null; }

run_cell() {
    local label="$1" model_path="$2" spec_args="$3" prompt_label="$4" prompt="$5"
    if [[ ! -d "$model_path" ]]; then echo "$label|$prompt_label|SKIP" ; return; fi
    if ! start_server "$model_path" "$spec_args"; then return 1; fi
    local toks_per_s_vals=()
    for r in $(seq 1 $RUNS); do
        : > /tmp/bench-srv.log
        local body=$(jq -nc --arg p "$prompt" '{model:"x",messages:[{role:"user",content:$p}],max_tokens:120,temperature:0,stream:false}')
        curl -sf -m 90 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d "$body" -o /dev/null
        # Server log line:  "<- N+M tokens (Xms) [prefill: P tok/s, decode: D tok/s] [stop]"
        local tps=$(LC_ALL=C grep -aoE 'decode: [0-9.]+ tok/s' /tmp/bench-srv.log | tail -1 | LC_ALL=C grep -aoE '[0-9]+\.[0-9]+' | head -1)
        if [[ -z "$tps" ]]; then tps="0"; fi
        toks_per_s_vals+=("$tps")
        echo "  $label | $prompt_label | run=$r | tps=$tps" >&2
    done
    stop_server
    local timed=("${toks_per_s_vals[@]:1}")
    local timed_csv=$(IFS=,; echo "${timed[*]}")
    local stats=$(python3 -c "import statistics as s; v=[float(x) for x in '$timed_csv'.split(',') if x]; print(f'{s.mean(v):.2f}|{min(v):.2f}|{max(v):.2f}')" 2>/dev/null)
    if [[ -z "$stats" ]]; then stats="0|0|0"; fi
    echo "$label|$prompt_label|$stats"
}

echo "label|prompt|mean_tps|min_tps|max_tps"

QWEN="$MODELS_DIR/Qwen3.5-4B-MLX-4bit"
run_cell "Qwen3.5-4B/none"    "$QWEN" ""        heavy-echo  "$HEAVY_ECHO"
run_cell "Qwen3.5-4B/pld"     "$QWEN" "--pld"   heavy-echo  "$HEAVY_ECHO"
run_cell "Qwen3.5-4B/none"    "$QWEN" ""        creative    "$CREATIVE"
run_cell "Qwen3.5-4B/pld"     "$QWEN" "--pld"   creative    "$CREATIVE"

GEMMA="$MODELS_DIR/gemma-4-e4b-it-4bit"
DRAFTER="$MODELS_DIR/gemma-4-E4B-it-assistant-bf16"
run_cell "Gemma-4-E4B/none"     "$GEMMA" ""                        heavy-echo  "$HEAVY_ECHO"
run_cell "Gemma-4-E4B/pld"      "$GEMMA" "--pld"                   heavy-echo  "$HEAVY_ECHO"
run_cell "Gemma-4-E4B/drafter"  "$GEMMA" "--drafter $DRAFTER"      heavy-echo  "$HEAVY_ECHO"
run_cell "Gemma-4-E4B/none"     "$GEMMA" ""                        creative    "$CREATIVE"
run_cell "Gemma-4-E4B/pld"      "$GEMMA" "--pld"                   creative    "$CREATIVE"
run_cell "Gemma-4-E4B/drafter"  "$GEMMA" "--drafter $DRAFTER"      creative    "$CREATIVE"

LFM="$MODELS_DIR/LFM2.5-350M-MLX-8bit"
run_cell "LFM2.5-350M/none"     "$LFM" ""        heavy-echo  "$HEAVY_ECHO"
run_cell "LFM2.5-350M/pld"      "$LFM" "--pld"   heavy-echo  "$HEAVY_ECHO"
run_cell "LFM2.5-350M/none"     "$LFM" ""        creative    "$CREATIVE"
run_cell "LFM2.5-350M/pld"      "$LFM" "--pld"   creative    "$CREATIVE"
