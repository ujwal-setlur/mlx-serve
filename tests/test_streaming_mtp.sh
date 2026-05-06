#!/bin/bash
# Streaming MTP byte-equivalence test.
#
# Verifies that running the same temp=0 chat completion request with
# `stream: true` against `--mtp` produces *identical* concatenated text to
# the same request without `--mtp`. MTP's streaming path uses the new
# StreamingTokenStream adapter which yields the (t1, t2) batches one at a
# time — this test guards against a token being swallowed or duplicated
# across the batch boundary.
#
# Requires:
#   - A built mlx-serve binary (run `zig build -Doptimize=ReleaseFast` first)
#   - A model directory whose safetensors include MTP head weights
#     (matching `*.mtp.*`); most MLX-converted Qwen3.5/3.6 checkpoints have
#     the config field but no weights.
#
# Usage:
#   MTP_TEST_MODEL=/path/to/qwen3.5-with-mtp ./tests/test_streaming_mtp.sh [port]
#
# Without MTP_TEST_MODEL set, falls back to ~/.mlx-serve/models/Qwen3.5-4B-MTPLX-Speed.
# Auto-skips if the directory doesn't exist OR lacks `*.mtp.*` weights.

set -e

PORT=${1:-8090}
BASE="http://127.0.0.1:$PORT"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

MODEL="${MTP_TEST_MODEL:-$HOME/.mlx-serve/models/Qwen3.5-4B-MTPLX-Speed}"

if [ ! -d "$MODEL" ]; then
    echo -e "${YELLOW}SKIP${NC} test_streaming_mtp: model directory ($MODEL) not found."
    echo "  Set MTP_TEST_MODEL to a model directory whose safetensors include"
    echo "  MTP head weights (keys matching '*.mtp.*')."
    exit 0
fi

if [ ! -f "$MODEL/config.json" ]; then
    echo -e "${RED}FAIL${NC} $MODEL/config.json missing."
    exit 1
fi

# Confirm MTP weights are actually present (not just config metadata).
HAS_MTP_WEIGHTS=$(python3 -c "
import os, sys
try:
    from safetensors import safe_open
except ImportError:
    print('?')
    sys.exit(0)
d = '$MODEL'
n = 0
for f in os.listdir(d):
    if f.endswith('.safetensors'):
        try:
            with safe_open(os.path.join(d, f), framework='pt') as st:
                for k in st.keys():
                    if '.mtp.' in k or k.startswith('mtp.'):
                        n += 1
                        break
            if n: break
        except: pass
print('1' if n else '0')
" 2>/dev/null)

if [ "$HAS_MTP_WEIGHTS" = "?" ]; then
    echo -e "${YELLOW}SKIP${NC} test_streaming_mtp: 'safetensors' Python package not available — cannot verify MTP weights are present."
    exit 0
fi

if [ "$HAS_MTP_WEIGHTS" != "1" ]; then
    echo -e "${YELLOW}SKIP${NC} test_streaming_mtp: $MODEL has no '*.mtp.*' tensors in its safetensors. The MLX conversion likely stripped them."
    exit 0
fi

BINARY="${MLX_SERVE_BINARY:-./zig-out/bin/mlx-serve}"
if [ ! -x "$BINARY" ]; then
    echo -e "${RED}FAIL${NC} $BINARY not found or not executable. Build first with 'zig build -Doptimize=ReleaseFast'."
    exit 1
fi

PROMPT="Write the first line of the Linux kernel boot message."
JSON_NONSTREAM=$(cat <<EOF
{
  "model": "mlx-serve",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "max_tokens": 64,
  "temperature": 0.0,
  "stream": false
}
EOF
)
JSON_STREAM=$(cat <<EOF
{
  "model": "mlx-serve",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "max_tokens": 64,
  "temperature": 0.0,
  "stream": true
}
EOF
)

# Extract concatenated `delta.content` from an SSE chat-completions stream.
sse_concat_content() {
    python3 -c '
import sys, json
out = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "):
        continue
    payload = line[6:].strip()
    if payload == "[DONE]" or not payload:
        continue
    try:
        ev = json.loads(payload)
    except Exception:
        continue
    for ch in ev.get("choices", []) or []:
        delta = ch.get("delta", {}) or {}
        text = delta.get("content")
        if isinstance(text, str):
            out.append(text)
sys.stdout.write("".join(out))
'
}

run_request() {
    local label="$1" mtp_flag="$2" mode="$3"
    echo -e "  starting server ($label)..." >&2
    local logfile=$(mktemp)
    "$BINARY" --model "$MODEL" --serve --port "$PORT" $mtp_flag > "$logfile" 2>&1 &
    local pid=$!
    local up=0
    for i in $(seq 1 60); do
        if curl -s -f "$BASE/health" > /dev/null 2>&1; then
            up=1
            break
        fi
        sleep 1
    done
    if [ "$up" != "1" ]; then
        echo -e "  ${RED}FAIL${NC} server did not become healthy in 60s" >&2
        cat "$logfile" | tail -20 >&2
        kill $pid 2>/dev/null || true
        rm -f "$logfile"
        return 1
    fi

    local body
    if [ "$mode" = "stream" ]; then
        body=$(echo "$JSON_STREAM" | curl -s -N -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" | sse_concat_content)
    else
        body=$(echo "$JSON_NONSTREAM" | curl -s -X POST -H "Content-Type: application/json" -d @- "$BASE/v1/chat/completions" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'], end='')")
    fi

    kill $pid 2>/dev/null || true
    wait $pid 2>/dev/null || true
    # Give the kernel a moment to fully release the port between back-to-back
    # server runs (previous behavior occasionally raced).
    sleep 1
    rm -f "$logfile"
    echo "$body"
}

echo "== streaming-MTP byte-equivalence test =="
echo "  model: $MODEL"
echo "  prompt: $PROMPT"
echo

pkill -f "mlx-serve.*--port $PORT" 2>/dev/null || true
sleep 1

OUT_REGULAR_STREAM=$(run_request "stream, no --mtp" "--no-mtp" "stream") || exit 1
echo "  baseline streaming output captured ($(echo -n "$OUT_REGULAR_STREAM" | wc -c) bytes)"

sleep 2

OUT_MTP_STREAM=$(run_request "stream, --mtp" "--mtp" "stream") || exit 1
echo "  MTP streaming output captured ($(echo -n "$OUT_MTP_STREAM" | wc -c) bytes)"

sleep 2

OUT_MTP_NONSTREAM=$(run_request "non-stream, --mtp" "--mtp" "nostream") || exit 1
echo "  MTP non-streaming output captured ($(echo -n "$OUT_MTP_NONSTREAM" | wc -c) bytes)"

if [ "$OUT_REGULAR_STREAM" = "$OUT_MTP_STREAM" ] && [ "$OUT_REGULAR_STREAM" = "$OUT_MTP_NONSTREAM" ]; then
    echo -e "${GREEN}PASS${NC} streaming + non-streaming MTP output is byte-identical to regular streaming"
    exit 0
else
    echo -e "${RED}FAIL${NC} outputs differ:"
    echo "  regular stream:"
    printf '    %s\n' "$OUT_REGULAR_STREAM"
    echo "  MTP stream:"
    printf '    %s\n' "$OUT_MTP_STREAM"
    echo "  MTP non-stream:"
    printf '    %s\n' "$OUT_MTP_NONSTREAM"
    diff <(echo "$OUT_REGULAR_STREAM") <(echo "$OUT_MTP_STREAM") | sed 's/^/    /' || true
    exit 1
fi
