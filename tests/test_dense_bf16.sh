#!/usr/bin/env bash
# Integration test for fully-dense bf16 MLX models — checkpoints with NO
# "quantization" key and no .scales/.biases tensors (quant_bits==0). Exercises:
#   • dense embedding + tied/untied lm_head (transposed-view projection)
#   • bf16 attention + MLP (pre-transposed weights, plain matmul)
#   • dense MoE expert dispatch via mlx_gather_mm  (Qwen 3.6 MoE)
#   • Per-Layer Embeddings (PLE) bf16                (Gemma 4 E-series)
#   • KV-layer sharing where shared layers drop k/v weights (Gemma E2B/E4B)
#
# Iterates a candidate list, testing each model present and skipping the rest
# (CI-friendly). Override the list with DENSE_BF16_MODELS (colon-separated abs
# paths). The ~65 GB Qwen MoE is opt-in via DENSE_BF16_HEAVY=1.
#
# Each model must (a) load, (b) answer a near-deterministic greedy probe with
# "Paris", and (c) produce a non-degenerate open-ended sentence — so the
# coherent-but-degraded failure mode is caught, not just emptiness.

set -euo pipefail

PORT="${PORT:-8090}"
HOST="127.0.0.1"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-360}"
MODELS_ROOT="${MODELS_ROOT:-$HOME/.mlx-serve/models/mlx-community}"

if [ -n "${DENSE_BF16_MODELS:-}" ]; then
    IFS=':' read -r -a CANDIDATES <<< "$DENSE_BF16_MODELS"
else
    CANDIDATES=(
        "$MODELS_ROOT/gemma-4-E2B-it-qat-bf16"   # initStandardLayers + PLE + KV-sharing
        "$MODELS_ROOT/Qwen3.5-0.8B-MLX-bf16"     # initMoeLayers dense (GatedDeltaNet)
    )
    [ "${DENSE_BF16_HEAVY:-0}" = "1" ] && CANDIDATES+=("$MODELS_ROOT/Qwen3.6-35B-A3B-bf16") # MoE, ~65GB
fi

if [ ! -x ./zig-out/bin/mlx-serve ]; then
    echo "FAIL: ./zig-out/bin/mlx-serve not built — run 'zig build -Doptimize=ReleaseFast' first"
    exit 1
fi

probe() {
    curl -sf "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":${2:-48},\"temperature\":0.0,\"stream\":false}" \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
}

tested=0
for MODEL in "${CANDIDATES[@]}"; do
    if [ ! -d "$MODEL" ]; then
        echo "skip: $MODEL not found"
        continue
    fi
    tested=$((tested + 1))
    LOG=$(mktemp -t mlx-dense-bf16.XXXXXX.log)
    echo "[dense-bf16] === $(basename "$MODEL") ==="
    echo "[dense-bf16] starting server (log=$LOG)"
    ./zig-out/bin/mlx-serve --model "$MODEL" --serve --host "$HOST" --port "$PORT" --no-vision --log-level info >"$LOG" 2>&1 &
    SERVER_PID=$!
    trap 'kill -9 $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true' EXIT

    for _ in $(seq 1 "$LOAD_TIMEOUT"); do
        curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && break
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "FAIL: server crashed during load"; tail -40 "$LOG"; exit 1
        fi
        sleep 1
    done
    if ! curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        echo "FAIL: $(basename "$MODEL") did not become healthy within ${LOAD_TIMEOUT}s"; tail -40 "$LOG"; exit 1
    fi

    fact=$(probe "What is the capital of France?" 48)
    echo "[dense-bf16] capital-of-France: $fact"
    if ! printf '%s' "$fact" | grep -qi "paris"; then
        echo "FAIL: greedy output did not contain 'Paris' — generation likely degraded"; exit 1
    fi

    greet=$(probe "Write a one-sentence greeting." 30)
    echo "[dense-bf16] greeting: $greet"
    # Degeneracy guard: catch a collapsed model that emits one token repeated
    # ("the the the the") or nothing at all. A terse-but-valid reply ("Hello!")
    # is fine, so only flag empty output or a run of ≥4 words with no variety.
    total_words=$(printf '%s' "$greet" | tr ' ' '\n' | grep -c . || true)
    uniq_words=$(printf '%s' "$greet" | tr ' ' '\n' | sort -u | grep -c . || true)
    if [ -z "${greet// /}" ] || { [ "$total_words" -ge 4 ] && [ "$uniq_words" -lt 2 ]; }; then
        echo "FAIL: degenerate/empty open-ended output: $greet"; exit 1
    fi

    # MoE-prefill coherence gate. The dense-bf16 MoE sorted-prefill path (Qwen 3.6
    # GatedDeltaNet + experts) once produced FLUENT-BUT-WRONG output — the model
    # answered clean prompts with "your message seems jumbled/unclear/incomplete"
    # because mlx_gather_mm(sorted_indices=true) silently mis-routed dense experts.
    # That failure mode slips past the "Paris"/degeneracy checks above (output was
    # non-degenerate and the high-confidence first token survived), so probe a tiny
    # multi-token reasoning prompt: a healthy model says "4"; the broken one deflects.
    math=$(probe "What is 2+2? Answer with just the number." 24)
    echo "[dense-bf16] 2+2: $math"
    if ! printf '%s' "$math" | grep -q "4"; then
        echo "FAIL: '2+2' answer lacks '4' — MoE prefill likely corrupted (sorted-gather regression)"; exit 1
    fi
    if printf '%s' "$math" | grep -qiE "jumbled|unclear|incomplete|corrupt|don't understand|can't understand"; then
        echo "FAIL: model called a clean prompt unintelligible — fluent-but-wrong MoE corruption"; exit 1
    fi

    echo "[dense-bf16] PASS — $(basename "$MODEL")"
    kill -9 $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
    trap - EXIT
done

if [ "$tested" -eq 0 ]; then
    echo "skip: no dense bf16 models present"
    exit 0
fi
echo "[dense-bf16] ALL PASS ($tested model(s))"
exit 0
