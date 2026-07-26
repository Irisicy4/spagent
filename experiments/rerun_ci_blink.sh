#!/usr/bin/env bash
# =============================================================================
# BLINK CI reruns (batch C3) — faithful four-tool stack (--with_moondream).
#   blink_qwen3b : det-image-only x3  +  det-text-only x3   (controller :8004)
#   blink_qwen72b: det-image-only x3                        (controller :8003,
#                  dedicated 72B replica so the CV-Bench sweep is unaffected)
# Requires: moondream proxy on :20024 (MOONDREAM_API_KEY provided at its
# launch), GDINO :20022, SAM2 :20020, Depth :20019.
# Sandboxed + resumable like the other CI drivers.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
DATASET_REL="dataset/BLINK_All_Tasks_500sample.jsonl"
N_REPEATS="${N_REPEATS:-3}"
RES="$ROOT/experiments/results/E1-det-encoding"
SANDBOXES="$ROOT/experiments/.parallel_sandboxes"

launch () {  # launch <name> <out> <base_url> <model> [extra args...]
    local name="$1" out="$2" burl="$3" model="$4"; shift 4
    if ls "$out"/spagent_evaluation_results_*_all.json >/dev/null 2>&1; then
        echo "skip $name (complete)"; return
    fi
    local sb="$SANDBOXES/$name"
    mkdir -p "$sb/outputs" "$out"
    ln -sfn "$ROOT/dataset"     "$sb/dataset"
    ln -sfn "$ROOT/checkpoints" "$sb/checkpoints"
    (
        cd "$sb"
        export OPENAI_BASE_URL="$burl"
        exec python "$ROOT/examples/evaluation/evaluate_dinosam.py" \
            --model "$model" --data_path "$DATASET_REL" \
            --with_moondream --output_dir "$out" "$@"
    ) > "$sb/run.log" 2>&1 &
    pids+=($!)
    echo "launched $name (pid ${pids[-1]})"
}

pids=()
for k in $(seq 1 "$N_REPEATS"); do
    launch "blink3b-image-only-r$k" \
        "$RES/blink_qwen3b/ci_runs/det-image-only/repeat$k" \
        http://localhost:8004/v1 Qwen2.5-VL-3B-Instruct
    launch "blink3b-text-only-r$k" \
        "$RES/blink_qwen3b/ci_runs/det-text-only/repeat$k" \
        http://localhost:8004/v1 Qwen2.5-VL-3B-Instruct --detection_text_only
    launch "blink72b-image-only-r$k" \
        "$RES/blink_qwen72b/ci_runs/det-image-only/repeat$k" \
        http://localhost:8003/v1 Qwen2.5-VL-72B-Instruct
done

echo "waiting on ${#pids[@]} BLINK runs..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "BLINK CI finished ($fail failures)."
