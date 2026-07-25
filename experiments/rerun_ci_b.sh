#!/usr/bin/env bash
# =============================================================================
# PARALLEL CI reruns — batch B: E2 depth-colormap (gray/plasma/turbo) and
# E3 best-combo (det text-only + depth turbo), x N_REPEATS each.
# Same sandbox isolation + resumable layout as rerun_ci_parallel.sh (batch A).
# Safe to run concurrently with batch A: vLLM batches all controller calls;
# sandboxes prevent artifact races.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8001/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
MODEL="${MODEL:-Qwen2.5-VL-72B-Instruct}"
DATASET_REL="dataset/cvbench_data_500sample.jsonl"
N_REPEATS="${N_REPEATS:-3}"
RES="$ROOT/experiments/results"
SANDBOXES="$ROOT/experiments/.parallel_sandboxes"

launch () {  # launch <sandbox-name> <out-dir> <script> [extra args...]
    local name="$1" out="$2" script="$3"; shift 3
    if ls "$out"/spagent_evaluation_results_*_all.json >/dev/null 2>&1; then
        echo "skip $name (complete)"; return
    fi
    local sb="$SANDBOXES/$name"
    mkdir -p "$sb/outputs" "$out"
    ln -sfn "$ROOT/dataset"     "$sb/dataset"
    ln -sfn "$ROOT/checkpoints" "$sb/checkpoints"
    (
        cd "$sb"
        exec python "$ROOT/examples/evaluation/$script" \
            --model "$MODEL" --data_path "$DATASET_REL" \
            --output_dir "$out" "$@"
    ) > "$sb/run.log" 2>&1 &
    pids+=($!)
    echo "launched $name (pid ${pids[-1]})"
}

pids=()
for k in $(seq 1 "$N_REPEATS"); do
    for cmap in gray plasma turbo; do
        launch "depth-$cmap-r$k" \
            "$RES/E2-depth-colormap/cvbench_qwen72b/ci_runs/$cmap/repeat$k" \
            evaluate_depth_encoding.py --colormap "$cmap"
    done
    launch "best-combo-r$k" \
        "$RES/E3-best-combo/cvbench_qwen72b/ci_runs/combo/repeat$k" \
        evaluate_best_combo.py --detection text-only --depth-colormap turbo
done

echo "waiting on ${#pids[@]} batch-B runs..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "batch B finished ($fail failures)."
