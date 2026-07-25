#!/usr/bin/env bash
# =============================================================================
# PARALLEL confidence-interval reruns (all variants x repeats at once).
#
# Each run gets a private sandbox cwd (own outputs/; dataset+checkpoints
# symlinked) because the tools write intermediate artifacts to relative
# outputs/<name based on image stem> — concurrent runs sharing one outputs/
# would race and cross-contaminate the images fed back to the controller.
# vLLM continuous-batches the concurrent controller calls; the flask tool
# servers serialize their small requests (acceptable queueing).
#
# Results land in the SAME layout as rerun_ci.sh:
#   experiments/results/E1-det-encoding/cvbench_qwen32b/ci_runs/<variant>/repeat<K>/
# Completed repeats are skipped, so this is resumable / compatible with the
# sequential driver. Aggregate with: python3 experiments/ci_aggregate.py --ci-runs
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8002/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

MODEL="${MODEL:-Qwen2.5-VL-32B-Instruct}"
DATASET_REL="dataset/cvbench_data_500sample.jsonl"      # william-native path
N_REPEATS="${N_REPEATS:-3}"
CI_OUT="$ROOT/experiments/results/E1-det-encoding/cvbench_qwen32b/ci_runs"
SANDBOXES="$ROOT/experiments/.parallel_sandboxes"       # scratch; gitignored via *. pattern below

declare -A SCRIPTS=(
    [det-image-only]="evaluate_dinosam.py"
    [det-image-text]="evaluate_dinosam_det-image0-and-text.py"
    [det-text-only]="evaluate_dinosam_det-text-only.py"
)

pids=()
for k in $(seq 1 "$N_REPEATS"); do
    for variant in det-text-only det-image-text det-image-only; do
        out="$CI_OUT/$variant/repeat$k"
        if ls "$out"/spagent_evaluation_results_*_all.json >/dev/null 2>&1; then
            echo "skip $variant repeat$k (complete)"
            continue
        fi
        sb="$SANDBOXES/32b-$variant-r$k"
        mkdir -p "$sb/outputs" "$out"
        ln -sfn "$ROOT/dataset"     "$sb/dataset"
        ln -sfn "$ROOT/checkpoints" "$sb/checkpoints"
        (
            cd "$sb"
            exec python "$ROOT/examples/evaluation/${SCRIPTS[$variant]}" \
                --model "$MODEL" --data_path "$DATASET_REL" \
                --output_dir "$out"
        ) > "$sb/run.log" 2>&1 &
        pids+=($!)
        echo "launched 32b-$variant repeat$k (pid ${pids[-1]}, sandbox $sb)"
    done
done

echo "waiting on ${#pids[@]} parallel runs..."
fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=$((fail+1))
done
echo "parallel CI reruns finished ($fail failures). Results: $CI_OUT"
