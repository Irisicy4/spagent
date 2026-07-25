#!/usr/bin/env bash
# =============================================================================
# Confidence-interval reruns for the REPORTED numbers (paper Table 4, SpAgent):
#   E1 detection-encoding x CV-Bench x Qwen2.5-VL-72B, N independent repeats.
#
# Run-to-run variance sources: controller sampling, tool stochasticity,
# parallel-worker ordering. Repeats land in:
#   experiments/results/E1-det-encoding/cvbench_qwen72b/ci_runs/<variant>/repeat<K>/
# Aggregate afterwards with experiments/ci_aggregate.py (mean +/- CI across
# repeats + per-item bootstrap).
#
# Prereqs: same as run_all.sh (controller vLLM on $OPENAI_BASE_URL,
# GroundingDINO :20022, SAM2 :20020, Depth :20019).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8001/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

MODEL="${MODEL:-Qwen2.5-VL-72B-Instruct}"
DATASET="${DATASET:-dataset_william/cvbench_data_500sample.jsonl}"
N_REPEATS="${N_REPEATS:-3}"
CI_OUT="experiments/results/E1-det-encoding/cvbench_qwen72b/ci_runs"

declare -A SCRIPTS=(
    [det-image-only]="evaluate_dinosam.py"
    [det-image-text]="evaluate_dinosam_det-image0-and-text.py"
    [det-text-only]="evaluate_dinosam_det-text-only.py"
)

for k in $(seq 1 "$N_REPEATS"); do
    for variant in det-text-only det-image-text det-image-only; do
        out="$CI_OUT/$variant/repeat$k"
        if ls "$out"/spagent_evaluation_results_*_all.json >/dev/null 2>&1; then
            echo "=== skip $variant repeat$k (already complete) ==="
            continue
        fi
        echo "=== [$(date +%H:%M:%S)] CI rerun: $variant repeat$k ==="
        python "examples/evaluation/${SCRIPTS[$variant]}" \
            --model "$MODEL" --data_path "$DATASET" --output_dir "$out"
    done
done
echo "CI reruns complete: $CI_OUT"
