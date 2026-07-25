#!/usr/bin/env bash
# =============================================================================
# Encoding experiments — clean, tagged driver
# (rewrite of the original run.sh + run_remaining.sh; see experiments/README.md)
#
# PREREQUISITES (start before running; adjust CUDA_VISIBLE_DEVICES/ports):
#   1. Controller LLM via vLLM (OpenAI-compatible) on $OPENAI_BASE_URL, e.g.:
#        vllm serve Qwen/Qwen2.5-VL-72B-Instruct --served-model-name Qwen2.5-VL-72B-Instruct \
#             --tensor-parallel-size 2 --port 8001 --limit-mm-per-prompt '{"image":8}'
#   2. Tool expert servers (see spagent/external_experts/*):
#        GroundingDINO :20022   (required by all E1 runs)
#        SAM2          :20020   (part of the dinosam tool stack)
#        Depth-AV2     :20019   (required by E2 and E3)
#        Moondream     :20024   (BLINK runs lean on it heavily)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8001/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

MODEL="${MODEL:-Qwen2.5-VL-72B-Instruct}"
DATASET="${DATASET:-dataset/cvbench_data_500sample.jsonl}"
BENCH_TAG="${BENCH_TAG:-cvbench_qwen72b}"        # matches results/ layout
OUT="experiments/results"

run () {  # run <script> <tag-path> [extra args...]
    local script="$1" out_dir="$2"; shift 2
    echo "=== [$(date +%H:%M:%S)] $script -> $out_dir ==="
    python "examples/evaluation/$script" \
        --model "$MODEL" --data_path "$DATASET" \
        --output_dir "$out_dir" "$@"
}

# -----------------------------------------------------------------------------
# E1  Detection-encoding ablation (depth colormap fixed to framework default)
#     image-only  : annotated bbox image, no text        (ObjectDetectionTool)
#     image+text  : annotated image + JSON text          (framework default)
#     text-only   : JSON xyxy text, no image             (Stage-I winner)
# -----------------------------------------------------------------------------
run evaluate_dinosam.py                     "$OUT/E1-det-encoding/$BENCH_TAG/det-image-only"
run evaluate_dinosam_det-image0-and-text.py "$OUT/E1-det-encoding/$BENCH_TAG/det-image-text"
run evaluate_dinosam_det-text-only.py       "$OUT/E1-det-encoding/$BENCH_TAG/det-text-only"

# -----------------------------------------------------------------------------
# E2  Depth-colormap ablation (detection fixed to image-only)
# -----------------------------------------------------------------------------
for cmap in gray plasma turbo; do
    run evaluate_depth_encoding.py "$OUT/E2-depth-colormap/$BENCH_TAG/$cmap" --colormap "$cmap"
done

# -----------------------------------------------------------------------------
# E3  Best combination (Stage-I winners: det text-only + depth turbo)
#     Was planned-but-commented-out in the original run.sh — enable when needed.
# -----------------------------------------------------------------------------
# run evaluate_best_combo.py "$OUT/E3-best-combo/$BENCH_TAG" \
#     --detection text-only --depth-colormap turbo

echo "All experiments done. Summaries: $OUT/**/spagent_evaluation_results_*_all.json"
