#!/usr/bin/env bash
# =============================================================================
# E5: tool-tailored benchmarks (task #10) — single-tool encoding sweeps on
# datasets chosen so each tool is load-bearing:
#   vstar   (191, detection) : none | det image/imagetext/textnorm/textpixel
#   tallyqa (300, seg count) : none | semseg overlay/instance/polygon
#   da2k    (300, depth)     : none | depth gray/plasma/turbo
# Usage: run_tailored_sweep.sh <batch> [n_repeats]   batch: vstar|tallyqa|da2k
# Env: BURL, MODEL. Sandboxed + resumable (skips completed repeats).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

BATCH="${1:?batch required: vstar|tallyqa|da2k}"
N_REPEATS="${2:-3}"
BURL="${BURL:-http://localhost:8003/v1}"
MODEL="${MODEL:-Qwen2.5-VL-72B-Instruct}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
TD="$ROOT/experiments/data_slices/tailored"
SANDBOXES="$ROOT/experiments/.parallel_sandboxes"
RES="$ROOT/experiments/results/E5-tailored/cvbench_qwen72b/ci_runs"

launch () {  # launch <arm-tag> <repeat> <tool> <encoding> <data-file> <workers>
    local tag="$1" k="$2" tool="$3" enc="$4" data="$5" workers="$6"
    local out="$RES/$tag/repeat$k"
    local name="e5-$tag-r$k"
    if ls "$out"/spagent_evaluation_results_*_all.json >/dev/null 2>&1; then
        echo "skip $name (complete)"; return
    fi
    local sb="$SANDBOXES/$name"
    mkdir -p "$sb/outputs" "$out"
    ln -sfn "$ROOT/dataset" "$sb/dataset"
    ln -sfn "$ROOT/checkpoints" "$sb/checkpoints"
    (
        cd "$sb"
        export OPENAI_BASE_URL="$BURL"
        exec python "$ROOT/examples/evaluation/evaluate_encoding_single_tool.py" \
            --tool "$tool" --encoding "$enc" \
            --data_path "$TD/$data" --image_base_path "$TD" \
            --model "$MODEL" --max_workers "$workers" --output_dir "$out"
    ) > "$sb/run.log" 2>&1 &
    pids+=($!)
    echo "launched $name (pid ${pids[-1]})"
}

pids=()
for k in $(seq 1 "$N_REPEATS"); do
    case "$BATCH" in
        vstar)
            launch "vstar-none" "$k" none none vstar.jsonl 4
            for enc in image imagetext textnorm textpixel; do
                launch "vstar-det-$enc" "$k" det "$enc" vstar.jsonl 4
            done
            ;;
        tallyqa)
            launch "tallyqa-none" "$k" none none tallyqa_count.jsonl 4
            for enc in overlay instance polygon; do
                launch "tallyqa-semseg-$enc" "$k" semseg "$enc" tallyqa_count.jsonl 2
            done
            ;;
        da2k)
            launch "da2k-none" "$k" none none da2k_depth.jsonl 4
            for enc in gray plasma turbo; do
                launch "da2k-depth-$enc" "$k" depth "$enc" da2k_depth.jsonl 4
            done
            ;;
        *) echo "unknown batch $BATCH"; exit 1;;
    esac
done

echo "waiting on ${#pids[@]} $BATCH runs..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "$BATCH finished ($fail failures)."
