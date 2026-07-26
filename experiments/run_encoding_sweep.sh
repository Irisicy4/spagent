#!/usr/bin/env bash
# =============================================================================
# E4: single-tool encoding sweep (judge-exp.md N1 + depth single-tool arm).
#
# Arms (each registers EXACTLY ONE tool; baseline registers none):
#   none   on all 3 slices                       (proves tool benefit)
#   depth  gray|plasma|turbo   on depth_distance (Depth+Distance tasks, 216)
#   refseg overlay|instance|polygon|maskonly on relation (120)
#   semseg overlay|instance|polygon|maskonly on count    (164)
#
# Usage: run_encoding_sweep.sh <batch> [n_repeats]
#   batch: base | depth | refseg | semseg | mirror32b
# Env: BURL (controller base_url), MODEL. Sandboxed + resumable.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

BATCH="${1:?batch required: base|depth|refseg|semseg|mirror32b}"
N_REPEATS="${2:-3}"
BURL="${BURL:-http://localhost:8001/v1}"
MODEL="${MODEL:-Qwen2.5-VL-72B-Instruct}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
SLICES="$ROOT/experiments/data_slices"
SANDBOXES="$ROOT/experiments/.parallel_sandboxes"

model_dir () { case "$1" in *72B*) echo cvbench_qwen72b;; *32B*) echo cvbench_qwen32b;; *3B*) echo cvbench_qwen3b;; *) echo "cvbench_$1";; esac; }
RES="$ROOT/experiments/results/E4-encoding-single-tool/$(model_dir "$MODEL")/ci_runs"

launch () {  # launch <arm-tag> <repeat> <tool> <encoding> <slice> <workers>
    local tag="$1" k="$2" tool="$3" enc="$4" slice="$5" workers="$6"
    local out="$RES/$tag/repeat$k"
    local name="enc-$(basename "$(model_dir "$MODEL")")-$tag-r$k"
    if ls "$out"/spagent_evaluation_results_*_all.json >/dev/null 2>&1; then
        echo "skip $name (complete)"; return
    fi
    local sb="$SANDBOXES/$name"
    mkdir -p "$sb/outputs" "$out"
    ln -sfn "$ROOT/dataset"     "$sb/dataset"
    ln -sfn "$ROOT/checkpoints" "$sb/checkpoints"
    (
        cd "$sb"
        export OPENAI_BASE_URL="$BURL"
        exec python "$ROOT/examples/evaluation/evaluate_encoding_single_tool.py" \
            --tool "$tool" --encoding "$enc" \
            --data_path "$SLICES/cvbench_$slice.jsonl" \
            --model "$MODEL" --max_workers "$workers" --output_dir "$out"
    ) > "$sb/run.log" 2>&1 &
    pids+=($!)
    echo "launched $name (pid ${pids[-1]})"
}

pids=()
for k in $(seq 1 "$N_REPEATS"); do
    case "$BATCH" in
        base)
            launch "none-depth_distance" "$k" none none depth_distance 4
            launch "none-relation"       "$k" none none relation 4
            launch "none-count"          "$k" none none count 4
            ;;
        depth)
            for cm in gray plasma turbo; do
                launch "depth-$cm" "$k" depth "$cm" depth_distance 4
            done
            ;;
        refseg)
            for enc in overlay instance polygon maskonly; do
                launch "refseg-$enc" "$k" refseg "$enc" relation 2
            done
            ;;
        semseg)
            for enc in overlay instance polygon maskonly; do
                launch "semseg-$enc" "$k" semseg "$enc" count 2
            done
            ;;
        mirror32b)
            # 32B single-GPU vLLM is deterministic: 1 repeat covers the mirror
            launch "none-depth_distance" "$k" none none depth_distance 4
            launch "none-relation"       "$k" none none relation 4
            launch "none-count"          "$k" none none count 4
            for cm in gray plasma turbo; do launch "depth-$cm" "$k" depth "$cm" depth_distance 4; done
            for enc in overlay instance polygon maskonly; do launch "refseg-$enc" "$k" refseg "$enc" relation 2; done
            for enc in overlay instance polygon maskonly; do launch "semseg-$enc" "$k" semseg "$enc" count 2; done
            ;;
        *) echo "unknown batch $BATCH"; exit 1;;
    esac
done

echo "waiting on ${#pids[@]} $BATCH runs..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "$BATCH finished ($fail failures)."
