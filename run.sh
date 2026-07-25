export OPENAI_BASE_URL='http://localhost:8001/v1'
export OPENAI_API_KEY='EMPTY'

MODEL="Qwen2.5-VL-72B-Instruct"
DATASET="dataset/cvbench_data_500sample.jsonl"
TAG="CVBench_Qwen2.5-VL-72B-Instruct"

# ===========================================================
# Exp 1: Detection encoding ablation (depth fixed to turbo)
# ===========================================================

# 1a. image-only detection
python examples/evaluation/evaluate_dinosam.py \
    --model $MODEL \
    --data_path $DATASET \
    --output_dir ./test/${TAG}-det-image0

# 1b. image+text detection
python examples/evaluation/evaluate_dinosam_det-image0-and-text.py \
    --model $MODEL \
    --data_path $DATASET \
    --output_dir ./test/${TAG}-det-image0-and-text

# 1c. text-only detection
python examples/evaluation/evaluate_dinosam_det-text-only.py \
    --model $MODEL \
    --data_path $DATASET \
    --output_dir ./test/${TAG}-det-text-only

# ===========================================================
# Exp 2: Depth colormap ablation (detection fixed to image-only)
# ===========================================================

python examples/evaluation/evaluate_depth_encoding.py \
    --model $MODEL \
    --data_path $DATASET \
    --colormap gray \
    --output_dir ./test/${TAG}-depth-gray

python examples/evaluation/evaluate_depth_encoding.py \
    --model $MODEL \
    --data_path $DATASET \
    --colormap plasma \
    --output_dir ./test/${TAG}-depth-plasma

python examples/evaluation/evaluate_depth_encoding.py \
    --model $MODEL \
    --data_path $DATASET \
    --colormap turbo \
    --output_dir ./test/${TAG}-depth-turbo

# ===========================================================
# Exp 3: Best combination (fill in after Exp 1 & 2 results)
# ===========================================================

# python examples/evaluation/evaluate_best_combo.py \
#     --model $MODEL \
#     --data_path $DATASET \
#     --detection text-only \
#     --depth-colormap turbo \
#     --output_dir ./test/${TAG}-best-combo
