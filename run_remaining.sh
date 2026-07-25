export OPENAI_BASE_URL='http://localhost:8001/v1'
export OPENAI_API_KEY='EMPTY'

MODEL="Qwen2.5-VL-72B-Instruct"
DATASET="dataset/cvbench_data_500sample.jsonl"
TAG="CVBench_Qwen2.5-VL-72B-Instruct"

# 1c. text-only detection
python examples/evaluation/evaluate_dinosam_det-text-only.py \
    --model $MODEL \
    --data_path $DATASET \
    --output_dir ./test/${TAG}-det-text-only

# Exp 2: Depth colormap ablation
python examples/evaluation/evaluate_depth_encoding.py \
    --model $MODEL --data_path $DATASET \
    --colormap gray \
    --output_dir ./test/${TAG}-depth-gray

python examples/evaluation/evaluate_depth_encoding.py \
    --model $MODEL --data_path $DATASET \
    --colormap plasma \
    --output_dir ./test/${TAG}-depth-plasma

python examples/evaluation/evaluate_depth_encoding.py \
    --model $MODEL --data_path $DATASET \
    --colormap turbo \
    --output_dir ./test/${TAG}-depth-turbo
