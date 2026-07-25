"""
Exp 3: Best detection encoding + best depth colormap combination.

Run after Exp 1 and Exp 2 to verify that the individually-optimal settings
remain optimal when combined.

Usage:
  python examples/evaluation/evaluate_best_combo.py \
      --model qwen2.5-vl-32b \
      --data_path dataset/cvbench_data_500sample.jsonl \
      --detection text-only \
      --depth-colormap turbo \
      --output_dir ./test/CVBench_Qwen2.5-VL-32B-Instruct-best-combo
"""

import json
import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from spagent.core import DataCollector
from spagent.core.prompts import GENERAL_VISION_SYSTEM_PROMPT, GENERAL_VISION_CONTINUATION_HINT
from spagent.external_experts.GroundingDINO import GroundingDINOImageAndTextTool
from spagent.external_experts.GroundingDINO_text_only import GroundingDINOTextOnlyTool
from spagent.tools import DepthEstimationTool, SegmentationTool, ObjectDetectionTool, MoondreamTool
from spagent.utils.utils import print_evaluation_results
from spagent_evaluation import evaluate_tool_config

TOOL_SERVERS = {
    "sam2": "http://localhost:20020",
    "grounding_dino": "http://localhost:20022",
    "depth": "http://localhost:20019",
    "moondream": "http://localhost:20024",
}

DETECTION_TOOLS = {
    "image0": lambda: ObjectDetectionTool(use_mock=False, server_url=TOOL_SERVERS["grounding_dino"]),
    "image0-and-text": lambda: GroundingDINOImageAndTextTool(server_url=TOOL_SERVERS["grounding_dino"]),
    "text-only": lambda: GroundingDINOTextOnlyTool(server_url=TOOL_SERVERS["grounding_dino"]),
}


def main():
    parser = argparse.ArgumentParser(description='Exp 3: best detection + best depth combination')

    parser.add_argument('--data_path', type=str, default='dataset/cvbench_data.jsonl')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_workers', type=int, default=4)
    parser.add_argument('--image_base_path', type=str, default='dataset')
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--max_iterations', type=int, default=3)
    parser.add_argument('--task', type=str, default="all")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument(
        '--detection', type=str, required=True,
        choices=list(DETECTION_TOOLS.keys()),
        help='Detection encoding to use (best from Exp 1)'
    )
    parser.add_argument(
        '--depth-colormap', type=str, required=True,
        choices=['gray', 'plasma', 'turbo'],
        dest='depth_colormap',
        help='Depth colormap to use (best from Exp 2)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return
    if not os.path.exists(args.image_base_path):
        print(f"Error: Image base path not found at {args.image_base_path}")
        return

    config_name = f"best_combo_det{args.detection}_depth{args.depth_colormap}"
    print(f"Running Exp 3: detection={args.detection}, depth={args.depth_colormap}")

    tools = [
        DETECTION_TOOLS[args.detection](),
        SegmentationTool(use_mock=False, server_url=TOOL_SERVERS["sam2"]),
        DepthEstimationTool(use_mock=False, server_url=TOOL_SERVERS["depth"], colormap=args.depth_colormap),
        MoondreamTool(use_mock=False, server_url=TOOL_SERVERS["moondream"]),
    ]

    results = evaluate_tool_config(
        config_name=config_name,
        tools=tools,
        data_path=args.data_path,
        image_base_path=args.image_base_path,
        model=args.model,
        max_samples=args.max_samples,
        max_workers=args.max_workers,
        max_iterations=args.max_iterations,
        system_prompt=GENERAL_VISION_SYSTEM_PROMPT,
        continuation_hint=GENERAL_VISION_CONTINUATION_HINT,
        temperature=args.temperature,
        seed=args.seed,
        top_p=args.top_p,
    )

    print(f"\nResults for {config_name}:")
    print_evaluation_results(results)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"spagent_evaluation_results_{args.model.replace('-', '_')}_{args.max_iterations}_{args.task}.json"
    )
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({config_name: results}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
