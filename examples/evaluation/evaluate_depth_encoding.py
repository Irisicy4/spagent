"""
Evaluate depth tool encoding (colormap) effect on performance.

Runs the same tool suite with three different depth colormaps:
  gray   - grayscale, white=close black=far
  plasma - yellow=close purple=far
  turbo  - blue=close green-yellow=mid red=far (default)

Each colormap is evaluated as a separate config. Use --colormap to run a
single condition, or omit it to run all three sequentially.

Results are saved to:
  {output_dir}/spagent_evaluation_results_{model}_{max_iterations}_{task}.json

Directory naming convention for collect_results.py:
  {Dataset}_{Model}-depth-{colormap}
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
from spagent.tools import DepthEstimationTool, SegmentationTool, ObjectDetectionTool
from spagent.utils.utils import print_evaluation_results
from spagent_evaluation import evaluate_tool_config

TOOL_SERVERS = {
    "sam2": "http://localhost:20020",
    "grounding_dino": "http://localhost:20022",
    "depth": "http://localhost:20019",
    "moondream": "http://localhost:20024",
}

COLORMAPS = ["gray", "plasma", "turbo"]


def build_tools(colormap: str):
    # Detection is fixed to image-only (baseline) so only depth varies across conditions.
    return [
        ObjectDetectionTool(use_mock=False, server_url=TOOL_SERVERS["grounding_dino"]),
        SegmentationTool(use_mock=False, server_url=TOOL_SERVERS["sam2"]),
        DepthEstimationTool(use_mock=False, server_url=TOOL_SERVERS["depth"], colormap=colormap),
    ]


def main():
    parser = argparse.ArgumentParser(description='Evaluate depth encoding (colormap) effect on performance')

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
        '--colormap', type=str, default=None,
        choices=COLORMAPS,
        help='Run a single colormap condition. Omit to run all three.'
    )
    parser.add_argument('--enable_data_collection', action='store_true')
    parser.add_argument('--data_output_dir', type=str, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return
    if not os.path.exists(args.image_base_path):
        print(f"Error: Image base path not found at {args.image_base_path}")
        return

    colormaps_to_run = [args.colormap] if args.colormap else COLORMAPS

    all_results = {}
    for colormap in colormaps_to_run:
        config_name = f"depth_{colormap}"
        print(f"\n{'='*60}")
        print(f"Running depth encoding experiment: colormap={colormap}")
        print(f"{'='*60}")

        tools = build_tools(colormap)

        data_collector = None
        if args.enable_data_collection:
            out_dir = args.data_output_dir or (
                f"training_data/{config_name}_{args.model.replace('-', '_')}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            data_collector = DataCollector(output_dir=out_dir, save_images=True, auto_save=True)
            print(f"Data collection enabled: {out_dir}")

        results = evaluate_tool_config(
            config_name=config_name,
            tools=tools,
            data_path=args.data_path,
            image_base_path=args.image_base_path,
            model=args.model,
            max_samples=args.max_samples,
            max_workers=args.max_workers,
            max_iterations=args.max_iterations,
            data_collector=data_collector,
            system_prompt=GENERAL_VISION_SYSTEM_PROMPT,
            continuation_hint=GENERAL_VISION_CONTINUATION_HINT,
            temperature=args.temperature,
            seed=args.seed,
            top_p=args.top_p,
        )
        all_results[config_name] = results

        print(f"\nResults for {config_name}:")
        print_evaluation_results(results)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"spagent_evaluation_results_{args.model.replace('-', '_')}_{args.max_iterations}_{args.task}.json"
    )
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {output_file}")


if __name__ == "__main__":
    main()
