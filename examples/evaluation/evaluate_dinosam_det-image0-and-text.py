import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from tqdm import tqdm
import cv2
import numpy as np
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from spagent import SPAgent
from spagent.core import DataCollector
from spagent.core.prompts import GENERAL_VISION_SYSTEM_PROMPT, GENERAL_VISION_CONTINUATION_HINT
from spagent.models import GPTModel, QwenModel
from spagent.external_experts.GroundingDINO import GroundingDINOImageAndTextTool
from spagent.tools import DepthEstimationTool, SegmentationTool
from spagent.utils.utils import (
    load_json_data,
    extract_question_and_answer,
    normalize_answer,
    print_evaluation_results,
    validate_sample_paths,
    save_result_to_csv
)
from spagent_evaluation import evaluate_tool_config, evaluate_single_sample
from datetime import datetime

# Define server URLs
TOOL_SERVERS = {
    "sam2": "http://localhost:20020",
    "grounding_dino": "http://localhost:20022",
    "depth": "http://localhost:20019",
    "moondream": "http://localhost:20024",
}

TOOL_CONFIGS = {
    "dinosam": [
        GroundingDINOImageAndTextTool(server_url=TOOL_SERVERS["grounding_dino"]),
        SegmentationTool(use_mock=False, server_url=TOOL_SERVERS["sam2"]),
        DepthEstimationTool(use_mock=False, server_url=TOOL_SERVERS["depth"]),
    ]
}


def main():
    parser = argparse.ArgumentParser(description='Evaluate dinosam with image+text detection encoding')

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
    parser.add_argument('--enable_data_collection', action='store_true')
    parser.add_argument('--data_output_dir', type=str, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return

    if not os.path.exists(args.image_base_path):
        print(f"Error: Image base path not found at {args.image_base_path}")
        return

    all_results = {}
    for config_name, tools in TOOL_CONFIGS.items():
        data_collector = None
        if args.enable_data_collection:
            if args.data_output_dir is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                data_output_dir = f"training_data/{config_name}_{args.model.replace('-', '_')}_{timestamp}"
            else:
                data_output_dir = args.data_output_dir

            data_collector = DataCollector(
                output_dir=data_output_dir,
                save_images=True,
                auto_save=True
            )
            print(f"Data collection enabled: {data_output_dir}")

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
