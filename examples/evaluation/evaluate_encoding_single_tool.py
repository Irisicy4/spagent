"""
Single-tool encoding experiments (judge-exp.md N1 + depth single-tool arm).

Registers EXACTLY ONE tool (or none for the baseline) so accuracy deltas are
attributable to the tool-output encoding, per the tool-isolation design rule.

  --tool none                            no-tool baseline
  --tool depth  --encoding gray|plasma|turbo
  --tool refseg --encoding overlay|instance|polygon|maskonly
  --tool semseg --encoding overlay|instance|polygon|maskonly
"""

import os
import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from spagent.core.prompts import GENERAL_VISION_SYSTEM_PROMPT, GENERAL_VISION_CONTINUATION_HINT
from spagent.tools import DepthEstimationTool
from spagent.tools.encoding_seg_tools import (
    ReferringSegmentationTool, SemanticSegmentationTool, SEG_ENCODINGS,
)
from spagent.utils.utils import print_evaluation_results
from spagent_evaluation import evaluate_tool_config

TOOL_SERVERS = {
    "sam2": "http://localhost:20020",
    "grounding_dino": "http://localhost:20022",
    "depth": "http://localhost:20019",
}

DEPTH_ENCODINGS = ["gray", "plasma", "turbo"]


def build_tools(tool: str, encoding: str):
    if tool == "none":
        return []
    if tool == "depth":
        assert encoding in DEPTH_ENCODINGS, f"depth encoding must be one of {DEPTH_ENCODINGS}"
        return [DepthEstimationTool(use_mock=False, server_url=TOOL_SERVERS["depth"],
                                    colormap=encoding)]
    if tool == "refseg":
        assert encoding in SEG_ENCODINGS, f"seg encoding must be one of {SEG_ENCODINGS}"
        return [ReferringSegmentationTool(encoding=encoding,
                                          gdino_url=TOOL_SERVERS["grounding_dino"],
                                          sam2_url=TOOL_SERVERS["sam2"])]
    if tool == "semseg":
        assert encoding in SEG_ENCODINGS, f"seg encoding must be one of {SEG_ENCODINGS}"
        return [SemanticSegmentationTool(encoding=encoding,
                                         gdino_url=TOOL_SERVERS["grounding_dino"],
                                         sam2_url=TOOL_SERVERS["sam2"])]
    raise ValueError(f"unknown tool '{tool}'")


def main():
    parser = argparse.ArgumentParser(description="Single-tool encoding evaluation")
    parser.add_argument('--tool', type=str, required=True,
                        choices=['none', 'depth', 'refseg', 'semseg'])
    parser.add_argument('--encoding', type=str, default='none')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_workers', type=int, default=4)
    parser.add_argument('--image_base_path', type=str, default='dataset')
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--max_iterations', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        sys.exit(1)
    if not os.path.exists(args.image_base_path):
        print(f"Error: Image base path not found at {args.image_base_path}")
        sys.exit(1)

    config_name = f"{args.tool}_{args.encoding}" if args.tool != 'none' else 'no_tool'
    tools = build_tools(args.tool, args.encoding)
    print(f"Config: {config_name} | tools registered: {[t.name for t in tools]}")

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

    all_results = {config_name: results}
    print_evaluation_results(results)

    os.makedirs(args.output_dir, exist_ok=True)
    import json
    model_tag = args.model.replace('-', '_').replace('.', '_')
    out = os.path.join(
        args.output_dir,
        f"spagent_evaluation_results_{model_tag}_{args.max_iterations}_all.json")
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
