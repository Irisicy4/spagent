"""
Retry failed samples from a previous evaluation run and merge results back.

Usage:
  python retry_failed.py \
    --result_json ./test/CVBench_Qwen2.5-VL-72B-Instruct-det-image0/spagent_evaluation_results_Qwen2.5_VL_72B_Instruct_3_all.json \
    --data_path dataset/cvbench_data_500sample.jsonl \
    --script examples/evaluation/evaluate_dinosam.py \
    --extra_args "--model Qwen2.5-VL-72B-Instruct"
"""
import json
import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from collections import Counter

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_failed_ids(result_json: str) -> list:
    with open(result_json) as f:
        data = json.load(f)
    config = list(data.values())[0]
    return [s["id"] for s in config.get("failed_samples_details", [])]


def filter_dataset(data_path: str, ids: set, out_path: str):
    kept = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample.get("id") in ids:
                kept.append(line)
    with open(out_path, "w") as f:
        f.write("\n".join(kept))
    print(f"Filtered dataset: {len(kept)} samples → {out_path}")
    return len(kept)


def merge_results(original_json: str, retry_json: str, output_json: str):
    with open(original_json) as f:
        orig = json.load(f)
    with open(retry_json) as f:
        retry = json.load(f)

    config_key = list(orig.keys())[0]
    orig_config = orig[config_key]
    retry_config = list(retry.values())[0]

    # Build lookup of retry results by ID
    retry_by_id = {r["id"]: r for r in retry_config.get("detailed_results", [])}

    # Rebuild detailed_results: replace failed entries with retry results
    new_results = []
    for r in orig_config["detailed_results"]:
        if not r["success"] and r["id"] in retry_by_id:
            new_results.append(retry_by_id[r["id"]])
        else:
            new_results.append(r)

    # Recalculate statistics
    successful = [r for r in new_results if r["success"]]
    failed = [r for r in new_results if not r["success"]]
    correct = [r for r in successful if r.get("is_correct")]
    accuracy = len(correct) / len(successful) if successful else 0
    total_time = sum(r.get("inference_time", 0) for r in successful)
    avg_time = total_time / len(successful) if successful else 0

    correct_ids = [r["id"] for r in successful if r.get("is_correct")]
    incorrect_ids = [r["id"] for r in successful if not r.get("is_correct")]

    task_stats = {}
    for r in successful:
        task = r.get("task", "unknown")
        if task not in task_stats:
            task_stats[task] = {"correct": 0, "total": 0}
        task_stats[task]["total"] += 1
        if r.get("is_correct"):
            task_stats[task]["correct"] += 1
    for t in task_stats:
        task_stats[t]["accuracy"] = task_stats[t]["correct"] / task_stats[t]["total"]

    tool_usage = {}
    for r in successful:
        for tool in r.get("used_tools", []):
            tool_usage[tool] = tool_usage.get(tool, 0) + 1

    merged = {
        config_key: {
            **orig_config,
            "total_samples": len(new_results),
            "successful_samples": len(successful),
            "failed_samples": len(failed),
            "overall_accuracy": accuracy,
            "average_inference_time": avg_time,
            "total_inference_time": total_time,
            "task_statistics": task_stats,
            "tool_usage_statistics": tool_usage,
            "failed_samples_details": failed,
            "detailed_results": new_results,
            "correct_question_ids": correct_ids,
            "incorrect_question_ids": incorrect_ids,
        }
    }

    os.makedirs(Path(output_json).parent, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\nMerged results saved to: {output_json}")
    print(f"  Total:      {len(new_results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed:     {len(failed)}")
    print(f"  Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_json", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--script", required=True, help="Evaluation script to re-run")
    parser.add_argument("--extra_args", default="", help="Extra args to pass to the script")
    parser.add_argument("--output_dir", default=None, help="Output dir for retry results (auto if not set)")
    parser.add_argument("--merge_only", action="store_true", help="Skip re-run, only merge existing retry results")
    parser.add_argument("--retry_result_json", default=None, help="Path to existing retry result JSON (for --merge_only)")
    args = parser.parse_args()

    result_json = args.result_json
    orig_dir = Path(result_json).parent
    retry_dir = args.output_dir or str(orig_dir) + "_retry"
    retry_result_json = args.retry_result_json or os.path.join(
        retry_dir,
        Path(result_json).name
    )
    merged_json = os.path.join(str(orig_dir) + "_merged", Path(result_json).name)

    if not args.merge_only:
        failed_ids = load_failed_ids(result_json)
        print(f"Found {len(failed_ids)} failed samples to retry")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp_path = tmp.name
        filter_dataset(args.data_path, set(failed_ids), tmp_path)

        cmd = [
            sys.executable, args.script,
            "--data_path", tmp_path,
            "--output_dir", retry_dir,
        ] + args.extra_args.split()

        print(f"\nRunning: {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)
        os.unlink(tmp_path)

    merge_results(result_json, retry_result_json, merged_json)


if __name__ == "__main__":
    main()
