#!/usr/bin/env python3
"""
Collect evaluation results from spagent/test subdirs and build a summary for LaTeX.
Parses subdir names: {Dataset}_{Model}-det-{encoding} -> dataset, base_model, encoding type.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# Paths relative to project root (spagent/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "test"
OUTPUT_DIR = Path(__file__).resolve().parent

ENCODING_LABELS = {
    # Detection encoding conditions
    "image0": "Image only",
    "image-only": "Image only",
    "text-only": "Text only",
    "image0-and-text": "Image and text",
    "image-and-text": "Image and text",
    # Depth colormap conditions
    "depth-gray": "Depth gray",
    "depth-plasma": "Depth plasma",
    "depth-turbo": "Depth turbo",
}


def parse_subdir_name(subdir_name: str) -> tuple[str, str, str] | None:
    """
    Parse subdir names into (dataset, base_model, encoding).

    Supported formats:
      'CVBench_Qwen2.5-VL-32B-Instruct-det-image0'   -> encoding = 'image0'
      'CVBench_Qwen2.5-VL-32B-Instruct-det-text-only' -> encoding = 'text-only'
      'CVBench_Qwen2.5-VL-32B-Instruct-depth-turbo'   -> encoding = 'depth-turbo'
    """
    # Determine the separator token (-det- or -depth-)
    for sep in ("-det-", "-depth-"):
        if sep in subdir_name:
            prefix, enc_suffix = subdir_name.rsplit(sep, 1)
            encoding = ("depth-" if sep == "-depth-" else "") + enc_suffix.strip()
            if "_" in prefix:
                parts = prefix.split("_", 1)
                dataset, model = parts[0], parts[1]
            else:
                dataset, model = prefix, ""
            return (dataset, model, encoding)
    return None


def encoding_display(enc: str) -> str:
    return ENCODING_LABELS.get(enc, enc.replace("-", " ").title())


def find_result_json(subdir: Path) -> Path | None:
    """Find first spagent_evaluation_results*.json in subdir."""
    for f in subdir.glob("spagent_evaluation_results*.json"):
        return f
    return None


def load_result(subdir: Path) -> dict | None:
    jpath = find_result_json(subdir)
    if not jpath or not jpath.is_file():
        return None
    with open(jpath, "r") as f:
        data = json.load(f)
    # Top-level key can be "dinosam" or similar
    for key in ("dinosam", "default"):
        if key in data and isinstance(data[key], dict):
            return data[key]
    if isinstance(data.get("overall_accuracy"), (int, float)):
        return data
    return None


def collect_all_results() -> list[dict]:
    """Scan test subdirs, load results, return list of records."""
    records = []
    if not TEST_DIR.is_dir():
        return records
    for subdir in sorted(TEST_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = parse_subdir_name(subdir.name)
        if not parsed:
            continue
        dataset, model, encoding = parsed
        result = load_result(subdir)
        if result is None:
            continue
        task_stats = result.get("task_statistics") or {}
        overall = result.get("overall_accuracy")
        if overall is None:
            continue
        rec = {
            "subdir": subdir.name,
            "dataset": dataset,
            "base_model": model,
            "encoding_raw": encoding,
            "encoding": encoding_display(encoding),
            "overall_accuracy": float(overall),
            "task_accuracy": {
                k: v.get("accuracy") for k, v in task_stats.items() if isinstance(v, dict) and "accuracy" in v
            },
            "total_samples": result.get("total_samples"),
        }
        records.append(rec)
    return records


def main():
    records = collect_all_results()
    out_json = OUTPUT_DIR / "results_summary.json"
    if not records:
        print("No results found under", TEST_DIR)
        with open(out_json, "w") as f:
            json.dump([], f)
        return []
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print("Collected", len(records), "results ->", out_json)
    return records
