#!/usr/bin/env python3
"""
Generate a LaTeX table from collected results (results_summary.json).
Overall accuracy + per-task breakdown, grouped by variation (dataset, model, encoding).
"""

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
SUMMARY_JSON = OUTPUT_DIR / "results_summary.json"


def gather_tasks(records: list[dict]) -> list[str]:
    """Collect all task names across records."""
    tasks = set()
    for r in records:
        tasks.update(r.get("task_accuracy") or {})
    return sorted(tasks)


def latex_escape(s: str) -> str:
    replace = {"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#", "$": r"\$"}
    for k, v in replace.items():
        s = s.replace(k, v)
    return s


def pct(x: float | None) -> str:
    if x is None:
        return "---"
    return f"{100 * x:.1f}"


def baseline_overall_by_group(records: list[dict]) -> dict[tuple[str, str], float]:
    """For each (dataset, base_model), get overall_accuracy of 'Image only' encoding (baseline)."""
    baseline = {}
    for r in records:
        if r.get("encoding") == "Image only":
            key = (r["dataset"], r["base_model"])
            baseline[key] = r["overall_accuracy"]
    return baseline


def pct_with_triangle(acc: float, baseline: float | None) -> str:
    """Format overall acc with up/down triangle vs baseline."""
    s = pct(acc)
    if baseline is None or acc == baseline:
        return s
    if acc > baseline:
        return s + r" $\uparrow$"
    return s + r" $\downarrow$"


def build_table(records: list[dict]) -> str:
    tasks = gather_tasks(records)
    n_tasks = len(tasks)
    # Sort by dataset, then model, then encoding for readable grouping
    records = sorted(records, key=lambda r: (r["dataset"], r["base_model"], r["encoding"]))
    baseline = baseline_overall_by_group(records)
    # Header: Dataset | Model | Object detection | Overall | Task1 | Task2 | ...
    cols = "lll|r|" + "r" * n_tasks
    task_headers = " & ".join(latex_escape(t) for t in tasks)
    header = "Dataset & Model & Obj. detection & Overall (\\%) & " + task_headers + r" \\"
    mid = r"\midrule"
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SpAgent evaluation results: overall and per-task accuracy (\%). $\uparrow$ / $\downarrow$ vs.\ image-only baseline.}",
        r"\label{tab:spagent-results}",
        r"\small",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        header,
        mid,
    ]
    for r in records:
        key = (r["dataset"], r["base_model"])
        bl = baseline.get(key) if r.get("encoding") != "Image only" else None
        overall_str = pct_with_triangle(r["overall_accuracy"], bl)
        row = [
            latex_escape(r["dataset"]),
            latex_escape(r["base_model"]),
            latex_escape(r["encoding"]),
            overall_str,
        ]
        for t in tasks:
            row.append(pct((r.get("task_accuracy") or {}).get(t)))
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def build_standalone_document(records: list[dict]) -> str:
    """Build a minimal LaTeX document that compiles on its own."""
    table = build_table(records)
    return r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=1in}

\begin{document}
""" + table + r"""
\end{document}
"""


def main():
    if not SUMMARY_JSON.is_file():
        print("Run collect_results.py first to create", SUMMARY_JSON)
        return
    with open(SUMMARY_JSON, "r") as f:
        records = json.load(f)
    if not records:
        print("No records in summary; run collect_results.py after adding result JSONs to test subdirs.")
        return
    tasks = gather_tasks(records)
    table = build_table(records)
    out_tex = OUTPUT_DIR / "results_table.tex"
    with open(out_tex, "w") as f:
        f.write(table)
    out_doc = OUTPUT_DIR / "results_table_standalone.tex"
    with open(out_doc, "w") as f:
        f.write(build_standalone_document(records))
    print("Wrote", out_tex, "and", out_doc, "with", len(records), "rows and tasks:", tasks)


if __name__ == "__main__":
    main()
