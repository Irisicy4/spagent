#!/usr/bin/env python3
"""
Build a LaTeX comparison table for exactly two result files:
  - CVBench Qwen2.5-VL-32B det-image0 (Image only)
  - CVBench Qwen2.5-VL-32B det-text-only (Text only)
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent

FILE_A = PROJECT_ROOT / "test/CVBench_Qwen2.5-VL-32B-Instruct-det-image0/spagent_evaluation_results_qwen2.5_vl_32b_3_all.json"
FILE_B = PROJECT_ROOT / "test/CVBench_Qwen2.5-VL-32B-Instruct-det-text-only/spagent_evaluation_results_qwen2.5_vl_32b_3_all.json"


def load_result(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    for key in ("dinosam", "default"):
        if key in data and isinstance(data[key], dict):
            return data[key]
    if isinstance(data.get("overall_accuracy"), (int, float)):
        return data
    return None


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def latex_escape(s: str) -> str:
    for k, v in [("_", r"\_"), ("&", r"\&"), ("%", r"\%")]:
        s = s.replace(k, v)
    return s


def main():
    a = load_result(FILE_A)
    b = load_result(FILE_B)
    if a is None or b is None:
        print("Missing file:", FILE_A if a is None else FILE_B)
        return

    tasks_a = a.get("task_statistics") or {}
    tasks_b = b.get("task_statistics") or {}
    task_names = sorted(set(tasks_a.keys()) | set(tasks_b.keys()))

    def acc(d: dict, task: str | None) -> float | None:
        if task is None:
            return d.get("overall_accuracy")
        st = (d.get("task_statistics") or {}).get(task)
        return st.get("accuracy") if isinstance(st, dict) else None

    # Rows: Overall, then each task
    rows = [("Overall", None)] + [(t, t) for t in task_names]
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{CVBench Qwen2.5-VL-32B: Image-only vs.\ text-only object detection encoding. Accuracy (\%); $\uparrow$/$\downarrow$ = change vs.\ image-only.}",
        r"\label{tab:cvbench-32b-compare}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        "Metric & Image only & Text only \\\\",
        r"\midrule",
    ]
    for label, task_key in rows:
        va = acc(a, task_key)
        vb = acc(b, task_key)
        delta = (vb - va) if (va is not None and vb is not None) else None
        c_a = pct(va) if va is not None else "---"
        if vb is None:
            c_b = "---"
        else:
            c_b = pct(vb)
            if delta is not None and delta != 0:
                c_b += r" $\uparrow$" if delta > 0 else r" $\downarrow$"
        lines.append(f"{latex_escape(label)} & {c_a} & {c_b} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    out_tex = OUTPUT_DIR / "comparison_cvbench_32b.tex"
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print("Wrote", out_tex)


if __name__ == "__main__":
    main()
