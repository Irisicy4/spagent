# SpAgent result analysis

Scripts in this folder collect evaluation results from each subdir of `spagent/test` and produce a LaTeX table.

## Subdir naming

Each test subdir is expected to follow:

- `{Dataset}_{Model}-det-{encoding}`

Where:

- **Dataset**: e.g. `CVBench`, `BLINK`
- **Model**: e.g. `Qwen2.5-VL-32B-Instruct`
- **Encoding** (object detection):
  - `image0` → **Image only** (original)
  - `text-only` → **Text only**
  - `image0-and-text` → **Image and text**

## Usage

```bash
./run.sh
```

Or step by step:

```bash
python3 collect_results.py   # writes results_summary.json
python3 generate_latex_table.py   # reads results_summary.json, writes results_table.tex
```

## Outputs

- **results_summary.json** – All parsed results (overall + per-task accuracy).
- **results_table.tex** – LaTeX table snippet (use `\usepackage{booktabs}` and `\usepackage{graphicx}` if using `\resizebox`).
- **results_table_standalone.tex** – Minimal compilable document (e.g. `pdflatex results_table_standalone.tex`).

Only subdirs that contain at least one `spagent_evaluation_results*.json` file are included.

### Two-run comparison (CVBench 32B)

To build a table comparing only the image-only and text-only runs for CVBench Qwen2.5-VL-32B:

```bash
python3 compare_two_results.py
```

This writes **comparison_cvbench_32b.tex** with rows: Overall, then each task (Count, Depth, Distance, Relation); columns: Image only, Text only; $\uparrow$/$\downarrow$ on the text-only column indicate change vs. image-only.

For BLINK Qwen2.5-VL-3B (default run vs. text-only object detection):

```bash
python3 compare_blink_3b.py
```

This writes **comparison_blink_3b.tex** (Default vs. Text only, with $\uparrow$/$\downarrow$ vs. default).
