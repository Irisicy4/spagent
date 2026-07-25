# Encoding Experiments (william's snapshot, reorganized)

Experiments behind the paper's Stage-I/Stage-II encoding study (Table 4 /
Fig. 4 / Fig. 7). Backend: GroundingDINO (+SAM/depth); controller served via
vLLM (OpenAI-compatible, `OPENAI_BASE_URL`). Dataset: 500-sample CV-Bench /
BLINK jsonl (see `dataset_william/`).

## Experiment tags

| Tag | Question | Variants |
|-----|----------|----------|
| **E1-det-encoding** | How should the detection tool's output be encoded for the controller? | `det-image-only` (annotated image, no text) · `det-image-text` (image + JSON text; framework default) · `det-text-only` (JSON xyxy text, no image) |
| **E2-depth-colormap** | Which colormap for the depth tool's rendered map? | `gray` · `plasma` · `turbo` |
| **E3-best-combo** | Do the per-tool winners stack? | `det-text-only` + `turbo` (planned in run.sh, commented out — never run in this snapshot) |
| **E0-legacy** | Early single runs pre-dating the sweep | gpt-4o, qwen-72b |

Encoding definitions & exact tool return shapes: `docs/detection_encoding.md`.
Tool implementations: image-only → `spagent/tools/detection_tool.py`;
image+text → `spagent/external_experts/GroundingDINO/grounding_dino_image_and_text_tool.py`;
text-only → `spagent/external_experts/GroundingDINO_text_only/`.

## Results on disk (final runs, accuracy over n=500)

| Result dir (under `results/`) | Acc | Status / note |
|---|---|---|
| E1/cvbench_qwen72b/det-image-only | **0.7125** | paper Table 4: 71.08 (paper notes 204/500 partial; this merged chain reached 500) |
| E1/cvbench_qwen72b/det-image-text | **0.7120** | paper ref 71.20 — exact match |
| E1/cvbench_qwen72b/det-text-only  | **0.7234** | paper 72.29 — Stage-I winner confirmed |
| E1/cvbench_qwen32b/det-*          | 0.0000 ×3 | **BROKEN runs** (parsing/scoring failure — do not use) |
| E1/blink_qwen3b/det-image-only    | 0.3700 | |
| E1/blink_qwen3b/det-text-only     | 0.3775 | text-only also wins on BLINK-3B |
| E1/blink_qwen72b/det-image-only   | 0.5588 | no other encodings run on BLINK-72B |
| E2/cvbench_qwen72b/{gray,plasma,turbo} | 0.7318 / 0.7220 / **0.7328** | turbo best → chosen for E3 |
| E2/cvbench_qwen32b/*              | 0.0000 ×3 | **BROKEN** (same failure as E1 32B) |

Missing entirely (empty dirs in the original, removed): BLINK-3B det-image-text,
CVBench-3B det-image-text, CVBench-3B det-text-only.

## Provenance / renaming map

Original dirs lived flat under `test/` with the scheme
`{BENCH}_{MODEL}[-det-...|-depth-...][_retry|_retry2|_merged|_merged_merged]`.
Long 72B runs crashed midway and were resumed: `base` + `_retry` (+`_retry2`)
were stitched by merge scripts into `_merged` (+`_merged_merged`). **The
deepest merge is the final result** and was moved to the tagged location
above; all intermediates are preserved untouched in `results/_partials/`.
Run/server logs are in `logs/`. Root-level CSV sweeps
(`depth_{gray,plasma,turbo}.csv`, `dinosam.csv` + `*_detailed_interactions.json`)
are per-run summaries → `results/E*/summaries/`.

NOTE: `analysze_result/collect_results.py` still scans the OLD `test/` layout
by name pattern; run it against `results/_partials` + tagged dirs only after
adapting its glob, or use the accuracies table above (extracted from each
run's `*_all.json`).

## Reproducing / rerunning

- `run_all.sh` — clean, tagged rewrite of the original `run.sh`+`run_remaining.sh`
  (kept verbatim in repo root for provenance).
- `rerun_ci.sh` — N-repeat rerun of the reported Table-4 numbers
  (E1 cvbench_qwen72b × 3 encodings) for run-to-run confidence intervals.
  Outputs land in `results/E1-det-encoding/cvbench_qwen72b/ci_runs/<variant>/repeatK/`.
- Prereqs for either: a vLLM server for the controller on `$OPENAI_BASE_URL`
  and the tool expert servers (GroundingDINO :20022 minimum; depth :20019 for
  E2/best-combo). See script headers for launch commands.
