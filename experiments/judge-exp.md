# Judge-Transfer Experiments — Design & Status

Companion doc to the Vision-Judge paper's **Stage II** (SpAgent as the
tool-using harness; CV-Bench workload). This repo snapshot IS the Stage-II
kitchen: the paper's Table 4 / depth-colormap numbers were produced here.
This doc tracks (a) confidence-interval reruns of every reported number,
(b) fidelity findings vs the paper, (c) new experiments extending Stage II
to the segmentation axis and to interventions the paper could not run.

---

## 1. Fidelity: does this repo match the paper?

| Paper element | This repo | Verdict |
|---|---|---|
| Table 4 SpAgent (71.08 / 71.20 / 72.29) | disk finals 71.25 / 71.20 / 72.34 | ✅ match; ±0.1 = merge-state (paper froze image-only at 204/500; retry chains later reached 500) |
| Detection encodings (image-only / image+text / text-xyxy) | three forked tools, spec in `docs/detection_encoding.md` | ✅ exact |
| Depth base=plasma, ablations turbo/gray (paper Table 6) | Stage-II disk: turbo .7328 > gray .7318 > plasma .7220 | ⚠️ plasma (Stage-I favourite) is LAST at agent runtime — Stage-I→II transfer questionable for depth; CI reruns will adjudicate |
| Tool stack (paper Table 7: GDINO+SAM2+DepthAV2+Moondream) | CV-Bench scripts wire 3 tools (no moondream); verified 0 moondream calls in real runs | ⚠️ harmless text/stack mismatch, note for camera-ready |
| 32B comparison (tex table 73.2 / 75.4) | on-disk 32B artifacts are a dead batch (500/500 prediction=None) | ❌ unbacked → rerunning (batch C1) |
| BLINK-3B table (37.0 / 37.8) | disk artifacts intact & consistent | ✅ backed; CI rerun unblocked by operator-provided moondream key |

## 2. CI reruns of reported numbers (3 repeats each)

Protocol: 500-item CV-Bench, `temperature=0`, per-run **sandbox** (private
`outputs/`; dataset+checkpoints symlinked) to prevent artifact races; all
runs concurrent; results in `results/<E>/cvbench_<model>/ci_runs/<variant>/repeatK/`;
aggregation = across-repeat t-CI + per-item bootstrap (`ci_aggregate.py --ci-runs`).

| # | Config | Reported | Rerun status | Interim mean [95% CI] |
|---|---|---|---|---|
| A1 | 72B det image-only | 71.08 (204/500) | running | — |
| A2 | 72B det image+text | 71.20 (ref) | **done ×3** | **71.10 [68.47, 73.74]** (.7174/.7169/.6988) |
| A3 | 72B det text-only | **72.29 (+1.09)** | **done ×3** | **70.45 [68.48, 72.41]** (.7100/.7080/.6954) |
| B1-3 | 72B depth gray/plasma/turbo | .7318/.7220/.7328 | running | — |
| B4 | 72B best-combo (text+turbo) | never run | running | — |
| C1 | 32B det ×3 encodings | 73.2 / — / 75.4 (tex) | running | — |
| C3 | BLINK-3B det image-only / text-only | 37.0 / 37.8 | running (four-tool stack) | — |
| A3' | 72B det **text-only-FIXED** | n/a (new arm) | running ×3 | — |
| C3' | BLINK-3B det **text-only-FIXED** | n/a (new arm) | running ×3 | — |

### ⚠️ Interim headline finding (A2 vs A3)
The paper's +1.09pp text-only advantage **does not reproduce** across 3
fresh repeats: every text-only repeat (max 71.00) lands below the original
72.34, and the rerun means invert the ordering (70.45 vs 71.10) with heavily
overlapping CIs. Run-to-run noise (~±1pp even at temperature 0 — vLLM
batching nondeterminism + tool stochasticity) is comparable to the claimed
delta. Final judgment awaits A1 + the **paired per-item analysis** (§5),
which is far more sensitive than unpaired CIs.
Failure contamination ruled out: 0–2 failed items per completed run.

### 🚨 Fidelity finding #7: the text-only arm was broken by construction
The agent loop injects tool text into prompts from exactly one key —
`result["description"]` — and `GroundingDINOTextOnlyTool` never returned it.
Its bbox JSON was computed and logged but NEVER shown to the controller
(verified in live continuation prompts). File dates show the original
published text-only runs (incl. CV-Bench 72.34) used this same path, so the
paper's SpAgent "text-only" condition is better read as "no detection
info". Fix: commit `aaa1d9d` (standalone, pushed for review) adds the
description channel. The 3 broken-arm CV-Bench repeats are retired to
`ci_runs/_broken_arm_det-text-only/` (kept as the faithful replication of
the published condition); canonical `det-text-only` slots now hold the
FIXED arm. BLINK text-only reruns restarted with the fix.

## 3. New experiment N1 — segmentation-encoding Stage II

Motivation: the paper sweeps seg encodings only in Stage I (judging) and
explicitly predicts "mask-driven QA benchmarks" would flip the seg axes'
zero/negative correlation (§5.2, Limitations). Nobody has run Stage II for
segmentation. We do.

### N1 encodings (from paper Table 6 seg axes / Fig 8 previews)

| ID | Encoding | Definition | Stage-I signal |
|---|---|---|---|
| S1 | overlay-base | α=0.5 overlay + text labels, standard palette | base config |
| S2 | opacity-100 | α=1.0 (image fully occluded under masks) | mid/low |
| S3 | color-by-instance | one colour per instance | counting-friendly |
| S4 | contour-only | boundaries only, minimal occlusion | mid |
| S5 | polygon-text | text-only polygon coords, no image | ties/beats overlay on instance-seg judging |
| S6 | mask-only *(negative control)* | mask without original image | known Stage-I failure mode |

Implementation: output-formatter forks of `SegmentationTool` (same
methodology as the detection forks), SAM2 backend unchanged.

**Tool isolation (design rule):** each N1 run registers ONLY the
segmentation tool (the encoding variant under test) — no detection/depth in
the stack. Rationale: in the existing E1 runs the agent freely chose among
det+seg+depth (measured usage: depth 217 calls vs detection 135 per 500
items), so encoding effects are diluted by items that never invoke the tool
being varied. A single-tool stack (a) attributes accuracy deltas to the
encoding cleanly, (b) raises the tool-invocation rate on mask-relevant
items, and (c) mirrors the RL pipeline's deliberate single-tool setup.
Trade-off (accepted): absolute accuracies won't be comparable to full-stack
runs — the comparison is WITHIN the encoding axis, which is the point.

### N1 benchmarks (mask-driven QA)

| Benchmark | Why segmentation-driven | Source / prep |
|---|---|---|
| CV-Bench Count slice | counting = count instance masks; the paper's own flip-prediction target | already on disk (subset of the 500) |
| TallyQA (complex) | compositional counting needs masks + filtering, boxes insufficient | HF pull, 500-sample |
| RefCOCOg→MCQ | candidate regions A–D rendered under each seg encoding; question = read the mask encoding | construct: HF refcoco + SAM2 masks from GT boxes |
| ReasonSeg (phase 2) | agent must produce masks (gIoU metric) | optional, different metric |

### N1 run matrix & hypotheses

- 6 encodings × 3 benchmarks × Qwen2.5-VL-72B × 3 repeats = **54 runs**
  (≈2 nights at current fleet throughput; +54 optional on 32B).
- **H1**: seg-encoding choice has a first-order effect on mask-driven tasks
  (flip of the paper's null result on CV-Bench-count/relation).
- **H2**: polygon-text ≥ overlay for instance-style tasks (Stage-I transfer).
- **H3**: opacity-100 and mask-only sink (occlusion cost at agent level).

## 4. New experiment N2 — visibility ablation (I1) on SpAgent

The paper ran intervention I1 (visibility of tool output) only on HuggingGPT
because "SpAgent always invokes its tool stack." The standardization branch's
**render module** (`feat/tool-standardization`, PR #230) makes it possible on
SpAgent without touching orchestration:

| Arm | render_config | Paper analogue |
|---|---|---|
| full | default projection (text + images) | "+ tool, visible" |
| text-summary | fields=minimal, no images attached | "summarised" |
| hidden | empty projection (tool runs, controller sees nothing) | no-tool proxy |

3 arms × CV-Bench × 72B × 3 repeats = 9 runs, executed on the **main repo**
(not this snapshot). Completes Table 4's missing SpAgent I1 column and
doubles as a real-world validation of the render architecture.

## 5. Analyses (post-sweep, zero GPU)

| Analysis | Method | Purpose |
|---|---|---|
| Paired encoding comparison | McNemar + paired bootstrap on shared item IDs across variants | sensitive test of the +1.09pp claim; final verdict on A3 |
| Tool-conditioned pairing | restrict the paired test to items where the varied tool was actually invoked (from `used_tools`) | undilutes encoding effects in the full-stack E1/E2 runs — the mixed stack means ~27% of items invoke detection |
| Depth plasma-vs-turbo significance | same paired machinery on B runs | adjudicate the Stage-I/Stage-II colormap discrepancy |
| Crowdedness stratification | bucket per-item accuracy by detected-object count | agent-level replica of paper Fig 6 |
| Timeout audit | failed-item overlap across repeats | rule out systematic item-level exclusion bias |

## 6. Infra & bookkeeping

- Controllers: vLLM — 72B TP2 (GPUs 0-1, :8001), 32B (GPU 2, :8002).
  Tools (GPU 7): GDINO :20022, SAM2 :20020, Depth :20019; Moondream server
  :20024 launches with `MOONDREAM_API_KEY` provided by the operator at
  runtime — the key is never written to disk or committed.
- Backfill: incomplete/timeout items are re-run with the repo's own
  `retry_failed.py` → merge flow (same as the original `_retry/_merged`
  chains), keeping methodology identical to the original runs.
- All drivers resumable; sandboxes under `experiments/.parallel_sandboxes/`
  (untracked); results force-added (`ci_runs/`) since the snapshot
  `.gitignore` ignores `*.json`.

## 7. Open decisions

| Question | Default |
|---|---|
| Include image+text in the 32B CI (tex table omits it)? | included (mirrors 72B) |
| BLINK CI: rerun 72B image-only too (55.88, no tabled partner)? | yes, cheap once moondream server is up |
| N1 on 32B as well? | after 72B results justify it |
| ReasonSeg (produce-mask setting)? | phase 2 only |
