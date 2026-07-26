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
| Depth colormaps: Stage-I ONLY in the paper (Table 6/Fig 4; §5.3/Table 4 is detection-only) | Stage-II disk singles (UNPUBLISHED): turbo .7328 > gray .7318 > plasma .7220 | ✅ resolved: the on-disk singles were flukes — ×3 reruns give gray=plasma (.758) ≫ turbo (.699), so our E2/E4 runs are the FIRST Stage-II depth-encoding evidence and they MATCH Stage-I plasma-over-turbo |
| Tool stack (paper Table 7: GDINO+SAM2+DepthAV2+Moondream) | CV-Bench scripts wire 3 tools (no moondream); verified 0 moondream calls in real runs | ⚠️ harmless text/stack mismatch, note for camera-ready |
| 32B comparison (tex table 73.2 / 75.4) | on-disk 32B artifacts are a dead batch (500/500 prediction=None) | ❌ unbacked → rerunning (batch C1) |
| BLINK-3B table (37.0 / 37.8) | disk artifacts intact & consistent | ✅ backed; CI rerun unblocked by operator-provided moondream key |

## 2. CI reruns of reported numbers (3 repeats each)

Protocol: 500-item CV-Bench, `temperature=0`, per-run **sandbox** (private
`outputs/`; dataset+checkpoints symlinked) to prevent artifact races; all
runs concurrent; results in `results/<E>/cvbench_<model>/ci_runs/<variant>/repeatK/`;
aggregation = across-repeat t-CI + per-item bootstrap (`ci_aggregate.py --ci-runs`).

| # | Config | Reported | Rerun status | Mean [95% t-CI] |
|---|---|---|---|---|
| A1 | 72B det image-only | 71.08 (204/500) | **done ×3** (backfilled ~470-474/500 ok) | **69.89** (.7004/.7006/.6957) |
| A2 | 72B det image+text | 71.20 (ref) | **done ×3** | **71.10 [68.47, 73.74]** (.7174/.7169/.6988) |
| A3 | 72B det text-only (broken, as published) | **72.29 (+1.09)** | **done ×3** (retired) | **70.45 [68.48, 72.41]** |
| B1-3 | 72B depth gray/plasma/turbo | .7318/.7220/.7328 | **done ×3 each** | **gray 75.79 [73.10,78.48] / plasma 75.79 [74.77,76.81] / turbo 69.90 [67.88,71.93]** |
| B4 | 72B combo-v1 (text+turbo) | never run | **done ×3** | **71.45 [68.88, 74.02]** |
| B5 | 72B **combo-v2 (FIXED text + plasma)** | n/a (new arm) | **done ×3** | **77.13 [76.37, 77.89]** (.7720/.7680/.7740) — best config measured |
| C1 | 32B det img-only / img+text / text-only(broken) | 73.2 / — / 75.4 (tex) | **done ×3 each** | **69.80 (deterministic) / 71.07 [68.62,73.52] / 71.53 [70.39,72.68]** |
| C1' | 32B det **text-only-FIXED** | n/a (new arm) | **done ×3** | **69.80 [67.52, 72.08]** (.690/.696/.708) — BELOW broken arm |
| C3 | BLINK-3B det image-only / text-only(broken) | 37.0 / 37.8 | **done ×3 each** | **37.46 [31.04,43.89] / 38.81 [36.15,41.47]** |
| C3' | BLINK-3B det **text-only-FIXED** | n/a (new arm) | **done ×3** | **38.81 [36.15, 41.47]** |
| A3' | 72B det **text-only-FIXED** | n/a (new arm) | relaunched ×3 on fresh :8003 (~07:55, prior attempt lost to a degraded vLLM server) | — |

### FINAL: detection text-format four-way (72B, CV-Bench-500, ×3 each)

| Text format | Mean | vs broken (paired) |
|---|---|---|
| cxcywh mislabeled xyxy (as published, post-description-fix) | 69.96 | −0.5pp, p=1.0 |
| broken (no info injected; published condition) | 70.45 | ref |
| normalized xyxy (corrected, 4dbb1b1) | 70.65 | +0.1pp, p=0.78 |
| **pixel xyxy + image WxH (operator-suggested)** | **71.96** | +1.5pp, p=0.16 |
| image+text-xyxy (image channel + corrected text) | **72.04** | best overall |

Monotone in coordinate-format quality/pretraining match (Qwen2.5-VL grounds
in absolute pixels). Pixel text-only ≈ image+text — correctly-formatted text
alone nearly recovers the annotated image's full value; normalized fractions
throw that value away. No single step is significant at n=3, but the
ordering + the 20:12 / 31:20 discordant patterns are consistent.
Counting is the opposite: semseg polygon-NORMALIZED 64.33 > polygon-PIXEL
62.09 (6:1 discordants) — count tasks consume instance structure, not
coordinates; localization tasks consume coordinates. Encoding must match
what the task extracts.

### Emerging story (updated 07/26 ~08:00)
1. **combo-v2 (fixed detection text + plasma depth) = 77.1%** — +5.7pp over the
   original combo-v1 and +4.8pp over the paper's best Table-4 number, with the
   tightest CI of any arm. Both stage-I-informed choices (plasma, real text)
   contribute.
2. At 32B the FIXED text arm (69.8) lands BELOW the broken no-info arm (71.5):
   raw bbox text can HURT a mid-size controller — consistent with the dilution
   +distraction hypothesis; awaiting the paired per-item analysis.
3. Depth: gray=plasma >> turbo, reversing the original single-run ordering and
   vindicating Stage-I's plasma pick; turbo (the published default) is worst.

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

## 2b. E4 — single-tool encoding sweep (RUNNING since 07/26 ~08:00)

Operator-prioritized instantiation of N1 plus a depth single-tool arm: test the
output ENCODING of (a) depth, (b) referring segmentation, (c) semantic
segmentation — each with exactly ONE tool registered, on the CV-Bench task
slice that tool should provably help. Tool benefit itself is proven by
matched no-tool baselines on the same slices.

| Axis | Tool (most confident of category) | Slice (from the same 500-sample) | Encodings |
|---|---|---|---|
| baseline | none | all 3 slices | — |
| depth | DepthAnythingV2 (single tool) | Depth+Distance (216) | gray / plasma / turbo |
| refseg | GDINO→SAM2 referring pipeline (`ReferringSegmentationTool`) | Relation (120) | overlay / instance / polygon / maskonly |
| semseg | GDINO→SAM2 open-vocab class pipeline (`SemanticSegmentationTool`) | Count (164) | overlay / instance / polygon / maskonly |

- Encodings implement paper Table-6 seg axes: S1 overlay-base, S3
  color-by-instance, S5 polygon-text (text-only, no image), S6 mask-only
  (negative control, black background).
- New code: `spagent/tools/encoding_seg_tools.py`,
  `examples/evaluation/evaluate_encoding_single_tool.py`,
  `experiments/run_encoding_sweep.sh`; slices in `experiments/data_slices/`.
- Verified before launch: description reaches controller (polygon arm's model
  answer references "outlined as a polygon"); images attach via output_path;
  renders visually QC'd. Found+fixed a real plumbing bug: the GDINO :20022
  server returns normalized CXCYWH under an xyxy-named key — naive reading
  produces garbage SAM2 prompts (mirror-32B seg arms restarted post-fix).
- Fleet: 72B ×3 repeats — :8001 base+depth+refseg, :8003 semseg; 32B mirror
  ×1 repeat (deterministic) on :8002. Results under
  `results/E4-encoding-single-tool/<model>/ci_runs/<arm>/repeatK/`.
- Hypotheses: H1 tool>none on its slice; H2 polygon-text ≥ overlay
  (Stage-I transfer); H3 maskonly sinks (occlusion control); depth single-tool
  should replicate gray/plasma >> turbo without stack dilution.

### E4 results (first 3 repeats complete ~11:00; repeats 4-6 running for power)

72B (×3 repeats, mean acc; paired-vs-baseline verdicts from `paired_analysis.py`):

| Arm | Depth+Distance (216) | Count (164) | Relation (120) |
|---|---|---|---|
| no-tool | **.8349** | .5833 | .8306 |
| depth gray / plasma / turbo | .8272 / .8102 / **.6914** | | |
| semseg overlay / instance / polygon / maskonly | | .6301 / .6382 / **.6443** / .6382 | |
| refseg overlay / instance / polygon / maskonly | | | .8533 / **.8620** / .8118 / .8442 |

32B mirror (deterministic, ×1):

| Arm | Depth+Distance | Count | Relation |
|---|---|---|---|
| no-tool | .8102 | **.6707** | **.9250** |
| depth gray / plasma / turbo | **.8148** / .8102 / **.6435** | | |
| semseg (best/worst) | | .6402 / .6159 | |
| refseg (best/worst) | | | .8667 / .8500 |

Headline findings:
1. **Turbo depth colormap is actively destructive at both scales in
   single-tool isolation**: 72B −14.4pp vs no-tool (McNemar p=0.0002),
   −13.6pp vs gray (p=0.0003); 32B −16.7pp. Gray/plasma are ~neutral.
   A bad encoding doesn't merely fail to help — it poisons the controller,
   even though QC samples show the model reading the legend correctly.
2. **Tool benefit is controller-capacity-dependent.** At 72B, semseg lifts
   Count by +4.7..+6.1pp (all four encodings; polygon-text +6.1 is the only
   CI excluding 0, p=0.029-0.076 range). At 32B EVERY single-tool arm is
   neutral-to-harmful on its slice (semseg −3..−5.5 on Count, refseg −6..−7.5
   on Relation, depth ≤+0.5).
3. **Encoding×task crossover**: polygon-text is the BEST semseg encoding for
   counting (+6.1) but the WORST refseg encoding for relation (−2.8) —
   symbolic text suits identity/count queries; spatial queries want image
   overlays (instance-color best, +1.9 ns). H2 holds for counting only;
   H3 (maskonly sinks) rejected for counting (identity, not appearance).
4. E1 close-out with the FIXED text arm (69.96 ×3 on fresh server): all four
   72B detection encodings sit in one 70-71 noise band; fixed-text vs
   broken-text p=1.0 → at 72B the detection text content is irrelevant on
   CV-Bench; the paper's +1.09 text-only advantage is conclusively artifact.

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

## 3.2 New experiment N3 — coordinate-format realization of "text xyxy" (DONE)

**Motivation (the §5.3 repro nuance).** The paper's Result B claims text-only
xyxy is the Stage-I success case and the strongest Stage-II run. On SpAgent
the published arm transmitted NOTHING (description bug), and the repo's
intended realization is *normalized [0,1]* xyxy (docs/detection_encoding.md)
— a convention the paper never states. N3 asks: which realization of "text
xyxy" actually carries the claimed value? Answer: **pixel coordinates + image
size**. This is the reproduction arm for Result B going forward.

| Realization of "text xyxy" (72B, ×3) | Count | Depth | Distance | Relation | Overall |
|---|---|---|---|---|---|
| broken / no info (as published) | 61.2 | 67.3 | 72.1 | 83.9 | 70.45 |
| cxcywh mislabeled (server passthrough) | — | — | — | — | 69.96 |
| normalized xyxy (repo's intended spec) | 62.4 | 66.7 | 69.8 | 85.3 | 70.65 |
| **pixel xyxy + image W×H (N3)** | 62.6 | 68.0 | **72.7** | **86.9** | **71.96** |
| image+text, corrected text (context) | 63.0 | 68.0 | 72.1 | 86.7 | 72.04 |

**Discussion points (for the paper's Discussion section):**
1. The §5.3 SpAgent ordering ("text-only xyxy strongest") is only realized
   when the text uses PIXEL coordinates with the image size stated —
   matching Qwen2.5-VL's absolute-coordinate grounding pretraining. The
   normalized variant is statistically indistinguishable from sending
   nothing (p=0.78 vs no-info), and mildly *harms* the 3D tasks
   (Depth 66.7 / Distance 69.8, both below the no-info arm).
2. Pixel text-only ≈ image+text on every task slice: correctly-formatted
   text fully substitutes for the annotated image. "Visual redundancy"
   (the framework default) buys nothing that well-formatted text doesn't.
3. Gains concentrate exactly where coordinates carry the answer
   (Distance +2.9, Relation +1.7 over normalized); Count is flat — and the
   seg counterpart inverts (normalized polygons BEAT pixel polygons for
   counting, 64.3 vs 62.1): **the coordinate convention must match what the
   task extracts, not a global best**. Stage I cannot see this axis because
   judging quality is insensitive to it; Stage II is where it binds.
4. Effects are ~2× diluted by tool usage (detection fires on ~57% of
   items); tool-conditioned paired deltas point the same way (+0.8pp
   pixel-vs-normalized on the 285 det-invoked items).

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
