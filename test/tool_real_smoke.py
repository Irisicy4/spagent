"""Real-backend smoke test for tools (with-compute).

Calls tools with use_mock=False against live backends and measures the
returned ARTIFACT, not the label — encoding the checks from the PR #230
verification campaign (degenerate boxes, cache stubs, mask/depth dims,
flow recovery). Requires the relevant servers to be up (see
docs/Tool/TOOL_USING.md for launch commands); zero servers are started here.

    # smoke one tool against an explicit server
    python test/tool_real_smoke.py --tool detection --url detection=http://localhost:20122

    # several tools, catalog-default URLs
    python test/tool_real_smoke.py --tool depth,segmentation,detection

    # optical flow with a synthetic-shift recovery check
    python test/tool_real_smoke.py --tool flowseek --flow-shift 12

Exit code is nonzero if any requested tool fails. Markdown report goes to
$GITHUB_STEP_SUMMARY when set, and/or --markdown-out.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

logging.disable(logging.WARNING)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "spagent"))
sys.path.insert(0, str(REPO / "test"))
os.chdir(str(REPO))

from tools.catalog import TOOL_CATALOG, DEFAULT_SERVER_URLS, build_tools  # noqa: E402
from core.tool_result import validate_payload                            # noqa: E402
from tool_ci_report import CALL_KW                                       # noqa: E402

IMG = "assets/dog.jpeg"
PASS, FAIL = "✅", "❌"

# What the smoke measures per category, and why. Rendered as a legend and in
# failure details. These encode the PR #230 verification-campaign rules:
# measure the artifact, never the label.
MEASURE_INFO = {
    "detection": "boxes project to in-bounds, non-degenerate pixel xyxy "
                 "(x2>x1, y2>y1, inside the image) and confidence aligns 1:1 "
                 "with boxes. 0 detections is a VALID finding. Guards the "
                 "normalized-cxcywh-under-an-xyxy-key bug (symptom: y2<y1).",
    "segmentation": "the mask file exists on disk and its dims equal the input "
                    "image dims; coverage is reported. A success without any "
                    "mask carrier fails.",
    "depth": "reported `shape` equals the input image dims and the output "
             "visualization exists on disk.",
    "3d_reconstruction": "`points_count` is a real positive integer and the "
                         "result is NOT served from the visualization cache — "
                         "the cached path once fabricated points_count=50000 "
                         "and contaminated a verification table.",
    "orientation": "azimuth/elevation are inside their valid ranges "
                   "(0–360 / −90–90).",
    "point_grounding": "every returned point lies inside the image bounds.",
    "optical_flow": "the raw flow .npy exists, loads, and is (H, W, 2); with "
                    "--flow-shift N, a synthetic N-px shift must be recovered "
                    "within 0.5 px.",
    "ocr": "the `text` key exists (empty text is a VALID finding — blank pages "
           "are real).",
    "image_generation": "the generated file exists, is non-empty, and decodes "
                        "as an image.",
}


def _img_dims(path=IMG):
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (W, H)


def measure(entry, res, notes, problems, flow_shift=None):
    """Category-specific artifact measurements.

    Observations go to ``notes``; each violation goes to ``problems`` with
    expected-vs-observed wording. Returns True when no problems were added.
    """
    import numpy as np
    cat = res.get("category") or entry.category
    W, H = _img_dims()
    before = len(problems)

    if cat == "detection":
        boxes = res.get("boxes") or []
        notes.append(f"{len(boxes)} box(es)")  # 0 detections is a valid finding
        payload = getattr(res, "payload", None)
        if boxes:
            conf = res.get("confidence")
            if conf is not None and len(conf) != len(boxes):
                problems.append(f"confidence misaligned: expected one score per box "
                                f"({len(boxes)}), observed {len(conf)} — a consumer "
                                "would attach scores to the wrong objects")
            if payload is not None and hasattr(payload, "to_xyxy_pixel"):
                px = payload.to_xyxy_pixel()
                bad = [b for b in px
                       if not (0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H)]
                if bad:
                    problems.append(f"expected pixel boxes with x2>x1, y2>y1 inside "
                                    f"{W}x{H}; observed {bad[:2]} — the declared "
                                    "box_format likely mislabels the actual convention")
                else:
                    notes.append(f"pixel boxes sane, e.g. {px[0]}")

    elif cat == "segmentation":
        mp = res.get("mask_path")
        if mp and os.path.exists(mp):
            from PIL import Image
            m = np.array(Image.open(mp).convert("L"))
            if m.shape != (H, W):
                problems.append(f"expected mask dims ({H},{W}) matching the input "
                                f"image; observed {m.shape}")
            else:
                notes.append(f"mask {m.shape}, coverage {(m > 127).mean():.3f}")
        elif not res.get("masks"):
            problems.append("success=True but NO mask artifact: neither mask_path "
                            "(file) nor masks (arrays) present — a segmentation "
                            "result with nothing segmented-shaped in it")

    elif cat == "depth":
        shape = res.get("shape")
        if list(shape or [])[:2] != [H, W]:
            problems.append(f"expected depth shape [{H}, {W}, ...] matching the "
                            f"input image; observed {shape}")
        else:
            notes.append(f"depth shape {shape}")
        out = res.get("output_path")
        if out and not os.path.exists(out):
            problems.append(f"output_path advertised but missing on disk: {out}")

    elif cat == "3d_reconstruction":
        raw = res.get("result") or {}
        if raw.get("cached"):
            problems.append("served from the visualization cache, not the backend "
                            "— delete outputs/<tool>_*png and rerun for a real "
                            "measurement")
        pc = raw.get("points_count")
        if not (isinstance(pc, int) and pc > 0):
            problems.append(f"expected a positive integer points_count from a real "
                            f"reconstruction; observed {pc!r}")
        else:
            notes.append(f"{pc:,} points")

    elif cat == "orientation":
        raw = res.get("result") or {}
        az, el = raw.get("azimuth"), raw.get("elevation")
        if az is None or not (0 <= az <= 360) or el is None or not (-90 <= el <= 90):
            problems.append(f"expected azimuth in [0,360] and elevation in "
                            f"[-90,90]; observed az={az} el={el}")
        else:
            notes.append(f"az={az} el={el} rot={raw.get('rotation')}")

    elif cat == "point_grounding":
        pts = res.get("points") or []
        bad = [p for p in pts
               if not (0 <= p.get("x", -1) <= W and 0 <= p.get("y", -1) <= H)]
        if bad:
            problems.append(f"expected points inside {W}x{H}; observed {bad[:2]}")
        else:
            notes.append(f"{len(pts)} point(s) in bounds")

    elif cat == "optical_flow":
        fp = res.get("flow_path")
        if not (fp and os.path.exists(fp)):
            problems.append("expected raw per-pixel flow saved as .npy "
                            "(flow_path); observed none — only the colorized "
                            "visualization exists, which cannot be consumed "
                            "numerically")
        else:
            flow = np.load(fp)
            if flow.ndim != 3 or flow.shape[-1] != 2:
                problems.append(f"expected flow array (H, W, 2); observed {flow.shape}")
            else:
                notes.append(f"flow {flow.shape}")
                if flow_shift:
                    dx = float(flow[:, 20:, 0].mean())
                    if abs(dx - flow_shift) > 0.5:
                        problems.append(f"synthetic {flow_shift}px shift NOT "
                                        f"recovered: expected mean dx≈{flow_shift}, "
                                        f"observed {dx:.2f}")
                    else:
                        notes.append(f"recovered dx={dx:.2f} (target {flow_shift})")

    elif cat == "ocr":
        if "text" not in res:
            problems.append("expected a `text` key (empty string is fine — blank "
                            "pages are valid findings); observed no text key at all")
        else:
            notes.append(f"text len {len(res.get('text') or '')}")

    elif cat == "image_generation":
        p = res.get("output_path")
        if not (p and os.path.exists(p) and os.path.getsize(p) > 0):
            problems.append(f"expected a non-empty generated image on disk; "
                            f"observed output_path={p!r} "
                            f"(exists={bool(p and os.path.exists(p))})")
        else:
            from PIL import Image
            with Image.open(p) as im:
                notes.append(f"image {im.size}, {os.path.getsize(p):,} bytes")

    else:
        notes.append(f"(no artifact measurement defined for category {cat!r})")

    return len(problems) == before


def _probe(url):
    """True if something answers HTTP at url (any status). Dead-backend guard:
    several tools convert connection failures into success=True zero-finding
    results, so reachability must be established out-of-band."""
    import requests
    for path in ("/health", "/"):
        try:
            requests.get(url.rstrip("/") + path, timeout=5)
            return True
        except requests.exceptions.RequestException:
            continue
    return False


def smoke_tool(entry, url, flow_shift):
    notes, problems = [], []
    kw = dict(CALL_KW.get(entry.key) or {})
    if not kw:
        return FAIL, notes, ["no CALL_KW entry in test/tool_ci_report.py — the "
                             "smoke cannot invoke this tool; add a minimal row"]

    effective_url = url or DEFAULT_SERVER_URLS.get(entry.key)
    if effective_url and not _probe(effective_url):
        return FAIL, notes, [
            f"backend unreachable at {effective_url} — start the server first. "
            "This is checked out-of-band because several tools convert a dead "
            "backend into a success=True zero-finding result, which would make "
            "this smoke pass vacuously"]
    tools, errs = build_tools([entry.key], use_mock=False,
                              overrides={entry.key: {"server_url": url}} if url else None)
    if not tools:
        return FAIL, [f"build: {'; '.join(errs)}"[:120]]
    tool = tools[0]

    if flow_shift and entry.key == "flowseek":
        import numpy as np
        from PIL import Image
        a = np.array(Image.open(IMG).convert("RGB"))
        b = np.roll(a, flow_shift, axis=1)
        os.makedirs("outputs", exist_ok=True)
        Image.fromarray(a).save("outputs/smoke_f1.png")
        Image.fromarray(b).save("outputs/smoke_f2.png")
        kw = dict(image1_path="outputs/smoke_f1.png", image2_path="outputs/smoke_f2.png",
                  output_path="outputs/smoke_flow.png")

    try:
        res = tool.call(**kw)
    except Exception as e:
        return FAIL, notes, [f"real call RAISED {type(e).__name__}: {e}"[:200]
                             + " — tools must return error dicts, not raise"]
    if not isinstance(res, dict) or not res.get("success"):
        return FAIL, notes, ["real call returned success=False: "
                             + str((res or {}).get("error"))[:200]]

    cat = res.get("category") or entry.category
    okc, unmet = validate_payload(res, cat)
    if not okc:
        problems.append(f"category contract `{cat}` unmet: the standardized "
                        f"result must carry ONE OF {unmet}; none present")
    measure(entry, res, notes, problems, flow_shift=flow_shift)
    return (FAIL if problems else PASS), notes, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tool", required=True, help="comma-separated catalog keys")
    ap.add_argument("--url", action="append", default=[],
                    help="key=http://host:port server override (repeatable)")
    ap.add_argument("--flow-shift", type=int,
                    help="flowseek only: synthetic px shift to recover")
    ap.add_argument("--markdown-out")
    args = ap.parse_args()

    urls = dict(u.split("=", 1) for u in args.url)
    keys = [k.strip() for k in args.tool.split(",") if k.strip()]
    entries = {e.key: e for e in TOOL_CATALOG}
    unknown = [k for k in keys if k not in entries]
    if unknown:
        print(f"❌ unknown tool key(s): {unknown}")
        return 1

    md = ["# Tool real-backend smoke report (with-compute)", "",
          "Live backends, artifact-measuring checks: what is verified per "
          "category is listed at the bottom.", "",
          "| tool | server | status | measurements |", "|---|---|---|---|"]
    failed, details = [], []
    for k in keys:
        e = entries[k]
        url = urls.get(k) or DEFAULT_SERVER_URLS.get(k, "")
        status, notes, problems = smoke_tool(e, urls.get(k), args.flow_shift)
        if status == FAIL:
            failed.append(k)
            details.append((k, e.category, problems))
        cells = "; ".join(notes + [f"✗ {p}" for p in problems])
        md.append(f"| {k} | {url or '(local)'} | {status} | {cells[:250]} |")
    md.append("")

    if details:
        md.append("## Failure details")
        md.append("")
        for k, cat, problems in details:
            md.append(f"### `{k}` ({cat})")
            doc = MEASURE_INFO.get(cat)
            if doc:
                md.append(f"*This category's smoke verifies that:* {doc}")
            md.append("")
            for p in problems:
                md.append(f"- ✗ {p}")
            md.append("")

    md.append("<details><summary>What the smoke verifies per category</summary>")
    md.append("")
    for cat, doc in MEASURE_INFO.items():
        md.append(f"- **{cat}** — {doc}")
    md.append("")
    md.append("All categories additionally require: backend reachable "
              "(probed out-of-band), real call succeeds without raising, and "
              "the category contract is satisfied.")
    md.append("</details>")
    md.append("")
    md.append(f"## {'❌ failed: ' + ', '.join(failed) + ' — see Failure details above' if failed else '✅ all requested tools passed'}")
    report = "\n".join(md)

    print(report)
    for path in filter(None, [args.markdown_out, os.environ.get("GITHUB_STEP_SUMMARY")]):
        with open(path, "a") as f:
            f.write(report + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
