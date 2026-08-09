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


def _img_dims(path=IMG):
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (W, H)


def measure(entry, res, notes, flow_shift=None):
    """Category-specific artifact measurements. Appends to notes, returns ok."""
    import numpy as np
    cat = res.get("category") or entry.category
    W, H = _img_dims()
    ok = True

    if cat == "detection":
        boxes = res.get("boxes") or []
        notes.append(f"{len(boxes)} box(es)")  # 0 detections is a valid finding
        payload = getattr(res, "payload", None)
        if boxes:
            conf = res.get("confidence")
            if conf is not None and len(conf) != len(boxes):
                ok = False
                notes.append(f"confidence misaligned: {len(conf)} vs {len(boxes)} boxes")
            if payload is not None and hasattr(payload, "to_xyxy_pixel"):
                px = payload.to_xyxy_pixel()
                bad = [b for b in px
                       if not (0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H)]
                if bad:
                    ok = False
                    notes.append(f"degenerate/out-of-bounds pixel boxes: {bad[:2]} (img {W}x{H})")
                else:
                    notes.append(f"pixel boxes sane, e.g. {px[0]}")

    elif cat == "segmentation":
        mp = res.get("mask_path")
        if mp and os.path.exists(mp):
            from PIL import Image
            m = np.array(Image.open(mp).convert("L"))
            if m.shape != (H, W):
                ok = False
                notes.append(f"mask dims {m.shape} != image ({H},{W})")
            else:
                notes.append(f"mask {m.shape}, coverage {(m > 127).mean():.3f}")
        elif not res.get("masks"):
            ok = False
            notes.append("no mask_path and no masks")

    elif cat == "depth":
        shape = res.get("shape")
        if list(shape or [])[:2] != [H, W]:
            ok = False
            notes.append(f"depth shape {shape} != image ({H},{W},3)")
        else:
            notes.append(f"depth shape {shape}")
        out = res.get("output_path")
        if out and not os.path.exists(out):
            ok = False
            notes.append(f"output_path missing on disk: {out}")

    elif cat == "3d_reconstruction":
        raw = res.get("result") or {}
        if raw.get("cached"):
            ok = False
            notes.append("served from stale cache — delete outputs/<tool>_*png and rerun")
        pc = raw.get("points_count")
        if not (isinstance(pc, int) and pc > 0):
            ok = False
            notes.append(f"points_count not a positive int: {pc!r}")
        else:
            notes.append(f"{pc:,} points")

    elif cat == "orientation":
        raw = res.get("result") or {}
        az, el = raw.get("azimuth"), raw.get("elevation")
        if az is None or not (0 <= az <= 360) or el is None or not (-90 <= el <= 90):
            ok = False
            notes.append(f"angles out of range: az={az} el={el}")
        else:
            notes.append(f"az={az} el={el} rot={raw.get('rotation')}")

    elif cat == "point_grounding":
        pts = res.get("points") or []
        bad = [p for p in pts
               if not (0 <= p.get("x", -1) <= W and 0 <= p.get("y", -1) <= H)]
        if bad:
            ok = False
            notes.append(f"out-of-bounds points: {bad[:2]}")
        else:
            notes.append(f"{len(pts)} point(s) in bounds")

    elif cat == "optical_flow":
        fp = res.get("flow_path")
        if not (fp and os.path.exists(fp)):
            ok = False
            notes.append("raw flow .npy missing")
        else:
            flow = np.load(fp)
            if flow.ndim != 3 or flow.shape[-1] != 2:
                ok = False
                notes.append(f"flow shape {flow.shape} not (H,W,2)")
            else:
                notes.append(f"flow {flow.shape}")
                if flow_shift:
                    dx = float(flow[:, 20:, 0].mean())
                    if abs(dx - flow_shift) > 0.5:
                        ok = False
                    notes.append(f"recovered dx={dx:.2f} (target {flow_shift})")

    elif cat == "ocr":
        if "text" not in res:
            ok = False
            notes.append("no text key")  # empty text itself is a valid finding
        else:
            notes.append(f"text len {len(res.get('text') or '')}")

    elif cat == "image_generation":
        p = res.get("output_path")
        if not (p and os.path.exists(p) and os.path.getsize(p) > 0):
            ok = False
            notes.append(f"generated image missing/empty: {p}")
        else:
            from PIL import Image
            with Image.open(p) as im:
                notes.append(f"image {im.size}, {os.path.getsize(p):,} bytes")

    else:
        notes.append(f"(no artifact measurement for category {cat})")

    return ok


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
    notes = []
    kw = dict(CALL_KW.get(entry.key) or {})
    if not kw:
        return FAIL, ["no CALL_KW entry"]

    effective_url = url or DEFAULT_SERVER_URLS.get(entry.key)
    if effective_url and not _probe(effective_url):
        return FAIL, [f"backend unreachable: {effective_url} — start the server first "
                      "(a dead backend can masquerade as a zero-finding success)"]
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
        return FAIL, [f"call raised {type(e).__name__}: {e}"[:140]]
    if not isinstance(res, dict) or not res.get("success"):
        return FAIL, [f"call failed: {str((res or {}).get('error'))[:120]}"]

    okc, unmet = validate_payload(res, res.get("category") or entry.category)
    if not okc:
        notes.append(f"contract unmet: {unmet}")
    ok = measure(entry, res, notes, flow_shift=flow_shift)
    return (PASS if (ok and okc) else FAIL), notes


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
          "| tool | server | status | measurements |", "|---|---|---|---|"]
    failed = []
    for k in keys:
        e = entries[k]
        url = urls.get(k) or DEFAULT_SERVER_URLS.get(k, "")
        status, notes = smoke_tool(e, urls.get(k), args.flow_shift)
        if status == FAIL:
            failed.append(k)
        md.append(f"| {k} | {url or '(local)'} | {status} | " + "; ".join(notes)[:200] + " |")
    md.append("")
    md.append(f"## {'❌ failed: ' + ', '.join(failed) if failed else '✅ all requested tools passed'}")
    report = "\n".join(md)

    print(report)
    for path in filter(None, [args.markdown_out, os.environ.get("GITHUB_STEP_SUMMARY")]):
        with open(path, "a") as f:
            f.write(report + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
