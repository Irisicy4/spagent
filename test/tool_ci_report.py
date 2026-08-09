"""Contract CI report for tool contributions (no-compute).

Runs every catalog tool in mock mode and checks it against the standardized
ToolResult contract (PR #230): build, parameter schema, mock call, contract
payload, render round-trip, box-convention sanity, and failure-path behavior.

Intended as the PR gate for new/changed tools:

    # full catalog, report only (exit 0 unless a *selected* tool fails)
    python test/tool_ci_report.py

    # gate only the tools touched by this PR (CI usage)
    python test/tool_ci_report.py --changed-from <base-sha>

    # gate specific tools
    python test/tool_ci_report.py --tools detection,zoom

A tool whose mock build/call needs unavailable heavy deps is reported as
DEP-SKIP and never fails the gate (mirrors verify_all_tools.py semantics).
Markdown report goes to $GITHUB_STEP_SUMMARY when set, and/or --markdown-out.
Zero/low VRAM; no servers required.
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.disable(logging.WARNING)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "spagent"))
os.chdir(str(REPO))

from tools.catalog import TOOL_CATALOG, build_tools            # noqa: E402
from core.tool_result import ToolResult, validate_payload      # noqa: E402
from core.render import render                                 # noqa: E402

IMG = "assets/dog.jpeg"

# Minimal mock-call kwargs per catalog key. A catalog entry with no row here
# is reported as NO-CALL-KW (a contribution gap), never a KeyError crash.
CALL_KW = {
    "depth":        dict(image_path=IMG),
    "segmentation": dict(image_path=IMG),
    "detection":    dict(image_path=IMG, text_prompt="dog"),
    "zoom":         dict(image_path=IMG, text_prompt="dog"),
    "localize":     dict(image_path=IMG, text_prompt="dog"),
    "supervision":  dict(image_path=IMG, task="image_det"),
    "yoloe":        dict(image_path=IMG, task="image", class_names=["dog"]),
    "yolo26":       dict(image_path=IMG),
    "qwenvl":       dict(image_path=IMG, text_prompt="dog"),
    "moondream":    dict(image_path=IMG, task="point", object_name="dog"),
    "molmo2":       dict(image_path=IMG, prompt="Point to the dog"),
    "pi3":          dict(image_path=[IMG], azimuth_angle=30, elevation_angle=10),
    "pi3x":         dict(image_path=[IMG], azimuth_angle=30, elevation_angle=10),
    "vggt":         dict(image_path=[IMG], azimuth_angle=30, elevation_angle=10),
    "mapanything":  dict(image_path=[IMG], azimuth_angle=30, elevation_angle=10),
    "orient_anything_v2": dict(image_path=IMG, object_category="dog"),
    "sana":         dict(prompt="a dog"),
    "veo":          dict(prompt="a dog running"),
    "sora":         dict(prompt="a dog running"),
    "wan":          dict(prompt="a dog running"),
    "vace":         dict(image_path=IMG, prompt="dog walks"),
    "flowseek":     dict(image1_path=IMG, image2_path=IMG,
                         output_path="outputs/vflow.png"),
    "paddleocr_vl": dict(image_path=IMG),
    "wilddet3d":    dict(image_path=IMG, prompt_text="dog"),
}

CHECKS = ["build", "schema", "call", "toolresult", "contract", "render", "boxes", "failpath"]

PASS, FAIL, SKIP, NA = "✅", "❌", "⏭ dep", "—"


def _is_dep_error(exc: Exception) -> bool:
    return isinstance(exc, (ImportError, ModuleNotFoundError, FileNotFoundError, OSError))


def _bad_image_kwargs(kw):
    """Clone kwargs with image path(s) pointing at a nonexistent file."""
    out = {}
    for k, v in kw.items():
        if "image" in k and "path" in k:
            out[k] = ["/nonexistent/ci_missing.jpg"] if isinstance(v, list) else "/nonexistent/ci_missing.jpg"
        else:
            out[k] = v
    return out


def check_tool(entry):
    """Run all checks for one catalog entry. Returns (results dict, notes list)."""
    r = {c: NA for c in CHECKS}
    notes = []
    key = entry.key

    # build
    try:
        tools, errs = build_tools([key], use_mock=True)
    except Exception as e:
        r["build"] = SKIP if _is_dep_error(e) else FAIL
        notes.append(f"build: {type(e).__name__}: {e}"[:100])
        return r, notes
    if not tools:
        r["build"] = SKIP if any("No module" in e or "import" in e.lower() for e in errs) else FAIL
        notes.append(f"build: {'; '.join(errs)}"[:100])
        return r, notes
    tool = tools[0]
    r["build"] = PASS

    # schema
    try:
        p = tool.parameters
        r["schema"] = PASS if isinstance(p, dict) and p.get("type") and "properties" in p else FAIL
        if r["schema"] == FAIL:
            notes.append("schema: parameters missing type/properties")
    except Exception as e:
        r["schema"] = FAIL
        notes.append(f"schema: {type(e).__name__}: {e}"[:80])

    # call kwargs available?
    if key not in CALL_KW:
        notes.append("no CALL_KW entry — add one for this tool")
        r["call"] = FAIL
        return r, notes

    # mock call
    try:
        res = tool.call(**CALL_KW[key])
    except Exception as e:
        r["call"] = SKIP if _is_dep_error(e) else FAIL
        notes.append(f"call: {type(e).__name__}: {e}"[:100])
        return r, notes
    if not isinstance(res, dict):
        r["call"] = FAIL
        notes.append(f"call returned {type(res).__name__}, not a dict/ToolResult")
        return r, notes
    if not res.get("success"):
        r["call"] = SKIP if "not found" in str(res.get("error", "")).lower() else FAIL
        notes.append(f"call failed: {str(res.get('error'))[:80]}")
        return r, notes
    r["call"] = PASS

    # ToolResult migration (informational for legacy dicts, but new tools must pass)
    r["toolresult"] = PASS if isinstance(res, ToolResult) else FAIL
    if r["toolresult"] == FAIL:
        notes.append("returns plain dict — new tools must return ToolResult")

    # contract
    category = res.get("category") or entry.category
    try:
        ok, unmet = validate_payload(res, category)
        r["contract"] = PASS if ok else FAIL
        if not ok:
            notes.append(f"contract unmet: {unmet}")
    except Exception as e:
        r["contract"] = FAIL
        notes.append(f"contract: {type(e).__name__}: {e}"[:80])

    # render round-trip, default + all presets
    try:
        for preset in (None, {"preset": "all"}):
            out = render(res, config=preset, tool_name=getattr(tool, "name", None))
            if not (out.text and out.text.strip()):
                r["render"] = FAIL
                notes.append(f"render produced empty text (preset={preset})")
                break
        else:
            r["render"] = PASS
    except Exception as e:
        r["render"] = FAIL
        notes.append(f"render: {type(e).__name__}: {e}"[:80])

    # box-convention sanity (detection-style payloads only)
    boxes = res.get("boxes")
    payload = getattr(res, "payload", None)
    if boxes and payload is not None and hasattr(payload, "to_xyxy_pixel"):
        try:
            px = payload.to_xyxy_pixel()
            bad = [b for b in px if not (b[2] > b[0] and b[3] > b[1])]
            r["boxes"] = FAIL if bad else PASS
            if bad:
                notes.append(f"degenerate pixel boxes (x2<=x1 or y2<=y1): {bad[:2]}")
        except ValueError as e:
            r["boxes"] = FAIL
            notes.append(f"boxes: {e}")
    elif boxes:
        r["boxes"] = FAIL
        notes.append("boxes present but payload lacks to_xyxy_pixel()")

    # failure path: bad image must yield success=False, never raise
    kw = CALL_KW[key]
    if any("image" in k and "path" in k for k in kw):
        try:
            bad_res = tool.call(**_bad_image_kwargs(kw))
            if isinstance(bad_res, dict) and not bad_res.get("success"):
                r["failpath"] = PASS
            else:
                r["failpath"] = FAIL
                notes.append("missing-image call did not return success=False")
        except Exception as e:
            r["failpath"] = FAIL
            notes.append(f"missing-image call raised {type(e).__name__}")

    return r, notes


def changed_tool_keys(base):
    """Map files changed since `base` to catalog keys. core/ changes select all."""
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        capture_output=True, text=True, cwd=str(REPO), check=True,
    ).stdout.splitlines()
    if any(f.startswith("spagent/core/") for f in diff):
        return [e.key for e in TOOL_CATALOG], []
    stems = {Path(f).stem for f in diff if f.startswith("spagent/tools/") and f.endswith(".py")}
    keys = [e.key for e in TOOL_CATALOG
            if e.cls.__module__.rsplit(".", 1)[-1] in stems]
    orphans = stems - {e.cls.__module__.rsplit(".", 1)[-1] for e in TOOL_CATALOG} - {"catalog", "__init__"}
    return keys, sorted(orphans)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", help="comma-separated catalog keys to gate")
    ap.add_argument("--changed-from", help="git base sha: gate tools changed since it")
    ap.add_argument("--markdown-out", help="also write the markdown report here")
    args = ap.parse_args()

    gated = None
    orphans = []
    if args.tools:
        gated = [k.strip() for k in args.tools.split(",") if k.strip()]
        unknown = [k for k in gated if k not in {e.key for e in TOOL_CATALOG}]
        if unknown:
            print(f"❌ unknown tool key(s): {unknown}")
            return 1
    elif args.changed_from:
        gated, orphans = changed_tool_keys(args.changed_from)

    rows, gate_failures = [], []
    for entry in TOOL_CATALOG:
        results, notes = check_tool(entry)
        rows.append((entry.key, entry.category, results, notes))
        is_gated = gated is None or entry.key in gated
        if is_gated and any(v == FAIL for v in results.values()):
            gate_failures.append(entry.key)

    # ---- report ----
    md = ["# Tool contract report (no-compute, mock mode)", ""]
    if gated is not None:
        md.append(f"**Gated tools** (changed in this PR): `{', '.join(gated) or 'none'}`  ")
        md.append("Other rows are informational.")
        md.append("")
    if orphans and not args.tools:
        md.append(f"⚠️ changed tool file(s) with **no catalog entry**: `{', '.join(orphans)}` "
                  "— new tools must be registered in `spagent/tools/catalog.py`.")
        md.append("")
        gate_failures.extend(f"unregistered:{o}" for o in orphans)
    md.append("| tool | category | " + " | ".join(CHECKS) + " | notes |")
    md.append("|" + "---|" * (len(CHECKS) + 3))
    for key, cat, results, notes in rows:
        gate_mark = "**" if (gated is not None and key in gated) else ""
        md.append(f"| {gate_mark}{key}{gate_mark} | {cat} | "
                  + " | ".join(results[c] for c in CHECKS)
                  + " | " + "; ".join(notes)[:160] + " |")
    md.append("")
    if gate_failures:
        md.append(f"## ❌ gate failed: {', '.join(gate_failures)}")
    else:
        md.append("## ✅ gate passed"
                  + ("" if gated is None else f" ({len(gated)} gated tool(s))"))
    report = "\n".join(md)

    print(report)
    for path in filter(None, [args.markdown_out, os.environ.get("GITHUB_STEP_SUMMARY")]):
        with open(path, "a") as f:
            f.write(report + "\n")

    return 1 if gate_failures else 0


if __name__ == "__main__":
    sys.exit(main())
