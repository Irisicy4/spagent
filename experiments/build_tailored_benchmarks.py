#!/usr/bin/env python3
"""
Build tool-tailored benchmark slices beyond CV-Bench (task #10):

  vstar.jsonl         detection-tailored : V*Bench (craigwu/vstar_bench), 191
                      small-object attribute/position MCQs — detection text
                      should shine here.
  tallyqa_count.jsonl seg-tailored       : TallyQA mirror sample (300), counting
                      questions MCQ-ified (4 numeric options around the truth).
                      Mirror lacks the simple/complex flag -> tagged "sample".
  da2k_depth.jsonl    depth-tailored     : DA-2K (depth-anything), 300 point-pair
                      "which marked point is closer" MCQs; the two query points
                      are rendered onto a copy of the image as labeled circles.

Output: experiments/data_slices/tailored/{*.jsonl, images/...}
Items follow the repo's CV-Bench JSON schema (image paths relative to the
tailored/ dir; pass --image_base_path experiments/data_slices/tailored).
Deterministic: seed 42 everywhere.
"""

import io
import json
import os
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data_slices" / "tailored"
IMG = OUT / "images"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", "/raid/icy/iris/.cache/huggingface")
rng = random.Random(42)


def item(id_, img_rel, question, answer_letter, task, source):
    return {
        "id": id_, "image": [img_rel], "video": [],
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer_letter},
        ],
        "task": task, "input_type": "image", "output_type": "MCQ",
        "data_source": source, "others": {}, "subtask": "",
    }


def build_vstar():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    repo = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    d = load_dataset("craigwu/vstar_bench", split="test")
    (IMG / "vstar").mkdir(exist_ok=True)
    rows = []
    for i, ex in enumerate(d):
        src = Path(repo) / ex["image"]
        if not src.exists():
            continue
        rel = f"images/vstar/{Path(ex['image']).name}"
        dst = OUT / rel
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        rows.append(item(f"vstar_{ex['category']}_{i}", rel, ex["text"].strip(),
                         ex["label"].strip(), ex["category"], "VStarBench"))
    with open(OUT / "vstar.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"vstar: {len(rows)} items")


def build_tallyqa(n=300):
    from datasets import load_dataset
    d = load_dataset("nimapourjafar/mm_tallyqa", split="train", streaming=True)
    (IMG / "tallyqa").mkdir(exist_ok=True)
    rows = []
    seen = 0
    for ex in d:
        seen += 1
        if seen > 40000 or len(rows) >= n:
            break
        try:
            data = ex["data"]
            q = next(t["data"] for t in data if t["role"] == "user" and t["modality"] == "text")
            a = next(t["data"] for t in data if t["role"] == "assistant")
            ans = int(str(a).strip().split()[0].rstrip("."))
        except (StopIteration, ValueError, IndexError):
            continue
        if not (0 <= ans <= 8) or not ex["images"]:
            continue
        # reservoir-free deterministic thinning: keep every 40th eligible row
        if seen % 40:
            continue
        img_bytes = ex["images"][0]["bytes"]
        rel = f"images/tallyqa/tqa_{len(rows):04d}.jpg"
        (OUT / rel).write_bytes(img_bytes)
        # 4 numeric options around the truth
        lo = max(0, ans - rng.randint(0, 3))
        opts = [lo, lo + 1, lo + 2, lo + 3]
        if ans not in opts:
            opts[0] = ans
            opts = sorted(set(opts))[:4]
            while len(opts) < 4:
                opts.append(max(opts) + 1)
        letters = "ABCD"
        ans_letter = letters[opts.index(ans)]
        qtext = (q.split("\n")[0].strip() + "\nSelect from the following choices:\n" +
                 "\n".join(f"({letters[j]}) {opts[j]}" for j in range(4)) + "\n")
        rows.append(item(f"tallyqa_{len(rows)}", rel, qtext, ans_letter,
                         "Count", "TallyQA-sample"))
    with open(OUT / "tallyqa_count.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"tallyqa: {len(rows)} items (scanned {seen})")


def build_da2k(n=300):
    import cv2
    import numpy as np
    from huggingface_hub import hf_hub_download
    zp = hf_hub_download("depth-anything/DA-2K", "DA-2K.zip", repo_type="dataset")
    zf = zipfile.ZipFile(zp)
    ann_name = next(x for x in zf.namelist() if x.endswith(".json"))
    ann = json.loads(zf.read(ann_name))
    (IMG / "da2k").mkdir(exist_ok=True)
    keys = sorted(ann.keys())
    rng.shuffle(keys)
    rows = []
    for k in keys:
        if len(rows) >= n:
            break
        entries = ann[k] if isinstance(ann[k], list) else [ann[k]]
        e = entries[0]
        try:
            p1, p2, closer = e["point1"], e["point2"], e["closer_point"]
        except (KeyError, TypeError):
            continue
        try:
            raw = zf.read(k if k in zf.namelist() else next(
                x for x in zf.namelist() if x.endswith(k)))
        except (KeyError, StopIteration):
            continue
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        # DA-2K points are [y, x]
        pts = {"A": (int(p1[1]), int(p1[0])), "B": (int(p2[1]), int(p2[0]))}
        r_px = max(6, int(0.008 * max(h, w)))
        for lab, (x, y) in pts.items():
            col = (0, 0, 255) if lab == "A" else (255, 0, 0)
            cv2.circle(img, (x, y), r_px, col, -1)
            cv2.circle(img, (x, y), r_px, (255, 255, 255), 2)
            cv2.putText(img, lab, (min(x + r_px + 2, w - 20), max(y, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        rel = f"images/da2k/da_{len(rows):04d}.jpg"
        cv2.imwrite(str(OUT / rel), img)
        ans = "A" if closer == "point1" else "B"
        q = ("Two points are marked on the image: point A (red circle) and "
             "point B (blue circle). Which marked point is closer to the "
             "camera?\nSelect from the following choices:\n(A) point A\n(B) point B\n")
        rows.append(item(f"da2k_{len(rows)}", rel, q, ans, "DA2K", "DA-2K"))
    with open(OUT / "da2k_depth.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"da2k: {len(rows)} items")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "vstar"):
        build_vstar()
    if which in ("all", "tallyqa"):
        build_tallyqa()
    if which in ("all", "da2k"):
        build_da2k()
