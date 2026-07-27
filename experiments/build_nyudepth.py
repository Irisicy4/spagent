#!/usr/bin/env python3
"""
NYU-Pairs: a second depth benchmark from a different source than DA-2K.

DA-2K is web imagery with human-annotated relative-depth pairs; NYU Depth v2 is
indoor scenes with a *sensor* depth map, so ground truth comes from measurement
rather than annotation and the scene statistics (cluttered rooms, short range)
are unlike DA-2K's outdoor/wild mix. Same MCQ form as DA-2K so the two are
directly comparable: two points are marked on the image, the model picks which
is closer to the camera.

Filters that keep items decidable:
  * both points at least 60 px apart in image space
  * true depths differ by >= 25% of the nearer depth (no near-ties)
  * both depths valid (>0) and within the sensor's reliable range
  * answer side balanced by construction (A/B drawn uniformly)
"""

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "data_slices" / "tailored"
IMG = OUT / "images" / "nyupairs"
IMG.mkdir(parents=True, exist_ok=True)
rng = random.Random(42)
N_ITEMS = 250
os.environ.setdefault("HF_HOME", "/raid/icy/iris/.cache/huggingface")


def main():
    from datasets import load_dataset
    ds = load_dataset("tanganke/nyuv2", split="train", streaming=True)
    rows, seen = [], 0
    for ex in ds:
        if len(rows) >= N_ITEMS:
            break
        seen += 1
        # tanganke/nyuv2 stores image/depth as nested lists, not PIL objects
        # tanganke/nyuv2 ships CHW float images in [0,1] and CHW metric depth
        arr = np.array(ex["image"], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        img = np.ascontiguousarray((arr[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1])
        depth = np.squeeze(np.array(ex["depth"], dtype=np.float32))
        h, w = depth.shape[:2]
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        # sample a decidable point pair
        pair = None
        for _ in range(60):
            y1, x1 = rng.randrange(h), rng.randrange(w)
            y2, x2 = rng.randrange(h), rng.randrange(w)
            d1, d2 = float(depth[y1, x1]), float(depth[y2, x2])
            if min(d1, d2) <= 0.3 or max(d1, d2) > 9.0:
                continue
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 60 ** 2:
                continue
            if abs(d1 - d2) < 0.25 * min(d1, d2):
                continue
            pair = ((y1, x1, d1), (y2, x2, d2))
            break
        if pair is None:
            continue
        (y1, x1, d1), (y2, x2, d2) = pair
        # A is drawn first; assign which physical point is A uniformly
        if rng.random() < 0.5:
            (ya, xa, da), (yb, xb, db) = (y1, x1, d1), (y2, x2, d2)
        else:
            (ya, xa, da), (yb, xb, db) = (y2, x2, d2), (y1, x1, d1)
        r_px = max(6, int(0.008 * max(h, w)))
        for (yy, xx), lab, col in (((ya, xa), "A", (0, 0, 255)), ((yb, xb), "B", (255, 0, 0))):
            cv2.circle(img, (xx, yy), r_px, col, -1)
            cv2.circle(img, (xx, yy), r_px, (255, 255, 255), 2)
            cv2.putText(img, lab, (min(xx + r_px + 2, w - 20), max(yy, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        rel = f"images/nyupairs/nyu_{len(rows):04d}.jpg"
        cv2.imwrite(str(OUT / rel), img)
        ans = "A" if da < db else "B"
        q = ("Two points are marked on the image: point A (red circle) and "
             "point B (blue circle). Which marked point is closer to the "
             "camera?\nSelect from the following choices:\n(A) point A\n(B) point B\n")
        rows.append({
            "id": f"nyupairs_{len(rows)}", "image": [rel], "video": [],
            "conversations": [{"from": "human", "value": q},
                              {"from": "gpt", "value": ans}],
            "task": "Depth", "input_type": "image", "output_type": "MCQ",
            "data_source": "NYUv2-Pairs",
            "others": {"depth_a": round(da, 3), "depth_b": round(db, 3)},
            "subtask": "",
        })
    with open(OUT / "nyupairs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"nyupairs: {len(rows)} items from {seen} scanned; "
          f"answers {dict(Counter(r['conversations'][1]['value'] for r in rows))}")


if __name__ == "__main__":
    main()
