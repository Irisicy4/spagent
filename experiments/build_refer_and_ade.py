#!/usr/bin/env python3
"""
Two more mask-driven benchmarks, one per remaining segmentation subtype.

COCO-Refer (REFERRING axis, 250 items)
    Four candidate instances are marked A-D with coloured boxes on the image.
    The question names one of them by a property that requires separating it
    from its neighbours -- leftmost/rightmost of its class, largest of its
    class, or the one nearest the bottom of the frame. Ground truth comes from
    COCO instance annotations, so answering needs per-instance extent rather
    than category detection alone.

ADE-Area (SEMANTIC axis, 250 items)
    Which of four named regions covers the largest area, decided from ADE20K
    polygon annotations. A second semantic source next to COCO-Area: different
    images (scene-typed: urban, nature, transportation), a far larger label
    vocabulary, and polygons drawn by annotators rather than panoptic merging.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

COCO = Path("/raid/william/project/data/coco")
ADE = Path("/raid/cathy/ADE20K_2021_17_01/images/ADE/validation")
OUT = Path(__file__).resolve().parent / "data_slices" / "tailored"
rng = random.Random(42)
LETTERS = "ABCD"
BOXCOL = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 200, 255)]


def _write(rows, name):
    with open(OUT / name, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"{name}: {len(rows)} items; answers "
          f"{dict(sorted(Counter(r['conversations'][1]['value'] for r in rows).items()))}")


def build_refer(n_items=250):
    img_dir = OUT / "images" / "cocorefer"
    img_dir.mkdir(parents=True, exist_ok=True)
    d = json.load(open(COCO / "annotations" / "instances_val2017.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    by_img = defaultdict(list)
    for a in d["annotations"]:
        if not a.get("iscrowd"):
            by_img[a["image_id"]].append(a)

    keys = sorted(by_img)
    rng.shuffle(keys)
    rows = []
    for img_id in keys:
        if len(rows) >= n_items:
            break
        anns = [a for a in by_img[img_id]
                if a["area"] >= 0.01 * imgs[img_id]["width"] * imgs[img_id]["height"]]
        if len(anns) < 4:
            continue
        cand = rng.sample(anns, 4)
        # pick a property that has a unique winner among the four
        props = []
        xs = [a["bbox"][0] for a in cand]
        areas = [a["area"] for a in cand]
        bots = [a["bbox"][1] + a["bbox"][3] for a in cand]
        for key, vals, best in (("leftmost", xs, min), ("largest", areas, max),
                                ("nearest the bottom of the image", bots, max)):
            b = best(vals)
            if sum(1 for v in vals if abs(v - b) < 1e-6) == 1:
                props.append((key, vals.index(b)))
        if not props:
            continue
        prop, gt = rng.choice(props)

        src = COCO / "val2017" / imgs[img_id]["file_name"]
        img = cv2.imread(str(src))
        if img is None:
            continue
        for i, a in enumerate(cand):
            x, y, w, h = [int(v) for v in a["bbox"]]
            cv2.rectangle(img, (x, y), (x + w, y + h), BOXCOL[i], 3)
            cv2.putText(img, LETTERS[i], (x + 4, max(y + 22, 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, BOXCOL[i], 3)
        rel = f"images/cocorefer/refer_{len(rows):04d}.jpg"
        cv2.imwrite(str(OUT / rel), img)

        q = (f"Four regions are marked A-D on the image. Which marked region is "
             f"the one that is {prop}?\nSelect from the following choices:\n" +
             "\n".join(f"({LETTERS[i]}) region {LETTERS[i]} "
                       f"({cats[cand[i]['category_id']]})" for i in range(4)) + "\n")
        rows.append({
            "id": f"cocorefer_{img_id}", "image": [rel], "video": [],
            "conversations": [{"from": "human", "value": q},
                              {"from": "gpt", "value": LETTERS[gt]}],
            "task": "Referring", "input_type": "image", "output_type": "MCQ",
            "data_source": "COCO-Refer", "others": {"property": prop}, "subtask": "",
        })
    _write(rows, "cocorefer.jsonl")


def build_ade(n_items=250):
    img_dir = OUT / "images" / "adearea"
    img_dir.mkdir(parents=True, exist_ok=True)
    jsons = sorted(ADE.rglob("*.json"))
    rng.shuffle(jsons)
    rows = []
    for jf in jsons:
        if len(rows) >= n_items:
            break
        try:
            ann = json.load(open(jf))["annotation"]
        except Exception:
            continue
        H, W = ann["imsize"][:2]
        per = defaultdict(float)
        for o in ann.get("object", []):
            poly = o.get("polygon") or {}
            xs, ys = poly.get("x"), poly.get("y")
            if not xs or not ys or len(xs) < 3:
                continue
            pts = np.array(list(zip(xs, ys)), dtype=np.int32)
            per[o["name"].split(",")[0].strip()] += abs(cv2.contourArea(pts))
        big = sorted(((k, v) for k, v in per.items() if v >= 0.02 * H * W),
                     key=lambda t: -t[1])
        if len(big) < 4 or big[0][1] < 1.5 * big[1][1]:
            continue
        cand = big[:4]
        pos = rng.randint(0, 3)
        order = [1, 2, 3]
        rng.shuffle(order)
        slots, j = [None] * 4, 0
        slots[pos] = 0
        for i in range(4):
            if slots[i] is None:
                slots[i] = order[j]
                j += 1
        names = [cand[i][0] for i in slots]

        src = jf.with_suffix(".jpg")
        if not src.exists():
            continue
        rel = f"images/adearea/ade_{len(rows):04d}.jpg"
        dst = OUT / rel
        if not dst.exists():
            os.symlink(src, dst)
        q = ("Which of the following covers the largest area in this image?\n"
             "Select from the following choices:\n" +
             "\n".join(f"({LETTERS[i]}) {names[i]}" for i in range(4)) + "\n")
        rows.append({
            "id": f"adearea_{len(rows)}", "image": [rel], "video": [],
            "conversations": [{"from": "human", "value": q},
                              {"from": "gpt", "value": LETTERS[pos]}],
            "task": "Area", "input_type": "image", "output_type": "MCQ",
            "data_source": "ADE-Area",
            "others": {"winner": cand[0][0], "scene": ann.get("scene", [""])[0]
                       if isinstance(ann.get("scene"), list) else ann.get("scene", "")},
            "subtask": "",
        })
    _write(rows, "adearea.jsonl")


if __name__ == "__main__":
    build_refer()
    build_ade()
