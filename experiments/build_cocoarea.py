#!/usr/bin/env python3
"""
COCO-Area: a SEMANTIC-segmentation benchmark (Stage-I's third seg subtype).

Everything we have run so far on the seg axis is instance-style counting or
referring. This targets the semantic axis directly: the question asks which
of four named regions covers the largest area of the image, which is decided
by region *extent*, not by instance identity or localisation. Ground truth
comes from COCO panoptic val2017 segment areas, and the candidate set mixes
"stuff" classes (sky, road, grass, wall) with "thing" classes, so a detector
alone cannot answer it -- only a semantic map can.

Filters: 4 candidate categories all present; the winner covers >=1.5x the
runner-up (no ambiguous ties); every candidate covers >=2% of the image;
answer position drawn uniformly so no letter carries a prior.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

COCO = Path("/raid/william/project/data/coco")
OUT = Path(__file__).resolve().parent / "data_slices" / "tailored"
IMG = OUT / "images" / "cocoarea"
IMG.mkdir(parents=True, exist_ok=True)
rng = random.Random(42)
N_ITEMS = 250


def main():
    pan = json.load(open(COCO / "annotations" / "panoptic_val2017.json"))
    cats = {c["id"]: c for c in pan["categories"]}
    imgs = {i["id"]: i for i in pan["images"]}

    rows = []
    anns = list(pan["annotations"])
    rng.shuffle(anns)
    for a in anns:
        if len(rows) >= N_ITEMS:
            break
        info = imgs[a["image_id"]]
        area_img = info["width"] * info["height"]
        # total area per category (merge instances of a class)
        per_cat = defaultdict(int)
        for seg in a["segments_info"]:
            per_cat[seg["category_id"]] += seg["area"]
        big = [(cid, ar) for cid, ar in per_cat.items() if ar >= 0.02 * area_img]
        if len(big) < 4:
            continue
        big.sort(key=lambda t: -t[1])
        # winner must be unambiguous against the best of the three distractors
        cand = big[:4]
        if cand[0][1] < 1.5 * cand[1][1]:
            continue
        winner = cand[0][0]
        names = [cats[c]["name"].replace("-merged", "").replace("-other", "").replace("-stuff", "")
                 for c, _ in cand]
        if len(set(names)) < 4:
            continue
        # place the winner uniformly among the four slots
        order = [1, 2, 3]
        rng.shuffle(order)
        pos = rng.randint(0, 3)
        slots = [None] * 4
        slots[pos] = 0
        j = 0
        for i in range(4):
            if slots[i] is None:
                slots[i] = order[j]
                j += 1
        opts = [names[i] for i in slots]

        fname = info["file_name"].replace(".png", ".jpg")
        rel = f"images/cocoarea/{fname}"
        dst = OUT / rel
        if not dst.exists():
            src = COCO / "val2017" / fname
            if not src.exists():
                continue
            os.symlink(src, dst)

        letters = "ABCD"
        q = ("Which of the following covers the largest area in this image?\n"
             "Select from the following choices:\n" +
             "\n".join(f"({letters[i]}) {opts[i]}" for i in range(4)) + "\n")
        rows.append({
            "id": f"cocoarea_{a['image_id']}",
            "image": [rel], "video": [],
            "conversations": [{"from": "human", "value": q},
                              {"from": "gpt", "value": letters[pos]}],
            "task": "Area", "input_type": "image", "output_type": "MCQ",
            "data_source": "COCO-Area", "others": {
                "winner": cats[winner]["name"],
                "winner_frac": round(cand[0][1] / area_img, 3),
                "is_stuff": not bool(cats[winner]["isthing"])},
            "subtask": "",
        })

    with open(OUT / "cocoarea.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stuff = sum(1 for r in rows if r["others"]["is_stuff"])
    print(f"cocoarea: {len(rows)} items; winner is a stuff class in {stuff} "
          f"({100 * stuff / max(len(rows), 1):.0f}%)")


if __name__ == "__main__":
    main()
