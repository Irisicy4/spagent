#!/usr/bin/env python3
"""
COCO-Count-Crowded: an instance-segmentation-driven counting benchmark.

Replaces TallyQA for the Stage-II instance-seg axis. Ground truth comes
directly from COCO instance *masks*, and items are filtered so that the answer
genuinely requires separating same-category instances:

  * 3 <= count <= 8 for the queried category (enumeration, not glance)
  * every counted instance covers >= 0.3% of the image (drop specks)
  * not iscrowd, and at least one same-category pair whose boxes overlap
    (IoU > 0.05) -- i.e. adjacency/occlusion, exactly where boxes blur
    together and per-instance masks carry information
  * distractor categories present, so "count everything" fails

Output: experiments/data_slices/tailored/cococount.jsonl (+ images symlinked),
in the repo's CV-Bench JSON schema. Deterministic (seed 42).
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

COCO = Path("/raid/william/project/data/coco")
OUT = Path(__file__).resolve().parent / "data_slices" / "tailored"
IMG = OUT / "images" / "cococount"
IMG.mkdir(parents=True, exist_ok=True)
rng = random.Random(42)
N_ITEMS = 300


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def main():
    d = json.load(open(COCO / "annotations" / "instances_val2017.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}

    by_img = defaultdict(lambda: defaultdict(list))
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        by_img[a["image_id"]][a["category_id"]].append(a)

    cand = []
    for img_id, per_cat in by_img.items():
        info = imgs[img_id]
        area_img = info["width"] * info["height"]
        for cid, anns in per_cat.items():
            big = [a for a in anns if a["area"] >= 0.003 * area_img]
            n = len(big)
            if not (3 <= n <= 8) or n != len(anns):
                continue  # require all instances countable, else GT is unfair
            boxes = [a["bbox"] for a in big]
            crowded = any(iou(boxes[i], boxes[j]) > 0.05
                          for i in range(len(boxes)) for j in range(i + 1, len(boxes)))
            if not crowded:
                continue
            if len(per_cat) < 2:      # need distractor categories
                continue
            cand.append((img_id, cid, n))

    rng.shuffle(cand)
    # cap per (category, count) so no single class/number dominates
    seen_pair, seen_img, rows = defaultdict(int), set(), []
    for img_id, cid, n in cand:
        if len(rows) >= N_ITEMS:
            break
        if img_id in seen_img or seen_pair[(cid, n)] >= 6:
            continue
        seen_img.add(img_id)
        seen_pair[(cid, n)] += 1

        fname = imgs[img_id]["file_name"]
        rel = f"images/cococount/{fname}"
        dst = OUT / rel
        if not dst.exists():
            os.symlink(COCO / "val2017" / fname, dst)

        lo = max(1, n - rng.randint(0, 3))
        opts = sorted({lo, lo + 1, lo + 2, lo + 3} | {n})
        opts = sorted(o for o in opts if o >= 1)[:4]
        while len(opts) < 4:
            opts.append(max(opts) + 1)
        if n not in opts:
            opts[-1] = n
            opts = sorted(set(opts))
        letters = "ABCD"
        name = cats[cid]
        plural = name if name.endswith("s") else name + "s"
        q = (f"How many {plural} are in the image?\n"
             f"Select from the following choices:\n" +
             "\n".join(f"({letters[j]}) {opts[j]}" for j in range(len(opts))) + "\n")
        rows.append({
            "id": f"cococount_{img_id}_{cid}",
            "image": [rel], "video": [],
            "conversations": [{"from": "human", "value": q},
                              {"from": "gpt", "value": letters[opts.index(n)]}],
            "task": "Count", "input_type": "image", "output_type": "MCQ",
            "data_source": "COCO-Count-Crowded", "others": {"gt_count": n,
                                                            "category": name},
            "subtask": "",
        })

    with open(OUT / "cococount.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    dist = defaultdict(int)
    for r in rows:
        dist[r["others"]["gt_count"]] += 1
    print(f"cococount: {len(rows)} items from {len(cand)} candidates; "
          f"count distribution {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
