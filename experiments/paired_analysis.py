#!/usr/bin/env python3
"""
Paired per-item comparison between two experimental arms (judge-exp.md §5).

Loads all repeats of each arm from ci_runs/<variant>/repeat*/, builds
per-item correctness (averaged across repeats), and reports:
  - per-arm mean accuracy over the COMMON item set
  - paired delta with a 95% bootstrap CI over items (10k resamples)
  - McNemar exact test on per-item majority-vote outcomes
  - optional tool-conditioned variant: restrict to items where a tool whose
    name contains --condition-tool was invoked in at least one repeat of
    either arm (undilutes encoding effects in mixed-stack runs)

Usage:
  paired_analysis.py ARM_DIR_A ARM_DIR_B [--condition-tool detection]
  (ARM_DIR = .../ci_runs/<variant>)
"""

import argparse
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict


def load_arm(arm_dir):
    """-> (per_item_correct: {id: [0/1 per repeat]}, tools_used: {id: set})"""
    per_item = defaultdict(list)
    tools = defaultdict(set)
    repeats = sorted(
        d for d in glob.glob(os.path.join(arm_dir, "repeat*"))
        if os.path.isdir(d) and not d.endswith("_retry")
    )
    # prefer merged over raw when both exist for the same repeat index
    by_idx = {}
    for d in repeats:
        base = os.path.basename(d).replace("_merged", "")
        if d.endswith("_merged") or base not in by_idx:
            by_idx[base] = d
    n_rep = 0
    for d in sorted(by_idx.values()):
        files = glob.glob(os.path.join(d, "spagent_evaluation_results_*_all.json"))
        if not files:
            continue
        data = json.load(open(files[0]))
        inner = data[list(data)[0]]
        dr = inner.get("detailed_results")
        if not dr:
            print(f"  WARNING: no detailed_results in {d}, skipped", file=sys.stderr)
            continue
        n_rep += 1
        for it in dr:
            correct = it.get("is_correct")
            correct = (correct is True) or (str(correct) == "True")
            per_item[it["id"]].append(1 if correct else 0)
            for t in it.get("used_tools") or []:
                tools[it["id"]].add(str(t))
    return per_item, tools, n_rep


def mcnemar_exact(b, c):
    """Two-sided exact binomial McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("--condition-tool", default=None,
                    help="restrict to items where a tool name containing this "
                         "substring was invoked in either arm")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a_items, a_tools, a_rep = load_arm(args.arm_a)
    b_items, b_tools, b_rep = load_arm(args.arm_b)
    name_a = os.path.basename(args.arm_a.rstrip("/"))
    name_b = os.path.basename(args.arm_b.rstrip("/"))
    common = sorted(set(a_items) & set(b_items))
    print(f"A: {name_a} ({a_rep} repeats)   B: {name_b} ({b_rep} repeats)")
    print(f"common items: {len(common)} "
          f"(A-only {len(set(a_items) - set(b_items))}, "
          f"B-only {len(set(b_items) - set(a_items))})")

    if args.condition_tool:
        sub = args.condition_tool.lower()
        common = [i for i in common
                  if any(sub in t.lower() for t in (a_tools[i] | b_tools[i]))]
        print(f"tool-conditioned ('{args.condition_tool}'): {len(common)} items")
    if not common:
        print("no common items — abort")
        return

    pa = {i: sum(a_items[i]) / len(a_items[i]) for i in common}
    pb = {i: sum(b_items[i]) / len(b_items[i]) for i in common}
    acc_a = sum(pa.values()) / len(common)
    acc_b = sum(pb.values()) / len(common)
    delta = [pb[i] - pa[i] for i in common]
    mean_d = sum(delta) / len(delta)

    rng = random.Random(args.seed)
    n = len(delta)
    boots = []
    for _ in range(args.boot):
        s = sum(delta[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo, hi = boots[int(0.025 * args.boot)], boots[int(0.975 * args.boot)]

    maj_a = {i: (sum(a_items[i]) * 2 > len(a_items[i])) for i in common}
    maj_b = {i: (sum(b_items[i]) * 2 > len(b_items[i])) for i in common}
    b_disc = sum(1 for i in common if maj_a[i] and not maj_b[i])  # A right, B wrong
    c_disc = sum(1 for i in common if maj_b[i] and not maj_a[i])  # B right, A wrong
    p = mcnemar_exact(b_disc, c_disc)

    print(f"\nacc  A={acc_a:.4f}  B={acc_b:.4f}   delta(B-A)={mean_d:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}] (95% paired bootstrap)")
    print(f"McNemar (majority-vote): A-right/B-wrong={b_disc}  "
          f"B-right/A-wrong={c_disc}  exact p={p:.4f}")
    sig = "SIGNIFICANT" if (lo > 0 or hi < 0) and p < 0.05 else "not significant"
    print(f"verdict: {sig}")


if __name__ == "__main__":
    main()
