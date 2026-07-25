#!/usr/bin/env python3
"""Aggregate confidence intervals for the encoding experiments.

Two independent estimates:
1. Per-item bootstrap CI on any single run (resample the 500 items with
   replacement) — captures dataset sampling noise; computable from the
   run's *_all.json alone.
2. Across-repeat mean +/- t-interval over rerun_ci.sh repeats — captures
   run-to-run stochasticity (controller sampling, tool nondeterminism).

Usage:
    python experiments/ci_aggregate.py                      # all final runs, bootstrap
    python experiments/ci_aggregate.py --ci-runs            # also aggregate ci_runs/ repeats
"""
import argparse
import glob
import json
import math
import os
import random

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def load_run(path):
    """Return (accuracy, n, per_item_correct_list or None) for a run dir."""
    js = glob.glob(os.path.join(path, "spagent_evaluation_results_*_all.json"))
    if not js:
        return None
    d = list(json.load(open(js[0])).values())[0]
    acc, n = d.get("overall_accuracy"), d.get("total_samples")
    items = None
    det = d.get("detailed_results")
    if isinstance(det, list) and det and isinstance(det[0], dict) and "is_correct" in det[0]:
        items = [bool(x.get("is_correct")) for x in det]
        # Some runs were stitched by merge scripts that updated the summary
        # accuracy but left detailed_results stale/duplicated. Refuse to
        # bootstrap inconsistent per-item data rather than print a wrong CI.
        if acc is not None and abs(sum(items) / len(items) - acc) > 0.005:
            return acc, n, None
    return acc, n, items


def bootstrap_ci(items, iters=10000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(items)
    stats = sorted(
        sum(items[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    lo = stats[int(alpha / 2 * iters)]
    hi = stats[int((1 - alpha / 2) * iters) - 1]
    return lo, hi


def t_interval(vals, alpha=0.05):
    """Small-sample t CI (t table for df 1..9, 95%)."""
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, float("nan"), float("nan")
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    t95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57,
           6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}.get(n - 1, 1.96)
    half = t95 * math.sqrt(var / n)
    return m, m - half, m + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci-runs", action="store_true",
                    help="also aggregate rerun_ci.sh repeat runs")
    args = ap.parse_args()

    print(f"{'run':<58} {'acc':>7} {'95% bootstrap CI':>19}")
    print("-" * 88)
    finals = sorted(
        p for p in glob.glob(f"{RESULTS}/E*/**/", recursive=True)
        if glob.glob(os.path.join(p, "spagent_evaluation_results_*_all.json"))
        and "_partials" not in p and "ci_runs" not in p
    )
    for p in finals:
        r = load_run(p)
        if not r:
            continue
        acc, n, items = r
        tag = os.path.relpath(p, RESULTS).rstrip("/")
        if items:
            lo, hi = bootstrap_ci(items)
            print(f"{tag:<58} {acc:>7.4f}   [{lo:.4f}, {hi:.4f}]")
        else:
            print(f"{tag:<58} {acc:>7.4f}   (no per-item data)")

    if args.ci_runs:
        print("\nAcross-repeat aggregation (rerun_ci.sh):")
        for vdir in sorted(glob.glob(f"{RESULTS}/E1-det-encoding/*/ci_runs/*/")):
            accs = []
            for rep in sorted(glob.glob(os.path.join(vdir, "repeat*/"))):
                r = load_run(rep)
                if r:
                    accs.append(r[0])
            if accs:
                m, lo, hi = t_interval(accs)
                tag = os.path.relpath(vdir, RESULTS).rstrip("/")
                print(f"{tag:<58} n_rep={len(accs)} mean={m:.4f} "
                      f"95% t-CI=[{lo:.4f}, {hi:.4f}] runs={['%.4f' % a for a in accs]}")


if __name__ == "__main__":
    main()
