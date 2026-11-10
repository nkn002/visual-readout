#!/usr/bin/env python
"""HW3: paired-bootstrap contrasts by box size and by category, from the saved per-item test correctness.

    python hw3/hw3_contrasts.py --tokens /scratch/rbc5xp/visual-readout
Reads hw3/results/arm_<i>_correct.npz (3 seeds x 4,957 test regions) and writes
hw3/results/contrasts_by_box_size.json and hw3/results/recall_delta_ci.json.

Each item's correctness is averaged over the 3 seeds first, so the CI covers test-set sampling only,
not training-seed variance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "hw3" / "results"
BASE = "mean, 1 hidden (HW2)"
PAIRS = [("attention 1 head, 1 hidden", BASE), ("attention 8 heads, 1 hidden", BASE),
         ("attention 8 heads, 2 hidden", "mean, 2 hidden"), ("mean, 2 hidden", BASE),
         ("attention 8 heads, 2 hidden", BASE), ("attention 1 head, 2 hidden", BASE)]
RECALL_ARMS = ["mean, 2 hidden", "attention 1 head, 1 hidden", "attention 1 head, 2 hidden",
               "attention 8 heads, 1 hidden", "attention 8 heads, 2 hidden"]


def boot(d: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return 100 * d.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", default="/scratch/rbc5xp/visual-readout")
    a = ap.parse_args()
    R = json.loads((RES / "results.json").read_text())
    arms, classes = list(R["arms"]), R["classes"]
    z = np.load(ROOT / "data" / "raw" / "regions20.npz")
    test = z["split"] == 2
    idx = np.load(Path(a.tokens) / "index.npz")
    L = (idx["offsets"][1:] - idx["offsets"][:-1])[test]
    y = z["y"][test]
    C = {arm: np.load(RES / f"arm_{i}_correct.npz")["correct"].astype(float).mean(0) for i, arm in enumerate(arms)}

    bins = {"1 token": L == 1, "2-9": (L >= 2) & (L <= 9), "10-49": (L >= 10) & (L <= 49), "50+": L >= 50, "all >1 token": L > 1}
    by_size = {}
    for A, B in PAIRS:
        d = C[A] - C[B]
        by_size[f"{A} vs {B}"] = {**{k: boot(d[m]) for k, m in bins.items()}, "all": boot(d)}
    (RES / "contrasts_by_box_size.json").write_text(json.dumps(by_size, indent=1))

    rng = np.random.default_rng(0)
    recall = {}
    for A in RECALL_ARMS:
        d = C[A] - C[BASE]
        recall[A] = {}
        for c, name in enumerate(classes):
            dc = d[y == c]
            b = [dc[rng.integers(0, len(dc), len(dc))].mean() for _ in range(2000)]
            recall[A][name] = [100 * dc.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5), int(len(dc))]
    (RES / "recall_delta_ci.json").write_text(json.dumps(recall, indent=1))
    for k, row in by_size.items():
        print(k, {b: f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" for b, (m, lo, hi) in row.items()})


if __name__ == "__main__":
    main()
