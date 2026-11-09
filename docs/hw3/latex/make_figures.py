#!/usr/bin/env python
"""Build the report figure at print size (7.0 in text width, 9 pt captions) as a vector PDF.

    python docs/hw3/latex/make_figures.py
Reads results/hw3/contrasts_by_box_size.json (from training/hw3_contrasts.py) and writes
docs/hw3/latex/fig_gain_by_box_size.{pdf,png}. Uses the lab's lab_style.py when it is installed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SKILL = Path(os.environ.get("LAB_STYLE_DIR", Path.home() / ".claude/skills/synced"))
for p in SKILL.glob("*/ai-conf-figures/scripts"):
    sys.path.insert(0, str(p))
try:
    import lab_style as ls
except ImportError:  # same settings, without the checker
    ls = None

WIDTH = 7.0  # \textwidth of main.tex (letter, 0.75 in margins)
ARMS = [  # (arm, legend label, colour): grey = mean pool, purple = 1 head, blue = 8 heads; dark = 2 hidden layers
    ("mean, 2 hidden", "Mean pool, 2 hidden (depth only)", "#A6A6A6"),
    ("attention 1 head, 1 hidden", "Attention 1 head, 1 hidden", "#C99BC0"),
    ("attention 1 head, 2 hidden", "Attention 1 head, 2 hidden (proposed)", "#932667"),
    ("attention 8 heads, 1 hidden", "Attention 8 heads, 1 hidden", "#9CC3EC"),
    ("attention 8 heads, 2 hidden", "Attention 8 heads, 2 hidden", "#2A78D6"),
]
BINS = [("1 token", "1 token\n(n = 843)"), ("2-9", "2 to 9\n(n = 1,621)"), ("10-49", "10 to 49\n(n = 1,606)"),
        ("50+", "50 or more\n(n = 887)"), ("all", "All boxes\n(n = 4,957)")]


def main() -> None:
    if ls:
        ls.setup("acm")  # 7.00 in full width, 9 pt captions -> 8 pt labels, 7 pt ticks
    else:
        plt.rcParams.update({"font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                             "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                             "figure.constrained_layout.use": True})
    C = json.loads((ROOT / "results/hw3/contrasts_by_box_size.json").read_text())
    x = np.array([0, 1, 2, 3, 4.35])  # gap before the all-boxes group
    w = 0.15
    fig, ax = plt.subplots(figsize=(WIDTH, WIDTH * 0.27))
    ax.axvspan(-0.45, 0.45, color="#F3EBD9", lw=0, zorder=0)
    for j, (arm, label, colour) in enumerate(ARMS):
        row = C[f"{arm} vs mean, 1 hidden (HW2)"]
        m = np.array([row[b][0] for b, _ in BINS])
        lo = np.array([row[b][1] for b, _ in BINS])
        hi = np.array([row[b][2] for b, _ in BINS])
        xs = x + (j - 2) * w
        ax.bar(xs, m, w * 0.92, color=colour, edgecolor="none", label=label, zorder=2)
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="none", ecolor="#404040", elinewidth=0.6, capsize=1.5, zorder=3)
    ax.axhline(0, color="#222222", lw=0.8, zorder=2)
    ax.text(0, -1.75, "attention cannot act", ha="center", va="bottom", fontsize=7, color="#6B5B3A")
    ax.set_xticks(x, [b for _, b in BINS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, x[-1] + 0.5)
    ax.set_ylim(-2, 6)
    ax.set_yticks([-2, 0, 2, 4, 6])
    ax.set_ylabel("Gain over HW2 (points)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.115, 1.0), ncol=2)
    out = HERE / "fig_gain_by_box_size"
    if ls:
        ls.save(fig, str(out), venue="acm", which="full")
    else:
        fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
