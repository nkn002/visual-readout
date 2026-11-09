#!/usr/bin/env python
"""Draw the HW3 design diagram: the HW2 baseline (mean pool -> MLP) and the proposed attention-pooling probe."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(14, 6.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 6.2); ax.axis("off")


def box(x, y, w, h, text, fc, ec="#333333", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11.5, weight="bold" if bold else "normal", wrap=True)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#333333"))


ROWS = [(4.35, "HW2 baseline", "mean pool\nevery token 1/L", "#e8e8e8", "MLP\n3584 → 512 → 20\n(ReLU, dropout 0.5)"),
        (1.15, "HW3 proposal", "attention pool\nlearned weights a_i\n(1 or 8 heads)", "#f6dbe8", "MLP\n3584 → 512 → 512 → 20\n(ReLU, dropout 0.5)")]
for y, title, pool, pc, mlp in ROWS:
    ax.text(0.1, y + 1.25, title, fontsize=13, weight="bold")
    box(0.1, y, 2.0, 1.0, "image region\n(box on photo)", "#ffffff")
    arrow(2.15, y + 0.5, 2.55, y + 0.5)
    box(2.6, y, 2.6, 1.0, "Qwen2.5-VL-7B (frozen)\nlayer-21 tokens\nunder the box: L × 3584", "#dde9f7")
    arrow(5.25, y + 0.5, 5.65, y + 0.5)
    box(5.7, y, 2.3, 1.0, pool, pc, bold=(pc != "#e8e8e8"))
    arrow(8.05, y + 0.5, 8.45, y + 0.5)
    box(8.5, y, 0.9, 1.0, "3584\nvector", "#ffffff")
    arrow(9.45, y + 0.5, 9.85, y + 0.5)
    box(9.9, y, 2.9, 1.0, mlp, pc)
    arrow(12.85, y + 0.5, 13.15, y + 0.5)
    box(13.2, y, 0.75, 1.0, "20\nclasses", "#ffffff")
ax.text(5.7, 0.35, "a_i = softmax over the box of  q · W_k x_i / 8   (q starts at 0, so training starts from the mean pool)", fontsize=11, style="italic")
ax.text(7.0, 3.55, "changed in HW3 (pink): how tokens are pooled, and one more hidden layer", ha="center", fontsize=11, color="#932667")
fig.suptitle("Region probe: HW2 baseline vs HW3 attention-pooling design", fontsize=14, weight="bold")
out = ROOT / "docs" / "hw3" / "architecture.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"-> {out}")
