#!/usr/bin/env python
"""HW3 figures from results/hw3/results.json (and the saved 1-head attention model for the attention maps).

    python training/hw3_figures.py --tokens /scratch/rbc5xp/visual-readout
Writes docs/hw3/{accuracy, curves, gain_by_box_size, recall_delta, attention_maps}.png.
Run training/hw3_contrasts.py first: it writes the two CI files these figures read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NLA_ROOT = Path(os.environ.get("NLA_ROOT", "/standard/AikyamLab/khai/nla"))
OUT = ROOT / "docs" / "hw3"
plt.rcParams.update({"font.size": 12})
POOL_COLOUR = {"mean": "#8a8a8a", "attention 1 head": "#932667", "attention 8 heads": "#2a78d6"}


def pooling_of(arm: str) -> str:
    return arm.split(",")[0]


def accuracy(R: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pools = ["mean", "attention 1 head", "attention 8 heads"]
    for j, depth in enumerate(["1 hidden", "2 hidden"]):
        for i, p in enumerate(pools):
            arm = next((a for a in R["arms"] if a.startswith(p + ",") and depth in a), None)
            if arm is None:
                continue
            r = R["arms"][arm]
            x = i + (j - 0.5) * 0.36
            ax.bar(x, 100 * r["test_acc_mean"], 0.34, yerr=100 * r["test_acc_std"], capsize=4, color=POOL_COLOUR[p],
                   alpha=1.0 if depth == "1 hidden" else 0.55, edgecolor="#333333", label=f"{depth}" if i == 0 else None)
            ax.text(x, 100 * r["test_acc_mean"] + 0.6, f"{100 * r['test_acc_mean']:.1f}", ha="center", fontsize=10)
    ax.set_xticks(range(3), ["mean pool (HW2)", "attention, 1 head", "attention, 8 heads"])
    lo = min(100 * r["test_acc_mean"] for r in R["arms"].values())
    ax.set_ylim(lo - 4, lo + 6)
    ax.set_ylabel("Test accuracy (%), mean ± std over 3 seeds")
    ax.set_title("Test accuracy by pooling and depth (20 classes, 4,957 test regions)")
    ax.legend(title="MLP depth (light = 2 hidden)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "accuracy.png", dpi=160); plt.close(fig)


def curves(R: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for arm, r in R["arms"].items():
        h = r["history"]; p = pooling_of(arm); ls = "-" if "1 hidden" in arm else "--"
        ep = [x["epoch"] for x in h]
        axes[0].plot(ep, [x["val_loss"] for x in h], color=POOL_COLOUR[p], ls=ls, label=arm)
        axes[1].plot(ep, [x["train_loss"] for x in h], color=POOL_COLOUR[p], ls=ls, label=arm)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Validation loss (selected setting, seed 0)")
    axes[1].set(xlabel="Epoch", ylabel="Cross-entropy", title="Training loss")
    for ax in axes:
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(OUT / "curves.png", dpi=160); plt.close(fig)


def recall_delta(R: dict) -> None:
    """Per-category recall change over HW2 with 95% CI (results/hw3/recall_delta_ci.json): your design and the best arm."""
    D = json.loads((ROOT / "results" / "hw3" / "recall_delta_ci.json").read_text())
    rb = 100 * np.mean([s["recall_per_class"] for s in R["arms"]["mean, 1 hidden (HW2)"]["test_seeds"]], 0)
    cls = R["classes"]
    show = [("attention 1 head, 2 hidden", "attention 1 head, 2 hidden (your design)", "#932667"),
            ("attention 8 heads, 2 hidden", "attention 8 heads, 2 hidden (best arm)", "#2a78d6")]
    order = sorted(range(len(cls)), key=lambda c: D[show[0][0]][cls[c]][0])
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for j, (arm, lab, col) in enumerate(show):
        for yi, c in enumerate(order):
            m, lo, hi, _ = D[arm][cls[c]]
            yy = yi + (j - 0.5) * 0.32
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.5)
            ax.plot(m, yy, "o", color=col, ms=5, label=lab if yi == 0 else None)
    ax.set_yticks(range(len(cls)), [f"{cls[c]} ({rb[c]:.0f}%)" for c in order])
    ax.axvline(0, color="#333333", lw=1)
    ax.set(xlabel="Change in test recall over HW2, points (3-seed mean, 95% CI)",
           title="Recall change per category (HW2 recall in brackets)\nabout 240 test regions per category")
    ax.grid(axis="x", alpha=0.3); ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "recall_delta.png", dpi=160); plt.close(fig)


GAIN_ARMS = [("mean, 2 hidden", "mean, 2 hidden (depth only)", "#8a8a8a"),
             ("attention 1 head, 1 hidden", "attention 1 head, 1 hidden", "#d38fb8"),
             ("attention 1 head, 2 hidden", "attention 1 head, 2 hidden (your design)", "#932667"),
             ("attention 8 heads, 1 hidden", "attention 8 heads, 1 hidden", "#8ab8ef"),
             ("attention 8 heads, 2 hidden", "attention 8 heads, 2 hidden", "#2a78d6")]


def gain_by_box_size() -> None:
    """Accuracy gain over HW2 within each box-size bin with 95% CI (results/hw3/contrasts_by_box_size.json)."""
    C = json.loads((ROOT / "results" / "hw3" / "contrasts_by_box_size.json").read_text())
    bins = ["1 token", "2-9", "10-49", "50+", "all"]
    labels = ["1\n(n=843)", "2-9\n(n=1,621)", "10-49\n(n=1,606)", "50+\n(n=887)", "all\n(n=4,957)"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.16
    ax.axvspan(-0.5, 0.5, color="#f4ecd8", zorder=0)
    for j, (arm, lab, col) in enumerate(GAIN_ARMS):
        row = C[f"{arm} vs mean, 1 hidden (HW2)"]
        for i, bn in enumerate(bins):
            m, lo, hi = row[bn]
            x = i + (j - 2) * w
            ax.bar(x, m, w * 0.95, color=col, edgecolor="#333333", lw=0.5, label=lab if i == 0 else None, zorder=2)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="#333333", capsize=2, lw=1, zorder=3)
    ax.axhline(0, color="#333333", lw=1)
    ax.text(0, -1.9, "attention\ncannot act here", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(bins)), labels)
    ax.set(xlim=(-0.5, len(bins) - 0.5), ylim=(-2, 6.6), ylabel="Test accuracy gain over HW2 (points)",
           xlabel="Tokens under the box (number of test regions)",
           title="Gain over the HW2 probe by box size (3-seed mean, 95% paired-bootstrap CI)")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9, ncol=2, loc="upper left", bbox_to_anchor=(0.2, 1.0))
    fig.tight_layout(); fig.savefig(OUT / "gain_by_box_size.png", dpi=160); plt.close(fig)


def attention_maps(tokens_dir: Path, n_show: int = 6) -> None:
    import torch
    from PIL import Image
    sys.path.insert(0, str(ROOT / "training"))
    from attention_probe import RegionProbe

    ck = torch.load(ROOT / "results" / "hw3" / "best_attn1.pt", map_location="cpu", weights_only=False)
    c = ck["config"]
    m = RegionProbe(c["d"], c["n_classes"], c["pooling"], c["heads"], c["key_dim"], tuple(c["hidden"]), c["activation"], c["dropout"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    z = np.load(ROOT / "data" / "raw" / "regions20.npz")
    idx = np.load(tokens_dir / "index.npz")
    tok = np.load(tokens_dir / "tokens.npy", mmap_mode="r")
    meta = [json.loads(line) for line in open(NLA_ROOT / "results" / "data" / "vgreg_meta.jsonl")]
    classes = json.loads((ROOT / "backend" / "model" / "classes.json").read_text())["classes"]
    off, L = idx["offsets"], idx["offsets"][1:] - idx["offsets"][:-1]
    rng = np.random.default_rng(3)
    cand = np.where((z["split"] == 2) & (L >= 12) & (L <= 60))[0]
    picks = rng.choice(cand, n_show, replace=False)
    files = {}
    fig, axes = plt.subplots(2, n_show // 2, figsize=(4.2 * (n_show // 2), 8))
    for ax, k in zip(axes.ravel(), picks):
        r = meta[z["row"][k]]
        iid = r["image_id"]
        p = next((NLA_ROOT / "results/data/vg_images" / d / f"{iid}.jpg" for d in ("VG_100K", "VG_100K_2")
                  if (NLA_ROOT / "results/data/vg_images" / d / f"{iid}.jpg").exists()))
        im = Image.open(p).convert("RGB"); W, H = im.size
        t = torch.tensor(np.asarray(tok[off[k]:off[k + 1]], dtype=np.float32))
        with torch.no_grad():
            pooled, w = m.pool(t, torch.zeros(len(t), dtype=torch.long), 1)
            pred = classes[int(m.forward_pooled(pooled).argmax())]
        w = w[:, 0].numpy()
        rows, cols = idx["grid"][k]
        x0, y0, bw, bh = r["box"]
        ax.imshow(im)
        for ti, wt in zip(idx["token_idx"][off[k]:off[k + 1]], w):
            rr, cc = divmod(int(ti), int(cols))
            a_ = 0.85 * (wt - w.min()) / (w.max() - w.min() + 1e-12)  # lowest weight in the box clear, highest yellow
            ax.add_patch(plt.Rectangle((cc * W / cols, rr * H / rows), W / cols, H / rows, color="#ffcc00", alpha=float(a_), lw=0))
        ax.add_patch(plt.Rectangle((x0, y0), bw, bh, fill=False, ec="#2a78d6", lw=2))
        pad = 0.25
        ax.set_xlim(max(0, x0 - pad * bw), min(W, x0 + bw * (1 + pad))); ax.set_ylim(min(H, y0 + bh * (1 + pad)), max(0, y0 - pad * bh))
        ax.set_title(f"\"{r['phrase'][:34]}\"\ntrue {classes[z['y'][k]]}, pred {pred}\nmax weight {w.max() * len(w):.1f}× uniform", fontsize=10)
        ax.axis("off")
    fig.suptitle("Where the 1-head, 1-hidden attention probe (seed 0) looks, on 6 random test boxes with 12-60 tokens\n"
                 "yellow runs from the lowest weight in the box (clear) to the highest; blue = the box", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "attention_maps.png", dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default="/scratch/rbc5xp/visual-readout")
    a = ap.parse_args()
    R = json.loads((ROOT / "results" / "hw3" / "results.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy(R); curves(R); gain_by_box_size(); recall_delta(R)
    if (ROOT / "results" / "hw3" / "best_attn1.pt").exists():
        attention_maps(Path(a.tokens))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
