#!/usr/bin/env python
"""HW3 figures from results/hw3/results.json (and the saved 1-head attention model for the attention maps).

    python training/hw3_figures.py --tokens /scratch/rbc5xp/visual-readout
Writes docs/hw3/{accuracy.png, curves.png, by_token_count.png, recall_delta.png, attention_maps.png}.
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


def by_token_count(R: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = list(next(iter(R["arms"].values()))["test_seeds"][0]["acc_by_token_count"])
    for arm, r in R["arms"].items():
        if "1 hidden" not in arm:
            continue
        vals = [100 * np.mean([s["acc_by_token_count"][b] for s in r["test_seeds"]]) for b in bins]
        ax.plot(range(len(bins)), vals, marker="o", color=POOL_COLOUR[pooling_of(arm)], label=arm)
    ax.set_xticks(range(len(bins)), [f"{b} tokens" for b in bins])
    ax.set(ylabel="Test accuracy (%), 3-seed mean", xlabel="Tokens under the box",
           title="Accuracy by box size (1-hidden-layer probes)")
    ax.grid(alpha=0.3); ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "by_token_count.png", dpi=160); plt.close(fig)


def recall_delta(R: dict, arm: str) -> None:
    base = "mean, 1 hidden (HW2)"
    cls = R["classes"]
    rb = np.mean([s["recall_per_class"] for s in R["arms"][base]["test_seeds"]], 0)
    ra = np.mean([s["recall_per_class"] for s in R["arms"][arm]["test_seeds"]], 0)
    d = 100 * (ra - rb)
    order = np.argsort(d)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(cls)), d[order], color=["#932667" if v > 0 else "#8a8a8a" for v in d[order]])
    ax.set_yticks(range(len(cls)), [f"{cls[i]} ({100 * rb[i]:.0f}%)" for i in order])
    ax.axvline(0, color="#333333", lw=1)
    ax.set(xlabel="Change in test recall, points (3-seed means)", title=f"{arm} minus HW2 mean pool, per category\n(label: HW2 recall)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "recall_delta.png", dpi=160); plt.close(fig)


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
            ax.add_patch(plt.Rectangle((cc * W / cols, rr * H / rows), W / cols, H / rows, color="#ffcc00", alpha=float(min(0.85, wt * len(w) * 0.35)), lw=0))
        ax.add_patch(plt.Rectangle((x0, y0), bw, bh, fill=False, ec="#2a78d6", lw=2))
        pad = 0.25
        ax.set_xlim(max(0, x0 - pad * bw), min(W, x0 + bw * (1 + pad))); ax.set_ylim(min(H, y0 + bh * (1 + pad)), max(0, y0 - pad * bh))
        ax.set_title(f"\"{r['phrase'][:34]}\"\ntrue {classes[z['y'][k]]}, pred {pred}\nmax weight {w.max() * len(w):.1f}× uniform", fontsize=10)
        ax.axis("off")
    fig.suptitle("Where the 1-head attention probe looks (yellow = weight; blue = the box)", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / "attention_maps.png", dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default="/scratch/rbc5xp/visual-readout")
    a = ap.parse_args()
    R = json.loads((ROOT / "results" / "hw3" / "results.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    accuracy(R); curves(R); by_token_count(R)
    best_attn = max((x for x in R["arms"] if x.startswith("attention")), key=lambda x: R["arms"][x]["val"]["acc"])
    if "mean, 1 hidden (HW2)" in R["arms"]:
        recall_delta(R, best_attn)
    if (ROOT / "results" / "hw3" / "best_attn1.pt").exists():
        attention_maps(Path(a.tokens))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
