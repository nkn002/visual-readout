#!/usr/bin/env python
"""HW3 experiments: does learned attention pooling over a region's tokens beat the HW2 mean pool?

Six arms, every one trained with the same budget and the same split as HW2:
    pooling in {mean, attention 1 head, attention 8 heads}  x  depth in {1 hidden layer of 512, 2 hidden layers of 512}
"mean + 1 hidden" is the HW2 architecture, rerun in this pipeline as the baseline.

Budget per arm: Adam, lr {3e-4, 1e-3} x weight decay {1e-4, 1e-3}, dropout 0.5, batch 256, at most 50 epochs,
early stopping on validation loss (patience 5). Each arm keeps the setting with the best validation accuracy,
which then reruns with seeds 1 and 2. The test set is scored for the selected setting only.

    python training/train_hw3.py --tokens /scratch/rbc5xp/visual-readout      # full run (GPU)
    python training/train_hw3.py --synthetic                                  # tiny fake data, checks the code

Outputs: results/hw3/results.json, results/hw3/*.png, results/hw3/best_attn1.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_probe import RegionProbe  # noqa: E402

ARMS = {  # name: (pooling, heads, hidden)
    "mean, 1 hidden (HW2)": ("mean", 1, (512,)),
    "mean, 2 hidden": ("mean", 1, (512, 512)),
    "attention 1 head, 1 hidden": ("attn", 1, (512,)),
    "attention 1 head, 2 hidden": ("attn", 1, (512, 512)),
    "attention 8 heads, 1 hidden": ("attn", 8, (512,)),
    "attention 8 heads, 2 hidden": ("attn", 8, (512, 512)),
}
GRID = [{"lr": lr, "weight_decay": wd} for lr in (3e-4, 1e-3) for wd in (1e-4, 1e-3)]
BASE = {"dropout": 0.5, "batch_size": 256, "epochs": 50, "patience": 5}
COUNT_BINS = [(1, 1), (2, 9), (10, 49), (50, 10**9)]


class Data:
    """Tokens on the GPU as one flat float16 tensor, plus offsets; batches are gathered without padding."""

    def __init__(self, tokens: torch.Tensor, offsets: np.ndarray, y: np.ndarray, split: np.ndarray, device):
        self.tokens = tokens
        self.off = torch.tensor(offsets, device=device)
        self.len = self.off[1:] - self.off[:-1]
        self.y = torch.tensor(y, device=device)
        self.split = split
        self.idx = {name: torch.tensor(np.where(split == s)[0], device=device) for s, name in enumerate(["train", "val", "test"])}
        self.device = device
        self._means: torch.Tensor | None = None

    def gather(self, regions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lens = self.len[regions]
        seg = torch.repeat_interleave(torch.arange(len(regions), device=self.device), lens)
        starts = torch.repeat_interleave(self.off[regions], lens)
        first = torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
        pos = torch.arange(int(lens.sum()), device=self.device) - first
        return self.tokens[starts + pos].float(), seg

    def means(self) -> torch.Tensor:
        """The mean of every region's tokens (the HW2 input), computed once."""
        if self._means is None:
            out = []
            for s in range(0, len(self.len), 2048):
                r = torch.arange(s, min(s + 2048, len(self.len)), device=self.device)
                t, seg = self.gather(r)
                out.append(torch.zeros((len(r), t.shape[1]), device=self.device).index_add(0, seg, t) / self.len[r, None].float())
            self._means = torch.cat(out)
        return self._means


def standardization(data: Data, model: RegionProbe) -> None:
    """Training-set statistics: per-dimension mean/std of pooled vectors (MLP input) and of tokens (attention keys)."""
    tr = data.idx["train"]
    P = data.means()[tr]
    model.pool_mean.copy_(P.mean(0)); model.pool_std.copy_(P.std(0).clamp_min(1e-6))
    sel = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(0))[:4000].to(tr.device)]
    T, _ = data.gather(sel)
    model.tok_mean.copy_(T.mean(0)); model.tok_std.copy_(T.std(0).clamp_min(1e-6))


def logits_for(model: RegionProbe, data: Data, regions: torch.Tensor) -> torch.Tensor:
    if model.pooling == "mean":
        return model.forward_pooled(data.means()[regions])
    t, seg = data.gather(regions)
    return model(t, seg, len(regions))


@torch.no_grad()
def evaluate(model: RegionProbe, data: Data, split: str, K: int) -> dict:
    model.eval()
    reg = data.idx[split]
    logits = torch.cat([logits_for(model, data, reg[i:i + 1024]) for i in range(0, len(reg), 1024)])
    y = data.y[reg]
    pred = logits.argmax(1)
    correct = (pred == y)
    f1s, rec = [], []
    for c in range(K):
        tp = ((pred == c) & (y == c)).sum().item(); fp = ((pred == c) & (y != c)).sum().item(); fn = ((pred != c) & (y == c)).sum().item()
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0); rec.append(r)
    lens = data.len[reg]
    by_count = {f"{lo}-{hi if hi < 10**9 else 'max'}": correct[(lens >= lo) & (lens <= hi)].float().mean().item() for lo, hi in COUNT_BINS}
    return {"loss": F.cross_entropy(logits, y).item(), "acc": correct.float().mean().item(),
            "top5": (logits.topk(5, 1).indices == y[:, None]).any(1).float().mean().item(),
            "macro_f1": float(np.mean(f1s)), "recall_per_class": rec, "acc_by_token_count": by_count,
            "correct": correct.cpu().numpy().astype(np.int8).tolist()}


def train_one(arm: str, cfg: dict, data: Data, K: int, seed: int, log=print) -> tuple[RegionProbe, list[dict], dict]:
    pooling, heads, hidden = ARMS[arm]
    torch.manual_seed(seed)
    model = RegionProbe(data.tokens.shape[1], K, pooling, heads, hidden=hidden, dropout=cfg["dropout"]).to(data.device)  # 1. init
    standardization(data, model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    g = torch.Generator().manual_seed(seed)
    tr = data.idx["train"]
    history, best, best_state, bad = [], float("inf"), None, 0
    torch.cuda.reset_peak_memory_stats() if data.device.type == "cuda" else None
    t_train = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = tr[torch.randperm(len(tr), generator=g).to(tr.device)]
        t0 = time.time()
        for i in range(0, len(order), cfg["batch_size"]):
            b = order[i:i + cfg["batch_size"]]
            opt.zero_grad()
            loss = F.cross_entropy(logits_for(model, data, b), data.y[b])   # 2. forward  3. loss
            loss.backward()                                                  # 4-5. backprop: gradients
            opt.step()                                                       # 6. gradient-descent update (Adam)
        if data.device.type == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t0
        trm, vam = evaluate(model, data, "train", K), evaluate(model, data, "val", K)
        history.append({"epoch": epoch, "train_loss": trm["loss"], "val_loss": vam["loss"], "train_acc": trm["acc"], "val_acc": vam["acc"]})
        log(f"    {arm} lr {cfg['lr']} wd {cfg['weight_decay']} seed {seed} | epoch {epoch:2d} train {trm['loss']:.3f}/{trm['acc']:.3f} val {vam['loss']:.3f}/{vam['acc']:.3f}")
        if vam["loss"] < best - 1e-4:
            best, best_state, bad = vam["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)                                        # 7. keep the best validation epoch
    cost = {"params": sum(p.numel() for p in model.parameters()), "sec_per_epoch": round(t_train / len(history), 2),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20) if data.device.type == "cuda" else None,
            "epochs": len(history), "best_epoch": int(np.argmin([h["val_loss"] for h in history]) + 1)}
    return model, history, cost


@torch.no_grad()
def attention_stats(model: RegionProbe, data: Data, split: str = "test") -> dict:
    """How far learned attention moved from the mean pool: normalized entropy (1 = uniform) and top-token weight."""
    if model.pooling != "attn":
        return {}
    model.eval()
    reg = data.idx[split]
    reg = reg[data.len[reg] > 1]
    ents, tops = [], []
    for i in range(0, len(reg), 1024):
        r = reg[i:i + 1024]
        t, seg = data.gather(r)
        _, w = model.pool(t, seg, len(r))
        L = data.len[r].float()
        ent = torch.zeros((len(r), w.shape[1]), device=w.device).index_add(0, seg, -(w * torch.log(w.clamp_min(1e-12))))
        ents.append((ent / torch.log(L)[:, None]).mean(1))
        top = torch.zeros((len(r), w.shape[1]), device=w.device).scatter_reduce(0, seg[:, None].expand(-1, w.shape[1]), w, "amax", include_self=True)
        tops.append((top * L[:, None]).mean(1))  # top weight relative to uniform (1 = uniform)
    e, tp = torch.cat(ents).cpu().numpy(), torch.cat(tops).cpu().numpy()
    return {"norm_entropy_mean": float(e.mean()), "norm_entropy_median": float(np.median(e)),
            "top_weight_vs_uniform_median": float(np.median(tp)), "n_regions": int(len(e))}


def paired_bootstrap(a: list[int], b: list[int], n: int = 2000, seed: int = 0) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = [float((a[i] - b[i]).mean()) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return {"mean": float((a - b).mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def load(args, device) -> tuple[Data, list[str], np.ndarray]:
    classes = json.loads((ROOT / "backend" / "model" / "classes.json").read_text())["classes"]
    if args.synthetic:
        rng = np.random.default_rng(0)
        N, d, K = 600, 3584, len(classes)
        y = rng.integers(0, K, N)
        lens = rng.integers(1, 12, N)
        off = np.concatenate([[0], np.cumsum(lens)])
        T = rng.normal(0, 1, (off[-1], d)).astype(np.float16)
        for k in range(N):  # one informative token per region, so attention has something to find
            T[off[k]] [y[k] * 100:(y[k] + 1) * 100] += 3.0
        split = np.array([0] * 400 + [1] * 100 + [2] * 100, dtype=np.int8)
        return Data(torch.tensor(T, device=device), off, y, split, device), classes, lens
    z = np.load(ROOT / "data" / "raw" / "regions20.npz")
    idx = np.load(Path(args.tokens) / "index.npz")
    assert (idx["rows"] == z["row"]).all(), "token store and dataset are not aligned"
    tok = np.load(Path(args.tokens) / "tokens.npy", mmap_mode="r")
    t = torch.empty(tok.shape, dtype=torch.float16, device=device)
    for s in range(0, tok.shape[0], 200_000):
        t[s:s + 200_000] = torch.from_numpy(np.ascontiguousarray(tok[s:s + 200_000])).to(device)
    print(f"tokens on {device}: {tuple(t.shape)}; cos(mean of tokens, HW2 vector) median {np.median(idx['cos_to_stored']):.4f}", flush=True)
    off = idx["offsets"]
    return Data(t, off, z["y"], z["split"], device), classes, off[1:] - off[:-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", default="/scratch/rbc5xp/visual-readout")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "hw3"))
    ap.add_argument("--arms", default="all", help="comma-separated arm indices (0-5) or 'all'")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, classes, lens = load(a, device)
    K = len(classes)
    out = Path(a.out if not a.synthetic else Path(a.out) / "synthetic")
    out.mkdir(parents=True, exist_ok=True)
    arms = list(ARMS) if a.arms == "all" else [list(ARMS)[int(i)] for i in a.arms.split(",")]
    if a.synthetic:
        BASE["epochs"] = 3

    report = {"classes": classes, "n": {s: int(len(data.idx[s])) for s in data.idx}, "budget": {"grid": GRID, **BASE},
              "token_count": {"median": float(np.median(lens)), "mean": float(np.mean(lens)), "share_single_token": float(np.mean(lens == 1))},
              "arms": {}}
    for arm in arms:
        runs = []
        for cfg in GRID if not a.synthetic else GRID[:1]:
            c = {**BASE, **cfg}
            model, hist, cost = train_one(arm, c, data, K, seed=0)
            va = evaluate(model, data, "val", K)
            runs.append({"config": c, "val_acc": va["acc"], "val_loss": va["loss"], "cost": cost, "_m": model, "_h": hist})
            print(f"  {arm} lr {c['lr']} wd {c['weight_decay']}: val acc {va['acc']:.4f} ({cost['epochs']} epochs, {cost['sec_per_epoch']} s/epoch)", flush=True)
        best = max(runs, key=lambda r: r["val_acc"])  # selection on validation only
        seeds = [{"seed": 0, "model": best["_m"], "history": best["_h"], "cost": best["cost"]}]
        for s in (1, 2) if not a.synthetic else ():
            m, h, cost = train_one(arm, best["config"], data, K, seed=s, log=lambda *x: None)
            seeds.append({"seed": s, "model": m, "history": h, "cost": cost})
        tests = [evaluate(sd["model"], data, "test", K) for sd in seeds]
        report["arms"][arm] = {
            "config": best["config"], "grid": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs],
            "cost": best["cost"], "history": best["_h"],
            "train": {k: v for k, v in evaluate(best["_m"], data, "train", K).items() if k != "correct"},
            "val": {k: v for k, v in evaluate(best["_m"], data, "val", K).items() if k != "correct"},
            "test_seeds": [{k: v for k, v in t.items() if k != "correct"} for t in tests],
            "test_acc_mean": float(np.mean([t["acc"] for t in tests])), "test_acc_std": float(np.std([t["acc"] for t in tests])),
            "test_f1_mean": float(np.mean([t["macro_f1"] for t in tests])),
            "_correct": tests[0]["correct"], "attention": attention_stats(best["_m"], data),
        }
        print(f"== {arm}: test acc {report['arms'][arm]['test_acc_mean']:.4f} ± {report['arms'][arm]['test_acc_std']:.4f} over {len(tests)} seeds", flush=True)
        if arm == "attention 1 head, 1 hidden" and not a.synthetic:
            torch.save({"config": best["_m"].config, "state_dict": {k: v.cpu() for k, v in best["_m"].state_dict().items()}}, out / "best_attn1.pt")
    base = "mean, 1 hidden (HW2)"
    for arm in report["arms"]:
        if base in report["arms"] and arm != base:
            report["arms"][arm]["gap_vs_hw2_seed0"] = paired_bootstrap(report["arms"][arm]["_correct"], report["arms"][base]["_correct"])
    for arm in report["arms"]:
        report["arms"][arm].pop("_correct")
    (out / "results.json").write_text(json.dumps(report, indent=1))
    print(f"-> {out / 'results.json'}")


if __name__ == "__main__":
    main()
