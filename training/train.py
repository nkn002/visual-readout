#!/usr/bin/env python
"""Train the HW2 region classifier and its shallow baseline, select on validation, test once.

    python training/train.py                     # full run: small grid for both models
    python training/train.py --quick             # one config each, for a smoke test

The training loop is written out step by step (see train_one):
  1. initialise parameters (MLP.init_weights)      2. forward pass         3. cross-entropy loss
  4-5. backpropagation computes every gradient      6. Adam gradient-descent update
  7. repeat over mini-batches and epochs; stop when validation loss has not improved for `patience` epochs.

Outputs: backend/model/mlp.pt, backend/model/baseline.pt, backend/model/metrics.json, docs/hw2/loss_curve.png
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
from model import MLP  # noqa: E402

SEED = 0


def load_data(path: Path, device):
    z = np.load(path)
    X = torch.tensor(z["X"].astype(np.float32), device=device)
    y = torch.tensor(z["y"], device=device)
    split = z["split"]
    parts = {name: (X[split == s], y[split == s]) for s, name in enumerate(["train", "val", "test"])}
    return parts, z


@torch.no_grad()
def evaluate(model: MLP, X: torch.Tensor, y: torch.Tensor, n_classes: int) -> dict:
    model.eval()
    logits = torch.cat([model(X[i:i + 4096]) for i in range(0, len(X), 4096)])
    loss = F.cross_entropy(logits, y).item()
    pred = logits.argmax(1)
    top5 = (logits.topk(5, dim=1).indices == y[:, None]).any(1).float().mean().item()
    f1s, recalls = [], []
    for c in range(n_classes):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        recalls.append(r)
    return {"loss": loss, "acc": (pred == y).float().mean().item(), "top5": top5, "macro_f1": float(np.mean(f1s)), "recall_per_class": recalls}


def train_one(cfg: dict, data: dict, n_classes: int, device, log=print, on_epoch=None, should_stop=None) -> tuple[MLP, list[dict]]:
    """Train one configuration. `on_epoch(row)` receives each epoch's metrics as it finishes (the app's live
    monitor uses it); `should_stop()` returning True ends training after the current epoch (the app's Cancel)."""
    torch.manual_seed(SEED)
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    model = MLP(Xtr.shape[1], n_classes, tuple(cfg["hidden"]), cfg["activation"], cfg["dropout"]).to(device)  # 1. init
    model.set_standardization(Xtr.mean(0), Xtr.std(0))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    g = torch.Generator(device="cpu").manual_seed(SEED)
    history, best, best_state, bad = [], float("inf"), None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = torch.randperm(len(Xtr), generator=g).to(device)
        total, seen = 0.0, 0
        for i in range(0, len(Xtr), cfg["batch_size"]):
            idx = order[i:i + cfg["batch_size"]]
            opt.zero_grad()
            logits = model(Xtr[idx])                 # 2. forward pass
            loss = F.cross_entropy(logits, ytr[idx])  # 3. loss
            loss.backward()                           # 4-5. backpropagation: gradients of every parameter
            opt.step()                                # 6. gradient-descent update (Adam)
            total += loss.item() * len(idx)
            seen += len(idx)
        tr = evaluate(model, Xtr, ytr, n_classes)
        va = evaluate(model, Xva, yva, n_classes)
        history.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"], "train_acc": tr["acc"], "val_acc": va["acc"], "batch_loss": total / seen})
        log(f"    epoch {epoch:3d}  train loss {tr['loss']:.4f} acc {tr['acc']:.3f} | val loss {va['loss']:.4f} acc {va['acc']:.3f}")
        if on_epoch is not None:
            on_epoch(history[-1])
        if va["loss"] < best - 1e-4:
            best, best_state, bad = va["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
        if should_stop is not None and should_stop():
            break
    model.load_state_dict(best_state)  # 7. keep the epoch with the lowest validation loss
    return model, history


def save(model: MLP, path: Path, extra: dict) -> None:
    torch.save({"config": model.config, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, **extra}, path)


def plot(curves: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colours = {"MLP": "#932667", "Softmax regression": "#2a78d6"}
    for name, h in curves.items():
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], color=colours[name], label=f"{name}, train")
        axes[0].plot(ep, [r["val_loss"] for r in h], color=colours[name], ls="--", label=f"{name}, validation")
        axes[1].plot(ep, [r["train_acc"] for r in h], color=colours[name], label=f"{name}, train")
        axes[1].plot(ep, [r["val_acc"] for r in h], color=colours[name], ls="--", label=f"{name}, validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss per epoch")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy per epoch (20 categories, chance 0.05)")
    for ax in axes:
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, frameon=False)
    fig.suptitle("Region classifier on Qwen2.5-VL layer-21 region vectors")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "raw" / "regions20.npz"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw docs/hw2/loss_curve.png from backend/model/metrics.json")
    a = ap.parse_args()
    if a.replot:
        m = json.loads((ROOT / "backend" / "model" / "metrics.json").read_text())["models"]
        plot({name: m[name]["history"] for name in ("Softmax regression", "MLP")}, ROOT / "docs" / "hw2" / "loss_curve.png")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data, z = load_data(Path(a.data), device)
    classes = json.loads((ROOT / "backend" / "model" / "classes.json").read_text())["classes"]
    K = len(classes)
    print(f"device {device} | train {len(data['train'][1])} val {len(data['val'][1])} test {len(data['test'][1])} | {K} classes")

    base = {"activation": "relu", "dropout": 0.2, "weight_decay": 1e-4, "batch_size": 256, "epochs": 50, "patience": 5}
    # The same tuning budget for both models: learning rate and weight decay, plus width and dropout for the MLP.
    lrs, wds = ([1e-3], [1e-4]) if a.quick else ([1e-3, 3e-4], [1e-4, 1e-3])
    grids = {
        "Softmax regression": [{**base, "hidden": [], "dropout": 0.0, "lr": lr, "weight_decay": wd} for lr in lrs for wd in wds],
        "MLP": [{**base, "hidden": [h], "dropout": dr, "lr": lr, "weight_decay": wd}
                for h in ([512] if a.quick else [256, 512, 1024]) for dr in ([0.2] if a.quick else [0.2, 0.5]) for lr in lrs for wd in wds],
    }
    results, curves, chosen = {}, {}, {}
    for name, grid in grids.items():
        runs = []
        for cfg in grid:
            t0 = time.time()
            print(f"{name} {cfg['hidden']} dropout {cfg['dropout']} lr {cfg['lr']} wd {cfg['weight_decay']}")
            model, history = train_one(cfg, data, K, device)
            va = evaluate(model, *data["val"], K)
            runs.append({"config": cfg, "val_acc": va["acc"], "val_loss": va["loss"], "epochs": len(history), "seconds": round(time.time() - t0, 1), "_model": model, "_history": history})
            print(f"  -> val acc {va['acc']:.4f}, val loss {va['loss']:.4f}, {len(history)} epochs, {time.time() - t0:.1f}s")
        best = max(runs, key=lambda r: r["val_acc"])  # model selection uses validation only
        chosen[name] = best
        curves[name] = best["_history"]
        results[name] = {"grid": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs], "selected": best["config"]}

    # test is scored once, for the two selected models
    ytr = data["train"][1]
    majority = int(torch.bincount(ytr, minlength=K).argmax())
    report = {"classes": classes, "n": {s: len(data[s][1]) for s in data}, "majority_class": classes[majority], "models": {}}
    for s in ("train", "val", "test"):
        report.setdefault("majority", {})[s] = {"acc": (data[s][1] == majority).float().mean().item()}
    for name, r in chosen.items():
        m = r["_model"]
        report["models"][name] = {
            "config": r["config"],
            "params": sum(p.numel() for p in m.parameters()),
            **{s: evaluate(m, *data[s], K) for s in ("train", "val", "test")},
            "history": r["_history"],
            "grid": results[name]["grid"],
        }
    # paired bootstrap of the test-accuracy gap, MLP minus softmax regression (same test regions, 2000 resamples)
    Xte, yte = data["test"]
    with torch.no_grad():
        c_mlp = (chosen["MLP"]["_model"].eval()(Xte).argmax(1) == yte).float().cpu().numpy()
        c_lin = (chosen["Softmax regression"]["_model"].eval()(Xte).argmax(1) == yte).float().cpu().numpy()
    rng = np.random.default_rng(SEED)
    boots = [float((c_mlp[i] - c_lin[i]).mean()) for i in (rng.integers(0, len(yte), len(yte)) for _ in range(2000))]
    report["test_gap_mlp_minus_softmax"] = {"mean": float((c_mlp - c_lin).mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n": int(len(yte))}

    out = ROOT / "backend" / "model"
    save(chosen["MLP"]["_model"], out / "mlp.pt", {"classes": classes, "test": report["models"]["MLP"]["test"]})
    save(chosen["Softmax regression"]["_model"], out / "baseline.pt", {"classes": classes, "test": report["models"]["Softmax regression"]["test"]})
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    plot(curves, ROOT / "docs" / "hw2" / "loss_curve.png")
    print("\nTEST (scored once):")
    print(f"  majority class ({classes[majority]}): acc {report['majority']['test']['acc']:.4f}")
    for name in ("Softmax regression", "MLP"):
        t = report["models"][name]["test"]
        c = report["models"][name]["config"]
        print(f"  {name:20s} acc {t['acc']:.4f}  top5 {t['top5']:.4f}  macro-F1 {t['macro_f1']:.4f}  ({report['models'][name]['params']} params, hidden {c['hidden']}, dropout {c['dropout']}, lr {c['lr']}, wd {c['weight_decay']})")
    g = report["test_gap_mlp_minus_softmax"]
    print(f"  MLP - softmax test accuracy: {g['mean']:+.4f}  95% CI [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]  (paired bootstrap, n={g['n']})")


if __name__ == "__main__":
    main()
