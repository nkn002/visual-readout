"""The trained region classifier (HW2): load training/model.py and backend/model/mlp.pt once, then predict.

The checkpoint stores the architecture, the weights, the input standardization and the class names,
so serving applies exactly the transform used in training.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache

from ..config import get_settings


class ClassifierUnavailable(RuntimeError):
    pass


# A model trained in the app can replace the shipped one for Predict, in memory only. mlp.pt is never overwritten.
_override: dict | None = None


def set_override(model, classes: list[str], label: str) -> None:
    global _override
    model.eval()
    _override = {"model": model, "classes": classes, "label": label}


def clear_override() -> None:
    global _override
    _override = None


def override_label() -> str | None:
    return _override["label"] if _override else None


def load():
    """(model, checkpoint dict). Raises ClassifierUnavailable when torch or the checkpoint is missing.
    Reloads when mlp.pt changes on disk, so a retrained model is served without restarting the backend."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ClassifierUnavailable("torch is not installed in the backend environment") from e
    s = get_settings()
    ckpt_path = s.classifier_dir / "mlp.pt"
    if not ckpt_path.exists():
        raise ClassifierUnavailable(f"no trained model at {ckpt_path}; run training/train.py")
    return _load(str(ckpt_path), ckpt_path.stat().st_mtime_ns)


@lru_cache(maxsize=2)
def _load(ckpt: str, _mtime: int):
    s = get_settings()
    ckpt_path = s.classifier_dir / "mlp.pt"
    spec = importlib.util.spec_from_file_location("vr_training_model", s.training_dir / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_checkpoint(ckpt_path)


def info() -> dict:
    if _override:
        c = _override["model"].config
        return {"name": _override["label"], "hidden": c["hidden"], "activation": c["activation"], "dropout": c["dropout"],
                "classes": _override["classes"], "test_acc": None}
    _, ckpt = load()
    c = ckpt["config"]
    return {"name": "MLP", "hidden": c["hidden"], "activation": c["activation"], "dropout": c["dropout"],
            "classes": ckpt["classes"], "test_acc": ckpt.get("test", {}).get("acc")}


def predict(vector: list[float], k: int = 5) -> list[dict]:
    """Top-k classes with softmax probabilities for one 3584-d region vector."""
    import torch

    if _override:
        model, classes = _override["model"], _override["classes"]
    else:
        model, ckpt = load()
        classes = ckpt["classes"]
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor([vector], dtype=torch.float32)), dim=1)[0]
    top = p.topk(min(k, len(p)))
    return [{"label": classes[i], "prob": round(float(v), 4)} for v, i in zip(top.values.tolist(), top.indices.tolist())]


def explain(vector: list[float], preview: int = 256) -> dict:
    """Every stage of the probe for one region vector, for the "Visualize the probe" view.

    input:  the standardized vector, averaged in `preview` equal chunks so the browser draws a small grid
    layers: each hidden layer's post-activation values (dropout is off at inference)
    output: logits and softmax probabilities for every category
    """
    import torch
    from torch import nn

    if _override:
        model, classes = _override["model"], _override["classes"]
    else:
        model, ckpt = load()
        classes = ckpt["classes"]
    model.eval()
    x = torch.tensor([vector], dtype=torch.float32)
    z = (x - model.mean) / model.std
    stages, h = [], z
    with torch.no_grad():
        mods = list(model.net)
        for i, m in enumerate(mods):
            h = m(h)
            nxt = mods[i + 1] if i + 1 < len(mods) else None
            if isinstance(m, nn.Linear) and nxt is not None and not isinstance(nxt, nn.Linear):
                continue  # report after the activation, not between Linear and activation
            if not isinstance(m, nn.Linear) and not isinstance(m, nn.Dropout):
                v = h[0]
                stages.append({"name": f"hidden layer {len(stages) + 1}", "size": int(v.numel()),
                               "activation": model.config["activation"], "values": [round(float(a), 4) for a in v.tolist()],
                               "active": int((v > 0).sum())})
        logits = h[0]
        probs = torch.softmax(logits, dim=0)
    d = z.shape[1]
    k = max(1, d // preview)
    chunks = z[0, : k * (d // k)].reshape(-1, k).mean(1)
    sizes = [model.config["d_in"], *model.config["hidden"], model.config["n_classes"]]
    return {
        "input": {"dim": d, "norm": float(x.norm()), "chunk": k, "preview": [round(float(a), 3) for a in chunks.tolist()]},
        "layers": stages,
        "sizes": sizes,
        "activation": model.config["activation"],
        "dropout": model.config["dropout"],
        "output": [{"label": c, "logit": round(float(l), 3), "prob": round(float(p), 4)} for c, l, p in zip(classes, logits.tolist(), probs.tolist())],
        "model": override_label() or "MLP (shipped)",
    }
