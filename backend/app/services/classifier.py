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


@lru_cache
def load():
    """(model, checkpoint dict). Raises ClassifierUnavailable when torch or the checkpoint is missing."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ClassifierUnavailable("torch is not installed in the backend environment") from e
    s = get_settings()
    ckpt_path = s.classifier_dir / "mlp.pt"
    if not ckpt_path.exists():
        raise ClassifierUnavailable(f"no trained model at {ckpt_path}; run training/train.py")
    spec = importlib.util.spec_from_file_location("vr_training_model", s.training_dir / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_checkpoint(ckpt_path)


def info() -> dict:
    _, ckpt = load()
    c = ckpt["config"]
    return {"name": "MLP", "hidden": c["hidden"], "activation": c["activation"], "dropout": c["dropout"],
            "classes": ckpt["classes"], "test_acc": ckpt.get("test", {}).get("acc")}


def predict(vector: list[float], k: int = 5) -> list[dict]:
    """Top-k classes with softmax probabilities for one 3584-d region vector."""
    import torch

    model, ckpt = load()
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor([vector], dtype=torch.float32)), dim=1)[0]
    top = p.topk(min(k, len(p)))
    return [{"label": ckpt["classes"][i], "prob": round(float(v), 4)} for v, i in zip(top.values.tolist(), top.indices.tolist())]
