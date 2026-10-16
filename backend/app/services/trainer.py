"""Interactive training (HW2 optional): train the region classifier in a background thread with settings the
user picks, and record every epoch so the browser can draw the curves live.

It reuses train_one and evaluate from training/train.py, so the app runs exactly the loop the report
describes. One job runs at a time; results live in memory; the committed mlp.pt is never overwritten.
"""
from __future__ import annotations

import importlib.util
import os
import threading
import time
import uuid
from functools import lru_cache

from ..config import get_settings
from . import classifier


class TrainingUnavailable(RuntimeError):
    pass


@lru_cache
def _train_module():
    s = get_settings()
    spec = importlib.util.spec_from_file_location("vr_training_train", s.training_dir / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # train.py puts training/ on sys.path and imports model.MLP
    return mod


_data_cache: dict = {}


def _data():
    """The labelled split as CPU tensors, loaded once: {train, val, test: (X, y)}, phrases, classes."""
    s = get_settings()
    if s.train_data not in _data_cache:
        try:
            import numpy as np
            import torch
        except ImportError as e:
            raise TrainingUnavailable("torch is not installed in the backend environment") from e
        if not s.train_data.exists():
            raise TrainingUnavailable(f"no training data at {s.train_data}; run training/prepare_data.py")
        z = np.load(s.train_data)
        X = torch.tensor(z["X"].astype(np.float32))
        y = torch.tensor(z["y"])
        split = z["split"]
        parts = {name: (X[split == k], y[split == k]) for k, name in enumerate(["train", "val", "test"])}
        phrases = z["phrase"][split == 2].tolist() if "phrase" in z.files else None
        _data_cache[s.train_data] = (parts, phrases)
    parts, phrases = _data_cache[s.train_data]
    return parts, phrases


class Job:
    def __init__(self, config: dict):
        self.id = uuid.uuid4().hex[:8]
        self.config = config
        self.status = "queued"
        self.rows: list[dict] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.started = time.time()
        self.finished: float | None = None
        self.stop = threading.Event()
        self.model = None

    def snapshot(self) -> dict:
        return {"id": self.id, "status": self.status, "config": self.config, "rows": list(self.rows),
                "result": self.result, "error": self.error, "elapsed": round((self.finished or time.time()) - self.started, 1),
                "in_use": classifier.override_label() == self.label(), "stopping": self.stop.is_set() and self.status == "running"}

    def label(self) -> str:
        return f"MLP trained in the app ({self.id})"


_lock = threading.Lock()
_current: Job | None = None


def current() -> Job | None:
    return _current


def start(config: dict) -> Job:
    global _current
    with _lock:
        if _current and _current.status in ("queued", "running"):
            raise RuntimeError("a training run is already in progress")
        parts, phrases = _data()  # fail fast, before the thread starts
        job = Job(config)
        _current = job
    threading.Thread(target=_run, args=(job, parts, phrases), daemon=True).start()
    return job


def _run(job: Job, parts: dict, phrases) -> None:
    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tm = _train_module()
    c = job.config
    classes = classifier.load()[1]["classes"]
    K = len(classes)
    try:
        job.status = "running"
        g = torch.Generator().manual_seed(0)
        Xtr, ytr = parts["train"]
        n = min(int(c["n_train"]), len(Xtr))
        idx = torch.randperm(len(Xtr), generator=g)[:n]
        data = {"train": (Xtr[idx], ytr[idx]), "val": parts["val"], "test": parts["test"]}
        cfg = {"hidden": list(c["hidden"]), "activation": c["activation"], "dropout": float(c["dropout"]), "lr": float(c["lr"]),
               "weight_decay": float(c["weight_decay"]), "batch_size": int(c["batch_size"]), "epochs": int(c["epochs"]),
               "patience": int(c["patience"])}
        model, _ = tm.train_one(cfg, data, K, torch.device("cpu"), log=lambda *a: None,
                                on_epoch=job.rows.append, should_stop=job.stop.is_set)
        job.model = model
        metrics = {s: {k: v for k, v in tm.evaluate(model, *data[s], K).items() if k != "recall_per_class"} for s in ("train", "val", "test")}
        Xte, yte = data["test"]
        pick = torch.randperm(len(yte), generator=g)[:8]
        with torch.no_grad():
            p = torch.softmax(model(Xte[pick]), dim=1)
        examples = [{"phrase": phrases[i] if phrases else None, "true": classes[int(yte[i])],
                     "pred": classes[int(p[j].argmax())], "prob": round(float(p[j].max()), 3)} for j, i in enumerate(pick.tolist())]
        job.result = {"metrics": metrics, "examples": examples, "n_train": n,
                      "params": sum(q.numel() for q in model.parameters()), "epochs_run": len(job.rows)}
        job.status = "cancelled" if job.stop.is_set() else "done"
    except Exception as e:  # noqa: BLE001  shown to the user in the panel
        job.status, job.error = "error", f"{type(e).__name__}: {e}"
    finally:
        job.finished = time.time()


def use_for_predict(job_id: str) -> None:
    job = _current
    if not job or job.id != job_id or job.model is None:
        raise KeyError(job_id)
    classifier.set_override(job.model, classifier.load()[1]["classes"], job.label())
