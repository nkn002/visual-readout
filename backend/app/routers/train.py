from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..schemas import TrainRequest
from ..services import classifier, trainer

router = APIRouter(prefix="/api", tags=["train"])
ACTIVATIONS = ("relu", "gelu", "tanh", "sigmoid")


@router.post("/train")
def start(body: TrainRequest) -> dict[str, Any]:
    """Start one training run in the background with the user's settings; poll GET /api/train/{id}."""
    if body.activation not in ACTIVATIONS:
        raise HTTPException(422, f"activation must be one of {ACTIVATIONS}")
    if not (1e-6 <= body.lr <= 1.0 and 1 <= body.epochs <= 200 and len(body.hidden) <= 4 and all(1 <= h <= 4096 for h in body.hidden)
            and 0.0 <= body.dropout < 1.0 and 8 <= body.batch_size <= 4096 and 100 <= body.n_train and 1 <= body.patience):
        raise HTTPException(422, "a setting is out of range")
    try:
        job = trainer.start(body.model_dump())
    except trainer.TrainingUnavailable as e:
        raise HTTPException(503, str(e))
    except classifier.ClassifierUnavailable as e:
        raise HTTPException(503, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return job.snapshot()


@router.get("/train")
def latest() -> dict[str, Any] | None:
    job = trainer.current()
    return job.snapshot() if job else None


@router.get("/train/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    job = trainer.current()
    if not job or job.id != job_id:
        raise HTTPException(404, f"no training run {job_id!r}")
    return job.snapshot()


@router.post("/train/{job_id}/cancel")
def cancel(job_id: str) -> dict[str, Any]:
    job = trainer.current()
    if not job or job.id != job_id:
        raise HTTPException(404, f"no training run {job_id!r}")
    job.stop.set()
    return job.snapshot()


@router.post("/train/{job_id}/use")
def use(job_id: str) -> dict[str, Any]:
    """Serve this run's model for Predict until reset (in memory; the shipped mlp.pt is untouched)."""
    try:
        trainer.use_for_predict(job_id)
    except KeyError:
        raise HTTPException(404, f"no finished training run {job_id!r}")
    return {"in_use": classifier.override_label()}


@router.post("/train/reset-model")
def reset_model() -> dict[str, Any]:
    classifier.clear_override()
    return {"in_use": None}
