from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..dependencies import SampleDep, SettingsDep
from typing import Any

from ..schemas import PredictRequest, PredictResult
from ..services import classifier
from .activations import check_box, region_grid

router = APIRouter(prefix="/api", tags=["predict"])
READOUT_LAYER = 21  # the classifier was trained on hidden state 21 (output of layer 20)


@router.post("/samples/{sample_id}/predict", response_model=PredictResult)
def predict(sample: SampleDep, body: PredictRequest, settings: SettingsDep) -> PredictResult:
    """Classify a drawn region: Qwen2.5-VL pools the tokens under the box at hidden state 21 (model server),
    then the trained MLP maps that 3584-d vector to probabilities over 20 object classes."""
    check_box(body.geometry)
    grid = region_grid(sample, body.geometry, READOUT_LAYER, body.condition, settings)
    try:
        top = classifier.predict(grid["vector"])
        info = classifier.info()
    except classifier.ClassifierUnavailable as e:
        raise HTTPException(503, str(e))
    return PredictResult(top=top, model={k: v for k, v in info.items() if k != "classes"}, n_tokens=len(grid.get("region_tokens") or []))


@router.post("/samples/{sample_id}/probe")
def probe(sample: SampleDep, body: PredictRequest, settings: SettingsDep) -> dict[str, Any]:
    """Every stage of the probe for one drawn region: input vector, hidden activations, output probabilities."""
    check_box(body.geometry)
    grid = region_grid(sample, body.geometry, READOUT_LAYER, body.condition, settings)
    try:
        out = classifier.explain(grid["vector"])
    except classifier.ClassifierUnavailable as e:
        raise HTTPException(503, str(e))
    out["n_tokens"] = len(grid.get("region_tokens") or [])
    out["layer"] = READOUT_LAYER
    return out
