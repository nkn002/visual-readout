from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException

from ..dependencies import SampleDep, SettingsDep
from ..schemas import ActivationRequest, ActivationVector
from ..services import activations as svc
from ..services import describe as dsvc
from .describe import NOT_RUNNING

router = APIRouter(prefix="/api", tags=["activations"])


def region_grid(sample, geometry: dict, layer: int, condition: str, settings) -> dict:
    """The model server's answer for one box (cached): pooled vector, PCA-64 scores, pooled token ids."""
    key = svc.key_for(sample.id, condition, layer, geometry)
    hit = svc.cached(key)
    if hit and hit.get("vector"):
        return hit
    rel = (sample.data.get("images") or {}).get(condition)
    if not rel:
        raise HTTPException(404, f"sample {sample.id!r} has no image for condition {condition!r}")
    url = dsvc.model_url(settings)
    if not url:
        raise HTTPException(503, NOT_RUNNING)
    try:
        return svc.fetch_grid(settings.data_path / rel, geometry, layer, url, key)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise HTTPException(503, NOT_RUNNING)
    except httpx.TimeoutException:
        raise HTTPException(504, "the model server took too long to answer")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"model server error {e.response.status_code}: {e.response.text[:300]}")


def check_box(g: dict) -> None:
    if g.get("type") != "bbox" or not all(isinstance(g.get(k), (int, float)) for k in ("x", "y", "w", "h")):
        raise HTTPException(422, "geometry must be a normalised bbox {type, x, y, w, h}")


@router.post("/samples/{sample_id}/activations", response_model=ActivationVector)
def activations(sample: SampleDep, body: ActivationRequest, settings: SettingsDep) -> ActivationVector:
    """The activation of one drawn region: the tokens under the box, mean-pooled at hidden_states[layer]
    (21 = the read-out layer L20) into the single vector the research connector reads."""
    check_box(body.geometry)
    grid = region_grid(sample, body.geometry, body.layer, body.condition, settings)
    return ActivationVector(**svc.public(grid))
