from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException

from ..dependencies import SampleDep, SettingsDep
from ..schemas import ActivationRequest, ActivationVector
from ..services import activations as svc
from ..services import describe as dsvc
from .describe import NOT_RUNNING

router = APIRouter(prefix="/api", tags=["activations"])


@router.post("/samples/{sample_id}/activations", response_model=ActivationVector)
def activations(sample: SampleDep, body: ActivationRequest, settings: SettingsDep) -> ActivationVector:
    """The activation of one drawn region: the tokens under the box, mean-pooled at hidden_states[layer]
    (21 = the read-out layer L20) into the single vector the research connector reads."""
    g = body.geometry
    if g.get("type") != "bbox" or not all(isinstance(g.get(k), (int, float)) for k in ("x", "y", "w", "h")):
        raise HTTPException(422, "geometry must be a normalised bbox {type, x, y, w, h}")
    key = svc.key_for(sample.id, body.condition, body.layer, g)
    hit = svc.cached(key)
    if hit:
        return ActivationVector(**hit)
    rel = (sample.data.get("images") or {}).get(body.condition)
    if not rel:
        raise HTTPException(404, f"sample {sample.id!r} has no image for condition {body.condition!r}")
    url = dsvc.model_url(settings)
    if not url:
        raise HTTPException(503, NOT_RUNNING)
    try:
        grid = svc.fetch_grid(settings.data_path / rel, g, body.layer, url, key)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise HTTPException(503, NOT_RUNNING)
    except httpx.TimeoutException:
        raise HTTPException(504, "the model server took too long to answer")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"model server error {e.response.status_code}: {e.response.text[:300]}")
    return ActivationVector(**grid)
