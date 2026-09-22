from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Query

from ..dependencies import SampleDep, SettingsDep
from ..schemas import ActivationGrid
from ..services import activations as svc
from ..services import describe as dsvc
from .describe import NOT_RUNNING

router = APIRouter(prefix="/api", tags=["activations"])


@router.get("/samples/{sample_id}/activations", response_model=ActivationGrid)
def activations(sample: SampleDep, settings: SettingsDep, layer: int = Query(21, ge=1, le=64), condition: str = "orig") -> ActivationGrid:
    """Per-image-token activation norms at hidden_states[layer] (21 = the read-out layer L20), on the token grid."""
    key = (sample.id, condition, layer)
    hit = svc.cached(key)
    if hit:
        return ActivationGrid(**hit)
    rel = (sample.data.get("images") or {}).get(condition)
    if not rel:
        raise HTTPException(404, f"sample {sample.id!r} has no image for condition {condition!r}")
    url = dsvc.model_url(settings)
    if not url:
        raise HTTPException(503, NOT_RUNNING)
    try:
        grid = svc.fetch_grid(settings.data_path / rel, layer, url, key)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise HTTPException(503, NOT_RUNNING)
    except httpx.TimeoutException:
        raise HTTPException(504, "the model server took too long to answer")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"model server error {e.response.status_code}: {e.response.text[:300]}")
    return ActivationGrid(**grid)
