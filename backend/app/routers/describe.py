from __future__ import annotations

from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException

from ..database import SessionDep
from ..dependencies import SampleDep, SettingsDep
from ..models import Annotation
from ..schemas import AnnotationRead, DescribeRequest, DescribeResult
from ..services import describe as svc

router = APIRouter(prefix="/api", tags=["describe"])
NOT_RUNNING = "the Qwen2.5-VL model server is not running. Start it with: sbatch scripts/model_server.sbatch"


@router.post("/samples/{sample_id}/describe", response_model=DescribeResult)
def describe(sample: SampleDep, body: DescribeRequest, session: SessionDep, settings: SettingsDep) -> DescribeResult:
    """Ask Qwen2.5-VL what the given box contains, and keep the answer as a `model` annotation."""
    g = body.geometry
    if g.get("type") != "bbox" or not all(isinstance(g.get(k), (int, float)) for k in ("x", "y", "w", "h")):
        raise HTTPException(422, "geometry must be a normalised bbox {type, x, y, w, h}")
    rel = (sample.data.get("images") or {}).get(body.condition)
    if not rel:
        raise HTTPException(404, f"sample {sample.id!r} has no image for condition {body.condition!r}")
    url = svc.model_url(settings)
    if not url:
        raise HTTPException(503, NOT_RUNNING)
    try:
        d = svc.describe_region(settings.data_path / rel, g, url)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        raise HTTPException(503, NOT_RUNNING)
    except httpx.TimeoutException:
        raise HTTPException(504, "the model server took too long to answer")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"model server error {e.response.status_code}: {e.response.text[:300]}")
    except ValueError as e:
        raise HTTPException(422, str(e))
    row = Annotation(id=str(uuid4()), sample_id=sample.id, label=d.label, note=d.description,
                     geometry={"type": "bbox", **{k: float(g[k]) for k in ("x", "y", "w", "h")}}, source="model")
    session.add(row)
    session.commit()
    session.refresh(row)
    return DescribeResult(label=d.label, description=d.description, model=d.model, annotation=AnnotationRead.model_validate(row))
