from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..dependencies import SampleDep, SettingsDep

router = APIRouter(prefix="/api", tags=["samples"])


@router.get("/samples/{sample_id}/file")
def sample_file(sample: SampleDep, settings: SettingsDep, condition: str = "orig") -> FileResponse:
    rel = (sample.data.get("images") or {}).get(condition)
    if not rel:
        raise HTTPException(404, f"sample {sample.id!r} has no image for condition {condition!r}")
    root = settings.data_path.resolve()
    path = (root / rel).resolve()
    if root not in path.parents or not path.exists():
        raise HTTPException(404, f"file for {sample.id!r} not found")
    return FileResponse(path)


@router.get("/samples/{sample_id}", response_model=dict[str, Any])
def get_sample(sample: SampleDep) -> dict[str, Any]:
    return sample.data
