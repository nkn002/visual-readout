from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException

from .config import Settings, get_settings
from .database import SessionDep
from .models import Sample

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_sample_or_404(sample_id: str, session: SessionDep) -> Sample:
    row = session.get(Sample, sample_id)
    if row is None:
        raise HTTPException(404, f"sample {sample_id!r} not found")
    return row


SampleDep = Annotated[Sample, Depends(get_sample_or_404)]
