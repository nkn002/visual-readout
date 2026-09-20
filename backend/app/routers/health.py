from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import func
from sqlmodel import select

from ..database import SessionDep
from ..dependencies import SettingsDep
from ..models import Dataset, Sample
from ..schemas import Health

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=Health)
def health(session: SessionDep, settings: SettingsDep) -> Health:
    n_ds = session.exec(select(func.count()).select_from(Dataset)).one()
    n_s = session.exec(select(func.count()).select_from(Sample)).one()
    return Health(status="ok", datasets=int(n_ds), samples=int(n_s), data_dir=str(settings.data_path))
