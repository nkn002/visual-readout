from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import func
from sqlmodel import select

from ..database import SessionDep
from ..dependencies import SettingsDep
from ..models import Dataset, Sample
from ..schemas import Health
from ..services import describe as svc

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=Health)
def health(session: SessionDep, settings: SettingsDep) -> Health:
    n_ds = session.exec(select(func.count()).select_from(Dataset)).one()
    n_s = session.exec(select(func.count()).select_from(Sample)).one()
    url = svc.model_url(settings)
    info = svc.model_health(url) if url else None
    return Health(status="ok", datasets=int(n_ds), samples=int(n_s), data_dir=str(settings.data_path),
                  model_url=url, model_ready=bool(info and info.get("ready")), model_name=(info or {}).get("model"))
