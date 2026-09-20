from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import or_
from sqlmodel import select

from ..database import SessionDep
from ..models import Dataset, Sample
from ..schemas import DatasetRead

router = APIRouter(prefix="/api", tags=["datasets"])


def _read(row: Dataset) -> DatasetRead:
    return DatasetRead(id=row.id, name=row.name, kind=row.kind, description=row.description,
                       conditions=row.meta.get("conditions") or [], arms=row.meta.get("arms") or [],
                       n_samples=row.n_samples)


@router.get("/datasets", response_model=list[DatasetRead])
def list_datasets(session: SessionDep) -> list[DatasetRead]:
    return [_read(r) for r in session.exec(select(Dataset).order_by(Dataset.id)).all()]


@router.get("/datasets/{dataset_id}/samples", response_model=list[dict[str, Any]])
def list_samples(dataset_id: str, session: SessionDep, offset: int = 0,
                 limit: int = Query(100, ge=1, le=1000), q: str | None = None) -> list[dict[str, Any]]:
    if session.get(Dataset, dataset_id) is None:
        raise HTTPException(404, f"dataset {dataset_id!r} not found")
    stmt = select(Sample).where(Sample.dataset_id == dataset_id)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(Sample.phrase.ilike(like), Sample.head.ilike(like), Sample.id.ilike(like)))
    stmt = stmt.order_by(Sample.position).offset(offset).limit(limit)
    return [r.data for r in session.exec(stmt).all()]
