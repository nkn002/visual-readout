"""Request and response models."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlmodel import SQLModel


class Health(SQLModel):
    status: str
    datasets: int
    samples: int
    data_dir: str


class DatasetRead(SQLModel):
    id: str
    name: str
    kind: str
    description: str
    conditions: list[dict[str, Any]]
    arms: list[dict[str, Any]]
    n_samples: int


class AnnotationCreate(SQLModel):
    id: str | None = None
    sample_id: str
    label: str = ""
    geometry: dict[str, Any]
    source: str = "human"
    created_at: datetime | None = None


class AnnotationRead(SQLModel):
    id: str
    sample_id: str
    label: str
    geometry: dict[str, Any]
    source: str
    created_at: datetime


class ImportResult(SQLModel):
    imported: int
    skipped: int
