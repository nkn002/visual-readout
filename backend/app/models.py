"""SQLModel tables. Each Sample row keeps the whole manifest record in `data`, so the API
returns exactly what static mode reads from manifest.json."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    kind: str
    description: str = ""
    meta: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))  # conditions, arms, source
    n_samples: int = 0
    created_at: datetime = Field(default_factory=utcnow)


class Sample(SQLModel, table=True):
    id: str = Field(primary_key=True)
    dataset_id: str = Field(foreign_key="dataset.id", index=True)
    position: int = 0
    phrase: str = ""
    head: str = ""
    data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class Annotation(SQLModel, table=True):
    id: str = Field(primary_key=True)
    sample_id: str = Field(foreign_key="sample.id", index=True)
    label: str = ""
    note: str = ""  # free text, e.g. the model's one-sentence description
    geometry: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    source: str = "human"
    created_at: datetime = Field(default_factory=utcnow)
