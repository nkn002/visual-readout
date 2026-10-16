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
    model_url: str | None = None
    model_ready: bool = False
    model_name: str | None = None


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
    note: str = ""
    note: str = ""
    geometry: dict[str, Any]
    source: str = "human"
    created_at: datetime | None = None


class AnnotationUpdate(SQLModel):
    label: str | None = None
    note: str | None = None


class AnnotationRead(SQLModel):
    id: str
    sample_id: str
    label: str
    note: str
    note: str
    geometry: dict[str, Any]
    source: str
    created_at: datetime


class ImportResult(SQLModel):
    imported: int
    skipped: int


class DescribeRequest(SQLModel):
    geometry: dict[str, Any]  # {"type": "bbox", "x", "y", "w", "h"} normalised to the image
    condition: str = "orig"


class DescribeResult(SQLModel):
    label: str
    description: str
    model: str
    annotation: AnnotationRead


class DescribeRequest(SQLModel):
    geometry: dict[str, Any]  # {"type": "bbox", "x", "y", "w", "h"} normalised to the image
    condition: str = "orig"


class DescribeResult(SQLModel):
    label: str
    description: str
    model: str
    annotation: AnnotationRead


class ActivationRequest(SQLModel):
    geometry: dict[str, Any]  # {"type": "bbox", "x", "y", "w", "h"} normalised to the image
    layer: int = 21
    condition: str = "orig"


class ActivationVector(SQLModel):
    layer: int
    dim: int  # size of the full pooled vector (3584 for Qwen2.5-VL-7B)
    reduced: list[float]  # the pooled vector projected onto 64 principal components
    explained: list[float]  # variance ratio each component explains
    basis: str  # how the components were fitted
    norm: float  # L2 norm of the full pooled vector
    top: list[dict[str, Any]]  # the 8 largest full dimensions: {dim, value}
    region_tokens: list[int]  # flat indices (row-major on rows x cols) of the tokens that were pooled
    rows: int
    cols: int
    pooling: str | None = None
    model: str | None = None


class PredictRequest(SQLModel):
    geometry: dict[str, Any]  # {"type": "bbox", "x", "y", "w", "h"} normalised to the image
    condition: str = "orig"


class PredictResult(SQLModel):
    top: list[dict[str, Any]]  # [{label, prob}], highest first
    model: dict[str, Any]  # name, hidden sizes, activation, test accuracy
    n_tokens: int  # how many image tokens were mean-pooled into the input vector
