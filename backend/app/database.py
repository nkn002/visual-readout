"""Engine and the per-request session dependency."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Iterator

from fastapi import Depends
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

from .config import get_settings


@lru_cache
def get_engine() -> Engine:
    url = get_settings().db_url
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: FastAPI may touch one request's session from more than one thread.
    return create_engine(url, connect_args={"check_same_thread": False})


def init_db() -> None:
    from . import models  # noqa: F401  registers the tables on SQLModel.metadata

    SQLModel.metadata.create_all(get_engine())


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
