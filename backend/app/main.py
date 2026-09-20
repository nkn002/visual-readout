"""visual-readout API. Run from backend/: uvicorn app.main:app --reload"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from .config import get_settings
from .database import get_engine, init_db
from .routers import annotations, datasets, health, samples
from .services.indexer import index


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    with Session(get_engine()) as session:
        app.state.index_counts = index(session, settings.data_path)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="visual-readout API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=get_settings().origins, allow_methods=["*"], allow_headers=["*"])
    for r in (health, datasets, samples, annotations):
        app.include_router(r.router)
    return app


app = create_app()
