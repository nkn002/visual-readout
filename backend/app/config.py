"""Settings, read from the environment and the repo-root .env (pydantic-settings)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(str(REPO_ROOT / ".env"), ".env"), extra="ignore")

    data_dir: Path = REPO_ROOT / "data" / "sample"
    database_url: str = f"sqlite:///{REPO_ROOT / 'data' / 'db' / 'app.db'}"
    cors_origins: str = "http://localhost:5173,https://nkn002.github.io"
    # Qwen2.5-VL model server (scripts/model_server.py on a GPU node). If unset, the URL file the
    # server writes at startup (data/db/model_server.url) is read at request time.
    model_url: str | None = None
    model_url_path: Path = REPO_ROOT / "data" / "db" / "model_server.url"
    # trained region classifier (HW2): checkpoint folder and the folder holding training/model.py
    classifier_dir: Path = REPO_ROOT / "backend" / "model"
    training_dir: Path = REPO_ROOT / "training"
    anthropic_api_key: str | None = None  # falls back to the SDK's own resolution (env, ant auth profile)
    describe_model: str = "claude-opus-5"

    @property
    def data_path(self) -> Path:
        """DATA_DIR as an absolute path; a relative value is taken from the repo root."""
        p = self.data_dir
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @property
    def db_url(self) -> str:
        """DATABASE_URL with a relative sqlite path made absolute from the repo root."""
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix) and not self.database_url.startswith(prefix + "/"):
            return prefix + str((REPO_ROOT / self.database_url[len(prefix):]).resolve())
        return self.database_url

    @property
    def model_url_file(self) -> Path:
        return self.model_url_path

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
