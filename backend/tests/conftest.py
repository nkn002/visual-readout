import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_TMP = Path(os.environ.get("PYTEST_TMP", "/tmp")) / "visual-readout-tests"
_TMP.mkdir(parents=True, exist_ok=True)
_DB = _TMP / "test.db"
if _DB.exists():
    _DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{_DB}"
os.environ["DATA_DIR"] = str(REPO / "data" / "sample")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
