"""Fill the Dataset and Sample tables from a manifest. Idempotent, keyed on ids."""
from __future__ import annotations

from pathlib import Path

from sqlmodel import Session

from ..models import Dataset, Sample
from .scan import load_manifest


def index(session: Session, data_dir: str | Path) -> dict[str, int]:
    m = load_manifest(data_dir)
    n_ds = n_s = 0
    for ds in m["datasets"]:
        row = session.get(Dataset, ds["id"]) or Dataset(id=ds["id"], name=ds["name"], kind=ds["kind"])
        row.name, row.kind = ds["name"], ds["kind"]
        row.description = ds.get("description", "")
        row.meta = {k: ds.get(k) for k in ("conditions", "arms", "source")}
        row.n_samples = len(ds["samples"])
        session.add(row)
        n_ds += 1
        for s in ds["samples"]:
            srow = session.get(Sample, s["id"]) or Sample(id=s["id"], dataset_id=ds["id"])
            srow.dataset_id = ds["id"]
            srow.position = int(s.get("index", 0))
            srow.phrase = s.get("phrase", "")
            srow.head = s.get("head", "")
            srow.data = s
            session.add(srow)
            n_s += 1
    session.commit()
    return {"datasets": n_ds, "samples": n_s}
