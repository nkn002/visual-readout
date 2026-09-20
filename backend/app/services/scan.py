#!/usr/bin/env python3
"""Load and validate the app manifest, and stage a data folder into the static bundle.

Standard library only, so the Pages workflow can run it without installing the backend.

    python backend/app/services/scan.py --data data/sample --out frontend/public/data

The manifest is written by scripts/export_sample.py; its shape is documented in the README.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REQUIRED = ("id", "phrase", "box", "images", "readouts", "judge")


def load_manifest(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    path = data_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no manifest.json in {data_dir}")
    m = json.loads(path.read_text())
    if m.get("version") != 1:
        raise ValueError(f"unsupported manifest version {m.get('version')!r}")
    for ds in m.get("datasets", []):
        for key in ("id", "name", "kind", "samples"):
            if key not in ds:
                raise ValueError(f"dataset missing {key!r}")
        for s in ds["samples"]:
            missing = [k for k in REQUIRED if k not in s]
            if missing:
                raise ValueError(f"sample {s.get('id')!r} missing {missing}")
            for cond, rel in s["images"].items():
                if not (data_dir / rel).exists():
                    raise FileNotFoundError(f"sample {s['id']} condition {cond}: {rel} not found")
    return m


def stage(data_dir: str | Path, out_dir: str | Path) -> Path:
    """Copy manifest.json and every referenced image into out_dir (replacing it)."""
    data_dir, out = Path(data_dir), Path(out_dir)
    m = load_manifest(data_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copyfile(data_dir / "manifest.json", out / "manifest.json")
    for ds in m["datasets"]:
        for s in ds["samples"]:
            for rel in s["images"].values():
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(data_dir / rel, dst)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--out", help="stage into this directory (omit to only validate)")
    a = ap.parse_args()
    m = load_manifest(a.data)
    n = sum(len(d["samples"]) for d in m["datasets"])
    print(f"{a.data}: {len(m['datasets'])} dataset(s), {n} samples: ok")
    if a.out:
        out = stage(a.data, a.out)
        print(f"-> staged into {out}")


if __name__ == "__main__":
    main()
