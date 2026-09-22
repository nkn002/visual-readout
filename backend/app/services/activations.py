"""Region activation vectors from the model server, cached in memory per (sample, condition, layer, box)."""
from __future__ import annotations

import base64
from pathlib import Path

import httpx

Key = tuple[str, str, int, tuple[float, float, float, float]]
_cache: dict[Key, dict] = {}


def key_for(sample_id: str, condition: str, layer: int, box: dict) -> Key:
    return (sample_id, condition, layer, tuple(round(float(box[k]), 4) for k in ("x", "y", "w", "h")))


def cached(key: Key) -> dict | None:
    return _cache.get(key)


def fetch_grid(image_path: str | Path, box: dict, layer: int, url: str, key: Key | None = None) -> dict:
    b64 = base64.standard_b64encode(Path(image_path).read_bytes()).decode("ascii")
    body = {"image": b64, "layer": layer, "box": {k: float(box[k]) for k in ("x", "y", "w", "h")}}
    r = httpx.post(f"{url}/activations", json=body, timeout=180)
    r.raise_for_status()
    grid = r.json()
    out = {k: grid.get(k) for k in ("layer", "dim", "reduced", "explained", "basis", "norm", "top", "region_tokens", "rows", "cols", "pooling", "model")}
    if key is not None:
        _cache[key] = out
    return out
