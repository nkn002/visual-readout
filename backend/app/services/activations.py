"""Per-token activation grids from the model server, cached in memory per (sample, condition, layer)."""
from __future__ import annotations

import base64
from pathlib import Path

import httpx

_cache: dict[tuple[str, str, int], dict] = {}


def cached(key: tuple[str, str, int]) -> dict | None:
    return _cache.get(key)


def fetch_grid(image_path: str | Path, layer: int, url: str, key: tuple[str, str, int] | None = None) -> dict:
    b64 = base64.standard_b64encode(Path(image_path).read_bytes()).decode("ascii")
    r = httpx.post(f"{url}/activations", json={"image": b64, "layer": layer}, timeout=180)
    r.raise_for_status()
    grid = r.json()
    out = {k: grid[k] for k in ("layer", "rows", "cols", "values", "stat")} | {"model": grid.get("model")}
    if key is not None:
        _cache[key] = out
    return out
