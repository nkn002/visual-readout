#!/usr/bin/env python3
"""Precompute the layer-21 activation grid for every sample through the running backend, so the static
page can show the heatmap without a server. Writes data/sample/activations/<sample id>.json.

    scripts/precompute_activations.py --api http://localhost:8000 --layer 21
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=300) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--data", default=str(ROOT / "data" / "sample"))
    a = ap.parse_args()
    out = Path(a.data) / "activations"
    out.mkdir(parents=True, exist_ok=True)
    for ds in get(f"{a.api}/api/datasets"):
        for s in get(f"{a.api}/api/datasets/{ds['id']}/samples?limit=1000"):
            g = get(f"{a.api}/api/samples/{s['id']}/activations?layer={a.layer}")
            (out / f"{s['id']}.json").write_text(json.dumps(g))
            print(f"{s['id']}: {g['rows']}x{g['cols']} tokens, max norm {max(g['values']):.0f}")


if __name__ == "__main__":
    main()
