#!/usr/bin/env python
"""Predict the 12 demo images' default regions with the trained MLP, for the static page.

The default regions' vectors already exist in the research repo (vgreg_V.npy, row = the sample's `key`),
so no GPU is needed. Writes data/sample/predictions.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NLA_ROOT = Path(os.environ.get("NLA_ROOT", "/standard/AikyamLab/khai/nla"))


def main() -> None:
    spec = importlib.util.spec_from_file_location("vr_training_model", ROOT / "training" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model, ckpt = mod.load_checkpoint(ROOT / "backend" / "model" / "mlp.pt")
    classes = ckpt["classes"]
    forms = json.loads((ROOT / "backend" / "model" / "classes.json").read_text())["forms"]
    to_class = {f: k for k, fs in forms.items() for f in fs}
    sys.path.insert(0, str(NLA_ROOT / "src"))
    from mmav.site import head_noun

    V = np.load(NLA_ROOT / "results" / "data" / "vgreg_V.npy", mmap_mode="r")
    out = {}
    for d in json.loads((ROOT / "data" / "sample" / "manifest.json").read_text())["datasets"]:
        for s in d["samples"]:
            x = torch.tensor(np.asarray(V[s["key"]], dtype=np.float32))[None]
            with torch.no_grad():
                p = torch.softmax(model(x), dim=1)[0]
            top = p.topk(5)
            out[s["id"]] = {
                "top": [{"label": classes[i], "prob": round(float(v), 4)} for v, i in zip(top.values.tolist(), top.indices.tolist())],
                "head": head_noun(s["phrase"]),
                "true_class": to_class.get(head_noun(s["phrase"]) or ""),
            }
            print(f"{s['id']}: '{s['phrase']}' -> {out[s['id']]['top'][0]['label']} ({out[s['id']]['top'][0]['prob']:.2f}); true class {out[s['id']]['true_class']}")
    (ROOT / "data" / "sample" / "predictions.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
