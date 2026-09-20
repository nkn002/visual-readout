#!/usr/bin/env python3
"""Export a small, self-contained sample from the NLA research results into data/sample/.

Reads (under NLA_ROOT/results/data):
  counterfactual_regions.json                     items, per-arm read-outs per condition, gold text
  judge/counterfactual_regions_gpt-5.6-luna.json  per-arm, per-condition judge verdicts
  counterfactual_imgs/NN_<condition>.jpg          the four edited images per item

Writes:
  <out>/manifest.json   the app's manifest (see README, "Data")
  <out>/images/*.jpg    copied images

Usage:
  python scripts/export_sample.py --out data/sample            # the 12 items that have images
  NLA_ROOT=/path/to/nla python scripts/export_sample.py --out data/raw --all
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path

NLA_ROOT = Path(os.environ.get("NLA_ROOT", "/standard/AikyamLab/khai/nla"))
DATA = NLA_ROOT / "results" / "data"
CF = DATA / "counterfactual_regions.json"
JUDGE = DATA / "judge" / "counterfactual_regions_gpt-5.6-luna.json"
IMG_DIR = DATA / "counterfactual_imgs"

# Copied from mmav.site.ARMS so this script stays dependency-free.
ARMS = [
    {"key": "affine-identity", "label": "Frozen NLA", "short": "frozen", "colour": "#8a8a8a",
     "note": "the released verbalizer on the raw region vector, nothing trained"},
    {"key": "affine-grpo-recon-b0-vgreg-nlamse-identinit-bridgeid", "label": "Cold connector, affine",
     "short": "cold affine", "colour": "#3f9b6d",
     "note": "identity-initialised affine connector and linear bridge, trained by reconstruction alone"},
    {"key": "meanmlp-grpo-recon-b0-vgreg-nlamse-identinit-bridgeid-bridgemlp", "label": "Cold connector, MLP + MLP",
     "short": "cold MLP", "colour": "#8452c9",
     "note": "the same recipe with an MLP connector and an MLP bridge"},
    {"key": "affine-grpo-recon-b0-vgreg-nlamse-identinit-cont", "label": "Gold through the bridge",
     "short": "gold bridge", "colour": "#2a78d6",
     "note": "identity connector, bridge fitted by ridge on gold text before training"},
    {"key": "affine-grpo-recon-b0-vgreg-nlamse-bs64-coselect", "label": "Gold through the connector",
     "short": "gold warm start", "colour": "#e07b39",
     "note": "connector warm-started by SFT on gold text, ridge bridge"},
]
CONDITIONS = [
    {"key": "orig", "label": "Original", "note": "the image as the VLM saw it"},
    {"key": "content_swap", "label": "Content swap", "note": "a donor region pasted into the box"},
    {"key": "context_swap", "label": "Context swap", "note": "the box kept, the rest of the image replaced"},
    {"key": "box_blank", "label": "Box blank", "note": "the box blanked out"},
]


def image_size(path: Path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return list(im.size)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--all", action="store_true", help="export every item, even without images")
    ap.add_argument("--dataset-id", default="cf")
    a = ap.parse_args()

    cf = json.loads(CF.read_text())
    judge = json.loads(JUDGE.read_text()) if JUDGE.exists() else {"per_item": {}}
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    conds = [c["key"] for c in CONDITIONS]
    arms = [m["key"] for m in ARMS if m["key"] in cf["texts"]]

    samples = []
    for i, it in enumerate(cf["items"]):
        images = {}
        for c in conds:
            src = IMG_DIR / f"{i:02d}_{c}.jpg"
            if src.exists():
                dst = out / "images" / src.name
                shutil.copyfile(src, dst)
                images[c] = f"images/{src.name}"
        if not images and not a.all:
            continue
        size = image_size(out / images["orig"]) if "orig" in images else None
        readouts = {arm: {c: cf["texts"][arm][c][i] for c in conds if c in cf["texts"][arm]} for arm in arms}
        verdicts = {}
        for arm in arms:
            per = judge["per_item"].get(arm, {})
            verdicts[arm] = {c: {w: (per.get(c, {}).get(w) or [None] * (i + 1))[i] for w in ("orig", "donor")}
                             for c in conds}
        samples.append({
            "id": f"{a.dataset_id}-{i:02d}",
            "index": i,
            "key": it["key"],
            "image_id": it["image_id"],
            "phrase": it["phrase"],
            "head": it["head"],
            "box": it["box"],
            "image_size": size,
            "n_tokens": it.get("n_tok"),
            "donor": {k[len("donor_"):]: v for k, v in it.items() if k.startswith("donor_")},
            "context_image_id": it.get("context_image_id"),
            "images": images,
            "gold": cf["gold"].get(str(it["key"])),
            "donor_gold": cf["gold"].get(str(it.get("donor_key"))),
            "readouts": readouts,
            "judge": verdicts,
        })

    manifest = {
        "version": 1,
        "datasets": [{
            "id": a.dataset_id,
            "name": "Counterfactual regions",
            "kind": "region",
            "description": "Visual Genome regions read out by five verbalizer arms under four image edits, "
                           "with a judge's object-mention verdict for each read-out.",
            "source": {"nla_root": str(NLA_ROOT), "seed": cf.get("seed"), "judge_model": judge.get("judge_model")},
            "conditions": CONDITIONS,
            "arms": [m for m in ARMS if m["key"] in arms],
            "samples": samples,
        }],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n_img = sum(len(s["images"]) for s in samples)
    print(f"-> {out}/manifest.json: {len(samples)} samples, {n_img} images, {len(arms)} arms")


if __name__ == "__main__":
    main()
