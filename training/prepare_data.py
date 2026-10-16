#!/usr/bin/env python
"""Build the HW2 classification dataset from the research repo's region vectors.

Input (NLA_ROOT/results/data):
  vgreg_V.npy        200,000 x 3584 float32. Row i is Qwen2.5-VL-7B's hidden state 21 (output of
                     layer 20), mean-pooled over the image tokens under Visual Genome box i.
                     Built by mmav.donor.extract_region_vectors with the prompt "Describe this image."
  vgreg_meta.jsonl   one line per row: image_id, phrase, box.

Label: the head noun of the phrase (mmav.site.head_noun), mapped to one of 20 object classes.
Selection: at most 1,500 regions per class (seed 0). The 12 demo images in data/sample are excluded.
Split: by image, 80/10/10 (seed 0), so no image has regions in two splits.

Output: data/raw/regions20.npz (git-ignored) and backend/model/classes.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NLA_ROOT = Path(os.environ.get("NLA_ROOT", "/standard/AikyamLab/khai/nla"))

# Class -> stemmed head nouns (mmav.site.head_noun stems plurals: "trees" -> "tre", "grass" -> "gras").
CLASSES: dict[str, list[str]] = {
    "person": ["man", "men", "woman", "women", "person", "people", "boy", "girl", "lady", "ladi",
               "guy", "child", "children", "kid", "player", "skier", "surfer", "skateboarder"],
    "window": ["window"],
    "sign": ["sign"],
    "tree": ["tree", "tre"],
    "cloud": ["cloud"],
    "sky": ["sky"],
    "building": ["building", "house", "hous"],
    "wall": ["wall"],
    "wheel": ["wheel", "tire"],
    "grass": ["gras", "grass"],
    "shirt": ["shirt", "t-shirt"],
    "car": ["car"],
    "dog": ["dog"],
    "plate": ["plate"],
    "water": ["water"],
    "door": ["door"],
    "cat": ["cat"],
    "table": ["table"],
    "elephant": ["elephant"],
    "bear": ["bear"],
}
CAP = 1500
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "regions20.npz"))
    a = ap.parse_args()

    sys.path.insert(0, str(NLA_ROOT / "src"))
    from mmav.site import head_noun

    data = NLA_ROOT / "results" / "data"
    rows = [json.loads(line) for line in open(data / "vgreg_meta.jsonl")]
    demo = {s["image_id"] for d in json.loads((ROOT / "data/sample/manifest.json").read_text())["datasets"] for s in d["samples"]}
    to_class = {form: k for k, forms in CLASSES.items() for form in forms}
    names = list(CLASSES)

    by_class: dict[int, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        k = to_class.get(head_noun(r["phrase"]) or "")
        if k is not None and r["image_id"] not in demo:
            by_class[names.index(k)].append(i)
    rng = np.random.default_rng(SEED)
    keep = []
    for c in range(len(names)):
        idx = np.array(by_class[c])
        keep.extend(rng.choice(idx, size=min(CAP, len(idx)), replace=False).tolist())
    keep = np.array(sorted(keep))
    y = np.array([names.index(to_class[head_noun(rows[i]["phrase"])]) for i in keep], dtype=np.int64)
    img = np.array([rows[i]["image_id"] for i in keep], dtype=np.int64)

    # split by image: shuffle the distinct images, 80/10/10
    uniq = np.unique(img)
    rng.shuffle(uniq)
    n_tr, n_va = int(0.8 * len(uniq)), int(0.1 * len(uniq))
    split_of = {im: 0 for im in uniq[:n_tr]} | {im: 1 for im in uniq[n_tr:n_tr + n_va]} | {im: 2 for im in uniq[n_tr + n_va:]}
    split = np.array([split_of[im] for im in img], dtype=np.int8)

    V = np.load(data / "vgreg_V.npy", mmap_mode="r")
    X = np.asarray(V[keep], dtype=np.float32)
    assert np.isfinite(X).all(), "non-finite values in the region vectors"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X.astype(np.float16), y=y, split=split, image_id=img, row=keep,
             phrase=np.array([rows[i]["phrase"] for i in keep]))
    (ROOT / "backend" / "model").mkdir(parents=True, exist_ok=True)
    (ROOT / "backend" / "model" / "classes.json").write_text(json.dumps({"classes": names, "forms": CLASSES}, indent=1))

    print(f"-> {out}: {len(keep)} regions, {len(uniq)} images, {X.shape[1]} dims")
    for s, name in enumerate(["train", "val", "test"]):
        m = split == s
        counts = np.bincount(y[m], minlength=len(names))
        print(f"  {name:5s} {m.sum():6d} regions, {len(np.unique(img[m])):5d} images, class sizes {counts.min()}..{counts.max()}")
    print("  overlap of images between splits:", len(set(img[split == 0]) & set(img[split == 2])), "(train/test)")


if __name__ == "__main__":
    main()
