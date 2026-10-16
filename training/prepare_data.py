#!/usr/bin/env python
"""Build the HW2 classification dataset from the research repo's region vectors.

Input (NLA_ROOT/results/data):
  vgreg_V.npy        200,000 x 3584 float32. Row i is Qwen2.5-VL-7B's hidden state 21 (output of
                     layer 20), mean-pooled over the image tokens under Visual Genome box i.
                     Built by mmav.donor.extract_region_vectors with the prompt "Describe this image."
  vgreg_meta.jsonl   one line per row: image_id, phrase, box.

Label: the head noun of the phrase (mmav.site.head_noun), mapped to one of 20 semantic categories
(person, domestic animal, wild animal, ..., sports equipment; see GROUPS). Three phrases override the head
noun: "license plate" -> vehicle, "hot dog" -> food, "teddy bear" -> household and electronics.
Selection: at most 2,500 regions per category (seed 0), so all 20 are balanced. The 12 demo images are excluded.
Split: by image, 80/10/10 (seed 0), so no image has regions in two splits.

Output: data/raw/regions20.npz (git-ignored) and backend/model/classes.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NLA_ROOT = Path(os.environ.get("NLA_ROOT", "/standard/AikyamLab/khai/nla"))

# 20 semantic categories over Visual Genome head nouns. Words are singular; label_of() adds the plural stems
# that mmav.site.head_noun produces ("tables" -> "tabl", "trees" -> "tre") and handles irregular plurals.
GROUPS = {
 "person": "man woman person boy girl lady guy child kid player skier surfer skateboarder snowboarder baby adult toddler teenager batter catcher pitcher umpire rider pedestrian tourist passenger spectator crowd chef worker officer policeman gentleman",
 "domestic animal": "dog cat horse cow sheep puppy kitten pony lamb calf goat pig cattle bull donkey",
 "wild animal": "bear elephant zebra giraffe bird lion tiger deer duck goose seagull pigeon swan monkey",
 "head and face": "head eye ear nose face mouth hair neck beak forehead cheek chin lip tooth eyebrow beard mustache",
 "limb": "leg arm hand foot finger paw tail knee elbow wrist thumb hoof toe shoulder",
 "clothing and accessory": "shirt hat jacket cap pant shoe short tie glove helmet boot sock coat jean dress sweater vest hoodie skirt scarf uniform sneaker sandal jersey outfit sleeve bag backpack umbrella purse handbag suitcase luggage watch necklace bracelet earring wristband",
 "vehicle": "car bus train truck boat plane airplane motorcycle bike bicycle van vehicle taxi jet aircraft ship sailboat scooter tractor trailer suv jeep locomotive carriage cart wheel tire headlight windshield bumper taillight hubcap",
 "sky and cloud": "sky cloud sun moon sunset",
 "plant": "tree leaf bush plant flower branch grass shrub weed vine palm petal stem foliage hedge moss lawn",
 "ground and water": "ground snow sand dirt field hill mountain rock water wave ripple ocean beach river lake sea shore stone mud puddle gravel pebble slope terrain",
 "building": "building house tower roof chimney church skyscraper barn hut shed steeple balcony awning porch arch archway",
 "window and door": "window door doorway shutter windowpane garage",
 "wall, floor and ceiling": "wall floor ceiling tile brick carpet rug stair staircase column pillar",
 "street and road": "street road sidewalk track pole post fence railing hydrant cone crosswalk curb lane pavement lamppost streetlight bench meter gate",
 "food": "pizza banana cake food sandwich donut doughnut cheese carrot fruit apple slice bread meat orange tomato onion broccoli vegetable salad sauce topping pepper lettuce egg fry sausage bun crust frosting icing cookie dessert pastry potato rice meal",
 "tableware": "plate bowl cup fork knife spoon tray napkin dish mug jar bottle lid container pan utensil saucer teapot chopstick",
 "furniture": "table chair bed couch sofa cabinet shelf counter desk pillow blanket curtain drawer dresser stool nightstand headboard cushion mattress bookshelf armchair",
 "household and electronics": "lamp mirror sink toilet faucet towel laptop keyboard monitor computer tv television phone cellphone screen remote clock vase book refrigerator fridge oven microwave stove toaster speaker outlet mouse candle painting bathtub shower soap toothbrush",
 "sign and text": "sign letter number logo writing word lettering sticker label text banner poster billboard graffiti name placard",
 "sports equipment": "kite ball surfboard frisbee racket racquet skateboard ski snowboard net",
}
IRREGULAR = {"men": "person", "women": "person", "people": "person", "children": "person", "feet": "limb", "teeth": "head and face",
             "leaves": "plant", "knives": "tableware", "mice": "household and electronics", "fries": "food"}
# phrase-level fixes where the head noun alone is misleading
PHRASE = [(r"\blicen[sc]e plates?\b", "vehicle"), (r"\bhot ?dogs?\b", "food"), (r"\bteddy bears?\b", "household and electronics")]


def build_label_map(stem) -> dict[str, str]:
    """Stemmed head noun -> category, using the same stemmer as mmav.site.head_noun."""
    to: dict[str, str] = {}
    for group, words in GROUPS.items():
        for w in words.split():
            forms = {stem(w), stem(w + "s")} | ({stem(w + "es")} if w.endswith(("s", "x", "ch", "sh")) else set())
            for f in forms:
                if to.get(f, group) != group:
                    raise ValueError(f"{f!r} maps to both {to[f]!r} and {group!r}")
                to[f] = group
    for w, group in IRREGULAR.items():
        to[stem(w)] = group
    return to


def label_of(phrase: str, head_noun, to: dict[str, str]) -> str | None:
    """The category of one region phrase, or None when its head noun is not in any category."""
    low = phrase.lower()
    for pattern, group in PHRASE:
        if re.search(pattern, low):
            return group
    return to.get(head_noun(phrase) or "")


CAP = 2500
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw" / "regions20.npz"))
    a = ap.parse_args()

    sys.path.insert(0, str(NLA_ROOT / "src"))
    from mmav.site import _stem, head_noun

    data = NLA_ROOT / "results" / "data"
    rows = [json.loads(line) for line in open(data / "vgreg_meta.jsonl")]
    demo = {s["image_id"] for d in json.loads((ROOT / "data/sample/manifest.json").read_text())["datasets"] for s in d["samples"]}
    to = build_label_map(_stem)
    names = list(GROUPS)
    labels = [label_of(r["phrase"], head_noun, to) for r in rows]

    by_class: dict[int, list[int]] = collections.defaultdict(list)
    for i, (r, k) in enumerate(zip(rows, labels)):
        if k is not None and r["image_id"] not in demo:
            by_class[names.index(k)].append(i)
    rng = np.random.default_rng(SEED)
    keep = []
    for c in range(len(names)):
        idx = np.array(by_class[c])
        keep.extend(rng.choice(idx, size=min(CAP, len(idx)), replace=False).tolist())
    keep = np.array(sorted(keep))
    y = np.array([names.index(labels[i]) for i in keep], dtype=np.int64)
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
    (ROOT / "backend" / "model" / "classes.json").write_text(json.dumps(
        {"classes": names, "groups": {k: v.split() for k, v in GROUPS.items()}, "irregular": IRREGULAR, "phrase_overrides": PHRASE}, indent=1))

    print(f"-> {out}: {len(keep)} regions, {len(uniq)} images, {X.shape[1]} dims")
    for s, name in enumerate(["train", "val", "test"]):
        m = split == s
        counts = np.bincount(y[m], minlength=len(names))
        print(f"  {name:5s} {m.sum():6d} regions, {len(np.unique(img[m])):5d} images, class sizes {counts.min()}..{counts.max()}")
    print("  overlap of images between splits:", len(set(img[split == 0]) & set(img[split == 2])), "(train/test)")


if __name__ == "__main__":
    main()
