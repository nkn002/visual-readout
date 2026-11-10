#!/usr/bin/env python
"""HW3: store the layer-21 image tokens under every box of the HW2 dataset, so a probe can attend over them.

HW2 fed the probe one vector per region: the mean of these tokens. HW3 needs the tokens themselves.
This script re-runs Qwen2.5-VL-7B exactly as the research extraction did (mmav.donor: prompt
"Describe this image.", bf16, hidden_states[21], mmav.vgregions.box_to_tokens with 50% cover) and keeps
every token under each box instead of averaging them.

Outputs (large, so under /scratch by default; never committed):
  tokens.npy    [T, 3584] float16, the tokens of all regions concatenated in data/raw/regions20.npz order
  index.npz     offsets [N+1] into tokens.npy, token_idx (flat grid index of each token), grid [N, 2] rows/cols,
                cos_to_stored [N] = cosine(mean of the extracted tokens, the stored HW2 vector) as a sanity check

    /home/rbc5xp/.conda/envs/nnedit/bin/python hw3/extract_tokens.py --out /scratch/rbc5xp/visual-readout
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NLA_ROOT = Path(os.environ.get("NLA_ROOT", "/standard/AikyamLab/khai/nla"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "raw" / "regions20.npz"))
    ap.add_argument("--out", default="/scratch/rbc5xp/visual-readout")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    a = ap.parse_args()

    sys.path.insert(0, str(NLA_ROOT / "src"))
    import torch
    from mmav.donor import DESCRIBE, LAYER, load_vl
    from mmav.vgregions import box_to_tokens

    z = np.load(a.data)
    rows, X = z["row"], z["X"]
    meta = [json.loads(line) for line in open(NLA_ROOT / "results" / "data" / "vgreg_meta.jsonl")]
    N = len(rows)
    n_expected = np.array([meta[r]["n_tokens"] for r in rows], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(n_expected)])
    T = int(offsets[-1])

    by_image: dict[int, list[int]] = defaultdict(list)
    for k, r in enumerate(rows):
        by_image[meta[r]["image_id"]].append(k)
    images = sorted(by_image)
    if a.limit:
        images = images[: a.limit]
    files = {int(p.stem): p for d in ("VG_100K", "VG_100K_2") for p in (NLA_ROOT / "results/data/vg_images" / d).glob("*.jpg")}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tokens = np.lib.format.open_memmap(out / "tokens.npy", mode="w+", dtype=np.float16, shape=(T, 3584))
    token_idx = np.full(T, -1, dtype=np.int32)
    grid = np.zeros((N, 2), dtype=np.int32)
    cos = np.full(N, np.nan, dtype=np.float32)
    mismatch = 0

    from PIL import Image

    model, proc = load_vl()
    proc.tokenizer.padding_side = "right"  # as in the research extraction
    img_tok = model.config.image_token_id
    t0 = time.time()
    for s in range(0, len(images), a.batch):
        ids = images[s:s + a.batch]
        imgs = [Image.open(files[i]).convert("RGB") for i in ids]
        texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": DESCRIBE}]}],
                                          tokenize=False, add_generation_prompt=True) for im in imgs]
        inp = proc(text=texts, images=imgs, padding=True, return_tensors="pt")
        grids = inp["image_grid_thw"]
        inp = inp.to(model.device)
        with torch.no_grad():
            h = model(**inp, output_hidden_states=True).hidden_states[LAYER]
        for n, iid in enumerate(ids):
            toks = h[n][inp.input_ids[n] == img_tok].float()
            g = grids[n].tolist()
            for k in by_image[iid]:
                idx, _ = box_to_tokens(meta[rows[k]]["box"], g, imgs[n].size)
                idx = idx[idx < len(toks)]
                want = int(n_expected[k])
                if len(idx) != want:
                    mismatch += 1
                    idx = idx[:want] if len(idx) > want else np.concatenate([idx, np.repeat(idx[-1:], want - len(idx))])
                v = toks[torch.as_tensor(idx, device=toks.device)].cpu().numpy()
                tokens[offsets[k]:offsets[k + 1]] = v.astype(np.float16)
                token_idx[offsets[k]:offsets[k + 1]] = idx
                grid[k] = (int(g[1]) // 2, int(g[2]) // 2)
                m, ref = v.mean(0), X[k].astype(np.float32)
                cos[k] = float(m @ ref / (np.linalg.norm(m) * np.linalg.norm(ref) + 1e-9))
        done = s + len(ids)
        if (s // a.batch) % 50 == 0 or done == len(images):
            el = time.time() - t0
            print(f"[tok] {done}/{len(images)} images, {el / done * 1000:.0f} ms/img, eta {el / done * (len(images) - done) / 60:.1f} min, "
                  f"cos to stored so far: median {np.nanmedian(cos):.4f} min {np.nanmin(cos):.4f}", flush=True)
    tokens.flush()
    np.savez(out / "index.npz", offsets=offsets, token_idx=token_idx, grid=grid, cos_to_stored=cos, rows=rows)
    done_mask = ~np.isnan(cos)
    print(f"-> {out}: {T} tokens for {done_mask.sum()} of {N} regions; token-count mismatches {mismatch}; "
          f"cos(mean of tokens, stored vector): median {np.median(cos[done_mask]):.4f}, 1st pct {np.percentile(cos[done_mask], 1):.4f}")


if __name__ == "__main__":
    main()
