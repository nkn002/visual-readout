#!/usr/bin/env python
"""Report figure: how the region probe reads Qwen2.5-VL's hidden state (one real test region).

    CUDA_VISIBLE_DEVICES="" python docs/hw3/latex/make_mechanism_figure.py
Writes docs/hw3/latex/fig_mechanism.{pdf,svg,png} at the COLM text width (5.5 in).
The attention weights and probabilities are computed here from results/hw3/best_attn1.pt
(attention 1 head, 1 hidden layer, seed 0) on test region 32332; nothing is typed in by hand.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NLA = Path(os.environ.get("NLA_ROOT", "/standard/AikyamLab/khai/nla"))
TOKENS = Path(os.environ.get("TOKENS_DIR", "/scratch/rbc5xp/visual-readout"))
for p in Path(os.environ.get("LAB_STYLE_DIR", Path.home() / ".claude/skills/synced")).glob("*/ai-conf-figures/scripts"):
    sys.path.insert(0, str(p))
from diagram_kit import IP, SL, Canvas  # noqa: E402

sys.path.insert(0, str(ROOT / "training"))
from attention_probe import RegionProbe  # noqa: E402

K = 32332                       # test region: "The man is wearing a baseball cap", label person
CROP = (110, 0, 300, 230)       # image crop (px) around the box
BOX_C = "#3D85C6"               # the box
TOK_C = "#E8B931"               # tokens under the box, in every panel
OURS = IP.clay                  # the trained probe


def example():
    ck = torch.load(ROOT / "results/hw3/best_attn1.pt", map_location="cpu", weights_only=False)
    c = ck["config"]
    m = RegionProbe(c["d"], c["n_classes"], c["pooling"], c["heads"], c["key_dim"], tuple(c["hidden"]),
                    c["activation"], c["dropout"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    z = np.load(ROOT / "data/raw/regions20.npz")
    idx = np.load(TOKENS / "index.npz")
    tok = np.load(TOKENS / "tokens.npy", mmap_mode="r")
    classes = json.loads((ROOT / "backend/model/classes.json").read_text())["classes"]
    meta = json.loads(open(NLA / "results/data/vgreg_meta.jsonl").readlines()[int(z["row"][K])])
    off = idx["offsets"]
    t = torch.tensor(np.asarray(tok[off[K]:off[K + 1]], dtype=np.float32))
    with torch.no_grad():
        pooled, w = m.pool(t, torch.zeros(len(t), dtype=torch.long), 1)
        prob = torch.softmax(m.forward_pooled(pooled), -1)[0]
    top = torch.topk(prob, 3)
    return dict(meta=meta, label=classes[int(z["y"][K])], token_idx=idx["token_idx"][off[K]:off[K + 1]].tolist(),
                grid=idx["grid"][K].tolist(), w=w[:, 0].tolist(),
                top=[(classes[int(i)], float(v)) for v, i in zip(top.values, top.indices)])


def dots(c, x, y, vertical=True, color=IP.stream):
    """Three small dots: an ellipsis that needs no glyph from the font."""
    for d in (-3, 0, 3):
        cx, cy = (x, y + d) if vertical else (x + d, y)
        c.raw(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="0.7" fill="{color}"/>')


def image_path(iid: int) -> Path:
    return next(NLA / "results/data/vg_images" / d / f"{iid}.jpg" for d in ("VG_100K", "VG_100K_2")
                if (NLA / "results/data/vg_images" / d / f"{iid}.jpg").exists())


def main() -> None:
    ex = example()
    meta, rows, cols = ex["meta"], *ex["grid"]
    im = Image.open(image_path(meta["image_id"])).convert("RGB")
    W, H = im.size
    cw, ch = W / cols, H / rows
    work = HERE / "_mech"
    work.mkdir(exist_ok=True)
    im.crop(CROP).save(work / "crop.png")

    c = Canvas(width_in=5.5, height_in=2.0)
    top, ph = 2, 141
    # ---------------- panel 1: image and box ----------------
    x1, w1 = 2, 96
    y0 = c.panel(x1, top, w1, ph, "Image and box", size=7.5)
    ix, iy, iw, ih = c.image(x1 + 6, y0 + 2, w1 - 12, 84, str(work / "crop.png"), border="#777777", bw=0.4)
    s = iw / (CROP[2] - CROP[0])
    for ti in ex["token_idx"]:
        r, cc = divmod(int(ti), cols)
        c.raw(f'<rect x="{ix + (cc * cw - CROP[0]) * s:.2f}" y="{iy + (r * ch - CROP[1]) * s:.2f}" '
              f'width="{cw * s:.2f}" height="{ch * s:.2f}" fill="{TOK_C}" fill-opacity="0.28" '
              f'stroke="{TOK_C}" stroke-width="0.8"/>')
    bx, by, bw, bh = meta["box"]
    c.region(ix + (bx - CROP[0]) * s, iy + (by - CROP[1]) * s, bw * s, bh * s, color=BOX_C, sw=1.4)
    yt = iy + ih + 4
    c.rich(x1 + 5, yt, [("“The man is wearing a baseball cap”", {"italic": True})], size=7, max_w=w1 - 10)
    c.rich(x1 + 5, yt + 19, [("label: ", {"color": IP.gray_text}), (ex["label"], {})], size=7, max_w=w1 - 10)  # noqa
    c.block_arrow(x1 + w1 + 2, 70, x1 + w1 + 14, width=7, head=7)

    # ---------------- panel 2: frozen Qwen2.5-VL ----------------
    x2, w2 = 114, 152
    y0 = c.panel(x2, top, w2, ph, "Qwen2.5-VL-7B (frozen)", size=7.5)
    lab_x = x2 + 34                          # right edge of the layer labels
    show = [19, 20, 21, None, 49, 50]        # image-token positions drawn (None = gap)
    xs = [lab_x + 10 + i * 14 for i in range(len(show))]
    xt = xs[-1] + 22                         # text-token column
    rows_y = {"layer 28": y0 + 14, "layer 20": y0 + 44, "layer 1": y0 + 74}
    dots_y = [y0 + 29, y0 + 59]
    y_in = y0 + 88
    c.box(lab_x + 3, rows_y["layer 20"] - 6.5, x2 + w2 - lab_x - 7, 13, fill="#F4EFD9", r=2)
    for name, yy in rows_y.items():
        c.text(lab_x, yy + 2.4, name, size=7, color=IP.gray_text, anchor="end")
    for xx in [x for x, p in zip(xs, show) if p is not None] + [xt]:
        c.line(xx, rows_y["layer 28"], xx, y_in, color=IP.stream, sw=0.7)
        for yy in rows_y.values():
            c.raw(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="2.6" fill="white" stroke="{IP.stream}" stroke-width="0.8"/>')
        for yy in dots_y:
            dots(c, xx, yy)
    for xx, p in zip(xs, show):
        if p is None:
            for yy in list(rows_y.values()) + [y_in + 6]:
                dots(c, xx, yy, vertical=False)
            continue
        r, cc = divmod(p, cols)
        cell = im.crop((int(cc * cw), int(r * ch), int((cc + 1) * cw), int((r + 1) * ch)))
        cell.save(work / f"cell_{p}.png")
        inside = p in ex["token_idx"]
        c.image(xx - 6, y_in, 12, 12, str(work / f"cell_{p}.png"), fit=False,
                border=TOK_C if inside else "#999999", bw=1.4 if inside else 0.5)
        if inside:
            c.raw(f'<circle cx="{xx:.2f}" cy="{rows_y["layer 20"]:.2f}" r="3.0" fill="{TOK_C}" '
                  f'stroke="#7A5A00" stroke-width="0.9"/>')
    c.box(xt - 6, y_in, 12, 12, fill="#E6E6E6", stroke="#999999", sw=0.5, r=1.5)
    c.text(xt, y_in + 8.6, "T", size=7, color=IP.ink, anchor="middle", mono=True)
    c.text(lab_x, y_in + 8.6, "input", size=7, color=IP.gray_text, anchor="end")
    c.text(x2 + 6, y_in + 23, "image tokens (vision encoder) + prompt", size=7, color=IP.gray_text)
    c.text(x2 + 6, y_in + 32, "Describe this image.", size=7, color=IP.ink, mono=True)
    c.block_arrow(x2 + w2 + 2, rows_y["layer 20"], x2 + w2 + 14, width=7, head=7)

    # ---------------- panel 3: the trained probe ----------------
    x3, w3 = 282, 112
    y0 = c.panel(x3, top, w3, ph, "Region probe (trained)", size=7.5, header_fill=OURS, title_color="white")
    w, L = ex["w"], len(ex["w"])
    sx = [x3 + 12 + i * 9 for i in range(L)]
    xw = c.mathtext(x3 + 6, y0 + 7, "x_{i}", size=8, color=IP.ink)
    c.text(x3 + 8 + xw, y0 + 7, "= hidden state 21, 3,584-d", size=7, color=IP.gray_text)
    for xx in sx:
        c.vector(xx - 2.5, y0 + 12, 5, 15, outline=TOK_C, sw=0.8, n=9, seed=int(xx))
    base, unit = y0 + 60, 24 / max(w)
    for xx, wi in zip(sx, w):
        c.box(xx - 3, base - wi * unit, 6, wi * unit, fill=OURS, r=0.6)
    c.line(sx[0] - 5, base, sx[-1] + 5, base, color=IP.ink, sw=0.6)
    yu = base - unit / L
    c.line(sx[0] - 5, yu, sx[-1] + 5, yu, color=IP.ink, sw=0.7, dash="1.6,1.4")
    c.text(sx[-1] + 8, yu + 2.4, "1/L (HW2)", size=7, color=IP.ink)
    aw = c.mathtext(sx[-1] + 8, base - max(w) * unit + 1, "a_{i}", size=8, color=OURS)
    c.text(sx[-1] + 10 + aw, base - max(w) * unit + 1, "learned", size=7, color=OURS)
    yf = base + 13
    c.mathtext(x3 + 6, yf + 3, "\u2211_{i} a_{i} x_{i}", size=8)
    c.arrow(x3 + 36, yf, x3 + 48, yf, color=IP.ink, sw=0.9, head=4)
    c.trapezoid(x3 + 50, yf - 9, 30, 18, narrow="right", text="MLP", size=7.5)
    c.arrow(x3 + 81, yf, x3 + 92, yf, color=IP.ink, sw=0.9, head=4)
    c.text(x3 + 94, yf + 2.4, "20", size=7, color=IP.ink)
    yb = yf + 13
    for k, (name, pv) in enumerate(ex["top"]):
        yy = yb + k * 10.5
        c.box(x3 + 6, yy, max(1.5, (w3 - 34) * pv), 8.5, fill=IP.readout, r=1)
        c.text(x3 + 8, yy + 6.3, name, size=7, color=IP.ink)
        c.text(x3 + w3 - 6, yy + 6.3, f"{pv:.2f}", size=7, color=IP.ink, anchor="end")
    c.save(str(HERE / "fig_mechanism"))
    print("weights", [round(v, 3) for v in w], "top", [(n, round(v, 3)) for n, v in ex["top"]])


if __name__ == "__main__":
    main()
