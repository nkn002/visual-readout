#!/usr/bin/env python
"""Google Slides batchUpdate requests that redraw fig_mechanism as native, editable shapes.

    python docs/hw3/slide_assets/make_slide_requests.py <commit> > /tmp/requests.json
Geometry follows docs/hw3/latex/make_mechanism_figure.py (a 396 x 144 pt canvas), scaled to a 9 in wide
figure centered on a 10 x 5.625 in slide. Values (weights, probabilities) are the same computed ones.
Writes batch_1.json (frames, panel 1, arrows), batch_2.json (panel 2), batch_3.json (panel 3, groups).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in Path(os.environ.get("LAB_STYLE_DIR", Path.home() / ".claude/skills/synced")).glob("*/ai-conf-figures/scripts"):
    sys.path.insert(0, str(p))
from diagram_kit import Canvas  # noqa: E402  (text measurement only)

SLIDE = "p"
S = 9.0 / 5.5                         # canvas pt -> slide pt
X0, Y0 = 36.0, (405 - 144 * S) / 2
FONT, MONO = "Arial", "Roboto Mono"
INK, GRAY, STREAM = "#393939", "#8A8A8A", "#8A8A88"
TOK, BOX, OURS, GOLD = "#E8B931", "#3D85C6", "#CC7C5E", "#AA8B3E"
RAW = "https://raw.githubusercontent.com/nkn002/visual-readout/{}/docs/hw3/slide_assets/{}"
W = [0.249, 0.218, 0.121, 0.121, 0.172, 0.119]          # attention weights (make_mechanism_figure.py output)
TOP3 = [("person", 0.812), ("clothing and accessory", 0.103), ("head and face", 0.080)]
measure = Canvas(1, 1)


def X(x): return round(X0 + x * S, 2)
def Y(y): return round(Y0 + y * S, 2)
def L(v): return round(v * S, 2)


def rgb(h):
    h = h.lstrip("#")
    return {"red": round(int(h[0:2], 16) / 255, 3), "green": round(int(h[2:4], 16) / 255, 3), "blue": round(int(h[4:6], 16) / 255, 3)}


class Batch:
    def __init__(self):
        self.r, self.ids = [], []

    def _props(self, x, y, w, h, extra=None):
        t = {"scaleX": 1, "scaleY": 1, "translateX": x, "translateY": y, "unit": "PT"}
        t.update(extra or {})
        return {"pageObjectId": SLIDE, "size": {"width": {"magnitude": w, "unit": "PT"}, "height": {"magnitude": h, "unit": "PT"}},
                "transform": t}

    def shape(self, oid, kind, x, y, w, h, fill=None, alpha=None, line=None, weight=0.75, dash=None, transform=None):
        """x, y, w, h in canvas pt."""
        self.ids.append(oid)
        if transform:
            props = self._props(0, 0, L(w), L(h), transform)
        else:
            props = self._props(X(x), Y(y), L(w), L(h))
        self.r.append({"createShape": {"objectId": oid, "shapeType": kind, "elementProperties": props}})
        sp, f = {}, []
        if fill:
            sf = {"color": {"rgbColor": rgb(fill)}}
            if alpha is not None:
                sf["alpha"] = alpha
            sp["shapeBackgroundFill"] = {"solidFill": sf}
            f.append("shapeBackgroundFill.solidFill")
        else:
            sp["shapeBackgroundFill"] = {"propertyState": "NOT_RENDERED"}
            f.append("shapeBackgroundFill.propertyState")
        if line:
            sp["outline"] = {"outlineFill": {"solidFill": {"color": {"rgbColor": rgb(line)}}}, "weight": {"magnitude": weight, "unit": "PT"}}
            f += ["outline.outlineFill", "outline.weight"]
            if dash:
                sp["outline"]["dashStyle"] = dash
                f.append("outline.dashStyle")
        else:
            sp["outline"] = {"propertyState": "NOT_RENDERED"}
            f.append("outline.propertyState")
        self.r.append({"updateShapeProperties": {"objectId": oid, "shapeProperties": sp, "fields": ",".join(f)}})

    def text(self, oid, x, base, s, size, color=INK, anchor="start", italic=False, bold=False, mono=False,
             subs=(), runs=(), shape_in=None):
        """Text at canvas baseline (x, base) like diagram_kit.text, or inside an existing shape (shape_in)."""
        fs = round(size * S, 1)
        if shape_in is None:
            w_c = measure.text_width(s, size, bold=bold, italic=italic, mono=mono) * 1.12 + 2
            bw, bh = L(w_c) + 14.4, fs * 1.25 + 7.2
            cx = {"start": X(x) - 7.2, "middle": X(x) - bw / 2, "end": X(x) + 7.2 - bw}[anchor]
            cy = Y(base) - 3.6 - 0.93 * fs          # top-aligned text box: baseline = top + inset + ascent
            self.ids.append(oid)
            self.r.append({"createShape": {"objectId": oid, "shapeType": "TEXT_BOX",
                                           "elementProperties": self._props(round(cx, 2), round(cy, 2), round(bw, 2), round(bh, 2))}})
        else:
            oid = shape_in
        self.r.append({"insertText": {"objectId": oid, "text": s}})
        st = {"fontFamily": MONO if mono else FONT, "fontSize": {"magnitude": fs, "unit": "PT"},
              "foregroundColor": {"opaqueColor": {"rgbColor": rgb(color)}}}
        f = "fontFamily,fontSize,foregroundColor"
        if italic:
            st["italic"] = True; f += ",italic"
        if bold:
            st["bold"] = True; f += ",bold"
        self.r.append({"updateTextStyle": {"objectId": oid, "textRange": {"type": "ALL"}, "style": st, "fields": f}})
        for a, b in subs:
            self.r.append({"updateTextStyle": {"objectId": oid, "textRange": {"type": "FIXED_RANGE", "startIndex": a, "endIndex": b},
                                               "style": {"baselineOffset": "SUBSCRIPT"}, "fields": "baselineOffset"}})
        for a, b, style, fields in runs:
            self.r.append({"updateTextStyle": {"objectId": oid, "textRange": {"type": "FIXED_RANGE", "startIndex": a, "endIndex": b},
                                               "style": style, "fields": fields}})
        al = {"start": "START", "middle": "CENTER", "end": "END"}[anchor]
        if al != "START":
            self.r.append({"updateParagraphStyle": {"objectId": oid, "textRange": {"type": "ALL"}, "style": {"alignment": al},
                                                    "fields": "alignment"}})

    def line(self, oid, x1, y1, x2, y2, color=INK, weight=0.8, dash=None, arrow=False):
        self.ids.append(oid)
        w, h = max(abs(x2 - x1), 0.01), max(abs(y2 - y1), 0.01)
        self.r.append({"createLine": {"objectId": oid, "category": "STRAIGHT",
                                      "elementProperties": self._props(X(min(x1, x2)), Y(min(y1, y2)), L(w), L(h))}})
        lp = {"lineFill": {"solidFill": {"color": {"rgbColor": rgb(color)}}}, "weight": {"magnitude": round(weight * S, 2), "unit": "PT"}}
        f = ["lineFill", "weight"]
        if dash:
            lp["dashStyle"] = dash; f.append("dashStyle")
        if arrow:
            lp["endArrow"] = "FILL_ARROW"; f.append("endArrow")
        self.r.append({"updateLineProperties": {"objectId": oid, "lineProperties": lp, "fields": ",".join(f)}})

    def image(self, oid, url, x, y, w, h, outline=None, weight=0.8):
        self.ids.append(oid)
        self.r.append({"createImage": {"objectId": oid, "url": url, "elementProperties": self._props(X(x), Y(y), L(w), L(h))}})
        if outline:
            self.r.append({"updateImageProperties": {"objectId": oid, "imageProperties": {"outline": {
                "outlineFill": {"solidFill": {"color": {"rgbColor": rgb(outline)}}}, "weight": {"magnitude": round(weight * S, 2), "unit": "PT"}}},
                "fields": "outline.outlineFill,outline.weight"}})

    def panel(self, key, x, y, w, h, title, header_fill="#E8E8E8", title_color="#1A1A1A"):
        self.shape(f"{key}_frame", "RECTANGLE", x, y, w, h, fill="#F3F3F3", line="#9A9A9A", weight=0.6)
        self.shape(f"{key}_head", "RECTANGLE", x, y, w, 15, fill=header_fill, line="#9A9A9A", weight=0.6)
        self.text(f"{key}_title", x + w / 2, y + 7.5 + 7.5 * 0.35, title, 7.5, color=title_color, anchor="middle")
        self.line(f"{key}_rule", x, y + 15, x + w, y + 15, color="#9A9A9A", weight=0.6)

    def group(self, oid, children):
        self.r.append({"groupObjects": {"groupObjectId": oid, "childrenObjectIds": children}})


def main() -> None:
    commit = sys.argv[1]
    url = lambda f: RAW.format(commit, f)
    # ---------------- batch 1: panel 1 and the two block arrows ----------------
    b1 = Batch()
    b1.panel("fa", 2, 2, 96, 141, "Image and box")
    ix, iy, iw, ih = 15.3, 22.0, 69.4, 84.0
    b1.image("fa_img", url("crop.png"), ix, iy, iw, ih, outline="#777777", weight=0.4)
    s = iw / 190
    cw, ch = 405 / 14, 500 / 18
    for ti in (20, 21, 34, 35, 48, 49):
        r, c = divmod(ti, 14)
        b1.shape(f"fa_tok{ti}", "RECTANGLE", ix + (c * cw - 110) * s, iy + r * ch * s, cw * s, ch * s, fill=TOK, alpha=0.28, line=TOK, weight=0.8)
    b1.shape("fa_box", "RECTANGLE", ix + (178 - 110) * s, iy + 32 * s, 52 * s, 71 * s, line=BOX, weight=1.4)
    b1.text("fa_quote1", 7, 116, "“The man is wearing a", 7, italic=True)
    b1.text("fa_quote2", 7, 125, "baseball cap”", 7, italic=True)
    b1.text("fa_label", 7, 135.5, "label: person", 7, runs=[(0, 6, {"foregroundColor": {"opaqueColor": {"rgbColor": rgb(GRAY)}}}, "foregroundColor")])
    for k, (xa, xb) in enumerate([(100, 112), (268, 280)]):
        b1.shape(f"arrow{k + 1}", "RIGHT_ARROW", xa, (70 if k == 0 else 64) - 6.6, xb - xa, 13.3, fill="#111111")
    g1 = list(b1.ids)

    # ---------------- batch 2: panel 2, the frozen VLM ----------------
    b2 = Batch()
    b2.panel("fb", 114, 2, 152, 141, "Qwen2.5-VL-7B (frozen)")
    lab_x, rows = 148, {"layer 28": 34, "layer 20": 64, "layer 1": 94}
    xs = [158, 172, 186, 200, 214, 228]
    show = [19, 20, 21, None, 49, 50]
    xt, y_in = 250, 108
    b2.shape("fb_band", "ROUND_RECTANGLE", lab_x + 3, 57.5, 111, 13, fill="#F4EFD9")
    for name, yy in rows.items():
        b2.text("fb_lab" + name.split()[1], lab_x, yy + 2.4, name, 7, color=GRAY, anchor="end")
    b2.text("fb_labin", lab_x, y_in + 8.6, "input", 7, color=GRAY, anchor="end")
    cols = [x for x, p in zip(xs, show) if p is not None] + [xt]
    for k, xx in enumerate(cols):
        b2.line(f"fb_col{k}", xx, 34, xx, y_in, color=STREAM, weight=0.7)
    for k, xx in enumerate(cols):
        p = (show[xs.index(xx)] if xx in xs else None)
        for name, yy in rows.items():
            hit = name == "layer 20" and p in (20, 21, 49)
            r = 3.0 if hit else 2.6
            b2.shape(f"fb_n{k}_{name.split()[1]}", "ELLIPSE", xx - r, yy - r, 2 * r, 2 * r,
                     fill=TOK if hit else "#FFFFFF", line="#7A5A00" if hit else STREAM, weight=0.9 if hit else 0.8)
    for name, yy in list(rows.items()) + [("in", y_in + 6)]:
        b2.text("fb_gap" + name.split()[-1], 200, yy + 2.4, "…", 7, color=STREAM, anchor="middle")
    for p, xx in zip(show, xs):
        if p is None:
            continue
        inside = p in (20, 21, 49)
        b2.image(f"fb_patch{p}", url(f"cell_{p}.png"), xx - 6, y_in, 12, 12, outline=TOK if inside else "#999999", weight=1.4 if inside else 0.5)
    b2.shape("fb_text", "ROUND_RECTANGLE", xt - 6, y_in, 12, 12, fill="#E6E6E6", line="#999999", weight=0.5)
    b2.text(None, 0, 0, "T", 7, mono=True, anchor="middle", shape_in="fb_text")
    b2.r.append({"updateShapeProperties": {"objectId": "fb_text", "shapeProperties": {"contentAlignment": "MIDDLE"}, "fields": "contentAlignment"}})
    b2.text("fb_note1", 120, 128.8, "image tokens (vision encoder) + prompt", 7, color=GRAY)
    b2.text("fb_note2", 120, 137, "Describe this image.", 7, mono=True)
    g2 = list(b2.ids)

    # ---------------- batch 3: panel 3, the trained probe ----------------
    b3 = Batch()
    b3.panel("fc", 282, 2, 112, 141, "Region probe (trained)", header_fill=OURS, title_color="#FFFFFF")
    b3.text("fc_xlab", 288, 25.8, "xi = hidden state 21, 3,584-d", 7, color=GRAY,
            subs=[(1, 2)], runs=[(0, 2, {"foregroundColor": {"opaqueColor": {"rgbColor": rgb(INK)}}, "italic": True,
                                         "fontFamily": "Times New Roman"}, "foregroundColor,italic,fontFamily")])
    sx = [294 + i * 9 for i in range(6)]
    greys = ["#D8D8D8", "#9E9E9E", "#C4C4C4", "#7C7C7C", "#E4E4E4"]
    for i, xx in enumerate(sx):
        for j in range(3):
            b3.shape(f"fc_v{i}_{j}", "RECTANGLE", xx - 2.5, 32 + j * 5, 5, 5, fill=greys[(i * 2 + j) % 5])
        b3.shape(f"fc_vo{i}", "RECTANGLE", xx - 2.5, 32, 5, 15, line=TOK, weight=0.8)
    base, unit = 80, 24 / max(W)
    for i, (xx, wi) in enumerate(zip(sx, W)):
        b3.shape(f"fc_bar{i}", "RECTANGLE", xx - 3, base - wi * unit, 6, wi * unit, fill=OURS)
    b3.line("fc_base", sx[0] - 5, base, sx[-1] + 5, base, weight=0.6)
    yu = base - unit / 6
    b3.line("fc_unif", sx[0] - 5, yu, sx[-1] + 5, yu, weight=0.7, dash="DOT")
    b3.text("fc_unif_lab", sx[-1] + 8, yu + 2.4, "1/L (HW2)", 7)
    b3.text("fc_ai", sx[-1] + 8, base - max(W) * unit + 1, "ai learned", 7, color=OURS, subs=[(1, 2)],
            runs=[(0, 2, {"italic": True, "fontFamily": "Times New Roman"}, "italic,fontFamily")])
    yf = base + 13
    b3.text("fc_sum", 288, yf + 3, "∑i ai xi", 8, subs=[(1, 2), (4, 5), (7, 8)],
            runs=[(0, 8, {"italic": True, "fontFamily": "Times New Roman"}, "italic,fontFamily")])
    b3.line("fc_arr1", 318, yf, 330, yf, weight=0.9, arrow=True)
    b3.shape("fc_mlp", "TRAPEZOID", 0, 0, 18, 30, fill="#FCF9F1", line=GOLD, weight=1.2,
             transform={"scaleX": 0, "scaleY": 0, "shearX": -1, "shearY": 1, "translateX": X(332) + L(30), "translateY": Y(yf - 9)})
    b3.text("fc_mlp_t", 347, yf + 2.6, "MLP", 7.5, color=GOLD, bold=True, anchor="middle")
    b3.line("fc_arr2", 363, yf, 374, yf, weight=0.9, arrow=True)
    b3.text("fc_20", 376, yf + 2.4, "20", 7)
    yb = yf + 13
    for k, (name, pv) in enumerate(TOP3):
        yy = yb + k * 10.5
        b3.shape(f"fc_p{k}", "RECTANGLE", 288, yy, max(1.5, 78 * pv), 8.5, fill="#EAC38D")
        b3.text(f"fc_pl{k}", 290, yy + 4.2, name, 7)
        b3.text(f"fc_pv{k}", 388, yy + 4.2, f"{pv:.2f}", 7, anchor="end")
    g3 = list(b3.ids)
    b3.group("grp_panel1", g1[:-2])
    b3.group("grp_panel2", g2)
    b3.group("grp_panel3", g3)

    for k, b in enumerate((b1, b2, b3), 1):
        (HERE / f"batch_{k}.json").write_text(json.dumps(b.r, separators=(",", ":")))
        print(f"batch_{k}: {len(b.r)} requests, {len(json.dumps(b.r, separators=(',', ':'))) / 1000:.1f} KB, {len(b.ids)} elements")


if __name__ == "__main__":
    main()
