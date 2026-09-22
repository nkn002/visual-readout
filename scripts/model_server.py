#!/usr/bin/env python
"""Qwen2.5-VL-7B-Instruct behind a tiny HTTP server, for the viewer's "Ask the model" button.

Runs on a GPU node with the research conda env (torch, transformers, qwen-vl-utils):
    sbatch scripts/model_server.sbatch          # or, on a GPU node:
    /home/rbc5xp/.conda/envs/nnedit/bin/python scripts/model_server.py --port 8100

API (JSON):
    GET  /health              -> {"ready": true, "model": "...", "device": "cuda:0"}
    POST /describe            {"images": ["<base64 jpeg/png>", ...], "prompt": "...", "max_new_tokens": 96}
                              -> {"text": "...", "model": "...", "seconds": 1.2}
    POST /activations         {"image": "<base64>", "box": {"x","y","w","h"} (0..1), "layer": 21}
                              -> {"layer", "dim", "reduced": [64 floats], "explained": [...], "basis", "norm",
                                  "top": [{"dim", "value"}...], "region_tokens": [...], "rows", "cols"}
                              The region's activation: the tokens under the box, chosen with the research
                              repo's rule (mmav.vgregions.box_to_tokens, 50% cover), mean-pooled at
                              hidden_states[layer] into one 3584-d vector, exactly the connector's input,
                              then projected onto 64 principal components fitted once on 20,000 of the
                              research repo's own region vectors (results/data/vgreg_V.npy) and cached in
                              --pca-file. hidden_states[21] is the read-out layer L20.

On start it writes its URL to --url-file (default data/db/model_server.url) so the backend finds it.
Loading and prompting follow mmav.donor.load_vl in the nla repo: bf16, device_map cuda, chat template
with image entries, processor(text=[...], images=[...]). Standard library HTTP, no extra deps.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VL = "Qwen/Qwen2.5-VL-7B-Instruct"
NLA_ROOT = Path(os.environ.get("NLA_ROOT", "/standard/AikyamLab/khai/nla"))


def _box_to_tokens():
    """Prefer the research repo's own function; fall back to a faithful copy if mmav is not importable."""
    try:
        import sys

        sys.path.insert(0, str(NLA_ROOT / "src"))
        from mmav.vgregions import box_to_tokens  # noqa: WPS433

        return box_to_tokens, "mmav.vgregions.box_to_tokens"
    except Exception:  # noqa: BLE001
        import numpy as np

        PATCH, MERGE = 14, 2
        CELL = PATCH * MERGE

        def box_to_tokens(box, grid_thw, orig_wh, min_cover: float = 0.5):
            _, gh, gw = [int(x) for x in grid_thw]
            rows, cols = gh // MERGE, gw // MERGE
            rw, rh = gw * PATCH, gh * PATCH
            ow, oh = orig_wh
            sx, sy = rw / ow, rh / oh
            x0, y0, bw, bh = box
            bx0, bx1 = max(0.0, x0 * sx), min(float(rw), (x0 + bw) * sx)
            by0, by1 = max(0.0, y0 * sy), min(float(rh), (y0 + bh) * sy)
            if bx1 <= bx0 or by1 <= by0:
                return np.empty(0, dtype=np.int64), 0.0
            c0, c1 = max(0, int(np.floor(bx0 / CELL))), min(cols, int(np.ceil(bx1 / CELL)))
            r0, r1 = max(0, int(np.floor(by0 / CELL))), min(rows, int(np.ceil(by1 / CELL)))
            keep = []
            for r in range(r0, r1):
                oy = min(by1, (r + 1) * CELL) - max(by0, r * CELL)
                for c in range(c0, c1):
                    ox = min(bx1, (c + 1) * CELL) - max(bx0, c * CELL)
                    if ox * oy >= min_cover * CELL * CELL:
                        keep.append(r * cols + c)
            if not keep:
                cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
                keep = [min(rows - 1, int(cy // CELL)) * cols + min(cols - 1, int(cx // CELL))]
            idx = np.array(sorted(keep), dtype=np.int64)
            return idx, float(len(idx) / (rows * cols))

        return box_to_tokens, "local copy of mmav.vgregions.box_to_tokens"


BOX_TO_TOKENS, BOX_TO_TOKENS_SRC = _box_to_tokens()
PCA_K = 64


def load_or_fit_pca(pca_file: Path, source: Path, n_rows: int = 20000, seed: int = 0):
    """(mean, components [K, d], explained variance ratio [K], description). Fit once, cache to pca_file."""
    import numpy as np

    if pca_file.exists():
        z = np.load(pca_file)
        return z["mean"], z["components"], z["explained"], str(z["basis"])
    if not source.exists():
        return None
    V = np.load(source, mmap_mode="r")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(V.shape[0], size=min(n_rows, V.shape[0]), replace=False))
    X = np.asarray(V[rows], dtype=np.float32)
    mean = X.mean(axis=0)
    Xc = X - mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2 / (len(X) - 1)
    explained = (var / var.sum())[:PCA_K]
    components = Vt[:PCA_K]
    basis = f"PCA-{PCA_K} fitted on {len(rows)} region vectors from {source.name} (seed {seed})"
    pca_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(pca_file, mean=mean, components=components, explained=explained, basis=basis)
    return mean, components, explained, basis


class Model:
    def __init__(self, model_id: str, device_map: str = "cuda", pca=None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.model_id = model_id
        self.pca = pca  # (mean, components, explained, basis) or None
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device_map).eval()
        self.lock = threading.Lock()

    def describe(self, images, prompt: str, max_new_tokens: int = 96) -> str:
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=images, padding=True, return_tensors="pt").to(self.model.device)
        with self.lock, self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inp.input_ids.shape[1] :], skip_special_tokens=True)[0].strip()

    def activations(self, image, box: dict, layer: int = 21, prompt: str = "Describe this image.") -> dict:
        """The region's activation vector: mean-pool the tokens under `box` (normalised 0..1) at hidden_states[layer]."""
        content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device)
        with self.lock, self.torch.no_grad():
            out = self.model(**inp, output_hidden_states=True)
        h = out.hidden_states[layer][0]
        pos = inp.input_ids[0] == self.model.config.image_token_id
        toks = h[pos].float()  # [T, d], raster order over the merged token grid
        grid_thw = inp.image_grid_thw[0].tolist()
        _, gh, gw = [int(x) for x in grid_thw]
        rows, cols = gh // 2, gw // 2
        if rows * cols != toks.shape[0]:
            raise RuntimeError(f"token grid {rows}x{cols} != {toks.shape[0]} image tokens")
        ow, oh = image.size
        px = (box["x"] * ow, box["y"] * oh, box["w"] * ow, box["h"] * oh)
        idx, frac = BOX_TO_TOKENS(px, grid_thw, (ow, oh))
        idx = [int(i) for i in idx]
        region = toks[idx].mean(dim=0)  # the connector's input: one mean-pooled vector
        v = region.cpu().numpy()
        if self.pca is not None:
            mean, comps, explained, basis = self.pca
        else:  # no research vectors at hand: fit the basis on this image's own tokens
            import numpy as np

            T = toks.cpu().numpy()
            mean = T.mean(axis=0)
            _, S, Vt = np.linalg.svd(T - mean, full_matrices=False)
            var = S**2 / max(1, len(T) - 1)
            k = min(PCA_K, len(S))
            comps, explained = Vt[:k], (var / var.sum())[:k]
            basis = f"PCA-{k} fitted on this image's {len(T)} tokens (no research vectors found)"
        reduced = (v - mean) @ comps.T
        order = sorted(range(len(v)), key=lambda i: -abs(float(v[i])))[:8]
        return {
            "layer": layer, "dim": int(v.shape[0]),
            "reduced": [round(float(x), 4) for x in reduced.tolist()],
            "explained": [round(float(x), 5) for x in explained.tolist()],
            "basis": basis,
            "norm": float(region.norm()),
            "top": [{"dim": int(i), "value": round(float(v[i]), 3)} for i in order],
            "region_tokens": idx, "rows": rows, "cols": cols, "grid_frac": float(frac),
            "pooling": BOX_TO_TOKENS_SRC, "n_layers": len(out.hidden_states) - 1,
        }


def make_handler(model: Model):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._json(200, {"ready": True, "model": model.model_id, "device": str(model.model.device)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.rstrip("/")
            if path not in ("/describe", "/activations"):
                return self._json(404, {"error": "not found"})
            try:
                from PIL import Image

                n = int(self.headers.get("content-length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                if path == "/activations":
                    box = req.get("box") or {}
                    if not req.get("image") or not all(k in box for k in ("x", "y", "w", "h")):
                        return self._json(400, {"error": "need image and box {x, y, w, h}"})
                    image = Image.open(io.BytesIO(base64.b64decode(req["image"]))).convert("RGB")
                    t0 = time.time()
                    grid = model.activations(image, {k: float(box[k]) for k in ("x", "y", "w", "h")}, int(req.get("layer", 21)))
                    return self._json(200, {**grid, "model": model.model_id, "seconds": round(time.time() - t0, 2)})
                images = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in req.get("images", [])]
                prompt = str(req.get("prompt", "Describe this image."))
                if not images:
                    return self._json(400, {"error": "no images"})
                t0 = time.time()
                text = model.describe(images, prompt, int(req.get("max_new_tokens", 96)))
                self._json(200, {"text": text, "model": model.model_id, "seconds": round(time.time() - t0, 2)})
            except Exception as e:  # noqa: BLE001  report to the caller, keep serving
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=VL)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--url-file", default=str(Path(__file__).resolve().parents[1] / "data" / "db" / "model_server.url"))
    ap.add_argument("--pca-file", default=str(Path(__file__).resolve().parents[1] / "data" / "db" / "pca64.npz"))
    ap.add_argument("--pca-source", default=str(NLA_ROOT / "results" / "data" / "vgreg_V.npy"))
    a = ap.parse_args()

    t0 = time.time()
    pca = load_or_fit_pca(Path(a.pca_file), Path(a.pca_source))
    print(f"pca basis: {pca[3] if pca else 'none (will fit per image)'} ({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time()
    print(f"loading {a.model} ...", flush=True)
    model = Model(a.model, pca=pca)
    print(f"loaded in {time.time() - t0:.0f}s on {model.model.device}", flush=True)

    url = f"http://{socket.gethostname()}:{a.port}"
    Path(a.url_file).parent.mkdir(parents=True, exist_ok=True)
    Path(a.url_file).write_text(url + "\n")
    print(f"serving {url}  (url written to {a.url_file})", flush=True)
    try:
        ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(model)).serve_forever()
    finally:
        Path(a.url_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
