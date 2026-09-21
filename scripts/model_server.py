#!/usr/bin/env python
"""Qwen2.5-VL-7B-Instruct behind a tiny HTTP server, for the viewer's "Ask the model" button.

Runs on a GPU node with the research conda env (torch, transformers, qwen-vl-utils):
    sbatch scripts/model_server.sbatch          # or, on a GPU node:
    /home/rbc5xp/.conda/envs/nnedit/bin/python scripts/model_server.py --port 8100

API (JSON):
    GET  /health              -> {"ready": true, "model": "...", "device": "cuda:0"}
    POST /describe            {"images": ["<base64 jpeg/png>", ...], "prompt": "...", "max_new_tokens": 96}
                              -> {"text": "...", "model": "...", "seconds": 1.2}

On start it writes its URL to --url-file (default data/db/model_server.url) so the backend finds it.
Loading and prompting follow mmav.donor.load_vl in the nla repo: bf16, device_map cuda, chat template
with image entries, processor(text=[...], images=[...]). Standard library HTTP, no extra deps.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VL = "Qwen/Qwen2.5-VL-7B-Instruct"


class Model:
    def __init__(self, model_id: str, device_map: str = "cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.model_id = model_id
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
            if self.path.rstrip("/") != "/describe":
                return self._json(404, {"error": "not found"})
            try:
                from PIL import Image

                n = int(self.headers.get("content-length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
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
    a = ap.parse_args()

    t0 = time.time()
    print(f"loading {a.model} ...", flush=True)
    model = Model(a.model)
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
