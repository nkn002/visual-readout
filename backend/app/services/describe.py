"""Ask Qwen2.5-VL what a drawn region contains, through the model server in scripts/model_server.py.

Two images go to the model: the whole photo with the region outlined in red, and the region cropped
and enlarged. The answer is parsed into a short label and a one-sentence description. The model
server is a separate process on a GPU node; this module only prepares inputs and talks HTTP.
"""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from ..config import Settings

PROMPT = (
    "The first image is a photo with a red box drawn on it. The second image shows the inside of that box, enlarged. "
    "What is inside the red box? Answer in exactly this format:\n"
    "Label: <two to five words naming what the box contains>\n"
    "Description: <one sentence of at most 25 words about what the box contains and how it looks>"
)
MIN_CROP = 320  # px; tiny crops are enlarged so the model can see them
PAD = 0.04  # fraction of the box added on each side of the crop, for a little context


@dataclass
class Described:
    label: str
    description: str
    model: str
    raw: str


def model_url(settings: Settings) -> str | None:
    """MODEL_URL from the environment, else the file the model server writes when it starts."""
    if settings.model_url:
        return settings.model_url.rstrip("/")
    f = settings.model_url_file
    if f.exists():
        url = f.read_text().strip()
        return url.rstrip("/") or None
    return None


def model_health(url: str, timeout: float = 1.5) -> dict | None:
    try:
        r = httpx.get(f"{url}/health", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPError, ValueError):
        return None


def _jpeg_b64(im: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def render_inputs(image_path: str | Path, box: dict) -> tuple[str, str]:
    """Return (photo with the box drawn, enlarged crop) as base64 JPEGs. `box` is normalised 0..1."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        x0, y0 = box["x"] * w, box["y"] * h
        x1, y1 = x0 + box["w"] * w, y0 + box["h"] * h
        marked = im.copy()
        draw = ImageDraw.Draw(marked)
        stroke = max(2, round(min(w, h) / 150))
        draw.rectangle([x0, y0, x1, y1], outline=(230, 40, 40), width=stroke)
        px, py = (x1 - x0) * PAD, (y1 - y0) * PAD
        crop = im.crop((max(0, round(x0 - px)), max(0, round(y0 - py)), min(w, round(x1 + px)), min(h, round(y1 + py))))
        if crop.width < 2 or crop.height < 2:
            raise ValueError("the box is too small")
        scale = MIN_CROP / min(crop.width, crop.height)
        if scale > 1:
            crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
        return _jpeg_b64(marked), _jpeg_b64(crop)


def parse_answer(text: str) -> tuple[str, str]:
    label = re.search(r"^\s*\**label\**\s*:\s*(.+)$", text, re.I | re.M)
    desc = re.search(r"^\s*\**description\**\s*:\s*(.+)$", text, re.I | re.M)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    lab = (label.group(1) if label else (lines[0] if lines else "region")).strip().strip("*").strip(".").strip()
    des = (desc.group(1) if desc else " ".join(lines[1:] or lines)).strip().strip("*").strip()
    return lab, des


def describe_region(image_path: str | Path, box: dict, url: str, max_new_tokens: int = 96) -> Described:
    marked, crop = render_inputs(image_path, box)
    r = httpx.post(f"{url}/describe", json={"images": [marked, crop], "prompt": PROMPT, "max_new_tokens": max_new_tokens}, timeout=180)
    r.raise_for_status()
    body = r.json()
    raw = str(body.get("text", ""))
    label, description = parse_answer(raw)
    return Described(label=label, description=description, model=str(body.get("model", "")), raw=raw)
