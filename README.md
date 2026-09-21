# Visual read-out

Browse what a natural-language activation verbalizer reads out of a vision-language model for one image region, and how that read-out changes when the region is edited.

Live: **https://nkn002.github.io/visual-readout/** · Course: DS7400 Deep Learning, Homework 1.

## What it shows

The research project maps Qwen2.5-VL region activations into a frozen text verbalizer (a natural language autoencoder) through a small trained connector. For each Visual Genome region the viewer shows:

- the plain image, with zoom and pan, on which you draw your own region;
- a **Default region** button that reveals the region the dataset defined for that image, together with the frozen verbalizer's read-out for it and an LLM judge's verdict on whether the read-out names the object;
- your own names-it / misses-it verdict on that read-out.

HW1 keeps the interface to that. The data file also carries four edited conditions of each image (content swap, context swap, box blank) and four trained connector arms, which later homeworks will surface.

**Data type:** images plus boxes plus text. A sample is one region: `image_id`, `box [x, y, w, h]`, the VG phrase, four image files, per-arm read-outs, and judge verdicts. The committed sample is 12 regions × 4 conditions (48 JPEGs, 3.3 MB) exported from the private research repo.

## Run it

Static mode (what the live page does; no backend):

```bash
python backend/app/services/scan.py --data data/sample --out frontend/public/data   # stage the sample
cd frontend && npm ci && npm run dev                                                # http://localhost:5173
```

Backend mode (FastAPI + SQLite; annotations persist in a database):

```bash
cd backend && python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn app.main:app --reload            # http://localhost:8000, docs at /docs
```

Then in the app open **Settings** (the pill in the top bar), choose **Backend**, and check the connection. In local dev the Vite proxy forwards `/api` to port 8000, so the URL can stay empty. From the public page use `http://localhost:8000`; Chrome asks once for permission to reach localhost.

Docker (both services, hot reload): `docker compose up`, then open http://localhost:5173.

Tests: `cd backend && .venv/bin/python -m pytest` (8 tests: indexing, endpoints, annotation round trip, import, manifest validation).

## Features

| Feature | Where |
|---|---|
| Image list with thumbnails, arrow-key navigation | `frontend/src/components/SampleList.tsx` |
| Image viewer with zoom, pan, and the box overlay | `frontend/src/features/viewer/RegionViewer.tsx` |
| Default region toggle and drawing tool | `frontend/src/components/StageBar.tsx` |
| Your regions, the default region's read-out with the judge's verdict | `frontend/src/components/Inspector.tsx` |
| **Annotation:** drag to draw labelled boxes on the image (default tool), give your own names-it / misses-it verdict on the read-out, delete, export and import JSON | `features/annotate/BoxTool.tsx`, `MentionTags.tsx` |
| **Backend:** FastAPI + SQLModel + SQLite; datasets, samples, file serving, annotations CRUD, bulk import | `backend/app/` |
| Two data modes behind one interface: static manifest + localStorage, or the API | `frontend/src/api/` |
| GitHub Pages deploy on every push to `main` | `.github/workflows/pages.yml` |

Implemented annotation: **yes**. Implemented backend/database: **yes** (SQLite via SQLModel; the schema is created at startup, migrations are a follow-up).

## How it is put together

```
visual-readout/
├── data/sample/                 manifest.json + images/, committed, shipped to Pages
├── scripts/export_sample.py     builds data/sample from the research repo (needs NLA_ROOT)
├── frontend/                    React 19 + Vite + TypeScript
│   └── src/
│       ├── api/                 DataSource interface; staticSource (manifest + localStorage); apiSource (fetch)
│       ├── stores/              zustand: viewer state; settings (mode, backend URL, persisted)
│       ├── features/viewer/     ViewerProps contract, registry (kind -> component), RegionViewer
│       ├── features/annotate/   BoxTool (drag to draw), MentionTags (yes/no verdict)
│       └── components/          Header, SampleList, StageBar, Inspector, SettingsPanel
├── backend/
│   └── app/
│       ├── main.py              app factory, CORS, lifespan (create tables, index the manifest)
│       ├── config.py            DATA_DIR, DATABASE_URL, CORS_ORIGINS from .env
│       ├── models.py            Dataset, Sample, Annotation (SQLModel)
│       ├── routers/             health, datasets, samples, annotations
│       └── services/            scan.py (validate + stage manifest, stdlib), indexer.py (manifest -> SQLite)
├── compose.yml                  frontend + backend with hot reload
└── .github/workflows/           pages.yml (deploy), test.yml (pytest + build)
```

Data flow: `export_sample.py` writes `data/sample/manifest.json`. In static mode the Pages build copies it into the bundle and the browser reads it. In backend mode `indexer.py` loads the same file into SQLite and the API serves it. The screens do not know which one answered.

Annotations are rows `{id, sample_id, label, geometry, source, created_at}`. `geometry` is JSON: `{"type": "bbox", x, y, w, h}` in normalised image coordinates, `{"type": "point", x, y}`, or `{"type": "tag", arm, condition, value}`. IDs are UUIDs made in the browser, so a JSON export from static mode imports into the database without collisions.

## Libraries

Frontend: React 19, Vite 8, TypeScript, TanStack Query (server data), Zustand (UI state), react-zoom-pan-pinch (zoom and pan). Backend: FastAPI, SQLModel (SQLAlchemy + Pydantic), SQLite, pydantic-settings, pytest.

## Working on it with an agent

The three places that change most: a new data type is a new component registered in `frontend/src/features/viewer/registry.ts` that takes `ViewerProps`; a new annotation shape is a new `Geometry` variant in `frontend/src/types/api.ts` drawn in `RegionViewer.tsx`; a new endpoint is a router in `backend/app/routers/` included in `main.py`, with a test in `backend/tests/`. A model for a later homework loads once in the `lifespan` block of `backend/app/main.py` and answers on `/api/samples/{id}/predict`; its outputs are stored as annotations with `source = "model"`.

## Roadmap

HW2: classify whether a read-out names the object, trained on the judge verdicts and your own tags, served from `/api/samples/{id}/predict` or in the browser with ONNX Runtime Web. HW3: predictions drawn as `source = "model"` annotations next to human ones. HW4: region-level edits and read-out comparison at scale, with background jobs.
