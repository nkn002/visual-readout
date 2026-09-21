# Visual read-out

A web viewer for one kind of research data: a photo, a region of that photo, and what a vision-language model says about that region.

Live: **https://nkn002.github.io/visual-readout/** · DS7400 Deep Learning, Homework 1 · Author: Khai-Nguyen Nguyen

## The goal

A user opens an image, draws a region, and asks Qwen2.5-VL what is in it. A **Default region** button reveals the region the research dataset defined for that image and the text a verbalizer produced for it. Users can record their own verdicts and save everything. HW1 delivers that viewer; later homeworks add trained models on top of it.

The research behind the data maps Qwen2.5-VL region activations into a frozen text verbalizer through a small trained connector. The viewer is the tool for looking at those regions and read-outs one at a time.

**Data type:** images plus boxes plus text. A sample is one Visual Genome region: an image, a box `[x, y, w, h]`, the phrase describing it, the verbalizer's read-out, and a judge's verdict. The committed sample has 12 regions.

## What you can do

| Feature | Where in the code |
|---|---|
| Browse the 12 images with thumbnails and arrow keys | `frontend/src/components/SampleList.tsx` |
| Zoom and pan the image | `frontend/src/features/viewer/RegionViewer.tsx` |
| **Annotation:** drag to draw a labelled region; delete; export and import as JSON | `frontend/src/features/annotate/BoxTool.tsx`, `Inspector.tsx` |
| **Ask Qwen2.5-VL** what a drawn region contains | `backend/app/services/describe.py`, `scripts/model_server.py` |
| **Default region:** reveal the dataset's box, its phrase, and the frozen verbalizer's read-out with the judge's verdict | `StageBar.tsx`, `Inspector.tsx` |
| Give your own names-it / misses-it verdict on that read-out | `frontend/src/features/annotate/MentionTags.tsx` |
| **Backend:** annotations persist in SQLite through a FastAPI service | `backend/app/` |
| Two data modes behind one interface: static (GitHub Pages) or backend | `frontend/src/api/` |
| Deploy to GitHub Pages on every push | `.github/workflows/pages.yml` |

Implemented annotation: **yes**. Implemented backend/database: **yes** (SQLite via SQLModel; tables are created at startup, migrations are a follow-up). Model in the loop: **yes**, Qwen2.5-VL-7B-Instruct on a cluster GPU, prompted directly.

## Run it

Static mode, what the live page does, no server:

```bash
python backend/app/services/scan.py --data data/sample --out frontend/public/data   # stage the sample
cd frontend && npm ci && npm run dev                                                # http://localhost:5173
```

Backend mode, annotations persist and the model can be asked:

```bash
cd backend && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn app.main:app --reload            # http://localhost:8000, docs at /docs
```

In the app open **Settings** (the pill in the top bar), choose **Backend**, and check the connection. In local dev the Vite proxy forwards `/api` to port 8000, so the URL can stay empty. From the public page use `http://localhost:8000`; Chrome asks once for permission to reach localhost.

Both at once, with the right URLs inside a UVA Open OnDemand VS Code session: `scripts/dev.sh`.

The model, on the UVA cluster (needs a GPU):

```bash
sbatch scripts/model_server.sbatch        # Qwen2.5-VL-7B-Instruct on an RTX 3090, about 90 s to load
tail -f data/db/model_server.<jobid>.log  # "serving http://<node>:8100" when ready
```

The server writes its address to `data/db/model_server.url`; the backend reads that file on every request, so nothing restarts. Without it, the Ask button answers with a clear "not running" message.

Docker, both services with hot reload: `docker compose up`, then open http://localhost:5173.

Tests: `cd backend && .venv/bin/python -m pytest` (11 tests: indexing, endpoints, annotation round trip, import, manifest validation, the model call against a stand-in server).

## The repository, top down

Four folders each own one part of the goal:

- `data/` holds the images and text the viewer shows.
- `frontend/` is the web page the user interacts with.
- `backend/` saves what the user does and talks to the model.
- `scripts/` starts everything, on a laptop or on the UVA cluster.

Beside them: `docs/DESIGN.md` (why it is built this way), `compose.yml` (frontend and backend together in Docker), `.env.example` (the settings the backend reads), and `.github/workflows/` (publishing and tests).

### `data/` — what the viewer shows

- `sample/` is the only data committed. It is small (12 images, 3 MB) so the app works from a fresh clone and on GitHub Pages.
  - `manifest.json` is the index of the 12 samples. For each one it lists the image files, the region box, the phrase, the read-out text of each verbalizer arm, and the judge's verdict. Two files read it: `frontend/src/api/staticSource.ts` in static mode and `backend/app/services/indexer.py` in backend mode, so the page and the API always agree on what a sample is.
  - `images/` holds the JPEGs, four per sample: the original and three edited versions. HW1 shows the original; the edits are for later homeworks.
- `raw/` and `db/` are ignored by git. `raw/` is for the full dataset, `db/` for the SQLite database and the model server's address.

### `frontend/` — the page the user sees

Built with React and Vite. The source is in `src/`; `npm run build` compiles it into `dist/`, a plain static site, which is what GitHub Pages serves.

A user action travels: `components/` → `stores/` for UI state and `api/hooks.ts` for data → `api/source.ts`, which hands the call to `staticSource.ts` (reads `manifest.json`, keeps annotations in the browser; what GitHub Pages runs) or to `apiSource.ts` (sends JSON to the backend's `/api/...` routes) → the answer comes back the same way and the screen re-renders. The files below are listed in that order.

- `main.tsx` starts the app. `App.tsx` lays out the three panes, handles keyboard shortcuts, and asks `hooks.ts` for the dataset, the sample list, and the selected image's annotations.
- `components/` — the panes and bars.
  - `SampleList.tsx`: the left column, one thumbnail per image. Thumbnails are `<img>` tags whose URL comes from the data source (a file in static mode, `GET /api/samples/{id}/file` in backend mode).
  - `StageBar.tsx`: above the image, the drawing tool and the **Default region** button. Writes only to `stores/viewerStore.ts`.
  - `Inspector.tsx`: the right column. Lists the user's regions; its **Ask Qwen2.5-VL** button calls `useDescribeRegion` in `hooks.ts`, its delete buttons call `useDeleteAnnotation`, and export/import go through the data source.
  - `Header.tsx`: the title and the data-source badge. `SettingsPanel.tsx`: switches between static and backend mode in `stores/settingsStore.ts`, and its **Check connection** button calls the backend's `GET /api/health` directly.
- `features/viewer/` — the image itself. `RegionViewer.tsx` shows the photo with zoom and pan and draws every box it is given. `types.ts` is the contract every viewer satisfies; `registry.ts` maps a data kind to its viewer, so a time-series or mesh viewer could plug in beside this one.
- `features/annotate/` — the two ways to annotate. `BoxTool.tsx` turns a drag into a box and hands it to `hooks.ts` as `createAnnotation`; `MentionTags.tsx` does the same for a names-it / misses-it verdict.
- `stores/` — small pieces of UI state that the panes share: `viewerStore.ts` (selected image, active tool, default region shown or not) and `settingsStore.ts` (static or backend, backend URL). No network calls here.
- `api/hooks.ts` — the one door to data. `useSamples`, `useAnnotations`, `useCreateAnnotation`, `useDeleteAnnotation`, `useDescribeRegion` wrap the calls in TanStack Query, so a save refreshes the list automatically. Every hook calls whatever `source.ts` returns.
- `api/source.ts` — the `DataSource` interface (list samples, get an image URL, list/create/delete annotations, describe a region) and `useDataSource()`, which returns `staticSource` or `apiSource` according to `settingsStore`. The panes never know which one answered.
- `api/staticSource.ts` — fulfils the interface with no server: fetches `data/manifest.json`, points image URLs at the bundled JPEGs, keeps annotations in `localStorage`, and rejects `describeRegion` with "needs the backend".
- `api/apiSource.ts` — the **only file that talks to the backend**. Each method is one HTTP call, answered by one backend file:
  - `listDatasets`, `listSamples` → `GET /api/datasets`, `GET /api/datasets/{id}/samples` → `backend/app/routers/datasets.py`
  - `fileUrl` → `GET /api/samples/{id}/file?condition=orig` → `routers/samples.py`
  - `listAnnotations`, `createAnnotation`, `deleteAnnotation`, `exportAnnotations`, `importAnnotations` → `GET/POST /api/samples/{id}/annotations`, `DELETE /api/annotations/{id}`, `GET /api/annotations`, `POST /api/annotations/import` → `routers/annotations.py`
  - `describeRegion` → `POST /api/samples/{id}/describe` with the box → `routers/describe.py`
  - `health` → `GET /api/health` → `routers/health.py`

  In local development Vite forwards `/api/*` to port 8000; from the public page the URL set in Settings is used, which is why the backend's `CORS_ORIGINS` lists the Pages origin.
- `types/api.ts` — the TypeScript shapes of samples, annotations, and responses: the same shapes `manifest.json` holds and the backend's `schemas.py` returns.
- `index.css` — all styling, light and dark.

### `backend/` — saving and asking the model

A FastAPI service with a SQLite database. Everything is in `app/`.

A request from `apiSource.ts` enters `main.py` → the file in `routers/` that owns that URL, with the body checked against `schemas.py` → `services/` for any real work → `models.py` and `database.py` to read or write SQLite, or `services/describe.py` → `scripts/model_server.py` on the GPU for a model answer → a JSON reply to the browser. The files below are listed in that order.

- `main.py` creates the service, allows the frontend's origins (CORS), and includes every router. At startup it creates the tables and runs `services/indexer.py` so the database holds the manifest before the first request.
- `config.py` reads settings from `.env`: where the data is, where the database is, which origins may call, and where the model server is. `main.py`, `database.py`, and `services/describe.py` read from it.
- `routers/` — the endpoints, one file per resource. Each states which frontend action reaches it and what it sends back:
  - `health.py` — Settings → **Check connection**. Returns whether the API is up and whether the model server answers.
  - `datasets.py` — page load and the image list. Returns the dataset and its samples, each exactly one manifest record.
  - `samples.py` — every `<img>` tag in backend mode. Returns the JPEG bytes for a sample and condition.
  - `annotations.py` — drawing a box, giving a verdict, deleting, export and import. Returns the stored row (an `Annotation`) or the count imported.
  - `describe.py` — the **Ask Qwen2.5-VL** button. Hands the box to `services/describe.py`, stores the answer as an annotation with `source = "model"`, and returns the label, the description, and that stored row.
- `schemas.py` defines what every request and response looks like. The frontend's `types/api.ts` mirrors it.
- `services/` — the logic behind the endpoints, with no HTTP of its own.
  - `scan.py` validates the manifest and, in the Pages workflow, copies it into the web build.
  - `indexer.py`, run by `main.py` at startup, loads the manifest into the Dataset and Sample tables through `models.py`.
  - `describe.py`, called by `routers/describe.py`: outlines the user's box on the photo, enlarges the crop, sends both with a prompt to `scripts/model_server.py` (`POST /describe`), and parses the "Label / Description" reply. It finds the model server through `MODEL_URL` or the file `data/db/model_server.url`.
- `database.py` opens SQLite and gives each request one session. `models.py` defines the three tables: Dataset, Sample, Annotation. An annotation is a row `{id, sample_id, label, note, geometry, source, created_at}`; `geometry` is a box, a point, or a verdict tag in normalised image coordinates, and `source` is `human` or `model`. IDs are UUIDs made in the browser, so a JSON export from static mode imports without collisions.
- `tests/` — 11 tests that run the whole service against the sample data, with the model server replaced by a stand-in.

### `scripts/` — running it

- `export_sample.py` builds `data/sample/` from the research repository. Run once when the data changes.
- `dev.sh` starts the frontend and backend and prints the URLs, including the proxy addresses needed inside the cluster's VS Code sessions.
- `model_server.py` loads Qwen2.5-VL-7B-Instruct on a GPU and answers over HTTP. `backend/app/services/describe.py` calls it with two images and a prompt; it returns `{text, model, seconds}`. On start it writes its address to `data/db/model_server.url`, so the backend finds it without a restart. `model_server.sbatch` runs it as a Slurm job.

### `.github/workflows/` — publishing

`pages.yml` builds the frontend and deploys it to `nkn002.github.io/visual-readout` on every push to `main`. `test.yml` runs the backend tests and the frontend build on every push.

## How the components interact

Diagrams at four levels of zoom, following the C4 model: who uses the system, what runs where in each mode, which files inside each part call which, how one request travels, and what the data looks like. **Key:** a box is one running part, with its technology in brackets and its job underneath. An arrow is a one-way call, labelled with what it does and how.

### 1. Context: who talks to what

```mermaid
---
title: System context
---
flowchart LR
  U["Person: student or grader<br/>opens the viewer in a browser"]
  V["Visual read-out<br/>[web app]<br/>shows an image, lets the user mark a region,<br/>and answers what is in it"]
  Q["Qwen2.5-VL-7B-Instruct<br/>[vision-language model on a cluster GPU]<br/>answers questions about an image"]
  R["nla research repository<br/>[files on the cluster]<br/>Visual Genome regions, read-outs, judge verdicts"]
  U -- "draws regions, asks questions [browser]" --> V
  V -- "asks: what is inside this box? [HTTP, JSON]" --> Q
  R -- "exported once into data/sample [export_sample.py]" --> V
```

### 2. Containers: what runs where, in each mode

The app runs in two modes. The screens are identical; only where the data comes from and where annotations go changes.

```mermaid
---
title: Static mode, the public page on GitHub Pages (no server)
---
flowchart LR
  B["Web app<br/>[React, runs in the browser]<br/>image list, viewer, region drawing, verdicts"]
  F["Static files<br/>[manifest.json + 48 JPEGs, served by GitHub Pages]<br/>the 12 samples and their read-outs"]
  L["Browser storage<br/>[localStorage]<br/>the user's annotations, exportable as JSON"]
  B -- "reads samples and images [HTTP GET]" --> F
  B -- "saves and reloads annotations" --> L
```

```mermaid
---
title: Backend mode, with the API and the model server running
---
flowchart LR
  B["Web app<br/>[React, runs in the browser]<br/>the same screens"]
  A["API<br/>[FastAPI, Python]<br/>serves samples, stores annotations,<br/>prepares model requests"]
  D["Database<br/>[SQLite file]<br/>samples and annotations"]
  M["Model server<br/>[Python on a GPU node]<br/>runs Qwen2.5-VL-7B-Instruct"]
  B -- "reads samples, saves annotations [HTTP, JSON]" --> A
  A -- "reads and writes [SQL]" --> D
  A -- "sends the photo with the box outlined, plus the crop [HTTP, JSON]" --> M
```

### 3. Components: how the files inside the web app interact

Read left to right: what the user sees, the UI state it shares, and the data layer that hides where data comes from. `types/api.ts` (not drawn) defines the shapes every file below uses.

```mermaid
---
title: Inside the web app (frontend/src)
---
flowchart LR
  subgraph UI["components/ and features/ — what the user sees"]
    direction TB
    App["App.tsx<br/>lays out the panes, keyboard shortcuts"]
    SL["SampleList.tsx<br/>image list"]
    SB["StageBar.tsx<br/>draw or pan tool, Default region button"]
    RV["RegionViewer.tsx + BoxTool.tsx<br/>image, zoom, boxes, drag to draw"]
    INS["Inspector.tsx + MentionTags.tsx<br/>your regions, Ask button, read-out, export"]
    SET["SettingsPanel.tsx<br/>static or backend"]
    App --> SL
    App --> SB
    App --> RV
    App --> INS
    App --> SET
  end
  subgraph STATE["stores/ — shared UI state"]
    VS["viewerStore.ts<br/>selected image, tool, default region on or off"]
    SS["settingsStore.ts<br/>mode, backend URL"]
  end
  subgraph DATA["api/ — where data comes from"]
    direction TB
    HK["hooks.ts<br/>useSamples, useAnnotations, useDescribeRegion"]
    SRC["source.ts<br/>DataSource interface, picked by mode"]
    ST["staticSource.ts<br/>reads manifest.json, keeps annotations in localStorage"]
    AP["apiSource.ts<br/>calls the API over HTTP"]
    HK --> SRC
    SRC --> ST
    SRC --> AP
  end
  UI -- "read and set" --> VS
  SET -- "sets" --> SS
  SS -- "chooses static or api" --> SRC
  SL -- "list samples" --> HK
  RV -- "create annotation" --> HK
  INS -- "list, delete, describe" --> HK
```

### 4. Components: how the files inside the API interact

Requests enter through `routers/`, logic lives in `services/`, and only `database.py` and `models.py` touch SQLite. Not drawn: `config.py` hands settings from `.env` to `main.py`, `database.py`, and `services/describe.py`; `schemas.py` defines the request and response shapes every router validates with.

```mermaid
---
title: Inside the API (backend/app)
---
flowchart LR
  MAIN["main.py<br/>creates the app, adds CORS,<br/>at startup: create tables, index the manifest"]
  subgraph RT["routers/ — one file per resource, validated by schemas.py"]
    direction TB
    RH["health.py<br/>is the API up, is the model up"]
    RD["datasets.py, samples.py<br/>list samples, serve image files"]
    RA["annotations.py<br/>create, list, delete, export, import"]
    RX["describe.py<br/>ask the model about a box"]
  end
  subgraph SVC["services/ — logic without HTTP"]
    direction TB
    IDX["indexer.py<br/>manifest rows into the tables"]
    SCAN["scan.py<br/>validates the manifest"]
    DSC["describe.py<br/>outline the box, enlarge the crop,<br/>call the model server, parse the answer"]
  end
  subgraph STORE["database.py + models.py — the only files that touch SQLite"]
    direction TB
    DBM["database.py<br/>engine, one session per request"]
    MOD["models.py<br/>Dataset, Sample, Annotation tables"]
  end
  DB[("SQLite<br/>data/db/app.db")]
  MS["scripts/model_server.py<br/>Qwen2.5-VL on a GPU node"]
  MAIN -- "includes" --> RT
  MAIN -- "runs at startup" --> IDX
  IDX -- "validates with" --> SCAN
  RT -- "reads and writes rows" --> STORE
  IDX -- "inserts rows" --> STORE
  RX -- "calls" --> DSC
  DSC -- "POST /describe [HTTP, JSON]" --> MS
  STORE -- "SQL" --> DB
```

### 5. Build and run: how the scripts move the data files

```mermaid
---
title: Data files, from the research repo to the running app
---
flowchart LR
  NLA[("nla research repo<br/>results/data")]
  EXP["scripts/export_sample.py<br/>run once when data changes"]
  MAN[("data/sample/<br/>manifest.json + images, committed")]
  SCAN["backend/app/services/scan.py<br/>runs in the Pages workflow and dev.sh"]
  PUB[("frontend/public/data/<br/>copied into the web build")]
  IDX["backend indexer.py<br/>runs at API startup"]
  DB[("SQLite<br/>data/db/app.db")]
  NLA -- "reads regions, read-outs, verdicts" --> EXP
  EXP -- "writes" --> MAN
  MAN -- "validated and copied" --> SCAN
  SCAN --> PUB
  MAN -- "loaded" --> IDX
  IDX --> DB
```

### 6. One request: asking the model about a drawn region

```mermaid
---
title: Sequence, from drawing a box to seeing the answer
---
sequenceDiagram
  actor U as User
  participant W as Web app
  participant A as API
  participant M as Model server
  U->>W: drag a box on the image
  W->>A: save the box as an annotation (source: human)
  A-->>W: saved
  U->>W: click "Ask Qwen2.5-VL"
  W->>A: describe this box
  A->>A: outline the box on the photo, enlarge the crop
  A->>M: two images + "what is inside the red box?"
  M-->>A: "Label: bear paws. Description: two brown paws..."
  A->>A: save the answer as an annotation (source: model)
  A-->>W: label and description
  W-->>U: answer shown under the region
```

### 7. Data: what is stored

The same three shapes appear in `manifest.json`, in the SQLite tables, and in the TypeScript types.

```mermaid
---
title: Data model
---
classDiagram
  direction LR
  class Dataset {
    id
    name
    conditions
    arms
  }
  class Sample {
    id
    image files, one per condition
    box: x, y, w, h
    phrase
    read-out text per arm
    judge verdict per arm
  }
  class Annotation {
    id
    label
    note
    geometry: box, point or tag
    source: human or model
    created_at
  }
  Dataset "1" --> "many" Sample : contains
  Sample "1" --> "many" Annotation : has
```

## Why React and not a plain HTML page

An academic page is a document: fixed content, no state. This page has state: the selected image, the active tool, boxes being dragged, annotations that change, two data sources. React splits that into components that re-render when state changes, and TypeScript checks that the data shapes match the backend. The compiled `dist/` is still a plain static site, the same kind of thing as a paper page; only the source is organised as an app.

## Libraries

Frontend: React 19, Vite 8, TypeScript, TanStack Query (server data), Zustand (UI state), react-zoom-pan-pinch (zoom and pan). Backend: FastAPI, SQLModel (SQLAlchemy + Pydantic), SQLite, pydantic-settings, Pillow, httpx, pytest. Model server: PyTorch, Transformers, Qwen2.5-VL-7B-Instruct.
## Working on it with an agent

A new data type is a new component registered in `frontend/src/features/viewer/registry.ts` that takes `ViewerProps`. A new annotation shape is a new `Geometry` variant in `frontend/src/types/api.ts`, drawn in `RegionViewer.tsx`. A new endpoint is a router in `backend/app/routers/` included in `main.py`, with a test in `backend/tests/`. A trained model for a later homework loads in `scripts/model_server.py` or in the backend's `lifespan`, and its outputs are stored as annotations with `source = "model"`.

## Roadmap

HW2: a classifier for whether a read-out names the object, trained on the judge verdicts and the user's tags. HW3: predictions drawn as `source = "model"` boxes next to human ones. HW4: region-level edits and read-out comparison at scale.
