import type { ActivationGrid, Annotation, AnnotationCreate, AnnotationUpdate, ConditionKey, Dataset, Geometry, Manifest, Page, Sample } from "../types/api";
import type { DataSource } from "./source";
import { newId } from "./source";

const BASE = import.meta.env.BASE_URL; // ends with "/"
const KEY = "visual-readout:annotations";

let manifestPromise: Promise<Manifest> | null = null;
function manifest(): Promise<Manifest> {
  manifestPromise ??= fetch(`${BASE}data/manifest.json`).then((r) => {
    if (!r.ok) throw new Error(`Could not load data/manifest.json (HTTP ${r.status}). Run scan.py to stage the sample data.`);
    return r.json() as Promise<Manifest>;
  });
  manifestPromise.catch(() => (manifestPromise = null));
  return manifestPromise;
}

function readAll(): Annotation[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "[]") as Annotation[];
  } catch {
    return [];
  }
}
function writeAll(list: Annotation[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {
    /* private window or blocked storage: annotations then live only in memory */
  }
}

function matches(s: Sample, q: string): boolean {
  const t = q.toLowerCase();
  return s.phrase.toLowerCase().includes(t) || s.head.toLowerCase().includes(t) || s.id.toLowerCase().includes(t);
}

export const staticSource: DataSource = {
  label: "static",
  async listDatasets(): Promise<Dataset[]> {
    const m = await manifest();
    return m.datasets.map(({ samples, ...d }) => ({ ...d, n_samples: samples.length }));
  },
  async listSamples(datasetId: string, { offset, limit, q }: Page): Promise<Sample[]> {
    const m = await manifest();
    const ds = m.datasets.find((d) => d.id === datasetId);
    if (!ds) throw new Error(`dataset ${datasetId} not found`);
    const rows = q ? ds.samples.filter((s) => matches(s, q)) : ds.samples;
    return rows.slice(offset, offset + limit);
  },
  async getSample(id: string): Promise<Sample> {
    const m = await manifest();
    for (const d of m.datasets) {
      const s = d.samples.find((x) => x.id === id);
      if (s) return s;
    }
    throw new Error(`sample ${id} not found`);
  },
  fileUrl(sample: Sample, condition: ConditionKey): string {
    const rel = sample.images[condition];
    return rel ? `${BASE}data/${rel}` : "";
  },
  async listAnnotations(sampleId: string): Promise<Annotation[]> {
    return readAll().filter((a) => a.sample_id === sampleId);
  },
  async createAnnotation(a: AnnotationCreate): Promise<Annotation> {
    const ann: Annotation = { source: "human", ...a, id: a.id ?? newId(), created_at: a.created_at ?? new Date().toISOString() };
    writeAll([...readAll(), ann]);
    return ann;
  },
  async updateAnnotation(id: string, patch: AnnotationUpdate): Promise<Annotation> {
    const all = readAll();
    const i = all.findIndex((a) => a.id === id);
    if (i < 0) throw new Error(`annotation ${id} not found`);
    all[i] = { ...all[i], ...(patch.label !== undefined && { label: patch.label }), ...(patch.note !== undefined && { note: patch.note }) };
    writeAll(all);
    return all[i];
  },
  async deleteAnnotation(id: string): Promise<void> {
    writeAll(readAll().filter((a) => a.id !== id));
  },
  async exportAnnotations(): Promise<Annotation[]> {
    return readAll();
  },
  async describeRegion(_sampleId: string, _geometry: Geometry): Promise<Annotation> {
    throw new Error("Asking the model needs the backend. Open Settings and switch to Backend.");
  },
  async getActivations(sampleId: string, layer: number): Promise<ActivationGrid> {
    // Precomputed by scripts/precompute_activations.py and copied into the build by scan.py.
    const r = await fetch(`${BASE}data/activations/${encodeURIComponent(sampleId)}.json`);
    if (!r.ok) throw new Error("No precomputed activations for this image. Switch to Backend mode to compute them.");
    const g = (await r.json()) as ActivationGrid;
    if (g.layer !== layer) throw new Error(`Precomputed grid is for layer ${g.layer}, not ${layer}.`);
    return g;
  },
  async importAnnotations(list: Annotation[]): Promise<number> {
    const have = new Set(readAll().map((a) => a.id));
    const add = list.filter((a) => a.id && !have.has(a.id));
    writeAll([...readAll(), ...add]);
    return add.length;
  },
};
