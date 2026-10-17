import type { ActivationVector, ProbeView, PredictResult, TrainConfig, TrainSnapshot, Annotation, AnnotationCreate, AnnotationUpdate, ConditionKey, Dataset, DescribeResult, Geometry, Health, Page, Sample } from "../types/api";
import type { DataSource } from "./source";

export class ApiSource implements DataSource {
  readonly label = "api" as const;
  private readonly base: string;
  constructor(base: string) {
    this.base = base;
  }

  private async req<T>(path: string, init?: RequestInit): Promise<T> {
    let r: Response;
    try {
      r = await fetch(`${this.base}/api${path}`, { headers: { "content-type": "application/json" }, ...init });
    } catch (e) {
      throw new Error(`Cannot reach ${this.base || "the backend"}: ${(e as Error).message}`);
    }
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try {
        const body = (await r.json()) as { detail?: unknown };
        if (body.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      } catch {
        /* no JSON body */
      }
      throw new Error(msg);
    }
    return r.status === 204 ? (undefined as T) : ((await r.json()) as T);
  }

  health(): Promise<Health> {
    return this.req<Health>("/health");
  }
  listDatasets(): Promise<Dataset[]> {
    return this.req<Dataset[]>("/datasets");
  }
  listSamples(datasetId: string, { offset, limit, q }: Page): Promise<Sample[]> {
    const p = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (q) p.set("q", q);
    return this.req<Sample[]>(`/datasets/${encodeURIComponent(datasetId)}/samples?${p}`);
  }
  getSample(id: string): Promise<Sample> {
    return this.req<Sample>(`/samples/${encodeURIComponent(id)}`);
  }
  fileUrl(sample: Sample, condition: ConditionKey): string {
    return sample.images[condition] ? `${this.base}/api/samples/${encodeURIComponent(sample.id)}/file?condition=${condition}` : "";
  }
  listAnnotations(sampleId: string): Promise<Annotation[]> {
    return this.req<Annotation[]>(`/samples/${encodeURIComponent(sampleId)}/annotations`);
  }
  createAnnotation(a: AnnotationCreate): Promise<Annotation> {
    return this.req<Annotation>(`/samples/${encodeURIComponent(a.sample_id)}/annotations`, { method: "POST", body: JSON.stringify(a) });
  }
  updateAnnotation(id: string, patch: AnnotationUpdate): Promise<Annotation> {
    return this.req<Annotation>(`/annotations/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(patch) });
  }
  deleteAnnotation(id: string): Promise<void> {
    return this.req<void>(`/annotations/${encodeURIComponent(id)}`, { method: "DELETE" });
  }
  exportAnnotations(): Promise<Annotation[]> {
    return this.req<Annotation[]>("/annotations");
  }
  async describeRegion(sampleId: string, geometry: Geometry): Promise<Annotation> {
    const r = await this.req<DescribeResult>(`/samples/${encodeURIComponent(sampleId)}/describe`, { method: "POST", body: JSON.stringify({ geometry }) });
    return r.annotation;
  }
  getActivations(sampleId: string, geometry: Geometry, layer: number): Promise<ActivationVector> {
    return this.req<ActivationVector>(`/samples/${encodeURIComponent(sampleId)}/activations`, { method: "POST", body: JSON.stringify({ geometry, layer }) });
  }
  predictRegion(sampleId: string, geometry: Geometry): Promise<PredictResult> {
    return this.req<PredictResult>(`/samples/${encodeURIComponent(sampleId)}/predict`, { method: "POST", body: JSON.stringify({ geometry }) });
  }
  probeRegion(sampleId: string, geometry: Geometry): Promise<ProbeView> {
    return this.req<ProbeView>(`/samples/${encodeURIComponent(sampleId)}/probe`, { method: "POST", body: JSON.stringify({ geometry }) });
  }
  startTraining(cfg: TrainConfig): Promise<TrainSnapshot> {
    return this.req<TrainSnapshot>("/train", { method: "POST", body: JSON.stringify(cfg) });
  }
  getTraining(id: string): Promise<TrainSnapshot> {
    return this.req<TrainSnapshot>(`/train/${encodeURIComponent(id)}`);
  }
  cancelTraining(id: string): Promise<TrainSnapshot> {
    return this.req<TrainSnapshot>(`/train/${encodeURIComponent(id)}/cancel`, { method: "POST" });
  }
  useTrainedModel(id: string): Promise<{ in_use: string | null }> {
    return this.req<{ in_use: string | null }>(`/train/${encodeURIComponent(id)}/use`, { method: "POST" });
  }
  resetModel(): Promise<{ in_use: string | null }> {
    return this.req<{ in_use: string | null }>("/train/reset-model", { method: "POST" });
  }
  async importAnnotations(list: Annotation[]): Promise<number> {
    const r = await this.req<{ imported: number }>("/annotations/import", { method: "POST", body: JSON.stringify(list) });
    return r.imported;
  }
}
