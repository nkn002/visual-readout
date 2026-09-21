import { useMemo } from "react";
import type { Annotation, AnnotationCreate, AnnotationUpdate, ConditionKey, Dataset, Geometry, Page, Sample } from "../types/api";
import { useSettings } from "../stores/settingsStore";
import { staticSource } from "./staticSource";
import { ApiSource } from "./apiSource";

/** Everything the screens need from data, in either mode. */
export interface DataSource {
  readonly label: "static" | "api";
  listDatasets(): Promise<Dataset[]>;
  listSamples(datasetId: string, page: Page): Promise<Sample[]>;
  getSample(id: string): Promise<Sample>;
  fileUrl(sample: Sample, condition: ConditionKey): string;
  listAnnotations(sampleId: string): Promise<Annotation[]>;
  createAnnotation(a: AnnotationCreate): Promise<Annotation>;
  updateAnnotation(id: string, patch: AnnotationUpdate): Promise<Annotation>;
  deleteAnnotation(id: string): Promise<void>;
  exportAnnotations(): Promise<Annotation[]>;
  importAnnotations(list: Annotation[]): Promise<number>;
  /** Ask the vision model what a box contains; the answer comes back as a `model` annotation. */
  describeRegion(sampleId: string, geometry: Geometry): Promise<Annotation>;
}

export function useDataSource(): DataSource {
  const mode = useSettings((s) => s.mode);
  const apiBase = useSettings((s) => s.apiBase);
  return useMemo(() => (mode === "api" ? new ApiSource(apiBase) : staticSource), [mode, apiBase]);
}

/** Part of every query key, so switching mode or backend refetches. */
export function useSourceKey(): string {
  const mode = useSettings((s) => s.mode);
  const apiBase = useSettings((s) => s.apiBase);
  return `${mode}:${apiBase}`;
}

export function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
