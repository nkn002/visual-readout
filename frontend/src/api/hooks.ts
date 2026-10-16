import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Annotation, AnnotationCreate, AnnotationUpdate, DefaultPrediction, Geometry, ModelReport } from "../types/api";
import { useDataSource, useSourceKey } from "./source";

export function useDatasets() {
  const src = useDataSource();
  const key = useSourceKey();
  return useQuery({ queryKey: ["datasets", key], queryFn: () => src.listDatasets(), retry: false });
}

export function useSamples(datasetId: string | undefined, q: string) {
  const src = useDataSource();
  const key = useSourceKey();
  return useQuery({
    queryKey: ["samples", key, datasetId, q],
    queryFn: () => src.listSamples(datasetId!, { offset: 0, limit: 500, q: q || undefined }),
    enabled: !!datasetId,
    retry: false,
  });
}

export function useAnnotations(sampleId: string | undefined) {
  const src = useDataSource();
  const key = useSourceKey();
  return useQuery({
    queryKey: ["annotations", key, sampleId],
    queryFn: () => src.listAnnotations(sampleId!),
    enabled: !!sampleId,
    retry: false,
  });
}

export function useCreateAnnotation() {
  const src = useDataSource();
  const key = useSourceKey();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (a: AnnotationCreate) => src.createAnnotation(a),
    onSuccess: (_r, a) => qc.invalidateQueries({ queryKey: ["annotations", key, a.sample_id] }),
  });
}

export function useUpdateAnnotation(sampleId: string | undefined) {
  const src = useDataSource();
  const key = useSourceKey();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: AnnotationUpdate }) => src.updateAnnotation(id, patch),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["annotations", key, sampleId] }),
  });
}

export function useDeleteAnnotation(sampleId: string | undefined) {
  const src = useDataSource();
  const key = useSourceKey();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => src.deleteAnnotation(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["annotations", key, sampleId] }),
  });
}

export function useImportAnnotations() {
  const src = useDataSource();
  const key = useSourceKey();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (list: Annotation[]) => src.importAnnotations(list),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["annotations", key] }),
  });
}

export function useDescribeRegion(sampleId: string | undefined) {
  const src = useDataSource();
  const key = useSourceKey();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (geometry: Geometry) => src.describeRegion(sampleId!, geometry),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["annotations", key, sampleId] }),
  });
}

export function useActivations(sampleId: string | undefined, geometry: Geometry | null, layer: number) {
  const src = useDataSource();
  const key = useSourceKey();
  return useQuery({
    queryKey: ["activations", key, sampleId, layer, geometry],
    queryFn: () => src.getActivations(sampleId!, geometry!, layer),
    enabled: !!sampleId && !!geometry,
    retry: false,
    staleTime: Infinity,
  });
}

// Static files shipped with every build (GitHub Pages included), so they work in both modes.
async function staticJson<T>(path: string): Promise<T> {
  const r = await fetch(`${import.meta.env.BASE_URL}${path}`);
  if (!r.ok || !(r.headers.get("content-type") ?? "").includes("json")) throw new Error(`${path} is not available`);
  return (await r.json()) as T;
}

export function useDefaultPredictions() {
  return useQuery({ queryKey: ["default-predictions"], queryFn: () => staticJson<Record<string, DefaultPrediction>>("data/predictions.json"), staleTime: Infinity, retry: false });
}

export function useModelReport(enabled: boolean) {
  return useQuery({ queryKey: ["model-report"], queryFn: () => staticJson<ModelReport>("data/model/metrics.json"), staleTime: Infinity, retry: false, enabled });
}

export function usePredictRegion(sampleId: string | undefined) {
  const src = useDataSource();
  return useMutation({ mutationFn: (geometry: Geometry) => src.predictRegion(sampleId!, geometry) });
}

/** Poll a training run once a second while it is queued or running. */
export function useTrainingRun(id: string | null) {
  const src = useDataSource();
  const key = useSourceKey();
  return useQuery({
    queryKey: ["training", key, id],
    queryFn: () => src.getTraining(id!),
    enabled: !!id,
    retry: false,
    refetchInterval: (q) => (q.state.data && !["queued", "running"].includes(q.state.data.status) ? false : 1000),
  });
}
