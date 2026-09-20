import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Annotation, AnnotationCreate } from "../types/api";
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
