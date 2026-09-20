import type { Annotation, AnnotationCreate, ConditionKey, Dataset, Sample } from "../../types/api";
import type { Tool } from "../../stores/viewerStore";

/** Every data type's viewer takes these props; the registry picks the component by dataset.kind. */
export interface ViewerProps {
  sample: Sample;
  dataset: Dataset;
  condition: ConditionKey;
  annotations: Annotation[];
  tool: Tool;
  showRegion: boolean;
  draftLabel: string;
  onCreate(a: AnnotationCreate): void;
}
