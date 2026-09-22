// Mirrors backend/app/schemas.py and the manifest written by scripts/export_sample.py.
export type ConditionKey = "orig" | "content_swap" | "context_swap" | "box_blank";

export interface Condition { key: ConditionKey; label: string; note?: string }
export interface Arm { key: string; label: string; short: string; colour: string; note?: string }
export interface Verdict { mention: "yes" | "no"; evidence: string }
export interface DonorInfo { key?: number; image_id?: number; phrase?: string; head?: string; box?: number[] }

export interface Sample {
  id: string;
  index: number;
  key: number;
  image_id: number;
  phrase: string;
  head: string;
  box: number[]; // [x, y, w, h] in image pixels
  image_size: number[] | null; // [w, h]
  n_tokens: number | null;
  donor: DonorInfo;
  context_image_id: number | null;
  images: Partial<Record<ConditionKey, string>>;
  gold: string | null;
  donor_gold: string | null;
  readouts: Record<string, Partial<Record<ConditionKey, string>>>;
  judge: Record<string, Partial<Record<ConditionKey, { orig: Verdict | null; donor: Verdict | null }>>>;
}

export interface Dataset {
  id: string;
  name: string;
  kind: string;
  description: string;
  conditions: Condition[];
  arms: Arm[];
  n_samples: number;
}

export interface Manifest {
  version: number;
  datasets: (Omit<Dataset, "n_samples"> & { samples: Sample[] })[];
}

export type Geometry =
  | { type: "bbox"; x: number; y: number; w: number; h: number } // normalised 0..1
  | { type: "point"; x: number; y: number }
  | { type: "tag"; arm: string; condition: ConditionKey; value: "yes" | "no" };

export type AnnotationSource = "human" | "model";

export interface Annotation {
  id: string;
  sample_id: string;
  label: string;
  note?: string;
  geometry: Geometry;
  source: AnnotationSource;
  created_at: string;
}

export interface AnnotationCreate {
  id?: string;
  sample_id: string;
  label: string;
  note?: string;
  geometry: Geometry;
  source?: AnnotationSource;
  created_at?: string;
}

export interface AnnotationUpdate {
  label?: string;
  note?: string;
}

export interface ActivationVector {
  layer: number;
  dim: number; // size of the full pooled vector (3584 for Qwen2.5-VL-7B)
  reduced: number[]; // the pooled vector projected onto 64 principal components
  explained: number[]; // variance ratio each component explains
  basis: string; // how the components were fitted
  norm: number; // L2 norm of the full pooled vector
  top: { dim: number; value: number }[]; // the 8 largest full dimensions
  region_tokens: number[]; // flat indices (row-major on rows x cols) of the tokens that were pooled
  rows: number;
  cols: number;
  pooling?: string | null;
  model?: string | null;
}

export interface Health {
  status: string;
  datasets: number;
  samples: number;
  data_dir: string;
  model_url?: string | null;
  model_ready?: boolean;
  model_name?: string | null;
}
export interface DescribeResult { label: string; description: string; model: string; annotation: Annotation }
export interface Page { offset: number; limit: number; q?: string }
