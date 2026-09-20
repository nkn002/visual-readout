import type { Annotation, AnnotationCreate, ConditionKey } from "../../types/api";

interface Props {
  sampleId: string;
  arm: string;
  condition: ConditionKey;
  annotations: Annotation[];
  onCreate(a: AnnotationCreate): void;
  onDelete(id: string): void;
}

/** Your own yes/no verdict on one read-out, stored as a `tag` annotation. */
export function MentionTags({ sampleId, arm, condition, annotations, onCreate, onDelete }: Props) {
  const current = annotations.find((a) => a.geometry.type === "tag" && a.geometry.arm === arm && a.geometry.condition === condition);
  const value = current?.geometry.type === "tag" ? current.geometry.value : null;
  const set = (v: "yes" | "no") => {
    if (current) onDelete(current.id);
    if (value !== v) onCreate({ sample_id: sampleId, label: `mention:${v}`, geometry: { type: "tag", arm, condition, value: v } });
  };
  return (
    <div className="tags" role="group" aria-label="Your verdict: does this read-out name the object?">
      <span>Your verdict</span>
      <button className={value === "yes" ? "on yes" : ""} onClick={() => set("yes")} type="button">
        names it
      </button>
      <button className={value === "no" ? "on no" : ""} onClick={() => set("no")} type="button">
        misses it
      </button>
    </div>
  );
}
