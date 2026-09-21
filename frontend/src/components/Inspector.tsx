import { useRef, useState } from "react";
import { useDataSource } from "../api/source";
import { useImportAnnotations } from "../api/hooks";
import { MentionTags } from "../features/annotate/MentionTags";
import type { Tool } from "../stores/viewerStore";
import type { Annotation, AnnotationCreate, Arm, ConditionKey, Dataset, Sample, Verdict } from "../types/api";

interface Props {
  sample: Sample | null;
  dataset: Dataset | undefined;
  condition: ConditionKey;
  annotations: Annotation[];
  onCreate(a: AnnotationCreate): void;
  onDelete(id: string): void;
  tool: Tool;
  setTool(t: Tool): void;
  draftLabel: string;
  setDraftLabel(s: string): void;
  showRegion: boolean;
  toggleRegion(): void;
}

/** HW1 shows one arm: the released verbalizer with nothing trained. The other arms stay in the data for later homeworks. */
export function primaryArm(dataset: Dataset): Arm | undefined {
  return dataset.arms.find((a) => a.key === "affine-identity") ?? dataset.arms[0];
}

function Badge({ v }: { v: Verdict | null | undefined }) {
  if (!v) return null;
  return (
    <span className={`badge ${v.mention}`} title={v.evidence ? `Judge evidence: ${v.evidence}` : "Judge verdict"}>
      judge: {v.mention === "yes" ? "names it" : "misses it"}
    </span>
  );
}

function describeGeometry(a: Annotation): string {
  const g = a.geometry;
  if (g.type === "bbox") return `box ${(g.w * 100).toFixed(0)}×${(g.h * 100).toFixed(0)}% at (${(g.x * 100).toFixed(0)}, ${(g.y * 100).toFixed(0)})`;
  if (g.type === "point") return `point (${(g.x * 100).toFixed(0)}, ${(g.y * 100).toFixed(0)})`;
  return "verdict";
}

export function Inspector(p: Props) {
  const { sample, dataset, condition, annotations } = p;
  const src = useDataSource();
  const importMut = useImportAnnotations();
  const fileRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<string>("");

  if (!sample || !dataset) return <aside className="inspector" />;
  const arm = primaryArm(dataset);
  const text = arm ? sample.readouts[arm.key]?.[condition] : undefined;
  const verdict = arm ? sample.judge[arm.key]?.[condition]?.orig : undefined;

  const exportJson = async () => {
    const all = await src.exportAnnotations();
    const blob = new Blob([JSON.stringify(all, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `visual-readout-annotations-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    setStatus(`Exported ${all.length} annotations.`);
  };
  const importJson = async (file: File) => {
    try {
      const list = JSON.parse(await file.text()) as Annotation[];
      const n = await importMut.mutateAsync(list);
      setStatus(`Imported ${n} of ${list.length} annotations.`);
    } catch (e) {
      setStatus(`Import failed: ${(e as Error).message}`);
    }
  };

  return (
    <aside className="inspector">
      <section>
        <h2>Region</h2>
        <p className="phrase">“{sample.phrase}”</p>
        <dl className="kv">
          <dt>Head noun</dt>
          <dd>{sample.head}</dd>
          <dt>Box (px)</dt>
          <dd>
            x {sample.box[0]}, y {sample.box[1]}, {sample.box[2]}×{sample.box[3]}
          </dd>
          <dt>Image</dt>
          <dd>
            Visual Genome {sample.image_id} · {sample.image_size?.join("×")}
          </dd>
        </dl>
        <label className="check">
          <input type="checkbox" id="show-region" checked={p.showRegion} onChange={p.toggleRegion} /> Show region box
        </label>
      </section>

      <section>
        <h2>Read-out</h2>
        {arm && (
          <article className="readout" style={{ ["--arm" as string]: arm.colour }}>
            <header>
              <span className="swatch" />
              <strong>{arm.label}</strong>
              <span className="badges">
                <Badge v={verdict} />
              </span>
            </header>
            {text ? <p className="text">{text}</p> : <p className="muted">No read-out for this region.</p>}
            <MentionTags sampleId={sample.id} arm={arm.key} condition={condition} annotations={annotations} onCreate={p.onCreate} onDelete={p.onDelete} />
          </article>
        )}
        {arm?.note && <p className="muted small">{arm.note}.</p>}
      </section>

      <section>
        <h2>Annotations</h2>
        <div className="toolrow">
          <div className="seg" role="group" aria-label="Tool">
            <button type="button" className={p.tool === "box" ? "on" : ""} onClick={() => p.setTool("box")} title="Drag on the image to draw a box (b)">
              Draw box
            </button>
            <button type="button" className={p.tool === "select" ? "on" : ""} onClick={() => p.setTool("select")} title="Drag to pan, wheel to zoom (esc)">
              Pan
            </button>
          </div>
          <input id="draft-label" type="text" placeholder={`Label (default: ${sample.head || "region"})`} value={p.draftLabel} onChange={(e) => p.setDraftLabel(e.target.value)} />
        </div>
        <ul className="annlist">
          {annotations.length === 0 && <li className="muted">None yet. Drag on the image to draw a box.</li>}
          {annotations.map((a) => (
            <li key={a.id}>
              <span className={`kind ${a.geometry.type}`}>{a.geometry.type === "tag" ? "verdict" : a.geometry.type}</span>
              <span className="lbl">
                {a.label}
                <small>{describeGeometry(a)}</small>
              </span>
              <button type="button" className="del" onClick={() => p.onDelete(a.id)} title="Delete">
                ×
              </button>
            </li>
          ))}
        </ul>
        <div className="io">
          <button type="button" onClick={exportJson}>
            Export JSON
          </button>
          <button type="button" onClick={() => fileRef.current?.click()}>
            Import JSON
          </button>
          <input
            ref={fileRef}
            id="import-file"
            type="file"
            accept="application/json"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void importJson(f);
              e.target.value = "";
            }}
          />
          <span className="muted">{status || (src.label === "static" ? "Stored in this browser." : "Stored in SQLite via the backend.")}</span>
        </div>
      </section>
    </aside>
  );
}
