import { useRef, useState } from "react";
import { useDataSource } from "../api/source";
import { useDescribeRegion, useImportAnnotations } from "../api/hooks";
import { MentionTags } from "../features/annotate/MentionTags";
import type { Annotation, AnnotationCreate, Arm, ConditionKey, Dataset, Geometry, Sample, Verdict } from "../types/api";

interface Props {
  sample: Sample | null;
  dataset: Dataset | undefined;
  condition: ConditionKey;
  annotations: Annotation[];
  onCreate(a: AnnotationCreate): void;
  onDelete(id: string): void;
  draftLabel: string;
  setDraftLabel(s: string): void;
  showRegion: boolean;
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

function describeGeometry(a: Annotation, size: number[] | null): string {
  const g = a.geometry;
  const [w, h] = size ?? [1, 1];
  if (g.type === "bbox") return `x ${Math.round(g.x * w)}, y ${Math.round(g.y * h)}, ${Math.round(g.w * w)}×${Math.round(g.h * h)} px`;
  if (g.type === "point") return `x ${Math.round(g.x * w)}, y ${Math.round(g.y * h)} px`;
  return "verdict on the default region's read-out";
}

export function Inspector(p: Props) {
  const { sample, dataset, condition, annotations, showRegion } = p;
  const src = useDataSource();
  const importMut = useImportAnnotations();
  const describe = useDescribeRegion(sample?.id);
  const fileRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<string>("");
  const [asking, setAsking] = useState<string | null>(null);
  const [askError, setAskError] = useState<{ id: string; message: string } | null>(null);

  if (!sample || !dataset) return <aside className="inspector" />;
  const arm = primaryArm(dataset);
  const text = arm ? sample.readouts[arm.key]?.[condition] : undefined;
  const verdict = arm ? sample.judge[arm.key]?.[condition]?.orig : undefined;
  const regions = annotations.filter((a) => a.geometry.type !== "tag" && a.source === "human");
  const tags = annotations.filter((a) => a.geometry.type === "tag");
  const sameBox = (a: Geometry, b: Geometry) =>
    a.type === "bbox" && b.type === "bbox" && ["x", "y", "w", "h"].every((k) => Math.abs((a as never as Record<string, number>)[k] - (b as never as Record<string, number>)[k]) < 1e-6);
  const answersFor = (r: Annotation) => annotations.filter((a) => a.source === "model" && sameBox(a.geometry, r.geometry));
  const ask = async (r: Annotation) => {
    setAsking(r.id);
    setAskError(null);
    try {
      await describe.mutateAsync(r.geometry);
    } catch (e) {
      setAskError({ id: r.id, message: (e as Error).message });
    } finally {
      setAsking(null);
    }
  };

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
        <h2>Image</h2>
        <dl className="kv">
          <dt>Source</dt>
          <dd>Visual Genome {sample.image_id}</dd>
          <dt>Size</dt>
          <dd>{sample.image_size?.join(" × ")} px</dd>
        </dl>
      </section>

      <section>
        <h2>Your regions</h2>
        <input id="draft-label" type="text" placeholder="Label for the next region you draw" value={p.draftLabel} onChange={(e) => p.setDraftLabel(e.target.value)} />
        {regions.length > 0 && (
          <div className="toolrow">
            <button
              type="button"
              className="clear"
              onClick={() => {
                if (window.confirm(`Remove all ${regions.length} region${regions.length === 1 ? "" : "s"} on this image, including the model's answers?`))
                  for (const a of annotations.filter((x) => x.geometry.type !== "tag")) p.onDelete(a.id);
              }}
              title="Delete every box on this image"
            >
              Clear all regions
            </button>
          </div>
        )}
        <ul className="annlist">
          {regions.length === 0 && <li className="muted">None yet. Drag on the image to draw one.</li>}
          {regions.map((a) => (
            <li key={a.id} className="region">
              <div className="row">
                <span className={`kind ${a.geometry.type}`}>{a.geometry.type}</span>
                <span className="lbl">
                  {a.label}
                  <small>{describeGeometry(a, sample.image_size)}</small>
                </span>
                <button type="button" className="ask" disabled={asking !== null} onClick={() => void ask(a)} title="Send this region to Qwen2.5-VL and ask what it is">
                  {asking === a.id ? "Asking…" : "Ask Qwen2.5-VL"}
                </button>
                <button type="button" className="del" onClick={() => p.onDelete(a.id)} title="Delete">
                  ×
                </button>
              </div>
              {askError?.id === a.id && <p className="err small">{askError.message}</p>}
              {answersFor(a).map((m) => (
                <div key={m.id} className="answer">
                  <span className="swatch model" />
                  <span>
                    <strong>{m.label}</strong>
                    {m.note && <> — {m.note}</>}
                  </span>
                  <button type="button" className="del" onClick={() => p.onDelete(m.id)} title="Delete this answer">
                    ×
                  </button>
                </div>
              ))}
            </li>
          ))}
        </ul>
        <p className="muted small">Ask Qwen2.5-VL, the same vision-language model the research reads activations from, what a region you drew contains. It answers from the image directly.</p>
      </section>

      <section>
        <h2>Default region</h2>
        {!showRegion ? (
          <p className="muted">Click <strong>Default region</strong> above the image to reveal the region this dataset defined, and the read-out the verbalizer produced for it.</p>
        ) : (
          <>
            <p className="phrase">“{sample.phrase}”</p>
            <dl className="kv">
              <dt>Head noun</dt>
              <dd>{sample.head}</dd>
              <dt>Box (px)</dt>
              <dd>
                x {sample.box[0]}, y {sample.box[1]}, {sample.box[2]}×{sample.box[3]}
              </dd>
            </dl>
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
                <MentionTags sampleId={sample.id} arm={arm.key} condition={condition} annotations={tags} onCreate={p.onCreate} onDelete={p.onDelete} />
              </article>
            )}
            {arm?.note && <p className="muted small">{arm.note}.</p>}
          </>
        )}
      </section>

      <section>
        <h2>Save</h2>
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
