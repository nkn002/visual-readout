import { useEffect, useMemo, useRef, useState } from "react";
import type { ActivationVector } from "../../types/api";
import { diverging, gridShape, symmetricRange } from "./heat";

interface Props {
  label: string;
  state: "loading" | "ready" | "error";
  error?: string;
  data?: ActivationVector;
  onClose(): void;
}

/** The region's activation vector as a heatmap: one cell per dimension, row-major, colours centred at zero. */
export function ActivationPanel({ label, state, error, data, onClose }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ i: number; v: number } | null>(null);
  const shape = useMemo(() => (data ? gridShape(data.dim) : { rows: 1, cols: 1 }), [data]);
  const range = useMemo(() => (data ? symmetricRange(data.vector) : 1), [data]);
  const stats = useMemo(() => {
    if (!data) return null;
    const v = data.vector;
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    const std = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length);
    const top = v.map((x, i) => ({ i, x })).sort((a, b) => Math.abs(b.x) - Math.abs(a.x)).slice(0, 8);
    return { mean, std, top };
  }, [data]);

  useEffect(() => {
    const el = canvas.current;
    if (!el || !data) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    el.width = shape.cols;
    el.height = shape.rows;
    const img = ctx.createImageData(shape.cols, shape.rows);
    for (let i = 0; i < data.vector.length; i++) {
      const m = /rgb\((\d+), (\d+), (\d+)\)/.exec(diverging(data.vector[i] / range))!;
      img.data.set([+m[1], +m[2], +m[3], 255], i * 4);
    }
    ctx.putImageData(img, 0, 0);
  }, [data, shape, range]);

  return (
    <section className="subwindow" aria-label="Activation vector">
      <header>
        <strong>Activation of “{label}”</strong>
        <span className="muted">
          {data ? ` · hidden state ${data.layer} (layer ${data.layer - 1} output) · ${data.dim} dimensions · mean of ${data.region_tokens.length} tokens` : ""}
        </span>
        <button type="button" className="hide" onClick={onClose} title="Close">
          ×
        </button>
      </header>
      {state === "loading" && <p className="muted">Computing: one forward pass of Qwen2.5-VL, then mean-pooling the tokens under the box…</p>}
      {state === "error" && <p className="err">{error}</p>}
      {state === "ready" && data && stats && (
        <div className="vec">
          <div className="grid">
            <canvas
              ref={canvas}
              style={{ aspectRatio: `${shape.cols} / ${shape.rows}` }}
              onMouseMove={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                const c = Math.min(shape.cols - 1, Math.floor(((e.clientX - r.left) / r.width) * shape.cols));
                const row = Math.min(shape.rows - 1, Math.floor(((e.clientY - r.top) / r.height) * shape.rows));
                const i = row * shape.cols + c;
                setHover({ i, v: data.vector[i] });
              }}
              onMouseLeave={() => setHover(null)}
            />
            <div className="axis">
              <span>dim 0</span>
              <span className="colorbar" />
              <span>dim {data.dim - 1}</span>
            </div>
            <p className="muted small">
              Row-major, {shape.rows} rows × {shape.cols} columns. Colour: blue −{range.toFixed(1)} … white 0 … red +{range.toFixed(1)} (99th percentile of |value|).
              {hover ? ` Dimension ${hover.i}: ${hover.v.toFixed(3)}` : " Hover a cell for its value."}
            </p>
          </div>
          <dl className="kv">
            <dt>L2 norm</dt>
            <dd>{data.norm.toFixed(1)}</dd>
            <dt>Mean</dt>
            <dd>{stats.mean.toFixed(3)}</dd>
            <dt>Std</dt>
            <dd>{stats.std.toFixed(3)}</dd>
            <dt>Largest</dt>
            <dd>
              {stats.top.map((t) => (
                <span key={t.i} className="dim">
                  d{t.i} {t.x > 0 ? "+" : ""}
                  {t.x.toFixed(1)}
                </span>
              ))}
            </dd>
            <dt>Pooling</dt>
            <dd className="small">{data.pooling}</dd>
          </dl>
        </div>
      )}
    </section>
  );
}
