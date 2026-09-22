import { useMemo, useState } from "react";
import type { ActivationVector } from "../../types/api";
import { diverging, symmetricRange } from "./heat";

interface Props {
  label: string;
  state: "loading" | "ready" | "error";
  error?: string;
  data?: ActivationVector;
  onClose(): void;
}

const COLS = 8;

/** The region's activation, reduced to 64 principal components and drawn as an 8 x 8 heatmap. */
export function ActivationPanel({ label, state, error, data, onClose }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const range = useMemo(() => (data ? symmetricRange(data.reduced) : 1), [data]);
  const rows = data ? Math.ceil(data.reduced.length / COLS) : 0;
  const explainedTotal = data ? data.explained.reduce((a, b) => a + b, 0) : 0;

  return (
    <section className="subwindow" aria-label="Activation vector">
      <header>
        <strong>Activation of “{label}”</strong>
        <span className="muted">
          {data ? ` · hidden state ${data.layer} (layer ${data.layer - 1} output) · mean of ${data.region_tokens.length} tokens · ${data.dim} dimensions reduced to ${data.reduced.length}` : ""}
        </span>
        <button type="button" className="hide" onClick={onClose} title="Close">
          ×
        </button>
      </header>
      {state === "loading" && <p className="muted">Computing: one forward pass of Qwen2.5-VL, mean-pooling the tokens under the box, projecting onto 64 components…</p>}
      {state === "error" && <p className="err">{error}</p>}
      {state === "ready" && data && (
        <div className="vec">
          <div className="grid">
            <div className="cells" style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)` }} onMouseLeave={() => setHover(null)}>
              {data.reduced.map((x, i) => (
                <div
                  key={i}
                  className={`cell ${hover === i ? "hot" : ""}`}
                  style={{ background: diverging(x / range) }}
                  onMouseEnter={() => setHover(i)}
                  title={`component ${i}: ${x.toFixed(2)} (explains ${(100 * data.explained[i]).toFixed(1)}% of variance)`}
                >
                  <span>{i}</span>
                </div>
              ))}
            </div>
            <div className="axis">
              <span>−{range.toFixed(1)}</span>
              <span className="colorbar" />
              <span>+{range.toFixed(1)}</span>
            </div>
            <p className="muted small">
              {rows} rows × {COLS} columns, component 0 top left. Blue negative, red positive; the range is the 99th percentile of the magnitudes.{" "}
              {hover !== null
                ? `Component ${hover}: ${data.reduced[hover].toFixed(2)}, explains ${(100 * data.explained[hover]).toFixed(1)}% of the variance.`
                : "Hover a cell for its value."}
            </p>
          </div>
          <dl className="kv">
            <dt>Basis</dt>
            <dd className="small">
              {data.basis}. These {data.reduced.length} components explain {(100 * explainedTotal).toFixed(0)}% of the variance of those vectors.
            </dd>
            <dt>Full vector</dt>
            <dd>
              {data.dim} dims, L2 norm {data.norm.toFixed(1)}
            </dd>
            <dt>Largest dims</dt>
            <dd>
              {data.top.map((t) => (
                <span key={t.dim} className="dim">
                  d{t.dim} {t.value > 0 ? "+" : ""}
                  {t.value.toFixed(1)}
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
