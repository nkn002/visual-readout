import type { ProbeView, Sample } from "../types/api";
import { diverging, symmetricRange } from "../features/viewer/heat";

interface Props {
  label: string;
  sample: Sample;
  box: { x: number; y: number; w: number; h: number };
  imageUrl: string;
  state: "loading" | "ready" | "error";
  error?: string;
  data?: ProbeView;
  onClose(): void;
}

/** Input -> probe architecture -> output for one drawn region, with the real values at every stage. */
export function ProbeViewPanel({ label, box, imageUrl, state, error, data, onClose }: Props) {
  const crop = {
    backgroundImage: `url(${imageUrl})`,
    backgroundSize: `${100 / box.w}% ${100 / box.h}%`,
    backgroundPosition: `${box.w < 1 ? (box.x / (1 - box.w)) * 100 : 0}% ${box.h < 1 ? (box.y / (1 - box.h)) * 100 : 0}%`,
    aspectRatio: `${box.w} / ${box.h}`,
  };
  const out = data ? [...data.output].sort((a, b) => b.prob - a.prob) : [];
  return (
    <div className="scrim" onClick={onClose}>
      <div className="panel wide" role="dialog" aria-labelledby="probe-title" onClick={(e) => e.stopPropagation()}>
        <button type="button" className="close" onClick={onClose} aria-label="Close">×</button>
        <h2 id="probe-title">The probe on “{label}”</h2>
        {state === "loading" && <p className="muted">Running Qwen2.5-VL on the image, then the probe…</p>}
        {state === "error" && <p className="err">{error}</p>}
        {state === "ready" && data && (
          <div className="pipeline">
            <section className="stage-col">
              <h3>1 · Input</h3>
              <div className="crop" style={crop} />
              <p className="small">
                Qwen2.5-VL reads the whole image. Its hidden state {data.layer} (output of layer {data.layer - 1}) is averaged over the{" "}
                <strong>{data.n_tokens} image tokens</strong> under the box: one vector of <strong>{data.input.dim.toLocaleString()} numbers</strong> (norm {data.input.norm.toFixed(0)}).
              </p>
              <Grid values={data.input.preview} cols={16} />
              <p className="small muted">The vector after standardization, each cell the mean of {data.input.chunk} consecutive numbers. Blue below the training mean, red above.</p>
            </section>
            <div className="arrow" aria-hidden>→</div>
            <section className="stage-col">
              <h3>2 · Probe ({data.model})</h3>
              <ol className="layers">
                <li><span className="lname">Standardize</span><span className="muted">(x − training mean) / training std</span></li>
                {data.layers.map((l, i) => (
                  <li key={l.name} className="block">
                    <span className="lname">Linear {data.sizes[i].toLocaleString()} → {l.size} + {l.activation}</span>
                    <Grid values={l.values} cols={32} positive />
                    <span className="muted small">{l.active} of {l.size} hidden units active (above 0); brighter = larger</span>
                  </li>
                ))}
                {data.layers.length > 0 && <li><span className="lname">Dropout {data.dropout}</span><span className="muted">used in training only, off for prediction</span></li>}
                <li><span className="lname">Linear {data.sizes[data.sizes.length - 2].toLocaleString()} → {data.sizes[data.sizes.length - 1]}</span><span className="muted">one score (logit) per category</span></li>
                <li><span className="lname">Softmax</span><span className="muted">turns the 20 scores into probabilities that sum to 1</span></li>
              </ol>
            </section>
            <div className="arrow" aria-hidden>→</div>
            <section className="stage-col">
              <h3>3 · Output</h3>
              <ol className="topk">
                {out.map((o, i) => (
                  <li key={o.label} className={i === 0 ? "winner" : ""}>
                    <span className="name" title={`logit ${o.logit.toFixed(2)}`}>{o.label}</span>
                    <span className="bar"><span style={{ width: `${Math.max(0.5, o.prob * 100)}%` }} /></span>
                    <span className="pct">{(o.prob * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ol>
            </section>
          </div>
        )}
      </div>
    </div>
  );
}

function Grid({ values, cols, positive = false }: { values: number[]; cols: number; positive?: boolean }) {
  const max = positive ? Math.max(...values, 1e-9) : symmetricRange(values);
  return (
    <div className="vgrid" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
      {values.map((v, i) => (
        <span key={i} style={{ background: positive ? `rgba(147, 38, 103, ${Math.min(1, v / max)})` : diverging(v / max) }} title={v.toFixed(3)} />
      ))}
    </div>
  );
}
