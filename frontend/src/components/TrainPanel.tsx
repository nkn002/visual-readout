import { useState } from "react";
import { useDataSource } from "../api/source";
import { useTrainingRun } from "../api/hooks";
import { useSettings } from "../stores/settingsStore";
import { useViewer } from "../stores/viewerStore";
import type { TrainConfig } from "../types/api";
import { LineChart } from "./LineChart";

const DEFAULTS: TrainConfig = { lr: 3e-4, epochs: 15, hidden: [512], activation: "relu", dropout: 0.5, weight_decay: 1e-4, batch_size: 256, n_train: 8000, patience: 5 };
const pct = (x?: number) => (x === undefined ? "–" : `${(100 * x).toFixed(1)}%`);

/** HW2 optional: choose hyperparameters, train in the backend, and watch loss and accuracy update live. */
export function TrainPanel() {
  const open = useViewer((s) => s.trainOpen);
  const setOpen = useViewer((s) => s.setTrainOpen);
  const mode = useSettings((s) => s.mode);
  const src = useDataSource();
  const [cfg, setCfg] = useState<TrainConfig>(DEFAULTS);
  const [hiddenText, setHiddenText] = useState("512");
  const [runId, setRunId] = useState<string | null>(null);
  const [msg, setMsg] = useState("");
  const run = useTrainingRun(runId);
  if (!open) return null;

  const snap = run.data;
  const busy = snap ? ["queued", "running"].includes(snap.status) : false;
  const start = async () => {
    setMsg("");
    const hidden = hiddenText.split(/[ ,]+/).filter(Boolean).map(Number);
    if (hidden.some((h) => !Number.isInteger(h) || h < 1)) return setMsg("Hidden layers: whole numbers separated by commas, or empty for none.");
    try {
      const s = await src.startTraining({ ...cfg, hidden });
      setRunId(s.id);
    } catch (e) {
      setMsg((e as Error).message);
    }
  };
  const num = (k: keyof TrainConfig) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => setCfg({ ...cfg, [k]: Number(e.target.value) });

  return (
    <div className="scrim" onClick={() => setOpen(false)}>
      <div className="panel wide" role="dialog" aria-labelledby="train-title" onClick={(e) => e.stopPropagation()}>
        <button type="button" className="close" onClick={() => setOpen(false)} aria-label="Close">×</button>
        <h2 id="train-title">Train a classifier</h2>
        <p className="small muted">
          Trains the same network and loop as <code>training/train.py</code> in the backend, on the precomputed region vectors.
          Each epoch is evaluated on the training subset and the full validation set; the test set is scored when training ends.
        </p>
        {mode !== "api" && <p className="err">Training runs in the backend. Open Settings and switch to Backend.</p>}
        <form className="trainform" onSubmit={(e) => { e.preventDefault(); void start(); }}>
          <label>Learning rate
            <select id="t-lr" value={cfg.lr} onChange={num("lr")} disabled={busy}>
              {[1e-4, 3e-4, 1e-3, 3e-3, 1e-2].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label>Epochs (max)
            <input id="t-epochs" type="number" min={1} max={200} value={cfg.epochs} onChange={num("epochs")} disabled={busy} />
          </label>
          <label>Hidden layers
            <input id="t-hidden" type="text" value={hiddenText} placeholder="e.g. 512 or 512,256; empty = none" onChange={(e) => setHiddenText(e.target.value)} disabled={busy} />
          </label>
          <label>Activation
            <select id="t-act" value={cfg.activation} onChange={(e) => setCfg({ ...cfg, activation: e.target.value })} disabled={busy}>
              {["relu", "gelu", "tanh", "sigmoid"].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label>Dropout
            <input id="t-dropout" type="number" min={0} max={0.9} step={0.1} value={cfg.dropout} onChange={num("dropout")} disabled={busy} />
          </label>
          <label>Weight decay
            <select id="t-wd" value={cfg.weight_decay} onChange={num("weight_decay")} disabled={busy}>
              {[0, 1e-4, 1e-3, 1e-2].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label>Batch size
            <select id="t-bs" value={cfg.batch_size} onChange={num("batch_size")} disabled={busy}>
              {[64, 128, 256, 512].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label>Training regions
            <select id="t-n" value={cfg.n_train} onChange={num("n_train")} disabled={busy}>
              {[2000, 4000, 8000, 20766].map((v) => <option key={v} value={v}>{v.toLocaleString()}{v === 20766 ? " (all)" : ""}</option>)}
            </select>
          </label>
          <label>Early-stop patience
            <input id="t-pat" type="number" min={1} max={200} value={cfg.patience} onChange={num("patience")} disabled={busy} />
          </label>
          <div className="row">
            <button type="submit" className="primary" disabled={busy || mode !== "api"}>Start training</button>
            {busy && snap && <button type="button" disabled={snap.stopping} onClick={async () => { await src.cancelTraining(snap.id); void run.refetch(); }}>Stop</button>}
            <span className="muted small">{msg}</span>
          </div>
        </form>

        {snap && (
          <>
            <p className="small">
              <strong>{snap.stopping ? "Stopping after this epoch, then scoring" : snap.status === "running" ? "Training" : snap.status === "queued" ? "Starting" : snap.status === "done" ? "Finished" : snap.status === "cancelled" ? "Stopped" : "Failed"}</strong>
              {" "}· epoch {snap.rows.length} of up to {snap.config.epochs} · {snap.elapsed.toFixed(0)} s · hidden {snap.config.hidden.length ? snap.config.hidden.join(" → ") : "none (softmax regression)"}, {snap.config.activation}, lr {snap.config.lr}
              {snap.error && <span className="err"> · {snap.error}</span>}
            </p>
            {snap.rows.length > 0 && (
              <div className="charts">
                <LineChart title="Loss per epoch (live)" xLabel="Epoch" yLabel="Cross-entropy"
                  series={[
                    { name: "train", colour: "#932667", points: snap.rows.map((h) => [h.epoch, h.train_loss]) },
                    { name: "validation", colour: "#932667", dashed: true, points: snap.rows.map((h) => [h.epoch, h.val_loss]) },
                  ]} />
                <LineChart title="Accuracy per epoch (live)" xLabel="Epoch" yLabel="Accuracy"
                  series={[
                    { name: "train", colour: "#2a78d6", points: snap.rows.map((h) => [h.epoch, h.train_acc]) },
                    { name: "validation", colour: "#2a78d6", dashed: true, points: snap.rows.map((h) => [h.epoch, h.val_acc]) },
                  ]} />
              </div>
            )}
            {snap.result && (
              <>
                <div className="tablewrap">
                  <table className="metrics">
                    <thead><tr><th>Split</th><th>Loss</th><th>Accuracy</th><th>Top-5</th><th>Macro-F1</th></tr></thead>
                    <tbody>
                      {(["train", "val", "test"] as const).map((k) => (
                        <tr key={k}>
                          <td>{k === "train" ? `Train (${snap.result!.n_train.toLocaleString()} regions)` : k === "val" ? "Validation" : "Test"}</td>
                          <td>{snap.result!.metrics[k].loss?.toFixed(3)}</td><td>{pct(snap.result!.metrics[k].acc)}</td>
                          <td>{pct(snap.result!.metrics[k].top5)}</td><td>{pct(snap.result!.metrics[k].macro_f1)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="small muted">
                  {snap.result.params.toLocaleString()} parameters; weights from the epoch with the lowest validation loss. The shipped model scores 82.6% on the same test set.
                </p>
                <h3>Predictions on random test regions</h3>
                <ul className="examples">
                  {snap.result.examples.map((x, i) => (
                    <li key={i} className={x.pred === x.true ? "ok" : "bad"}>
                      <span className="mark">{x.pred === x.true ? "✓" : "✗"}</span>
                      <span className="ph">“{x.phrase}”</span>
                      <span>true <strong>{x.true}</strong>, predicted <strong>{x.pred}</strong> ({(x.prob * 100).toFixed(0)}%)</span>
                    </li>
                  ))}
                </ul>
                <div className="row">
                  {snap.in_use ? (
                    <button type="button" onClick={async () => { await src.resetModel(); void run.refetch(); }}>Go back to the shipped model for Predict</button>
                  ) : (
                    <button type="button" className="primary" onClick={async () => { await src.useTrainedModel(snap.id); void run.refetch(); }}>Use this model for Predict</button>
                  )}
                  <span className="muted small">{snap.in_use ? "Predict now uses this run's model (in memory only)." : "Predict uses the shipped model."}</span>
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
