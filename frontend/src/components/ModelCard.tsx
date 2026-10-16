import { useModelReport } from "../api/hooks";
import { useViewer } from "../stores/viewerStore";
import { LineChart } from "./LineChart";

const MLP = "MLP";
const LIN = "Softmax regression";
const pct = (x?: number) => (x === undefined ? "–" : `${(100 * x).toFixed(1)}%`);

/** HW2 model card: what the classifier is, how it trained, and how it scores on held-out data. */
export function ModelCard() {
  const open = useViewer((s) => s.modelOpen);
  const setOpen = useViewer((s) => s.setModelOpen);
  const q = useModelReport(open);
  if (!open) return null;
  const r = q.data;
  return (
    <div className="scrim" onClick={() => setOpen(false)}>
      <div className="panel wide" role="dialog" aria-labelledby="model-title" onClick={(e) => e.stopPropagation()}>
        <button type="button" className="close" onClick={() => setOpen(false)} aria-label="Close">×</button>
        <h2 id="model-title">Region classifier</h2>
        {q.isPending && <p className="muted">Loading the model report…</p>}
        {q.error && <p className="err">{(q.error as Error).message}</p>}
        {r && (
          <>
            <p>
              A multi-layer perceptron reads one region's activation, Qwen2.5-VL-7B's hidden state 21 mean-pooled over the image tokens under the box (3,584 numbers),
              and predicts which of {r.classes.length} categories the region shows. Each label is the head noun of the region's Visual Genome phrase,
              grouped into a category: for example dog, cat and horse are "domestic animal", and tree, grass and flower are "plant".
            </p>
            <dl className="kv">
              <dt>Architecture</dt>
              <dd>3584 → {r.models[MLP].config.hidden.join(" → ")} ({r.models[MLP].config.activation}, dropout {r.models[MLP].config.dropout}) → {r.classes.length}; {r.models[MLP].params.toLocaleString()} parameters</dd>
              <dt>Training</dt>
              <dd>Adam, learning rate {r.models[MLP].config.lr}, weight decay {r.models[MLP].config.weight_decay}, batch {r.models[MLP].config.batch_size}, cross-entropy; early stopping on validation loss (patience {r.models[MLP].config.patience}); chosen on validation accuracy from {r.models[MLP].grid.length} settings</dd>
              <dt>Data</dt>
              <dd>{r.n.train.toLocaleString()} train / {r.n.val.toLocaleString()} validation / {r.n.test.toLocaleString()} test regions, split by image so no photo appears in two sets</dd>
              <dt>Baseline</dt>
              <dd>Softmax regression: the same code with no hidden layer, tuned on the same grid of learning rate and weight decay</dd>
            </dl>
            <div className="charts">
              <LineChart
                title="Cross-entropy loss per epoch" xLabel="Epoch" yLabel="Loss"
                series={[
                  { name: "MLP train", colour: "#932667", points: r.models[MLP].history.map((h) => [h.epoch, h.train_loss]) },
                  { name: "MLP validation", colour: "#932667", dashed: true, points: r.models[MLP].history.map((h) => [h.epoch, h.val_loss]) },
                  { name: "Softmax train", colour: "#2a78d6", points: r.models[LIN].history.map((h) => [h.epoch, h.train_loss]) },
                  { name: "Softmax validation", colour: "#2a78d6", dashed: true, points: r.models[LIN].history.map((h) => [h.epoch, h.val_loss]) },
                ]}
              />
              <LineChart
                title="Accuracy per epoch" xLabel="Epoch" yLabel="Accuracy"
                series={[
                  { name: "MLP train", colour: "#932667", points: r.models[MLP].history.map((h) => [h.epoch, h.train_acc]) },
                  { name: "MLP validation", colour: "#932667", dashed: true, points: r.models[MLP].history.map((h) => [h.epoch, h.val_acc]) },
                  { name: "Softmax train", colour: "#2a78d6", points: r.models[LIN].history.map((h) => [h.epoch, h.train_acc]) },
                  { name: "Softmax validation", colour: "#2a78d6", dashed: true, points: r.models[LIN].history.map((h) => [h.epoch, h.val_acc]) },
                ]}
              />
            </div>
            <p className="muted small">
              Training continues for {r.models[MLP].config.patience} epochs past the lowest validation loss, then stops; the served weights are the ones from that lowest-loss epoch.
              The gap between train and validation accuracy shows overfitting, which dropout and early stopping limit.
            </p>
            <div className="tablewrap">
              <table className="metrics">
                <thead>
                  <tr><th>Model</th><th>Train acc</th><th>Val acc</th><th>Test acc</th><th>Test top-5</th><th>Test macro-F1</th></tr>
                </thead>
                <tbody>
                  <tr><td>Majority class ({r.majority_class})</td><td>{pct(r.majority.train.acc)}</td><td>{pct(r.majority.val.acc)}</td><td>{pct(r.majority.test.acc)}</td><td>–</td><td>–</td></tr>
                  {[LIN, MLP].map((m) => (
                    <tr key={m} className={m === MLP ? "ours" : ""}>
                      <td>{m === MLP ? "MLP (served)" : m}</td>
                      <td>{pct(r.models[m].train.acc)}</td><td>{pct(r.models[m].val.acc)}</td><td>{pct(r.models[m].test.acc)}</td>
                      <td>{pct(r.models[m].test.top5)}</td><td>{pct(r.models[m].test.macro_f1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="small">
              On the {r.test_gap_mlp_minus_softmax.n.toLocaleString()} test regions the MLP's accuracy is {(100 * r.test_gap_mlp_minus_softmax.mean).toFixed(1)} points higher than softmax regression,
              with a 95% paired-bootstrap interval of {(100 * r.test_gap_mlp_minus_softmax.ci95[0]).toFixed(1)} to {(100 * r.test_gap_mlp_minus_softmax.ci95[1]).toFixed(1)} points.
              {r.test_gap_mlp_minus_softmax.ci95[0] > 0
                ? " The interval excludes zero, so the MLP's lead is outside noise."
                : " The interval includes zero, so this test set does not separate the two models."}{" "}
              The MLP reaches {pct(r.models[MLP].test.acc)}, against {pct(r.majority.test.acc)} for always predicting the most common category.
            </p>
            <h3>Categories</h3>
            <p className="small muted">{r.classes.join(" · ")}</p>
            <h3>Test recall per category (MLP)</h3>
            <ol className="topk recall">
              {r.classes
                .map((c, i) => ({ c, v: r.models[MLP].test.recall_per_class?.[i] ?? 0 }))
                .sort((a, b) => b.v - a.v)
                .map(({ c, v }) => (
                  <li key={c}>
                    <span className="name">{c}</span>
                    <span className="bar"><span style={{ width: `${v * 100}%` }} /></span>
                    <span className="pct">{(v * 100).toFixed(0)}%</span>
                  </li>
                ))}
            </ol>
          </>
        )}
      </div>
    </div>
  );
}
