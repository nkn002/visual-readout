import { useSettings } from "../stores/settingsStore";
import { useViewer } from "../stores/viewerStore";
import type { Dataset } from "../types/api";

export function Header({ dataset }: { dataset: Dataset | undefined }) {
  const mode = useSettings((s) => s.mode);
  const apiBase = useSettings((s) => s.apiBase);
  const open = useViewer((s) => s.setSettingsOpen);
  const openModel = useViewer((s) => s.setModelOpen);
  const openTrain = useViewer((s) => s.setTrainOpen);
  return (
    <header className="topbar">
      <div className="brand">
        <strong>Visual read-out</strong>
        {dataset && (
          <span className="sub">
            {dataset.name} · {dataset.n_samples} regions
          </span>
        )}
      </div>
      <div className="right">
        <button type="button" className={`mode ${mode}`} onClick={() => open(true)} title="Data source settings">
          <span className="dot" />
          {mode === "static" ? "Static · annotations stay in this browser" : `Backend · ${apiBase || "dev proxy"}`}
        </button>
        <button type="button" className="mode model" onClick={() => openModel(true)} title="The trained region classifier: training curves and test results">
          Model
        </button>
        <button type="button" className="mode model" onClick={() => openTrain(true)} title="Train a classifier with your own settings and watch it learn">
          Train
        </button>
        <a href="https://github.com/nkn002/visual-readout" target="_blank" rel="noreferrer">
          GitHub
        </a>
      </div>
    </header>
  );
}
