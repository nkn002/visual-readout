import { useSettings } from "../stores/settingsStore";
import { useViewer } from "../stores/viewerStore";
import type { Dataset } from "../types/api";

export function Header({ dataset }: { dataset: Dataset | undefined }) {
  const mode = useSettings((s) => s.mode);
  const apiBase = useSettings((s) => s.apiBase);
  const open = useViewer((s) => s.setSettingsOpen);
  return (
    <header className="topbar">
      <div className="brand">
        <strong>Visual read-out</strong>
        {dataset && (
          <span className="sub">
            {dataset.name} · {dataset.n_samples} regions · {dataset.arms.length} arms
          </span>
        )}
      </div>
      <div className="right">
        <button type="button" className={`mode ${mode}`} onClick={() => open(true)} title="Data source settings">
          <span className="dot" />
          {mode === "static" ? "Static · annotations stay in this browser" : `Backend · ${apiBase || "dev proxy"}`}
        </button>
        <a href="https://github.com/nkn002/visual-readout" target="_blank" rel="noreferrer">
          GitHub
        </a>
      </div>
    </header>
  );
}
