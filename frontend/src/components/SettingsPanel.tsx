import { useState } from "react";
import { ApiSource } from "../api/apiSource";
import { DEFAULT_API_BASE, useSettings } from "../stores/settingsStore";
import { useViewer } from "../stores/viewerStore";

export function SettingsPanel() {
  const open = useViewer((s) => s.settingsOpen);
  const setOpen = useViewer((s) => s.setSettingsOpen);
  const { mode, apiBase, setMode, setApiBase } = useSettings();
  const [draft, setDraft] = useState(apiBase);
  const [result, setResult] = useState<string>("");
  if (!open) return null;

  const check = async () => {
    setResult("Checking…");
    try {
      const h = await new ApiSource(draft.trim().replace(/\/+$/, "")).health();
      const m = h.model_ready ? `Qwen2.5-VL ready at ${h.model_url}` : h.model_url ? `model server at ${h.model_url} not answering` : "model server not running (sbatch scripts/model_server.sbatch)";
      setResult(`Connected: ${h.datasets} dataset(s), ${h.samples} samples. ${m}.`);
    } catch (e) {
      setResult(`Failed: ${(e as Error).message}`);
    }
  };
  const docsUrl = `${draft.trim().replace(/\/+$/, "") || ""}/docs`;

  return (
    <div className="scrim" onClick={() => setOpen(false)}>
      <div className="panel" role="dialog" aria-labelledby="settings-title" onClick={(e) => e.stopPropagation()}>
        <h2 id="settings-title">Data source</h2>
        <label className="radio">
          <input type="radio" name="mode" id="mode-static" checked={mode === "static"} onChange={() => setMode("static")} />
          <span>
            <strong>Static</strong> — read the bundled sample; annotations stay in this browser (export them as JSON).
          </span>
        </label>
        <label className="radio">
          <input type="radio" name="mode" id="mode-api" checked={mode === "api"} onChange={() => setMode("api")} />
          <span>
            <strong>Backend</strong> — FastAPI + SQLite; annotations persist in the database.
          </span>
        </label>
        <label className="field">
          Backend URL
          <input id="api-base" type="url" value={draft} placeholder={DEFAULT_API_BASE || "(same origin, via the dev proxy)"} onChange={(e) => setDraft(e.target.value)} />
        </label>
        <div className="row">
          <button type="button" onClick={check}>
            Check connection
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => {
              setApiBase(draft);
              setMode("api");
              setOpen(false);
            }}
          >
            Use backend
          </button>
          <a href={docsUrl} target="_blank" rel="noreferrer">
            API docs
          </a>
        </div>
        {result && <p className="result">{result}</p>}
        <p className="muted small">
          Start it with <code>uvicorn app.main:app --reload</code> in <code>backend/</code>. From this public page, Chrome asks once for
          permission to reach localhost.
        </p>
        <button type="button" className="close" onClick={() => setOpen(false)} aria-label="Close">
          ×
        </button>
      </div>
    </div>
  );
}
