import { useEffect, useState } from "react";
import { useAnnotations, useCreateAnnotation, useDatasets, useDeleteAnnotation, useSamples } from "./api/hooks";
import { Header } from "./components/Header";
import { Inspector } from "./components/Inspector";
import { SampleList } from "./components/SampleList";
import { SettingsPanel } from "./components/SettingsPanel";
import { viewers } from "./features/viewer/registry";
import { useSettings } from "./stores/settingsStore";
import { useViewer } from "./stores/viewerStore";
import type { ConditionKey } from "./types/api";

// HW1 shows the original image only. The edited conditions stay in the data for later homeworks.
const CONDITION: ConditionKey = "orig";

export default function App() {
  const [q, setQ] = useState("");
  const datasets = useDatasets();
  const dataset = datasets.data?.[0];
  const samples = useSamples(dataset?.id, q);
  const v = useViewer();
  const setMode = useSettings((s) => s.setMode);
  const mode = useSettings((s) => s.mode);

  const list = samples.data;
  const sample = list?.find((s) => s.id === v.selectedId) ?? null;
  useEffect(() => {
    if (list && list.length && !list.some((s) => s.id === v.selectedId)) v.select(list[0].id);
  }, [list, v.selectedId, v.select]);

  const annotations = useAnnotations(sample?.id);
  const create = useCreateAnnotation();
  const remove = useDeleteAnnotation(sample?.id);

  // Keyboard: arrows or j/k move between regions, b = draw box, esc = pan.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (!list?.length) return;
      const i = list.findIndex((s) => s.id === v.selectedId);
      if (e.key === "ArrowDown" || e.key === "j") {
        v.select(list[Math.min(list.length - 1, i + 1)].id);
        e.preventDefault();
      } else if (e.key === "ArrowUp" || e.key === "k") {
        v.select(list[Math.max(0, i - 1)].id);
        e.preventDefault();
      } else if (e.key === "b") v.setTool("box");
      else if (e.key === "Escape") v.setTool("select");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [list, v]);

  const Viewer = viewers[dataset?.kind ?? "region"] ?? viewers.region;
  const error = datasets.error ?? samples.error ?? annotations.error;

  return (
    <div className="app">
      <div className="chrome">
        <Header dataset={dataset} />
        {error && (
          <div className="banner" role="alert">
            <span>{(error as Error).message}</span>
            {mode === "api" && (
              <button type="button" onClick={() => setMode("static")}>
                Switch to static mode
              </button>
            )}
          </div>
        )}
      </div>
      <main className="panes">
        <SampleList samples={list} loading={samples.isPending} selectedId={v.selectedId} onSelect={v.select} q={q} onQuery={setQ} />
        <section className="stage" aria-label="Viewer">
          {sample && dataset ? (
            <Viewer
              sample={sample}
              dataset={dataset}
              condition={CONDITION}
              annotations={annotations.data ?? []}
              tool={v.tool}
              showRegion={v.showRegion}
              draftLabel={v.draftLabel}
              onCreate={(a) => create.mutate(a)}
            />
          ) : (
            <div className="empty">{datasets.isPending || samples.isPending ? "Loading…" : "Select a region"}</div>
          )}
        </section>
        <Inspector
          sample={sample}
          dataset={dataset}
          condition={CONDITION}
          annotations={annotations.data ?? []}
          onCreate={(a) => create.mutate(a)}
          onDelete={(id) => remove.mutate(id)}
          tool={v.tool}
          setTool={v.setTool}
          draftLabel={v.draftLabel}
          setDraftLabel={v.setDraftLabel}
          showRegion={v.showRegion}
          toggleRegion={v.toggleRegion}
        />
      </main>
      <SettingsPanel />
    </div>
  );
}
