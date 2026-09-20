import { useEffect, useState } from "react";
import { useAnnotations, useCreateAnnotation, useDatasets, useDeleteAnnotation, useSamples } from "./api/hooks";
import { ConditionTabs } from "./components/ConditionTabs";
import { Header } from "./components/Header";
import { Inspector } from "./components/Inspector";
import { SampleList } from "./components/SampleList";
import { SettingsPanel } from "./components/SettingsPanel";
import { viewers } from "./features/viewer/registry";
import { useSettings } from "./stores/settingsStore";
import { useViewer } from "./stores/viewerStore";
import type { ConditionKey } from "./types/api";

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

  // Keyboard: arrows or j/k move, 1-4 pick a condition, b toggles the box tool, esc goes back to select.
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
      } else if (/^[1-4]$/.test(e.key) && dataset) {
        const c = dataset.conditions[Number(e.key) - 1];
        if (c) v.setCondition(c.key);
      } else if (e.key === "b") v.setTool(v.tool === "box" ? "select" : "box");
      else if (e.key === "Escape") v.setTool("select");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [list, dataset, v]);

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
          {dataset && <ConditionTabs conditions={dataset.conditions} sample={sample} value={v.condition} onChange={(c: ConditionKey) => v.setCondition(c)} />}
          {sample && dataset ? (
            <Viewer
              sample={sample}
              dataset={dataset}
              condition={v.condition}
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
          condition={v.condition}
          annotations={annotations.data ?? []}
          onCreate={(a) => create.mutate(a)}
          onDelete={(id) => remove.mutate(id)}
          tool={v.tool}
          setTool={v.setTool}
          draftLabel={v.draftLabel}
          setDraftLabel={v.setDraftLabel}
          showRegion={v.showRegion}
          toggleRegion={v.toggleRegion}
          hiddenArms={v.hiddenArms}
          toggleArm={v.toggleArm}
        />
      </main>
      <SettingsPanel />
    </div>
  );
}
