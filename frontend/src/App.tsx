import { useEffect } from "react";
import { useActivations, useAnnotations, useCreateAnnotation, useDatasets, useDeleteAnnotation, useSamples } from "./api/hooks";
import { normalise } from "./features/viewer/heat";
import { Header } from "./components/Header";
import { Inspector } from "./components/Inspector";
import { SampleList } from "./components/SampleList";
import { SettingsPanel } from "./components/SettingsPanel";
import { StageBar } from "./components/StageBar";
import { viewers } from "./features/viewer/registry";
import { useSettings } from "./stores/settingsStore";
import { useViewer } from "./stores/viewerStore";
import type { ConditionKey } from "./types/api";

// HW1 shows the original image only. The edited conditions stay in the data for later homeworks.
const CONDITION: ConditionKey = "orig";
// hidden_states[21] is the output of layer 20: the read-out layer the research verbalizer reads.
const READOUT_LAYER = 21;

export default function App() {
  const datasets = useDatasets();
  const dataset = datasets.data?.[0];
  const samples = useSamples(dataset?.id, "");
  const v = useViewer();
  const setMode = useSettings((s) => s.setMode);
  const mode = useSettings((s) => s.mode);

  const list = samples.data;
  const sample = list?.find((s) => s.id === v.selectedId) ?? null;
  useEffect(() => {
    if (list && list.length && !list.some((s) => s.id === v.selectedId)) v.select(list[0].id);
  }, [list, v.selectedId, v.select]);
  // Every image starts plain: the dataset's own region is only shown on request.
  useEffect(() => v.setShowRegion(false), [v.selectedId, v.setShowRegion]);

  const annotations = useAnnotations(sample?.id);
  const heat = useActivations(sample?.id, READOUT_LAYER, v.showHeat);
  const heatState = !v.showHeat ? "idle" : heat.isPending ? "loading" : heat.error ? "error" : "ready";
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
        <SampleList samples={list} loading={samples.isPending} selectedId={v.selectedId} onSelect={v.select} />
        <section className="stage" aria-label="Viewer">
          <StageBar
            tool={v.tool}
            setTool={v.setTool}
            showRegion={v.showRegion}
            toggleRegion={v.toggleRegion}
            hasRegion={!!sample}
            showHeat={v.showHeat}
            toggleHeat={v.toggleHeat}
            heatState={heatState}
            heatError={heat.error ? (heat.error as Error).message : undefined}
            heatRange={heat.data ? (({ lo, hi }) => ({ lo, hi }))(normalise(heat.data)) : undefined}
          />
          {sample && dataset ? (
            <Viewer
              sample={sample}
              dataset={dataset}
              condition={CONDITION}
              annotations={annotations.data ?? []}
              tool={v.tool}
              showRegion={v.showRegion}
              draftLabel={v.draftLabel}
              heat={v.showHeat && heat.data ? heat.data : null}
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
          draftLabel={v.draftLabel}
          setDraftLabel={v.setDraftLabel}
          showRegion={v.showRegion}
        />
      </main>
      <SettingsPanel />
    </div>
  );
}
