import type { Tool } from "../stores/viewerStore";

interface Props {
  tool: Tool;
  setTool(t: Tool): void;
  showRegion: boolean;
  toggleRegion(): void;
  hasRegion: boolean;
  heat: { label: string; state: "loading" | "ready" | "error"; error?: string; range?: { lo: number; hi: number }; tokens?: number } | null;
  onHideHeat(): void;
}

/** Controls that act on the image: the drawing tool and the dataset's own region. */
export function StageBar({ tool, setTool, showRegion, toggleRegion, hasRegion, heat, onHideHeat }: Props) {
  return (
    <div className="stagebar">
      <div className="seg" role="group" aria-label="Tool">
        <button type="button" className={tool === "box" ? "on" : ""} onClick={() => setTool("box")} title="Drag on the image to draw your region (b)">
          Draw region
        </button>
        <button type="button" className={tool === "select" ? "on" : ""} onClick={() => setTool("select")} title="Drag to pan, wheel to zoom (esc)">
          Pan
        </button>
      </div>
      <button
        type="button"
        id="default-region"
        className={`toggle ${showRegion ? "on" : ""}`}
        disabled={!hasRegion}
        aria-pressed={showRegion}
        onClick={toggleRegion}
        title="Show the region this dataset defined for the image, with the read-out the verbalizer produced for it"
      >
        Default region
      </button>
      {heat && (
        <span className="heatinfo" title="Each image token is coloured by its cosine similarity to the mean-pooled vector of the tokens under the region, at hidden state 21 (the read-out layer, output of layer 20). Both are centred on the image's mean token. Outlined cells are the pooled tokens.">
          <span className="bar" /> heatmap: <strong>{heat.label}</strong>
          {heat.state === "loading" && " · computing…"}
          {heat.state === "error" && <span className="err"> · {heat.error}</span>}
          {heat.state === "ready" && heat.range && ` · ${heat.tokens} tokens pooled · cosine ${heat.range.lo.toFixed(2)} to ${heat.range.hi.toFixed(2)}`}
          <button type="button" className="hide" onClick={onHideHeat} title="Hide the heatmap">
            ×
          </button>
        </span>
      )}
    </div>
  );
}
