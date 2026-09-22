import type { Tool } from "../stores/viewerStore";

interface Props {
  tool: Tool;
  setTool(t: Tool): void;
  showRegion: boolean;
  toggleRegion(): void;
  hasRegion: boolean;
  showHeat: boolean;
  toggleHeat(): void;
  heatState: "idle" | "loading" | "ready" | "error";
  heatError?: string;
  heatRange?: { lo: number; hi: number };
}

/** Controls that act on the image: the drawing tool and the dataset's own region. */
export function StageBar({ tool, setTool, showRegion, toggleRegion, hasRegion, showHeat, toggleHeat, heatState, heatError, heatRange }: Props) {
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
      <button
        type="button"
        id="heatmap"
        className={`toggle heat ${showHeat ? "on" : ""}`}
        disabled={!hasRegion}
        aria-pressed={showHeat}
        onClick={toggleHeat}
        title="Colour each image token by the size of its activation at the read-out layer (hidden state 21, the output of layer 20 of Qwen2.5-VL)"
      >
        {heatState === "loading" ? "Computing…" : "Activation heatmap"}
      </button>
      {showHeat && heatState === "error" && <span className="err small">{heatError}</span>}
      {showHeat && heatState === "ready" && heatRange && (
        <span className="legend" title="L2 norm of the token's hidden state; colours span the 2nd to 98th percentile">
          <span className="bar" /> {Math.round(heatRange.lo)} – {Math.round(heatRange.hi)}
        </span>
      )}
    </div>
  );
}
