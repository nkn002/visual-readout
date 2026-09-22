import type { Tool } from "../stores/viewerStore";

interface Props {
  tool: Tool;
  setTool(t: Tool): void;
  showRegion: boolean;
  toggleRegion(): void;
  hasRegion: boolean;
}

/** Controls that act on the image: the drawing tool and the dataset's own region. */
export function StageBar({ tool, setTool, showRegion, toggleRegion, hasRegion }: Props) {
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
    </div>
  );
}
