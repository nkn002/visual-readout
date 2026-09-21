import { useEffect, useRef } from "react";
import { useDataSource } from "../api/source";
import type { Sample } from "../types/api";

interface Props {
  samples: Sample[] | undefined;
  loading: boolean;
  selectedId: string | null;
  onSelect(id: string): void;
}

export function SampleList({ samples, loading, selectedId, onSelect }: Props) {
  const src = useDataSource();
  const listRef = useRef<HTMLUListElement>(null);
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>('[aria-current="true"]')?.scrollIntoView({ block: "nearest" });
  }, [selectedId]);
  return (
    <aside className="samples">
      <p className="listhead">Images</p>
      <ul ref={listRef}>
        {loading && <li className="muted">Loading…</li>}
        {samples?.map((s) => (
          <li key={s.id}>
            <button type="button" aria-current={s.id === selectedId} onClick={() => onSelect(s.id)}>
              <img src={src.fileUrl(s, "orig")} alt="" loading="lazy" />
              <span className="txt">
                <span className="phrase">Image {s.index.toString().padStart(2, "0")}</span>
                <span className="meta">Visual Genome {s.image_id}</span>
              </span>
            </button>
          </li>
        ))}
        {samples && samples.length === 0 && <li className="muted">No images.</li>}
      </ul>
      <p className="hint">↑ ↓ change image · drag on the image to draw your region · esc to pan · b to draw again</p>
    </aside>
  );
}
