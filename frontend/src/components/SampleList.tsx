import { useEffect, useRef } from "react";
import { useDataSource } from "../api/source";
import type { Sample } from "../types/api";

interface Props {
  samples: Sample[] | undefined;
  loading: boolean;
  selectedId: string | null;
  onSelect(id: string): void;
  q: string;
  onQuery(q: string): void;
}

export function SampleList({ samples, loading, selectedId, onSelect, q, onQuery }: Props) {
  const src = useDataSource();
  const listRef = useRef<HTMLUListElement>(null);
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>('[aria-current="true"]')?.scrollIntoView({ block: "nearest" });
  }, [selectedId]);
  return (
    <aside className="samples">
      <label className="search">
        <span className="visually-hidden">Filter regions</span>
        <input id="filter" type="search" placeholder="Filter by phrase or head noun" value={q} onChange={(e) => onQuery(e.target.value)} />
      </label>
      <ul ref={listRef}>
        {loading && <li className="muted">Loading…</li>}
        {samples?.map((s) => (
          <li key={s.id}>
            <button type="button" aria-current={s.id === selectedId} onClick={() => onSelect(s.id)}>
              <img src={src.fileUrl(s, "orig")} alt="" loading="lazy" />
              <span className="txt">
                <span className="phrase">{s.phrase}</span>
                <span className="meta">
                  #{s.index.toString().padStart(2, "0")} · {s.head} · {s.n_tokens ?? "?"} tokens
                </span>
              </span>
            </button>
          </li>
        ))}
        {samples && samples.length === 0 && <li className="muted">No regions match.</li>}
      </ul>
      <p className="hint">↑ ↓ move · drag on the image to draw a box · esc to pan · b to draw again</p>
    </aside>
  );
}
