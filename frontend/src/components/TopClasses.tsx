import type { TopClass } from "../types/api";

/** Top-k classes as labelled probability bars. */
export function TopClasses({ top, truth }: { top: TopClass[]; truth?: string | null }) {
  return (
    <ol className="topk">
      {top.map((t) => (
        <li key={t.label} className={truth && t.label === truth ? "truth" : ""}>
          <span className="name">{t.label}</span>
          <span className="bar">
            <span style={{ width: `${Math.max(1, t.prob * 100)}%` }} />
          </span>
          <span className="pct">{(t.prob * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ol>
  );
}
