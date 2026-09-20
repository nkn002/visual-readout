import type { Condition, ConditionKey, Sample } from "../types/api";

export function ConditionTabs({ conditions, sample, value, onChange }: { conditions: Condition[]; sample: Sample | null; value: ConditionKey; onChange(c: ConditionKey): void }) {
  return (
    <div className="tabs" role="tablist" aria-label="Image condition">
      {conditions.map((c, i) => {
        const has = !sample || !!sample.images[c.key];
        return (
          <button
            key={c.key}
            type="button"
            role="tab"
            aria-selected={value === c.key}
            className={value === c.key ? "on" : ""}
            disabled={!has}
            title={`${c.note ?? c.label} (key ${i + 1})`}
            onClick={() => onChange(c.key)}
          >
            <kbd>{i + 1}</kbd> {c.label}
          </button>
        );
      })}
    </div>
  );
}
