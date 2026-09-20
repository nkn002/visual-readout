import { useRef, useState } from "react";
import type { PointerEvent } from "react";
import type { Geometry } from "../../types/api";

interface Draft { x0: number; y0: number; x1: number; y1: number }

/** Drag-to-draw layer inside the viewer's SVG. Coordinates are in image pixels; the result is normalised. */
export function BoxTool({ active, width, height, onDone }: { active: boolean; width: number; height: number; onDone(g: Geometry): void }) {
  const [draft, setDraft] = useState<Draft | null>(null);
  const ref = useRef<SVGRectElement>(null);
  if (!active) return null;

  const toImage = (e: PointerEvent<SVGRectElement>) => {
    const r = ref.current!.getBoundingClientRect();
    const x = Math.min(width, Math.max(0, ((e.clientX - r.left) / r.width) * width));
    const y = Math.min(height, Math.max(0, ((e.clientY - r.top) / r.height) * height));
    return { x, y };
  };
  const rect = draft && {
    x: Math.min(draft.x0, draft.x1),
    y: Math.min(draft.y0, draft.y1),
    w: Math.abs(draft.x1 - draft.x0),
    h: Math.abs(draft.y1 - draft.y0),
  };

  return (
    <>
      <rect
        ref={ref}
        className="capture"
        x={0}
        y={0}
        width={width}
        height={height}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          const p = toImage(e);
          setDraft({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
        }}
        onPointerMove={(e) => {
          if (!draft) return;
          const p = toImage(e);
          setDraft({ ...draft, x1: p.x, y1: p.y });
        }}
        onPointerUp={() => {
          if (rect && rect.w >= 3 && rect.h >= 3) {
            onDone({ type: "bbox", x: rect.x / width, y: rect.y / height, w: rect.w / width, h: rect.h / height });
          }
          setDraft(null);
        }}
      />
      {rect && <rect className="draft" x={rect.x} y={rect.y} width={rect.w} height={rect.h} />}
    </>
  );
}
