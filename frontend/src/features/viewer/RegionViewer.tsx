import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { TransformComponent, TransformWrapper } from "react-zoom-pan-pinch";
import type { ReactZoomPanPinchRef } from "react-zoom-pan-pinch";
import { useDataSource } from "../../api/source";
import { BoxTool } from "../annotate/BoxTool";
import type { ViewerProps } from "./types";

function useFit(w: number, h: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: w, height: h });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const fit = () => {
      const cw = el.clientWidth - 24;
      const ch = el.clientHeight - 24;
      const width = Math.max(160, Math.min(cw, (ch * w) / h));
      setSize({ width, height: (width * h) / w });
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
  }, [w, h]);
  return { ref, size };
}

export function RegionViewer({ sample, condition, annotations, tool, showRegion, draftLabel, heat, onCreate }: ViewerProps) {
  const src = useDataSource();
  const url = src.fileUrl(sample, condition);
  const [w, h] = sample.image_size ?? [500, 334];
  const [bx, by, bw, bh] = sample.box;
  const { ref, size } = useFit(w, h);
  const zoom = useRef<ReactZoomPanPinchRef>(null);
  // When the frame is refitted (window or panel resized), re-centre instead of remounting the zoom state.
  useEffect(() => {
    zoom.current?.resetTransform(0);
  }, [size.width, size.height]);
  const fontSize = Math.max(11, w / 40);

  return (
    <div className={`viewer tool-${tool}`} ref={ref}>
      <TransformWrapper
        key={sample.id}
        ref={zoom}
        minScale={0.5}
        maxScale={12}
        centerOnInit
        limitToBounds={false}
        panning={{ disabled: tool === "box" }}
        doubleClick={{ disabled: true }}
        wheel={{ step: 0.15 }}
      >
        {({ zoomIn, zoomOut, resetTransform }) => (
          <>
            <div className="viewer-controls">
              <button type="button" onClick={() => zoomIn()} title="Zoom in">
                +
              </button>
              <button type="button" onClick={() => zoomOut()} title="Zoom out">
                −
              </button>
              <button type="button" onClick={() => resetTransform()} title="Fit the image to the window again (undo zoom and pan)">
                Fit view
              </button>
            </div>
            <TransformComponent wrapperClass="viewer-wrap" contentClass="viewer-content">
              <div className="frame" style={{ width: size.width, height: size.height }}>
                {url ? <img src={url} alt={sample.phrase} draggable={false} /> : <div className="no-image">No image for this condition</div>}
                <svg className="overlay" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
                  {heat && (
                    <g className="heat" aria-label={`${heat.region_tokens.length} pooled image tokens`}>
                      {heat.region_tokens.map((i) => {
                        const r = Math.floor(i / heat.cols);
                        const c = i % heat.cols;
                        return <rect key={`p${i}`} className="pooled" x={(c * w) / heat.cols} y={(r * h) / heat.rows} width={w / heat.cols} height={h / heat.rows} />;
                      })}
                    </g>
                  )}
                  {showRegion && (
                    <g className="region">
                      <rect x={bx} y={by} width={bw} height={bh} />
                      <text x={bx + 4} y={Math.max(fontSize + 2, by - 5)} fontSize={fontSize}>
                        {sample.head}
                      </text>
                    </g>
                  )}
                  {annotations.map((a) => {
                    const g = a.geometry;
                    if (g.type === "bbox")
                      return (
                        <g key={a.id} className={`ann ${a.source}`}>
                          <rect x={g.x * w} y={g.y * h} width={g.w * w} height={g.h * h} />
                          <text x={g.x * w + 4} y={g.y * h + fontSize + 2} fontSize={fontSize}>
                            {a.label}
                          </text>
                        </g>
                      );
                    if (g.type === "point")
                      return (
                        <g key={a.id} className={`ann ${a.source}`}>
                          <circle cx={g.x * w} cy={g.y * h} r={fontSize / 2} />
                          <text x={g.x * w + fontSize} y={g.y * h + fontSize / 2} fontSize={fontSize}>
                            {a.label}
                          </text>
                        </g>
                      );
                    return null;
                  })}
                  <BoxTool
                    active={tool === "box"}
                    width={w}
                    height={h}
                    onDone={(geometry) => onCreate({ sample_id: sample.id, label: draftLabel.trim() || sample.head || "region", geometry })}
                  />
                </svg>
              </div>
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
    </div>
  );
}
