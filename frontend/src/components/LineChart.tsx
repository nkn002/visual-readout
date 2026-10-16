export interface Series { name: string; colour: string; dashed?: boolean; points: [number, number][] }

/** A small SVG line chart: epochs on x, one value on y, with axis labels and a legend. */
export function LineChart({ series, xLabel, yLabel, title }: { series: Series[]; xLabel: string; yLabel: string; title: string }) {
  const W = 420, H = 270, L = 48, R = 12, T = 26, B = 76;
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = (y1 - y0) * 0.08 || 0.1;
  const lo = y0 - pad, hi = y1 + pad;
  const sx = (x: number) => L + ((x - x0) / Math.max(1e-9, x1 - x0)) * (W - L - R);
  const sy = (y: number) => T + (1 - (y - lo) / (hi - lo)) * (H - T - B);
  const yTicks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  const xTicks = Array.from(new Set([x0, Math.round((x0 + x1) / 2), x1]));
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
      <text x={W / 2} y={15} textAnchor="middle" className="title">{title}</text>
      {yTicks.map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} className="grid" />
          <text x={L - 6} y={sy(v) + 4} textAnchor="end" className="tick">{v.toFixed(2)}</text>
        </g>
      ))}
      {xTicks.map((v) => (
        <text key={v} x={sx(v)} y={H - B + 16} textAnchor="middle" className="tick">{v}</text>
      ))}
      <text x={(L + W - R) / 2} y={H - B + 32} textAnchor="middle" className="axis">{xLabel}</text>
      <text x={12} y={(T + H - B) / 2} textAnchor="middle" className="axis" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>{yLabel}</text>
      {series.map((s) => (
        <polyline key={s.name} fill="none" stroke={s.colour} strokeWidth={2} strokeDasharray={s.dashed ? "5 4" : undefined}
          points={s.points.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")} />
      ))}
      {series.map((s, i) => (
        <g key={`l${s.name}`} transform={`translate(${L + (i % 2) * 180}, ${H - B + 50 + Math.floor(i / 2) * 15})`}>
          <line x1={0} x2={18} y1={0} y2={0} stroke={s.colour} strokeWidth={2} strokeDasharray={s.dashed ? "5 4" : undefined} />
          <text x={24} y={4} className="legend">{s.name}</text>
        </g>
      ))}
    </svg>
  );
}
