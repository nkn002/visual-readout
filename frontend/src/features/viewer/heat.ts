import type { ActivationGrid } from "../../types/api";

// Five-stop "inferno"-like ramp, dark to bright.
const STOPS: [number, number, number][] = [
  [0, 0, 4],
  [66, 10, 104],
  [147, 38, 103],
  [221, 81, 58],
  [252, 165, 10],
  [252, 255, 164],
];

export function colorAt(t: number): string {
  const x = Math.min(1, Math.max(0, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(x));
  const f = x - i;
  const c = STOPS[i].map((a, k) => Math.round(a + (STOPS[i + 1][k] - a) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/** Normalise values to 0..1 between the 2nd and 98th percentile, so one attention-sink token does not flatten the rest. */
export function normalise(g: ActivationGrid): { t: number[]; lo: number; hi: number } {
  const sorted = [...g.values].sort((a, b) => a - b);
  const q = (p: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))];
  const lo = q(0.02);
  const hi = Math.max(q(0.98), lo + 1e-9);
  return { t: g.values.map((v) => (v - lo) / (hi - lo)), lo, hi };
}
