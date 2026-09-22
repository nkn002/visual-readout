// Diverging blue-white-red ramp for a signed vector, centred at zero.
export function diverging(t: number): string {
  const x = Math.min(1, Math.max(-1, t));
  if (x < 0) {
    const a = -x;
    return `rgb(${Math.round(255 - 200 * a)}, ${Math.round(255 - 180 * a)}, 255)`;
  }
  return `rgb(255, ${Math.round(255 - 190 * x)}, ${Math.round(255 - 215 * x)})`;
}

/** Symmetric colour range: the 99th percentile of |v|, so a few huge dimensions do not hide the rest. */
export function symmetricRange(v: number[]): number {
  const abs = v.map(Math.abs).sort((a, b) => a - b);
  return Math.max(abs[Math.min(abs.length - 1, Math.round(0.99 * (abs.length - 1)))], 1e-9);
}

