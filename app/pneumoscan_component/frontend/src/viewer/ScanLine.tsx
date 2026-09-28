interface Props {
  /** 0..1, real backend progress fraction — never a fake timer. */
  fraction: number;
  reducedMotion: boolean;
}

/** The cyan sweep line shown while analysis is running, following real
 * progress (README: "During analysis: a cyan scan line with glow sweeps
 * down"). With `prefers-reduced-motion`, renders a static line at the
 * current position instead of relying on a CSS transition to animate it. */
export function ScanLine({ fraction, reducedMotion }: Props) {
  const topPct = Math.min(100, Math.max(0, fraction * 100));
  return (
    <>
      <div className="psc-scan-shade" style={{ height: `${topPct}%` }} />
      <div
        className={reducedMotion ? "psc-scan-line psc-scan-line--static" : "psc-scan-line"}
        style={{ top: `${topPct}%` }}
      />
    </>
  );
}
