// Real multi-point Grad-CAM sampling for the 3D lung model (owner-directed
// follow-up to Phase 2, 2026-09-28: "can grad cam be shown in 3d model...
// same way its shown in normal 2d way by changing opacity"). Downsamples
// the same real raw 224x224 heatmap the 2D viewer colorizes
// (viewer/heatmap.ts) into a small set of weighted points the 3D shader
// renders as soft gaussian splats -- never a fabricated/procedural
// pattern. Reuses the 2D viewer's own opacity/highlight-area semantics
// exactly (viewer/heatmap.ts's `renderHeatmapCanvas`: opacity scales
// overall strength, highlightArea is a 0..1 threshold below which a
// pixel/cell contributes nothing) so the two views represent the same
// underlying data the same way, just rendered differently.
export interface HeatPoint {
  side: "right" | "left";
  u: number;
  v: number;
  w: number; // 0..1, already opacity-scaled
}

const HEATMAP_SIZE = 224;
const GRID = 16; // cells per axis -- coarse enough to stay well under HEAT_MAX after
// top-K selection, fine enough to trace the real heatmap's shape rather than
// reducing to a handful of zone blobs.
const HEAT_MAX_POINTS = 16; // must match lung-model.html's own `HEAT_MAX` GLSL/JS constant

export function sampleHeatPoints(raw: Uint8Array | null, opacity: number, highlightArea: number): HeatPoint[] {
  if (!raw || raw.length < HEATMAP_SIZE * HEATMAP_SIZE) return [];

  const cell = HEATMAP_SIZE / GRID;
  const threshold = Math.round(highlightArea * 255);
  const candidates: { u: number; v: number; value: number }[] = [];

  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      // Max-pool within the cell: a representative peak, not a blurred
      // average, so a small sharp hotspot doesn't get diluted away.
      let peak = 0;
      const y0 = Math.floor(gy * cell);
      const y1 = Math.floor((gy + 1) * cell);
      const x0 = Math.floor(gx * cell);
      const x1 = Math.floor((gx + 1) * cell);
      for (let py = y0; py < y1; py++) {
        const rowOffset = py * HEATMAP_SIZE;
        for (let px = x0; px < x1; px++) {
          const value = raw[rowOffset + px] ?? 0;
          if (value > peak) peak = value;
        }
      }
      if (peak < threshold) continue;
      candidates.push({ u: (gx + 0.5) / GRID, v: (gy + 0.5) / GRID, value: peak });
    }
  }

  candidates.sort((a, b) => b.value - a.value);

  return candidates.slice(0, HEAT_MAX_POINTS).map((c) => ({
    side: c.u < 0.5 ? "right" : "left", // radiographic convention, matches app/presentation.py::compute_focus
    u: c.u,
    v: c.v,
    w: Math.min(1, (c.value / 255) * opacity),
  }));
}
