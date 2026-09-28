// Colorizes the real, raw 224x224 uint8 Grad-CAM heatmap the backend sends
// (`app/presentation.py::heatmap_to_uint8_bytes`) onto an offscreen canvas,
// client-side — never a fabricated/procedural heatmap. Also builds the
// blend used by the viewer's "Opacity"/"Highlight area" sliders and the
// small "Original"/"AI highlight" thumbnails.
export const HEATMAP_SIZE = 224;

// A small, standard jet-style ramp (dark blue -> cyan -> yellow -> red),
// matching the "warmer = more attention" language used throughout the
// design (README/HTML). Linear interpolation between stops.
const JET_STOPS: Array<[number, number, number, number]> = [
  [0.0, 8, 15, 82],
  [0.25, 20, 120, 200],
  [0.5, 120, 210, 130],
  [0.75, 245, 200, 40],
  [1.0, 220, 40, 30],
];

function jetColor(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  for (let i = 0; i < JET_STOPS.length - 1; i += 1) {
    const [t0, r0, g0, b0] = JET_STOPS[i];
    const [t1, r1, g1, b1] = JET_STOPS[i + 1];
    if (clamped >= t0 && clamped <= t1) {
      const f = t1 === t0 ? 0 : (clamped - t0) / (t1 - t0);
      return [r0 + (r1 - r0) * f, g0 + (g1 - g0) * f, b0 + (b1 - b0) * f];
    }
  }
  return [JET_STOPS[JET_STOPS.length - 1][1], JET_STOPS[JET_STOPS.length - 1][2], JET_STOPS[JET_STOPS.length - 1][3]];
}

/**
 * Renders the raw intensity map to an offscreen canvas.
 *
 * `opacity` (0..1) is the sliders' overall alpha multiplier. `highlightArea`
 * (0..1) is the "Highlight area" slider — since we have real per-pixel data
 * (unlike the design prototype's static procedural blob, which faked this
 * with a CSS clip-path over a fixed image), it's implemented as a real
 * intensity cutoff: pixels below the threshold are fully transparent, so a
 * low value shows only the AI's strongest few regions and a high value
 * shows its full attention map.
 */
export function renderHeatmapCanvas(
  raw: Uint8Array,
  opacity: number,
  highlightArea: number,
  size = HEATMAP_SIZE
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;

  const imageData = ctx.createImageData(size, size);
  const threshold = Math.round(highlightArea * 255);
  for (let i = 0; i < size * size; i += 1) {
    const value = raw[i] ?? 0;
    const [r, g, b] = jetColor(value / 255);
    const alpha = value < threshold ? 0 : Math.round(opacity * 255);
    imageData.data[i * 4] = r;
    imageData.data[i * 4 + 1] = g;
    imageData.data[i * 4 + 2] = b;
    imageData.data[i * 4 + 3] = alpha;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

/** Full-opacity/full-area static render, for the small "AI highlight" thumbnail. */
export function renderHeatmapThumbnail(raw: Uint8Array, size = HEATMAP_SIZE): HTMLCanvasElement {
  return renderHeatmapCanvas(raw, 0.75, 0, size);
}
