import { useEffect, useRef } from "react";
import { renderHeatmapThumbnail } from "./heatmap";

interface Props {
  imageUrl: string;
  heatmapRaw: Uint8Array | null;
  showHeatmap: boolean;
  aspect: string;
  label: string;
  onClick?: () => void;
}

/** A small "Original"/"AI highlight" thumbnail — reused by the clinician
 * "Where the AI looked" card and the patient result card's own thumbnails. */
export function HeatmapThumb({ imageUrl, heatmapRaw, showHeatmap, aspect, label, onClick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!showHeatmap || !heatmapRaw || !canvasRef.current) return;
    const rendered = renderHeatmapThumbnail(heatmapRaw);
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    canvasRef.current.width = rendered.width;
    canvasRef.current.height = rendered.height;
    ctx.clearRect(0, 0, rendered.width, rendered.height);
    ctx.drawImage(rendered, 0, 0);
  }, [heatmapRaw, showHeatmap]);

  const frame = (
    <div className="psc-thumb-frame" style={{ aspectRatio: aspect }}>
      <img src={imageUrl} alt={label} className="psc-thumb-image" />
      {showHeatmap && heatmapRaw && <canvas ref={canvasRef} className="psc-thumb-heatmap" />}
    </div>
  );

  if (onClick) {
    return (
      <button type="button" className="psc-thumb" onClick={onClick}>
        {frame}
        <span className="psc-thumb-label">{label}</span>
      </button>
    );
  }
  return (
    <div className="psc-thumb">
      {frame}
      <span className="psc-thumb-label">{label}</span>
    </div>
  );
}
