import { useEffect, useRef } from "react";
import type { Focus } from "../contract";
import { renderHeatmapCanvas } from "../viewer/heatmap";
import type { Selection } from "./types";

interface Props {
  displayImageUrl: string | null;
  heatmapRaw: Uint8Array | null;
  focus: Focus | null;
  onSelectFocusBox: (sel: Selection) => void;
}

// Zone boundary lines, exact percentages from the README's own X-ray-view
// spec ("dashed zone lines at 34.7% / 57.3%").
const UPPER_MIDDLE_PCT = 34.7;
const MIDDLE_LOWER_PCT = 57.3;

export function XraySection({ displayImageUrl, heatmapRaw, focus, onSelectFocusBox }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!heatmapRaw || !canvasRef.current) return;
    const rendered = renderHeatmapCanvas(heatmapRaw, 0.6, 0);
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    canvasRef.current.width = rendered.width;
    canvasRef.current.height = rendered.height;
    ctx.clearRect(0, 0, rendered.width, rendered.height);
    ctx.drawImage(rendered, 0, 0);
  }, [heatmapRaw]);

  return (
    <div className="psc-explorer-xray">
      {displayImageUrl && (
        <div className="psc-explorer-xray-stage">
          <img src={displayImageUrl} alt="Chest radiograph" className="psc-explorer-xray-image" draggable={false} />
          {heatmapRaw && <canvas ref={canvasRef} className="psc-explorer-xray-heatmap" />}

          <div className="psc-explorer-zone-line" style={{ top: `${UPPER_MIDDLE_PCT}%` }} />
          <div className="psc-explorer-zone-line" style={{ top: `${MIDDLE_LOWER_PCT}%` }} />
          <span className="psc-explorer-zone-label" style={{ top: `${UPPER_MIDDLE_PCT / 2}%` }}>
            Upper
          </span>
          <span className="psc-explorer-zone-label" style={{ top: `${(UPPER_MIDDLE_PCT + MIDDLE_LOWER_PCT) / 2}%` }}>
            Middle
          </span>
          <span className="psc-explorer-zone-label" style={{ top: `${(MIDDLE_LOWER_PCT + 100) / 2}%` }}>
            Lower
          </span>

          <div className="psc-marker psc-marker--left">R</div>
          <div className="psc-marker psc-marker--right">L</div>

          {focus && (
            <button
              type="button"
              className="psc-explorer-focus-box"
              style={{
                left: `${focus.u * 100}%`,
                top: `${focus.v * 100}%`,
              }}
              onClick={() => onSelectFocusBox({ kind: "zone", side: focus.side, zone: focus.zone })}
              title="Select this region in 3D"
            />
          )}
        </div>
      )}
      <div className="psc-explorer-xray-caption">Illustrative anatomical mapping — not a patient-specific 3D reconstruction.</div>
    </div>
  );
}
