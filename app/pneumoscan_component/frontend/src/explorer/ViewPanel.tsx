import type { CameraView } from "./types";

interface Props {
  activeView: CameraView | null;
  onView: (v: CameraView) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  autoRotate: boolean;
  setAutoRotate: (v: boolean) => void;
  breathing: boolean;
  setBreathing: (v: boolean) => void;
}

const VIEWS: CameraView[] = ["front", "left", "right", "top"];

export function ViewPanel({ activeView, onView, onZoomIn, onZoomOut, onReset, autoRotate, setAutoRotate, breathing, setBreathing }: Props) {
  return (
    <div className="psc-explorer-panel psc-explorer-panel--view">
      <div className="psc-explorer-caption">View</div>
      <div className="psc-explorer-view-radio">
        {VIEWS.map((v) => (
          <button
            key={v}
            type="button"
            className={`psc-explorer-radio-btn${activeView === v ? " psc-explorer-radio-btn--active" : ""}`}
            onClick={() => onView(v)}
          >
            {v[0].toUpperCase() + v.slice(1)}
          </button>
        ))}
      </div>
      <div className="psc-explorer-zoom-row">
        <button type="button" className="psc-icon-btn" title="Zoom out" onClick={onZoomOut}>
          −
        </button>
        <button type="button" className="psc-tool-btn" onClick={onReset}>
          Reset
        </button>
        <button type="button" className="psc-icon-btn" title="Zoom in" onClick={onZoomIn}>
          +
        </button>
      </div>
      <button type="button" className="psc-explorer-switch-row" onClick={() => setAutoRotate(!autoRotate)}>
        <span className={`psc-switch${autoRotate ? " psc-switch--on" : ""}`}>
          <span className="psc-switch-knob" />
        </span>
        <span>Auto-rotate</span>
      </button>
      <button type="button" className="psc-explorer-switch-row" onClick={() => setBreathing(!breathing)}>
        <span className={`psc-switch${breathing ? " psc-switch--on" : ""}`}>
          <span className="psc-switch-knob" />
        </span>
        <span>Breathing</span>
      </button>
    </div>
  );
}
