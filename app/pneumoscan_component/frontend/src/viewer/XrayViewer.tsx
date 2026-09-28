import { useEffect, useRef, useState } from "react";
import type { Focus, Phase, StudyProps } from "../contract";
import { fmt } from "../lib/na";
import { EmptyDropZone } from "./EmptyDropZone";
import { ScanLine } from "./ScanLine";
import { CompareSplit } from "./CompareSplit";
import { useZoomPan } from "./useZoomPan";
import { renderHeatmapCanvas } from "./heatmap";

export type ViewMode = "original" | "overlay" | "compare";

interface Props {
  study: StudyProps | null;
  displayImageUrl: string | null;
  heatmapRaw: Uint8Array | null;
  focus: Focus | null;
  focusSpread: boolean;
  phase: Phase;
  progressFraction: number;
  reducedMotion: boolean;
  onFileChosen: (file: File) => void;
  onUseSample: () => void;
  clientError: string | null;
  setClientError: (m: string | null) => void;
  viewMode: ViewMode;
  setViewMode: (m: ViewMode) => void;
  aiHighlightOn: boolean;
  setAiHighlightOn: (v: boolean) => void;
  opacity: number;
  setOpacity: (v: number) => void;
  highlightArea: number;
  setHighlightArea: (v: number) => void;
  comparePct: number;
  setComparePct: (v: number) => void;
  isFullscreen: boolean;
  onToggleFullscreen: () => void;
}

export function XrayViewer(props: Props) {
  const {
    study,
    displayImageUrl,
    heatmapRaw,
    focus,
    focusSpread,
    phase,
    progressFraction,
    reducedMotion,
    onFileChosen,
    onUseSample,
    clientError,
    setClientError,
    viewMode,
    setViewMode,
    aiHighlightOn,
    setAiHighlightOn,
    opacity,
    setOpacity,
    highlightArea,
    setHighlightArea,
    comparePct,
    setComparePct,
    isFullscreen,
    onToggleFullscreen,
  } = props;

  const viewportRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement>(null);
  const zp = useZoomPan(viewportRef);
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);

  const hasImage = !!displayImageUrl;
  const hasHeatmap = !!heatmapRaw && aiHighlightOn;
  const aspect = study ? `${study.width} / ${study.height}` : "1 / 1";

  const showOverlayLayer = viewMode === "overlay" && hasHeatmap;
  const showCompare = viewMode === "compare" && hasHeatmap;

  // The <canvas> ref below is only mounted in the DOM while
  // `showOverlayLayer || showCompare` (never rendered in "original" view, to
  // avoid paying for an offscreen colorize pass nobody sees) -- found live,
  // 2026-09-27: switching INTO overlay/compare view re-mounts a fresh,
  // never-drawn canvas, but this effect's own deps (heatmapRaw/opacity/
  // highlightArea) don't change on a view-mode switch, so it silently never
  // ran again and the canvas stayed blank at its default 300x150 size.
  // `showOverlayLayer || showCompare` must be a dependency so a fresh mount
  // gets drawn immediately, not just a later data/slider change.
  useEffect(() => {
    if (!heatmapRaw || !heatmapCanvasRef.current || !(showOverlayLayer || showCompare)) return;
    const rendered = renderHeatmapCanvas(heatmapRaw, opacity, highlightArea);
    const ctx = heatmapCanvasRef.current.getContext("2d");
    if (!ctx) return;
    heatmapCanvasRef.current.width = rendered.width;
    heatmapCanvasRef.current.height = rendered.height;
    ctx.clearRect(0, 0, rendered.width, rendered.height);
    ctx.drawImage(rendered, 0, 0);
  }, [heatmapRaw, opacity, highlightArea, showOverlayLayer, showCompare]);

  const focusText = phase === "review" ? (focusSpread ? "The AI's focus is spread out rather than in one area" : focus ? `Strongest focus: ${focus.side} lung, ${focus.zone} zone` : null) : null;

  return (
    <section className={`psc-viewer${isFullscreen ? " psc-viewer--fullscreen" : ""}`}>
      <div className="psc-viewer-toolbar">
        <div className="psc-segmented psc-segmented--dark">
          {(["original", "overlay", "compare"] as ViewMode[]).map((mode) => (
            <button
              key={mode}
              type="button"
              disabled={mode !== "original" && !hasHeatmap}
              className={`psc-seg-btn${viewMode === mode ? " psc-seg-btn--active" : ""}`}
              onClick={() => setViewMode(mode)}
            >
              {mode === "original" ? "Original" : mode === "overlay" ? "AI overlay" : "Compare"}
            </button>
          ))}
        </div>
        <div className="psc-viewer-tools">
          <button type="button" className="psc-icon-btn" title="Zoom out" onClick={zp.zoomOut}>
            −
          </button>
          <div className="psc-zoom-label">{zp.zoomPct}%</div>
          <button type="button" className="psc-icon-btn" title="Zoom in" onClick={zp.zoomIn}>
            +
          </button>
          <div className="psc-toolbar-sep" />
          <button type="button" className="psc-tool-btn" onClick={zp.fit}>
            Fit
          </button>
          <button type="button" className="psc-tool-btn" onClick={zp.reset}>
            Reset
          </button>
          <button
            type="button"
            className={`psc-tool-btn${zp.state.invert ? " psc-tool-btn--active" : ""}`}
            onClick={zp.toggleInvert}
          >
            Invert
          </button>
          <button type="button" className="psc-tool-btn" onClick={onToggleFullscreen}>
            {isFullscreen ? "Exit full screen" : "Full screen"}
          </button>
        </div>
      </div>

      <div
        ref={viewportRef}
        className="psc-viewport"
        style={{ cursor: zp.isPanning ? "grab" : hasImage ? "default" : "auto" }}
        onWheel={hasImage ? zp.onWheel : undefined}
        onPointerDown={hasImage ? zp.onPointerDown : undefined}
        onPointerMove={(e) => {
          zp.onPointerMove(e);
          const rect = viewportRef.current?.getBoundingClientRect();
          if (rect) setHover({ x: e.clientX - rect.left, y: e.clientY - rect.top });
        }}
        onPointerUp={zp.onPointerUp}
        onPointerLeave={(e) => {
          zp.onPointerLeave(e);
          setHover(null);
        }}
      >
        {hasImage ? (
          <>
            <div
              ref={stageRef}
              className="psc-stage"
              style={{
                aspectRatio: aspect,
                transform: `translate(-50%, -50%) translate(${zp.state.panX}px, ${zp.state.panY}px) scale(${zp.state.zoom})`,
              }}
            >
              <img
                src={displayImageUrl ?? undefined}
                alt="Chest radiograph"
                className="psc-stage-image"
                style={{ filter: zp.state.invert ? "invert(1)" : "none" }}
                draggable={false}
              />
              {(showOverlayLayer || showCompare) && (
                <canvas
                  ref={heatmapCanvasRef}
                  className="psc-stage-heatmap"
                  style={
                    showCompare
                      ? { clipPath: `inset(0 0 0 ${comparePct}%)` }
                      : { opacity: 1, transition: "opacity .9s ease" }
                  }
                />
              )}
              {showCompare && <CompareSplit pct={comparePct} onChange={setComparePct} containerRef={stageRef} />}
              {phase === "analyzing" && <ScanLine fraction={progressFraction} reducedMotion={reducedMotion} />}
            </div>

            <div className="psc-marker psc-marker--left">R</div>
            <div className="psc-marker psc-marker--right">L</div>

            <div className="psc-viewport-meta">
              <span className="psc-viewport-meta-strong">{study?.projection ? `${study.projection} chest radiograph` : "Chest radiograph"}</span>
              <span>{fmt(study?.study_id)}</span>
              <span>{study ? `${study.width} x ${study.height}px` : ""}</span>
            </div>

            {showCompare && (
              <div className="psc-compare-labels">
                <span>Left: original</span>
                <span>Right: AI highlight</span>
              </div>
            )}

            {hover && (
              <div className="psc-hover-readout">
                <span>
                  {Math.round(hover.x)}, {Math.round(hover.y)} px
                </span>
              </div>
            )}
            <div className="psc-scroll-hint">Scroll to zoom · drag to pan</div>
          </>
        ) : (
          <EmptyDropZone onChooseFile={onFileChosen} onUseSample={onUseSample} clientError={clientError} setClientError={setClientError} />
        )}
      </div>

      {hasImage && (
        <div className="psc-viewer-controls">
          <button type="button" className="psc-toggle-row" onClick={() => setAiHighlightOn(!aiHighlightOn)}>
            <span className={`psc-switch${aiHighlightOn ? " psc-switch--on" : ""}`}>
              <span className="psc-switch-knob" />
            </span>
            <span className="psc-toggle-label">AI highlight</span>
          </button>
          <div className="psc-viewer-sliders">
            <label className="psc-slider-row">
              <span>Opacity</span>
              <input
                type="range"
                min={0}
                max={100}
                value={Math.round(opacity * 100)}
                onChange={(e) => setOpacity(Number(e.target.value) / 100)}
              />
              <span className="psc-slider-value">{Math.round(opacity * 100)}%</span>
            </label>
            <label className="psc-slider-row">
              <span>Highlight area</span>
              <input
                type="range"
                min={0}
                max={100}
                value={Math.round(highlightArea * 100)}
                onChange={(e) => setHighlightArea(Number(e.target.value) / 100)}
              />
              <span className="psc-slider-value">{Math.round(highlightArea * 100)}%</span>
            </label>
          </div>
        </div>
      )}
      {focusText && <div className="psc-viewer-focus-caption">{focusText}</div>}
    </section>
  );
}
