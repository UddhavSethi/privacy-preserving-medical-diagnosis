import { useEffect, useMemo, useRef, useState } from "react";
import type { Phase, ProgressProps, ResultProps, StudyProps } from "../contract";
import { LungFrame, type LungFrameHandle, type LungFrameState } from "./LungFrame";
import { LayersPanel } from "./LayersPanel";
import { ViewPanel } from "./ViewPanel";
import { InfoPanel } from "./InfoPanel";
import { ZoneChips } from "./ZoneChips";
import { XraySection } from "./XraySection";
import { ConfidenceRing } from "./ConfidenceRing";
import { sampleHeatPoints } from "./heatPoints";
import { DEFAULT_LAYERS, type CameraView, type Selection } from "./types";

export type ExplorerView = "3d" | "xray";

interface Props {
  mode: "clinician" | "patient";
  phase: Phase;
  study: StudyProps | null;
  displayImageUrl: string | null;
  heatmapRaw: Uint8Array | null;
  result: ResultProps | null;
  progress: ProgressProps | null;
  reducedMotion: boolean;
  aiHighlightOn: boolean;
  setAiHighlightOn: (v: boolean) => void;
  /** Same Opacity/Highlight-area state the 2D viewer's own sliders already
   * own (App.tsx) -- reused as-is so the 3D heat cloud represents the
   * identical real data the same way, not a separate/duplicate control. */
  opacity: number;
  highlightArea: number;
  /** Bumped by the result card's "Open 3D" button to force the 3D view and scroll here. */
  openSignal: number;
}

const STAGE_ORDER = ["mc_dropout", "ood", "gradcam"] as const;
const ANALYZING_LABELS = ["Analyzing chest X-ray", "Mapping model attention", "Generating visualization"];

export function Explorer3D({
  mode,
  phase,
  study,
  displayImageUrl,
  heatmapRaw,
  result,
  progress,
  reducedMotion,
  aiHighlightOn,
  setAiHighlightOn,
  opacity,
  highlightArea,
  openSignal,
}: Props) {
  const sectionRef = useRef<HTMLDivElement>(null);
  const lungFrameRef = useRef<LungFrameHandle>(null);

  const [explorerView, setExplorerView] = useState<ExplorerView>("xray");
  const [selection, setSelection] = useState<Selection>(null);
  const [layers, setLayers] = useState(DEFAULT_LAYERS);
  const [breathing, setBreathing] = useState(!reducedMotion);
  const [autoRotate, setAutoRotate] = useState(!reducedMotion);
  const [cameraView, setCameraView] = useState<CameraView | null>(null);
  const [webglFailed, setWebglFailed] = useState(false);

  // A genuinely new study resets 3D state too -- otherwise a selection or
  // camera position from a previous image would silently carry over onto
  // an unrelated one (same reasoning as App.tsx's own viewer-prefs reset).
  const studyId = study?.study_id ?? null;
  useEffect(() => {
    setSelection(null);
    setLayers(DEFAULT_LAYERS);
    setExplorerView("xray");
  }, [studyId]);

  useEffect(() => {
    if (reducedMotion) {
      setBreathing(false);
      setAutoRotate(false);
    }
  }, [reducedMotion]);

  // "Open 3D" (ResultClinician's footer button) forces the 3D view and
  // scrolls the section into view. Skips the very first render (openSignal
  // starts at 0) so mounting doesn't itself trigger a scroll.
  const firstOpenSignal = useRef(true);
  useEffect(() => {
    if (firstOpenSignal.current) {
      firstOpenSignal.current = false;
      return;
    }
    if (webglFailed) return;
    setExplorerView("3d");
    sectionRef.current?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openSignal]);

  const focus = aiHighlightOn ? result?.focus ?? null : null;

  // Real multi-point cloud from the same raw heatmap bytes the 2D viewer
  // colorizes -- recomputed only when the underlying data/controls actually
  // change (heatmapRaw is itself already content-stabilized upstream by
  // App.tsx's useStableBytes, so this doesn't re-run on every ~0.3s poll
  // tick for an unchanged heatmap).
  const heatPoints = useMemo(
    () => (aiHighlightOn ? sampleHeatPoints(heatmapRaw, opacity, highlightArea) : []),
    [aiHighlightOn, heatmapRaw, opacity, highlightArea]
  );

  const lungState: LungFrameState = {
    focus,
    showFocus: aiHighlightOn,
    heatPoints,
    select: selection,
    mode,
    layers,
    breathing,
    autoRotate,
    scan: phase === "analyzing" ? { active: true, progress: progress?.fraction ?? 0 } : { active: false },
  };

  const stageDone = (id: (typeof STAGE_ORDER)[number]) => progress?.stages.find((s) => s.id === id)?.status === "done";

  const header =
    mode === "clinician"
      ? { eyebrow: "3D anatomical explorer", title: "Lungs and AI focus region", sub: "Explore lung anatomy alongside where the AI focused, in 3D or on the original X-ray." }
      : { eyebrow: "3D lung explanation", title: "Explore the lungs", sub: "See a 3D model of the lungs alongside what the AI noticed." };

  return (
    <section className="psc-explorer" ref={sectionRef}>
      <div className="psc-explorer-header">
        <div className="psc-explorer-eyebrow">{header.eyebrow}</div>
        <h2 className="psc-explorer-title">{header.title}</h2>
        <p className="psc-explorer-sub">{header.sub}</p>
      </div>

      <div className="psc-explorer-frame">
        <div
          className="psc-explorer-layer-3d"
          style={{
            opacity: explorerView === "3d" ? 1 : 0,
            transform: reducedMotion ? "none" : explorerView === "3d" ? "scale(1)" : "scale(1.04)",
            pointerEvents: explorerView === "3d" ? "auto" : "none",
          }}
        >
          <LungFrame
            ref={lungFrameRef}
            state={lungState}
            onSelect={setSelection}
            onViewName={setCameraView}
            onReady={() => {}}
            onTimeout={() => {
              setWebglFailed(true);
              setExplorerView("xray");
            }}
          />
        </div>
        <div
          className="psc-explorer-layer-xray"
          style={{
            opacity: explorerView === "xray" ? 1 : 0,
            transform: reducedMotion ? "none" : explorerView === "xray" ? "scale(1)" : "scale(0.97)",
            pointerEvents: explorerView === "xray" ? "auto" : "none",
          }}
        >
          <XraySection displayImageUrl={displayImageUrl} heatmapRaw={heatmapRaw} focus={focus} onSelectFocusBox={(sel) => { setSelection(sel); setExplorerView("3d"); }} />
        </div>

        <div className="psc-explorer-topcenter">
          <div className="psc-segmented psc-segmented--dark">
            <button type="button" className={`psc-seg-btn${explorerView === "xray" ? " psc-seg-btn--active" : ""}`} onClick={() => setExplorerView("xray")}>
              X-ray
            </button>
            <button
              type="button"
              className={`psc-seg-btn${explorerView === "3d" ? " psc-seg-btn--active" : ""}`}
              disabled={webglFailed}
              onClick={() => setExplorerView("3d")}
            >
              3D anatomy
            </button>
          </div>
        </div>

        {mode === "clinician" && explorerView === "3d" && (
          <div className="psc-explorer-topleft">
            <LayersPanel layers={layers} setLayers={setLayers} aiFocusOn={aiHighlightOn} setAiFocusOn={setAiHighlightOn} hasFocus={!!result?.focus} />
          </div>
        )}

        {explorerView === "3d" && (
          <div className="psc-explorer-topright">
            <ViewPanel
              activeView={cameraView}
              onView={(v) => lungFrameRef.current?.sendView(v)}
              onZoomIn={() => lungFrameRef.current?.sendView("zoomIn")}
              onZoomOut={() => lungFrameRef.current?.sendView("zoomOut")}
              onReset={() => lungFrameRef.current?.sendView("reset")}
              autoRotate={autoRotate}
              setAutoRotate={setAutoRotate}
              breathing={breathing}
              setBreathing={setBreathing}
            />
          </div>
        )}

        {explorerView === "3d" && selection && (
          <div className="psc-explorer-right">
            <InfoPanel selection={selection} mode={mode} result={result} />
          </div>
        )}

        <div className="psc-explorer-bottomleft">
          {mode === "clinician" ? (
            result ? <ConfidenceRing result={result} /> : null
          ) : result ? (
            <div className="psc-explorer-panel psc-explorer-patient-note">
              {result.focus_spread || !result.focus
                ? "The AI didn't point to one specific area."
                : `The AI noticed something in the ${result.focus.side} lung, ${result.focus.zone} area.`}
            </div>
          ) : null}
        </div>

        {phase === "analyzing" && (
          <div className="psc-explorer-bottomcenter">
            <div className="psc-explorer-panel psc-explorer-analyzing">
              <div className="psc-explorer-analyzing-stages">
                {STAGE_ORDER.map((id, i) => (
                  <span key={id} className={`psc-explorer-analyzing-stage${stageDone(id) ? " psc-explorer-analyzing-stage--done" : ""}`}>
                    {ANALYZING_LABELS[i]}
                  </span>
                ))}
              </div>
              <div className="psc-progress-track">
                <div className="psc-progress-fill" style={{ width: `${(progress?.fraction ?? 0) * 100}%` }} />
              </div>
            </div>
          </div>
        )}

        {webglFailed && (
          <div className="psc-explorer-webgl-note">3D view isn't available in this browser — showing the X-ray view instead.</div>
        )}
      </div>

      <ZoneChips onSelectZone={(zone) => { setSelection({ kind: "zone", side: "both", zone }); setExplorerView("3d"); }} focus={focus} />
    </section>
  );
}
