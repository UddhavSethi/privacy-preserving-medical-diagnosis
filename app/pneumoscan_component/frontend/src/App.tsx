import { useEffect, useMemo, useState } from "react";
import type { ComponentProps } from "streamlit-component-lib";
import { withStreamlitConnection } from "streamlit-component-lib";
import { fileToBase64, markReady, sendEvent, syncFrameHeightToViewport } from "./bridge";
import type { PneumoScanArgs, UploadPayload } from "./contract";
import { useObjectUrl, useStableBytes } from "./lib/useObjectUrl";
import { Header, type Mode } from "./components/Header";
import { StepBar } from "./components/StepBar";
import { StudyCard } from "./components/StudyCard";
import { ImageCheckCard } from "./components/ImageCheckCard";
import { UploadIntro } from "./components/right/UploadIntro";
import { QualityCard } from "./components/right/QualityCard";
import { AnalyzingCard } from "./components/right/AnalyzingCard";
import { ResultClinician } from "./components/right/ResultClinician";
import { ResultPatient } from "./components/right/ResultPatient";
import { WhereAILooked } from "./components/right/WhereAILooked";
import { RejectedCard } from "./components/right/RejectedCard";
import { XrayViewer, type ViewMode } from "./viewer/XrayViewer";
import { Explorer3D } from "./explorer/Explorer3D";

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(query.matches);
    const listener = () => setReduced(query.matches);
    query.addEventListener("change", listener);
    return () => query.removeEventListener("change", listener);
  }, []);
  return reduced;
}

function App({ args }: ComponentProps): JSX.Element {
  const data = (args as PneumoScanArgs).data;
  const rawArgs = args as PneumoScanArgs;

  useEffect(() => {
    markReady();
    return syncFrameHeightToViewport();
  }, []);

  const reducedMotion = usePrefersReducedMotion();
  const [mode, setMode] = useState<Mode>("clinician");
  const [viewMode, setViewMode] = useState<ViewMode>("original");
  const [aiHighlightOn, setAiHighlightOn] = useState(true);
  const [opacity, setOpacity] = useState(0.75);
  const [highlightArea, setHighlightArea] = useState(0);
  const [comparePct, setComparePct] = useState(50);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [clientError, setClientError] = useState<string | null>(null);
  const [open3DSignal, setOpen3DSignal] = useState(0);

  const displayImageUrl = useObjectUrl(rawArgs.display_image, "image/jpeg");
  const heatmapRaw = useStableBytes(rawArgs.heatmap);

  // A genuinely new study (new upload, or back to "upload") resets every
  // client-only viewer preference -- otherwise e.g. "Compare" mode or a
  // zoomed-in view would silently carry over onto an unrelated image.
  const studyId = data.study?.study_id ?? null;
  useEffect(() => {
    setViewMode("original");
    setAiHighlightOn(true);
    setOpacity(0.75);
    setHighlightArea(0);
    setComparePct(50);
    setIsFullscreen(false);
    setClientError(null);
  }, [studyId]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isFullscreen) setIsFullscreen(false);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [isFullscreen]);

  const onFileChosen = async (file: File) => {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const data_b64 = await fileToBase64(bytes);
    const payload: UploadPayload = { name: file.name, mime: file.type, size: bytes.length, data_b64 };
    sendEvent("upload", payload);
  };
  const onUseSample = () => sendEvent("use_sample");
  const onAnalyze = () => sendEvent("analyze");
  const onNewScreening = () => sendEvent("new_screening");
  const onDismissError = () => sendEvent("dismiss_error");

  const hasImage = data.study !== null;
  const pageTitle = mode === "clinician" ? "New screening" : "Check a chest X-ray";
  const pageSub =
    mode === "clinician"
      ? "Upload a chest X-ray, run the AI screening model, and review its explanation."
      : "Upload a chest X-ray photo to see what the AI notices.";

  const focus = data.result?.focus ?? null;
  const focusSpread = data.result?.focus_spread ?? true;

  const rightPanel = useMemo(() => {
    if (data.phase === "upload") return <UploadIntro mode={mode} />;
    if (data.phase === "quality") return <QualityCard quality={data.quality} mode={mode} onAnalyze={onAnalyze} />;
    if (data.phase === "analyzing") return <AnalyzingCard progress={data.progress} />;
    if (data.phase === "rejected") return <RejectedCard quality={data.quality} onNewScreening={onNewScreening} />;
    if (data.phase === "review" && data.result) {
      return mode === "clinician" ? (
        <>
          <ResultClinician result={data.result} setViewMode={setViewMode} onOpen3D={() => setOpen3DSignal((n) => n + 1)} />
          <WhereAILooked
            study={data.study}
            displayImageUrl={displayImageUrl}
            heatmapRaw={heatmapRaw}
            focus={focus}
            focusSpread={focusSpread}
            setViewMode={setViewMode}
          />
        </>
      ) : (
        <ResultPatient result={data.result} study={data.study} displayImageUrl={displayImageUrl} heatmapRaw={heatmapRaw} />
      );
    }
    return null;
  }, [data.phase, data.quality, data.progress, data.result, data.study, mode, displayImageUrl, heatmapRaw, focus, focusSpread]);

  if (data.phase === "error") {
    return (
      <div className="psc-app">
        <Header mode={mode} setMode={setMode} />
        <main className="psc-error-main">
          <section className="psc-card psc-card--warn psc-error-card">
            <h2 className="psc-card-heading">Something went wrong</h2>
            <p className="psc-card-body">{data.error ?? "An unexpected error occurred."}</p>
            <button type="button" className="psc-btn psc-btn--primary" onClick={onDismissError}>
              Start a new screening
            </button>
          </section>
        </main>
      </div>
    );
  }

  return (
    <div className="psc-app">
      <Header mode={mode} setMode={setMode} />
      <main className="psc-main">
        <div className="psc-page-head">
          <div className="psc-page-head-text">
            <h1 className="psc-page-title">{pageTitle}</h1>
            <div className="psc-page-sub">{pageSub}</div>
          </div>
          {hasImage && (
            <button type="button" className="psc-btn psc-btn--outline" onClick={onNewScreening}>
              Start a new screening
            </button>
          )}
        </div>

        <StepBar phase={data.phase} progress={data.progress} mode={mode} />

        <div className="psc-columns">
          <aside className="psc-col-left">
            <StudyCard study={data.study} phase={data.phase} />
            <ImageCheckCard quality={data.quality} />
          </aside>

          <XrayViewer
            study={data.study}
            displayImageUrl={displayImageUrl}
            heatmapRaw={heatmapRaw}
            focus={focus}
            focusSpread={focusSpread}
            phase={data.phase}
            progressFraction={data.progress?.fraction ?? 0}
            reducedMotion={reducedMotion}
            onFileChosen={onFileChosen}
            onUseSample={onUseSample}
            clientError={clientError}
            setClientError={setClientError}
            viewMode={viewMode}
            setViewMode={setViewMode}
            aiHighlightOn={aiHighlightOn}
            setAiHighlightOn={setAiHighlightOn}
            opacity={opacity}
            setOpacity={setOpacity}
            highlightArea={highlightArea}
            setHighlightArea={setHighlightArea}
            comparePct={comparePct}
            setComparePct={setComparePct}
            isFullscreen={isFullscreen}
            onToggleFullscreen={() => setIsFullscreen((v) => !v)}
          />

          <aside className="psc-col-right">{rightPanel}</aside>
        </div>

        {hasImage && (
          <Explorer3D
            mode={mode}
            phase={data.phase}
            study={data.study}
            displayImageUrl={displayImageUrl}
            heatmapRaw={heatmapRaw}
            result={data.result}
            progress={data.progress}
            reducedMotion={reducedMotion}
            aiHighlightOn={aiHighlightOn}
            setAiHighlightOn={setAiHighlightOn}
            opacity={opacity}
            highlightArea={highlightArea}
            openSignal={open3DSignal}
          />
        )}
      </main>
    </div>
  );
}

export default withStreamlitConnection(App);
