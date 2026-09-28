import type { Focus, StudyProps } from "../../contract";
import { HeatmapThumb } from "../../viewer/HeatmapThumb";
import type { ViewMode } from "../../viewer/XrayViewer";

interface Props {
  study: StudyProps | null;
  displayImageUrl: string | null;
  heatmapRaw: Uint8Array | null;
  focus: Focus | null;
  focusSpread: boolean;
  setViewMode: (m: ViewMode) => void;
}

export function WhereAILooked({ study, displayImageUrl, heatmapRaw, focus, focusSpread, setViewMode }: Props) {
  if (!displayImageUrl) return null;
  const aspect = study ? `${study.width} / ${study.height}` : "1 / 1";
  const focusText = focusSpread
    ? "The AI's focus is spread out rather than in one area"
    : focus
      ? `Strongest focus: ${focus.side} lung, ${focus.zone} zone`
      : null;

  return (
    <section className="psc-card">
      <h2 className="psc-card-title">Where the AI looked</h2>
      <div className="psc-thumb-grid">
        <HeatmapThumb
          imageUrl={displayImageUrl}
          heatmapRaw={heatmapRaw}
          showHeatmap={false}
          aspect={aspect}
          label="Original"
          onClick={() => setViewMode("original")}
        />
        <HeatmapThumb
          imageUrl={displayImageUrl}
          heatmapRaw={heatmapRaw}
          showHeatmap
          aspect={aspect}
          label="AI highlight"
          onClick={() => setViewMode("overlay")}
        />
      </div>
      {focusText && <div className="psc-focus-plain">{focusText}</div>}
      <p className="psc-card-footnote">Coloured areas influenced the result most. They are not an outline of disease.</p>
    </section>
  );
}
