import type { ResultProps, StudyProps } from "../../contract";
import { HeatmapThumb } from "../../viewer/HeatmapThumb";

interface Props {
  result: ResultProps;
  study: StudyProps | null;
  displayImageUrl: string | null;
  heatmapRaw: Uint8Array | null;
}

function headline(result: ResultProps): string {
  if (result.label === "Pneumonia") return "The AI noticed a pattern linked to pneumonia";
  if (result.label === "Normal") return "The AI did not notice a pattern linked to pneumonia";
  return "The AI could not give a clear result";
}

function body(result: ResultProps): string {
  if (result.label === "Pneumonia") {
    return "This does not mean you have pneumonia — it means the AI saw something worth a professional review.";
  }
  if (result.label === "Normal") {
    return "This does not rule out pneumonia or any other condition — it means the AI did not see a strong pattern in this image.";
  }
  return "The result was too close to the AI's decision boundary to call confidently either way.";
}

function imageCheckText(imageCheck: ResultProps["image_check"]): string {
  if (imageCheck === "Unusual") {
    return "Your image looked different from the ones the AI has learned from, so treat this result with extra caution.";
  }
  if (imageCheck === "Typical") {
    return "Your image looked similar to the ones the AI has learned from.";
  }
  return "Not available.";
}

const BANDS: Array<"Low" | "Medium" | "High"> = ["Low", "Medium", "High"];

export function ResultPatient({ result, study, displayImageUrl, heatmapRaw }: Props) {
  const aspect = study ? `${study.width} / ${study.height}` : "1 / 1";
  const band = result.certainty;

  return (
    <section className="psc-card psc-result-reveal">
      <div className="psc-result-body">
        <h2 className="psc-card-title">Your screening result</h2>
        <div className="psc-patient-headline">{headline(result)}</div>
        <p className="psc-card-body">{body(result)}</p>
      </div>

      <div className="psc-patient-section">
        <div className="psc-patient-row-head">
          <span className="psc-patient-row-label">How confident is the AI?</span>
          <span className="psc-patient-row-value">{band ?? "Not available"}</span>
        </div>
        <div className="psc-sure-segs">
          {BANDS.map((b) => (
            <div key={b} className="psc-sure-seg">
              <div className={`psc-sure-bar${band === b ? " psc-sure-bar--active" : ""}`} />
              <span className="psc-sure-name">{b}</span>
            </div>
          ))}
        </div>
        {result.deferred && (
          <div className="psc-note psc-note--warn">
            The AI is unsure about this image, so it has marked it for a closer look by a professional.
          </div>
        )}
      </div>

      {displayImageUrl && (
        <div className="psc-patient-section">
          <span className="psc-patient-row-label">Why did the AI flag this image?</span>
          <div className="psc-thumb-grid">
            <HeatmapThumb imageUrl={displayImageUrl} heatmapRaw={heatmapRaw} showHeatmap={false} aspect={aspect} label="Your X-ray" />
            <HeatmapThumb imageUrl={displayImageUrl} heatmapRaw={heatmapRaw} showHeatmap aspect={aspect} label="Areas the AI focused on" />
          </div>
          <p className="psc-card-footnote">
            The coloured areas show where the AI looked most closely when making its decision. Warmer colours mean
            more attention. They do not show the exact location of any illness.
          </p>
        </div>
      )}

      <div className="psc-patient-section">
        <span className="psc-patient-row-label">Was the image suitable?</span>
        <p className="psc-card-footnote">{imageCheckText(result.image_check)}</p>
      </div>

      <div className="psc-patient-cta">
        <span className="psc-patient-cta-title">What should I do?</span>
        <p className="psc-patient-cta-text">
          AI screening results should be reviewed by a qualified healthcare professional. This application does not
          provide a medical diagnosis.
        </p>
      </div>
    </section>
  );
}
