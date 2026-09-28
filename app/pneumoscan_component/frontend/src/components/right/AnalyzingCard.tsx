import { JOB_STAGE_LABELS } from "../../contract";
import type { ProgressProps } from "../../contract";

const STAGE_DETAIL: Record<string, string> = {
  gate: "Confirming this is a chest X-ray",
  preprocess: "Preparing the image for the model",
  mc_dropout: "Running the screening model",
  ood: "Comparing against training data",
  gradcam: "Building the visual explanation",
};

interface Props {
  progress: ProgressProps | null;
}

export function AnalyzingCard({ progress }: Props) {
  if (!progress) return null;

  if (progress.queued) {
    return (
      <section className="psc-card">
        <h2 className="psc-card-heading">Waiting for the analysis service…</h2>
        <p className="psc-card-body">Another screening is in progress. This one will start as soon as it's free.</p>
      </section>
    );
  }

  const pct = Math.round(progress.fraction * 100);

  return (
    <section className="psc-card">
      <div className="psc-analyzing-head">
        <h2 className="psc-card-heading">Analyzing</h2>
        <span className="psc-progress-pct">{pct}%</span>
      </div>
      <div className="psc-progress-track">
        <div className="psc-progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="psc-substeps">
        {progress.stages.map((stage) => (
          <div key={stage.id} className="psc-substep">
            <div className={`psc-substep-mark psc-substep-mark--${stage.status}`}>{stage.status === "done" ? "✓" : ""}</div>
            <div className="psc-substep-copy">
              <span className={`psc-substep-label psc-substep-label--${stage.status}`}>{JOB_STAGE_LABELS[stage.id]}</span>
              <span className="psc-substep-detail">{STAGE_DETAIL[stage.id]}</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
