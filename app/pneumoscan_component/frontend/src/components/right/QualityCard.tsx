import type { QualityProps } from "../../contract";

interface Props {
  quality: QualityProps | null;
  mode: "clinician" | "patient";
  onAnalyze: () => void;
}

export function QualityCard({ quality, mode, onAnalyze }: Props) {
  const hasIssue = !!quality && (!quality.resolution_ok || !quality.grayscale);
  return (
    <section className="psc-card">
      <div className="psc-card-intro-head">
        <h2 className="psc-card-heading">{mode === "clinician" ? "Image looks ready" : "Ready to check"}</h2>
        <p className="psc-card-body">
          {hasIssue
            ? "This image passed the basic checks, but see the Image check card for details before analyzing."
            : mode === "clinician"
              ? "This image passed the quality and chest X-ray checks. Run the AI screening model."
              : "This image is ready to be screened by the AI."}
        </p>
      </div>
      <button type="button" className="psc-btn psc-btn--primary psc-btn--full psc-btn--tall" onClick={onAnalyze}>
        Analyze X-ray
      </button>
    </section>
  );
}
