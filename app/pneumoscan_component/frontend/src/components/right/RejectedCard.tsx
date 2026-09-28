import type { QualityProps } from "../../contract";

interface Props {
  quality: QualityProps | null;
  onNewScreening: () => void;
}

export function RejectedCard({ quality, onNewScreening }: Props) {
  const p = quality?.xray_gate.p_xray;
  return (
    <section className="psc-card psc-card--warn">
      <h2 className="psc-card-heading">This doesn't look like a chest X-ray</h2>
      <p className="psc-card-body">
        The image check didn't recognize this as a chest X-ray
        {p !== null && p !== undefined ? ` (${(p * 100).toFixed(0)}% confidence it is one)` : ""}, so no analysis was
        run. Please upload a chest X-ray image.
      </p>
      <button type="button" className="psc-btn psc-btn--primary psc-btn--full" onClick={onNewScreening}>
        Start a new screening
      </button>
    </section>
  );
}
