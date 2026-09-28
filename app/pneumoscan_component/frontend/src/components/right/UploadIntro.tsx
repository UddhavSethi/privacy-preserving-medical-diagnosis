interface Props {
  mode: "clinician" | "patient";
}

const POINTS_CLINICIAN = [
  { title: "Upload a chest X-ray", text: "JPEG, PNG, or DICOM. A frontal (PA) view works best." },
  { title: "Automatic quality + gate check", text: "The image is checked for resolution, color, and whether it's a chest X-ray at all." },
  { title: "Run AI screening", text: "The model returns a finding, a confidence level, and a Grad-CAM explanation of what it looked at." },
];

const POINTS_PATIENT = [
  { title: "Add your chest X-ray", text: "A JPEG, PNG, or DICOM file from your provider." },
  { title: "A quick quality check", text: "The app checks the image is clear and readable." },
  { title: "See the AI's screening result", text: "You'll see a plain-language result and how confident the AI is." },
];

export function UploadIntro({ mode }: Props) {
  const points = mode === "clinician" ? POINTS_CLINICIAN : POINTS_PATIENT;
  return (
    <section className="psc-card">
      <div className="psc-card-intro-head">
        <h2 className="psc-card-heading">{mode === "clinician" ? "Start a new screening" : "Check a chest X-ray"}</h2>
        <p className="psc-card-body">
          {mode === "clinician"
            ? "Upload a chest X-ray to run it through the federated, privacy-preserving screening model."
            : "Upload a chest X-ray photo and see what the AI notices."}
        </p>
      </div>
      <div className="psc-intro-points">
        {points.map((p, i) => (
          <div key={p.title} className="psc-intro-point">
            <div className="psc-intro-n">{i + 1}</div>
            <div className="psc-intro-copy">
              <span className="psc-intro-title">{p.title}</span>
              <span className="psc-intro-text">{p.text}</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
