import type { Phase, ProgressProps } from "../contract";

type StepStatus = "done" | "current" | "pending";

interface Step {
  n: string;
  label: string;
  sub: string;
  status: StepStatus;
}

function computeSteps(phase: Phase, progress: ProgressProps | null, mode: "clinician" | "patient"): Step[] {
  const labels =
    mode === "clinician"
      ? ["Upload X-ray", "Image check", "AI analysis", "AI highlight", "Your review"]
      : ["Upload X-ray", "Image check", "AI screening", "AI explanation", "Your result"];
  const subs = ["Add an image", "Quality + gate", "Screening model", "Explainability", "Review & decide"];

  const statuses: StepStatus[] = ["pending", "pending", "pending", "pending", "pending"];

  if (phase === "upload") {
    statuses[0] = "current";
  } else if (phase === "quality" || phase === "rejected") {
    statuses[0] = "done";
    statuses[1] = phase === "rejected" ? "current" : "current";
  } else if (phase === "analyzing") {
    statuses[0] = "done";
    statuses[1] = "done";
    const gradcamStage = progress?.stages.find((s) => s.id === "gradcam");
    if (gradcamStage && gradcamStage.status !== "pending") {
      statuses[2] = "done";
      statuses[3] = gradcamStage.status === "done" ? "done" : "current";
    } else {
      statuses[2] = "current";
    }
  } else if (phase === "review") {
    statuses[0] = "done";
    statuses[1] = "done";
    statuses[2] = "done";
    statuses[3] = "done";
    statuses[4] = "current";
  } else if (phase === "error") {
    statuses[0] = "current";
  }

  return labels.map((label, i) => ({ n: String(i + 1).padStart(2, "0"), label, sub: subs[i], status: statuses[i] }));
}

interface Props {
  phase: Phase;
  progress: ProgressProps | null;
  mode: "clinician" | "patient";
}

export function StepBar({ phase, progress, mode }: Props) {
  const steps = computeSteps(phase, progress, mode);
  return (
    <div className="psc-stepbar">
      {steps.map((step, i) => (
        <div key={step.label} className="psc-step">
          <div className="psc-step-row">
            <div className={`psc-step-mark psc-step-mark--${step.status}`}>{step.status === "done" ? "✓" : i + 1}</div>
            <div className={`psc-step-line psc-step-line--${step.status === "pending" ? "pending" : "active"}`} />
          </div>
          <div className="psc-step-copy">
            <span className="psc-step-n">{step.n}</span>
            <span className={`psc-step-label psc-step-label--${step.status}`}>{step.label}</span>
            <span className="psc-step-sub">{step.sub}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
