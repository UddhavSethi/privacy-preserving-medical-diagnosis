import type { Phase, StudyProps } from "../contract";
import { fmt } from "../lib/na";

interface Props {
  study: StudyProps | null;
  phase: Phase;
}

function formatBytes(bytes: number | null): string {
  if (bytes === null || bytes === undefined) return "Not available";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatLoadedAt(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  } catch {
    return iso;
  }
}

function statusFor(phase: Phase): { label: string; tone: "neutral" | "info" | "good" | "warn" } {
  switch (phase) {
    case "quality":
      return { label: "Ready to analyze", tone: "info" };
    case "analyzing":
      return { label: "Analyzing", tone: "info" };
    case "review":
      return { label: "Reviewed", tone: "good" };
    case "rejected":
      return { label: "Not a chest X-ray", tone: "warn" };
    case "error":
      return { label: "Error", tone: "warn" };
    default:
      return { label: "Loaded", tone: "neutral" };
  }
}

export function StudyCard({ study, phase }: Props) {
  if (!study) return null;
  const status = statusFor(phase);
  const rows: Array<[string, string, boolean?]> = [
    ["Study ID", study.study_id, true],
    ["Image", study.filename],
    ["Projection", fmt(study.projection)],
    ["Dimensions", `${study.width} x ${study.height}px`, true],
    ["File size", formatBytes(study.file_size_bytes)],
    ["Loaded", formatLoadedAt(study.loaded_at)],
    ["Patient", "Not recorded"],
  ];

  return (
    <section className="psc-card">
      <div className="psc-card-head">
        <h2 className="psc-card-title">Study</h2>
        <span className={`psc-pill psc-pill--${status.tone}`}>{status.label}</span>
      </div>
      <div className="psc-rows">
        {rows.map(([k, v, mono]) => (
          <div key={k} className="psc-row">
            <span className="psc-row-key">{k}</span>
            <span className={`psc-row-value${mono ? " psc-row-value--mono" : ""}`}>{v}</span>
          </div>
        ))}
      </div>
      <div className="psc-card-footnote">No patient identifiers are collected. Nothing is saved after this session.</div>
    </section>
  );
}
