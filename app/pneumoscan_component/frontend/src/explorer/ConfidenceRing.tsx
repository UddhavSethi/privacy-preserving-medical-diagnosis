import type { ResultProps } from "../contract";
import { fmt } from "../lib/na";

interface Props {
  result: ResultProps;
}

function findingText(result: ResultProps): string {
  if (result.label === "Pneumonia") return "Pneumonia pattern detected";
  if (result.label === "Normal") return "No pneumonia pattern detected";
  return "Inconclusive result";
}

// Clinician-only (README: "Bottom-left: clinician gets a confidence ring
// (66px conic gradient #56c7cd, finding, 'Certainty: …')").
export function ConfidenceRing({ result }: Props) {
  const pct = Math.round(result.confidence * 100);
  return (
    <div className="psc-explorer-panel psc-explorer-confidence">
      <div className="psc-explorer-ring" style={{ background: `conic-gradient(#56c7cd ${pct}%, rgba(255,255,255,0.12) ${pct}%)` }}>
        <div className="psc-explorer-ring-inner">{pct}%</div>
      </div>
      <div className="psc-explorer-confidence-copy">
        <span className="psc-explorer-confidence-finding">{findingText(result)}</span>
        <span className="psc-explorer-confidence-certainty">Certainty: {fmt(result.certainty)}</span>
      </div>
    </div>
  );
}
