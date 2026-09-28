import type { ResultProps } from "../../contract";
import type { ViewMode } from "../../viewer/XrayViewer";

interface Props {
  result: ResultProps;
  setViewMode: (m: ViewMode) => void;
  onOpen3D: () => void;
}

function findingTitle(result: ResultProps): string {
  if (result.label === "Pneumonia") return "Pneumonia pattern detected";
  if (result.label === "Normal") return "No pneumonia pattern detected";
  return "Inconclusive — near the decision boundary";
}

export function ResultClinician({ result, setViewMode, onOpen3D }: Props) {
  const rows: Array<[string, string]> = [
    ["How sure the AI is", result.certainty ?? "Not available"],
    ["Image check", result.image_check ?? "Not available"],
    ["Flagged for a second look", result.deferred === null ? "Not available" : result.deferred ? "Yes" : "No"],
  ];

  return (
    <section className="psc-card psc-result-reveal">
      <div className="psc-result-body">
        <h2 className="psc-card-title">AI result</h2>
        <div className="psc-result-main">
          <div className="psc-finding-title">{findingTitle(result)}</div>
          <div className="psc-conf-row">
            <span className="psc-conf-value">{(result.confidence * 100).toFixed(0)}%</span>
            <span className="psc-conf-word">confidence</span>
          </div>
          <div className="psc-conf-track">
            <div className="psc-conf-fill" style={{ width: `${result.confidence * 100}%` }} />
          </div>
        </div>

        {result.image_check === "Unusual" && (
          <div className="psc-note psc-note--warn">
            This image looks different from the X-rays the AI learned from, for example an unusual scan or different
            equipment. Treat the result with extra caution.
          </div>
        )}
        {result.abstained && (
          <div className="psc-note psc-note--info">
            The result is too close to call, so the AI isn't giving an answer. Read this X-ray without relying on the
            AI.
          </div>
        )}

        <div className="psc-rows">
          {rows.map(([k, v]) => (
            <div key={k} className="psc-row">
              <span className="psc-row-key">{k}</span>
              <span className="psc-row-value psc-row-value--strong">{v}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="psc-result-footer">
        <div className="psc-result-footer-head">
          <span className="psc-result-footer-title">Review recommended</span>
          <span className="psc-result-footer-sub">The AI result is a second opinion. The decision is yours.</span>
        </div>
        <div className="psc-next-step">
          <span className="psc-next-n">1</span>
          <span className="psc-next-text">Compare the AI highlight against the original image</span>
          <button type="button" className="psc-btn psc-btn--outline-teal" onClick={() => setViewMode("compare")}>
            Compare
          </button>
        </div>
        <div className="psc-next-step">
          <span className="psc-next-n">2</span>
          <span className="psc-next-text">Check exactly where the AI focused</span>
          <button type="button" className="psc-btn psc-btn--outline-teal" onClick={() => setViewMode("overlay")}>
            View
          </button>
        </div>
        <div className="psc-next-step">
          <span className="psc-next-n">3</span>
          <span className="psc-next-text">See the AI's focus on a 3D lung model</span>
          <button type="button" className="psc-btn psc-btn--outline-teal" onClick={onOpen3D}>
            Open 3D
          </button>
        </div>
      </div>
    </section>
  );
}
