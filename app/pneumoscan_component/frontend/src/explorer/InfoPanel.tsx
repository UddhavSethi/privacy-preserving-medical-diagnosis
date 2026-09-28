import type { ResultProps } from "../contract";
import { fmt, fmtPct } from "../lib/na";
import { STRUCTURE_INFO, zoneDescription, zoneSub, zoneTitle, type Selection } from "./types";

interface Props {
  selection: Selection;
  mode: "clinician" | "patient";
  result: ResultProps | null;
}

function matchesFocus(selection: Selection, result: ResultProps | null): boolean {
  if (!selection || selection.kind !== "zone" || !result?.focus) return false;
  return selection.side === result.focus.side && selection.zone === result.focus.zone;
}

export function InfoPanel({ selection, mode, result }: Props) {
  if (!selection) return null;

  const isFocus = matchesFocus(selection, result);
  const title = selection.kind === "zone" ? zoneTitle(selection.side) : STRUCTURE_INFO[selection.id].title;
  const sub = selection.kind === "zone" ? zoneSub(selection.zone) : STRUCTURE_INFO[selection.id].sub;
  const description = selection.kind === "zone" ? zoneDescription(selection.side, selection.zone) : STRUCTURE_INFO[selection.id].description;

  return (
    <div className="psc-explorer-panel psc-explorer-panel--info psc-explorer-info-enter">
      <div className="psc-explorer-info-head">
        <span className="psc-explorer-info-title">{title.toUpperCase()}</span>
        {isFocus && <span className="psc-explorer-focus-badge">AI focus</span>}
      </div>
      <div className="psc-explorer-info-sub">{sub}</div>

      {mode === "clinician" && result && (
        <div className="psc-explorer-info-rows">
          <div className="psc-explorer-info-row">
            <span>Model confidence</span>
            <span>{fmtPct(result.confidence)}</span>
          </div>
          <div className="psc-explorer-info-row">
            <span>Certainty</span>
            <span>{fmt(result.certainty)}</span>
          </div>
        </div>
      )}

      <p className="psc-explorer-info-desc">{description}</p>
    </div>
  );
}
