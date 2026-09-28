import type { Focus } from "../contract";
import type { LungZone } from "./types";

interface Props {
  onSelectZone: (zone: LungZone) => void;
  focus: Focus | null;
}

const ZONES: LungZone[] = ["upper", "middle", "lower"];

// "Explore anatomy" row (README): clicking a zone chip selects that zone in
// BOTH lungs at once (not a single side) -- these are zone-only chips, the
// side-specific selection only happens by clicking a lung directly in 3D
// or a region in the X-ray view.
export function ZoneChips({ onSelectZone, focus }: Props) {
  return (
    <div className="psc-explorer-chip-row">
      {ZONES.map((zone) => (
        <button key={zone} type="button" className="psc-explorer-chip" onClick={() => onSelectZone(zone)}>
          <span className="psc-explorer-chip-glyph" aria-hidden="true">
            <span className={zone === "upper" ? "psc-explorer-chip-bar--on" : ""} />
            <span className={zone === "middle" ? "psc-explorer-chip-bar--on" : ""} />
            <span className={zone === "lower" ? "psc-explorer-chip-bar--on" : ""} />
          </span>
          <span>{zone[0].toUpperCase() + zone.slice(1)} lung</span>
          {focus?.zone === zone && <span className="psc-explorer-chip-tag">AI focus</span>}
        </button>
      ))}
    </div>
  );
}
