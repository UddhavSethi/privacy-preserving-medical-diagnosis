// PneumoScan redesign, Phase 2 — types mirroring `lung-model.html`'s own
// postMessage protocol exactly (verified against the file's own `pick()`
// function and `S` state object, not just the README's prose description).
export type LungSide = "right" | "left" | "both";
export type LungZone = "upper" | "middle" | "lower";
export type StructureId = "trachea" | "main" | "tree" | "heart" | "ribs" | "spine";

export interface ZoneSelection {
  kind: "zone";
  side: LungSide;
  zone: LungZone;
}
export interface StructureSelection {
  kind: "structure";
  id: StructureId;
}
export type Selection = ZoneSelection | StructureSelection | null;

export interface LungLayers {
  lungs: boolean;
  lobes: boolean;
  bronchi: boolean;
  trachea: boolean;
  heart: boolean;
  ribs: boolean;
  spine: boolean;
}

// `lung-model.html`'s own default (line ~329): ribs/spine off, everything
// else on. Kept here as the single source of truth for the initial value.
export const DEFAULT_LAYERS: LungLayers = {
  lungs: true,
  lobes: true,
  bronchi: true,
  trachea: true,
  heart: true,
  ribs: false,
  spine: false,
};

export type CameraView = "front" | "left" | "right" | "top" | "bottom" | "back";

// Short region copy for the info panel. Structure descriptions are the
// exact same strings `lung-model.html`'s own `NAMES` constant uses for its
// hover tooltips (kept consistent with what a user already saw on hover) --
// this is general anatomy reference copy, not a backend-derived value, so
// it isn't subject to the "never fabricate" rule that applies to AI
// results themselves.
export const STRUCTURE_INFO: Record<StructureId, { title: string; sub: string; description: string }> = {
  trachea: { title: "Trachea", sub: "Main airway", description: "The windpipe: a rigid, ringed tube carrying air from the throat down to the main bronchi." },
  main: { title: "Main bronchi", sub: "Airways into each lung", description: "The two large airways branching from the trachea, one into each lung." },
  tree: { title: "Bronchial tree", sub: "Branching airways", description: "The branching network of airways that carries air from the main bronchi out to the smallest passages in each lung." },
  heart: { title: "Heart", sub: "Shown for reference", description: "Shown for anatomical reference only — not a target of this chest X-ray screening model." },
  ribs: { title: "Rib cage", sub: "Shown for reference", description: "Shown for anatomical reference only — not a target of this chest X-ray screening model." },
  spine: { title: "Spine", sub: "Shown for reference", description: "Shown for anatomical reference only — not a target of this chest X-ray screening model." },
};

const ZONE_LABEL: Record<LungZone, string> = { upper: "Upper", middle: "Middle", lower: "Lower" };

export function zoneTitle(side: LungSide): string {
  if (side === "both") return "Both lungs";
  return `${side === "right" ? "Right" : "Left"} lung`;
}

export function zoneSub(zone: LungZone): string {
  return `${ZONE_LABEL[zone]} region`;
}

export function zoneDescription(side: LungSide, zone: LungZone): string {
  if (side === "both") {
    return `An approximate ${ZONE_LABEL[zone].toLowerCase()}-region division of both lungs. Region boundaries here are illustrative, not a patient-specific reconstruction.`;
  }
  const lobeCount = side === "right" ? "three lobes (upper, middle, lower)" : "two lobes (upper, lower)";
  return `An approximate ${ZONE_LABEL[zone].toLowerCase()}-region division of the ${side} lung, which has ${lobeCount}. Region boundaries here are illustrative, not a patient-specific reconstruction.`;
}
