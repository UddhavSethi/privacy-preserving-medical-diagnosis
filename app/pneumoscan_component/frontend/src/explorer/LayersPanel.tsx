import type { LungLayers } from "./types";

interface Props {
  layers: LungLayers;
  setLayers: (l: LungLayers) => void;
  aiFocusOn: boolean;
  setAiFocusOn: (v: boolean) => void;
  hasFocus: boolean;
}

const ROWS: Array<{ key: keyof LungLayers; label: string }> = [
  { key: "lungs", label: "Lungs" },
  { key: "lobes", label: "Lobes" },
  { key: "bronchi", label: "Bronchi" },
  { key: "trachea", label: "Trachea" },
  { key: "heart", label: "Heart (Reference)" },
  { key: "ribs", label: "Rib cage" },
  { key: "spine", label: "Spine" },
];

// Clinician-mode only (README). The "AI focus" row is deliberately NOT part
// of `LungLayers` -- it shares state with the X-ray viewer's own "AI
// highlight" switch (README: "The AI focus layer = the viewer's 'AI
// highlight' switch (same state)"), so it's a separate prop pair here
// rather than a duplicated toggle with its own independent state.
export function LayersPanel({ layers, setLayers, aiFocusOn, setAiFocusOn, hasFocus }: Props) {
  const toggle = (key: keyof LungLayers) => setLayers({ ...layers, [key]: !layers[key] });

  return (
    <div className="psc-explorer-panel psc-explorer-panel--layers">
      <div className="psc-explorer-caption">Layers</div>
      <div className="psc-explorer-layer-rows">
        {ROWS.map((row) => (
          <button key={row.key} type="button" className="psc-explorer-layer-row" onClick={() => toggle(row.key)}>
            <span className={`psc-explorer-checkbox${layers[row.key] ? " psc-explorer-checkbox--on" : ""}`} aria-hidden="true" />
            <span>{row.label}</span>
          </button>
        ))}
        <button
          type="button"
          className="psc-explorer-layer-row"
          onClick={() => setAiFocusOn(!aiFocusOn)}
          disabled={!hasFocus}
        >
          <span className={`psc-explorer-checkbox psc-explorer-checkbox--amber${aiFocusOn && hasFocus ? " psc-explorer-checkbox--on" : ""}`} aria-hidden="true" />
          <span>AI focus</span>
          <span className="psc-explorer-layer-note">Approx.</span>
        </button>
      </div>
    </div>
  );
}
