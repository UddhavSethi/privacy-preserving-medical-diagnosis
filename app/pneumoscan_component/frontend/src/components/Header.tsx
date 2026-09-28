export type Mode = "clinician" | "patient";

interface Props {
  mode: Mode;
  setMode: (m: Mode) => void;
}

const NAV_ITEMS = ["New Screening", "Previous Studies", "How It Works", "About"];

export function Header({ mode, setMode }: Props) {
  return (
    <header className="psc-header">
      <div className="psc-header-inner">
        <div className="psc-brand">
          <div className="psc-logo" aria-hidden="true">
            <span />
            <span />
          </div>
          <div className="psc-brand-text">
            <div className="psc-brand-name">PneumoScan</div>
            <div className="psc-brand-sub">AI-assisted chest X-ray screening</div>
          </div>
        </div>

        <nav className="psc-nav">
          {NAV_ITEMS.map((label) => {
            const active = label === "New Screening";
            return (
              <button key={label} type="button" className={`psc-nav-btn${active ? " psc-nav-btn--active" : ""}`} disabled={!active} aria-current={active ? "page" : undefined}>
                {label}
              </button>
            );
          })}
        </nav>

        <div className="psc-header-right">
          <div className="psc-mode-switch">
            <button type="button" className={`psc-mode-btn${mode === "clinician" ? " psc-mode-btn--active" : ""}`} onClick={() => setMode("clinician")}>
              Clinician mode
            </button>
            <button type="button" className={`psc-mode-btn${mode === "patient" ? " psc-mode-btn--active" : ""}`} onClick={() => setMode("patient")}>
              Patient mode
            </button>
          </div>
          <div className="psc-user-chip">
            <div className="psc-user-avatar">{mode === "clinician" ? "C" : "P"}</div>
            <div className="psc-user-text">
              <span className="psc-user-name">{mode === "clinician" ? "Clinician" : "Patient"}</span>
              <span className="psc-user-session">Local session</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
