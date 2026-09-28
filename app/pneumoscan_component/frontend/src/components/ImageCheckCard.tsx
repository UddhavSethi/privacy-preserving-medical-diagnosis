import type { QualityProps } from "../contract";

interface Props {
  quality: QualityProps | null;
}

type RowState = "pass" | "warn" | "fail" | "unknown";

function icon(state: RowState): string {
  if (state === "pass") return "✓";
  if (state === "fail") return "✕";
  if (state === "warn") return "!";
  return "?";
}

export function ImageCheckCard({ quality }: Props) {
  if (!quality) return null;

  const gateStatus = quality.xray_gate.status;
  const rows: Array<{ label: string; state: RowState; value: string }> = [
    { label: "Image decoded", state: quality.decoded ? "pass" : "fail", value: quality.decoded ? "Yes" : "No" },
    { label: "Format supported", state: quality.format_supported ? "pass" : "fail", value: quality.format_supported ? "Yes" : "No" },
    {
      label: "Resolution",
      state: quality.resolution_ok ? "pass" : "warn",
      value: `${quality.min_side_px}px min side`,
    },
    { label: "Grayscale radiograph", state: quality.grayscale ? "pass" : "warn", value: quality.grayscale ? "Yes" : "Unusual color" },
    {
      label: "Chest X-ray detected",
      state: gateStatus === "passed" ? "pass" : gateStatus === "failed" ? "fail" : "unknown",
      value:
        gateStatus === "unavailable"
          ? "Not available"
          : quality.xray_gate.p_xray !== null
            ? `${(quality.xray_gate.p_xray * 100).toFixed(0)}% confidence`
            : "Not available",
    },
  ];

  return (
    <section className="psc-card">
      <h2 className="psc-card-title">Image check</h2>
      <div className="psc-check-rows">
        {rows.map((row) => (
          <div key={row.label} className="psc-check-row">
            <span className={`psc-check-icon psc-check-icon--${row.state}`}>{icon(row.state)}</span>
            <span className="psc-check-label">{row.label}</span>
            <span className="psc-check-value">{row.value}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
