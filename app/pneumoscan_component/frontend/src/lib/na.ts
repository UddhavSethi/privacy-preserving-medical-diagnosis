// "Never fabricate data" (docs/pneumoscan_redesign_plan.md decision 4): the
// single place every component goes through to render a value the backend
// might not have sent. `null`/`undefined` always becomes "Not available" —
// never a guessed default like "Low" or "Typical".
export const NOT_AVAILABLE = "Not available";

export function fmt<T>(value: T | null | undefined, formatter?: (v: T) => string): string {
  if (value === null || value === undefined) return NOT_AVAILABLE;
  if (typeof value === "number" && Number.isNaN(value)) return NOT_AVAILABLE;
  return formatter ? formatter(value) : String(value);
}

export function fmtPct(value: number | null | undefined, digits = 0): string {
  return fmt(value, (v) => `${(v * 100).toFixed(digits)}%`);
}
