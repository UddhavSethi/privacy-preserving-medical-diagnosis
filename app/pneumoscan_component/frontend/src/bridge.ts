// Thin wrapper around streamlit-component-lib's imperative API.
//
// We deliberately do NOT let the component auto-size its iframe to content
// height (the library's default StreamlitComponentBase behavior). This app
// owns a fixed, full-viewport layout (sticky header, internal scrolling,
// full-screen viewer, a 3D explorer) that breaks if the iframe grows to fit
// content instead of being pinned to the real browser viewport height. See
// the plan's Phase 0 exit criteria: verify this on the actual deployed
// Cloud URL, not just locally, since iframe sizing behaves differently
// across browsers/embedding contexts.
import { Streamlit } from "streamlit-component-lib";

let readySent = false;

export function markReady(): void {
  if (readySent) return;
  readySent = true;
  Streamlit.setComponentReady();
}

// Pins the iframe's rendered height to the real window viewport height, and
// keeps it in sync across resizes/orientation changes. The component's own
// CSS then uses `height: 100dvh` internally, which becomes correct because
// the iframe's own viewport now equals the outer window's.
//
// Deliberately reads `window.top.innerHeight`, NOT this frame's own
// `window.innerHeight`: Streamlit renders every new component into an
// iframe at a small placeholder height until setFrameHeight first reports a
// larger one, so the iframe's own innerHeight is circular -- it just
// reflects whatever tiny height Streamlit already gave it, not the real
// browser viewport. Streamlit's component iframe is same-origin (served
// from the same Streamlit server as the parent page), so reading
// window.top.innerHeight directly is not a cross-origin violation.
export function syncFrameHeightToViewport(): () => void {
  const getViewportHeight = () => window.top?.innerHeight ?? window.innerHeight;
  const report = () => Streamlit.setFrameHeight(getViewportHeight());
  report();
  window.addEventListener("resize", report);
  return () => window.removeEventListener("resize", report);
}

export interface OutboundEvent {
  event_id: string;
  type: string;
  payload?: unknown;
}

let eventCounter = 0;

export function sendEvent(type: string, payload?: unknown): void {
  eventCounter += 1;
  const event: OutboundEvent = {
    event_id: `${Date.now()}-${eventCounter}`,
    type,
    payload,
  };
  Streamlit.setComponentValue(event);
}
