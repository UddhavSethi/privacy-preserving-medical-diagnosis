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
// browser viewport. In production the component is served as the built
// `dist/` bundle from Streamlit's own server, so the iframe is same-origin
// and this is not a cross-origin violation (verified live on Cloud, Phase 0).
//
// In LOCAL DEV, though, `PNEUMOSCAN_COMPONENT_DEV_URL` points the iframe at
// the separate Vite dev server (a different origin/port than Streamlit) --
// found live, 2026-09-27: `window.top.innerHeight` throws a real
// `SecurityError` there, which an uncaught exception in a passive effect
// crashes the whole component ("Component Error" in the parent page, not
// just a wrong height). `screen.availHeight` is always readable
// cross-origin and is a reasonable full-viewport stand-in for local dev
// testing; this fallback never runs in production.
export function syncFrameHeightToViewport(): () => void {
  const getViewportHeight = () => {
    try {
      return window.top?.innerHeight ?? window.innerHeight;
    } catch {
      return window.screen.availHeight;
    }
  };
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

// Streamlit's component bridge hands a `bytes`-typed Python arg to us as a
// real binary value (see custom_component.py's `is_bytes_like` special-args
// path — no base64 round-trip), but which concrete JS type that is
// (`Uint8Array` vs. a plain `ArrayBuffer`) isn't pinned across
// streamlit/streamlit-component-lib versions. Every caller that needs
// `display_image`/`heatmap` goes through this instead of assuming one.
export function toUint8Array(value: Uint8Array | ArrayBuffer | null | undefined): Uint8Array | null {
  if (value == null) return null;
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  return null;
}

// Reads a file (from a drop/pick or the sample-image fetch) into the
// base64 string the `upload` event's payload carries — chunked to avoid
// blowing the call stack on `String.fromCharCode(...bytes)` for large files.
export async function fileToBase64(bytes: Uint8Array): Promise<string> {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}
