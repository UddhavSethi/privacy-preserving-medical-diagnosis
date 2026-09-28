import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { CameraView, LungLayers, Selection } from "./types";
import type { HeatPoint } from "./heatPoints";
import type { Focus } from "../contract";

export interface LungFrameState {
  focus: Focus | null;
  showFocus: boolean;
  select: Selection;
  mode: "clinician" | "patient";
  layers: LungLayers;
  breathing: boolean;
  autoRotate: boolean;
  scan: { active: boolean; progress?: number };
  /** Real multi-point Grad-CAM cloud (owner-directed follow-up, 2026-09-28)
   * -- see `heatPoints.ts` and `lung-model.html`'s own `HEAT_MAX`/`uHeat*`
   * additions. Empty/omitted renders nothing, never a fabricated pattern. */
  heatPoints: HeatPoint[];
}

export interface LungFrameHandle {
  sendView: (view: CameraView | "reset" | "zoomIn" | "zoomOut") => void;
}

interface Props {
  state: LungFrameState;
  onSelect: (sel: Selection) => void;
  onViewName: (name: CameraView) => void;
  onReady: () => void;
  onTimeout: () => void;
}

const READY_TIMEOUT_MS = 8000;

// Wraps `public/lung-model.html` (three.js, reused as-is from the design
// handoff -- see docs/pneumoscan_redesign_plan.md Phase 2 step 1). Owns
// nothing about anatomy/rendering itself; only the postMessage protocol
// documented in the design README's "postMessage protocol" section and
// verified directly against the file's own `addEventListener('message', ...)`
// handler and `pick()`/`view()` functions.
export const LungFrame = forwardRef<LungFrameHandle, Props>(function LungFrame({ state, onSelect, onViewName, onReady, onTimeout }, ref) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [ready, setReady] = useState(false);
  const lastStateJson = useRef<string | null>(null);

  useImperativeHandle(ref, () => ({
    sendView: (view) => {
      const win = iframeRef.current?.contentWindow;
      if (!win) return;
      win.postMessage({ type: "pneumoscan-view", view }, "*");
    },
  }));

  // 8s no-`ready` timeout (Risk #11, plan Phase 2 step 2): if the iframe
  // never reports ready (most likely cause: no WebGL), fall back to the
  // X-ray-only view rather than showing a dead dark box forever.
  useEffect(() => {
    if (ready) return;
    const t = window.setTimeout(() => {
      if (!ready) onTimeout();
    }, READY_TIMEOUT_MS);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  useEffect(() => {
    function onMessage(e: MessageEvent) {
      if (e.source !== iframeRef.current?.contentWindow) return; // hardens the prototype's own unchecked '*' target
      const d = e.data || {};
      if (d.type === "pneumoscan-lung-ready") {
        setReady(true);
        onReady();
      } else if (d.type === "pneumoscan-lung-select") {
        onSelect(d.sel ?? null);
      } else if (d.type === "pneumoscan-lung-view") {
        onViewName(d.name);
      }
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-send state on every change, but de-duplicated (matching the
  // prototype's own `sendLung` behavior) so we're not posting a redundant
  // message every render -- e.g. while `scan.progress` isn't actually
  // moving between two polling ticks.
  useEffect(() => {
    if (!ready) return;
    const win = iframeRef.current?.contentWindow;
    if (!win) return;
    const json = JSON.stringify(state);
    if (json === lastStateJson.current) return;
    lastStateJson.current = json;
    win.postMessage({ type: "pneumoscan-lung", state }, "*");
  }, [ready, state]);

  return <iframe ref={iframeRef} src="./lung-model.html" title="3D anatomical explorer" className="psc-explorer-iframe" />;
});
