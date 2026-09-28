import { useCallback, useRef, useState } from "react";

export interface ZoomPanState {
  zoom: number; // 1..8
  panX: number; // px, viewport space
  panY: number;
  invert: boolean;
}

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;

function clampZoom(z: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z));
}

/**
 * Wheel-zoom-around-cursor + drag-to-pan + invert, for the New Screening
 * viewer's "Original"/"AI overlay"/"Compare" stage (README: "wheel-zoom
 * around the cursor (1-8x), drag to pan ... invert"). `viewportRef` is the
 * outer, fixed-size viewport element wheel/pointer events are read
 * relative to; the returned `panX`/`panY`/`zoom` transform the inner
 * "stage" element (`translate(panX, panY) scale(zoom)`).
 */
export function useZoomPan(viewportRef: React.RefObject<HTMLDivElement>) {
  const [state, setState] = useState<ZoomPanState>({ zoom: 1, panX: 0, panY: 0, invert: false });
  const dragRef = useRef<{ pointerId: number; startX: number; startY: number; startPanX: number; startPanY: number } | null>(
    null
  );

  const zoomAround = useCallback((nextZoomRaw: number, clientX?: number, clientY?: number) => {
    setState((prev) => {
      const nextZoom = clampZoom(nextZoomRaw);
      if (nextZoom === prev.zoom) return prev;
      const rect = viewportRef.current?.getBoundingClientRect();
      if (!rect || clientX === undefined || clientY === undefined) {
        return { ...prev, zoom: nextZoom };
      }
      // Keep the point under the cursor visually fixed: the stage is
      // translated by (panX, panY) then scaled by zoom around its own
      // center, so solving for the pan that keeps (cx, cy) fixed gives:
      const cx = clientX - rect.left - rect.width / 2;
      const cy = clientY - rect.top - rect.height / 2;
      const ratio = nextZoom / prev.zoom;
      return {
        ...prev,
        zoom: nextZoom,
        panX: cx - (cx - prev.panX) * ratio,
        panY: cy - (cy - prev.panY) * ratio,
      };
    });
  }, [viewportRef]);

  const zoomIn = useCallback(() => zoomAround(state.zoom * 1.25), [state.zoom, zoomAround]);
  const zoomOut = useCallback(() => zoomAround(state.zoom / 1.25), [state.zoom, zoomAround]);

  const fit = useCallback(() => setState((prev) => ({ ...prev, zoom: 1, panX: 0, panY: 0 })), []);
  const reset = useCallback(() => setState((prev) => ({ ...prev, zoom: 1, panX: 0, panY: 0 })), []);
  const toggleInvert = useCallback(() => setState((prev) => ({ ...prev, invert: !prev.invert })), []);

  const onWheel = useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * 0.0015);
      zoomAround(state.zoom * factor, e.clientX, e.clientY);
    },
    [state.zoom, zoomAround]
  );

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (state.zoom <= 1) return;
      (e.target as Element).setPointerCapture?.(e.pointerId);
      dragRef.current = { pointerId: e.pointerId, startX: e.clientX, startY: e.clientY, startPanX: state.panX, startPanY: state.panY };
    },
    [state.zoom, state.panX, state.panY]
  );

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) return;
    setState((prev) => ({
      ...prev,
      panX: drag.startPanX + (e.clientX - drag.startX),
      panY: drag.startPanY + (e.clientY - drag.startY),
    }));
  }, []);

  const endDrag = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === e.pointerId) dragRef.current = null;
  }, []);

  const zoomPct = Math.round(state.zoom * 100);
  const isPanning = state.zoom > 1;

  return {
    state,
    zoomIn,
    zoomOut,
    fit,
    reset,
    toggleInvert,
    onWheel,
    onPointerDown,
    onPointerMove,
    onPointerUp: endDrag,
    onPointerLeave: endDrag,
    zoomPct,
    isPanning,
  };
}
