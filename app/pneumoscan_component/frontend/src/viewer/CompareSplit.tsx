import { useCallback, useRef } from "react";

interface Props {
  pct: number; // 0..100
  onChange: (pct: number) => void;
  containerRef: React.RefObject<HTMLDivElement>;
}

/** The draggable vertical split handle for the viewer's "Compare" mode
 * (README: "a draggable vertical split with a 36px white handle"). */
export function CompareSplit({ pct, onChange, containerRef }: Props) {
  const draggingRef = useRef(false);

  const updateFromClientX = useCallback(
    (clientX: number) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const next = ((clientX - rect.left) / rect.width) * 100;
      onChange(Math.min(96, Math.max(4, next)));
    },
    [containerRef, onChange]
  );

  return (
    <div className="psc-compare-split" style={{ left: `${pct}%` }}>
      <div className="psc-compare-line" />
      <div
        className="psc-compare-handle"
        onPointerDown={(e) => {
          draggingRef.current = true;
          (e.target as Element).setPointerCapture?.(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (draggingRef.current) updateFromClientX(e.clientX);
        }}
        onPointerUp={(e) => {
          draggingRef.current = false;
          (e.target as Element).releasePointerCapture?.(e.pointerId);
        }}
      >
        <span>‹</span>
        <span>›</span>
      </div>
    </div>
  );
}
