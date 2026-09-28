import { useEffect, useRef, useState } from "react";
import { toUint8Array } from "../bridge";

function signature(bytes: Uint8Array): string {
  // Cheap content signature, not a real hash — good enough to detect "this
  // is the same image/heatmap Python already sent" across the ~0.3s polling
  // reruns (Phase 1B) without re-decoding a fresh Blob/<img> every tick,
  // which would otherwise flicker the viewer and leak object URLs for the
  // life of the session.
  let acc = bytes.length;
  const step = Math.max(1, Math.floor(bytes.length / 64));
  for (let i = 0; i < bytes.length; i += step) acc = (acc * 31 + bytes[i]) >>> 0;
  return `${bytes.length}:${acc}`;
}

/** Builds (and revokes on change/unmount) an object URL from a Streamlit
 * bytes special-arg, only recreating it when the underlying bytes actually
 * change content -- not merely a new JS object reference for the same
 * content, which Streamlit's own re-serialization produces on every rerun. */
export function useObjectUrl(bytes: Uint8Array | ArrayBuffer | null | undefined, mime: string): string | null {
  const [url, setUrl] = useState<string | null>(null);
  const sigRef = useRef<string | null>(null);
  const urlRef = useRef<string | null>(null);

  useEffect(() => {
    const arr = toUint8Array(bytes ?? null);
    if (!arr) {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
      sigRef.current = null;
      setUrl(null);
      return;
    }
    const sig = signature(arr);
    if (sig === sigRef.current && urlRef.current) return;
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    const nextUrl = URL.createObjectURL(new Blob([arr], { type: mime }));
    sigRef.current = sig;
    urlRef.current = nextUrl;
    setUrl(nextUrl);
  }, [bytes, mime]);

  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    },
    []
  );

  return url;
}

/** Same content-signature stabilization as `useObjectUrl`, but returns the
 * raw bytes themselves (for the heatmap canvas colorizer, which doesn't
 * need a Blob) instead of an object URL -- avoids re-running the colorize
 * effect every ~0.3s poll for bytes that haven't actually changed. */
export function useStableBytes(bytes: Uint8Array | ArrayBuffer | null | undefined): Uint8Array | null {
  const [stable, setStable] = useState<Uint8Array | null>(null);
  const sigRef = useRef<string | null>(null);

  useEffect(() => {
    const arr = toUint8Array(bytes ?? null);
    if (!arr) {
      sigRef.current = null;
      setStable(null);
      return;
    }
    const sig = signature(arr);
    if (sig === sigRef.current) return;
    sigRef.current = sig;
    setStable(arr);
  }, [bytes]);

  return stable;
}
