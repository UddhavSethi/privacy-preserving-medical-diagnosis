// Phase 0 spike: proves two risky things work on the REAL Streamlit Cloud
// deployment before any real UI is built on top of them.
//
//   1. Bytes round-trip: Python -> JS (an image arg) and JS -> Python (an
//      uploaded-file event), matching the final design's upload flow.
//   2. Full-viewport iframe layout: a sticky header, a scrollable body, and
//      a position:fixed overlay all behave correctly when the iframe is
//      pinned to the real window height instead of auto-sizing to content.
//
// This file is intentionally thrown away once Phase 0's exit criteria pass
// (see the plan: "round-trip works on the live Cloud URL, not just
// locally") -- it is not meant to survive into Phase 1's real screens.
import { useEffect, useRef, useState } from "react";
import type { ComponentProps } from "streamlit-component-lib";
import { withStreamlitConnection } from "streamlit-component-lib";
import { markReady, sendEvent, syncFrameHeightToViewport } from "./bridge";

interface SpikeArgs {
  image_b64?: string;
  message?: string;
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function App({ args }: ComponentProps): JSX.Element {
  const typedArgs = args as SpikeArgs;
  const [scrollProbe, setScrollProbe] = useState(0);
  const [lastSent, setLastSent] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    markReady();
    return syncFrameHeightToViewport();
  }, []);

  const handleFile = async (file: File) => {
    const buf = new Uint8Array(await file.arrayBuffer());
    const b64 = bytesToBase64(buf);
    sendEvent("upload", { name: file.name, mime: file.type, size: buf.length, data_b64: b64 });
    setLastSent(`${file.name} (${buf.length} bytes)`);
  };

  return (
    <div style={{ height: "100dvh", display: "flex", flexDirection: "column", fontFamily: "sans-serif" }}>
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          background: "rgba(255,255,255,0.9)",
          borderBottom: "1px solid #ddd",
          padding: "10px 16px",
        }}
      >
        <strong>PneumoScan Phase 0 spike</strong> — sticky header check (scroll offset: {scrollProbe}px)
      </header>

      <main
        style={{ flex: 1, overflowY: "auto", padding: 16 }}
        onScroll={(e) => setScrollProbe(Math.round((e.target as HTMLDivElement).scrollTop))}
      >
        <section style={{ marginBottom: 16 }}>
          <h3>1. Python → JS (bytes arg)</h3>
          <p>args.message = {JSON.stringify(typedArgs.message ?? null)}</p>
          {typedArgs.image_b64 ? (
            <img
              src={`data:image/png;base64,${typedArgs.image_b64}`}
              alt="from Python"
              style={{ maxWidth: 240, border: "1px solid #ccc" }}
            />
          ) : (
            <p>(no image arg received)</p>
          )}
        </section>

        <section style={{ marginBottom: 16 }}>
          <h3>2. JS → Python (component value)</h3>
          <input
            ref={fileInputRef}
            type="file"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleFile(f);
            }}
          />
          <p>Last sent: {lastSent ?? "(nothing yet)"}</p>
          <button onClick={() => sendEvent("ping", { t: Date.now() })}>Send ping event</button>
        </section>

        <section style={{ height: 1400, background: "linear-gradient(#f4f6f7, #dfe4e6)" }}>
          <p>Scroll probe filler — confirms the body scrolls independently of the sticky header
             and that the iframe itself does not also scroll (it should stay pinned to the
             viewport height set via setFrameHeight).</p>
        </section>
      </main>

      <div
        style={{
          position: "fixed",
          bottom: 16,
          right: 16,
          background: "#0f7c83",
          color: "white",
          padding: "8px 14px",
          borderRadius: 8,
          fontSize: 13,
        }}
      >
        position:fixed check
      </div>
    </div>
  );
}

export default withStreamlitConnection(App);
