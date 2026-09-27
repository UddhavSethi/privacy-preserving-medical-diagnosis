import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base: "./" -- Streamlit serves this bundle's dist/index.html from a
// component-specific path (not the site root), so asset URLs must be
// relative or they 404 on the real deployment. outDir stays the default
// "dist" but is stated explicitly since app/pneumoscan_component/__init__.py
// (Phase 0 step 3) points declare_component's `path` at it directly, and
// that path is also what the CI bundle test (Phase 1) and the .gitignore
// negation rule reference -- keeping it explicit here avoids the three
// drifting if Vite's own default ever changes.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist",
  },
});
