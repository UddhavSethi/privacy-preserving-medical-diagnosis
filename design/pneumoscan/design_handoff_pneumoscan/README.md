# Handoff: PneumoScan — clinician/patient screening UI + 3D anatomical explorer

## Overview
PneumoScan is a redesigned frontend for the repo `UddhavSethi/privacy-preserving-medical-diagnosis` (Streamlit app in `app/streamlit_app.py`, inference in `app/inference.py`, config in `conf/app.yaml`). It replaces the research-dashboard look with a clinical product: upload a chest X-ray, check image quality, run AI screening, review an explainable result (Grad-CAM highlight), and explore an illustrative 3D lung model. There are two audiences, switched in the header: **Clinician mode** and **Patient mode** (same result, different wording and density).

The app uses **one fixed model configuration: `fedavg_secagg`** (FedAvg + Secure Aggregation). There is **no model selector** anywhere in the UI. Model/architecture details appear only on the *How It Works* page under a collapsible "Technical details" section.

## About the Design Files
The files in this bundle are **design references built in HTML**: prototypes showing the intended look and behavior. They are not production code to copy directly. The task is to **recreate these designs inside the existing app**, keeping `app/inference.py` and the ML pipeline unchanged and wiring the UI to real outputs.

Recommended implementation path (keeps Streamlit Cloud deploy working):
- Build the UI as a **Streamlit bidirectional custom component** (`streamlit.components.v1.declare_component`, React or plain TS frontend), and have `app/streamlit_app.py` render just that component.
- Python → component: the `InferenceResult` fields listed below, plus the uploaded image and the Grad-CAM heatmap.
- Component → Python: the uploaded file bytes and "run analysis" / "new screening" events.
- `lung-model.html` (three.js) can be embedded as-is in an iframe inside the component, or ported to React Three Fiber. It is driven entirely by `postMessage` (protocol below).

Alternative: expose `run_full_inference()` through a small FastAPI endpoint and serve the frontend separately.

**Do not invent data.** Every number in the prototype is a *sample* (the "Sample output" chip). If the backend doesn't provide a value, show "Not available". Never fabricate it.

## Fidelity
**High-fidelity.** Colors, type, spacing, copy and interactions are final. Recreate them pixel-accurately.

## Data contract (map UI → backend)
The UI is driven by these values, all of which already exist in `app/inference.py` / `conf/app.yaml`:

| UI element | Source |
|---|---|
| Finding: "Pneumonia pattern detected" / "No pneumonia pattern detected" / "Inconclusive — near the decision boundary" | `InferenceResult.label` (`Pneumonia` / `Normal` / `Uncertain`, with the abstention band applied) |
| Confidence % | calibrated `P(pneumonia)` for Pneumonia; `1 − p` for Normal; `max(p, 1−p)` for Uncertain |
| "How sure the AI is" High / Medium / Low | MC-dropout predictive entropy ÷ deferral threshold: < 0.5 → High, < 1 → Medium, ≥ 1 → Low |
| "Flagged for a second look" Yes/No | deferral flag (entropy ≥ deferral threshold) |
| "Image check": Typical / Unusual | per-hospital OOD flags; show **Unusual only if all 3 flag** (matches existing `all()` logic) |
| Chest X-ray detected (quality list) | existing X-ray gate |
| AI highlight overlay | Grad-CAM for the predicted class |
| 3D AI focus position + "Strongest focus: right lung, lower zone" | argmax of the Grad-CAM map (see mapping below) |
| Unavailable values | for checkpoints without calibration caches, show "Not available" (never "Low") |

**Backend additions needed:**
1. Return the **raw Grad-CAM heatmap** (e.g. 7×7 or 224×224 float array, 0–1), not only the finished overlay PNG. The UI colorizes it client-side so the *Opacity* and *Highlight area* sliders work, and it finds the peak for the 3D focus.
2. Report **analysis progress/steps** (gate → preprocess → MC passes → OOD → Grad-CAM). The scan animation must follow real progress, not a timer. The prototype uses a 4.2 s stand-in timer.
3. DICOM: decode server-side (the prototype only opens JPEG/PNG in the browser).

**Peak → lung-zone mapping (image space, u = x/width, v = y/height):**
- Side: `u < 0.5` → patient **right** lung (radiographic convention: patient right on viewer left); otherwise left.
- Zone: `t = (v − 0.12) / 0.68`; `t < 0.34` upper, `< 0.67` middle, else lower.
- Only mapped when the label is Pneumonia and the peak lies inside `0.1<u<0.9, 0.12<v<0.85`; otherwise show "The AI's focus is spread out rather than in one area."

## Screens / Views

### Global header (sticky)
- White at 86% opacity + `backdrop-filter: blur(14px)`, bottom border `1px #e3e7ea`, max-width 1440, padding `10px 24px`.
- Logo: 32×32 teal (`#0f7c83`) rounded-8 tile with two white lung shapes. "PneumoScan" 16/600; subtitle "AI-assisted chest X-ray screening" 12px `#5b656d`.
- Nav buttons: New Screening · Previous Studies · How It Works · About. 14/500, padding `9px 14px`, radius 8. Active: bg `#e9f3f3`, text `#0a5c61`; hover bg `#eef2f3`.
- Mode switch: segmented, track `#eef1f2` radius 9, pad 3. Active pill white with shadow `0 1px 2px rgba(20,25,29,.12)`.
- User chip: 30px circle `#e4f1f1` with initial (C/P), "Clinician"/"Patient" + "Local session".
- No disclaimer banner and no model selector.

### New Screening
Page title 24/600 ("New screening" / patient: "Check a chest X-ray") + subline 14px `#5b656d`. "Start a new screening" button at right once an image is loaded.

**Step bar:** a white card, 5 equal columns. 26px circle (done = teal fill + ✓; current = teal outline; pending = `#d3dadd` outline) plus a 2px connector line. Labels (clinician): Upload X-ray · Image check · AI analysis · AI highlight · Your review.

**Three columns** (flex-wrap, gap 16):
1. **Left (flex 1 1 250px):**
   - *Study card*: Study ID, Image, Projection, Dimensions, File size, Loaded, Patient "Not recorded".
   - *Image check card*: Image decoded · Format supported · Resolution (warn if < 256px) · Grayscale radiograph · Chest X-ray detected.
2. **Center viewer (flex 3 1 520px):** dark `#0e1215`, radius 12.
   - Toolbar: [Original | AI overlay | Compare], −/zoom%/+, Fit, Reset, Invert, Full screen.
   - Viewport `min(68vh,700px)`: wheel-zoom around the cursor (1–8×), drag to pan, R/L markers, top-left metadata (mono 11.5), hover readout "x y px · AI focus NN%".
   - Compare: a draggable vertical split with a 36px white handle.
   - Empty state: dashed drop zone with "Choose X-ray file" and "Use sample X-ray" buttons.
   - During analysis: a cyan scan line with glow sweeps down.
   - Bottom bar: "AI highlight" switch, "Opacity" slider, "Highlight area" slider.
3. **Right (flex 1.4 1 340px):** the card depends on phase.
   - *Upload*: intro with 3 numbered points.
   - *Quality*: "Image looks ready" + primary button **Analyze X-ray** (teal, 48px tall).
   - *Analyzing*: progress bar + 5 sub-steps.
   - *Result, clinician*:
     - "AI result" card: finding 24/600, confidence 32px mono + animated bar, caution notes (unusual image / too close to call), and rows: How sure the AI is · Image check · Flagged for a second look.
     - Footer (bg `#f0f7f7`): "Review recommended", "The AI result is a second opinion. The decision is yours.", and 3 steps with Compare / Open 3D buttons.
     - "Where the AI looked" card: original and highlight thumbnails, "Strongest focus: …", "Coloured areas influenced the result most. They are not an outline of disease."
   - *Result, patient*: plain headline + body; "How confident is the AI?" Low/Medium/High; "Why did the AI flag this image?" thumbnails; "Was the image suitable?"; "What should I do?" (teal panel: "AI screening results should be reviewed by a qualified healthcare professional. This application does not provide a medical diagnosis.").

### 3D Anatomical Explorer (below the columns, full width)
- Header: eyebrow 12/600, letter-spacing .12em, teal ("3D anatomical explorer" / patient "3D lung explanation"); title 28/600 ("Lungs and AI focus region" / "Explore the lungs"); sub 15px.
- Dark viewer: `height: clamp(560px, 78vh, 780px)`, radius 16, bg `#06080a`, shadow `0 30px 60px -36px rgba(6,24,28,.7)`. Contains the `lung-model.html` iframe plus floating glass panels.
  - Glass panel style: `background: rgba(12,17,20,.74); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,.08); border-radius: 12px`.
  - Panel captions: 11/600, uppercase, letter-spacing .12em, `#8fa0a7`.
- Top-center: [X-ray | 3D anatomy] segmented control. It crossfades (opacity .5s; the 3D layer scales 1.04 → 1, the X-ray 0.97 → 1).
- Top-left (clinician only): **Layers** list, 200px wide. Lungs, Lobes, Bronchi, Trachea, Heart (Reference), Rib cage, Spine (ribs and spine off by default), and AI focus (amber, note "Approx."). The **AI focus layer = the viewer's "AI highlight" switch** (same state).
- Top-right, 188px: **View** radio (Front/Left/Right/Top; the active one reflects the actual camera direction reported by the iframe), −/+/Reset, and Auto-rotate + Breathing switches (default off when `prefers-reduced-motion`).
- Right, top 300px, 244px wide: **info panel** for the selected region. Title caps teal, e.g. "RIGHT LUNG" / "Lower region"; AI-focus badge when it matches the focus; rows for Model confidence and Certainty (clinician); description clamped to 4 lines; `max-height: calc(100% - 448px)`. Animates in with opacity and translateX(14px → 0) over .28s.
- Bottom-left: clinician gets a **confidence ring** (66px conic gradient `#56c7cd`, finding, "Certainty: …"); patient gets a "What the AI noticed" sentence.
- Bottom-center, while analyzing: stage list "Analyzing chest X-ray → Mapping model attention → Generating visualization" plus a progress bar.
- X-ray view: the image with the heatmap, dashed zone lines at 34.7% / 57.3%, Upper/Middle/Lower labels, R/L markers, and an amber focus box. Clicking the box selects that region in 3D. Caption: "Illustrative anatomical mapping — not a patient-specific 3D reconstruction".
- Below the viewer: "Explore anatomy" row with pill chips Upper/Middle/Lower lung (44px tall, a 3-bar zone glyph, and an "AI focus" tag on the matching zone). Clicking one highlights the zone in both lungs, moves the camera to it and opens the info panel.

### Previous Studies
Table header (Study · Date · AI finding · Review status) and an empty state: "No previous studies" / "This prototype doesn't store images, results, or patient details…" / "Start a new screening". Do not generate fake records.

### How It Works
- 5 step cards.
- "Privacy-preserving federated AI" card with a hospital A/B/C → Secure aggregation → Global model flow diagram (animated SVG dots) and the text: "Model training uses federated learning and secure aggregation so participating sites can collaborate without sharing their training images."
- Collapsible **Technical details** (for researchers): model, training, aggregation, transport, uncertainty, threshold, image checks, explanation, datasets, plus the privacy-protections table.

### About
Project description, known limitations, team and faculty guide.

## 3D viewer (`lung-model.html`)
three.js 0.184 (unpkg import map), a single file. Procedural anatomy:
- Lungs: deformed spheres with a tapered apex, domed base and cardiac notch, using a custom fresnel `ShaderMaterial`. Zone highlight bands, a selection outline and pulse, an AI-focus gaussian glow with ring, a scan band, and a slow light sweep all run in the shader.
- Lobe fissure lines, built by ray-casting plane/surface intersections.
- Trachea with rings, main bronchi, a 5-root depth-4 bronchial tree, heart, 10 rib pairs with sternum and clavicles, and 21 vertebrae.
- Environment: polar floor grid, reference rings with ticks, drifting dust, faint scan lines, vignette.
- AI focus: additive glow sprite, expanding rings, orbiting particles, and an HTML "AI FOCUS · Illustrative AI focus region" label.
- Breathing (5 s cycle), float, auto-rotate, eased camera tweens (slerped direction), orientation gizmo (R/L, S/I, A/P), and hover tooltips with the lobe name.

**postMessage protocol**
- Parent → iframe `{type:'pneumoscan-lung', state:{ focus:{side,zone,u,v}|null, showFocus, select:{kind:'zone',side:'right'|'left'|'both',zone}|{kind:'structure',id:'trachea'|'main'|'tree'|'heart'|'ribs'|'spine'}|null, mode:'clinician'|'patient', layers:{lungs,lobes,bronchi,trachea,heart,ribs,spine}, breathing, autoRotate, scan:{active,progress 0–1} }}`
- Parent → iframe `{type:'pneumoscan-view', view:'front'|'left'|'right'|'top'|'reset'|'zoomIn'|'zoomOut'}`
- Iframe → parent: `pneumoscan-lung-ready`, `pneumoscan-lung-select {sel}`, `pneumoscan-lung-view {name}`
- Send `focus` once Grad-CAM is available (the scan beam then settles on it). Send `scan.progress` from real backend progress.

## State (from the prototype's logic class)
- `mode`, `screen`
- `phase`: upload → quality → analyzing → review
- `img` (url, w, h, format, size, id, time), `qStep`, `elapsed`
- Viewer: `view` (original/overlay/compare), `zoom`, `panX`, `panY`, `invert`, `max`, `opacity`, `intensity`, `showCam`, `compare`
- 3D: `sel`, `lastSel`, `layers`, `breathing`, `autoRotate`, `explorerView` ('3d'|'xray'), `lungViewName`, `techOpen`
- The model is fixed: `fedavg_secagg`.

## Interactions & motion
- Result cards reveal with opacity + translateY(8–14px) over .45–.6s, staggered.
- The confidence bar animates its width over 1s with `cubic-bezier(.2,.7,.2,1)`.
- The heatmap fades in over .9s.
- Buttons: hover background shifts. Segmented controls and switches transition over .15–.2s.
- Camera tweens: 900–1000 ms ease-in-out cubic; zoom 350 ms.
- Respect `prefers-reduced-motion`: breathing and rotation off, tweens instant.
- Esc exits full-screen viewer.

## Design tokens
- **Font:** IBM Plex Sans (400/500/600); IBM Plex Mono (400/500) for numbers and metadata.
- **Colors:**
  - Page and ink: page `#f4f6f7`, surface `#ffffff`, border `#e3e7ea`, divider `#f0f2f3`, ink `#14191d`, secondary `#5b656d`, tertiary `#7a848b`
  - Teal: `#0f7c83`, hover `#0d6d73`, deep `#0a5c61`, tint `#e4f1f1` / `#e9f3f3` / `#f0f7f7`, bright (on dark) `#56c7cd` / `#3cb1b8`
  - Warning: `#8a520a` on `#fbf0dc` / `#fbf4e6`, border `#f0e2c6`
  - Dark surfaces: imaging `#0e1215` / `#06080a`, dark border `#222a30` / `#2a343b`
  - AI focus amber: `#e0a040` / `#f0b35a` / `#f6cf94`
- **Radii:** 6–8 (controls), 12 (cards), 16 (3D viewer), 999 (pills).
- **Type scale:** 11 caps labels, 12–13.5 body/meta, 14–15 body, 18 card titles, 24 page titles, 28 section titles, 32 mono confidence.
- **Spacing:** 4/6/8/10/12/14/16/18/20/24.

## Assets
- Sample X-ray: CC0 radiograph by Mikael Häggström (Wikimedia Commons), loaded by URL. Swap in the uploaded image.
- The Grad-CAM in the prototype is procedurally generated from sample blobs. **Replace it with the real heatmap.**
- No other images; the logo is CSS shapes.

## Files
- `PneumoScan v2.dc.html`: full UI prototype (template + logic class; open it in a browser, `support.js` is its runtime).
- `lung-model.html`: standalone 3D explorer (open it directly to try it).
- `support.js`: prototype runtime only. Not needed in the real implementation.

## Suggested Claude Code prompt
> Read `design_handoff_pneumoscan/README.md` and the HTML files in that folder. Implement this UI as a Streamlit bidirectional custom component that replaces the current layout in `app/streamlit_app.py`. Keep `app/inference.py` behavior unchanged; add the raw Grad-CAM array and progress reporting described in "Backend additions". Use the fixed `fedavg_secagg` configuration with no model selector. Embed `lung-model.html` via the postMessage protocol. Show "Not available" for any value the backend doesn't produce, and never fabricate metrics.
