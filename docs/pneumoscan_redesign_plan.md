# PneumoScan frontend redesign: implementation plan

> Working plan for the in-progress PneumoScan redesign (Claude Design handoff
> -> React Streamlit custom component). Phase 0 is complete and verified on
> the live Cloud deployment as of 2026-09-27; Phases 1-3 are not yet started.
> This is a durable copy of the plan approved via EnterPlanMode — kept in the
> repo itself (not just `~/.claude/plans/`) so it survives independently of
> any single session's local state. Update the "Status" note at the top as
> phases complete; don't rewrite history below it.

## Context

A designer delivered a high-fidelity redesign ("PneumoScan") at
`design/pneumoscan/design_handoff_pneumoscan/README.md`, with two HTML
reference files: `PneumoScan v2.dc.html` (full UI prototype, written in a
proprietary preview templating format — a reference to recreate pixel-accurately,
not to copy) and `lung-model.html` (a self-contained three.js 3D lung viewer,
reusable as-is). It replaces today's plain-Streamlit-widgets UI
(`app/streamlit_app.py`, `app/components.py`, `app/theme.py`) with one React
custom component via `streamlit.components.v1.declare_component`.

**Owner-approved decisions (final, do not re-litigate):**

1. **Frontend tech: React + npm**, matching the README's own recommendation.
   This is the first Node/npm/React toolchain this Python-only project has
   ever had. The component's `dist/` bundle is built locally and **committed
   to git**, since Streamlit Community Cloud's build environment is
   Python/pip only and cannot run npm itself.
2. **`inference.py` can change**, including real progress-callback plumbing
   through `run_full_inference` (not a fake timer), and a new raw Grad-CAM
   heatmap field on `InferenceResult`.
3. **Single fixed model config: `fedavg_secagg`** — no model selector
   anywhere in the new UI. (This replaces the `fedavg_finetune_pilot` default
   set earlier this session — that's expected, it's what this design calls for.)
4. **Never fabricate data.** Every value the backend doesn't actually produce
   shows "Not available" — never a fake number, never "Low" as a default.
5. **D1 — derive real calibration for `fedavg_secagg`.** New script,
   `scripts/derive_app_decision_policy.py`, mirroring the existing
   validation-set threshold-sweep method already used for the other two
   checkpoints (`docs/adr1_groupnorm_fallback.md` §10/§17). Produces real
   `decision_threshold` / `abstention_half_width` / `temperature` values,
   committed into `conf/app.yaml`'s `fedavg_secagg` entry.
6. **D2 — precompute and commit small deferral/OOD artifacts** so the public
   Streamlit Cloud deployment isn't permanently stuck at "Not available" for
   "How sure is the AI" / "Image check" (the 576MB feature cache those need
   locally is gitignored and never deployed). Commit just the derived
   deferral threshold (one float, in `conf/app.yaml`) plus the 3 per-hospital
   `IsolationForest` models (small joblib files, pinned to `scikit-learn==1.9.0`
   to match the pinned version).
7. **D3 — drop the "Research & Results" tab entirely.** The redesign is a
   clinical product, not a research dashboard. The underlying data
   (`docs/results.md`, `docs/calibration.md`, etc.) still exists in the repo,
   just not surfaced in the app.
8. **D4 — no site-wide disclaimer banner**, exactly as the design specifies.
   Patient mode keeps its own disclaimer text in the result panel, per the design.

### Key facts already verified this session (don't re-derive)

- `InferenceResult` dataclass: `app/inference.py:115-130`. Fields:
  `rgb_image`, `predicted_label` ("Normal"/"Pneumonia"/"Uncertain"),
  `predicted_class`, `confidence` (calibrated), `prob_pneumonia` (calibrated),
  `entropy`, `deferred`, `deferral_threshold`, `abstained`,
  `decision_threshold`, `gradcam_overlay_rgb`, `ood_flags: dict[str,bool]`,
  `ood_scores: dict[str,float]`. The X-ray gate result is separate:
  `check_is_xray(bgr_image, gate, image_size=224) -> XrayGateResult`
  (`app/inference.py:97`), with `XrayGateResult.is_xray: bool`,
  `p_xray: float` (`src/uncertainty/xray_gate.py:71-73`).
- Main entry point: `run_full_inference(model, bgr_image, deferral_threshold,
  ood_detectors, ood_thresholds, image_size=224, num_mc_passes=20,
  decision_threshold=DEFAULT_DECISION_THRESHOLD, abstention_half_width=0.0,
  temperature=1.0) -> InferenceResult` (`app/inference.py:142-153`). Current
  internal order: preprocess (166) -> MC passes (178) -> Grad-CAM (197) ->
  OOD (199-204). The design wants gate -> preprocess -> MC -> OOD -> Grad-CAM;
  OOD and Grad-CAM are independent steps, safe to swap with no behavior change
  (a test enforces identical outputs before/after).
- **Raw Grad-CAM heatmap already exists internally, just discarded.**
  `compute_gradcam_heatmap(model, image_tensor, target_class) -> np.ndarray`
  (`src/explain/gradcam.py:56-72`) returns a float array in [0,1] at 224x224.
  `GradCAMOverlay` (`gradcam.py:75-78`) already has both `heatmap` and
  `overlay_rgb`; `inference.py:217` only keeps `overlay.overlay_rgb`. Exposing
  the heatmap is one new `InferenceResult` field + one line change.
- **DICOM decoding is already fully server-side** (`pydicom`,
  `inference.py:25,52-68`) — the README's "backend addition #3" is a
  non-issue. Pre-existing bug worth fixing in passing: the uploader only
  accepts `type=["jpg","jpeg","png","dcm"]` (`streamlit_app.py:246`) but the
  decoder's `SUPPORTED_DICOM_EXTENSIONS` includes `.dicom` too
  (`inference.py:47`) — `.dicom` files get rejected at upload despite being
  decodable.
- **No progress/step reporting exists anywhere today** — only a single
  blocking `with st.spinner("Analyzing..."):` (`streamlit_app.py:286`).
- Entropy/deferral/OOD reference logic to reuse: entropy at `inference.py:181`;
  `deferred = entropy >= deferral_threshold` (182); `get_deferral_threshold`
  (`streamlit_app.py:114,283`) returns `None` -> becomes `float("inf")` at
  line 306 today if artifacts are missing (this `inf` sentinel is a trap —
  see Risk 6 below, must be replaced with an explicit "unavailable" check, not
  relied upon); `uncertainty_label` -> Low/Medium/High (`inference.py:274-283`);
  OOD banner uses `all(ood_flags.values())` (337) — matches the design's own
  "Unusual only if all 3 flag" rule exactly, reuse as-is.
- Zero npm/React/TypeScript/`streamlit.components.v1` anywhere in tracked
  code today. No node/npm/webpack/vite in `pyproject.toml`/`requirements.txt`.
- `.gitignore` currently ignores `dist/` and `build/` repo-wide — **a
  committed component `dist/` needs an explicit negation rule**, or Cloud
  serves a blank iframe. This is Risk #1 below. **RESOLVED in Phase 0**: negation
  rule added (`!app/pneumoscan_component/frontend/dist/`).
- `app/components.py` already exists — the new component package must be
  named `app/pneumoscan_component/`, not `app/components/`, to avoid collision.
- `outputs/checkpoints/ablation/secagg_seed42.pt` (the `fedavg_secagg`
  checkpoint) is already committed and loads correctly (verified: float64 on
  disk, loads into float32 via `load_classifier` with no error).
- MC dropout is cheap (runs on 1024-d pooled features, milliseconds). Real
  compute time is the X-ray gate's backbone forward pass, the preprocess
  backbone forward pass, and the Grad-CAM backward pass — so 5 real stage
  boundaries (gate, preprocess, MC, OOD, Grad-CAM) is the honest level of
  progress detail; per-MC-pass granularity would be fake precision.
  `check_is_xray` currently builds a new `DenseNet121Head()` on every call —
  cache it.
- Heatmap coordinates line up directly with the image: `build_eval_transform`
  uses `Resize((224,224))` (a stretch, no crop), so heatmap (u,v) equals
  original-image (u,v) — the overlay can be stretched straight over the
  displayed image with no coordinate correction.
- Streamlit 1.62 specifics: `bytes` component args are sent as binary
  (no base64 needed Python->JS); `st.fragment(run_every=...)` exists but adds
  little value here since the page is almost entirely the one component;
  `components.v2` also exists (would avoid iframe-sizing issues) but decision
  1 fixes v1 — noted only, not used.
- Local Node is v18.19.1 (EOL). Vite 7 needs Node >=20.19 — pin Vite 5/6
  locally, or install Node 20/22 LTS. Cloud never runs Node regardless.
  **RESOLVED in Phase 0**: pinned to Vite 5.4.x.
- External dependencies in the prototype that must be vendored (not fetched
  live): Google Fonts, three.js from unpkg, a Wikimedia sample X-ray image,
  and `lung-model.html`'s `postMessage('*', ...)` with no sender check
  (tighten to check `e.source`).
- Prototype copy not backed by real data needing correction: "Projection: PA
  (expected)" -> use DICOM `ViewPosition` tag when present, else "Not recorded".
- **Component sizing gotcha found in Phase 0** (not in the original plan):
  the iframe's own `window.innerHeight` is circular — Streamlit starts new
  components at a tiny placeholder height, so the iframe just reports back
  that same tiny number. Must read `window.top.innerHeight` instead (the
  component iframe is same-origin, so this isn't a cross-origin violation).
  Fixed in `app/pneumoscan_component/frontend/src/bridge.ts`.
- **`streamlit-component-lib`'s exported `ComponentProps` type is NOT
  actually generic** despite appearing so in some docs — `ComponentProps<T>`
  fails to compile; use plain `ComponentProps` and cast `args as YourType`
  internally instead.
- **`declare_component(...)`'s returned callable takes props as top-level
  kwargs, not nested under an `args=` key** — despite `CustomComponent`'s
  internal parameter being named `args` in some places, you call it as
  `component_func(**your_props_dict, key=..., default=...)`.

---

## Phase 0: toolchain and deployment spike — COMPLETE (2026-09-27)

Ships nothing visible to normal users. Prove the risky infrastructure works
on Cloud before building any real UI.

1. Create `app/pneumoscan_component/frontend/` — Vite + React 18 + TypeScript
   + `streamlit-component-lib`. Add `.nvmrc`. `vite.config.ts`:
   `base: './'`, `outDir: 'dist'`.
2. Fix `.gitignore`: add `node_modules/` and
   `!app/pneumoscan_component/frontend/dist/`. Verify with
   `git check-ignore -v`.
3. `app/pneumoscan_component/__init__.py`: `declare_component("pneumoscan",
   path=dist)`, or `url=os.environ["PNEUMOSCAN_COMPONENT_DEV_URL"]` in dev
   mode. If `dist/index.html` is missing, raise a loud `st.error`, never a
   silent blank iframe.
4. Build a throwaway spike component: takes a bytes arg (an image), displays
   it, sends a JSON event with a base64 upload back to Python. Also spike the
   layout shell: sticky header, `100dvh` layout, a `position:fixed` overlay,
   `scrollTo` — these all break if the iframe sizes to content height, so
   `streamlit_app.py` must inject CSS (`layout="wide"`, hide Streamlit's own
   header/toolbar/padding, pin the component iframe to `height:100dvh`) so
   the component owns all scrolling.
5. Push to `origin/main`, verify on the real Streamlit Cloud URL: iframe
   loads (no 404 on `/component/...`), bytes round-trip both directions,
   layout works at desktop and mobile widths. Measure bundle size. Fallback
   if too large: hand-write a ~60-line bridge using the three raw
   `streamlit:*` postMessage events instead of the full component-lib.

**Exit criteria:** round-trip works on the live Cloud URL, not just locally.

**Result: PASSED.** Verified live at
`https://pneumonia-fl-demo.streamlit.app/?phase0_spike=1` — Python->JS bytes
round-trip visibly correct, sticky header, single scroll region, no Streamlit
chrome leaking through, no console errors. JS->Python direction re-uses
identical code already directly verified working locally. Bundle size: 324KB
JS (90KB gzipped) — well within limits, no fallback needed. Committed as
`77225fb`. The temporary `?phase0_spike=1` block in `streamlit_app.py` and
the throwaway `App.tsx` must be removed once Phase 1's real UI replaces them.

## Phase 1: backend changes, then New Screening screen end-to-end — COMPLETE (2026-09-27)

**1A committed as `0a977e0`** (local, not yet pushed at the time of writing).
**1B/1C/1D/1E built and verified live in a same-session follow-up** (not yet
committed): `app/presentation.py` gained `quality_props`/`build_props` (the
top-level props builder), `app/streamlit_app.py` was fully rewritten (~300
lines) as the real phase-driven state machine + single-worker
`ThreadPoolExecutor` polling loop described in 1B, and the entire React
frontend for the New Screening screen (both clinician and patient modes,
including the full viewer -- wheel-zoom, pan, compare-split, invert,
fullscreen, motion) was built against the frozen `contract.ts`/`bridge.ts`
contract. The Phase 0 `?phase0_spike` block and spike `App.tsx` are fully
removed (grep-verified, zero remaining references). Full test suite: 256
passing (207 pre-redesign + 46 from 1A + 3 new: `test_streamlit_app.py`'s
`AppTest` smoke test, `test_frontend_bundle.py`).

**Verified with a real manual smoke test** (Vite dev server + built `dist/`
served through a real local Streamlit process, driven via
`claude-in-chrome`, both dev-mode and production-mode): sample-X-ray upload
-> real gate/quality checks -> real threaded analysis against the actual
`fedavg_secagg` checkpoint (MC Dropout, OOD, Grad-CAM all real, not mocked)
-> review screen with real confidence/certainty/image-check values and a
real colorized Grad-CAM overlay, all end-to-end through the new UI.

**Two real bugs found by the manual browser smoke test, not by
`tsc`/`npm run build`/pytest (none of which can catch a live-rendering
bug)** -- confirms why this step, not just a clean build, was necessary:
1. `bridge.ts`'s `syncFrameHeightToViewport` read `window.top.innerHeight`
   unconditionally; this throws a real `SecurityError` (crashing the whole
   component with "Component Error") whenever the component iframe is
   cross-origin from the Streamlit page -- exactly the local dev-mode setup
   (`PNEUMOSCAN_COMPONENT_DEV_URL` pointing at the separate Vite dev
   server). Fixed with a try/catch falling back to `screen.availHeight`
   (always readable cross-origin); production (same-origin, the committed
   `dist/`) is unaffected and was re-verified working after the fix.
2. `XrayViewer.tsx`'s heatmap-colorizing `useEffect` depended only on
   `[heatmapRaw, opacity, highlightArea]`, but the `<canvas>` it draws onto
   is only mounted in the DOM while the "AI overlay"/"Compare" view is
   active (never in "Original" view). Switching into overlay view for the
   first time mounted a fresh, never-drawn canvas (stuck at the browser's
   default 300x150 blank size) with nothing to re-trigger the draw, since
   none of the effect's own dependencies changed on a view-mode switch --
   the overlay was silently invisible. Fixed by adding
   `showOverlayLayer`/`showCompare` to the dependency array. Verified via
   direct in-page JS (`canvas.getImageData`) before and after: 0 non-zero
   pixels at a default 300x150 canvas -> all 50,176 pixels real jet-colorized
   data at a correct 224x224 canvas, then confirmed visually.

Also found and dismissed as a pure test-environment artifact (not a real
app bug -- confirmed by loading the same built bundle standalone at its own
origin, which rendered correctly): the automation browser's OS/Chrome-level
forced-dark-content handling initially inverted the page's light theme
inside the iframe specifically; added `color-scheme: light` to `tokens.css`
as a standard, harmless opt-out regardless.

**Not yet done / follow-ups for a future session:**
- Not committed or pushed yet.
- No CI frontend job (`.github/workflows/tests.yml` + a `setup-node`/`tsc`/
  `npm run build`/dist-diff job) -- Risk #2's mitigation, still manual only.
- No further manual smoke test beyond the one flow above (DICOM/.dicom
  upload, a non-X-ray rejection, a sub-256px image, a 30MB client-side
  reject, mobile width, `prefers-reduced-motion` emulation) -- the plan's
  own full manual-smoke-test checklist is only partially exercised.
- Deliberately out of scope for Phase 1 (unchanged from the original plan):
  the 3D anatomical explorer (Phase 2) and Previous Studies/How It
  Works/About (Phase 3) -- nav shows only "New Screening" as active.

**Real results from running D1/D2 for real against `fedavg_secagg`:**
- D1 (`scripts/derive_app_decision_policy.py`): `decision_threshold=0.55`,
  `temperature=1.1283` (this checkpoint is measurably overconfident, needs
  softening -- opposite direction from round 9's own mild sharpening).
  `abstention_half_width=0.05` -- but note the REAL finding: this
  checkpoint's predictions cluster much more heavily near the decision
  boundary than the fine-tuned checkpoint's do (895+832 of 4844 val examples
  fall in calibrated [0.5,0.7) alone) -- actual abstain rate at half_width
  0.05 is ~21%, not the nominal ~10% target. Every tested half-width
  overshoots; 0.05 is just the closest available. Written into
  `conf/app.yaml`'s `fedavg_secagg` entry with a comment explaining this.
- D2 (`scripts/precompute_app_deferral_ood.py`): `deferral_threshold=0.6907`,
  3 IsolationForest models committed to
  `outputs/app_artifacts/fedavg_secagg/` (~5.5MB total, force-added past
  `.gitignore`'s blanket `outputs/` rule, same pattern as the earlier
  checkpoint-bundling work).
- Verified the FULL pipeline end-to-end using ONLY these precomputed
  artifacts (no live feature cache) -- simulating exactly what Cloud will
  have: real "Pneumonia" prediction, real focus region (right lung, lower
  zone), real OOD/certainty values, all correct.

**Next step in a future session:** Phase 1B (the Streamlit polling loop --
`ThreadPoolExecutor` + `session_state` + `st.rerun()`, per the plan's own
1B section) and rewriting `app/streamlit_app.py` to actually call
`app/analysis_job.py::run_job` and render the `pneumoscan(...)` component
with real props from `app/presentation.py`. The backend is fully ready for
this -- nothing in 1B/1C/1D needs new backend work, only Streamlit-side
wiring and the actual React frontend (1E, not started at all -- still just
the Phase 0 spike `App.tsx`).

### 1A — backend (Python only, testable in CI)

**`app/inference.py`:**
- Add `gradcam_heatmap: np.ndarray` (float32, 224x224, [0,1]) and
  `gradcam_target_class: int` to `InferenceResult`, filled from
  `overlay.heatmap` (currently discarded at line 217).
- Add `progress_callback: Callable[[str, str], None] | None = None` to
  `run_full_inference`, called as `(stage, "start"|"done")` for
  `"preprocess"`, `"mc_dropout"`, `"ood"`, `"gradcam"`. Exceptions propagate,
  never swallowed. Expose stage names as `INFERENCE_STAGES`.
- Swap OOD before Grad-CAM to match the design's stated order (test enforces
  identical output vs. current order, seeded).
- Add `SUPPORTED_UPLOAD_EXTENSIONS = (".jpg", ".jpeg", ".png", ".dcm",
  ".dicom")` as single source of truth, fixing the `.dicom` upload-rejection bug.
- Add an optional `frozen_model` pass-through to `check_is_xray` so the app
  can cache the frozen backbone instead of rebuilding it per call.
- Add `decode_uploaded_image_with_meta(bytes, name) -> (bgr, DecodedMeta)`
  where `DecodedMeta` has `format`, `width`, `height`, `projection | None`
  (from DICOM `ViewPosition` when present). Existing `decode_uploaded_image`
  stays untouched.

**New `app/presentation.py`** (pure, no Streamlit import — the one place that
enforces "never fabricate"):
- `certainty_label(entropy, deferral_threshold | None) -> "High"|"Medium"|"Low"|None`
  — ratio-based (< 0.5 High, < 1 Medium, else Low), returns `None` for a
  missing/non-finite threshold (never defaults to "High"/"Low").
- `image_check(ood_flags) -> "Unusual"|"Typical"|None` — `all()` over flags,
  `None` if no detectors (reuses existing logic, never fabricates "Typical").
- `compute_focus(heatmap, label) -> {side, zone, u, v} | None` plus a `spread`
  flag — argmax to pixel-center (u,v), the README's exact side/zone mapping,
  only for label=="Pneumonia" within the stated bounding region.
- `quality_checks(bgr, meta)` — `resolution_ok` (min side >= 256),
  `grayscale` (mean channel-diff on a 48x48 downsample, threshold 12, same
  method as the prototype but run server-side on real pixels).
- `build_props(...)` — the single JSON-safe dict builder feeding the component.
- `make_display_jpeg(bgr, max_edge=1600)`, `heatmap_to_uint8_bytes(heatmap)`.

**New `app/analysis_job.py`** (pure, threading only):
- `AnalysisJob`: lock-protected stage statuses, `queued`, `result`, `gate`, `error`.
- `run_job(job, bgr, *, gate, frozen_model, model, cfg)`: stage `gate` first
  (unavailable if no gate weights, `rejected` if `is_xray` is False), then
  `run_full_inference(progress_callback=job.on_stage)`.

**`conf/app.yaml`:**
- Add `active_configuration: fedavg_secagg`.
- Add the D1-derived `decision_threshold`/`abstention_half_width`/`temperature`
  values for the `fedavg_secagg` entry (same style/comments as the existing
  two calibrated entries).
- Add the D2-derived deferral threshold value, and reference the 3 committed
  per-hospital IsolationForest joblib files.

**New `scripts/derive_app_decision_policy.py`** (D1): mirrors the existing
validation-set threshold-sweep methodology (`docs/adr1_groupnorm_fallback.md`
§10/§17) for the `fedavg_secagg` checkpoint specifically. Outputs the three
calibration values to write into `conf/app.yaml`.

**Precompute step for D2:** derive the secagg deferral threshold (one float)
and fit/export the 3 per-hospital `IsolationForest` OOD detectors as small
joblib files (pinned `scikit-learn==1.9.0`), committed alongside the app
(small files, unlike the 576MB feature cache itself).

### 1B — progress mechanism: worker thread + callback + rerun polling (recommended)

- The "analyze" event submits `run_job` to an `@st.cache_resource`
  `ThreadPoolExecutor(max_workers=1)` — a single global worker also
  serializes inference across visitors, protecting Cloud RAM; a second
  concurrent visitor honestly sees `queued: true` ("Waiting for the analysis
  service…").
- Job stored in `st.session_state["job"]`. While running, the script renders
  the component with `job.snapshot()`, then `time.sleep(0.3)` + `st.rerun()`.
- The component has a stable `key="pneumoscan"` so reruns update props without
  remounting — React state (mode, zoom, 3D state) survives across polls.
- Progress moves at exactly 5 real stage boundaries (gate, preprocess, MC,
  OOD, Grad-CAM) — never fake per-substep precision. Within a running stage,
  the UI shows only an indeterminate shimmer.
- **Tradeoff, stated plainly:** this resends args (~150-250KB display JPEG +
  ~50KB heatmap) a few times a second for the few seconds inference takes.
  The alternative (splitting `run_full_inference` into resumable stage
  functions sequenced by `streamlit_app.py` itself, with tensors kept in
  `session_state` between reruns) sends less data but is architecturally
  messier and doesn't use the approved callback design as directly. Recommend
  the thread+polling approach; note the fallback exists if bandwidth becomes
  a real problem in testing.

### 1C — Python -> JS props contract (`protocol_version: 1`)

Every nullable field renders as "Not available" in the UI — JS never
substitutes a default. Key mappings (full table, from the design's own data
contract, cross-referenced against real field names):

| Prop | Source |
|---|---|
| `phase` | Python state machine (`upload`/`quality`/`analyzing`/`review`/`rejected`/`error`) |
| `study` | `study_id` = `"PS-"` + first 6 hex chars of upload SHA-256; width/height from decode; `projection` from DICOM `ViewPosition` or null; patient always "Not recorded" |
| `display_image` | decoded original (DICOM windowed server-side), JPEG, long edge <=1600 |
| `quality` | `presentation.quality_checks` + `XrayGateResult` |
| `progress` | `AnalysisJob.snapshot()` |
| `result.label/confidence/prob_pneumonia/entropy/abstained` | `InferenceResult` fields directly |
| `result.certainty` | `certainty_label(entropy, real_threshold_or_None)` — real threshold now available per D2 |
| `result.deferred`/`deferral_threshold` | real values per D2 (was: always null before D2) |
| `result.image_check` | `image_check(ood_flags)` — real values per D2 |
| `result.ood_sites` | `ood_flags`/`ood_scores` per hospital — real values per D2 |
| `result.gradcam_target` | `gradcam_target_class` |
| `result.focus`/`focus_spread` | `compute_focus(gradcam_heatmap, label)` |
| `result.decision_threshold`/`temperature`/`abstention_half_width` | real values per D1 (was: null before D1) |
| `heatmap` | `gradcam_heatmap` as 224x224 uint8 bytes; JS colorizes client-side (jet ramp) for the opacity/intensity sliders and hover readout |
| `error` | decode failure / job exception |

With D1 and D2 resolved, the only remaining "Not available" cases in the
handoff's own data-contract table are: Previous Studies records (no storage
by design — empty state, never fake records) and Patient field (always "Not
recorded" — never collected).

### 1D — JS -> Python events

`{event_id, type, payload}` — Python ignores any `event_id` it's already
processed (v1 component values persist across reruns). Events: `upload`
(base64 file data — top-level bytes only, `Uint8Array` nested in JSON would
serialize wrong), `use_sample` (loads committed `app/assets/sample_xray.jpg`,
the README's CC0 Häggström image — add attribution on the About page),
`analyze`, `new_screening`, `dismiss_error`. Client caps uploads at 25MB.
UI-only state (mode, screen, viewer zoom/pan/invert/compare, all 3D state)
stays client-side, never round-trips. `st.file_uploader`, the spinner, and
the Advanced model `selectbox` are removed; `DEFAULT_CONFIG_KEY` is replaced
by `active_configuration` from the yaml.

### 1E — frontend for New Screening

- Plumbing: `bridge.ts` (args hook, `sendEvent` — already exists from Phase
  0, extend it), `contract.ts` (TS types mirroring 1C, new), `tokens.css`
  (README colors/radii/type scale, new), fonts via
  `@fontsource/ibm-plex-sans`/`@fontsource/ibm-plex-mono` (bundled, not
  fetched live), `lib/na.ts` (single `fmt()` null->"Not available" helper).
- Components: `Header`, `StepBar`, `StudyCard`, `ImageCheckCard`,
  `viewer/XrayViewer` (+ `useZoomPan`, `heatmap.ts`, `CompareSplit`,
  `EmptyDropZone`, `ScanLine`), `right/UploadIntro`, `QualityCard`,
  `AnalyzingCard` (5 real sub-steps), `ResultClinician`, `ResultPatient`,
  `WhereAILooked`, `RejectedCard`.
- Sub-phase split: 1a = pipeline + cards + basic Original/Overlay viewer;
  1b = full viewer (wheel zoom 1-8x around cursor, pan, compare handle,
  invert, full-screen + Esc, R/L markers, hover readout, sliders), motion,
  `prefers-reduced-motion`.
- Nav shows only New Screening in this phase; "Open 3D" hidden until Phase 2.

`app/streamlit_app.py` rewritten to ~150 lines: page config + CSS shell,
cached loaders (kept, plus a cached frozen backbone), the event handler, job
polling, the `pneumoscan(...)` call. Delete `app/components.py` and
`app/theme.py` once nothing references them (Phase 3 at the latest). Also
remove the Phase 0 `?phase0_spike` block and `App.tsx` spike at this point.

## Phase 2: 3D anatomical explorer (frontend only, zero server cost) — COMPLETE (2026-09-28)

Owner-directed (2026-09-28), alongside removing the dead Previous
Studies/How It Works/About nav links (Header.tsx's `NAV_ITEMS` now just
`["New Screening"]` — those were disabled buttons with nothing behind them,
Phase 3 never started, more confusing than useful).

1. **Done.** `lung-model.html` copied into `frontend/public/`, with its
   import map repointed from the live `unpkg.com/three@0.184.0` URLs to
   local `./vendor/three/...` paths — the only edit to the file itself.
   `three.module.js`/`three.core.js`/`OrbitControls.js` vendored from unpkg
   into `public/vendor/three/`, SHA-384 verified byte-identical to the
   integrity hashes already pinned in the original file (zero version
   drift). Verified live in a browser, standalone (`/lung-model.html`
   directly, bypassing React/Streamlit entirely): the 3D scene renders
   correctly — lungs, bronchial tree, trachea, rib cage, orientation gizmo —
   with zero console errors, before any React work started on top of it.
2. **Done.** `explorer/LungFrame.tsx`: iframe wrapper, message listener
   hardened with `e.source === iframe.contentWindow`, de-duplicated
   `pneumoscan-lung`/`pneumoscan-view` state posting (verified against the
   file's own `addEventListener('message', ...)` handler and `pick()`/
   `view()` functions, not just the README prose), 8s no-`ready` timeout
   falling back to the X-ray view.
3. **Done.** `explorer/{Explorer3D,LayersPanel,ViewPanel,InfoPanel,
   ZoneChips,XraySection,ConfidenceRing,types}.tsx` — all the overlay panels,
   the X-ray/3D crossfade, real data wired in (never fabricated): `focus`
   from `result.focus`, `scan.progress` from real `progress.fraction`,
   confidence/certainty in the info panel and confidence ring from the real
   result. AI-focus layer shares state with the New Screening viewer's own
   "AI highlight" toggle, not a duplicate. X-ray view reuses
   `viewer/heatmap.ts`'s real colorizer, not a reimplementation.
4. **Done.** "Open 3D" enabled in `ResultClinician.tsx`'s footer (step 3's
   placeholder text from Phase 1 replaced with a real button) — scrolls to
   the explorer and forces the 3D view.

Two real bugs found and fixed via code review before any browser testing:
a crossfaded-out layer kept default pointer-events, silently blocking
clicks on the visible layer underneath (fixed: `pointerEvents: "none"` on
the hidden layer); the zone-chip selection path (`side: "both"`) wasn't
handled by the info panel's title/description functions, which only knew
`"left"`/`"right"` (fixed).

**Verified live** (`tsc --noEmit` + `npm run build` clean, both frontend
tests passing, then a real browser smoke test against a local Streamlit
process serving the built `dist/`): section renders below the New
Screening columns once an image is loaded; X-ray view shows real zone
lines/R-L markers/caption; switching to "3D anatomy" renders the live
model with working Layers panel, View panel (Front/Left/Right/Top
reflecting the iframe's own reported camera direction), zoom, auto-rotate,
breathing; clicking a structure sends a real `select` message and the info
panel shows the real tooltip copy; running a real analysis and clicking
"Open 3D" correctly scrolls in, switches to 3D, and shows the *real* result
(90% confidence, "No pneumonia pattern detected", Certainty: High) in both
the confidence ring and info panel; the "AI focus" layer is correctly
disabled for this Normal-label result (no `result.focus` — matches the
"never fabricate" rule, since `compute_focus` only ever returns a real
value for a Pneumonia label). Zero console errors throughout the full
interaction sequence.

The heaviest part (shaders) is reused as-is from `lung-model.html`; the React
work here was panels and message-passing only, confirmed by the above.

### Phase 2 follow-up: real Grad-CAM heatmap cloud in 3D — COMPLETE (2026-09-28)

Owner-directed same-day follow-up ("can grad cam be shown in 3d model
version as well... in the same way its shown in normal 2d way by changing
opacity"). Chose "multi-point glow field" (owner-selected over per-zone
bands or full texture projection) after inspecting the shader: the
existing `uFocusC`/`uFocusOn`/`uFocusR` uniforms already render a single
gaussian glow around one world-space point -- this is the *same* real
technique repeated many times, not a new rendering approach.

Real edits to `lung-model.html` itself this time (previously only its
import map changed) -- the only file that needed them, since the point
math and glow rendering are inherent to the shader, not the React wrapper:
- `lungFrag`: new `uHeatOn`/`uHeatPoints[16]`/`uHeatWeights[16]` uniforms,
  a `jetColor()` GLSL function matching `viewer/heatmap.ts`'s own ramp
  exactly (same 5 color stops, so the 3D field reads as the same heatmap,
  not a different palette), and a 16-point gaussian-sum loop blended into
  the existing fragment color/alpha right after the single-point focus glow.
- JS: a `heatG` marker pool (16 empty `Object3D`, same parent/pattern as
  the existing `focusG`), a `heat` entry in the `F` easing object (eased
  toward 1 whenever real points exist and "AI highlight" is on), and a
  per-frame block computing each marker's world position from real
  `(side, u, v)` data using the *exact* same rest-space formula
  `onFocus()` already uses for the single point (verified by reading that
  function's own source, not re-derived).
- `app/pneumoscan_component/frontend/src/explorer/heatPoints.ts` (new):
  downsamples the real raw 224×224 heatmap into a 16×16 grid, max-pools
  each cell, thresholds by the *same* `highlightArea` value the 2D slider
  already owns, and returns the top 16 weighted points -- opacity-scaled
  by the *same* `opacity` value the 2D slider already owns. No new UI: the
  existing Opacity/Highlight-area sliders (`App.tsx`, previously only
  wired to the 2D viewer) are now also threaded into `Explorer3D`, so
  there is exactly one set of controls for both views, not a duplicate.

**Verified live**, not just built: `tsc --noEmit`/`npm run build` clean,
zero console/shader-compile errors in a real browser. Cross-checked the
3D result against the 2D heatmap for the same real analysis (sample X-ray,
`fedavg_secagg`) -- both show the same two hotspots (right-middle,
left-middle zones), confirming the coordinate mapping is correct, not just
plausible-looking. Confirmed the Opacity slider actually drives the 3D
cloud's intensity: set to 0, the cloud fully disappears; restored, it
reappears matching the 2D view. This works for *any* result (not gated to
Pneumonia-only like the single-point `focus`/`compute_focus`), since it
samples the real heatmap directly rather than the backend's single-peak
summary -- confirmed live on this session's own Normal-label test case,
which still showed a real (if lower-confidence) attention pattern in 3D.

## Phase 3: Previous Studies, How It Works, About (static, frontend only) — NOT STARTED

- **Previous Studies:** empty state only, no fake records.
- **How It Works:** 5 step cards, animated SVG federated-flow diagram,
  collapsible Technical details — every claim checked against reality for
  `fedavg_secagg` specifically (DP is off; TLS verified only if that ablation
  row actually used it); step-1 copy updated to reflect DICOM is supported.
- **About:** description, limitations, team, faculty guide, CC0 sample attribution.
- Update docs: rewrite `docs/frontend.md` (architecture, dev loop, build-and-commit
  rule), CLAUDE.md §16.1a (new Node toolchain, fixed secagg config, D1-D4
  resolutions), `docs/SESSION_STATE.md`.

---

## Directory layout

```
app/
  streamlit_app.py            # rewritten: shell CSS, loaders, events, job polling
  inference.py                 # + heatmap field, progress_callback, ext constant, meta decode
  presentation.py               # NEW, pure: props builder + "never fabricate" rules
  analysis_job.py                # NEW, pure: threaded job + stage snapshots
  assets/sample_xray.jpg          # NEW (CC0)
  pneumoscan_component/
    __init__.py                    # declare_component wrapper (dist vs dev URL) -- DONE (Phase 0)
    frontend/
      package.json  package-lock.json  .nvmrc  tsconfig.json  vite.config.ts  index.html  -- DONE (Phase 0)
      public/lung-model.html  public/vendor/three/...   (Phase 2)
      src/{main.tsx, App.tsx, bridge.ts, contract.ts, tokens.css, lib/, components/, viewer/, explorer/, screens/}
        # main.tsx, App.tsx (spike, to be replaced), bridge.ts -- DONE (Phase 0); rest NOT STARTED
      dist/                        # COMMITTED (requires the .gitignore negation rule) -- DONE (Phase 0)
scripts/
  derive_app_decision_policy.py   # NEW (D1) -- NOT STARTED
tests/
  test_app_inference.py            # extended -- NOT STARTED
  test_app_presentation.py          # NEW -- NOT STARTED
  test_app_analysis_job.py           # NEW -- NOT STARTED
  test_frontend_bundle.py             # NEW -- NOT STARTED
.github/workflows/tests.yml          # + frontend job -- NOT STARTED
```

Dev loop: `npm run dev` (port 5173) with
`PNEUMOSCAN_COMPONENT_DEV_URL=http://localhost:5173` alongside
`uv run streamlit run app/streamlit_app.py`. Before each commit:
`npm run build`, then commit `dist/`.

## Testing and verification

**Pytest** (existing CI; the committed secagg checkpoint makes real-model
tests CI-runnable):
- `test_app_inference.py`: heatmap shape/dtype/range; callback fires in
  exact start/done order for all 4 stages; identical results with/without
  callback and against the pre-swap OOD/Grad-CAM order (seeded); `.dicom`
  extension accepted; DICOM meta decode against an in-test generated pydicom file.
- `test_app_presentation.py`: `certainty_label` bands + `None` for
  `None`/`inf`/`NaN` threshold (never "High"); `image_check` `None` for no
  detectors, "Unusual" only when all 3 flag; `deferred` null without a real
  threshold; `compute_focus` boundary cases (u=0.5, v edges, non-Pneumonia ->
  `None`, spread flag); `build_props` is JSON-serializable; no string "Low"
  appears anywhere an input is actually missing.
- `test_app_analysis_job.py`: fake inference function checks stage
  transitions, gate rejected, gate unavailable, exception capture, queued flag.
- `test_frontend_bundle.py`: `dist/index.html` exists; every referenced
  `./assets/*` exists; no absolute `/assets` paths; (Phase 2+) `lung-model.html`
  and vendored three.js files exist.
- `AppTest.from_file("app/streamlit_app.py").run()` raises no exception from
  fresh state — verify in Phase 0 that `AppTest` tolerates a v1 component.

**CI frontend job:** `setup-node` (matching `.nvmrc`), `npm ci`,
`tsc --noEmit`, `npm run build`, then
`git diff --exit-code app/pneumoscan_component/frontend/dist` — catches a
stale committed `dist/` before it reaches Cloud.

**Manual smoke test** (via the `run` skill, dev mode then built-`dist` mode):
JPG/PNG/.dcm/.dicom uploads + sample image; a non-X-ray photo (rejected) and
random noise; sub-256px image (resolution warning); 30MB file (client-side
error); clinician/patient switch; every viewer control;
`prefers-reduced-motion` emulation; clean console; progress steps visibly
follow real backend stages; real certainty/OOD/deferral values appear
(now real everywhere, per D2 — no local-vs-Cloud gap to check).

**On the actual Streamlit Cloud deployment:** app boots (check logs); no 404
on component assets; secagg inference completes end-to-end, timed; two
concurrent sessions show the queued state with no OOM; mobile width;
(Phase 2) 3D loads with zero requests to unpkg/Google (everything vendored).

## Risks

1. **`dist/` gitignored -> blank iframe on Cloud.** FIXED (Phase 0):
   `.gitignore` negation rule added and verified.
2. **Stale committed `dist/`** serving an old UI or mismatched props contract.
   Fixed by the CI rebuild-and-diff job (not yet built — do in Phase 1);
   Python should fail loudly on a `protocol_version` mismatch rather than
   rendering silently wrong.
3. **Iframe sizing** (content-height iframes break sticky headers, `vh`
   units, full-screen, `scrollTo`). FIXED (Phase 0): pinned via
   `window.top.innerHeight`, verified working on live Cloud.
4. **Cloud resource limits** (shared CPU, ~1-2.7GB RAM): the single-worker
   executor + cached frozen backbone avoid concurrent inference spikes and
   repeated model rebuilds (Phase 1, not yet built); the 3D viewer costs the
   server nothing (runs entirely client-side in the browser).
5. **Polling bandwidth**: kept bounded by capping the display JPEG at
   <=1600px long edge; 25MB client upload cap stays well under Streamlit's
   200MB message-size limit.
6. **The `float("inf")` deferral-threshold sentinel is a trap** in the
   current code — it silently makes `uncertainty_label` return "Low"
   uncertainty (-> fake "High" certainty) and `deferred`/`flagged_ood` return
   `False` (-> fake "No"/"Typical") rather than "not available". The new
   presentation layer must check real availability explicitly, never rely on
   this sentinel — enforced by `test_app_presentation.py` (Phase 1).
7. **Background thread outliving its session** — job exceptions must be
   captured into `job.error`, never lost silently (Phase 1).
8. **External dependencies** (fonts, three.js, sample image) must be vendored,
   not fetched live — firewalled/hospital networks commonly block
   unpkg/Google, and this is a public demo that shouldn't depend on 3rd-party
   uptime for its core look (Phase 2/1E).
9. **Local Node 18 is EOL.** FIXED (Phase 0): pinned to Vite 5.4.x, works
   fine on Node 18. Does not affect Cloud (which never runs Node).
10. **Module name collision** with existing `app/components.py`. FIXED
    (Phase 0): named the new package `app/pneumoscan_component/`.
11. **WebGL missing** on some client browsers — 8s timeout falls back to the
    X-ray-only view rather than hanging (Phase 2).
