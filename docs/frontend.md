# Frontend / Demo Interface (OPT-6)

A Streamlit presentation layer over the already-trained federated checkpoints from
Stage 21's ablation campaign. **This is a demonstration layer only** — it does not
touch training, evaluation, privacy guarantees, or the federated pipeline in any
way; it only ever consumes a finished checkpoint for one-off inference, exactly as
scoped when this extension was concept-approved (CLAUDE.md §16.1a).

> **This is a research prototype for a B.Tech academic project. It is NOT a
> medical device, has NOT been validated for clinical use, and must NEVER be used
> to inform an actual diagnosis or treatment decision.** Every screen in the app
> repeats this.

## Running it locally

```bash
uv run streamlit run app/streamlit_app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`). No
separate build step, no separate frontend toolchain — it's a single Python
process reading directly from this repository's own files.

## Required artifacts

The app degrades gracefully (clear in-app messaging, never a crash) when any of
these are missing, but each feature needs its own prerequisite to actually work:

| Feature | Requires |
|---|---|
| Model selector + prediction | At least one checkpoint under `outputs/checkpoints/{ablation,centralized_baseline}/` (see `conf/app.yaml` for the exact expected paths) |
| Uncertainty / deferral decision | `data/partitions/hospitals_natural.json` + `data/feature_cache/` (Stages 4–9) |
| OOD detection gate | Same as above — detectors are built at app startup from real cached hospital features |
| Grad-CAM | Nothing beyond the checkpoint — computed live on the uploaded image |
| Research Results tab | `outputs/results/{calibration,privacy_attack,gradcam_localization,conformal,ood_detector}.json` (OPT-1–5's own scripts) and, for the ablation table specifically, a reachable `mlruns.db` (Stage 21) |

A fresh checkout that has run the full pipeline (Stages 0–23 + OPT-1–5) has
everything already in place.

## Expected input

A single chest X-ray image: JPEG, PNG, or a raw DICOM (`.dcm`). It is run through
the exact same preprocessing every other part of this project uses — OpenCV CLAHE
contrast enhancement (ADR-6, fixed parameters), then resize + ImageNet
normalization (Stage 7's eval transform). Nothing is re-implemented for the demo;
`app/inference.py` calls `src/data/preprocessing.py` and `src/data/transforms.py`
directly.

## What each displayed metric means

- **Prediction** — the class (Normal / Pneumonia) with the higher mean
  probability across 20 stochastic MC Dropout forward passes (Stage 19) — this
  *is* the prediction mechanism, not a separate point estimate layered on top.
- **Confidence** — that mean probability for the predicted class.
- **Uncertainty (Low / Medium / High)** — a human-readable band over the MC
  Dropout predictive entropy, relative to a threshold calibrated once against the
  real held-out validation set (see "Deferral decision" below).
- **Decision (accepted / review recommended)** — Stage 19's DG-10 deferral
  policy (owner-approved 2026-08-30: defer the highest-uncertainty 10% of
  predictions). DG-10 is defined relative to a *batch's* uncertainty
  distribution, which has no meaning for one newly-uploaded image in isolation —
  so the app calibrates the entropy cutoff once, at startup, against the real
  pooled validation set (`app/inference.py::calibrate_deferral_threshold`, which
  calls the exact same `compute_deferral` function DG-10 itself uses) and
  applies that frozen threshold to new images. This is the standard way to
  deploy a batch-relative policy for real one-at-a-time inference, not a new
  policy invented for the demo.
- **Input status (in-distribution / anomalous)** — OPT-5's per-hospital
  Isolation Forest gate. Shown per hospital (Hospital A / B / C each have their
  own detector, trained only on their own data, per OPT-5's architecture) rather
  than collapsed into one verdict, since collapsing them would misrepresent the
  genuinely per-hospital design.
- **Grad-CAM overlay** — which regions of the (preprocessed) X-ray most
  influenced the predicted class, via the same `src/explain/gradcam.py`
  Stage 18 already uses and Stage 18/OPT-3 already validated (quantitatively,
  against real RSNA bounding boxes — see the Research Results tab).

## Model / checkpoint selector

Exactly the 7 real trained configurations from Stage 21's live ablation campaign
are exposed (seed 42 each, matching `scripts/generate_explanations.py`'s existing
precedent of one canonical checkpoint per configuration): FedAvg with no privacy
protection, FedAvg + DP at each swept epsilon (1, 2, 4, 8), FedAvg + Secure
Aggregation, and the centralized (non-federated, non-private) ceiling — the last
one explicitly labeled "comparison only," never presented as a real deployment
candidate. No configuration is shown that the backend does not actually have a
trained checkpoint for.

## Limitations

- **Presentation-only, by design.** A broken or unavailable demo does not affect
  any reported research result — every number on the Research Results tab is
  read from files OPT-1–5 already generated and validated independently.
- **The OOD gate's 5% flag-rate target is a placeholder**, not an owner-approved
  clinical policy the way DG-10's 10% deferral rate is (see `docs/ood_detection.md`).
- **No authentication, no persistence, no multi-user concerns** — this is a
  single-user local demo, not a deployed service. (Deployment is explicitly
  out of scope for this pass — see below.)
- Uploading an image that isn't a chest X-ray at all will still produce a
  prediction unless the OOD gate flags it — the gate is a safety signal to
  weigh, not a hard block on the prediction being computed at all.

## Deployment (not done in this pass)

Per explicit instruction, this pass stops at "runs correctly locally." A future
deployment step would need, at minimum: a decision on hosting (this app makes no
network calls, has no secrets, and reads only local files, so it is portable to
most standard Streamlit hosting options), a decision on whether/how to ship the
`outputs/`, `data/feature_cache/`, and `data/partitions/` artifacts it depends on
(none of which are currently committed to the repository — see
`docs/reproducibility.md`), and the standard CLAUDE.md governance around any new
infrastructure change. None of that is implied or started by this document.

## Engineering notes

- `app/inference.py` contains zero Streamlit imports and zero new inference
  logic — it only sequences calls into `src/models`, `src/uncertainty`,
  `src/explain`, and `src/data`, so it is independently testable
  (`tests/test_app_inference.py`) without a running Streamlit process.
- `app/results_loader.py` reads `outputs/results/*.json` and calls
  `src.evaluation.tables.build_ablation_table()` directly — no research number
  is hardcoded in the frontend.
- `conf/app.yaml` is the one `conf/*.yaml` file in this project that the code
  it documents actually loads at runtime (Streamlit doesn't run under
  `hydra.main`, unlike the rest of this project, so OmegaConf is used directly).
