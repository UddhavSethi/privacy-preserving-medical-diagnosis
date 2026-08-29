# Session State — Continuation Reference

**Purpose:** Enable a fresh Claude Code session to resume this project correctly without
re-reading full conversation history. `CLAUDE.md` remains the governing architecture
document — this file is a status snapshot, not a replacement for it. Read `CLAUDE.md`
and `docs/IMPLEMENTATION_PLAN.md` in full before acting; this is a pointer/summary layer
on top of them.

**Last updated:** 2026-08-29, end of Phase 1 / Stage 3.

---

## 1. What this project is

Privacy-preserving pneumonia detection via federated learning across simulated
hospitals. Full detail: `CLAUDE.md` §1–§2. One-line summary: hospitals train locally on
chest X-rays that never leave their storage; only DP-noised, SecAgg+-masked model
updates travel over TLS+client-auth to a FedAvg server. Explainability (Grad-CAM) and
uncertainty (MC Dropout) are added on top. The paper's core deliverable is a 6-row
ablation table quantifying what each privacy layer costs (`CLAUDE.md` §11.1).

## 2. Architecture (pointer, not restated in full — see CLAUDE.md §3, §5–§10)

- **Topology:** star/hub-and-spoke FL. One `ClientApp`/`ServerApp` codebase, two
  execution engines (simulation for measurement, Docker Compose deployment for
  demonstration) — ADR-8.
- **Model:** DenseNet121, ImageNet-pretrained, **frozen backbone** (BatchNorm in
  `eval()`, frozen running stats) + small trainable head — **ADR-1, the load-bearing
  decision of the whole project**. Not yet implemented (Stage 8).
- **Privacy stack:** Opacus DP-SGD (sample-level, per-client/local DP — ADR-2) +
  Flower SecAgg+ (ADR-3, no custom crypto) + TLS with client auth (ADR-4). None of
  this is implemented yet — Stages 14–16.
- **This is a CLI/script/notebook-driven ML research pipeline — there is no web
  frontend, no page routing, and no CMS/content-management layer.** ("Navigation
  architecture" and "content-management architecture" from a generic project template
  do not apply here; the closest analogues are the Hydra config tree (§6 below) and the
  dataset/manifest pipeline (§4 below), both covered there instead.)

## 3. Approved technology stack (exact pins — `pyproject.toml` / `uv.lock`)

Python 3.11.16 (pinned `>=3.11,<3.12` — not 3.12, ADR-5). Environment managed by `uv`.

| Package | Version | Notes |
|---|---|---|
| torch | 2.13.0+cu126 | CUDA 12.6 build, matched to RTX 3050 Laptop (4GB VRAM, driver 595.84) |
| torchvision | 0.28.0+cu126 | Pinned exactly to torch's required version |
| flwr | 1.35.0 | SecAgg+ verified importable: `flwr.client.mod.secaggplus_mod`, `flwr.server.workflow.SecAggPlusWorkflow` |
| opacus | 1.6.0 | `opacus.validators.ModuleValidator` verified importable |
| opencv-python-headless | 5.0.0.93 | CLAHE only (ADR-6) |
| hydra-core / omegaconf | 1.3.5 / 2.3.1 | |
| mlflow | 3.15.2 | **Tracking backend is SQLite, not filesystem — see §6** |
| numpy | 2.4.6 | **Not latest (2.5.2)** — 2.5.x dropped Python 3.11 support |
| pandas | 2.3.3 | **Not latest (3.0.5)** — mlflow 3.15.2 requires `pandas<3` |
| scikit-learn, pydicom, pillow, tqdm, matplotlib | 1.9.0 / 3.0.2 / 12.3.0 / 4.70.0 / 3.11.1 | |
| grad-cam | 1.5.7 | Chosen over Captum — narrower, purpose-built |
| kaggle | 2.2.4 | RSNA acquisition |
| requests | 2.34.2 | Added explicitly in Stage 3 (was already transitively resolved at this version — no-op for the installed env, just makes an implicit dep explicit) |
| pytest (dev group) | 9.1.1 | |

**Deliberately excluded, do not introduce:** TensorFlow/TF-Federated/TF-Privacy, NVFlare,
OpenFL, PySyft, CrypTen, W&B/cloud tracking, blockchain, ViTs/medical foundation models,
Kubernetes. (`CLAUDE.md` §4.)

## 4. Datasets (Stage 3 — see §7 for exactly what's done)

**Decision (resolved 2026-08-29, recorded in `CLAUDE.md` §14):** two-source strategy —
**Kermany = Hospital A**, **RSNA Pneumonia Detection Challenge = Hospitals B & C**.

| Dataset | Source | Size (measured, not estimated) | Files | Location |
|---|---|---|---|---|
| Kermany | Mendeley Data, doi:10.17632/rscbjbr9sj.3 | 8.4GB zip (bundled w/ unrelated OCT data — plan's 1.2GB estimate was for a 3rd-party Kaggle mirror, not the authoritative source) | 5,856 JPEG images | `data/raw/kermany/CellData/chest_xray/{train,test}/{NORMAL,PNEUMONIA}/` |
| RSNA | Kaggle competition `rsna-pneumonia-detection-challenge` | 3.96GB (measured by paginating the Kaggle API before download — plan's ~12GB estimate was wrong) | 29,684 DICOM files | `data/raw/rsna/{stage_2_train_images,stage_2_test_images}/` + label CSVs at `data/raw/rsna/*.csv` |

Both validated: **zero checksum mismatches, zero corrupt files** (see
`data/manifests/{kermany,rsna}_validation_report.json`).

**RSNA DICOM findings (relevant to Stage 6 CLAHE preprocessing):** all images uniformly
1024×1024, `PhotometricInterpretation=MONOCHROME2` for all files, **no**
`RescaleSlope`/`RescaleIntercept` present on any file (raw pixel values — no linear
rescale step needed before CLAHE, simplifying Stage 6 vs. the generic DICOM risk
`CLAUDE.md` flags).

**RSNA label structure** (26,684 unique patients, from `stage_2_train_labels.csv` +
`stage_2_detailed_class_info.csv`): Normal 8,851 / No Lung Opacity–Not Normal 11,821 /
Lung Opacity (pneumonia) 6,012. `stage_2_train_labels.csv` has one row per bounding box
— multiple rows per patient for some Lung Opacity cases; dedupe by `patientId` before
counting (the validator in `scripts/validate_datasets.py` already does this correctly).

**Decision Gate DG-2 — RESOLVED 2026-08-29, not yet written back into CLAUDE.md/docs.**
Owner chose **option (a): keep RSNA's native Target grouping** (Normal +
No-Lung-Opacity-Not-Normal both map to the negative class), for benchmark
comparability and to preserve the full 20,672-patient negative class. The clinical
caveat (model learns "abnormal-but-not-pneumonia" = "normal") is to be stated as an
honest limitation in the paper, not engineered around. **Next session should write this
into `CLAUDE.md` §14 / `docs/IMPLEMENTATION_PLAN.md` Stage 4 section when Stage 4 is
implemented** (propose the diff first, per `CLAUDE.md` §17.1 governance).

Raw data and the two big archive zips are **not** committed (gitignored) but **do**
exist on disk right now — see §9 known-state notes.

## 5. Reproducibility spine (Stage 2 — implemented)

- `conf/config.yaml`: `seed=42`, `data_partition_seed=1000`, `client_sampling_seed=2000`
  tracked as three separate values (per `CLAUDE.md` §12's warning that the latter two
  are the most commonly forgotten in FL work).
- `src/utils/seeding.py`: `set_global_seed()`, `seed_worker()`, `make_generator()`.
  `torch.use_deterministic_algorithms(True, warn_only=<config flag>)`.
- `src/utils/logging.py`: `configure_logging()` / `get_logger()`.
- `src/utils/mlflow_utils.py`: `tracked_run()` context manager — logs fully resolved
  config (dotted-key flatten) + git SHA at run start.

## 6. Configuration decisions worth knowing

- **MLflow tracking backend is `sqlite:///mlruns.db`, not `file:./mlruns`.** MLflow
  3.15.2 deprecated the plain filesystem store (raises `MlflowException` by default).
  SQLite is still fully local/offline — does not conflict with `CLAUDE.md`'s "local,
  self-hosted, no third-party cloud" requirement. `mlruns.db`/`mlflow.db` are
  gitignored.
- **CUDA build is `+cu126`** (not cu129/cu130/cu132, which are also available for
  torch 2.13.0/cp311) — chosen as the stable middle ground for the RTX 3050 (driver
  595.84).
- **`torch.use_deterministic_algorithms` has a `warn_only` escape hatch** in
  `conf/config.yaml` (`deterministic.warn_only`) for the known risk that some CUDA
  kernels have no deterministic implementation — currently `false` (strict).

## 7. Stage-by-stage status (`docs/IMPLEMENTATION_PLAN.md` Part IV is authoritative for stage definitions)

| Stage | Status | Commit(s) |
|---|---|---|
| 0 — Repository foundation | **Done** | `038b7ab` |
| 1 — Pinned Python 3.11 env | **Done** | `4ab5922` |
| 2 — Config, seeding, MLflow tracking | **Done**, 8/8 tests passing | `86e36a7` |
| 3 — Dataset acquisition & validation | **Done**, both datasets, zero data-integrity issues | `3aea1f0` (Kermany), `74c6ea5` (RSNA) |
| 4 — Label harmonization & patient-level splitting | **Not started.** DG-2 is now resolved (see §4) — this was the blocker, Stage 4 can proceed | — |
| 5 — Hospital partitioning | Not started (DG-3 — client-size-imbalance handling — still open, raise before finishing this stage) | — |
| 6 — CLAHE preprocessing + cache | Not started | — |
| 7 — torchvision transforms/Dataset/DataLoader | Not started | — |
| 8–23 | Not started | — |

**Phase 0 (Stages 0–2) and Phase 1's Stage 3 are complete. Stage 4 is next.**

## 8. Pending decisions / open decision gates

- **DG-2 (label harmonization): RESOLVED** this session — option (a), see §4. Needs
  writing back into `CLAUDE.md`/plan docs when Stage 4 is implemented (propose diff
  first).
- **DG-3 (hospital-size imbalance handling, Stage 5):** open. Kermany (~5,860 images)
  vs. RSNA (~29,684 images) is a large natural imbalance. Plan recommends reporting
  both balanced and natural-imbalance results rather than picking one — not yet
  discussed with owner in depth.
- **Dropout placement** (head-only vs. after dense blocks, `CLAUDE.md` §14 item, needed
  at Stage 8): open.
- **Target epsilon values** for the DP sweep + delta relative to dataset size (needed
  at Stage 14): open.
- **Client count** / default partition scheme for headline results (needed at Stage 5):
  open, related to DG-3.

No dependency, architecture, or technology-stack changes are currently awaiting
approval — the last one (adding `requests`) was resolved and committed in Stage 3.

## 9. Known state / things to be aware of (not bugs, but worth knowing)

- **~15.6GB of raw data + archives currently on local disk**, all gitignored, none of
  it committed: `data/raw/kermany` (1.2G), `data/raw/rsna` (3.8G),
  `data/_downloads/ZhangLabData.zip` (7.9G, kept for provenance),
  `data/_downloads/rsna-pneumonia-detection-challenge.zip` (3.7G). Disk has 299GB free
  as of last check — not urgent, but the two zips can be deleted once you're confident
  the extracted+checksummed data is sufficient (the manifests retain the source hashes
  for provenance either way).
- **Kermany has no official validation split** in the Mendeley source (train/test
  only) — a val split must be carved from train at Stage 4/5.
- **Kermany's NORMAL-class filenames carry no patient identifier** (only PNEUMONIA
  filenames do, as `personNNN_bacteria/virus_NNN.jpeg`) — patient-level grouping
  (ADR-7) is only partial for this dataset; must be disclosed as a limitation.
- **`docs/IMPLEMENTATION_PLAN.md`'s two dataset size estimates were both wrong** (Kermany
  low by ~7x, RSNA high by ~3x) — corrected in commit messages and `CLAUDE.md` §14, but
  the plan document's own Stage 3 prose (line ~436, "approximately 1.2 GB... approximately
  12 GB") has **not** been edited to match. Low priority; doesn't block anything.
- All 8 pytest tests pass; working tree is clean; `main` is in sync with `origin/main`
  as of commit `74c6ea5`.

## 10. Git status

**Branch:** `main`, in sync with `origin/main` at `74c6ea5`. Working tree clean.

```
74c6ea5 Stage 3 complete: RSNA acquired, checksummed, validated (DICOM)
3aea1f0 Stage 3 (partial): Kermany acquisition, checksummed and validated
86e36a7 Stage 2: configuration, seeding and MLflow tracking
4ab5922 Stage 1: pinned Python 3.11 environment (ADR-5)
038b7ab Stage 0: repository foundation
```

**Standing authorization:** owner granted full autonomy for Phase 1 (commands,
downloads, commits, pushes) — stopping only at decision gates or genuine architecture
conflicts. Not yet confirmed whether this extends past Phase 1; ask if a fresh session
is unsure before pushing through a later phase boundary unprompted.

## 11. Exact next recommended step

1. Propose (then, on approval, make) the `CLAUDE.md` §14 / `docs/IMPLEMENTATION_PLAN.md`
   edit recording DG-2's resolution (option a).
2. Implement Stage 4: `src/data/labels.py` (RSNA label mapping using DG-2's resolution;
   Kermany label standardization), `src/data/splitting.py` (patient-grouped train/val/test
   splitting — handle Kermany's partial-patient-ID limitation explicitly), `scripts/build_splits.py`,
   committed split manifests at `data/partitions/*.json`. Required test: zero patient
   overlap between splits, deterministic given seed (`CLAUDE.md` §11.3).
3. Before finishing Stage 5 (hospital partitioning), raise DG-3 explicitly.
