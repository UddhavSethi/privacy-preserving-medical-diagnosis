# Session State — Continuation Reference

**Purpose:** Enable a fresh Claude Code session to resume this project correctly without
re-reading full conversation history. `CLAUDE.md` remains the governing architecture
document — this file is a status snapshot, not a replacement for it. Read `CLAUDE.md`
and `docs/IMPLEMENTATION_PLAN.md` in full before acting; this is a pointer/summary layer
on top of them.

**Last updated:** 2026-08-29, end of Phase 1 / Stage 5. Phase 1 (Stages 3-5) is complete.

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

**Decision Gate DG-2 — RESOLVED 2026-08-29, written back into `CLAUDE.md` §14 and
`docs/IMPLEMENTATION_PLAN.md` Stage 4.** Owner chose **option (a): keep RSNA's native
Target grouping** (Normal + No-Lung-Opacity-Not-Normal both map to the negative class),
for benchmark comparability and to preserve the full 20,672-patient negative class. The
clinical caveat (model learns "abnormal-but-not-pneumonia" = "normal") is to be stated
as an honest limitation in the paper, not engineered around. Implemented in
`src/data/labels.py::load_rsna_records()`.

**Correction (found while implementing Stage 4): Kermany's normal-class patient-ID
limitation, as originally stated in `CLAUDE.md`/the plan, does not apply to the actual
data used.** The authoritative Mendeley source's filenames are
`<CLASS>-<accession-id>-<seq>.jpeg` for *both* NORMAL and PNEUMONIA — the accession id
is a genuine, groupable patient/study identifier for both classes (verified: zero id
collisions across classes or the source's own train/test split, across all 5,856
files). This is a different, better-structured naming convention than the third-party
Kaggle mirror the original limitation note was written against. `CLAUDE.md` §15 item 8
has been corrected accordingly.

Raw data and the two big archive zips are **not** committed (gitignored) but **do**
exist on disk right now — see §9 known-state notes.

**Frozen splits (Stage 4, `seed=1000`, `val_frac=test_frac=0.15`, patient-grouped +
label-stratified via `src/data/splitting.py::grouped_stratified_split`):**

| Source | train | val | test |
|---|---|---|---|
| Kermany (images / patients) | 4,180 / 2,155 | 841 / 463 | 835 / 463 |
| RSNA (images = patients) | 18,678 | 4,003 | 4,003 |

Full per-split label counts are in `data/partitions/{kermany,rsna}_splits.json`
(`summary` key) — both correctly proportion Normal:Pneumonia within ~1.5x of the overall
ratio per split (verified by `tests/test_splitting.py::test_class_balance_preserved_per_split`
against the general splitting logic; the real-data split was inspected manually, not
re-asserted by a dedicated test against the live files).

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
| 4 — Label harmonization & patient-level splitting | **Done** — DG-2 applied, both sources fully patient-grouped, 18 tests passing, splits frozen in `data/partitions/{kermany,rsna}_splits.json` | `28131be` |
| 5 — Hospital partitioning | **Done.** DG-3 resolved (report both). 29/29 tests passing. | `6d667cf` |
| 6 — CLAHE preprocessing + cache | **Done.** 37/37 tests passing. Full cache built: 5,856 Kermany + 26,684 RSNA images (46GB on local disk, not committed — see §9). MLflow artifact logged (experiment `clahe_cache`). | `91d7da8` |
| 7 — torchvision transforms/Dataset/DataLoader | **Done.** 48/48 tests passing. Smoke-tested end-to-end on real cached data from both sources, including multi-worker DataLoader. | (pending commit this session) |
| 8–23 | Not started | — |

**Phase 0 (Stages 0–2), all of Phase 1 (Stages 3–5), and Phase 2's Stages 6–7 are
complete. Stage 8 is next — see §11, this is the single largest technical-risk stage
in the whole project.**

## 8. Pending decisions / open decision gates

- **DG-2 (label harmonization): RESOLVED and applied**, see §4.
- **DG-3 (hospital-size imbalance handling, Stage 5): RESOLVED 2026-08-29** — owner
  chose "report both". `hospitals_natural.json` (A=5,856/B=C=13,342 images, ~4.5x
  imbalance) and `hospitals_natural_balanced.json` (A=B=C=5,856 images, B/C
  label-stratified-subsampled down via `src/data/partitioning.py::subsample_to_size()`,
  never upsampling A) are both frozen and committed. Both should appear in the ablation
  results/paper.
- **Dropout placement** (head-only vs. after dense blocks, `CLAUDE.md` §14 item, needed
  at Stage 8): open.
- **Target epsilon values** for the DP sweep + delta relative to dataset size (needed
  at Stage 14): open.
- **Client count** / default partition scheme for headline results (needed at Stage 5):
  open, related to DG-3.

No dependency, architecture, or technology-stack changes are currently awaiting
approval — the last one (adding `requests`) was resolved and committed in Stage 3.

## 9. Known state / things to be aware of (not bugs, but worth knowing)

- **~62GB of data on local disk, all gitignored, none of it committed**:
  `data/raw/kermany` (1.2G), `data/raw/rsna` (3.8G), `data/_downloads/ZhangLabData.zip`
  (7.9G, kept for provenance), `data/_downloads/rsna-pneumonia-detection-challenge.zip`
  (3.7G), and **`data/clahe_cache/` (46G — Stage 6's CLAHE output, keyed by
  `source/param-hash/relative_path.png`, `clip_limit=2.0`/`tile_grid_size=(8,8)`)**.
  Disk has 252GB free as of last check — not urgent, but the two archive zips can be
  deleted once you're confident the extracted+checksummed data is sufficient (the
  manifests retain the source hashes for provenance either way). If CLAHE parameters
  ever change, re-run `scripts/build_clahe_cache.py` — it writes to a new
  parameter-hash subdirectory rather than overwriting, so stale cache from an old
  parameter set will accumulate on disk unless manually cleaned up.
- **Kermany's official train/test split (this Mendeley source) has no validation
  fold** — resolved by Stage 4's own split, which carves train/val/test from scratch
  across *all* of Kermany's images regardless of the source's original train/test
  labels (kept only as `source_native_split` metadata, unused for our splitting).
- ~~Kermany's NORMAL-class filenames carry no patient identifier~~ — **this was wrong,
  corrected during Stage 4, see §4.** Both classes are fully patient-groupable in the
  actual source used.
- **RSNA's `stage_2_test_images/` (3,000 DICOMs) is unlabeled** (Kaggle's held-out
  competition test set, ground truth never released) and is correctly excluded from
  `load_rsna_records()` — all usable RSNA data comes from the 26,684-patient
  `stage_2_train_images/` + `stage_2_train_labels.csv` pool, which Stage 4 then splits
  into our own train/val/test.
- `docs/IMPLEMENTATION_PLAN.md`'s stale dataset-size estimates, the Kermany
  patient-ID claim, and the "no implementation has begun" closing line have all been
  corrected during Stage 4 — the plan document is now consistent with actual state
  (though `docs/SESSION_STATE.md`, this file, remains the authoritative live-status
  pointer per its own closing note in the plan doc).
- 18 pytest tests pass (8 from Stage 2 + 3 label tests + 7 splitting tests); working
  tree has Stage 4 changes staged/ready to commit as of writing this file.

## 10. Git status

**Branch:** `main`. `91d7da8` (Stage 6) is pushed to `origin/main`; Stage 7
(`src/data/transforms.py`, `src/data/datasets.py`, `conf/data/transforms.yaml`,
`tests/test_transforms.py`, `tests/test_datasets.py`, this file's update) is about to
be committed as the next commit on top of that. Check `git log --oneline -5` on
resume — this file is not re-updated after every single commit within a session, only
at natural pause points.

```
91d7da8 Stage 6: OpenCV CLAHE preprocessing and cache (ADR-6)
6d667cf Stage 5 complete: DG-3 resolved as "report both"
3ae7e78 Stage 5: hospital partitioning code + natural/Dirichlet outputs (DG-3 open)
28131be Stage 4: label harmonization and patient-grouped splitting (DG-2 applied)
74c6ea5 Stage 3 complete: RSNA acquired, checksummed, validated (DICOM)
```

**Standing authorization:** owner granted full autonomy for Phase 1 (commands,
downloads, commits, pushes) — stopping only at decision gates or genuine architecture
conflicts. Phase 2 (Stage 6) proceeded on repeated one-word "continue" replies rather
than a fresh explicit autonomy grant — treated as continued authorization by the same
session, but a **fresh session should not assume this transfers automatically**; if
unsure, ask before pushing through a later phase boundary unprompted.

## 11. Exact next recommended step

Phase 1 and Stages 6–7 are complete (`src/data/transforms.py` — separate
`build_train_transform`/`build_eval_transform`, no horizontal flip since chest X-ray
laterality is clinically meaningful; `src/data/datasets.py` — `ChestXrayDataset` +
`build_dataloader` reading from `data/clahe_cache/`; smoke-tested end-to-end on real
data from both sources including multi-worker loading).

Next is **Stage 8 — DenseNet121 frozen backbone + trainable head**, marked in the plan
as **the single largest technical risk in the project** and a **CRITICAL GATE (DG-4,
DG-6)**. This is where ADR-1 (freeze the backbone, train only a small head) either
gets proven out or fails and triggers the GroupNorm fallback — do this validation now,
not later, because failure here is cheap and failure discovered downstream (e.g. at
Stage 14's DP integration) is not.

What it needs: `src/models/densenet_head.py`, `src/models/freezing.py`,
`conf/model/densenet121.yaml`. Build an ImageNet-pretrained DenseNet121 with the
backbone frozen (BatchNorm forced to `eval()`, running stats frozen) and a small
trainable classifier head with deliberately inserted dropout (this also resolves
CLAUDE.md §14's still-open "dropout placement" pending decision — head-only vs. after
dense blocks — so that choice needs to be made and documented as part of this stage,
not silently defaulted).

**The stage's entire point is its acceptance tests, not just building the model:**
1. `opacus.validators.ModuleValidator` accepts the frozen-backbone model (already
   confirmed importable in Stage 1; this is the real test — does it pass validation).
2. Per-sample gradients compute correctly, and only for head parameters.
3. BatchNorm running statistics are provably unchanged after a training step.
4. A spike test: one DP-SGD step fits within the 4GB VRAM budget (RTX 3050) at the
   intended batch size.
5. Trainable parameter count lands in the expected ~1e5–1e6 range (not ~7M).

**If any of this fails**, the ADR-1 fallback (Opacus `ModuleValidator.fix()` to
GroupNorm, fine-tune more layers) requires explicit owner approval before adopting —
do not silently switch to it. This is exactly the kind of "requires changing approved
architecture" situation that should stop and ask, not proceed autonomously.
