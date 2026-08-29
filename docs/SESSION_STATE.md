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
  decision of the whole project**. Implemented and validated in Stage 8
  (`src/models/densenet_head.py`) — Opacus accepts it cleanly, 262,914 trainable
  params, 0.37GB/4GB VRAM for one DP-SGD step. No training loop exists yet (Stage 11+).
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
| 7 — torchvision transforms/Dataset/DataLoader | **Done.** 48/48 tests passing. Smoke-tested end-to-end on real cached data from both sources, including multi-worker DataLoader. | `dda0bed` |
| 8 — DenseNet121 frozen backbone + head | **Done — ADR-1's premise fully validated, no GroupNorm fallback needed.** 57/57 tests passing (9 new). See note below the table. | `80f533d` |
| 9 — Frozen-backbone feature cache | **Done.** DG-5 resolved (K=5 augmented views). 62/62 tests passing (5 new). Full cache built and measured — see note below the table. | (pending commit this session) |
| 10–23 | Not started | — |

**Stage 8 was the project's single largest technical-risk stage, and it passed cleanly
on the first attempt** (`src/models/densenet_head.py`, `src/models/freezing.py`,
`conf/model/densenet121.yaml`):
- Opacus's `ModuleValidator` accepts the frozen-backbone model outright (0 errors).
- Per-sample gradients compute correctly for exactly the 4 head parameters (2 Linear
  layers' weights/biases); 0 backbone parameters tracked.
- All 121 BatchNorm layers' running stats provably unchanged after a training step,
  even after calling `.train()` on the whole model (the `DenseNet121Head.train()`
  override is what makes that permanent).
- CUDA VRAM spike test: one DP-SGD-style step at batch_size=32 uses **0.37GB of the
  4GB RTX 3050 budget** — an 11x margin, not a near-miss.
- Trainable parameters: **262,914** (head: `Linear(1024,256) -> ReLU -> Dropout ->
  Linear(256,2)`), squarely in the plan's expected ~1e5-1e6 range. Total: ~7.2M,
  confirming the backbone itself is intact.
- **Dropout placement resolved: head-only** (a single `Dropout(p=0.3)` between the
  head's hidden ReLU and final classification layer) — this closes CLAUDE.md §14's
  previously-open pending decision. Tradeoff accepted: MC Dropout (Stage 19) will only
  capture last-layer uncertainty, not backbone-level uncertainty.
- **One real bug found and fixed during validation, not by inspection**: the head's
  `nn.ReLU(inplace=True)` broke Opacus's `GradSampleModule` backward hooks (in-place
  ops conflict with its view-tracking) — changed to `inplace=False`. Worth remembering
  if any future model code adds in-place ops anywhere in a path Opacus wraps.

**Stage 9 (`src/data/feature_cache.py`, `scripts/build_feature_cache.py`):**
- **`DenseNet121Head` was refactored** (`src/models/densenet_head.py`) to split the old
  single `self.head` Sequential into a parameter-free `self.pool`
  (`AdaptiveAvgPool2d`+`Flatten`) and a trainable `self.classifier` (the same
  `Linear->ReLU->Dropout->Linear` as before), plus a new `pooled_features(x)` method
  covering backbone->ReLU->pool. This is what lets Stage 9 cache the 1024-dim pooled
  vector and later train `classifier` directly on it. Backward compatible — all 9
  Stage 8 tests still pass unchanged after the refactor (no test referenced `.head`).
- **DG-5 resolved: K=5 augmented views** cached per training image (plus 1 deterministic
  eval-style view); val/test get 1 eval-style view each. Every training image's view
  index is seeded deterministically (`VIEW_SEED + i*1000 + v`) for reproducibility.
- **Cache size: 576MB** — matches the plan's own ~600MB estimate almost exactly.
  `data/feature_cache/<param-hash>/{kermany,rsna}_{train,val,test}.pt`.
- **Measured speedup (the stage's actual acceptance criterion, not a guess): 8.3x**
  per training step on real data — 2.61ms/step from a cached feature (classifier-only)
  vs. 21.76ms/step live (full frozen-backbone forward + classifier), benchmarked on 64
  real Kermany training images on the RTX 3050. Lower end of the plan's "one-to-two
  orders of magnitude" estimate but real and measured, not theoretical.
- Core correctness claim proven by test, not assumed: training the classifier from a
  cached eval-style feature is bit-for-bit identical (`atol=1e-5`) to training it via a
  live full-model forward pass, given the same seed and no augmentation.

**Phase 0 (Stages 0–2), all of Phase 1 (Stages 3–5), and Phase 2's Stages 6–9 are
complete. Stage 10 is next (required, no open decision gates).**

**Also recorded (documentation only, not implemented):** two optional extensions were
raised, evaluated, and approved-in-concept by the owner on 2026-08-29 — **OPT-5**
(Isolation Forest OOD detection gate, scoped to chest-X-ray anomaly detection only, not
federated-update anomaly detection — that interpretation was evaluated and explicitly
rejected as conflicting with SecAgg+) and **OPT-6** (a Streamlit demo interface,
presentation-only). Both are documented in CLAUDE.md §16.1a and
`docs/IMPLEMENTATION_PLAN.md` Phase 6 (OPT-5/OPT-6), committed in `5b52bc0`. **Neither
is part of the 24-stage critical path, and implementation of either still needs a
separate explicit go-ahead** — do not start building either just because they're
documented. Earliest they could start: OPT-5 after Stage 11 (needs a trained model);
OPT-6 after Stages 11/18/19 (needs a trained model, Grad-CAM, and MC Dropout).

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

- **~62.5GB of data on local disk, all gitignored, none of it committed**:
  `data/raw/kermany` (1.2G), `data/raw/rsna` (3.8G), `data/_downloads/ZhangLabData.zip`
  (7.9G, kept for provenance), `data/_downloads/rsna-pneumonia-detection-challenge.zip`
  (3.7G), `data/clahe_cache/` (46G — Stage 6's CLAHE output, keyed by
  `source/param-hash/relative_path.png`, `clip_limit=2.0`/`tile_grid_size=(8,8)`), and
  **`data/feature_cache/` (576MB — Stage 9's cached pooled backbone features, keyed by
  `<transform-param-hash>/{kermany,rsna}_{train,val,test}.pt`, K=5 augmented + 1
  eval-style view per training image)**. Disk has 252GB free as of last check — not
  urgent, but the two archive zips can be
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

**Branch:** `main`. `5b52bc0` (OPT-5/OPT-6 documentation) is pushed to `origin/main`;
Stage 9 (`src/data/feature_cache.py`, `scripts/build_feature_cache.py`,
`tests/test_feature_cache.py`, the `src/models/densenet_head.py` pool/classifier
refactor, this file's update) is about to be committed as the next commit on top of
that. Check `git log --oneline -5` on resume — this file is not re-updated after every
single commit within a session, only at natural pause points.

```
5b52bc0 Record OPT-5 (Isolation Forest OOD gate) and OPT-6 (Streamlit demo) as approved-in-concept optional extensions
80f533d Stage 8: DenseNet121 frozen backbone + head — ADR-1 validated
dda0bed Stage 7: torchvision transforms, Dataset and DataLoader
91d7da8 Stage 6: OpenCV CLAHE preprocessing and cache (ADR-6)
6d667cf Stage 5 complete: DG-3 resolved as "report both"
```

**Standing authorization:** owner granted full autonomy for Phase 1 (commands,
downloads, commits, pushes) — stopping only at decision gates or genuine architecture
conflicts. Phase 2 (Stage 6) proceeded on repeated one-word "continue" replies rather
than a fresh explicit autonomy grant — treated as continued authorization by the same
session, but a **fresh session should not assume this transfers automatically**; if
unsure, ask before pushing through a later phase boundary unprompted.

## 11. Exact next recommended step

Stages 8 and 9 are both complete. ADR-1 is fully validated (Stage 8) and the feature
cache is built and measured (Stage 9, DG-5 resolved as K=5 views, 8.3x speedup). No
open decision gates remain in Phase 2.

Next is **Stage 10 — evaluation and metrics module** (`REQ`, no open decision gates):
`src/evaluation/metrics.py` (AUROC as the primary metric, plus AUPRC, sensitivity at
fixed specificity, specificity, F1, balanced accuracy, confusion matrix — accuracy
alone is explicitly not acceptable on this imbalanced data per CLAUDE.md §11.2),
`src/evaluation/bootstrap.py` (bootstrap 95% CIs on AUROC), `src/evaluation/reporting.py`
(multi-seed mean/std aggregation, results serialization, MLflow logging).

**Build this before Stage 11 (the first baseline) produces any number** — CLAUDE.md
explicitly warns that building the metrics module after the first baseline is "the
classic mistake": results get recomputed and tables silently disagree. Required tests:
metrics match scikit-learn references on synthetic data; degenerate cases (single-class,
all-correct, all-wrong) handled without crashing; bootstrap CIs reproducible under seed;
a known-input regression test. Threshold-dependent metrics (F1, sensitivity) need an
explicit, fixed threshold policy — decide and document it here, don't leave it implicit.

After Stage 10: **Stage 11 — local single-hospital baseline** (ablation row 1) is the
first stage that actually trains anything and produces a real number.
