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
| 9 — Frozen-backbone feature cache | **Done.** DG-5 resolved (K=5 augmented views). 62/62 tests passing (5 new). Full cache built and measured — see note below the table. | `dfb1d9f` |
| 10 — Evaluation and metrics module | **Done.** 85/85 tests passing (23 new). No open decision gates. See note below the table. | `ece7535` |
| 11 — Local single-hospital baseline (ablation row 1) | **Done.** 94/94 tests passing (8 new + 1 partitioning regression test). Real results below — no architecture concerns triggered. | `c57ed27` |
| 12 — Centralized pooled baseline (ablation row 2) | **Done.** 97/97 tests passing (3 new). Centralized model matched/exceeded every local baseline — all [OK]. See note below the table. | `f684195` |
| 13 — Flower FedAvg in simulation (ablation row 3) | **Done — real 20-round FedAvg run completed end-to-end.** 101/101 tests passing (4 new). No privacy layers yet (by design — DP/SecAgg/TLS come next). Several real Flower-tooling issues hit and fixed; see note below the table. | (pending commit this session) |
| 14 — Differential Privacy with formal accounting (ablation row 5) | **Done — DG-7 resolved and applied.** Opacus DP-SGD wired into `client_app.py` as a config-switchable layer (`dp-enabled`, default `false`; Stage 13's no-DP path untouched). 111/111 tests passing (10 new). Two real live `flwr run` DP runs at epsilon=4 and epsilon=1, confirming the expected privacy-utility tradeoff. See note below the table. | 90bf95f |
| 15 — Secure Aggregation with Flower SecAgg+ (ablation row 4) | **Done — real live `flwr run` with SecAgg+ actually masking updates.** SecAgg+ at flwr==1.35.0 only integrates via Flower's legacy Strategy/workflow API (not Stage 13/14's Message-API `strategy.start()`), confirmed against installed source + Flower's own reference example — a genuine API-surface finding, raised with and approved by the owner before implementation. Separate `client_app_secagg.py`/`server_app_secagg.py` app pair. 114/114 tests passing (3 new). See note below the table. | 20cc551 |
| 16 — TLS + client authentication (ADR-4) | **Done — real TLS + Flower node-authentication validated end-to-end, deployment engine (not simulation), using Stage 13/14's canonical app unmodified (ADR-8 confirmed).** `scripts/generate_certs.sh`, `src/federated/security.py`, `conf/federated/tls.yaml`, `tests/{test_tls_auth,test_security}.py`. 119/119 tests passing (5 new), including a real subprocess-level negative test (unregistered SuperNode rejected) and positive test (registered SuperNode accepted). See note below the table. | e078cee |
| 17 — Docker Compose multi-client deployment | **Done — real `docker compose` run, 4 separate containers over a real Docker network, all 3 hospitals training every round, clean teardown.** Scope (owner-approved 2026-08-30): FedAvg + TLS/auth only, CPU-only (DG-9). `docker/{Dockerfile.client,Dockerfile.server,docker-compose.yml}`, `scripts/{run_deployment.sh,prepare_deployment_shards.py}`, `client_app.py`'s `_resolve_config()` addition. Three real bugs found and fixed via live runs, not inspection — see note below the table. Still 119/119 (no new unit tests — infra/deployment stage). | 3ece93a |
| 18 — Grad-CAM explainability | **Done — real, non-degenerate, class-discriminative heatmaps on real trained checkpoints and real chest X-rays.** A real ADR-1/Grad-CAM interaction the plan itself flagged as needing empirical verification turned out to be a real bug (crash, not silent wrong output) on the first live test run — see note below the table. `src/explain/gradcam.py`, `scripts/generate_explanations.py`. 125/125 tests passing (6 new). | 9fcc8af |
| 19 — Monte Carlo Dropout and deferral | **Done — DG-10 resolved (fixed coverage target, defer worst 10%), verified on the real pooled test set: retained-case accuracy exceeds overall accuracy.** No implementation bugs this time (all 9 new tests passed on first run) — a clean stage after Grad-CAM's real bug. `src/uncertainty/{mc_dropout,deferral}.py`, `conf/experiment/uncertainty.yaml`. 134/134 tests passing (9 new). | (pending commit this session) |
| 20 — Overhead instrumentation | **Done — real per-round communication and compute cost, DP's and SecAgg's overhead attributed separately from real live-run comparisons.** `src/evaluation/overhead.py`, small additive instrumentation in `client_app.py`/`client_app_secagg.py`. 140/140 tests passing (6 new). See note below the table. | (pending commit this session) |
| 21 — Full ablation campaign | **Done — 27/27 real live runs succeeded, every row of the ablation table has real numbers, 3 seeds each.** Epsilon sweep shows a clean monotonic privacy-utility tradeoff; Dirichlet sweep shows the expected heterogeneity-hurts-FedAvg effect. Real bugs found and fixed along the way — see note below the table. 144/144 tests passing (3 new). | (pending commit this session) |
| 22 — Test suite hardening | **Done — audit found the existing suite already covers every CLAUDE.md §11.3 requirement except one; filled that gap and added optional CI.** `tests/test_integration_smoke.py` (new), `.github/workflows/tests.yml` (new). 145/145 tests passing (1 new). See note below the table. | (pending commit this session) |
| 23 | Not started | — |

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

**Stage 10 (`src/evaluation/{metrics,bootstrap,reporting}.py`):**
- `compute_metrics()`: AUROC (primary) + AUPRC + sensitivity/specificity/F1/balanced
  accuracy at threshold + sensitivity-at-target-specificity (swept via the ROC curve,
  default target 90%) + confusion matrix. **Threshold policy made explicit**: 0.5
  default on the positive-class probability, overridable — stated per Stage 10's own
  flagged risk that this must never be left implicit.
- Degenerate inputs handled without crashing: single-class y_true reports `NaN` for
  AUROC/AUPRC/sensitivity-at-specificity (mathematically undefined) rather than raising
  or fabricating a number; all-correct and all-wrong predictions verified against
  hand-computed expected values.
- `bootstrap_auroc_ci()`: percentile bootstrap, raises immediately and clearly on
  single-class input (AUROC undefined) rather than crashing inside sklearn.
- `aggregate_over_seeds()` / `aggregate_metrics_over_seeds()`: mean/std over seeds,
  always records `n_seeds` explicitly so a table built from fewer than the recommended
  3 seeds is visible in the data, never silently presented as final.
- **Known-input regression test**: the canonical sklearn `roc_auc_score` docstring
  example (`y_true=[0,0,1,1]`, `y_score=[0.1,0.4,0.35,0.8]` → AUROC=0.75), independent
  of this project's own code.
- MLflow logging (`log_metrics_to_mlflow`) smoke-tested end-to-end against the real
  local tracking DB, not just unit-tested in isolation — confirmed metrics actually
  land in a queryable run.
- 23 new tests (85 total), all passing.

**Real bug found and fixed before Stage 11 could produce valid results (committed
separately, `099e320`):** `natural_shard_rsna` (Stage 5) defaulted to `seed=1000`,
identical to `data_partition_seed` — the seed Stage 4's `grouped_stratified_split`
already used on this exact patient population. Both sort-then-`random.Random(seed)`
-shuffle, so the same seed reproduces the identical permutation; cutting it in half
for Hospitals B/C silently reproduced Stage 4's test/val/train cut points instead of
an independent split. **Symptom: Hospital C had zero val/test records (100% train);
Hospital B absorbed all of RSNA's val+test.** Fixed by changing
`RSNA_SHARD_SEED` to 5000 in `scripts/build_partitions.py`, regenerating both
`hospitals_natural{,_balanced}.json` (total per-hospital image counts unchanged, only
internal split composition was wrong), and adding a regression test
(`tests/test_partitioning.py::test_natural_shard_proportional_across_upstream_split_when_seed_differs`)
that exercises the real `grouped_stratified_split` + `natural_shard_rsna` together.
**General lesson recorded in `natural_shard_rsna`'s docstring**: any function using
this sort-then-seed-shuffle pattern must use a seed independent of any prior split's
seed applied to the same population — worth checking if this pattern is reused again
later in the project (e.g. Stage 13's client sampling).

**Stage 11 (`src/training/trainer.py`, `scripts/train_local.py`,
`conf/experiment/local.yaml`) — real results, 3 seeds each, on cached features:**

| Partition | Hospital | Test AUROC (mean ± std) |
|---|---|---|
| natural | A (Kermany) | 0.9849 ± 0.0005 |
| natural | B (RSNA shard) | 0.8339 ± 0.0004 |
| natural | C (RSNA shard) | 0.8584 ± 0.0007 |
| balanced | A (Kermany) | 0.9849 ± 0.0005 (unchanged — A was never subsampled) |
| balanced | B (RSNA shard) | 0.8204 ± 0.0013 |
| balanced | C (RSNA shard) | 0.8505 ± 0.0011 |

All well above chance and with tight seed variance — **no ADR-1/GroupNorm-fallback
concern triggered**, so this was not raised as an architecture question. Kermany
outperforms RSNA noticeably, plausibly reflecting DG-2's accepted clinical caveat
(RSNA's negative class mixes true-Normal with abnormal-but-not-pneumonia) plus
Kermany's cleaner pediatric single-center population being an easier separation task.
The balanced regime's RSNA hospitals score marginally lower than natural's — smaller
training set, as expected from DG-3's tradeoff.

Class imbalance handled via inverse-frequency loss weighting (not oversampling — no
data duplication needed). Checkpoints: `outputs/checkpoints/local_baseline/` (18
files, gitignored). Full results: `outputs/results/local_baseline.json` (gitignored —
MLflow, experiment `local_baseline`, is the authoritative record; the JSON is a local
convenience export). 8 new trainer tests + 1 partitioning regression test (94 total).

**Stage 12 (`src/training/trainer.py::load_pooled_features`, `scripts/train_centralized.py`,
`conf/experiment/centralized.yaml`) — real results, 3 seeds each:**

| Partition | Pooled test AUROC | Centralized model on A | on B | on C |
|---|---|---|---|---|
| natural | 0.9053 ± 0.0006 | 0.9793 (local: 0.9849) [OK] | 0.8377 (local: 0.8339) [OK] | 0.8621 (local: 0.8584) [OK] |
| balanced | 0.9290 ± 0.0010 | 0.9795 (local: 0.9849) [OK] | 0.8200 (local: 0.8204) [OK] | 0.8572 (local: 0.8505) [OK] |

**Every per-hospital comparison flagged OK** — the centralized model matched or
slightly exceeded every one of Stage 11's local baselines (within the script's 0.02
tolerance), which is exactly the sanity check Stage 12's own testing criterion
requires ("centralized should at least match local, or investigate a
label-harmonization/partitioning defect"). No investigation needed. The pooled test
AUROC (0.91–0.93) sits between Kermany's easy ~0.98 and RSNA's harder ~0.83–0.86, as
expected for a blended test set — not itself a concern.

3 new tests (97 total): `load_hospital_features`/`load_pooled_features` correctness
against synthetic fixtures mirroring the real on-disk structure (shape checks,
hospital-filtering correctness, pooling concatenation).

**Stage 13 (`src/federated/{client_app,server_app,serialization,strategy}.py`,
`conf/federated/fedavg.yaml`) — the first real federated round ever run in this
project, via `flwr run .` (Flower's actual simulation runtime, not a mocked test).**

Real 20-round FedAvg run, natural partition, 3 simulated nodes (A/B/C), 1 local epoch
per round, lr=0.001, `Adam` on the classifier only (backbone never transmitted —
ADR-1's head-only federated payload, ~1MB `ArrayRecord`). Round 0 is the random-init
model; round 1 already jumps to 0.79 pooled test AUROC from a single round.
Client-side val AUROC (federated-weighted across clients) climbs steadily to ~0.86 and
plateaus; pooled test AUROC plateaus around 0.80–0.83.

**The headline check — federated vs. local vs. centralized, same per-hospital test
sets throughout (final round's saved checkpoint, evaluated post-hoc):**

| Hospital | Local (Stage 11) | Centralized (Stage 12) | FedAvg (Stage 13) |
|---|---|---|---|
| A (Kermany) | 0.9849 | 0.9793 | 0.9482 |
| B (RSNA) | 0.8339 | 0.8377 | 0.8320 |
| C (RSNA) | 0.8584 | 0.8621 | 0.8574 |

**Honest finding, not a red flag: FedAvg does not beat local/centralized here** — it
lands close to both (within ~1-4 points), slightly below on every hospital, most
noticeably on Kermany (A), which already does very well locally on its own clean,
small dataset and has the least to gain from federating with two harder, differently-
distributed RSNA shards. This is a well-known, expected, and reportable non-IID
federation-cost finding — exactly what the ablation table (row 3 vs. rows 1/2) exists
to quantify, not a system malfunction. Only 20 rounds x 1 local epoch were run; more
rounds/epochs or FedAvg tuning were not attempted, since finding this exact number was
the goal, not chasing a specific outcome.

4 new tests (101 total): serialization round-trip losslessness (dtype/shape/values),
and single-client-round determinism/reproducibility (a real bug was found and fixed
here too — `train_local_round` originally only seeded a local `Generator` for
shuffle/view-selection, leaving Dropout's mask draw on the *global* torch RNG
unseeded, so two "identical" calls silently diverged; fixed by also calling
`torch.manual_seed(seed)`, matching how Stage 11/12's `train_classifier` already
achieves full reproducibility via `set_global_seed`).

**Real Flower-tooling issues hit and fixed while implementing this stage — exactly
the kind of API churn ADR-5 warned about, worth knowing before touching Stages 14-17:**
1. **The Message API (`@app.train()`/`@app.evaluate()`/`@app.main()`, `ArrayRecord`,
   `MetricRecord`, `RecordDict`, `Grid`, `strategy.start()`) has fully superseded the
   old `NumPyClient`/`client_fn` style** at flwr 1.35.0. Verified by generating a real
   official app skeleton with `flwr new @flwrlabs/quickstart-pytorch` against the
   pinned CLI (not by trusting web docs, which can drift from the exact pinned
   version) — this is the reliable way to check Flower's current API going forward.
2. **`flwr run` requires a persistent local "SuperLink" control-plane daemon**, even
   for pure local simulation — auto-started on first run, but it's a real background
   process (`flower-superlink`, `flower-superexec`) that outlives the `flwr run`
   command and can accumulate stale state if a run is killed mid-flight. If a future
   run hangs or errors mysteriously, check `pgrep -af flower-super`, kill it, delete
   `~/.flwr/local-superlink/state.db*`, and retry clean.
3. **Default simulated SuperNode count changed from 10 to 2 as of flwr>=1.32** — this
   project needs 3 (one per hospital). Must pass
   `--federation-config "num-supernodes=3"` explicitly every run, or configure it
   permanently via `flwr federation simulation-config`.
4. **The FAB (Flower App Bundle) has a hard 10MB size limit**, and its default
   include patterns (`**/*.py`, `**/*.json`, `**/*.yaml`, etc.) matched far more than
   intended: `.venv`'s installed packages (~350MB of matching files) and
   `data/*.json` (multi-MB partition files) both got swept in. Fixed with a positive
   `fab-include = ["src/**/*.py", "pyproject.toml"]` allowlist in `[tool.flwr.app]`
   rather than trying to exclude everything unwanted.
5. **That `fab-include` key must be a direct child of `[tool.flwr.app]`** — placing
   it after `[tool.flwr.app.components]`'s keys silently nested it under
   `.components` instead (TOML tables extend to the next `[header]`), and Flower
   simply ignored the unrecognized key with no error. If a config key seems to have
   no effect, check with `flwr.cli.config_utils.load_and_validate` directly rather
   than assuming placement was correct.
6. **`[project].name` doubles as the Flower App name**, which has a hard 32-character
   limit with no override — `privacy-preserving-medical-diagnosis` (37 chars) had to
   be shortened to `pneumonia-fl` (cosmetic only; nothing imports the package by this
   name).
7. **`fab-format-version = 1` requires a declared `[project].license` file** —
   omitted rather than inventing a license unilaterally (a real decision, not a
   build-tool formality); version defaults to `0` without it, which works fine
   locally.
8. **Flower's simulation runtime copies the app's source into an isolated directory**
   (`~/.flwr/apps/<app-id>/`) and executes it from there — any relative path, or any
   `Path(__file__).resolve().parents[N]`-style "repo root" computation (the pattern
   every other script in this project uses), silently resolves against the wrong
   directory. Fixed by passing `partition-path`, `feature-cache-dir`, and
   `output-checkpoint` as **absolute paths** via `[tool.flwr.app.config]`, explicitly
   threaded through `context.run_config` into `client_app.py`/`server_app.py` rather
   than relying on any default. This is specific to `flwr run`'s simulation
   execution — Stage 17's deployment engine is a different execution mode and will
   need its own path handling.

**Stage 14 (`src/privacy/{dp,accounting}.py`, `src/federated/client_app.py`,
`src/federated/server_app.py`, `pyproject.toml`'s `[tool.flwr.app.config]`
DP keys) — sample-level DP-SGD (ADR-2), config-switchable, verified with two real
live `flwr run` executions, not just unit tests.**

**DG-7 resolved 2026-08-29 (owner-approved):** delta = 1e-5 (well below 1/N for
every hospital — the smallest, Hospital A, has N=4,180 train, so 1/N ≈ 2.4e-4 >>
1e-5); target-epsilon sweep = {1, 2, 4, 8}, with 4 (the sweep's midpoint) set as
`pyproject.toml`'s config default. Recorded in `pyproject.toml`'s `[tool.flwr.app.config]`
comment block and in CLAUDE.md.

**Design:** DP is a config-switchable layer, not a fork of Stage 13's client logic.
`dp-enabled` (default `false`) selects between Stage 13's original path
(`train_local_round` — plain Adam, cycles through K=5 augmented + 1 eval view) and
the new DP path (`train_local_round_dp` — Opacus `PrivacyEngine.make_private()`,
trained on the deterministic eval-style view only, since Opacus's Poisson-sampling
DataLoader — required for its accounting to be valid — doesn't compose simply with
"a different random augmented view per epoch"). A `PrivacyEngine` instance is cached
per hospital (`_privacy_engine_cache` in `client_app.py`) and reused across every
round that hospital participates in — required for the accountant's spent-epsilon to
accumulate correctly rather than resetting each round (the stage's own flagged risk,
directly tested — see below). Per-client noise multiplier is calibrated once from
that client's own dataset size via `src/privacy/accounting.py::compute_noise_multiplier`
(Opacus's `get_noise_multiplier`, RDP accountant, given target epsilon/delta, sample
rate, and total training steps across all rounds).

**Real bugs found and fixed while implementing this stage (not found by inspection —
found via failing tests or a live run):**
1. **Accountant-type mismatch**: `compute_noise_multiplier()` explicitly used
   `accountant="rdp"`, but a bare `PrivacyEngine()` call defaults to `"prv"` — noise
   would have been calibrated for one accounting method while epsilon was reported
   under another, internally inconsistent. Fixed by adding `make_privacy_engine()` in
   `src/privacy/dp.py`, used everywhere a `PrivacyEngine` is constructed, explicitly
   pinning `accountant="rdp"`.
2. **`OverflowError` on near-zero-noise epsilon queries**: `privacy_engine.get_epsilon()`
   is mathematically infinite as `noise_multiplier -> 0` (no noise = no privacy), and
   Opacus's accountant can overflow computing that limit rather than returning it.
   Fixed with `try/except OverflowError: epsilon_spent = float("inf")` in
   `train_local_round_dp`.
3. **Opacus refuses to re-attach hooks to an already-wrapped model.** This isn't a bug
   to work around — it's a correct guard that validates the design already in place:
   `client_app.py`'s `@app.train()` always constructs a fresh `DenseNet121Head()` each
   round and loads the previous round's returned `classifier_state` into it, so the
   same model object is never re-wrapped. The bug was in the *tests*, which initially
   reused one model object across simulated rounds; fixed by matching the real
   fresh-model-per-round pattern.
4. **Methodological lesson in the tests themselves**: the first version of
   `tests/test_dp.py` tried to verify "clipping bounds the update" and "noise scales
   with the multiplier" by comparing final Adam-optimized parameter positions across
   configurations — Adam's adaptive per-parameter normalization actively confounds
   raw-gradient-magnitude comparisons (produced a wrong-direction assertion failure).
   Rewritten to directly inspect Opacus's internal mechanism instead: call
   `dp_opt.clip_and_accumulate()` by hand, inspect `param.grad_sample` (pre-clip
   per-sample norms) and `param.summed_grad` (post-clip — must satisfy
   `summed_norm <= max_grad_norm * batch_size` by the triangle inequality), then
   `dp_opt.add_noise()` and inspect `param.grad` to isolate the injected noise
   magnitude. Worth remembering for any future test of a clipping/noising mechanism —
   test the mechanism directly, not through an optimizer that will confound it.

**Real live validation — two full `flwr run` executions (5 rounds each, natural
partition, 3 nodes), not mocks, comparing directly against Stage 13's no-DP round-5
result (`pooled_test_auroc=0.8182`):**

| Run | target-epsilon | Round 5 pooled_test_auroc | Round 5 client_val_auroc | epsilon_spent trend (aggregated across hospitals) |
|---|---|---|---|---|
| Stage 13 (no DP) | — | 0.8182 | — | — |
| Stage 14, looser budget | 4.0 | 0.8152 | 0.8185 | 3.20 → 3.43 → 3.65 (rounds 3–5), noise_multiplier=0.6365 |
| Stage 14, tighter budget | 1.0 | 0.8022 | 0.8032 | 0.835 → 0.848 → 0.849 → 0.888 → 0.873 (rounds 1–5), noise_multiplier=1.0544 |

**Confirms Stage 14's own testing criterion — decreasing epsilon monotonically
decreases accuracy — with real measured numbers, not an assumption:**
epsilon=4 costs ~0.3 AUROC points vs. no-DP (0.8182 → 0.8152); epsilon=1 costs ~1.6
points (0.8182 → 0.8022). Both are honest, moderate utility costs at these budgets —
neither run collapsed to chance, and the accountant visibly accumulates budget every
round (never resets), which is exactly what Stage 14 needed to demonstrate to be
trustworthy. Per-round `epsilon_spent` here is a weighted mean across the 3 hospitals'
individually-tracked accountants (each has its own `PrivacyEngine`, since each
hospital has a different dataset size and therefore a different noise multiplier for
the same target epsilon) — the slight non-monotonic dip at epsilon=1's round 5 (0.888
→ 0.873) reflects that aggregation, not the underlying per-hospital accountants, each
of which is unit-tested to increase strictly monotonically
(`test_accountant_consumes_budget_monotonically_across_rounds`).

10 new tests (111 total): `tests/test_accounting.py` (4 — `compute_total_steps`
arithmetic, `compute_noise_multiplier` increases for tighter epsilon and for more
steps) and `tests/test_dp.py` (6 — clipping bounds the summed gradient, looser clip
norm permits a larger summed gradient, noise scales with the multiplier, the
accountant consumes budget monotonically across rounds using the same `PrivacyEngine`,
zero-noise DP-SGD still sensibly reduces loss on separable synthetic data, and the
returned classifier state loads cleanly into a plain unwrapped classifier with no
leftover Opacus `_module.` prefix).

**Stage 15 (`src/federated/{client_app_secagg,server_app_secagg}.py`,
`tests/test_secagg.py`, `pyproject.toml`'s Stage 15 config keys) — Secure
Aggregation via Flower SecAgg+ (ADR-3), ablation row 4.**

**Real, material API-surface finding, verified against flwr==1.35.0's actual
installed source (not memory/docs — ADR-5) and raised with the owner before
writing code, per CLAUDE.md's "STOP and ask" rule for implementation-approach
conflicts:** SecAgg+ is only wired through Flower's *legacy* server pipeline
at this pinned version — `SecAggPlusWorkflow` requires a `LegacyContext`, an
old-style `flwr.server.strategy.Strategy` (not the new
`flwr.serverapp.strategy.FedAvg` Stage 13/14 use), and `DefaultWorkflow`,
called directly as `workflow(grid, legacy_context)`. It has **not** been
ported to the new Message-API `strategy.start(grid=...)` loop — confirmed via
source inspection (`flwr.server.compat`, `flwr.server.workflow`) and
cross-checked against Flower's own official
`examples/flower-secure-aggregation` reference app (fetched from
`flwrlabs/flower` on GitHub via `gh api`), which uses exactly this
`LegacyContext`/`DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(...))`
pattern from inside an `@app.main()` function. Flower's own "Upgrade to
Message API" doc doesn't mention SecAgg+ at all, confirming no documented
migration path exists yet. **Owner approved proceeding with the legacy-API
path** (the only ADR-3-compliant option — no custom crypto) via an explicit
choice, 2026-08-30.

**A second, deeper finding surfaced during implementation** (not covered in
the initial owner check-in, since it only became clear once building the
client side): `secaggplus_mod`'s handshake is driven through Flower's
`flwr.compat` glue, which packs/unpacks messages under `fitins.parameters` /
`fitres.parameters` RecordDict keys — a different wire format from Stage
13/14's Message-API convention (`msg.content["arrays"]`,
`RecordDict({"arrays": ..., "metrics": ...})`). Since a `ClientApp` can only
be built with `client_fn` (legacy) *or* the new `@app.train()`/`@app.evaluate()`
decorators — never both (`ClientApp.__init__`'s `_call` short-circuits
`_registered_funcs` entirely when `client_fn` is set) — SecAgg+ cannot be a
runtime config flag inside the existing `client_app.py`; it structurally
requires its own `ClientApp` built the legacy `NumPyClient` + `client_fn` +
`mods=[secaggplus_mod]` way, matching the official example exactly. This is
why Stage 15 ships as a **fully separate app pair**
(`client_app_secagg.py`/`server_app_secagg.py`), not a branch in
`client_app.py`/`server_app.py` the way Stage 14's `dp-enabled` flag was.

**Operational consequence — how to actually run this stage**: `pyproject.toml`'s
`[tool.flwr.app.components]` points at Stage 13/14's canonical
`server_app.py`/`client_app.py` by default (covers ablation rows 1/2/3/5/6).
To run row 4 (this stage), temporarily edit `[tool.flwr.app.components]` to:
```
serverapp = "src.federated.server_app_secagg:app"
clientapp = "src.federated.client_app_secagg:app"
```
run `flwr run`, then **revert it back** before committing — verified via
`git diff pyproject.toml` after each validation run this session. A future
session picking this project back up must know this swap-and-revert pattern
exists; it is not a one-off, it is how ablation row 4 is run every time.

**Real bug found via the live validation run, not by inspection**:
`SecAggPlusWorkflow`'s own weight-encoding math (`ratio = num_examples /
max_weight`) silently triggered its "potential overflow" warning every round
— its default `max_weight=1000.0` is far below this project's real
per-hospital train counts (Hospital B/C's natural shards are ~13,342 each).
Fixed by adding `max-weight = 20000` to `pyproject.toml`'s Stage 15 config
block and threading it into `SecAggPlusWorkflow(..., max_weight=...)`
explicitly rather than relying on the library default. Confirmed the warning
disappeared and results stayed sensible on re-run.

**Real live validation — 3-round `flwr run` with SecAgg+ actually masking
every client's update (not mocked)**, natural partition, 3 nodes, after the
`max_weight` fix:

| Round | pooled_test_auroc (SecAgg+) |
|---|---|
| 0 (init) | 0.2919 |
| 1 | 0.8048 |
| 2 | 0.8089 |
| 3 | 0.8126 |

Tracks Stage 13's plain-FedAvg trajectory closely (round 1 ~0.79-0.80 in both;
Stage 13 by round 3 is in the same ~0.80-0.81 band) — exactly what ablation
row 4 vs. row 3 should show: SecAgg+'s only expected cost is quantization
noise, not a change in what's being learned, and that's what these numbers
demonstrate.

**"Masks cancel exactly" (CLAUDE.md §11.3's own named test) — 3 new tests,
`tests/test_secagg.py` (114 total)**: replicates the *exact* client-side
quantize/weight-encode (`secaggplus_mod`) and server-side
unmask/dequantize (`secaggplus_workflow.unmask_stage`) math using Flower's own
functions (`quantize`, `dequantize`, `factor_combine`, `factor_extract`,
`parameters_addition`, `parameters_mod`), for synthetic multi-client scenarios
with realistic (and deliberately skewed) per-hospital weights — but
deliberately skips generating/exchanging the actual pairwise/private masks,
since their exact cancellation is Flower's own tested cryptographic guarantee
(Shamir secret sharing + ECDH), not something to re-derive per ADR-3. What's
tested is the part specific to this project's usage: that the
quantize-encode-sum-unmask-decode round-trip reproduces the same weighted
average plain FedAvg would compute, with only quantization-level error, not
some other integration bug. A third test isolates `quantize`/`dequantize`
alone to confirm the round-trip error is real (lossy, non-vacuous) but
correctly bounded.

**Stage 16 (`scripts/generate_certs.sh`, `src/federated/security.py`,
`conf/federated/tls.yaml`, `tests/{test_tls_auth,test_security}.py`) — TLS +
client authentication (ADR-4), the project's first stage to leave pure
simulation.**

**Mechanism verified against flwr==1.35.0's actual installed CLI (`flower-superlink
--help`, `flower-supernode --help`, `flwr federation/supernode --help`), not
memory or older docs — ADR-4 itself flags this area as changed across
releases, ADR-5 requires verifying against the pinned version:**
- Server TLS: `--ssl-ca-certfile`/`--ssl-certfile`/`--ssl-keyfile` on
  `flower-superlink`.
- Client (SuperNode) authentication: `--enable-supernode-auth` on the
  SuperLink (hard-requires TLS — refuses `--insecure` outright) +
  `--auth-supernode-private-key` on each SuperNode + the public half
  pre-registered via `flwr supernode register`. **The old
  `--auth-list-public-keys` flag this project's earlier notes might assume
  is gone entirely at this version — it's a hard error, not just
  deprecated**, with the CLI itself pointing at `--enable-supernode-auth` +
  `flwr supernode register` as the replacement.
- Keys are ECDSA-384, OpenSSH format (`ssh-keygen -t ecdsa -b 384`), verified
  against Flower's own tested `framework/e2e/e2e-bare-auth/generate.sh`
  reference (fetched from `flwrlabs/flower` on GitHub via `gh api`) — not
  Ed25519, which is a *different*, unrelated Flower mechanism
  (`--trusted-entities`).
- Dynamic `RegisterNode` is disabled server-side once
  `--enable-supernode-auth` is set — only pre-registered public keys can
  connect at all, which is exactly ADR-4's claim.

**A real, non-obvious operational finding, found via this stage's own live
validation, not documentation:** `flwr supernode register`/`flwr run <app>
<name>` resolve their target through a **named SuperLink connection** stored
in the user-level `~/.flwr/config.toml` (`[superlink.<name>]`, the same
registry holding the built-in `local` simulation alias and Flower's cloud
`supergrid` alias) — **not** a project-level `[tool.flwr.federations.<name>]`
block in `pyproject.toml`, even though that block is Flower's own documented
mechanism elsewhere (confirmed via the same `e2e-bare-auth`/`test_superlink.sh`
reference, which *does* inject a `[tool.flwr.federations.e2e]` block). Defining
both under the same name simultaneously (as first attempted here) triggers a
"legacy TOML migration" code path that fails outright with a confusing error
("No federation name was provided..."). There is also no OSS CLI command to
*create* a new named connection against a fresh local address (`flwr login`/
`flwr federation create` both require the connection to already exist in
`~/.flwr/config.toml` — likely gated behind Flower Enterprise's
`FederationManager`, given the OSS build's `NoOpFederationManager` and the
`flwr.ee`-guarded imports throughout `flwr.superlink.config_loader`). The
practical fix: **the `pyproject.toml` federations block was removed**, and
instead `~/.flwr/config.toml` gets a `[superlink.<name>]` entry written
directly (it's a plain, user-editable TOML file, not a protected format) —
this is exactly what `tests/test_tls_auth.py`'s fixture does automatically
via an isolated `FLWR_HOME`, and what a human operator running this stage
manually must also do (see the exact entry format in the test fixture or in
this stage's commit).

**Real, live end-to-end validation (not mocked) — deployment engine, not
simulation, using Stage 13/14's canonical `client_app.py`/`server_app.py`
completely unmodified:**
1. `scripts/generate_certs.sh` generated a local CA, SuperLink server cert,
   and 3 hospital ECDSA-384 keypairs.
2. `flower-superlink` started with `--enable-supernode-auth` + TLS.
3. **Negative case confirmed manually first**: an unregistered hospital_A key
   attempting to connect got `FAILED_PRECONDITION` / "Failed to activate
   SuperNode" — proving auth isn't decorative.
4. All 3 hospitals registered via `flwr supernode register`; all 3
   `flower-supernode` processes then connected successfully.
5. `flwr run . deployment --run-config "num-server-rounds=2"` completed 2 real
   rounds over real TLS-encrypted, authenticated gRPC connections between 4
   separate OS processes (not simulated actors): round 2
   `pooled_test_auroc=0.8081`, `client_val_auroc=0.8350` — consistent with
   the same early-round trajectory seen in every simulation-mode run this
   session (Stages 13-15), **confirming ADR-8's claim that the same
   ClientApp/ServerApp code runs correctly under both execution engines**,
   validated for the first time this session.

**5 new tests (119 total)**: `tests/test_tls_auth.py` (3) — a real
subprocess-orchestrated integration suite (not mocked), isolated via the
`FLWR_HOME` env var (which `flwr.supercore.utils.get_flwr_home` itself
supports) so it never touches the developer's real `~/.flwr/` state:
`test_supernode_auth_requires_tls` (the encryption-coupling claim: `--insecure`
+ `--enable-supernode-auth` together must be rejected), the negative test
CLAUDE.md names explicitly (`test_unregistered_supernode_is_rejected`), and
its positive counterpart (`test_registered_supernode_is_accepted`, without
which the negative test alone can't distinguish "auth works" from "the
SuperLink rejects everyone"). A real bug in the tests' own process-teardown
logic was found and fixed along the way: `flower-supernode` doesn't exit
cleanly on rejection (its "SuperExec" sidecar keeps the stdout pipe open),
which hung a plain `subprocess.run(..., timeout=...)`/`Popen.communicate()`
indefinitely — fixed by using `start_new_session=True` and killing the whole
process group (`os.killpg`), not just the parent process.
`tests/test_security.py` (2) — `src/federated/security.py` verifies Stage
16's own testing criterion ("confirmation that the configured message length
exceeds the actual update size") directly: the real head-only classifier
payload (~1MB, matching Stage 13's own live-logged `ArrayRecord` sizes) sits
comfortably under `flwr.common.GRPC_MAX_MESSAGE_LENGTH`.

**Real, stale-assumption finding, corrected rather than silently worked
around**: ADR-4's text (and this project's earlier planning notes) assume
Flower's classic 4MB default gRPC message-size ceiling, "requiring explicit
override." At the pinned flwr==1.35.0, **no CLI flag to override gRPC
message size exists any more** — `flwr.common.GRPC_MAX_MESSAGE_LENGTH` is a
hardcoded ~2GB constant with no exposed knob. This doesn't block anything
(2GB vastly exceeds the ~1MB real payload), but the premise "must be
explicitly configured" is stale for this pinned version — documented in
`security.py`'s docstring and `conf/federated/tls.yaml` rather than silently
claiming something was "configured" that has no CLI surface any more.

**Stage 17 (`docker/{Dockerfile.client,Dockerfile.server,docker-compose.yml}`,
`scripts/{run_deployment.sh,prepare_deployment_shards.py}`,
`src/federated/client_app.py`'s `_resolve_config()`) — Docker Compose
multi-client deployment, the single most convincing demonstration artifact:
real separate containers, real gRPC, real TLS, real client authentication,
real per-hospital data isolation.**

**Scope decisions, both owner-approved 2026-08-30**: (1) DG-9 resolved as
CPU-only, few rounds — this is a demonstration, not a measurement (real
numbers come from simulation, Stages 11-15); (2) FedAvg + TLS/auth only, using
Stage 13/14/16's canonical app completely unmodified — SecAgg+'s separate
legacy-API app pair (Stage 15) is not part of this demonstration; combining
every layer at once (ablation row 6) is deferred as its own later integration
step, since it requires reconciling two different Flower API surfaces first,
which is real integration work, not Docker packaging.

**Real per-hospital filesystem isolation (not just code-level patient_id
filtering)**: `scripts/prepare_deployment_shards.py` pre-slices the shared
feature cache (which stores Hospital B and C as patient-disjoint shards of the
*same* `rsna_*.pt` files) into one directory per hospital containing ONLY that
hospital's records — real counts: Hospital A 4180/841/835 train/val/test
(Kermany, unchanged since it's already 100% of that source), Hospital B
9330/2034/1978, Hospital C 9348/1969/2025 (both RSNA shards, confirmed disjoint
and summing to the known ~13,342 per hospital across splits). Docker Compose
bind-mounts each hospital's directory, read-only, into only that hospital's own
container — no other hospital's data exists at any path inside that container,
not merely code that promises not to read it. Enabled by one small, additive
change to `client_app.py`: `_resolve_config()` checks `context.node_config`
before falling back to `context.run_config` for `feature-cache-dir` and
`partition-path` — simulation mode (Stages 13-16) never sets these at the node
level, so this is a pure no-op there; Docker Compose gives each hospital
container its own `--node-config "... feature-cache-dir=/data/hospital
partition-path=/data/hospital/partition.json"`. No change to the actual
training/aggregation logic itself.

**Three real bugs found and fixed via live runs — none by inspection, all by
watching an actual attempt fail and diagnosing from evidence:**

1. **CUDA dependency bloat in the Docker build.** `uv sync --frozen
   --no-install-package torch --no-install-package torchvision` still pulled
   in ~2GB of unused `nvidia-*-cu12`/`triton` packages — excluding a package by
   name does not prune its transitive dependencies from a frozen lockfile sync
   when the lockfile's own resolution is CUDA-oriented (this project's
   `uv.lock` targets `torch==2.13.0+cu126` for the host dev GPU). Fixed by
   abandoning `uv sync` for the Docker build entirely and using direct `uv pip
   install --system` calls at the exact same pinned versions as
   `pyproject.toml` (CPU torch/torchvision from a separate index, everything
   else from PyPI) — bypasses the lockfile's CUDA-oriented resolution for the
   Docker build context only; the committed `pyproject.toml`/`uv.lock` and host
   dev environment are completely unchanged. Also switched to plain
   `flwr==1.35.0` instead of `flwr[simulation]==1.35.0` in the images, since
   the deployment engine doesn't need the simulation backend's `ray`
   dependency. Final image size: 2.4GB per container (was headed toward 5GB+
   before the fix). **Worth remembering for any future CPU-variant Docker
   image built from a CUDA-oriented lockfile: `--no-install-package` is not
   suffient on its own.**

2. **TLS SAN mismatch for the Docker Compose service name.** The first live
   deployment run got stuck indefinitely — all 3 hospital SuperNode containers
   showed repeated "Connection attempt failed, retrying" with zero nodes ever
   connecting, and no clear error surfaced in the logs. Raw TCP connectivity
   between containers tested fine (`socket.create_connection` succeeded),
   isolating the failure to the TLS layer. Root cause: the server
   certificate's SAN list (`scripts/generate_certs.sh`'s `certificate.conf`)
   only covered `localhost`/`127.0.0.1`/`::1`/`0.0.0.0` — correct for Stage
   16's host-only validation, but hospital containers reach the SuperLink via
   the Docker Compose service name `superlink`, which wasn't in the SAN list,
   so TLS hostname verification silently failed. Fixed by adding `DNS.2 =
   superlink` to the SAN list. **Worth remembering: TLS hostname-verification
   failures in gRPC can present as silent, uninformative retry loops rather
   than a clear certificate error — when a container-to-container TLS
   connection won't come up, check the server cert's SAN against the actual
   hostname being dialed before looking anywhere else.**

3. **Registration-before-connection race condition.** With the SAN fix alone,
   the second live run got 2 of 3 hospitals connected but the third
   (hospital-c) got permanently stuck. Its logs showed `FAILED_PRECONDITION` /
   "Failed to activate SuperNode" followed by a Python traceback and
   `RuntimeError: generator didn't yield` — then the container sat "Up"
   forever, its SuperExec sidecar still polling but its actual Fleet
   connection dead. Root cause: `docker compose up -d` for all 4 services
   starts every hospital SuperNode immediately, but `flwr supernode register`
   for each hospital runs afterward, sequentially, in the script — so a
   SuperNode's very first activation attempt can land in the window before its
   own key is registered. Unlike a plain connection failure (which Flower does
   retry, as seen with the other two hospitals eventually succeeding), a hard
   `FAILED_PRECONDITION` activation rejection crashes that connection attempt
   permanently with no further retry from inside that process. Fixed by
   restructuring `scripts/run_deployment.sh`: start only the `superlink`
   container first, wait for its Control API, register all 3 hospitals, *then*
   start `hospital-a`/`hospital-b`/`hospital-c` — eliminating the race
   entirely rather than trying to out-time it with delays. **Worth
   remembering: Flower's SuperNode auth doesn't tolerate "start everything at
   once and let it sort itself out" the way an unauthenticated setup would —
   registration must complete before the first connection attempt, every
   time, for every node.**

**Real live validation — full 3-round `flwr run . deployment` over the actual
Docker Compose network** (superlink + hospital-a + hospital-b + hospital-c,
4 separate containers, real TLS, real authenticated gRPC, real per-hospital
data isolation, using Stage 13/14/16's canonical app completely unmodified):
all 3 hospitals sampled and contributed every round, 0 failures.

| Round | pooled_test_auroc | client_val_auroc |
|---|---|---|
| 0 (init) | 0.2919 | — |
| 1 | 0.8048 | 0.8174 |
| 2 | 0.8081 | 0.8350 |
| 3 | 0.8148 | 0.8428 |

Matches the same early-round trajectory seen in every other run this session
(Stages 13-16) — **this is the strongest confirmation yet of ADR-8's "same
code, both engines" claim**: the identical `client_app.py`/`server_app.py`
that produced Stage 13's 20-round simulation result and Stage 16's host-process
deployment result now also works, unmodified, across 4 genuinely separate
containers on a real Docker network with real TLS and real authentication.
Clean teardown confirmed both in-script (the `trap cleanup EXIT` output) and
independently (`docker ps -a` showed nothing running afterward).

**Phase 0 (Stages 0–2), all of Phase 1 (Stages 3–5), all of Phase 2 (Stages 6–12),
and all of Phase 3 (Stages 13–17) are now complete — the entire FL + DP + SecAgg +
TLS/auth + Docker deployment core (rows 1-6 of the ablation ladder are all
individually implementable, and the demonstration artifact is real and working).**

**Stage 18 (`src/explain/gradcam.py`, `scripts/generate_explanations.py`,
`tests/test_gradcam.py`) — Grad-CAM explainability (CLAUDE.md §9, objective 6's
explanation half). The first Phase 4 stage, and the first genuinely lower-risk
stage this session (no Flower protocol surface, no distributed timing) — but
still surfaced a real bug on its own first live test run, exactly per the
plan's own explicit warning that the ADR-1/Grad-CAM interaction "must be
verified, not assumed."**

**Real bug found and fixed, empirically, not by inspection:** `pytorch-grad-cam`'s
plain `GradCAM` class does not mark its input tensor `requires_grad=True` on
its own — that only happens when its (unexposed-by-`GradCAM`) `compute_input_gradient`
flag is `True`, which it isn't by default. For an ordinarily fully-trainable
model this is invisible, since *some* parameter upstream already requires
grad, so PyTorch builds the graph anyway. With ADR-1's entirely-frozen
backbone (`requires_grad=False` on every backbone parameter) **and** an input
tensor that also doesn't require grad, *nothing* in the forward pass needed a
gradient at all — PyTorch never built an autograd graph, the target layer's
forward hook fired fine (real activations captured), but its backward hook
never fired (`grads=None`), producing a hard `AttributeError` crash the
moment the very first test ran. Fixed in `src/explain/gradcam.py` by
explicitly calling `image_tensor.clone().requires_grad_(True)` before handing
the tensor to the library — load-bearing for this project's architecture,
harmless/unnecessary-but-safe for an ordinary model.

**A second, softer finding from the same verification process**: querying
Grad-CAM for a class the model is *confidently rejecting* (e.g. asking "what
looks Normal about this?" on an image the model is >99.99% sure is
Pneumonia) can legitimately produce an all-zero heatmap after Grad-CAM's own
final ReLU — there's no localized positive evidence for a class the model
has no support for anywhere in the image. This isn't a bug; it's correct
Grad-CAM behavior. The test suite was written to distinguish this correctly:
"overlays render for both classes" is a functional (shape/dtype/range)
claim, not a non-degeneracy claim for every class — the actual non-degeneracy
claim is checked specifically against the model's own *predicted* class,
where it has real evidence.

**Target layer**: `model.features.norm5` (DenseNet121's final backbone
BatchNorm2d, pre-ReLU/pre-pool) — matches CLAUDE.md §9's already-approved
text exactly, no new decision needed.

**Real batch generation, not a toy example**: `scripts/generate_explanations.py`
evaluated Stage 12's trained centralized checkpoint (`natural_seed42.pt`) on
the full real pooled test set (4,838 images across all 3 hospitals) at the
Stage 10 default 0.5 threshold — TP=1158, FP=476, TN=2860, FN=344 (implied
accuracy ≈83.1%, sensitivity ≈77.1%, specificity ≈85.7%, consistent with
Stage 12's own reported pooled AUROC of ~0.90-0.93). A real, if minor, bucket-composition
bug was found and fixed here too: records are collected Hospital-A-first, and
since Kermany's own classifier is highly accurate, its small TN/FP/FN buckets
filled up the `[:EXAMPLES_PER_BUCKET]` slice before any RSNA record was ever
reached — the first run's saved output was accidentally 100% Kermany. Fixed
with a fixed-seed shuffle before bucketing, confirmed via the second run's
actual output files spanning both sources. Visually inspected a handful of
real overlays (not just the automated border-energy proxy test) — a real
pneumonia-positive case's heatmap concentrated over the correct (opacity-side)
lung field, not the image border or the "R"/"L" laterality marker; a real
false-negative case's heatmap (targeting its wrongly-predicted "Normal"
class) concentrated over central mediastinal structures, a plausible
"looks unremarkable" justification for the model's mistake.

6 new tests (125 total), `tests/test_gradcam.py`: target-layer resolution,
gradient-flow-through-the-frozen-backbone (using a real trained checkpoint
and the model's own predicted class — an earlier version of this test used
an untrained random-init model and was flaky, since an untrained classifier
head's Grad-CAM sign pattern is essentially a coin flip, not a reliable
signal of whether gradients are flowing correctly), functional both-classes
rendering, non-degeneracy for the predicted class specifically,
class-discriminativeness (the two per-class heatmaps for one image must
differ), and the border-energy proxy for "not latching onto scanner
artifacts/text" (CLAUDE.md's own named chest-X-ray failure mode to check
for).

**Stage 19 (`src/uncertainty/{mc_dropout,deferral}.py`,
`conf/experiment/uncertainty.yaml`) — Monte Carlo Dropout and deferral
(CLAUDE.md §10, objective 6's confidence half). The rare clean stage this
session: all 9 new tests passed on their first real run, no bugs found —
notable specifically because Grad-CAM (Stage 18, also flagged as needing
empirical verification) did surface a real bug on its first run, so this
wasn't for lack of trying to break it.**

**DG-10 resolved, owner-approved 2026-08-30 (asked explicitly, not
defaulted silently — the plan's own framing: "a clinical policy choice
rather than a tuning parameter"):** fixed coverage target — defer the
highest-predictive-entropy 10% of predictions to human clinician review
(`src/uncertainty/deferral.py::DEFAULT_TARGET_DEFER_FRACTION = 0.10`); the
actual entropy cutoff is *derived* from each run's real uncertainty
distribution rather than a hand-picked number on an otherwise-uninterpretable
scale. Uncertainty metric: predictive entropy (owner-approved, over
predictive variance) of the mean predictive distribution across T=20
stochastic forward passes.

**Mechanism relies entirely on `DenseNet121Head.train()`'s existing Stage 8
override** (`self.features.eval()` forced permanently, everything else
follows normal `.train()`/`.eval()` semantics) — calling `model.train()` for
MC Dropout activates the classifier's `Dropout` while the backbone's
BatchNorm stays frozen, with zero new code needed to keep that invariant.
Verified directly, not assumed: a real test confirms the backbone's
`running_mean` is bit-for-bit unchanged after MC Dropout's repeated forward
passes.

**Real validation, Stage 12's trained centralized checkpoint (`natural_seed42.pt`)
against the full real pooled test set (4,838 images) — all of Stage 19's own
named testing criteria checked with real numbers, not asserted:**

| Metric | Value |
|---|---|
| Overall accuracy (all 4,838 predictions) | 83.17% |
| Mean predictive entropy, correctly classified | 0.371 |
| Mean predictive entropy, misclassified | 0.613 |
| Deferral threshold (entropy, derived from 10% target) | 0.688 |
| Realized coverage | 90.0% (484/4,838 deferred) |
| Retained-case accuracy (the 90% not deferred) | **86.27%** |
| Deferred-set accuracy (the 10% sent to review) | 55.37% |

**The core "does this human-in-the-loop mechanism actually help" claim is
answered with a real number, not assumed**: retained accuracy (86.3%) exceeds
overall accuracy (83.2%) by ~3 points after deferring only the worst 10% —
and the deferred set's own accuracy (55.4%, barely above chance) confirms the
mechanism genuinely isolates the hardest, most ambiguous cases rather than
deferring arbitrarily. This is a meaningful result *despite* CLAUDE.md's own
honest framing that head-only MC Dropout is a known-weak, often-poorly-
calibrated estimator — the separation between correct/misclassified mean
entropy (0.371 vs. 0.613, a ~65% relative gap) and the accuracy-lift number
both show it carries real signal even in this documented-weaker form.

**9 new tests (134 total)**: `tests/test_mc_dropout.py` (4) — dropout
genuinely active (T passes differ, the stage's own named "most common bug in
this area"), backbone BatchNorm provably stays frozen during MC Dropout,
predictive entropy is correctly bounded (zero for a certain distribution,
>ln(2)/2 for maximal 2-class uncertainty), and the real
misclassified-vs-correct entropy separation above. `tests/test_deferral.py`
(5) — invalid-fraction rejection, realized deferral rate matches the target
fraction on synthetic data, deferral rate demonstrably responds to the
threshold (Stage 19's own named criterion), a zero-fraction edge case defers
nothing, and the real retained-vs-overall accuracy claim above.

**Stage 20 (`src/evaluation/overhead.py`, plus small additive instrumentation
in `src/federated/client_app.py` and `client_app_secagg.py`) — overhead
instrumentation (CLAUDE.md §11.2): communication and compute cost as a
first-class measured output, not an afterthought. The first Phase 5 stage —
the mode of work is starting to shift from "prove each mechanism works" to
"produce the paper's actual numbers."**

**Design**: `classifier_payload_size_bytes()` measures the real serialized
size of a classifier-head update (the same `torch.save` mechanism
`ArrayRecord` uses under the hood); `theoretical_payload_size_bytes()`
computes the raw parameter-count lower bound for cross-checking (Stage 20's
own testing criterion). `measure_wall_clock()`/`measure_peak_memory()` are
small context managers. Wired into `client_app.py`'s `train()` and
`client_app_secagg.py`'s `fit()` by wrapping exactly the local-training call
(not feature loading, which is cached/amortized across rounds) — both now
report `wall_clock_seconds` and `payload_bytes` in their per-round metrics,
which Flower's own built-in aggregated-metrics logging already surfaces with
zero further server-side changes needed.

**Real live-run comparison — three real 5-round runs (natural partition, 3
nodes), not estimates — this is what makes "DP's and SecAgg's overhead
attributed separately" (CLAUDE.md's own explicit requirement) a real,
checkable claim rather than a plan-only line item:**

| Configuration | Client-side compute (median, wall_clock_seconds/round) | Payload bytes/round | Total round-trip time (5 rounds) |
|---|---|---|---|
| Baseline (no DP, no SecAgg) | ~0.45s | 1,053,900 (constant) | 18.03s (~3.61s/round) |
| + Differential Privacy (ε=4) | ~7.50s (**~16.7x**) | 1,053,900 (**unchanged**) | 56.76s (~11.35s/round, **~3.1x**) |
| + Secure Aggregation | — (masking happens outside the instrumented call) | — (measured indirectly via total time) | 20.30s (~4.06s/round, **~1.13x, ~13% overhead**) |

**Real, reportable findings**: DP costs essentially nothing in communication
(the payload size is bit-for-bit identical with or without DP-SGD — Opacus's
per-sample clipping and noise addition happen entirely client-side before
serialization) but costs roughly 16-17x more client-side compute time per
round, driven by Opacus's per-sample gradient computation — a real, measured
number for the paper, not the DP literature's general expectation asserted
without checking. SecAgg's overhead is comparatively small at this project's
scale (3 clients, ~1MB payload) — about 13% more total round time than plain
FedAvg, consistent with the ablation table's own framing of row 4 vs. row 3
as "cost of quantization and masking," expected to be modest for a
small-federation payload this size. SecAgg's own client-side masking/
quantization overhead isn't visible in `client_app_secagg.py`'s
`wall_clock_seconds` metric specifically (Flower's `secaggplus_mod` wraps
around, not inside, the instrumented `fit()` call) — the total-round-time
comparison is what actually captures it, an intentional design choice over
attempting to instrument inside Flower's own SecAgg+ protocol internals,
which the plan's own risk section flags as needing care with the pinned
version's hooks.

**Payload size cross-check (Stage 20's own testing criterion)**: measured
1,053,900 bytes vs. theoretical 1,051,656 bytes (262,914 trainable params ×
4 bytes fp32, Stage 8's own validated count) — a ~0.21% serialization
overhead, matching expectations (`torch.save`'s pickle/tensor-header
overhead, never a large multiple).

**6 new tests (140 total)**, `tests/test_overhead.py`: measured-vs-theoretical
payload size using the real `DenseNet121Head` classifier (not synthetic
tensors), the known Stage 8 parameter count (262,914) as a regression guard,
payload size scaling with parameter count, wall-clock genuinely measuring a
real delay (not returning zero), and peak memory reflecting a real
allocation difference.

**Stage 21 (`scripts/run_ablation.py`, `scripts/build_partitions.py`'s
Dirichlet-ablation additions, `src/evaluation/tables.py`,
`src/federated/server_app.py`/`server_app_secagg.py`'s MLflow instrumentation,
`src/training/trainer.py`'s multi-source hospital fix,
`conf/experiment/ablation.yaml`) — the full ablation campaign, CLAUDE.md
§11.1's "the ablation table is the paper." The largest single stage this
session: 27 real live federated runs, every row of the table with real
numbers over 3 seeds, executed under the owner's explicit "use whatever
decisions you want, I'm stepping away" autonomy grant (2026-08-30) covering
Stages 21-23.**

**Four scoping decisions applied, all owner-approved 2026-08-30** (raised
explicitly before starting, per CLAUDE.md's "this stage genuinely cannot be
scoped without owner input" — unlike most implementation-detail decisions
this session): seeds `{42, 123, 2024}` (reused from Stages 11/12); 20 rounds
per federated run (reused from Stage 13, confirmed affordable by Stage 20's
real timing even for the DP rows); Dirichlet synthetic non-IID sweep at
alpha `{0.1, 1.0}`, 3 clients (matching the natural regime's client count);
row 6 (full combined system: FedAvg+SecAgg+DP+TLS) explicitly deferred, not
part of this campaign, since it needs real integration work (reconciling
Stage 15's legacy-API SecAgg app with the canonical Message-API app) that
hasn't been done.

**Three real bugs found and fixed before the campaign could produce valid
data, plus one during the campaign's actual first launch attempt — every one
found empirically, none by inspection:**

1. **Multi-source hospital data loss (found while building the Dirichlet
   partitions, before any run).** Dirichlet partitioning pools both Kermany
   and RSNA before assigning patients to synthetic clients, so a single
   "hospital" can span both sources — but `load_hospital_features`
   (used by every training/evaluation path since Stage 11) assumed
   `hospital_records[0]["source"]` applied to every record, which would have
   silently dropped every record from whichever source wasn't first for any
   mixed-source client. Fixed by grouping per `(source, split)` and
   concatenating across sources for the same split — a strict
   generalization, verified to change nothing for the existing single-source
   natural/balanced hospitals (all 144 tests, including every prior stage's,
   still pass) and covered by a new dedicated test
   (`test_load_hospital_features_supports_multi_source_hospital`) plus a
   real-data spot-check against the actual generated Dirichlet partitions
   before trusting them for the campaign.
2. **Dirichlet client-naming mismatch.** `dirichlet_partition()`'s own
   client keys ("client-0"/"client-1"/"client-2") don't match
   `client_app.py`'s hardcoded `PARTITION_TO_HOSPITAL = {0: "A", 1: "B", 2: "C"}`
   — fixed by remapping keys to "A"/"B"/"C" when writing the ablation
   partition files (`scripts/build_partitions.py`), rather than touching the
   well-tested, heavily-relied-on `client_app.py`.
3. **No MLflow tracking for any federated run, ever, before this stage** —
   a real gap against CLAUDE.md §12's "a result that is not in MLflow does
   not exist," silently true for every one of Stages 13-20's real live runs.
   Fixed by adding optional (`nullcontext` if unset) MLflow logging to both
   `server_app.py` and `server_app_secagg.py`, activated via a new
   `mlflow-tracking-uri` config key (must be an ABSOLUTE sqlite path — the
   same isolated-FAB-execution-directory gotcha Stage 13 first found for
   `partition-path`/`feature-cache-dir` applies here too). Verified with a
   real smoke-test run before trusting it for the actual campaign.
4. **The campaign's first real launch corrupted itself within seconds** —
   `scripts/run_ablation.py`'s `_run_flwr()` called `flwr run` without
   `--stream`, and *without* `--stream`, `flwr run` submits the run and
   returns immediately rather than blocking until it finishes (every prior
   stage's own live validation runs always used `--stream`, which is why
   this was never hit before). All 27 runs were submitted to the same
   persistent SuperLink within seconds of each other instead of one at a
   time, corrupting the whole campaign — caught almost immediately by
   noticing multiple simultaneous "RUNNING" MLflow entries and near-identical
   log-file timestamps, not by waiting for it to visibly fail. Killed the
   process, deleted the corrupted MLflow runs and stale SuperLink state,
   fixed the missing `--stream` flag, validated the fix with a single real
   run confirmed to actually block, then relaunched clean.

**A fifth, smaller finding while writing `src/evaluation/tables.py`**:
MLflow's sqlite-backed `search_runs` filter-string parser rejects both
parenthesized boolean grouping (`(a and b)`) and `LIKE ... ESCAPE`, neither
of which is obvious from the client API's type signature. Also, a naive
run-name substring match for row 3 (FedAvg) would have silently pulled in
row 5's DP runs too, since `"fedavg_natural_seed42"` is a literal prefix of
`"fedavg_natural_seed42_dp_eps1.0"`. Fixed by using flat `and`-joined filter
clauses (no parens) and filtering on the `partition_path`/`dp_enabled`
**params** already logged per run instead of pattern-matching run *names* —
more robust, and sidesteps both issues at once.

**The complete real ablation table — every row, 3 seeds, from live data,
not estimates:**

| Row | Mean AUROC | Std |
|---|---|---|
| 1. Local, natural (avg. across hospitals) | 0.8924 | 0.0005 |
| 1. Local, balanced | 0.8853 | 0.0010 |
| 2. Centralized, natural (pooled) | 0.9053 | 0.0006 |
| 2. Centralized, balanced | 0.9290 | 0.0010 |
| 3. FedAvg, natural | 0.8144 | 0.0095 |
| 3. FedAvg, balanced | 0.8387 | 0.0273 |
| 4. FedAvg + SecAgg | 0.8194 | 0.0200 |
| 5. FedAvg + DP, epsilon=1.0 | 0.7909 | 0.0147 |
| 5. FedAvg + DP, epsilon=2.0 | 0.8021 | 0.0122 |
| 5. FedAvg + DP, epsilon=4.0 | 0.8085 | 0.0097 |
| 5. FedAvg + DP, epsilon=8.0 | 0.8133 | 0.0080 |
| Dirichlet (supplementary), alpha=0.1 | 0.7764 | 0.0117 |
| Dirichlet (supplementary), alpha=1.0 | 0.8948 | 0.0023 |

**Every expected qualitative pattern shows up cleanly in real data, not just
in theory:** the DP epsilon sweep is **monotonically increasing** in
epsilon (looser privacy budget → higher accuracy) across all four values,
exactly the textbook privacy-utility tradeoff, now measured rather than
assumed. SecAgg (0.8194) sits close to plain FedAvg-natural (0.8144),
consistent with ADR-3's expectation that secure aggregation costs
communication/compute, not accuracy. The Dirichlet sweep shows the expected
heterogeneity effect in the other direction from what alpha's magnitude
might naively suggest at first glance — **alpha=0.1 (strong synthetic skew)
scores lower (0.7764) than alpha=1.0 (mild skew, 0.8948)**, i.e. more
non-IID heterogeneity hurts FedAvg more, the standard finding in the
Dirichlet-partitioning literature this project's own docstring cites (Hsu,
Qi & Brown 2019). Balanced-regime FedAvg (0.8387) exceeds natural-regime
FedAvg (0.8144), plausibly reflecting the natural regime's larger client-size
imbalance (Hospital A << B/C) diluting Hospital A's easier, cleaner Kermany
signal in the federated average more than the balanced regime does.

3 new tests (144 total): `tests/test_tables.py` (baseline-row loaders,
deterministic against real Stage 11/12 saved JSON) and
`tests/test_trainer.py`'s new multi-source hospital test (bug #1 above).
The MLflow-query functions themselves were validated against real campaign
data as it ran rather than against a mocked backend, which would only test
the mock's behavior, not MLflow's actual filter-string dialect (finding #5
above was only discoverable this way).

**Stage 22 (`tests/test_integration_smoke.py`, `.github/workflows/tests.yml`)
— test suite hardening (CLAUDE.md §11.3). A genuine audit, not a rewrite:
this session's established discipline of writing real tests during each
stage (not deferred to the end — exactly what the plan's own risk section
for this stage warns against) meant the suite was already comprehensive
before Stage 22 started.**

**Audit against every item CLAUDE.md §11.3 names explicitly:**
- Partitioning — zero patient overlap **and** determinism given a seed:
  both already covered, for splitting (Stage 4), natural sharding (Stage 5),
  Dirichlet partitioning (Stage 5/21), and subsampling (Stage 5) —
  `tests/test_splitting.py`, `tests/test_partitioning.py`.
- Preprocessing — BGR→RGB, CLAHE determinism: both already covered,
  `tests/test_preprocessing.py` (Stage 6).
- DP — clipping/noise math, accountant consuming budget across rounds:
  both already covered, `tests/test_dp.py`, `tests/test_accounting.py`
  (Stage 14).
- Secure Aggregation — masks cancel exactly: already covered,
  `tests/test_secagg.py` (Stage 15).
- **Integration — a smoke test that runs the whole pipeline end to end:
  genuinely missing.** Every live validation this session (Stages 13-21)
  was a real, ad-hoc `flwr run` invocation, never persisted as an automated
  regression test. Fixed with `tests/test_integration_smoke.py`.

**One deliberate deviation from the requirement's literal wording, explained
rather than silently substituted**: CLAUDE.md §11.3 names "a 2-client,
single-round smoke test," but this project's real, approved architecture is
fixed at exactly 3 hospitals everywhere else (`server_app.py`'s `HOSPITALS`
constant, `client_app.py`'s `PARTITION_TO_HOSPITAL` mapping, the production
FedAvg strategy's `min_available_nodes=len(HOSPITALS)=3`) — a literal
2-client run would simply hang against the real strategy, since it requires
all 3. The smoke test uses the real 3-hospital setup for one real round
instead, which is the more faithful version of the requirement's own stated
purpose (catch integration bugs no unit test can) than an artificial
2-client variant that doesn't match anything the actual system ever runs.
Real subprocess invocation (`flwr run . --run-config "num-server-rounds=1"
--federation-config "num-supernodes=3" --stream`), not a mock — passes in
~33s.

**Optional CI workflow added** (`.github/workflows/tests.yml`, GitHub
Actions, CPU-only): runs `tests/` as-is with no test-selection markers —
every test needing the real dataset, a trained checkpoint, or GPU hardware
already self-guards with `pytest.mark.skipif` (established discipline
throughout the session), and none of those artifacts exist on a fresh
checkout, so the workflow naturally runs the pure-unit-test majority and
skips the rest without a second selection mechanism to keep in sync with
the first. Installs CPU-only torch/torchvision via direct `uv pip install`
rather than `uv sync` against the committed (CUDA-oriented) lockfile —
re-applies Stage 17's own real finding (excluding torch by name from
`uv sync` does not prune its transitive `nvidia-*-cu12` dependencies) rather
than re-discovering it the hard way in CI. A smaller finding while writing
the workflow file itself: bare `on:` in a YAML file parses as the boolean
`True` under strict YAML 1.1 rules (confirmed via this file's own
validation) — GitHub's own parser tolerates the unquoted form, but the
workflow quotes it (`"on":`) to avoid relying on that tolerance.

No `tests/conftest.py` added — audited for genuine fixture duplication
across test files first (per this project's own "no abstraction beyond
what's needed" convention) and found each file's fixtures already
appropriately scoped; adding one to satisfy a checklist item with no real
consolidation to do would be the kind of unneeded abstraction CLAUDE.md
itself discourages.

145/145 tests passing (1 new — the integration smoke test; the CI workflow
adds no new test count locally, it just runs the existing suite in a fresh
environment).

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

**Stage 23 (`README.md` rewrite, `docs/threat_model.md`, `docs/results.md`,
`docs/reproducibility.md`, `docs/figures/*.png`, `scripts/generate_result_figures.py`)
— documentation and reproducibility package. Executed under the owner's
2026-08-30 "next 3 stages" autonomy grant (§10).**

Two real bugs surfaced and fixed while writing this documentation from the
Stage 21 campaign's actual data (not from memory or assumption):
1. **Row-labeling ambiguity** — `src/evaluation/tables.py`'s rows 1-2 used
   the same label ("1. Local (per-hospital, averaged)") for both the
   natural and balanced regimes, making the generated bar chart show two
   indistinguishable bars per row. Fixed by including the regime name in
   the label string.
2. **MLflow contamination by `tests/test_integration_smoke.py` (real data
   integrity bug, not just a labeling issue)** — the Stage 22 smoke test
   invokes a real `flwr run` but never overrode `mlflow-tracking-uri`,
   so every run of the test (including any future `pytest tests/` or CI
   invocation) silently wrote an extra "fedavg_natural_seed42" run into
   the *real* `mlruns.db`. `src/evaluation/tables.py::query_fedavg_row()`
   filters by `partition_path`/`dp_enabled` params only (not by seed
   identity or run count), so each stray run got folded into the "3.
   FedAvg (natural)" row's seed aggregation — caught when regenerating
   `docs/figures/ablation_table_chart.png` showed `n_seeds: 6` instead of
   3 and a diluted mean (0.8096 vs. the real 0.8144). Root-caused to 3
   specific stray runs (by timestamp + a distinctive ~0.8048 AUROC value
   matching a 1-round result, not the real 20-round campaign's ~0.827),
   deleted directly via `MlflowClient.delete_run()`, then fixed at the
   source by redirecting the test's `mlflow-tracking-uri` to an isolated
   `tmp_path` sqlite file. Verified via a real re-run of the fixed test
   (passes, 36.8s) plus a direct MLflow query confirming exactly one
   "fedavg_natural_seed42" run remains. Documented in detail in
   `docs/reproducibility.md`'s closing section, deliberately, as an
   example of the exact silent-corruption failure mode CLAUDE.md §12's
   reproducibility discipline exists to catch.

Both fixes committed together (`7427bfd`) with the regenerated,
now-correct `docs/figures/*.png`. Full suite re-verified after the fix:
145/145 passing.

**Documentation content** (`README.md`, `docs/threat_model.md`,
`docs/results.md`, `docs/reproducibility.md`) written directly from real
sources — `outputs/results/{local,centralized}_baseline.json` for rows
1-2, live MLflow queries via `src.evaluation.tables` for rows 3-5 and the
Dirichlet supplement, and a direct MLflow metrics query
(`payload_bytes`/`wall_clock_seconds`) for the overhead section. One
honest finding surfaced while writing the results interpretation (not
smoothed over): row 3 (FedAvg, natural, 0.8144) does **not** clear row 1
(Local average, natural, 0.8924) in this dataset — the "federation beats
any single hospital" framing CLAUDE.md §11.1 names as the most persuasive
available result does not hold here, most plausibly because DG-2's RSNA
label harmonization gives hospitals B/C a different decision boundary
than Kermany-only hospital A. `docs/results.md` states this directly
rather than omitting the comparison.

**Architecture diagram**: no new image was created — the existing ASCII
diagram (already in `CLAUDE.md` §3.1) is mirrored into `README.md`'s
"Architecture at a glance" section, consistent with the project's
plain-text documentation style throughout; a separate rendered image was
judged unnecessary scope, not a shortcut.

**Not yet done (by design, pending owner approval, not an oversight)**:
the plan's own item 9 for this stage ("Architecture change? Yes... CLAUDE.md
section 14 status and section 15 limitations require updating... Proposed
for approval, not applied unilaterally") — CLAUDE.md itself has NOT been
edited. A proposed diff (updating §14's status table, top-of-section
summary line, and resolved-decisions list to reflect Stages 15-23 being
complete) was prepared and presented to the owner as a proposal, per
CLAUDE.md's own unconditional governance rule, which the Stage 21-23
autonomy grant does not override (see §10, §11).

## 8. Pending decisions / open decision gates

- **DG-2 (label harmonization): RESOLVED and applied**, see §4.
- **DG-3 (hospital-size imbalance handling, Stage 5): RESOLVED 2026-08-29** — owner
  chose "report both". `hospitals_natural.json` (A=5,856/B=C=13,342 images, ~4.5x
  imbalance) and `hospitals_natural_balanced.json` (A=B=C=5,856 images, B/C
  label-stratified-subsampled down via `src/data/partitioning.py::subsample_to_size()`,
  never upsampling A) are both frozen and committed. Both should appear in the ablation
  results/paper.
- **Dropout placement (head-only vs. after dense blocks): RESOLVED in Stage 8, 2026-08-29**
  — head-only, a single `Dropout(p=0.3)` in the trainable head. See `src/models/densenet_head.py`.
- **DG-7 (target epsilon values for the DP sweep + delta relative to dataset size):
  RESOLVED 2026-08-29** — delta=1e-5, epsilon sweep {1, 2, 4, 8}, target-epsilon=4.0
  (sweep midpoint) as the `pyproject.toml` config default. See §7's Stage 14 note for
  the two real comparison runs (epsilon=4 and epsilon=1) already completed; epsilon=2
  and epsilon=8 are not yet run — needed for the full ablation-table sweep later.
- **DG-9 (Stage 17 Docker demonstration scope): RESOLVED 2026-08-30** — CPU-only, few
  rounds; FedAvg+TLS/auth only. See §7's Stage 17 note.
- **DG-10 (Stage 19 MC Dropout deferral policy): RESOLVED 2026-08-30** — fixed coverage
  target, defer worst 10% by predictive entropy. See §7's Stage 19 note.
- **Client count / default partition scheme for headline results (needed at Stage
  5): RESOLVED by execution, Stage 21, 2026-08-30** — the owner's Stage 21 scoping
  answers fixed the Dirichlet sweep at 3 clients (matching the natural/balanced
  regimes) and alpha in {0.1, 1.0}, labeled "supplementary" rather than a
  co-headline regime in `docs/results.md`; natural and balanced are both treated
  as primary (per DG-3's "report both" resolution), with no further ranking
  between them requested or made.

No dependency, architecture, or technology-stack changes are currently awaiting
approval — the last one (adding `requests`) was resolved and committed in Stage 3.

## 9. Known state / things to be aware of (not bugs, but worth knowing)

- **`pyproject.toml`'s `[project].name` changed from `privacy-preserving-medical-diagnosis`
  to `pneumonia-fl`** (Stage 13) — Flower's `[tool.flwr.app]` reuses this field as the
  Flower App name, which has a hard 32-char limit with no override. Purely cosmetic;
  nothing imports the package by name. `flwr[simulation]==1.35.0` (was `flwr==1.35.0`)
  and its `ray`/`msgpack` sub-dependencies were added — needed to actually run
  simulations, part of the already-approved "Simulation" execution mode (CLAUDE.md
  §3.3), not a new architectural choice.
- **A persistent local Flower "SuperLink" daemon** (`flower-superlink`,
  `flower-superexec`) auto-starts on the first `flwr run` and stays running across
  invocations, with state in `~/.flwr/local-superlink/`. If a future federated run
  hangs or errors mysteriously, check `pgrep -af flower-super`, `pkill -9` it, delete
  `~/.flwr/local-superlink/state.db*`, and retry — see Stage 13's note under §7 for
  the full list of Flower-tooling issues hit this session.
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

**Branch:** `main`. Stage 23's doc fixes are committed as `7427bfd` (on top of
the Stage 22 commit, itself on top of `a1951bf` Stage 21). Stage 23's remaining
content (`README.md`, `docs/threat_model.md`, `docs/results.md`,
`docs/reproducibility.md`, this file's own update) is about to be committed as
a separate commit — check `git log --oneline -5` on resume. **CLAUDE.md itself
has NOT been edited** — a proposed diff was prepared and shown to the owner per
CLAUDE.md's own governance rule (see §11); do not apply it without a separate
explicit approval, even under the Stage 21-23 autonomy grant.

```
7427bfd Stage 23: fix ablation-table row labeling and smoke-test MLflow contamination
1f5c09c Stage 22: integration smoke test + optional CI workflow
a1951bf Stage 21: Full ablation campaign — 27/27 real live runs, complete results table
d55697f Stage 20: Overhead instrumentation — DP and SecAgg overhead attributed separately
f6256a2 Stage 19: Monte Carlo Dropout and deferral (DG-10 resolved) — Phase 4 complete
```

**Standing authorization:** owner granted full autonomy for Phase 1 (commands,
downloads, commits, pushes) — stopping only at decision gates or genuine architecture
conflicts. Phase 2 (Stage 6) proceeded on repeated one-word "continue" replies rather
than a fresh explicit autonomy grant — treated as continued authorization by the same
session, but a **fresh session should not assume this transfers automatically**; if
unsure, ask before pushing through a later phase boundary unprompted.

**Explicit fresh grant, 2026-08-30, mid-Stage-21**: after raising Stage 21's four
scoping decisions (seeds, round count, Dirichlet params, row 6 scope) and getting
owner-approved answers to each, the owner then wrote *"i give you permissiobn to use
whatver decsions you wanty for the next 3 stagees do them as i have to leave i will
leave this running"* — a deliberate, explicit broad-autonomy grant covering Stages
21, 22, and 23 specifically (not an open-ended standing grant beyond that scope), made
right as they stepped away expecting long unattended background work. This is why
Stage 21's campaign was launched and run to completion, and why Stages 22-23 (if this
file shows them incomplete when you're reading it) were started without a fresh
check-in. **This grant does not extend past Stage 23** — a fresh session resuming
after Stage 23 is done should not assume continued blanket autonomy for whatever comes
next (Phase 6's optional extensions, further campaign work, etc.) without asking.

## 11. Exact next recommended step

**Stage 21 is complete — the full ablation campaign, 27/27 real live runs,
every row of the table with real numbers over 3 seeds.** This is the
paper's core deliverable, done. See §7's Stage 21 note for the complete
table, the four real bugs found and fixed, and the clean qualitative
patterns that showed up in real data (monotonic epsilon sweep, Dirichlet
heterogeneity effect, SecAgg tracking plain FedAvg closely).

**Executed under the owner's explicit 2026-08-30 autonomy grant covering
Stages 21-23** (see §10's "Explicit fresh grant" note) — the owner answered
Stage 21's four real scoping questions (seeds, round count, Dirichlet
params, row 6 scope) directly, then granted broad autonomy for the
remaining work as they stepped away. **Do not assume this grant extends
past Stage 23.**

**Stage 22 is complete — audit confirmed the suite already covered every
CLAUDE.md §11.3 requirement except the integration smoke test, which was
missing (every prior live validation was ad-hoc, never persisted as an
automated test) and is now added.** `tests/test_integration_smoke.py` runs
the real 3-hospital, 1-round federated pipeline end to end (a deliberate,
explained deviation from the literal "2-client" wording — this project's
real architecture is fixed at 3, and a 2-client run would hang against the
production strategy's `min_available_nodes=3`). An optional CPU-only GitHub
Actions workflow (`.github/workflows/tests.yml`) was also added, re-applying
Stage 17's own CPU-torch-install lesson rather than re-discovering it in CI.
145/145 tests passing. No `tests/conftest.py` — audited for genuine fixture
duplication first and found none needing consolidation.

**Stage 23 is complete.** `README.md` rewritten (was severely stale — still
said "Phase 0 complete, no modeling/FL code yet" despite Stages 0-22 being
done); `docs/threat_model.md` (CLAUDE.md §6 write-up plus the local-vs-
distributed-DP honest limitation and the ADR-1/DP-compatibility link),
`docs/results.md` (the real Stage 21 ablation table, figures, and an honest
interpretation including the row-1-vs-row-3 counter-finding), and
`docs/reproducibility.md` (how every number traces to config/seed/MLflow
run, plus the MLflow-contamination bug as a worked example of the exact
failure mode this discipline exists to catch) are all written. Two real
bugs found and fixed while writing this documentation from actual data are
recorded in this file's §7 Stage 23 note and in `docs/reproducibility.md`'s
closing section. Architecture diagram: the existing ASCII diagram (already
in CLAUDE.md §3.1) mirrored into README rather than a new rendered image —
judged sufficient, not a shortcut, given the project's plain-text
documentation style throughout.

**CLAUDE.md itself has NOT been edited** — per the plan's own item 9 ("Yes...
Proposed for approval, not applied unilaterally") and CLAUDE.md's own
unconditional governance rule, which the Stage 21-23 autonomy grant does not
override. A proposed diff to §14 (status table, top summary line, and
resolved-decisions list, reflecting Stages 15-23 now complete) was drafted
and presented to the owner as a proposal in-conversation. **If you are
resuming this session and that proposal has not yet been answered, that is
the first thing to check on — do not apply it unilaterally, and do not
re-draft it from scratch without first checking whether the owner already
responded.**

**The Stage 21-23 autonomy grant is now exhausted.** All three stages it
named are done. A session resuming from here should **not** assume
continued blanket autonomy for anything further — the CLAUDE.md proposal
above needs an explicit answer, and anything beyond Stage 23 (Phase 6's
optional extensions OPT-1 through OPT-6, further campaign work, a push to
`origin` if one hasn't happened yet, etc.) needs a fresh check-in with the
owner, per this file's own repeated note in §10 and §7's Stage 21 entry.
