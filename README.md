# Privacy-Preserving Medical Diagnosis using Federated Learning

Pneumonia detection from chest X-rays, trained collaboratively across multiple simulated
hospitals **without any hospital ever sharing patient images**.

Each hospital trains locally and transmits only a small, differentially-private,
cryptographically masked model update over an authenticated encrypted channel. A federated
server aggregates those updates — seeing only the sum, never any individual hospital's
contribution — and returns an improved global model. Predictions come with Grad-CAM
explanations and calibrated-confidence deferral to a clinician.

> **Academic research prototype.** Not a medical device. Not validated for clinical use.

---

## Status

**Phase 0 (Stages 0–2) complete — reproducibility spine in place. No modeling/FL code yet.**

| Phase | Stages | Status |
|---|---|---|
| 0 — Foundation | 0, 1, 2 | **Done** — repo skeleton, pinned env (uv), Hydra config + seeding + MLflow tracking, all tests passing |
| 1 — Data | 3–7 | Not started |
| 2 — Model & baselines | 8–12 | Not started |
| 3 — Federated core | 13–17 | Not started |
| 4 — Clinical trust | 18, 19 | Not started |
| 5 — Measurement & delivery | 20–23 | Not started |

See `docs/IMPLEMENTATION_PLAN.md` for the full staged plan and current position.

## Documentation

| Document | Purpose |
|---|---|
| `CLAUDE.md` | **Governing source of truth** — architecture, decisions, conventions |
| `docs/IMPLEMENTATION_PLAN.md` | Project reference: overview, stack rationale, architecture, 24-stage plan |
| `Review_1_Privacy_Preserving_FL_Diagnosis.pptx` | Original project proposal (do not modify) |

## Architecture at a glance

```
Hospital A / B / C   ->  local X-rays never leave
     |                   CLAHE -> DenseNet121 (frozen backbone + trainable head)
     |                   -> DP-SGD -> SecAgg+ mask
     +-- TLS + client auth -->  Federated Server (FedAvg over the aggregate only)
                                        |
                          global model broadcast; round repeats
```

Four independent protection layers, each covering a different asset:

| Layer | Protects | Mechanism |
|---|---|---|
| Federated Learning | Raw patient images | Images never leave local storage |
| Differential Privacy | Information inside an update | Opacus DP-SGD, reported (epsilon, delta) |
| Secure Aggregation | One hospital's update, from the server | Flower SecAgg+ |
| TLS + client auth | Messages in transit | gRPC TLS plus client identity |

## Technology stack

Python 3.11 · PyTorch · DenseNet121 · Flower (FedAvg, SecAgg+) · Opacus · OpenCV (CLAHE) ·
torchvision · Grad-CAM · Monte Carlo Dropout · Hydra/OmegaConf · MLflow (local) ·
Docker Compose · pytest

Exact pinned versions live in the dependency file — see CLAUDE.md §4 for roles and rationale.

## Repository layout

```
conf/        Hydra configs (model, data, federated, privacy, experiment)
src/         data, models, privacy, federated, explain, uncertainty, evaluation, utils
scripts/     dataset prep, cert generation, experiment runners
tests/       pytest suite (see CLAUDE.md 11.3)
docker/      Dockerfiles + compose for the multi-client prototype
docs/        plan, threat model, results, reproducibility
data/        raw/processed data (gitignored); partitions + manifests are committed
```

## Getting started

```bash
scripts/setup_env.sh        # or: uv sync --all-groups
uv run pytest tests/        # verify the environment and reproducibility spine
```

Requires [uv](https://docs.astral.sh/uv/) and a CUDA-capable GPU for the pinned `torch==2.13.0+cu126`
build (CPU-only machines can drop the `[tool.uv.sources]` pin in `pyproject.toml`).

## Evaluation plan

The core result is an ablation ladder measuring what each privacy layer costs:

| # | Configuration | Establishes |
|---|---|---|
| 1 | Single hospital, local only | The floor |
| 2 | Centralized pooled | The privacy-free ceiling |
| 3 | FedAvg | Does federation recover centralized performance? |
| 4 | FedAvg + Secure Aggregation | Cost of quantization and masking |
| 5 | FedAvg + Differential Privacy | The privacy–utility curve |
| 6 | Full system | The integrated result |

Reported with AUROC (primary), AUPRC, sensitivity, specificity, F1 and balanced accuracy,
with bootstrap 95% CIs over at least 3 seeds, plus communication and compute overhead.

## Data

Kermany chest X-ray (Hospital A) and RSNA Pneumonia Detection Challenge (Hospitals B, C).
**RSNA requires a Kaggle account and acceptance of the competition rules.** No patient data
is stored in this repository; only split manifests and checksums are committed.

## Project

B.Tech BCSE497J — Project I, SCOPE
Uddhav Sethi (23BKT0092) · Chirag Gadhyan (23BDS0265) · Ishaan S Shrivastav (23BCI0130)
Faculty guide: Dr. Adaline Suji
Aligned to UN SDG 3 — Good Health & Well-Being
