# Privacy-Preserving Medical Diagnosis using Federated Learning

## Project Reference and Implementation Plan

**Project:** B.Tech BCSE497J — Project I, SCOPE
**Team:** Uddhav Sethi (23BKT0092), Chirag Gadhyan (23BDS0265), Ishaan S Shrivastav (23BCI0130)
**Faculty guide:** Dr. Adaline Suji
**Repository:** UddhavSethi/privacy-preserving-medical-diagnosis
**Date:** 29 August 2026
**Status:** Architecture approved. No implementation has begun.

**Purpose of this document.** A single living reference for tracking the project: what it is, how the technology stack changed from the Review 1 proposal and why, the resulting architecture, and the staged plan from an empty repository to a finished research prototype. `CLAUDE.md` in the repository remains the governing source of truth for the architecture; this document explains and expands on it.

**How this document is organised**

| Part | Contents |
|---|---|
| I | About this project — what it is, the problem, the gap, the objectives |
| II | Technology stack — original proposal, revised stack, every change and its justification |
| III | System architecture — topology, federated round, privacy layers, key decisions |
| IV | Staged implementation plan — 24 required stages plus 4 optional extensions |

---

# PART I — ABOUT THIS PROJECT

## 1. Overview

This project builds a system that lets several hospitals jointly train a pneumonia-detection model on chest X-ray images **without any hospital ever sending its patient images anywhere**.

Each hospital trains a copy of the model on its own images, inside its own walls. Only a small mathematical summary of what the model learned — never an image — leaves the hospital. A central server combines those summaries into one improved shared model and sends it back. The cycle repeats until the shared model is accurate.

Three further protections sit on top of that arrangement, because the summaries themselves are not automatically safe, and because a model a doctor cannot interrogate is of little clinical use:

- **Differential Privacy** mathematically limits how much any individual patient's data can influence the summary, so patient information cannot be reverse-engineered from it.
- **Secure Aggregation** cryptographically masks each hospital's summary so that even the central server sees only the combined total, never one hospital's contribution on its own.
- **TLS with client authentication** encrypts every message in transit and ensures only registered hospitals can participate.

Finally, the model explains itself and knows when it is unsure. **Grad-CAM** produces a heatmap showing which regions of the lung drove each prediction, and **Monte Carlo Dropout** attaches a confidence score, with low-confidence cases automatically deferred to a human clinician rather than acted upon.

## 2. The problem being solved

**Data cannot be centralised.** Modern chest X-ray classifiers are trained by pooling images onto a single server. Hospitals cannot legally or ethically do this with patient data. Most published pneumonia-detection work therefore assumes the privacy problem away rather than solving it, and depends on data-sharing agreements that in practice rarely exist.

**Model updates are not automatically safe.** Where federated learning is used, the shared updates are commonly treated as safe by default. Research shows that updates can still leak information about the images used to produce them, so federation alone is a necessary but insufficient protection.

**Clinical trust is missing.** A prediction with no explanation and no calibrated confidence is not actionable in a hospital. Doctors will not, and should not, act on a black box.

## 3. Research gap

Each individual ingredient of a solution already exists in the literature, but they have not been assembled.

- Centralised training remains the default in published pneumonia-detection work.
- Where federated learning is used, protection of the updates themselves is usually absent.
- Security concerns — encrypted transport, protecting a hospital's update from the server itself — are frequently listed as future work.
- Clinical usability, meaning explanation and confidence, is rarely built in at all.

**The gap in one sentence:** no single published framework combines Federated Learning, Differential Privacy, Secure Aggregation, TLS, explainability and uncertainty into one working pneumonia-detection pipeline.

## 4. Objectives

1. Build a federated pipeline where hospitals train locally and share only model updates, never images.
2. Apply Differential Privacy with a formal mechanism and accountant, reporting privacy as (epsilon, delta).
3. Add Secure Aggregation so even the server cannot see any single hospital's update.
4. Secure all communication with TLS plus client authentication.
5. Achieve strong detection performance using DenseNet121 transfer learning despite limited per-hospital data.
6. Make the model explainable through Grad-CAM and confidence-aware through Monte Carlo Dropout, with deferral to human review.
7. Produce a reproducible artifact: seeded, configuration-driven, tracked and tested.

**Prioritisation order when requirements conflict:** privacy, then security, correctness, reproducibility, academic credibility, explainability, uncertainty estimation, maintainability, and finally raw accuracy. Accuracy is deliberately last — a model that scores higher by weakening a privacy guarantee or by becoming irreproducible is a regression, not an improvement.

## 5. What is novel here

**The novelty is integration, not algorithmic invention.** DenseNet121, DP-SGD, Secure Aggregation, Grad-CAM and Monte Carlo Dropout are each individually well studied. Nothing here is newly invented, and the project does not claim otherwise.

The contribution is showing that these components compose into a single coherent, reproducible, end-to-end pipeline — **and quantifying what that composition costs** in accuracy, calibration, explanation quality, communication and compute. Most published work adopts one or two of these pieces; this project demonstrates they can work together in a form a real hospital network could adopt.

Because the claim is integration, the ablation table in Part IV is the central result of the work. Any activity that neither builds the pipeline nor measures the cost of a layer within it is out of scope by default.

**Alignment and intended outcome:** UN Sustainable Development Goal 3 (Good Health and Well-Being); a Scopus-indexed conference paper.

---

# PART II — TECHNOLOGY STACK: ORIGINAL AND REVISED

## 6. The originally proposed stack

The Review 1 presentation proposed the following components.

| Component | As originally proposed |
|---|---|
| Language | Python (version unspecified) |
| Deep learning | PyTorch |
| Model | DenseNet121, transfer learning |
| Preprocessing | OpenCV — resizing, CLAHE, normalization |
| FL framework | Flower |
| Transport | gRPC |
| Aggregation | Federated Averaging (FedAvg) |
| Privacy | Differential Privacy — "calibrated noise added to the update" |
| Security | Secure Aggregation — protocol unspecified |
| Transport security | TLS encryption |
| Explainability | Grad-CAM |
| Uncertainty | Monte Carlo Dropout |

This selection was fundamentally sound. The review that followed did not replace the core of it.

**However, several things were entirely absent from the proposal**, and for an academic prototype these omissions were a larger risk than any of the tool choices: no dataset was named; no evaluation metrics or baselines were defined; there was no configuration management, no experiment tracking, no seeding or reproducibility strategy, no testing, and no explicit threat model. The privacy mechanism was also described only informally, with no unit of protection and no privacy budget.

## 7. The revised and approved stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python 3.11 (pinned) | Ecosystem compatibility |
| Deep learning | PyTorch | Model and training loop |
| Model | DenseNet121, ImageNet-pretrained | Frozen backbone plus small trainable head |
| FL framework | Flower (exact version pin) | Orchestration, both execution engines |
| Aggregation | FedAvg | Server-side aggregation |
| Secure Aggregation | Flower SecAgg+ | Established masking protocol, no custom cryptography |
| Differential Privacy | Opacus DP-SGD | Per-sample clipping, Gaussian noise, formal accountant |
| Transport security | gRPC with TLS and client authentication | Confidentiality, integrity, identity |
| Preprocessing | OpenCV | CLAHE only, cached |
| Transforms | torchvision | Resize, normalize, augment |
| Explainability | Grad-CAM via an established library | Localization heatmaps |
| Uncertainty | Monte Carlo Dropout | Confidence and deferral |
| Configuration | Hydra and OmegaConf | Every run fully specified by config |
| Tracking | MLflow, self-hosted locally | Metrics, parameters, artifacts, offline |
| Containerization | Docker Compose | Multi-container prototype demonstration |
| Testing | pytest | Unit and integration tests |
| Datasets | Kermany and RSNA | Hospital A, and Hospitals B and C |

## 8. Summary of every change

| # | Component | Review 1 | Approved now | Verdict |
|---|---|---|---|---|
| 1 | Python | Unversioned | Pinned to 3.11 | Refined |
| 2 | PyTorch | PyTorch | Unchanged | Kept |
| 3 | DenseNet121 | Model trained and federated as a whole | Backbone frozen; only a small head trained and federated | **Changed (major)** |
| 4 | OpenCV | Entire preprocessing pipeline | CLAHE only, cached; torchvision for everything else | Narrowed |
| 5 | Flower | Flower | Unchanged, but version pinned exactly | Kept |
| 6 | gRPC | gRPC | Unchanged, message-size limit configured explicitly | Kept |
| 7 | Differential Privacy | "Calibrated noise added to the update" | Opacus DP-SGD, sample-level, per-sample clipping, formal accountant, reported (epsilon, delta) | **Changed (major)** |
| 8 | Secure Aggregation | Protocol unspecified | Flower SecAgg+, custom cryptography prohibited | **Changed** |
| 9 | TLS | Server-side TLS | TLS plus client authentication | **Changed** |
| 10 | Grad-CAM | Grad-CAM | Unchanged; established library, not hand-rolled | Kept |
| 11 | Monte Carlo Dropout | Monte Carlo Dropout | Unchanged, but dropout layers must be inserted deliberately | Kept |
| 12 | Configuration | Absent | Hydra and OmegaConf | **Added** |
| 13 | Experiment tracking | Absent | Local MLflow | **Added** |
| 14 | Reproducibility | Absent | Full seeding, lockfile, committed split manifests | **Added** |
| 15 | Dataset | Not named | Kermany plus RSNA, patient-level splits | **Added** |
| 16 | Evaluation | Absent | Six-row ablation ladder, AUROC-led metric suite, bootstrap CIs, three or more seeds | **Added** |
| 17 | Overhead measurement | Absent | Communication and compute cost, attributed per layer | **Added** |
| 18 | Testing | Absent | pytest over privacy-critical paths | **Added** |
| 19 | Deployment | Docker and Kubernetes both future work | Docker Compose pulled into the prototype; Kubernetes stays future work | **Changed** |
| 20 | Threat model | Implicit | Explicitly stated, with exclusions named | **Added** |

## 9. Why each major change was made

### 9.1 DenseNet121 — freeze the backbone, federate only a small head

**This is the single most consequential change in the project.**

The original plan was to train and federate DenseNet121 as a whole. That turns out to be impossible as specified, for five independent reasons that all arrive at once:

1. **Differential Privacy is incompatible with BatchNorm.** DenseNet121 is built on BatchNorm layers, which mix information across samples within a batch. That breaks per-sample gradient computation, which is exactly what DP-SGD requires, and voids the privacy guarantee. Opacus rejects such a model outright rather than training it incorrectly.
2. **Federated averaging is unstable with BatchNorm.** Averaging BatchNorm running statistics across hospitals with different data distributions produces statistics that match no hospital's actual data — a well-documented FedAvg failure mode.
3. **Differential Privacy loses all utility at that scale.** DenseNet121 has roughly 7 million parameters. The accuracy cost of DP noise grows badly with the number of parameters being noised; at 7 million the model is destroyed at any meaningful privacy budget.
4. **The updates are too large to send.** A full update is about 28 MB, which exceeds gRPC's 4 MB default message limit, and is expensive to mask under Secure Aggregation.
5. **It does not fit the available hardware.** DP-SGD stores per-sample gradients, and DenseNet121 is activation-heavy. The 4 GB of GPU memory available for development cannot hold this at any usable batch size.

**The change:** keep DenseNet121, but freeze the pretrained backbone with its BatchNorm layers held in evaluation mode with frozen statistics, and train and federate only a small classifier head.

**Why this works.** Frozen BatchNorm in evaluation mode is a fixed mathematical transformation, so it is safe for per-sample gradients and DP-SGD applies cleanly to the head. No BatchNorm statistics are averaged across hospitals. The federated update shrinks from roughly 7 million parameters to a few hundred thousand, which simultaneously restores DP utility, fits the message limit, makes Secure Aggregation cheap, and fits in 4 GB of memory. It is also ordinary transfer learning, which is exactly what the original proposal already justified on the grounds that no single hospital has enough data to train a large network from scratch.

**What it costs.** Maximum achievable accuracy is lower than full fine-tuning would allow. This is an accepted, reportable trade-off. If head-only accuracy proves insufficient, the approved fallback is to replace BatchNorm with GroupNorm and fine-tune more layers — at the cost of discarding pretrained statistics and requiring more training and memory.

### 9.2 Differential Privacy — from informal noise to a formal guarantee

The original description, adding "carefully calibrated noise" to the update, does not by itself constitute differential privacy. **Noise added without per-sample gradient clipping and without a privacy accountant provides no formal guarantee at all.** It is a common and easily identified weakness, and it would be the first point a reviewer attacks.

**The change:** use Opacus DP-SGD with per-sample gradient clipping to a fixed norm, calibrated Gaussian noise, and a formal accountant that reports (epsilon, delta). The unit of protection is defined as the **training sample**, because the project's stated objective — that no patient's data can be reverse-engineered — is a patient-level claim rather than a hospital-level one.

**An honest limitation that follows.** Because each hospital adds noise independently, this is effectively local differential privacy, which has poorer accuracy than central differential privacy at the same privacy budget. The principled alternative, distributed differential privacy, has each hospital add only a share of the noise while Secure Aggregation ensures only the total is ever revealed. Implementing that rigorously requires specialised noise distributions integrated into the cryptographic protocol, which is beyond a prototype of this scope. The approved approach is to implement per-hospital noise, report the privacy budget under a clearly stated trust assumption, and discuss distributed differential privacy as the principled extension.

### 9.3 Secure Aggregation — use an established protocol, never custom cryptography

The original proposal named no protocol. The failure mode that follows is a hand-rolled masking scheme that is subtly wrong, cannot be verified, and would not be trusted by any reviewer.

**The change:** use Flower's built-in SecAgg+ implementation, which provides an established masking protocol with secret sharing and resilience to hospitals dropping out mid-round. **Writing custom cryptographic protocols is prohibited by the approved architecture.**

**A consequence worth planning for.** Secure Aggregation operates on integers in a finite mathematical field, so updates must be quantized before masking. Quantization interacts with both accuracy and DP noise. Rather than hiding this, the project measures and reports it as one row of the ablation table — the cost of the masking layer is part of the contribution.

### 9.4 TLS — add client authentication

The original proposal claimed TLS would block "eavesdropping, tampering and impersonation." **Server-side TLS does not block impersonation.** It proves the server's identity to the hospital, but does nothing to stop an unauthorised party from connecting while claiming to be a hospital. As specified, the impersonation claim was unsupported.

**The change:** add client authentication, through either mutual TLS with per-hospital certificates or Flower's node-authentication mechanism.

**Why it is worth doing now.** It makes the original claim true, and it delivers at low cost part of what the Review 1 presentation had deferred to future work, namely confirming that an update genuinely came from a registered hospital.

### 9.5 OpenCV — narrowed to CLAHE only

CLAHE contrast enhancement is OpenCV's genuinely irreplaceable contribution here; torchvision has no equivalent. Everything else — resizing, normalization, augmentation — moves to torchvision for composability, correct seeding of random augmentation, and tensor-native execution.

Two specific hazards drove this decision. First, OpenCV reads images in BGR channel order while an ImageNet-pretrained DenseNet121 expects RGB; silently mismatching these costs accuracy in a way that is very hard to notice, so the conversion is made explicit and covered by a test. Second, CLAHE parameters must be fixed, logged and the results cached, otherwise preprocessing becomes a source of run-to-run variation and a throughput bottleneck.

### 9.6 The additions — reproducibility, evaluation and testing

These were absent from the original proposal, and for a project whose output is a research paper they matter more than any individual library choice.

- **Configuration and tracking.** The project runs a matrix of experiments across privacy budgets, hospital counts, data-distribution schemes and random seeds. Without configuration management every run becomes untraceable, and results cannot be defended. Hydra makes each run fully specified by a config file; MLflow records the resolved config alongside the results.
- **MLflow was chosen over cloud tracking deliberately.** For a project whose entire thesis is data sovereignty, shipping experiment telemetry to a third-party service is the wrong posture, and a local tracker also works offline during a demonstration.
- **Seeding.** Beyond the usual framework seeds, the **data-partition seed and the hospital-sampling seed** are recorded. These two are the most commonly forgotten in federated-learning work and they dominate run-to-run variance.
- **Evaluation.** Accuracy alone is not acceptable on imbalanced medical data. The project reports AUROC as the primary metric alongside AUPRC, sensitivity, specificity, F1 and balanced accuracy, with bootstrap confidence intervals and results averaged over at least three seeds. Single-run numbers are not credible in federated learning.
- **Overhead measurement.** Because the contribution is integration, the cost of each layer — in communication and computation — is a first-class result, not an afterthought.
- **Testing.** Privacy and security failures are silent. A test that Secure Aggregation masks cancel exactly, and that the privacy accountant actually consumes budget across rounds, is what converts an implementation claim into a defensible one.
- **Docker Compose was pulled forward** from future work into the prototype. A demonstration in which one server and three hospital containers, each with its own certificate and its own data, communicate over real encrypted connections is far more convincing than three processes on one laptop, and costs roughly a day on top of work already planned. Kubernetes correctly remains future work.

## 10. Considered and deliberately rejected

The following were evaluated and not adopted. They should not be introduced without an explicit decision to revisit.

| Technology | Why rejected |
|---|---|
| TensorFlow, TF-Federated, TF-Privacy | Opacus, the best-maintained DP-SGD implementation, is PyTorch-only. TF-Federated is tightly coupled, has a steep learning curve, and is effectively simulation-only, which would make the real encrypted-transport demonstration harder |
| NVIDIA FLARE | Genuinely strong and production-grade, with real medical deployments, but far heavier: provisioning ceremony, admin APIs and job configuration. Named instead as the migration path for real multi-hospital deployment in future work |
| OpenFL | Credible and worth citing, since the FeTS consortium used it, but a smaller ecosystem than Flower and no built-in advantage here |
| Vision Transformers, medical foundation models | Worse under DP-SGD, worse with limited per-hospital data, far more compute than available, and they abandon the DenseNet121 and CheXNet lineage that gives the work its clinical credibility |
| Blockchain for audit trails | A common addition in student federated-learning projects that adds substantial complexity while providing no guarantee the existing layers do not already provide |
| Full homomorphic encryption | Computationally infeasible at this model size for a prototype. A narrow comparison on the small head remains a possible optional extension |
| Kubernetes now | Docker Compose delivers the entire demonstrative benefit at a fraction of the cost. Kubernetes stays as future work |
| PySyft, CrypTen | Heavy, with significant API churn, and offering nothing beyond what Flower and Opacus already provide |
| Weights and Biases or any cloud tracker | Sends experiment telemetry to a third party, which is the wrong posture for a data-sovereignty project, and requires connectivity |
| Custom cryptographic protocols | Likely to be subtly wrong and impossible for a reviewer to trust. Prohibited by the approved architecture |
| Byzantine and poisoning defences now | Deferred to future work, and not simply additive — see the architectural tension noted in Part III |

---

# PART III — SYSTEM ARCHITECTURE

## 11. Topology

The system is a star arrangement. Every hospital runs an identical local pipeline; the server only aggregates and redistributes, and never sees data.

```
  Hospital A                Hospital B                Hospital C
  +--------------+          +--------------+          +--------------+
  | Local X-rays |          | Local X-rays |          | Local X-rays |  never leave
  |      |       |          |      |       |          |      |       |
  | CLAHE + tv   |          | CLAHE + tv   |          | CLAHE + tv   |  preprocessing
  |      |       |          |      |       |          |      |       |
  | DenseNet121  |          | DenseNet121  |          | DenseNet121  |  frozen backbone
  | (head only)  |          | (head only)  |          | (head only)  |  + trainable head
  |      |       |          |      |       |          |      |       |
  |   DP-SGD     |          |   DP-SGD     |          |   DP-SGD     |  (epsilon, delta)
  |      |       |          |      |       |          |      |       |
  | SecAgg+ mask |          | SecAgg+ mask |          | SecAgg+ mask |  masked update
  +------+-------+          +------+-------+          +------+-------+
         |                         |                         |
         +--- TLS + client auth ---+--- TLS + client auth ---+
                                   |
                                   v
                    +------------------------------+
                    |      Federated Server        |
                    |  SecAgg+ unmask (aggregate)  |
                    |          FedAvg              |
                    |  sees the SUM only, never    |
                    |  any single hospital         |
                    +--------------+---------------+
                                   |
                 global model broadcast; the round repeats
```

**The system invariant:** the only thing that ever crosses a hospital boundary is a small, DP-noised, cryptographically masked parameter update, sent over an authenticated encrypted channel.

## 12. One federated training round

1. The server broadcasts the current global head parameters to the selected hospitals.
2. Each hospital loads its local X-rays. **The images never leave local storage.**
3. Preprocessing runs: OpenCV CLAHE, then torchvision resize, normalize and augment.
4. The trainable head is trained locally using Opacus DP-SGD, with per-sample gradient clipping and calibrated noise. The accountant accumulates the privacy budget.
5. The resulting update is quantized and masked by Flower SecAgg+.
6. The masked update is transmitted over TLS with client authentication.
7. The server unmasks only the aggregate and applies FedAvg.
8. The improved global model is broadcast, and the round repeats until the round budget or stopping criterion is reached.

## 13. The four protection layers

Each layer protects a different asset. They are not interchangeable, and each is independently switchable so that its individual cost can be measured.

| Layer | Protects | Against | Mechanism |
|---|---|---|---|
| Federated Learning | Raw patient images | Anyone outside the hospital | Images never leave local storage |
| Differential Privacy | Information encoded inside an update | Inference or reconstruction from updates | Opacus DP-SGD with (epsilon, delta) |
| Secure Aggregation | An individual hospital's update | The server itself | Flower SecAgg+ masking |
| TLS with client authentication | Messages in transit | Eavesdropping, tampering, impersonation | gRPC TLS plus client identity |

**Threat model in scope:** an honest-but-curious server that follows the protocol but tries to infer information; a passive network adversary; an unregistered party attempting to impersonate a hospital; and hospital collusion up to the SecAgg+ threshold.

**Explicitly out of scope for this phase:** malicious hospitals submitting poisoned updates; collusion above the SecAgg+ threshold; side-channel and physical attacks; and compromise of a hospital's own infrastructure.

## 14. Key architectural decisions

| Ref | Decision |
|---|---|
| ADR-1 | Freeze the DenseNet121 backbone; federate only a small trainable head. The load-bearing decision of the design |
| ADR-2 | Differential Privacy is sample-level DP-SGD with a formal accountant reporting (epsilon, delta) |
| ADR-3 | Use Flower's SecAgg+; custom cryptography is prohibited |
| ADR-4 | TLS must include client authentication |
| ADR-5 | Pin the Flower version exactly; its API has changed significantly across releases |
| ADR-6 | OpenCV is used only for CLAHE, with fixed parameters and a cache |
| ADR-7 | Data is split at the patient level, never the image level, wherever identifiers permit |
| ADR-8 | Simulation is used for measurement; containerised deployment for demonstration. One client implementation serves both |

## 15. Execution modes

The same client and server code runs in two modes, and the client logic is never forked between them.

| Mode | Purpose |
|---|---|
| Simulation | Fast experiment sweeps across seeds, privacy budgets, hospital counts and partition schemes. All reported results come from here |
| Containerised deployment | Real separate processes, real encrypted transport, real client authentication, real Secure Aggregation. The demonstration that the architecture works as described |

## 16. Known architectural tensions

These are genuine conflicts between components, worth stating explicitly in the paper rather than discovering later.

1. **Secure Aggregation and Byzantine detection are directly opposed.** Secure Aggregation exists precisely to stop the server from seeing individual updates, while detecting a malicious hospital requires inspecting individual updates to find outliers. The future-work item is therefore not simply additive; reconciling the two requires specialised techniques.
2. **Differential Privacy trades against explainability and calibration.** Heatmap quality and confidence calibration plausibly degrade as the privacy budget tightens. Privacy and clinical trust are presented as complementary, but they may partly conflict. Measuring this is itself a contribution.
3. **Privacy trades against accuracy.** Objectives 2 and 5 pull in opposite directions, and per-hospital local noise is the most expensive way to spend a privacy budget. Shrinking the federated update under ADR-1 is the primary mitigation.

---

# PART IV — STAGED IMPLEMENTATION PLAN

## Preface

This plan takes the project from an empty repository to a completed research prototype. It implements the architecture recorded in `CLAUDE.md`, which remains the governing source of truth. Where this plan and `CLAUDE.md` disagree, `CLAUDE.md` wins and the conflict must be raised rather than worked around.

**Dataset decision (approved):** Kermany as Hospital A; RSNA Pneumonia Detection Challenge shards as Hospitals B and C.

Two consequences of that decision surfaced while building this plan and should be understood before Stage 3:

1. **RSNA requires a Kaggle account and acceptance of the competition rules** before download. This is a manual action that only the project owner can perform.
2. **The two datasets do not share label semantics.** Kermany is pediatric *pneumonia vs. normal*; RSNA is adult *lung opacity vs. not*, and RSNA's negative class includes cases that are abnormal but not pneumonia. This is genuine label shift stacked on top of domain shift. It is good for realism, requires an explicit decision (Decision Gate DG-2), and must be disclosed as a limitation.

**Legend**

| Tag | Meaning |
|---|---|
| REQ | Required for the core project |
| REC | Recommended |
| OPT | Optional research extension — requires explicit approval |

**Size estimates:** S is approximately half a day; M is one to two days; L is three or more days.

---

# PHASE 0 — Foundation

## Stage 0 — Repository foundation
**Class:** REQ  **Size:** S

**1. Goal.** A navigable, governed repository skeleton before any logic exists.

**2. What we implement.** Directory tree per CLAUDE.md section 13; a README covering purpose, quickstart and reproduction; package init files; the gitignore (already created).

**3. Files created.** `README.md`, `src/**/__init__.py`, `conf/`, `scripts/`, `tests/`, `docker/`, `data/partitions/.gitkeep`.

**4. Dependencies required.** None.

**5. Prerequisites.** CLAUDE.md approved (complete).

**6. Testing and validation.** Tree matches section 13; `git status` shows no ignored-path leakage; no secret is stageable.

**7. Expected output.** An empty but correct skeleton; the first commit candidate.

**8. Risks.** Low. The main hazard is scaffolding empty modules that are never used — keep the tree minimal and grow into it.

**9. Architecture change?** No.

## Stage 1 — Python 3.11 environment and pinned dependencies
**Class:** REQ  **Size:** S  **DECISION GATE DG-1**

**1. Goal.** A reproducible, exactly pinned environment, honouring ADR-5.

**2. What we implement.** Python 3.11 virtual environment; `pyproject.toml` plus a lockfile; verification that every version resolves together before the file is written.

**3. Files created.** `pyproject.toml`, lockfile, `scripts/setup_env.sh`.

**4. Dependencies required — all need approval under CLAUDE.md section 17.3.**

Approved in principle by the technology stack: `torch` and `torchvision` (CUDA build for the RTX 3050), `flwr` (exact pin, must expose SecAgg+ and the DP mods), `opacus`, `opencv-python-headless`, `hydra-core`, `omegaconf`, `mlflow`, `pytest`.

New, not currently in CLAUDE.md section 4: `numpy`, `pandas`, `scikit-learn` (metrics), `pydicom` (RSNA is DICOM), `pillow`, `tqdm`, `matplotlib` (figures), `grad-cam` or `captum`, and optionally `kaggle` (download automation).

**5. Prerequisites.** Stage 0.

**6. Testing and validation.** Clean install from the lockfile into a fresh virtual environment; `torch.cuda.is_available()` returns true; `import flwr, opacus, cv2` all succeed; the pinned Flower version exposes SecAgg+ at the expected import path.

**7. Expected output.** A single command reproduces the environment.

**8. Risks.** This is the highest-probability early failure. Flower, Opacus and torch version incompatibility; CUDA build mismatch; the pinned Flower may not expose SecAgg+ where the documentation claims. Mitigation: verifying the SecAgg+ and DP-mod imports is an acceptance criterion of this stage, not a later discovery.

**9. Architecture change?** Yes. CLAUDE.md section 4 gains roughly eight dependencies. That edit will be proposed separately for approval.

## Stage 2 — Configuration, seeding and tracking
**Class:** REQ  **Size:** M

**1. Goal.** Establish the reproducibility spine (CLAUDE.md section 12) before anything produces a number.

**2. What we implement.** The Hydra configuration tree; a global seeding utility covering torch, numpy, random, deterministic algorithms, DataLoader worker seeds and — critically — the data-partition seed and the client-sampling seed; an MLflow wrapper logging the resolved config, git SHA, metrics and artifacts; structured logging.

**3. Files created.** `conf/config.yaml`, `conf/model/`, `conf/data/`, `conf/federated/`, `conf/privacy/`, `conf/experiment/`, `src/utils/seeding.py`, `src/utils/logging.py`, `src/utils/mlflow_utils.py`.

**4. Dependencies required.** The Stage 1 set.

**5. Prerequisites.** Stage 1.

**6. Testing and validation.** A pytest asserting that the same seed produces bit-identical tensors both within a process and across processes; an MLflow run containing the fully resolved config; a config override demonstrably changing the logged parameters.

**7. Expected output.** Every later stage becomes config-driven and tracked by construction.

**8. Risks.** `torch.use_deterministic_algorithms(True)` raises on certain CUDA kernels, particularly some pooling backward operations. Mitigation: expose it as a config flag, record whenever it is relaxed, and document exactly which operation forced the relaxation.

**9. Architecture change?** No.

---

# PHASE 1 — Data

## Stage 3 — Dataset acquisition and integrity validation
**Class:** REQ  **Size:** M

**1. Goal.** Both datasets on disk, checksum-verified, and never committed.

**2. What we implement.** Download and ingest scripts for Kermany (approximately 1.2 GB, JPEG) and RSNA Stage 2 (approximately 12 GB, DICOM); a SHA-256 manifest; a validation report covering counts, class balance, corrupt files, image size distribution and DICOM tag sanity.

**3. Files created.** `scripts/download_kermany.py`, `scripts/download_rsna.py`, `scripts/validate_datasets.py`, `data/manifests/*.json`.

**4. Dependencies required.** `pydicom`, `pandas`, `pillow`, optionally `kaggle`.

**5. Prerequisites.** Stage 2. Requires the project owner to create a Kaggle account and accept the RSNA competition rules — this cannot be automated on the owner's behalf.

**6. Testing and validation.** Checksums match; expected file counts confirmed; zero unreadable images; a written data-characteristics report committed as an artifact.

**7. Expected output.** Verified local corpora plus reproducible manifests.

**8. Risks.** The Kaggle authentication and rules-acceptance blocker; roughly 13 GB of download; DICOM decoding requires correct pixel handling (rescale slope and intercept, photometric interpretation) — incorrect windowing degrades everything downstream invisibly.

**9. Architecture change?** Yes. CLAUDE.md section 14, pending decision 1, becomes resolved. Proposed separately.

## Stage 4 — Label harmonization and patient-level splitting
**Class:** REQ  **Size:** M  **DECISION GATE DG-2**

**1. Goal.** One consistent binary label across two differently annotated sources, with ADR-7 honoured.

**2. What we implement.** Label mapping (RSNA Target equal to 1 maps to Pneumonia; the treatment of the negative class is DG-2); patient identifier extraction — RSNA carries a clean `patientId`, whereas Kermany is only partially groupable because pneumonia filenames carry a person identifier and normal filenames do not; grouped train, validation and test splitting; committed split manifests.

**3. Files created.** `src/data/labels.py`, `src/data/splitting.py`, `scripts/build_splits.py`, `data/partitions/*.json`.

**4. Dependencies required.** `pandas`, `scikit-learn`.

**5. Prerequisites.** Stage 3.

**6. Testing and validation.** A test asserting zero patient overlap across splits. Splits deterministic given a seed. Class balance reported per split. A test recording Kermany's normal-class grouping limitation explicitly rather than ignoring it silently.

**7. Expected output.** Frozen, committed, reproducible splits.

**8. Risks.** DG-2 is substantive. Including RSNA's "No Lung Opacity / Not Normal" cases as Normal teaches the model that abnormal-but-not-pneumonia equals normal, which is clinically wrong but matches the RSNA challenge framing. Excluding them shrinks and cleans the negative class. This choice materially changes results and must be justified in the paper. Separately, Kermany's normal-class patient leakage is unavoidable and must be disclosed.

**9. Architecture change?** Adds a documented limitation to CLAUDE.md section 15.

## Stage 5 — Hospital partitioning
**Class:** REQ  **Size:** S  **DECISION GATE DG-3**

**1. Goal.** Turn two corpora into N simulated hospitals, both naturally and synthetically heterogeneous.

**2. What we implement.** Natural non-IID assignment (Kermany becomes Hospital A; RSNA is split into patient-disjoint shards B and C); a synthetic Dirichlet partitioner with configurable alpha; per-client statistics reporting.

**3. Files created.** `src/data/partitioning.py`, `conf/data/partition_natural.yaml`, `conf/data/partition_dirichlet.yaml`.

**4. Dependencies required.** None new.

**5. Prerequisites.** Stage 4.

**6. Testing and validation.** Partitions disjoint at patient level; deterministic per seed; an alpha sweep visibly changing per-client class distributions; committed partition indices.

**7. Expected output.** Reproducible hospital assignments for every experiment.

**8. Risks.** Severe client imbalance — Kermany holds roughly 5,900 images against RSNA's roughly 30,000 — will skew FedAvg's sample-weighted average toward RSNA. DG-3 asks whether to balance shard sizes, retain the natural imbalance, or report both. Reporting both is recommended, since the imbalance is itself a realistic finding.

**9. Architecture change?** Resolves CLAUDE.md section 14, pending decision 4.

## Stage 6 — OpenCV CLAHE preprocessing and cache
**Class:** REQ  **Size:** M

**1. Goal.** Implement ADR-6: OpenCV used for CLAHE only, deterministically and cached.

**2. What we implement.** CLAHE with fixed, logged `clipLimit` and `tileGridSize`; explicit BGR to RGB conversion; a disk cache keyed by image identifier and a hash of the CLAHE parameters; a before-and-after visual sample artifact.

**3. Files created.** `src/data/preprocessing.py`, `scripts/build_clahe_cache.py`.

**4. Dependencies required.** `opencv-python-headless`.

**5. Prerequisites.** Stage 5.

**6. Testing and validation.** A test for BGR to RGB correctness using a synthetic asymmetric-colour image — this is the invisible-failure case identified in ADR-6. CLAHE output byte-identical across runs given fixed parameters. Cache hit and miss correctness. A visual inspection artifact logged to MLflow.

**7. Expected output.** A preprocessed cache, with CLAHE removed from the per-epoch hot path.

**8. Risks.** Cache invalidation bugs, such as a stale cache surviving a parameter change — mitigated by hashing parameters into the cache key. The DICOM to 8-bit conversion choice interacts with CLAHE and must be fixed and logged.

**9. Architecture change?** No.

## Stage 7 — torchvision transforms, Dataset and DataLoader
**Class:** REQ  **Size:** S

**1. Goal.** torchvision handles everything except CLAHE.

**2. What we implement.** Training and evaluation transform pipelines (resize to 224, ImageNet normalization, seeded augmentation); a Dataset reading from the CLAHE cache; a DataLoader with seeded workers.

**3. Files created.** `src/data/datasets.py`, `src/data/transforms.py`, `conf/data/transforms.yaml`.

**4. Dependencies required.** `torchvision`.

**5. Prerequisites.** Stage 6.

**6. Testing and validation.** Batch shapes, dtypes and value ranges correct; normalization statistics verified; augmentation reproducible under a fixed seed; no leakage of training transforms into evaluation.

**7. Expected output.** Batches ready for the model.

**8. Risks.** Double normalization; augmentation accidentally applied at evaluation; forgotten worker seeding, which is a classic silent source of nondeterminism.

**9. Architecture change?** No.

---

# PHASE 2 — Model and non-federated baselines

## Stage 8 — DenseNet121 frozen backbone and trainable head
**Class:** REQ  **Size:** M  **CRITICAL GATE DG-4 and DG-6**

**1. Goal.** Build ADR-1 and prove it works with Opacus before anything is built on top of it.

**2. What we implement.** ImageNet-pretrained DenseNet121; the backbone frozen with BatchNorm forced into eval mode with frozen running statistics; a trainable classifier head with deliberately inserted dropout; parameter-count reporting.

**3. Files created.** `src/models/densenet_head.py`, `src/models/freezing.py`, `conf/model/densenet121.yaml`.

**4. Dependencies required.** `torch`, `torchvision`, `opacus`.

**5. Prerequisites.** Stage 7.

**6. Testing and validation — this is the entire point of the stage.** A test that Opacus `ModuleValidator` accepts the model. A test that per-sample gradients compute correctly and only for head parameters. A test that BatchNorm running statistics are unchanged after a training step. A spike confirming one DP-SGD step fits within 4 GB of VRAM at the intended batch size. Trainable parameter count within the expected range of roughly 1e5 to 1e6.

**7. Expected output.** A DP-compatible, federatable model — the load-bearing assumption of the whole project, retired as a risk on the first day of modelling.

**8. Risks.** This is the single largest technical risk in the project. If Opacus rejects the frozen-BatchNorm model, or per-sample gradients still exceed the 4 GB budget, ADR-1's premise fails and the fallback is `ModuleValidator.fix()` with GroupNorm — which discards pretrained BatchNorm statistics, requires more training, and increases VRAM pressure. Performing this validation at Stage 8 rather than Stage 14 is deliberate: failure here is cheap, failure later is not.

**9. Architecture change?** Potentially major. A failure triggers the ADR-1 GroupNorm fallback, which CLAUDE.md requires the owner to approve explicitly. This stage also resolves section 14, pending decision 2, on dropout placement.

## Stage 9 — Frozen-backbone feature cache
**Class:** REC  **Size:** M  **DECISION GATE DG-5**

**1. Goal.** Make the full ablation campaign feasible on a 4 GB laptop GPU.

**2. What we implement.** Because the backbone is frozen, its 1024-dimensional output is precomputed once per image and the head is trained on cached features. Optionally K augmented views are cached per image to retain augmentation. Full-image forward passes remain in use for inference and Grad-CAM.

**3. Files created.** `src/data/feature_cache.py`, `scripts/build_feature_cache.py`.

**4. Dependencies required.** None new.

**5. Prerequisites.** Stage 8.

**6. Testing and validation.** Cached features match a live forward pass within floating-point tolerance; head training on the cache matches training on images under the same seed with augmentation disabled; the measured speedup is recorded.

**7. Expected output.** A likely one-to-two order-of-magnitude speedup in head training, which is what makes six configurations across multiple epsilon values and three or more seeds realistic on this hardware. It also nearly eliminates the DP-SGD VRAM risk.

**8. Risks.** DG-5: caching without augmentation loses regularization on small data, whereas caching K views costs K times the storage — still modest, roughly 600 MB for five views across 36,000 images. The cache must be invalidated on any backbone or preprocessing change. If ADR-1's GroupNorm fallback is adopted, this stage becomes invalid because the backbone is no longer frozen.

**9. Architecture change?** No architectural change, but a significant implementation strategy worth recording in CLAUDE.md if adopted.

## Stage 10 — Evaluation and metrics module
**Class:** REQ  **Size:** M

**1. Goal.** One trusted metrics implementation, built before any baseline produces numbers.

**2. What we implement.** AUROC as the primary metric, plus AUPRC, sensitivity at fixed specificity, specificity, F1, balanced accuracy and confusion matrix; bootstrap 95 percent confidence intervals; multi-seed aggregation reporting mean and standard deviation; results serialization and MLflow logging.

**3. Files created.** `src/evaluation/metrics.py`, `src/evaluation/bootstrap.py`, `src/evaluation/reporting.py`.

**4. Dependencies required.** `scikit-learn`, `numpy`.

**5. Prerequisites.** Stage 2.

**6. Testing and validation.** Metrics match scikit-learn references on synthetic data; degenerate cases such as single-class, all-correct and all-wrong are handled; bootstrap confidence intervals reproducible under seed; a known-input regression test.

**7. Expected output.** Every subsequent stage reports comparably.

**8. Risks.** Building this module after the baselines is the classic mistake — results get recomputed and tables silently disagree. Threshold-dependent metrics such as F1 and sensitivity require an explicit, fixed threshold policy.

**9. Architecture change?** No.

## Stage 11 — Local single-hospital baseline
**Class:** REQ  **Size:** M  **Ablation row 1**

**1. Goal.** Establish the floor — what one hospital achieves alone — and the first end-to-end training loop.

**2. What we implement.** A non-federated training loop for the head; explicit class-imbalance handling by weighted loss or sampling, with the choice recorded; checkpointing; independent runs for Hospitals A, B and C.

**3. Files created.** `src/training/trainer.py`, `scripts/train_local.py`, `conf/experiment/local.yaml`.

**4. Dependencies required.** None new.

**5. Prerequisites.** Stages 8 and 10, plus Stage 9 if adopted.

**6. Testing and validation.** Loss decreases; AUROC meaningfully above 0.5; three-seed variance reported; overfitting checked against validation; results present in MLflow.

**7. Expected output.** Ablation row 1 for each hospital — the comparison the federated result must beat.

**8. Risks.** Frozen-backbone accuracy may disappoint; this stage reveals whether ADR-1's accuracy cost is acceptable or whether the GroupNorm fallback conversation is required. Kermany's small size makes its local model high-variance.

**9. Architecture change?** Possibly triggers the ADR-1 fallback discussion.

## Stage 12 — Centralized pooled baseline
**Class:** REQ  **Size:** S  **Ablation row 2**

**1. Goal.** Establish the privacy-free ceiling.

**2. What we implement.** Training on all hospitals' data pooled, using an identical model, optimizer and schedule to Stage 11.

**3. Files created.** `scripts/train_centralized.py`, `conf/experiment/centralized.yaml`.

**4. Dependencies required.** None new.

**5. Prerequisites.** Stage 11.

**6. Testing and validation.** Centralized performance at least matches each local model — if it does not, investigate before proceeding, as this usually indicates a label-harmonization or partitioning defect. Identical test set to all other rows. Three seeds.

**7. Expected output.** The upper bound against which every privacy layer is measured.

**8. Risks.** This row must use exactly the same evaluation set and protocol as rows 1 and 3 through 6, or the entire ablation table is meaningless. Note that this row deliberately violates the project's privacy premise and exists solely as a reference point.

**9. Architecture change?** No.

---

# PHASE 3 — Federated core

Strict ordering applies throughout this phase: no federated learning before a working local model; no differential privacy before working federated learning; no secure aggregation before a verified DP-free federated workflow.

## Stage 13 — Flower FedAvg in simulation
**Class:** REQ  **Size:** L  **Ablation row 3**

**1. Goal.** Working federated training with no privacy layers, isolating federated-learning correctness.

**2. What we implement.** A `ClientApp` performing fit and evaluate over the local partition; a `ServerApp` running FedAvg; parameter serialization and deserialization for head-only updates; per-round metric aggregation; round orchestration. Written once so it runs under both execution engines, per ADR-8.

**3. Files created.** `src/federated/client_app.py`, `src/federated/server_app.py`, `src/federated/strategy.py`, `src/federated/serialization.py`, `conf/federated/fedavg.yaml`, `scripts/run_simulation.py`.

**4. Dependencies required.** `flwr` at the pinned version.

**5. Prerequisites.** Stage 11. Stage 12 recommended, since it supplies the target.

**6. Testing and validation.** A two-client, single-round smoke test running end to end. A test that with a single client, FedAvg output equals local training — this isolates aggregation bugs. Lossless parameter round-trip. The global model improves across rounds. The headline check: does the federated model beat the best single-hospital local model?

**7. Expected output.** Ablation row 3 — the project's core value proposition, demonstrated.

**8. Risks.** Flower API churn per ADR-5 — write against the pinned version's documentation only. MLflow logging from inside simulation actors is a known friction point involving concurrent and nested runs. Non-IID convergence may be poor or unstable. Client imbalance skews the weighted average, per DG-3.

**9. Architecture change?** No, provided the pinned Flower behaves as documented.

## Stage 14 — Differential Privacy with formal accounting
**Class:** REQ  **Size:** L  **DECISION GATE DG-7**  **Ablation row 5**

**1. Goal.** Implement ADR-2: sample-level DP-SGD with a genuine accountant and reported epsilon and delta.

**2. What we implement.** The Opacus `PrivacyEngine` inside the client fit loop; per-sample clipping and calibrated Gaussian noise; `BatchMemoryManager` for the 4 GB budget; an accountant accumulating budget across rounds rather than merely across epochs; epsilon reported per configuration; differential privacy as a config-switchable layer.

**3. Files created.** `src/privacy/dp.py`, `src/privacy/accounting.py`, `conf/privacy/dp_*.yaml`.

**4. Dependencies required.** `opacus`.

**5. Prerequisites.** Stage 13 verified working, and Stage 8's Opacus compatibility proven.

**6. Testing and validation.** A test that gradients are genuinely clipped to the configured norm. A test that noise scales correctly with the multiplier. A test that the accountant consumes budget monotonically across rounds — a static epsilon across rounds is the classic silent bug here. Decreasing epsilon should monotonically decrease accuracy. Setting sigma to zero should reproduce Stage 13.

**7. Expected output.** Ablation row 5 — the privacy-utility curve across epsilon.

**8. Risks.** DG-7 covers the choice of target epsilon values, delta relative to dataset size, and clipping norm; these determine whether the results are publishable or trivially weak. Cross-round accounting is easy to get wrong. Utility may collapse at tight epsilon — this is expected and is itself the finding, but ADR-2's local-DP caveat means it must be framed honestly. Opacus memory pressure becomes a factor if Stage 9 was not adopted.

**9. Architecture change?** Resolves CLAUDE.md section 14, pending decision 3.

## Stage 15 — Secure Aggregation using Flower SecAgg+
**Class:** REQ  **Size:** L  **DECISION GATE DG-8**  **Ablation row 4**

**1. Goal.** Implement ADR-3 — the server sees only the aggregate — using Flower's implementation and never custom cryptography.

**2. What we implement.** The SecAgg+ workflow on the server and the corresponding client mod; update quantization for the finite field; dropout-resilience configuration; a config-switchable layer.

**3. Files created.** `src/privacy/secagg.py`, `conf/privacy/secagg.yaml`.

**4. Dependencies required.** Flower SecAgg+, verified available during Stage 1.

**5. Prerequisites.** Stage 13 verified. Technically independent of Stage 14, but implemented afterwards so the DP-noise and quantization interaction becomes measurable immediately.

**6. Testing and validation.** A test that masks cancel exactly — the unmasked aggregate equals plain FedAvg within tolerance. This is the test that makes the Secure Aggregation claim defensible rather than merely asserted. A test that quantization round-trip error is bounded. A client-dropout scenario handled correctly. The accuracy delta against row 3 measured and reported.

**7. Expected output.** Ablation row 4 — the measured cost of quantization and masking.

**8. Risks.** DG-8 concerns quantization bit-width, which trades accuracy against field size; too coarse a setting interacts badly with DP noise, which is a real and reportable interaction. SecAgg+ introduces multi-stage round communication that complicates the round loop. Custom cryptography is absolutely prohibited — if Flower's implementation does not fit, stop and escalate rather than hand-rolling, per ADR-3.

**9. Architecture change?** No, unless SecAgg+ proves unusable in the pinned version, which would be a significant escalation.

## Stage 16 — TLS and client authentication
**Class:** REQ  **Size:** M

**1. Goal.** Implement ADR-4 and make the impersonation-resistance claim actually true.

**2. What we implement.** A scripted local certificate authority with per-hospital certificates and keys; a TLS-enabled Flower server; client authentication by mutual TLS or Flower node authentication, with the mechanism verified against the pinned version; explicit configuration of the gRPC maximum message length.

**3. Files created.** `scripts/generate_certs.sh`, `src/federated/security.py`, `conf/federated/tls.yaml`.

**4. Dependencies required.** None new; uses the OpenSSL command-line tool.

**5. Prerequisites.** Stage 13.

**6. Testing and validation.** A negative test confirming that an unregistered or unauthenticated client is rejected — without this, client authentication is decorative. Confirmation that traffic is encrypted. Verification that `certs/` is gitignored and no key is stageable. Confirmation that the configured message length exceeds the actual update size.

**7. Expected output.** An authenticated, encrypted channel that partially delivers what the source presentation deferred to future work.

**8. Risks.** Flower's client-authentication mechanism has changed across releases and may differ from the first documentation encountered. Certificate paths and permissions differ between host and container, which affects Stage 17. Accidentally committing a key is a serious hazard — mitigated by the gitignore, but must be verified before the first commit touching `certs/`.

**9. Architecture change?** Possibly refines ADR-4's stated mechanism once verified.

## Stage 17 — Docker Compose multi-client deployment
**Class:** REQ  **Size:** M  **DECISION GATE DG-9**

**1. Goal.** Produce the demonstration artifact: separate containers, real gRPC, real TLS, real SecAgg+, and genuine per-hospital data isolation.

**2. What we implement.** Client and server images; a compose file defining one server and three hospital containers, each with its own certificate and its own mounted data shard; a run script.

**3. Files created.** `docker/Dockerfile.client`, `docker/Dockerfile.server`, `docker/docker-compose.yml`, `scripts/run_deployment.sh`.

**4. Dependencies required.** Docker and Docker Compose.

**5. Prerequisites.** Stages 13 through 16 all working in simulation.

**6. Testing and validation.** `docker compose up` completes federated rounds end to end with DP, SecAgg and TLS all active; each container can access only its own data; the demonstration is reproducible from a clean checkout.

**7. Expected output.** The single most convincing demonstration available — federation that is not merely three processes on one laptop.

**8. Risks.** DG-9: GPU access inside Docker requires the NVIDIA container toolkit, and three GPU containers will not fit within 4 GB. The recommendation is that the demonstration runs CPU-only with few rounds, since it is a demonstration rather than a measurement — measurements come from simulation per ADR-8. Image size with a CUDA torch build is large. Certificate paths and networking differ from host runs.

**9. Architecture change?** No; it clarifies the division of labour described in CLAUDE.md section 3.3.

---

# PHASE 4 — Clinical trust layer

## Stage 18 — Grad-CAM explainability
**Class:** REQ  **Size:** M

**1. Goal.** Deliver the explanation half of objective 6.

**2. What we implement.** Grad-CAM applied to the global model targeting the final dense block or `features.norm5`; heatmap overlay rendering; batch generation across true positive, false positive, true negative and false negative cases; client-side execution so that no image moves.

**3. Files created.** `src/explain/gradcam.py`, `scripts/generate_explanations.py`.

**4. Dependencies required.** `grad-cam` or `captum`.

**5. Prerequisites.** A trained global model, so Stage 13 onwards.

**6. Testing and validation.** Heatmaps are non-degenerate rather than uniform or empty; the target layer resolves correctly; overlays render for both classes; a sanity check that pneumonia-positive cases highlight lung fields rather than image borders or text markers.

**7. Expected output.** Explanation artifacts logged to MLflow, and figures for the paper.

**8. Risks.** With a frozen backbone, gradients flow only through the head. Whether Grad-CAM still produces meaningful maps at the chosen target layer must be verified, not assumed — this is a real interaction between ADR-1 and Grad-CAM. Separately, models sometimes latch onto scanner artifacts or text overlays, a known chest X-ray failure mode worth checking for. If the GroupNorm fallback is adopted, the target layer name changes.

**9. Architecture change?** Possibly a note in CLAUDE.md section 9 if the target layer must move.

## Stage 19 — Monte Carlo Dropout and deferral
**Class:** REQ  **Size:** M  **DECISION GATE DG-10**

**1. Goal.** Deliver the confidence half of objective 6, with a working human-in-the-loop path.

**2. What we implement.** T stochastic forward passes with dropout active at inference; an uncertainty metric such as predictive entropy or variance; an actual deferral path flagging low-confidence cases for review; configurable T and threshold.

**3. Files created.** `src/uncertainty/mc_dropout.py`, `src/uncertainty/deferral.py`, `conf/experiment/uncertainty.yaml`.

**4. Dependencies required.** None new.

**5. Prerequisites.** Stage 8 with dropout inserted, and Stage 13.

**6. Testing and validation.** A test that dropout is genuinely active at inference, confirmed by T passes producing differing outputs — if they do not, MC Dropout is silently doing nothing, which is the most common bug in this area. Uncertainty should be higher on misclassified cases than correct ones. The deferral rate should respond to the threshold. Accuracy on retained cases should exceed overall accuracy.

**7. Expected output.** Confidence-aware predictions plus a deferral mechanism that exists in code rather than only in prose.

**8. Risks.** DG-10: the deferral threshold is a clinical policy choice rather than a tuning parameter and requires explicit justification. Head-only dropout may yield weak, poorly separated uncertainty, which CLAUDE.md section 10 already acknowledges. Without calibration metrics, delivered by optional extension OPT-1, the confidence numbers remain unvalidated.

**9. Architecture change?** Confirms the consequences of section 14, pending decision 2.

---

# PHASE 5 — Measurement and delivery

## Stage 20 — Overhead instrumentation
**Class:** REQ  **Size:** S

**1. Goal.** Deliver the overhead reporting required by CLAUDE.md section 11.2, treated as a first-class result rather than an afterthought.

**2. What we implement.** Bytes transmitted per client per round; total communication per experiment; wall-clock timing per round and per phase; client compute and peak memory; overhead attributed separately to differential privacy and to secure aggregation.

**3. Files created.** `src/evaluation/overhead.py`, plus instrumentation hooks within `src/federated/`.

**4. Dependencies required.** None new.

**5. Prerequisites.** Stages 13 through 15.

**6. Testing and validation.** Measured payload matches the theoretical parameter-count size; SecAgg's multi-stage overhead is visible; measurements are stable across repeated runs.

**7. Expected output.** The overhead columns of the results table, directly supporting the integration-cost claim.

**8. Risks.** Instrumenting inside Flower's transport requires care with the pinned version's hooks. Timing on a shared laptop is noisy, so medians over repeats should be reported.

**9. Architecture change?** No.

## Stage 21 — Full ablation campaign
**Class:** REQ  **Size:** L

**1. Goal.** Produce the results table, which is the paper's core deliverable.

**2. What we implement.** Experiment configurations for all six ablation rows; the epsilon sweep; three or more seeds per configuration; a batch runner; results aggregation into publication-ready tables and figures.

**3. Files created.** `conf/experiment/ablation_*.yaml`, `scripts/run_ablation.py`, `src/evaluation/tables.py`.

**4. Dependencies required.** `matplotlib`.

**5. Prerequisites.** Stages 11 through 17 and Stage 20 — every layer must be verified individually before the campaign runs.

**6. Testing and validation.** All rows share one evaluation protocol and test set; mean and standard deviation reported over seeds; bootstrap confidence intervals computed; every number traceable to a config hash, seed and MLflow run.

**7. Expected output.** The complete ablation table, the privacy-utility curve, and the overhead comparison.

**8. Risks.** Compute time is the binding constraint on a 4 GB laptop, and Stage 9's feature cache is what makes the campaign tractable. A defect discovered mid-campaign invalidates completed runs, so every prior stage must be genuinely validated first. Seed variance may exceed the effect size between adjacent rows, in which case more seeds are required.

**9. Architecture change?** No.

## Stage 22 — Test suite hardening
**Class:** REQ  **Size:** M

**1. Goal.** Consolidate CLAUDE.md section 11.3 into a suite that runs green in a single command.

**2. What we implement.** Completion of the required tests — patient overlap, BGR to RGB, CLAHE determinism, DP clipping, noise and accounting, SecAgg mask cancellation, and the two-client smoke test; fixtures using tiny synthetic data; optionally a GitHub Actions workflow running CPU-only fast tests.

**3. Files created.** `tests/**`, `tests/conftest.py`, optionally `.github/workflows/tests.yml`.

**4. Dependencies required.** `pytest`, optionally `pytest-cov`.

**5. Prerequisites.** All implementation stages.

**6. Testing and validation.** The full suite passes; fast tests run without a GPU or real data; coverage focuses specifically on privacy-critical paths.

**7. Expected output.** A defensible correctness claim for the privacy and security layers.

**8. Risks.** Tests written only at the end tend to codify existing bugs. Most of these tests should be written during their own stages, with this stage consolidating and filling gaps.

**9. Architecture change?** No.

## Stage 23 — Documentation and reproducibility package
**Class:** REQ  **Size:** M

**1. Goal.** Enable a third party to reproduce every number.

**2. What we implement.** A full README covering setup, data acquisition, reproduction of each ablation row and running the Docker demonstration; results documentation; the threat model write-up from CLAUDE.md section 6; the limitations from section 15; an architecture diagram; and a proposed CLAUDE.md status update for approval.

**3. Files created.** `README.md`, `docs/threat_model.md`, `docs/results.md`, `docs/reproducibility.md`, `docs/figures/`.

**4. Dependencies required.** None.

**5. Prerequisites.** Stage 21.

**6. Testing and validation.** A clean-checkout dry run following only the README; every reported number traceable to a config, seed and MLflow run.

**7. Expected output.** The submission-ready research artifact.

**8. Risks.** Documentation drift from code. The Kaggle-gated RSNA download cannot be fully automated for a third party and must be documented as a manual step.

**9. Architecture change?** Yes. CLAUDE.md section 14 status and section 15 limitations require updating. Proposed for approval, not applied unilaterally.

---

# PHASE 6 — Optional research extensions

None of these begin without explicit approval, per CLAUDE.md section 16.1.

## OPT-1 — Calibration metrics
**Class:** OPT  **Size:** S — highest value per unit of effort

Expected Calibration Error, Brier score, reliability diagrams and risk-coverage curves, together with an analysis of how calibration degrades as epsilon tightens. This is pure post-hoc analysis over existing predictions and is therefore very cheap. Without it, the confidence-aware claim remains unvalidated, since MC Dropout confidence may be badly calibrated. The risk is that it may reveal the confidence scores are poor — which is a finding worth reporting honestly. Prerequisites: Stages 19 and 21.

## OPT-2 — Empirical privacy attacks
**Class:** OPT  **Size:** L — highest academic payoff

Membership inference and optionally gradient inversion against model updates, run with differential privacy disabled and at several epsilon values. This converts the project's central premise from a literature citation into a measured result and produces the most compelling figure available. Risks: attack implementation is finicky; a weak attack proves nothing, so a negative result must not be over-claimed as evidence that DP works; gradient inversion is easier against the small head, which cuts both ways. Prerequisites: Stages 14 and 21.

## OPT-3 — Quantitative Grad-CAM evaluation
**Class:** OPT  **Size:** M

Pointing game or intersection-over-union evaluation against RSNA bounding boxes, which are available thanks to the dataset decision, plus an analysis of whether explanation quality degrades under differential privacy and federated averaging. This converts heatmaps into a measured result. The risk is that only RSNA carries boxes, so Kermany is excluded from this analysis. Prerequisites: Stages 18 and 21.

## OPT-4 — Stronger uncertainty methods
**Class:** OPT  **Size:** M

Snapshot ensembling across recent federated rounds, which is nearly free since it requires no extra training, or conformal prediction providing distribution-free coverage guarantees on the deferral decision. Conformal prediction is the more interesting contribution and maps directly onto the human-in-the-loop objective. The risk is that federated conformal calibration under non-IID clients is genuinely subtle. Prerequisites: Stages 19 and 21.

---

## 1. Final stage roadmap

| Number | Stage | Class | Size | Ablation row |
|---|---|---|---|---|
| 0 | Repository foundation | REQ | S | — |
| 1 | Environment and pinned dependencies | REQ | S | — |
| 2 | Config, seeding, MLflow | REQ | M | — |
| 3 | Dataset acquisition and validation | REQ | M | — |
| 4 | Label harmonization and patient-level splits | REQ | M | — |
| 5 | Hospital partitioning | REQ | S | — |
| 6 | CLAHE preprocessing and cache | REQ | M | — |
| 7 | Transforms, Dataset, DataLoader | REQ | S | — |
| 8 | DenseNet121 frozen backbone and head | REQ | M | — |
| 9 | Feature cache | REC | M | — |
| 10 | Evaluation and metrics module | REQ | M | — |
| 11 | Local single-hospital baseline | REQ | M | row 1 |
| 12 | Centralized baseline | REQ | S | row 2 |
| 13 | Flower FedAvg in simulation | REQ | L | row 3 |
| 14 | Differential Privacy and accounting | REQ | L | row 5 |
| 15 | Secure Aggregation (SecAgg+) | REQ | L | row 4 |
| 16 | TLS and client authentication | REQ | M | — |
| 17 | Docker Compose deployment | REQ | M | row 6 |
| 18 | Grad-CAM | REQ | M | — |
| 19 | Monte Carlo Dropout and deferral | REQ | M | — |
| 20 | Overhead instrumentation | REQ | S | — |
| 21 | Full ablation campaign | REQ | L | all |
| 22 | Test suite hardening | REQ | M | — |
| 23 | Documentation and reproducibility | REQ | M | — |
| OPT-1 | Calibration metrics | OPT | S | — |
| OPT-2 | Empirical privacy attacks | OPT | L | — |
| OPT-3 | Quantitative Grad-CAM | OPT | M | — |
| OPT-4 | Stronger uncertainty | OPT | M | — |

## 2. Dependency order between stages

```
0 -> 1 -> 2 -+-> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -+-> 9  (optional accelerator)
             |                                |
             +----------> 10 ----------------+-> 11 -> 12
                                                        |
                                                        v
                                                       13   (FedAvg - the FL gate)
                                                        |
                   +-------------+---------------+------+------+
                   v             v               v             v
                  14 (DP)       15 (SecAgg)     16 (TLS)      18 (Grad-CAM)
                   |             |               |             |
                   +-------------+------+--------+             v
                                        v                     19 (MC Dropout)
                                       17 (Docker)             |
                                        |                      |
                                        +-------+--------------+
                                                v
                                               20 (overhead)
                                                v
                                               21 (ablation campaign)
                                                v
                                            22 -> 23
                                                v
                                    OPT-1 . OPT-2 . OPT-3 . OPT-4
```

**Hard ordering constraints**

- No federated learning before a working local model: Stage 13 requires Stage 11.
- No differential privacy or secure aggregation before FedAvg is verified DP-free: Stages 14 and 15 require Stage 13.
- No deployment before every layer works in simulation: Stage 17 requires Stages 13 through 16.
- No ablation campaign before every layer is individually validated: Stage 21 requires Stages 11 through 17 and Stage 20.
- Stage 8 must retire the ADR-1 risk before anything is built on top of it.
- Stages 14, 15, 16 and 18 are mutually independent and could be reordered; the sequence above deliberately front-loads the riskiest, which is differential privacy.

## 3. Decision gates

| Gate | Stage | Decision | Why it belongs to the owner |
|---|---|---|---|
| DG-1 | 1 | Dependency set and exact pins | Section 17.3 requires approval; adds roughly eight dependencies to CLAUDE.md section 4 |
| DG-2 | 4 | RSNA negative-class semantics | Materially changes labels, results and clinical claims |
| DG-3 | 5 | Client imbalance: balance shards, keep natural, or report both | Changes the headline FedAvg result |
| DG-4 | 8 | ADR-1 validated, or invoke the GroupNorm fallback | The load-bearing architectural assumption |
| DG-5 | 9 | Feature caching and augmentation strategy | Determines whether the full campaign is feasible on this hardware |
| DG-6 | 8 | Dropout placement | Trades MC Dropout quality against pretrained-feature integrity |
| DG-7 | 14 | Target epsilon values, delta, clipping norm | Determines whether the privacy claims are publishable |
| DG-8 | 15 | Quantization bit-width | Trades accuracy against SecAgg field size; interacts with DP noise |
| DG-9 | 17 | Demonstration scope: CPU-only, round count | A hardware-constrained scoping choice |
| DG-10 | 19 | Deferral threshold policy | A clinical policy decision, not a hyperparameter |
| DG-11 | Phase 6 | Which optional extensions to run | Scope and timeline |

In addition, the standing gates from CLAUDE.md apply throughout: every dependency change, every CLAUDE.md edit, and every push to GitHub.

## 4. Expected final repository structure

```
.
|-- CLAUDE.md                          living architecture doc (approval-gated)
|-- README.md                          setup, reproduce, demo
|-- pyproject.toml + lockfile          exact pins (single source)
|-- .gitignore                         created
|-- Review_1_Privacy_Preserving_FL_Diagnosis.pptx
|-- conf/
|   |-- config.yaml
|   |-- model/densenet121.yaml
|   |-- data/{kermany,rsna,partition_natural,partition_dirichlet,transforms}.yaml
|   |-- federated/{fedavg,tls}.yaml
|   |-- privacy/{dp_eps1,dp_eps3,dp_eps8,secagg}.yaml
|   `-- experiment/{local,centralized,fedavg,secagg,dp,full,ablation_*}.yaml
|-- src/
|   |-- data/          labels, splitting, partitioning, preprocessing, datasets,
|   |                  transforms, feature_cache
|   |-- models/        densenet_head, freezing, dropout
|   |-- privacy/       dp, accounting, secagg
|   |-- federated/     client_app, server_app, strategy, serialization, security
|   |-- explain/       gradcam
|   |-- uncertainty/   mc_dropout, deferral
|   |-- training/      trainer
|   |-- evaluation/    metrics, bootstrap, overhead, reporting, tables
|   `-- utils/         seeding, logging, mlflow_utils
|-- scripts/           download_*, validate_datasets, build_splits, build_clahe_cache,
|                      build_feature_cache, train_local, train_centralized,
|                      run_simulation, run_deployment, run_ablation,
|                      generate_certs.sh, generate_explanations
|-- tests/             data, model, privacy (dp + secagg), federated, evaluation
|-- docker/            Dockerfile.client, Dockerfile.server, docker-compose.yml
|-- docs/              threat_model, results, reproducibility, figures/
|-- data/
|   |-- partitions/    committed split manifests (JSON)
|   |-- manifests/     committed checksums
|   `-- raw/ processed/ cache/    gitignored
|-- certs/             gitignored
`-- mlruns/            gitignored
```

## 5. Recommended implementation sequence

The stages group into review batches, each ending at a natural approval point.

| Batch | Stages | Outcome | Review focus |
|---|---|---|---|
| A — Skeleton | 0, 1, 2 | Reproducible environment plus config and seeding spine | DG-1 dependency pins |
| B — Data | 3, 4, 5 | Verified corpora, patient-level splits, hospital partitions | DG-2 and DG-3: label semantics and imbalance |
| C — Pipeline | 6, 7 | Preprocessing and loaders, with BGR and CLAHE tests green | Preprocessing correctness |
| D — Model | 8, 9 | ADR-1 proven DP-compatible and VRAM-feasible | DG-4, DG-5, DG-6 — the critical gate |
| E — Baselines | 10, 11, 12 | Ablation rows 1 and 2; accuracy floor and ceiling known | Is frozen-backbone accuracy acceptable? |
| F — Federated core | 13 | Ablation row 3; does federation beat local-only? | The project's core claim |
| G — Privacy layers | 14, 15 | Ablation rows 4 and 5 | DG-7 and DG-8: epsilon values and quantization |
| H — Security and demo | 16, 17 | Authenticated TLS; the Docker Compose demonstration | DG-9 |
| I — Clinical trust | 18, 19 | Grad-CAM and MC Dropout with a working deferral path | DG-10 |
| J — Results | 20, 21 | The complete ablation table and figures | Statistical rigour, seed variance |
| K — Delivery | 22, 23 | Green test suite, reproducible artifact, documentation | CLAUDE.md status update proposal |
| L — Extensions | OPT-1 through OPT-4 | Paper strengthening | DG-11 |

**Prioritization if time compresses.** Batches A through F are non-negotiable, since they establish that federated learning works at all. Batches G and H deliver the privacy and security contribution. Batch I is required by the stated objectives. Batches J and K make the work publishable. Within Phase 6, OPT-1 is the cheapest and OPT-2 carries the highest academic payoff; if only one extension runs, OPT-2 is the stronger choice.

**Critical path.** Stages 0, 1, 2, 3, 4, 5, 6, 7, **8**, 11, **13**, 14, 15, 21. Stages 8 and 13 are the two make-or-break points; everything else is either preparation for them or measurement built on top of them.

---

# Pending CLAUDE.md updates

Under CLAUDE.md section 17.1, two changes are already warranted and await approval. Neither has been applied.

1. **Section 14, pending decision 1** — the dataset strategy is now resolved to Kermany as Hospital A and RSNA as Hospitals B and C. This would move from pending and blocking to a recorded decision, with the Kermany normal-class patient-grouping limitation added to section 15.
2. **Section 4** — the dependency table requires roughly eight additions: `numpy`, `pandas`, `scikit-learn`, `pydicom`, `pillow`, `tqdm`, `matplotlib`, `grad-cam` or `captum`, and optionally `kaggle`. This is best proposed together with DG-1 at Stage 1, so that pins and the documentation edit are approved in a single pass.

---

*End of plan. No implementation has begun; no dependency has been installed; no dataset has been downloaded; nothing has been committed or pushed.*
