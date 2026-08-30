# CLAUDE.md

> **This file is the project's living context and architecture document.**
>
> **Do NOT modify this file automatically during development.**
> If a change to the architecture, requirements, dependencies, project structure, or an
> important technical decision would require updating this document, then:
> 1. State what needs to change.
> 2. Explain why.
> 3. Show the proposed change.
> 4. Ask for explicit permission.
> 5. Only edit this file after approval.
>
> Treat every statement below as the approved source of truth until the owner approves a change.

---

## 1. Project Overview

**Privacy-Preserving Medical Diagnosis using Federated Learning**

An academic research prototype for **pneumonia detection from chest X-ray images** in which
multiple simulated hospitals collaboratively train a shared model **without any hospital ever
transmitting patient images** to a central server or to each other.

| | |
|---|---|
| Course | B.Tech BCSE497J — Project I, SCOPE |
| Team | Uddhav Sethi (23BKT0092), Chirag Gadhyan (23BDS0265), Ishaan S Shrivastav (23BCI0130) |
| Faculty guide | Dr. Adaline Suji |
| Alignment | UN Sustainable Development Goal 3 — Good Health & Well-Being |
| Target outcome | Scopus-indexed conference paper |
| Source of truth for scope | `Review_1_Privacy_Preserving_FL_Diagnosis.pptx` (do not modify) |
| Repository | `UddhavSethi/privacy-preserving-medical-diagnosis` |

### Nature of the contribution

The novelty of this project is **integration, not algorithmic invention**. DenseNet121,
DP-SGD, Secure Aggregation, Grad-CAM and Monte Carlo Dropout are each individually
well-studied. The contribution is demonstrating that they compose into a single coherent,
reproducible, end-to-end medical diagnosis pipeline — **and quantifying what that composition
costs** in accuracy, calibration, explanation quality, communication and compute.

Because the claim is integration, **the ablation table is the paper** (see §11). Any work that
does not either build the pipeline or measure the cost of a layer in it is out of scope by default.

### Problem being solved

1. **Data cannot be centralized.** Conventional chest X-ray classifiers require pooling images
   on one server, which hospitals cannot do legally or ethically. Most published work assumes
   this problem away rather than solving it.
2. **Model updates are not automatically safe.** Where federated learning is used, shared
   updates are commonly treated as safe by default, though they can leak information about
   training images.
3. **Clinical trust is missing.** Black-box predictions without explanation or calibrated
   confidence are not actionable in a clinical workflow.

### Research gap

No single published framework combines Federated Learning + Differential Privacy + Secure
Aggregation + TLS + explainability + uncertainty into one working pneumonia-detection
pipeline. This project fills that gap.

---

## 2. Objectives

1. Build a federated learning pipeline where hospitals train locally and share only model
   updates — never patient images.
2. Apply **Differential Privacy** with a formal mechanism and accountant so patient
   information cannot be reverse-engineered from updates. Privacy **must** be reported as
   (epsilon, delta).
3. Add **Secure Aggregation** so even the central server cannot observe any single hospital's
   update — only the aggregate.
4. Secure all communication with **TLS plus client authentication**, preventing eavesdropping,
   tampering and impersonation.
5. Achieve strong pneumonia-detection performance with **DenseNet121 transfer learning**
   despite limited per-hospital data.
6. Make the model **explainable (Grad-CAM)** and **confidence-aware (Monte Carlo Dropout)**,
   with low-confidence cases deferred for human review.
7. Produce a **reproducible** artifact: seeded, configuration-driven, tracked, and testable.

### Prioritization order

When requirements conflict, resolve in this order:

**privacy → security → correctness → reproducibility → academic credibility → explainability →
uncertainty estimation → maintainability → raw accuracy**

Accuracy is deliberately last. A model that scores higher by weakening a privacy guarantee,
leaking patient identity across a split, or becoming irreproducible is a regression, not an
improvement.

---

## 3. Architecture

### 3.1 System topology

Star / hub-and-spoke federation. Every hospital runs an **identical local pipeline**; the server
only aggregates and redistributes.

```
  Hospital A                Hospital B                Hospital C
  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
  │ Local X-rays │          │ Local X-rays │          │ Local X-rays │   <- never leave
  │      ↓       │          │      ↓       │          │      ↓       │
  │ CLAHE + tv   │          │ CLAHE + tv   │          │ CLAHE + tv   │   preprocessing
  │      ↓       │          │      ↓       │          │      ↓       │
  │ DenseNet121  │          │ DenseNet121  │          │ DenseNet121  │   frozen backbone
  │ (head only)  │          │ (head only)  │          │ (head only)  │   + trainable head
  │      ↓       │          │      ↓       │          │      ↓       │
  │   DP-SGD     │          │   DP-SGD     │          │   DP-SGD     │   (epsilon, delta)
  │      ↓       │          │      ↓       │          │      ↓       │
  │ SecAgg+ mask │          │ SecAgg+ mask │          │ SecAgg+ mask │   masked update
  └──────┬───────┘          └──────┬───────┘          └──────┬───────┘
         │                         │                         │
         └── TLS + client auth ────┼──── TLS + client auth ──┘
                                   ▼
                    ┌──────────────────────────────┐
                    │      Federated Server        │
                    │  SecAgg+ unmask (aggregate)  │
                    │        FedAvg                │
                    │  sees SUM only, never one    │
                    └──────────────┬───────────────┘
                                   │
                    global model broadcast; round repeats
```

**Invariant:** the only thing that ever leaves a hospital boundary is a small, DP-noised,
cryptographically masked parameter update, transmitted over an authenticated encrypted channel.

### 3.2 One federated round

1. Server broadcasts current global head parameters to selected clients.
2. Client loads local X-rays — **images never leave the client's filesystem**.
3. Preprocessing: OpenCV CLAHE, then torchvision resize/normalize/augment.
4. Local training of the trainable head via **Opacus DP-SGD** (per-sample gradient clipping +
   calibrated Gaussian noise); privacy budget accumulated by the accountant.
5. Update is quantized and masked by **Flower SecAgg+**.
6. Masked update transmitted over **TLS with client authentication**.
7. Server unmasks only the aggregate and applies **FedAvg**.
8. Improved global model broadcast; repeat until the round budget or stopping criterion is met.

### 3.3 Execution modes

The same `ClientApp` / `ServerApp` code must run in both modes. Do not fork the client logic.

| Mode | Engine | Purpose |
|---|---|---|
| **Simulation** | Flower simulation engine | Fast experiment sweeps: seeds, epsilon values, client counts, partition schemes |
| **Deployment** | Flower deployment engine, Docker Compose | Demonstration: real separate processes, real gRPC, real TLS + client auth, real SecAgg+ |

Results tables are produced in simulation mode. The TLS/SecAgg/isolation claims are
demonstrated in deployment mode.

---

## 4. Technology Stack

Approved components. **Exact pinned versions live in the dependency file
(`requirements.txt` / `pyproject.toml`), never duplicated here.**

| Layer | Technology | Role |
|---|---|---|
| Language | **Python 3.11** | Pinned to 3.11 (not 3.12) for FL/DP ecosystem compatibility |
| DL framework | **PyTorch** | Model, training loop |
| Model | **DenseNet121** | Backbone (ImageNet-pretrained, CheXNet lineage) |
| FL framework | **Flower** | Client/server orchestration, both execution engines |
| Aggregation | **FedAvg** | Server-side aggregation strategy |
| Secure Aggregation | **Flower SecAgg+** | Established masking protocol — no custom crypto |
| Differential Privacy | **Opacus DP-SGD** | Per-sample clipping + Gaussian noise + accountant |
| Transport security | **gRPC + TLS + client authentication** | Confidentiality, integrity, client identity |
| Preprocessing | **OpenCV** | CLAHE contrast enhancement only |
| Transforms | **torchvision** | Resize, normalize, augment, tensor conversion |
| Explainability | **Grad-CAM** | Class-discriminative localization heatmaps |
| Uncertainty | **Monte Carlo Dropout** | Predictive confidence + deferral |
| Configuration | **Hydra / OmegaConf** | All runs fully specified by config; sweeps |
| Tracking | **MLflow (local, self-hosted)** | Metrics, params, artifacts — offline, no third-party cloud |
| Containerization | **Docker Compose** | Multi-container prototype demonstration |
| Testing | **pytest** | Unit + integration tests |
| Numerics | **NumPy** | Array operations |
| Tabular data | **pandas** | Manifests, splits, results tables |
| Metrics | **scikit-learn** | AUROC/AUPRC and related evaluation metrics |
| DICOM decoding | **pydicom** | Reading RSNA's DICOM images |
| Image I/O | **Pillow** | Image loading/saving support |
| Progress reporting | **tqdm** | Training/preprocessing progress bars |
| Plotting | **Matplotlib** | Paper and report figures |
| Grad-CAM implementation | **grad-cam** | Established library backing the Grad-CAM stage (ADR/section 9) |
| Dataset download | **kaggle** | RSNA acquisition automation (Stage 3) |

**Deliberately excluded** (do not introduce without approval): TensorFlow / TF-Federated /
TF-Privacy, NVFlare, OpenFL, PySyft, CrypTen, Weights & Biases or any cloud tracking service,
blockchain components, Vision Transformers or medical foundation models, Kubernetes.

MLflow is chosen over cloud tracking deliberately: a project whose thesis is data sovereignty
should not ship its experiment telemetry to a third-party service.

---

## 5. Important Architectural Decisions

Each decision is recorded with its rationale. **Do not reverse any of these silently.** If
implementation pressure suggests reversing one, stop and raise it.

### ADR-1 — Freeze the DenseNet121 backbone; federate only a small trainable head

**Status:** Approved. This is the load-bearing decision of the entire design.

**Context.** Stock DenseNet121 is built on BatchNorm, and this creates several simultaneous,
independent failures:

- **DP incompatibility.** BatchNorm mixes information across samples in a batch, which breaks
  per-sample gradient computation and voids the DP guarantee. Opacus's `ModuleValidator`
  rejects the model outright. This is a hard incompatibility, not a tuning issue.
- **FL instability.** Averaging BatchNorm running statistics across non-IID clients yields
  statistics that match no client's actual distribution — a known FedAvg failure mode.
- **DP utility collapse.** Noising ~7M parameters destroys utility at any meaningful epsilon,
  because the noise cost scales badly with dimension.
- **Communication.** A full DenseNet121 update is ~28 MB in fp32, exceeding gRPC's 4 MB default
  message limit.
- **Secure Aggregation cost.** Masking and quantizing a 7M-dimensional vector is expensive.
- **Hardware.** Opacus stores per-sample gradients; DenseNet121 is activation-heavy. This will
  not fit in 4 GB of VRAM at any usable batch size.

**Decision.** Freeze the pretrained backbone with **BatchNorm layers in `eval()` mode with
frozen running statistics**, and train + federate only a small classifier head (classifier
layers, optionally the last dense block).

**Consequences.** Frozen BatchNorm in eval mode is a fixed affine transform and is therefore
per-sample safe, so DP-SGD applies cleanly to head parameters only. No BatchNorm statistics are
averaged across clients. The federated update drops from ~7M to roughly 1e5–1e6 parameters,
which simultaneously fixes DP utility, message size, SecAgg cost and VRAM. This is also
straightforward transfer learning, exactly as justified by the source deck.

**Cost.** Caps maximum achievable accuracy relative to full fine-tuning. This is an accepted
and reportable trade-off, and it must be stated in the paper.

**Approved fallback if head-only accuracy proves insufficient:** use Opacus's
`ModuleValidator.fix()` to replace BatchNorm with **GroupNorm** and fine-tune more layers.
GroupNorm is the standard choice for FL under non-IID data and is DP-compatible. Cost:
pretrained BatchNorm statistics are discarded, more training is required, and VRAM pressure
increases substantially. **Requires approval before adopting.**

### ADR-2 — Differential Privacy is sample-level DP-SGD with a formal accountant

Objective 2 is a **patient-level** claim ("no patient's data can be reverse-engineered"), so the
unit of protection is the **training sample**, not the client. Use Opacus DP-SGD: per-sample
gradient clipping to a fixed norm, calibrated Gaussian noise, and a formal accountant (RDP/PRV)
reporting **(epsilon, delta)** with delta stated relative to dataset size.

**Prohibited:** adding noise to weights without per-sample clipping, or reporting a privacy
claim without an accountant. Noise without clipping and accounting provides **no formal
guarantee** and is the single most likely point of reviewer attack.

**Known limitation, to be stated honestly in the paper.** With each client adding full noise
independently, this is effectively **local DP**, which has worse utility than central DP at
equal epsilon. The principled alternative is **distributed DP** — each client adds a share of
the noise and Secure Aggregation ensures only the sum is revealed, achieving central-DP utility
with no trusted server. Implementing that rigorously requires discrete Gaussian or Skellam
noise over a finite field integrated with the SecAgg protocol. **For this prototype: implement
per-client Gaussian DP + SecAgg, report epsilon under a clearly stated trust assumption, and
discuss distributed DP as the principled extension.**

### ADR-3 — Use Flower's SecAgg+, never custom cryptography

Secure Aggregation uses Flower's established SecAgg+ workflow (masking with secret sharing and
dropout resilience). **Writing custom cryptographic protocols is prohibited.** A hand-rolled
masking scheme is likely to be subtly wrong and will not be trusted by reviewers.

**Consequence to plan for:** SecAgg operates over integers in a finite field, so updates must be
**quantized** before masking. Quantization interacts with both accuracy and DP noise. This is
not a nuisance to hide — **measure and report it** as part of the ablation (row 4 in §11).

### ADR-4 — TLS must include client authentication

Server-only TLS authenticates the server to the client and does **not** prevent an attacker from
connecting as a fake hospital. Since the architecture claims impersonation resistance, client
authentication is required — mutual TLS with per-hospital certificates, or Flower's node
authentication with per-node key pairs.

*The exact mechanism and flags must be verified against the pinned Flower version; this area has
changed across releases.*

This also partially delivers what the source deck deferred to future work (confirming an update
came from a registered hospital), at low cost. Certificate/key generation **must be scripted and
committed** so the security setup is reproducible. **Generated certificates, keys and other
secrets must never be committed.**

### ADR-5 — Pin the Flower version exactly

Flower has had significant API churn across 1.x (legacy `start_client`/`start_server` versus the
newer `ClientApp`/`ServerApp` + `flwr run` model). SecAgg+ and the DP integration require the
newer API. Write code against the pinned version's official documentation, not against blog
posts or older examples. **This is the project's single largest reproducibility hazard.**

### ADR-6 — OpenCV is used only for CLAHE

OpenCV provides CLAHE, which torchvision does not. Everything else — resize, normalize,
augment — uses torchvision transforms, for composability and correct seeding of random
augmentation.

Mandatory practices:
- **CLAHE parameters (`clipLimit`, `tileGridSize`) are fixed and logged**, and CLAHE output
  should be **precomputed/cached to disk** so it is not a per-run source of nondeterminism or a
  throughput bottleneck.
- **`cv2.imread` returns BGR; ImageNet-pretrained DenseNet121 expects RGB.** This conversion
  must be explicit and covered by a test. Silently mismatching it costs accuracy invisibly.
- Use `opencv-python-headless` so containers do not need GUI libraries.

### ADR-7 — Patient-level data separation

Splits are performed at the **patient level**, never the image level, wherever patient
identifiers permit. Patient leakage between train and test inflates results and is a standard
examiner probe. Where a dataset lacks patient identifiers, this limitation must be recorded
explicitly in §14 and in the paper.

### ADR-8 — Simulation for measurement, deployment for demonstration

See §3.3. One client implementation, two execution modes. Do not fork client logic between them.

---

## 6. Privacy & Security Design

Four layers, each protecting a distinct asset. The layers are **not interchangeable** and each
must be independently switchable for the ablation study.

| Layer | Protects | Against | Mechanism |
|---|---|---|---|
| **Federated Learning** | Raw patient images | Any party outside the hospital | Images never leave local storage |
| **Differential Privacy** | Information encoded *inside* an update | Inference/reconstruction from updates | Opacus DP-SGD, (epsilon, delta) |
| **Secure Aggregation** | An individual hospital's update | The server itself | Flower SecAgg+ masking |
| **TLS + client auth** | Messages in transit | Network eavesdropping, tampering, impersonation | gRPC TLS + client authentication |

### Threat model (must be stated explicitly in the paper)

**In scope:**
- **Honest-but-curious server** — follows the protocol but attempts to infer information from
  what it receives. Countered by DP (update contents) and SecAgg (individual attribution).
- **Passive network adversary** — eavesdropping or tampering in transit. Countered by TLS.
- **Unregistered party impersonating a hospital.** Countered by client authentication.
- **Client collusion up to the SecAgg+ threshold**, per the protocol's stated bound.

**Explicitly out of scope for this phase:**
- **Malicious clients** submitting poisoned or Byzantine updates.
- Collusion **above** the SecAgg+ threshold.
- Side-channel and physical attacks; compromise of a hospital's own infrastructure.

### Known architectural tensions (name these in the paper — they are contributions, not flaws)

1. **Secure Aggregation vs. Byzantine detection are directly opposed.** SecAgg exists to prevent
   the server from seeing individual updates; Byzantine robustness requires inspecting individual
   updates to find outliers. The future-work item is therefore **not additive** — reconciling it
   needs specific techniques (robust aggregation over secret shares, or zero-knowledge validity
   proofs).
2. **DP noise vs. explainability and calibration.** Grad-CAM quality and confidence calibration
   plausibly degrade as epsilon tightens. Privacy and clinical trust are presented as
   complementary but may partly trade off. Measuring this is a genuine contribution.
3. **Local DP utility vs. the accuracy objective.** Objectives 2 and 5 pull against each other,
   and per-client local DP is the most expensive way to spend the budget. ADR-1's parameter
   reduction is the primary mitigation; ADR-2's distributed-DP discussion is the principled one.

---

## 7. ML Design

- **Task:** binary classification — **Pneumonia vs. Normal** — from chest X-ray images.
- **Architecture:** DenseNet121, ImageNet-pretrained. Chosen for its CheXNet lineage and
  reviewer familiarity in chest X-ray work. **Do not substitute another architecture.**
- **Training regime:** frozen backbone (BatchNorm in `eval()`, running statistics frozen) with a
  small trainable classifier head. See **ADR-1**.
- **Dropout:** stock DenseNet121 contains **no dropout layers**. Dropout must be inserted
  deliberately to enable MC Dropout (see §9). The placement choice is a documented trade-off,
  not an incidental detail.
- **Preprocessing:** OpenCV CLAHE (fixed, logged, cached) → torchvision resize → normalize with
  ImageNet statistics → augment. Explicit BGR→RGB conversion. See **ADR-6**.
- **Input resolution:** 224×224 by default, constrained by available VRAM.
- **Class imbalance:** must be handled explicitly (weighted loss or sampling) and the choice
  recorded. Chest X-ray pneumonia datasets are typically imbalanced.
- **Hardware target:** development on a single 4 GB VRAM GPU (RTX 3050 Laptop), 14 GB RAM,
  16 cores. Clients run **sequentially** in simulation; they cannot be resident concurrently.
  Opacus requires memory-managed batching (e.g. `BatchMemoryManager`) under this constraint.

---

## 8. Federated Learning Design

- **Strategy:** FedAvg. Do not substitute FedProx, FedBN or others without approval.
- **Clients:** multiple simulated hospitals, each holding a disjoint local dataset. Client count
  is a configurable experimental variable.
- **Federated payload:** trainable head parameters only (ADR-1) — DP-noised, quantized, masked.
- **Server:** aggregates only; never has access to raw data, and never to an individual
  unmasked update.
- **Data heterogeneity:** both regimes must be supported and reported:
  - **Natural non-IID** — genuinely different data sources acting as different hospitals
    (pending §12 dataset decision). Preferred, as it reflects real cross-institutional shift.
  - **Synthetic non-IID** — Dirichlet partitioning with a configurable alpha, as a controlled
    sweep.
- **Configurable per experiment:** number of rounds, clients per round / participation fraction,
  local epochs, batch size, optimizer, learning rate, DP parameters (clipping norm, noise
  multiplier, target epsilon), partition scheme and alpha, and all seeds.
- **Communication accounting:** bytes transmitted per client per round and wall-clock per round
  are **first-class measured outputs**, not afterthoughts (see §11).
- **gRPC message size** must be configured explicitly rather than relying on the 4 MB default.

---

## 9. Explainability

- **Method:** Grad-CAM, applied to the global model at inference.
- **Target layer:** final dense block / `features.norm5` of DenseNet121.
- **Output:** heatmap over the chest X-ray showing which lung regions drove the Pneumonia/Normal
  prediction, so a clinician can visually verify the model's reasoning.
- **Implementation:** use an established library (`pytorch-grad-cam` or Captum) rather than a
  hand-rolled implementation. An established library also makes comparing Grad-CAM variants
  nearly free.
- **Execution site:** client-side, on the received global model. Explanations are generated where
  the images are, so no image ever needs to move for explanation.
- **Note:** if the ADR-1 GroupNorm fallback is ever adopted, the Grad-CAM target layer name
  changes and must be updated.

Qualitative heatmaps alone are illustrative, not a result. Quantitative Grad-CAM evaluation is a
**pending optional direction** (§16.1), not part of the approved baseline scope.

---

## 10. Uncertainty Estimation

- **Method:** Monte Carlo Dropout. The same image is passed through the network T times with
  dropout active at inference; the predictive distribution yields a prediction plus a confidence
  estimate.
- **Required implementation detail:** dropout layers must be **inserted deliberately** (§7).
  Head-only dropout is simple but captures only last-layer uncertainty; dropout after dense
  blocks captures more but perturbs pretrained features and requires more fine-tuning.
  **Document which was chosen and why.**
- **Configurable:** number of stochastic forward passes T, dropout rate, uncertainty metric
  (e.g. predictive entropy or variance), and the **deferral threshold**.
- **Clinical behavior:** predictions below the confidence threshold are **flagged for manual
  clinician review rather than acted upon** — this is the human-in-the-loop mechanism required
  by the objectives, and the deferral path must actually exist in the code, not merely be
  described.
- **Honest framing:** MC Dropout is a known-weak uncertainty estimator and is often poorly
  calibrated. It is the approved baseline because the source deck mandates it and it is cheap.
  Stronger methods and calibration metrics are **pending optional directions** (§16.1).

---

## 11. Evaluation Strategy

### 11.1 The ablation ladder — this is the core result

Because the contribution is integration, the required deliverable is the **cost of each layer**:

| # | Configuration | Establishes |
|---|---|---|
| 1 | Single hospital, local only | The floor — what one hospital achieves alone |
| 2 | Centralized pooled training | The ceiling — the privacy-free upper bound |
| 3 | FedAvg | Does FL recover centralized performance? |
| 4 | FedAvg + Secure Aggregation | Cost of quantization and masking |
| 5 | FedAvg + Differential Privacy, swept over epsilon | The privacy–utility curve |
| 6 | Full system (FedAvg + SecAgg + DP + TLS/auth) | The integrated system |

The most persuasive single result available is **the federated global model outperforming any
individual hospital's locally-trained model** (row 3 vs. row 1) — this is the value proposition
of federated learning, and it is strongest under genuine cross-institutional heterogeneity.

### 11.2 Metrics

- **Classification:** **AUROC (primary)**, AUPRC, sensitivity/recall at fixed specificity,
  specificity, F1, balanced accuracy, confusion matrix.
  **Accuracy alone is not acceptable** on imbalanced medical data.
- **Statistical rigor:** bootstrap 95% confidence intervals on AUROC.
  Report **mean ± standard deviation over at least 3 seeds** — single-run numbers are not
  credible in FL, where run-to-run variance is high.
- **Privacy:** reported **(epsilon, delta)** per configuration, from the accountant.
- **Overhead (required, not optional):** bytes transmitted per client per round, total
  communication per experiment, wall-clock time per round, and client-side compute/memory cost.
  Report the overhead attributable to DP and to SecAgg separately.

### 11.3 Testing

`pytest` coverage is required specifically where a silent failure would be invisible:

- **Partitioning:** zero patient overlap between train/val/test splits; partitions are
  deterministic given a seed.
- **Preprocessing:** BGR→RGB correctness; CLAHE determinism given fixed parameters.
- **DP:** clipping and noise math match the intended mechanism; the accountant is actually
  consuming budget across rounds.
- **Secure Aggregation:** **masks cancel exactly** — the unmasked aggregate equals plain FedAvg
  within tolerance. This is the test that converts "we implemented SecAgg" into a defensible
  claim.
- **Integration:** a 2-client, single-round smoke test that runs end to end.

---

## 12. Reproducibility Requirements

Non-negotiable. Reproducibility ranks above accuracy in this project (§2).

- **Configuration:** every run is fully specified by a Hydra/OmegaConf config. No magic numbers
  in code, no hand-edited constants between runs. Sweeps are launched by config.
- **Seeding:** `torch`, `numpy`, `random`, `torch.use_deterministic_algorithms(True)`, DataLoader
  worker seeds, and — critically — **the data-partition seed and the client-sampling seed**.
  These last two are the ones most commonly forgotten in FL work and they dominate variance.
- **Tracking:** local MLflow logs params, metrics, artifacts and the resolved config for every
  run. A result that is not in MLflow does not exist.
- **Dependencies:** exact pinned versions in the dependency file with a lockfile. Flower, Opacus
  and PyTorch versions are pinned exactly (ADR-5).
- **Dataset determinism:** a script that acquires the data, **verifies checksums**, builds the
  partition, and **commits the resulting partition indices** (e.g. as JSON) so a third party can
  reproduce the exact splits.
- **Environment:** Docker Compose definition for the multi-client prototype so the demonstrated
  system is reproducible off this machine.
- **Reporting:** every reported number traces to a config hash + seed set + MLflow run.

---

## 13. Project Structure

**Planned** layout. Nothing below exists yet (§14). This is the target to grow into, not a
structure to scaffold empty in advance.

```
.
├── CLAUDE.md                     # this file — living architecture doc
├── README.md
├── pyproject.toml / requirements.txt   # exact pinned dependencies (single source)
├── Review_1_Privacy_Preserving_FL_Diagnosis.pptx   # source of truth for scope; do not modify
├── conf/                         # Hydra configs
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   ├── federated/
│   ├── privacy/                  # DP + SecAgg settings
│   └── experiment/               # ablation ladder configurations
├── src/
│   ├── data/                     # loading, CLAHE caching, partitioning, patient-level splits
│   ├── models/                   # DenseNet121 head, dropout insertion, freezing logic
│   ├── privacy/                  # DP-SGD wiring, accountant, SecAgg+ configuration
│   ├── federated/                # ClientApp, ServerApp, FedAvg strategy, metrics
│   ├── explain/                  # Grad-CAM
│   ├── uncertainty/              # MC Dropout, deferral logic
│   ├── evaluation/               # metrics, bootstrap CIs, overhead accounting
│   └── utils/                    # seeding, logging, MLflow helpers
├── scripts/                      # dataset prep, certificate generation, experiment runners
├── tests/                        # pytest — see §11.3
├── docker/                       # Dockerfiles + compose for the multi-client prototype
└── certs/                        # generated TLS material — GITIGNORED, never committed
```

**Never commit:** certificates, private keys, credentials, API keys, patient data, or private
datasets. `certs/` and any data directory must be gitignored before the first commit that could
touch them.

---

## 14. Current Implementation Status

**As of 2026-08-28: nothing is implemented.**

| Item | Status |
|---|---|
| Source deck studied and analyzed | Done |
| Technology stack reviewed and approved | Done |
| CLAUDE.md (this document) | Done — awaiting owner approval |
| Git repository | Initialized, remote configured, **no commits yet** |
| Dependency file | `pyproject.toml` / `uv.lock`, pinned |
| Dataset | **Decided (2026-08-29): Kermany = Hospital A; RSNA = Hospitals B & C** |
| Code | Phase 0, Phase 1, Phase 2 (Stages 6–12), and Phase 3's Stage 13 complete — FedAvg verified working end-to-end (real 20-round run) with no privacy layers yet. See `docs/SESSION_STATE.md` for current detail; this table is a coarse summary. |
| Docker configuration | None |
| Tests | 101 passing |

### Resolved decisions

1. **Dataset strategy (resolved 2026-08-29).** Two-source strategy approved: Kermany chest
   X-ray dataset as Hospital A, RSNA Pneumonia Detection Challenge shards as Hospitals B and
   C. See `docs/IMPLEMENTATION_PLAN.md` Part IV preface for the label-semantics and
   Kaggle-access consequences that follow from this choice (Decision Gate DG-2, resolved
   below). **Correction to the acquisition plan:** the Kermany dataset's authoritative source
   (Mendeley Data, doi:10.17632/rscbjbr9sj.3) no longer offers a standalone chest-X-ray-only
   download — as of dataset version 3 it is bundled with an unrelated OCT dataset in one
   ~8.4GB zip (not the ~1.2GB originally estimated from the common Kaggle mirror).
   `scripts/download_kermany.py` downloads the full archive, verifies it against Mendeley's
   published SHA-256, and extracts only the `chest_xray/` subtree.

2. **Decision Gate DG-2 — RSNA label harmonization (resolved 2026-08-29).** Option (a)
   approved: keep RSNA's native `Target` grouping (Normal + "No Lung Opacity / Not Normal"
   both map to the negative class), for benchmark comparability and to preserve the full
   20,672-patient negative class rather than shrinking Hospitals B/C by ~44%. The resulting
   clinical caveat — the model learns "abnormal-but-not-pneumonia" = "normal" — is to be
   stated as an honest limitation in the paper (§15), not engineered around.

3. **Decision Gate DG-3 — hospital-size imbalance (resolved 2026-08-29).** Owner chose to
   report both regimes: the natural partition (Hospital A/Kermany 5,856 images vs. Hospitals
   B/C/RSNA shards 13,342 images each — roughly 4.5x) and a size-balanced companion (B and C
   label-stratified-subsampled down to Hospital A's size, never upsampling A). Both are
   frozen in `data/partitions/hospitals_natural{,_balanced}.json`; both should appear in the
   ablation results.

4. **Dropout placement (resolved in Stage 8, 2026-08-29).** Head-only: a single
   `Dropout(p=0.3)` inside the trainable head, between its hidden `Linear(1024,256)+ReLU`
   and the final `Linear(256,2)`. Chosen for simplicity and to keep the backbone/head
   separation ADR-1 depends on completely clean. Accepted tradeoff: MC Dropout (§10, Stage
   19) will only capture last-layer uncertainty, not backbone-level uncertainty. See
   `src/models/densenet_head.py`.

### Pending decisions (blocking — must be resolved with the owner before related work)

1. **Target epsilon values** for the DP sweep, and delta relative to dataset size.
2. **Client count** and default partition scheme for the headline results — partially
   resolved by DG-3 (client count of 3 for the natural regime); still open for the
   Dirichlet synthetic sweep's client count and which regime is the paper's primary
   headline vs. secondary comparison.

---

## 15. Known Limitations

To be stated honestly in the paper. Concealing these weakens credibility more than admitting them.

1. **Simulated hospitals, not a real deployment.** Clients are simulated processes/containers on
   one machine. No real cross-institutional network, governance or data-use agreements.
2. **Frozen backbone caps accuracy** relative to full fine-tuning (ADR-1). Accepted trade-off.
3. **Effectively local DP**, which has worse utility than central DP at equal epsilon.
   Distributed DP is discussed but not implemented (ADR-2).
4. **Malicious clients are out of scope.** No Byzantine or poisoning defense in this phase.
5. **MC Dropout is a weak uncertainty estimator** and may be poorly calibrated; calibration
   metrics are not in the approved baseline scope (§16.1).
6. **Grad-CAM is evaluated qualitatively** unless the optional quantitative evaluation is
   approved.
7. **Hardware ceiling.** 4 GB VRAM constrains input resolution, batch size, model capacity and
   the feasibility of ensemble-based methods; clients run sequentially in simulation.
8. **Patient-level separation depends on identifier availability** in the chosen dataset
   (ADR-7). Resolved for both approved sources: RSNA carries a clean `patientId`; Kermany's
   authoritative Mendeley source (unlike the third-party Kaggle mirror commonly used instead)
   encodes a groupable accession id in *every* filename, Normal and Pneumonia alike — verified
   empirically, zero id collisions across classes or the source's own train/test split, across
   all 5,856 files. Both sources therefore get full patient-level grouping in Stage 4; this
   limitation does not apply to this project's actual data.
9. **Empirical privacy leakage is asserted from literature, not demonstrated**, unless the
   optional privacy-attack study (§16.1) is approved.

---

## 16. Future Work

### 16.1 Optional research directions — require discussion and approval

**Do not implement these automatically.** Each is a candidate for strengthening the paper; each
must be raised, discussed and approved before any work begins.

- **Empirical privacy attacks against model updates** — membership inference and/or gradient
  inversion, run with DP off and at several epsilon values. This would convert the project's
  central premise ("updates can leak") from a literature citation into a measured, demonstrated
  result, and would show what the DP layer actually mitigates. Highest potential academic payoff.
- **Calibration metrics** — Expected Calibration Error, Brier score, reliability diagrams, and a
  risk–coverage curve quantifying the deferral/human-review story. Also: how calibration degrades
  as epsilon tightens.
- **Quantitative Grad-CAM evaluation** — pointing game or IoU against bounding-box annotations
  (dataset permitting), plus whether explanation quality degrades under DP noise and federated
  averaging.
- **Stronger uncertainty methods** — snapshot/round ensembling over recent FL rounds (nearly
  free, no extra training), or conformal prediction for distribution-free coverage guarantees on
  the deferral decision.
- **Extended privacy/utility trade-off analysis** beyond the baseline epsilon sweep.

### 16.1a Approved optional extensions (approved in concept 2026-08-29; implementation awaits a separate explicit go-ahead)

Two extensions the owner raised, evaluated, and approved as concepts — **neither is part of
the core 24-stage critical path**, both remain clearly optional, and neither is to be
implemented until the owner separately approves the documentation diff that added them here.
Full stage-style write-ups (goal, files, dependencies, prerequisites, testing, risks): `docs/IMPLEMENTATION_PLAN.md`, Phase 6, OPT-5 and OPT-6.

- **OPT-5 — Isolation Forest OOD detection gate.** Detects anomalous/out-of-distribution
  chest X-rays (wrong modality, corrupted scans, unfamiliar population) at inference time —
  **not** anomalous federated client/model updates. That interpretation was considered and
  rejected: CLAUDE.md section 6 already documents that Secure Aggregation and Byzantine
  update-inspection are directly opposed, and section 16.2 already prohibits Byzantine/
  poisoning detection without separate approval. This extension does not touch Secure
  Aggregation, FedAvg, or the federated update path in any way — it runs entirely client-side
  on already-local data, using the 1024-dim pooled backbone feature vector Stage 8/9 already
  produce. One `IsolationForest` per hospital (not federated — it isn't a parametric model
  that can be `FedAvg`'d; each hospital trains its own, consistent with data never leaving a
  hospital). No new dependency (`scikit-learn` is already pinned). Prerequisites: Stages 9
  and 11. Proposed placement: Phase 4, alongside Stage 18 (Grad-CAM) and Stage 19 (MC
  Dropout).
- **OPT-6 — Streamlit demo interface.** A presentation-only layer over an already-trained
  checkpoint: prediction, MC Dropout confidence/deferral (Stage 19), Grad-CAM overlay (Stage
  18), and the OOD flag (OPT-5) if built. Does not touch training, evaluation, privacy
  guarantees, or the FL pipeline. Requires one new dependency (`streamlit`), which itself
  needs the section 17.3 dependency-approval process before it is added — approving this
  extension's concept is not the same as approving that dependency addition. Prerequisites:
  Stages 11, 18, 19 (OPT-5 optional). Proposed placement: Phase 5, after Stage 21, in a new
  top-level `app/` directory (not `src/`, since it is an entry point, not library code).

### 16.2 Explicit future work — do NOT implement unless explicitly approved

- Byzantine participant detection
- Model-poisoning detection
- Advanced digital-signature mechanisms
- Real multi-hospital deployment
- Kubernetes production deployment
- PACS integration
- Full homomorphic encryption

---

## 17. Development Conventions

### 17.1 Governance — read before acting

| Artifact | Rule |
|---|---|
| **CLAUDE.md** | Never modified automatically. Propose → explain → show → ask → only then edit. |
| **Dependencies** | Never added/removed/upgraded/downgraded/replaced without asking first. |
| **Git push / PRs / remote branches** | Never without explicit approval. |
| **`Review_1_...pptx`** | Never modified. It is the source of truth for scope. |
| **Approved architecture (§4, §5)** | Never silently replaced or substituted. |

### 17.2 Working style

- **Work incrementally.** Do not implement the whole project in one pass. Build one component,
  make it testable, then move on.
- **Before any major implementation change, state:** what is changing, why, which components are
  affected, which dependencies are required, how it will be tested, and the architectural impact.
- **If a proposed implementation conflicts with this document or the approved architecture,
  STOP and ask.** Do not work around the conflict silently.
- Do not introduce a technology from the "deliberately excluded" list in §4.
- Do not begin any item in §16 without approval.

### 17.3 Dependency changes

1. State which dependency is changing and why.
2. Ask permission **before** making the change.
3. Update the dependency file only after approval.
4. If the change also affects CLAUDE.md, propose **that** update separately and ask again.

Exact versions live in `requirements.txt` / `pyproject.toml`. **The full dependency list is
deliberately not duplicated in this document** — §4 records roles and rationale, the dependency
file records versions.

### 17.4 Git and GitHub

- Remote: `UddhavSethi/privacy-preserving-medical-diagnosis`.
- Inspecting status, history, branches, diffs and the remote is fine at any time.
- **Never push without explicit approval.** Before any push: show what will be pushed, show the
  diff/stat, explain the changes, ask, and only then push.
- Do not create pull requests or modify remote branches without permission.
- **Never commit** secrets, credentials, API keys, private keys, certificates, patient data or
  private datasets.

### 17.5 Code conventions

- All experimental parameters come from Hydra config — no hardcoded values that vary between runs.
- Every run seeded and logged to MLflow, including the resolved config.
- Every privacy-relevant claim is backed by a test (§11.3) or by an accountant output.
- Prefer established libraries over hand-rolled implementations for anything cryptographic or
  privacy-critical (ADR-3).
- Client logic is written once and runs under both execution engines (ADR-8).
