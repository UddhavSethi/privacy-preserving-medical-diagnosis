# Deployed model: final metric report

Model: `fedavg_dp_eps4` — FedAvg + Differential Privacy (target ε=4, δ=1e-5), frozen
DenseNet121 backbone + trainable head (~263K params), natural 3-hospital partition
(Kermany = A, RSNA = B, C). Report date: 2026-10-21.

Two kinds of numbers appear below; do not mix them:
- **3-seed** (seeds 42/123/2024, mean ± std): the research results, threshold-free or at 0.5.
- **Deployed checkpoint** (seed 42, the file the app loads): operating-point numbers.
  Threshold, temperature, abstention band chosen on the **validation** set only;
  all figures quoted here are on the held-out **pooled test set (n = 4,838, 31.0% pneumonia)**.

## 1. Deployed decision policy

| Setting | Value |
|---|---|
| Temperature scaling | T = 2.2534 (fit on validation) |
| Decision threshold | 0.35, applied to temperature-scaled P(pneumonia) (best-F1 on validation) |
| "Inconclusive" band | ±0.05 around threshold |
| MC Dropout | T = 20 passes, predictive entropy |
| Deferral | entropy ≥ 0.673 → clinician review (top 10% of validation) |
| OOD gate | one Isolation Forest per hospital, 5% flag target |

Correction made today: the previous threshold (0.2) had been chosen on raw probabilities
but applied to temperature-scaled ones in `app/inference.py`, giving ~89% sensitivity /
~47% specificity live instead of the intended balance. The threshold is now selected
after temperature scaling (`scripts/derive_app_decision_policy.py`), so selection and
use are on the same scale.

## 2. Performance, deployed checkpoint, test set

| Metric | Threshold 0.35 (deployed) | Threshold 0.5 (naive) |
|---|---|---|
| Sensitivity (recall) | **73.1%** | 46.0% |
| Specificity | **72.2%** | 86.9% |
| Precision (PPV) | 54.2% | 61.3% |
| NPV | 85.6% | 78.1% |
| F1 | 0.623 | 0.525 |
| Accuracy | 72.5% | 74.2% |
| Balanced accuracy | 72.7% | 66.5% |
| Confusion (TN/FP/FN/TP) | 2409 / 927 / 404 / 1098 | 2899 / 437 / 811 / 691 |

Threshold-free: AUROC **0.787** (bootstrap 95% CI 0.774–0.800), AUPRC 0.599, Brier 0.168,
ECE (predicted-class confidence, 10 bins, after temperature scaling) 0.026.

Moving from 0.5 to the deployed threshold recovers ~27 points of sensitivity for ~15 points
of specificity — the right direction for a screening tool.

## 3. Human-in-the-loop layers (deployed checkpoint, test set)

| Stage | Cases kept | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| All cases | 100% | 72.5% | 73.1% | 72.2% | 0.623 |
| After "Inconclusive" abstention (12.7% abstained) | 87.3% | 75.2% | 76.1% | 74.8% | 0.653 |
| After abstention + MC-Dropout deferral (10.1% deferred) | 77.1% | 78.7% | 70.7% | 81.8% | 0.653 |

Deferred cases are genuinely the hard ones: accuracy on deferred cases is 48.7% (chance)
versus 75.2% on the rest.

## 4. Same policy over 3 seeds (ε=4; T = 2.2534, threshold 0.35, MC-Dropout mean probabilities)

The deployed operating point applied to the seed 42 / 123 / 2024 checkpoints (temperature fit on seed 42 only), pooled test set, mean ± std:

| Metric | Value |
|---|---|
| AUROC | 0.7996 ± 0.0121 |
| AUPRC | 0.6176 ± 0.0247 |
| Sensitivity | 0.7392 ± 0.0109 |
| Specificity | 0.7317 ± 0.0185 |
| Precision | 0.5540 ± 0.0163 |
| NPV | 0.8617 ± 0.0048 |
| F1 | 0.6332 ± 0.0106 |
| Accuracy | 0.7340 ± 0.0118 |
| Balanced accuracy | 0.7355 ± 0.0087 |
| ECE (after calibration) | 0.0258 ± 0.0007 |
| Brier (after calibration) | 0.1639 ± 0.0048 |

Note: the ablation table in section 5 reports AUROC 0.8085 ± 0.0097 for the same checkpoints because
it scores a single deterministic forward pass; the 0.7996 here averages 20 MC-Dropout passes (what the
app actually does). For reference, the same checkpoints at the naive 0.5 threshold, uncalibrated:
sensitivity 0.4763 ± 0.0178, specificity 0.8793 ± 0.0073, F1 0.5460 ± 0.0198, ECE 0.1023 ± 0.0037.

## 5. Cost of each layer (AUROC, 3 seeds, natural regime)

| Configuration | AUROC |
|---|---|
| Centralized (privacy-free ceiling) | 0.9053 |
| Local, per-hospital average | 0.8924 |
| FedAvg | 0.8144 ± 0.0095 |
| FedAvg + Secure Aggregation | 0.8194 ± 0.0200 |
| FedAvg + DP ε=1 / 2 / 4 / 8 | 0.7909 / 0.8021 / **0.8085** / 0.8133 |

## 6. Privacy and security

- Formal guarantee: ε = 4, δ = 1e-5 (Opacus RDP accountant, per-sample clipping + Gaussian noise).
- Membership-inference attack AUROC at ε=4: 0.4966 ± 0.0018 (≈ chance; no-DP centralized: 0.5084).
- Secure aggregation (Flower SecAgg+), TLS with per-hospital client authentication demonstrated in Docker deployment mode.
- Trust assumption: per-client (effectively local) DP; server is honest-but-curious.

## 7. Explainability and overhead

- Grad-CAM vs RSNA boxes at ε=4: pointing game 0.147 ± 0.058, IoU 0.084 ± 0.038 (centralized: 0.184 / 0.099).
- Communication: 1,053,853 bytes per client per round (≈1 MB, vs ~28 MB full DenseNet121), identical with or without DP.
- Wall-clock per round: 6.74 s with DP vs 0.52 s without (seed 42, last round).

## 8. Honest limitations to state

- Frozen backbone caps accuracy (ADR-1); DP costs sensitivity far more than specificity.
- RSNA label harmonization folds "abnormal-not-pneumonia" into Normal, inflating false positives.
- Simulated hospitals; malicious clients out of scope.
- Operating-point numbers come from a single seed (42) checkpoint.

Source data: `outputs/results/fedavg_dp_eps4_deployed_final_report.json`,
`outputs/results/fedavg_dp_eps4_threshold_calibration_analysis.json`, `docs/results.md`,
`docs/calibration.md`, `docs/privacy_attack.md`, `docs/gradcam_localization.md`.

## 9. Sensitivity analysis: excluding the ambiguous RSNA subgroup (evaluation only, no retraining)

RSNA's "No Lung Opacity / Not Normal" cases (abnormal, not pneumonia) are labeled Normal
(DG-2). Re-scoring the same deployed predictions with that subgroup (1,761 of 4,838 test
images) removed from evaluation only, threshold 0.35:

| Metric | Full test set (n=4,838) | Excluding ambiguous subgroup (n=3,077) |
|---|---|---|
| AUROC | 0.787 (0.774–0.800) | **0.884** (0.872–0.896) |
| AUPRC | 0.599 | 0.892 |
| Sensitivity | 73.1% | 73.1% (unchanged) |
| Specificity | 72.2% | **89.0%** |
| Precision | 54.2% | 86.3% |
| NPV | 85.6% | 77.6% |
| F1 | 0.623 | 0.792 |
| Accuracy | 72.5% | 81.2% |
| False positives | 927 | 174 |

False-positive rate by true subgroup at the deployed threshold: truly Normal RSNA 9.3%
(n=1,340), Kermany normal 21.3% (n=235), ambiguous "Not Normal" **42.8%** (n=1,761).
With the ambiguous subgroup excluded, abstention keeps 87.8% of cases at 84.6% accuracy /
92.5% specificity; abstention + deferral keeps 78.8% at 84.5% accuracy / 95.2% specificity.

Reading: the precision problem is largely label ambiguity, not a weak model. Sensitivity does
not move, since pneumonia cases are untouched. This is a cleaner-cohort figure, not a
deployment figure: real use will include patients with other findings, so report both.
Source: `outputs/results/fedavg_dp_eps4_excl_ambiguous_report.json` (local, gitignored).
