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

## 4. Research results, 3 seeds (mean ± std, pooled test set, threshold 0.5 unless noted)

| Metric | Value |
|---|---|
| AUROC | 0.8085 ± 0.0097 |
| Accuracy | 0.7542 ± 0.0105 |
| Sensitivity | 0.4763 ± 0.0178 |
| Specificity | 0.8793 ± 0.0073 |
| F1 | 0.5460 ± 0.0198 |
| Balanced accuracy | 0.6778 ± 0.0125 |
| ECE (uncalibrated) / Brier | 0.1023 ± 0.0037 / 0.1744 ± 0.0043 |

Note the deployed seed-42 checkpoint (AUROC 0.787) is below the 3-seed mean; seed variance, not a bug.

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
