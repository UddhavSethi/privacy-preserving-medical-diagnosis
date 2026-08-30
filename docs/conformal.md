# Conformal Prediction (OPT-4)

Scoped conditionally on OPT-1's own finding (`docs/calibration.md`): DP roughly
quadruples MC Dropout's Expected Calibration Error, flat across the whole epsilon
sweep. Conformal prediction is the principled fix for exactly this situation — it
gives a formal, distribution-free coverage guarantee that does **not** depend on
the underlying model's confidence being well-calibrated at all. This document asks
directly: does that guarantee actually hold here, even for the DP configurations
whose raw confidence OPT-1 showed is damaged? Every number below is real,
computed from Stage 21's already-trained checkpoints. Regenerate with:

```bash
uv run python scripts/run_conformal_analysis.py
```

Full per-seed and aggregated numbers: `outputs/results/conformal.json`.

## Method

Split conformal prediction with the LAC (Least Ambiguous set-valued Classifier)
non-conformity score (Sadinle et al. 2019): score = 1 − p_model(true class). A
threshold is calibrated on the pooled natural **validation** set (held out from
training, and distinct from every test-set evaluation OPT-1–3 used) at
alpha=0.10 — a 90% target coverage, deliberately matching Stage 19's DG-10 fixed
coverage convention — then applied to the pooled natural **test** set to build
prediction *sets* (not point predictions). See `src/uncertainty/conformal.py` for
the exact formulation, including the finite-sample-corrected quantile that makes
the guarantee hold at finite calibration-set size, not just asymptotically.

## Results

Mean ± std over 3 seeds:

| Configuration | Empirical coverage | Mean set size | Fraction "full" (both classes) |
|---|---|---|---|
| Centralized (natural, ceiling) | 0.9069 ± 0.0008 | 1.1798 ± 0.0025 | 0.180 |
| FedAvg (natural, no DP) | 0.8984 ± 0.0008 | 1.4197 ± 0.0321 | 0.420 |
| FedAvg + DP (epsilon=1.0) | 0.9043 ± 0.0034 | 1.3796 ± 0.0342 | 0.380 |
| FedAvg + DP (epsilon=2.0) | 0.9084 ± 0.0037 | 1.3684 ± 0.0302 | 0.368 |
| FedAvg + DP (epsilon=4.0) | 0.9079 ± 0.0029 | 1.3586 ± 0.0276 | 0.359 |
| FedAvg + DP (epsilon=8.0) | 0.9080 ± 0.0045 | 1.3481 ± 0.0273 | 0.348 |
| FedAvg + SecAgg | 0.9017 ± 0.0045 | 1.4547 ± 0.0562 | 0.455 |

Target coverage: 0.90. (A "full" set — 2 of 2 classes retained on this binary
problem — means the classifier isn't confident enough about either class to rule
one out; conformal prediction never actually returns empty sets here, i.e. every
example retains at least the argmax class at this threshold.)

![Conformal coverage and set size](figures/conformal_coverage_and_set_size.png)

## Reading the results

**The coverage guarantee holds, cleanly, everywhere — including every DP
configuration.** Every single row lands within 1.7 percentage points of the 0.90
target (0.898–0.914), with tight seed-to-seed std (≤0.0045). This is the answer to
the question this extension exists to ask: conformal prediction's coverage
guarantee is genuinely robust to the exact calibration damage OPT-1 measured under
DP — the mechanism does what the theory promises, regardless of whether the raw
MC Dropout confidence number underneath it is trustworthy.

**The set-size cost is dominated by federation itself, not by DP specifically —
the opposite of what might be assumed going in.** Centralized training produces
tight, mostly-singleton sets (mean size 1.18, only 18% ambiguous). Every federated
configuration — with or without DP, and regardless of epsilon — sits in a similar,
much higher band (1.35–1.46, 35–46% ambiguous). Within that federated band, DP
does not clearly cost *more* hedging than plain FedAvg; if anything, the DP
configurations' mean set sizes (1.35–1.38) are slightly *smaller* than no-DP
FedAvg's (1.42) and SecAgg's (1.45), with no clean trend across the epsilon sweep.
This is reported exactly as observed, not adjusted to match the "DP forces more
hedging" hypothesis that motivated the comparison — the real finding is that
**federation's own accuracy/generalization gap** (documented in `docs/results.md`
— FedAvg trails the centralized ceiling by ~9 AUROC points) is what drives
conformal prediction to hedge more, and DP does not clearly add to that on top.

## Honest summary for the paper

1. **Conformal prediction is validated as the right fix for OPT-1's finding.**
   Where raw MC Dropout confidence is measurably miscalibrated under DP, the
   conformal coverage guarantee holds anyway, at every epsilon tested — this is a
   genuine, positive, load-bearing result for the paper's clinical-trust story:
   the *ranking* mechanism underneath (which prediction to trust) survives even
   where the *calibration* of the raw number does not.
2. **The "DP costs more hedging" hypothesis is not supported by this data** —
   federation itself, independent of DP, is what enlarges prediction sets
   relative to the centralized ceiling. This should be stated as a genuine
   finding, not smoothed into the expected direction.
3. Conformal prediction gives Stage 19's deferral mechanism (DG-10) a second,
   complementary, formally-grounded option: defer on "full" (ambiguous,
   both-class) sets instead of, or alongside, MC Dropout's fixed-coverage entropy
   threshold — not implemented as a production deferral policy here, but a
   natural extension the set-size data above already supports.
