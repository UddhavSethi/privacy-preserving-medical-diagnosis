# Tests run and results measured, in order

All numbers: seeds {42, 123, 2024}, ± = sample std (n−1) unless stated. Natural partition (Hospital A = Kermany, pediatric,
72% pneumonia; B and C = RSNA shards, 22%). Test split = 4,838 images, 31.0% pneumonia. "Macro AUROC" = mean of the three
within-hospital test AUROCs. Authoritative tables: `docs/audit_tables.md`; write-up: `docs/audit_2026-10.md`.

**Status tags.** Each numbered item in section B is tagged **[CURRENT]** (the figure stands and may be quoted), **[HISTORICAL]** (a measurement of
the original pipeline, kept for traceability; the figure is correct for what it measured but is superseded as a headline) or **[DEFECT]**
(a measurement exposing a bug that was later fixed; the figure is not a valid result). The current headline numbers are in items 10, 12, 13, 16 and 19.

## A. Baseline and test-suite history
| When | Command | Result |
|---|---|---|
| Start of audit | `uv run pytest -q` | 255 passed (2 min 11 s) |
| After DP/harness/MIA/reporting additions | per-file runs | `test_dp_validity` 9 passed, `test_mia` 10, `test_fl_sim` 7, `test_evaluation_protocol` 2, `test_comm_accounting` 3, `test_accountant_state` 2, `test_reporting` +1 |
| Final tree (before the commit) | full suite | **289 passed** (255 + 34 new) |
| After the commit, re-run | full suite | **289 passed** (2 min 8 s) |
| Follow-up commit `bf59584` | full suite incl. `tests/test_verify_deployment_epsilon.py` (4 tests: configuration read from files, recomputation pinned to the recorded six-round ε, calibration sanity, dependence on configuration not logs) | **293 passed** (2 min 21 s) |
| Clean-environment check | minimal venv (numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0 from PyPI) | 28 scipy-dependent tests passed; full `uv sync` offline impossible (CUDA wheels uncached) so NOT run |
| Lockfile | `uv lock --check --offline` | consistent after hand-adding `scipy` (a plain `uv lock` upgraded dozens of unrelated packages and was reverted) |

What the 34 new tests assert: batch-dependence of the legacy weighted loss (60% change) vs 0 for `per_sample_mean`; injected DP
noise std = σ·C (±2%, 262 k coords, two settings); neighbouring-batch sensitivity ≤ C; RDP ε ≥ exact analytic-Gaussian ε
and ≤ 1.2× it; RDP within 25% of PRV; ε monotone in noise/steps; accountant reset defect demonstrated and fix equals a long-lived engine;
LiRA vectorised == per-example reference; ε-lower-bound never exceeds true ε on randomized response; byte counter never raises
(incl. empty-list ConfigRecord crash in Flower 1.35); harness round 1 == manual FedAvg of the shared trainers; deterministic per seed;
threshold-for-specificity uses negatives only; sample-std/t-interval math.

## B. Findings and measurements, in the order they were obtained
1. **Row 1 not like-for-like** [HISTORICAL — pooled column superseded; per-hospital columns CURRENT] (`scripts/audit_per_hospital_eval.py`, no training). AUROC of each checkpoint on test A | B | C | pooled:
   local_A 0.985|0.720|0.738|0.752; local_B 0.723|0.834|0.859|0.799; local_C 0.649|0.834|0.858|0.795; centralized 0.979|0.838|0.862|0.905;
   FedAvg 0.946|0.828|0.854|0.814; FedAvg+SecAgg 0.950|0.823|0.849|0.819; DP ε1/2/4/8 (original recipe) A 0.780/0.833/0.864/0.886, B ≈0.770–0.780,
   C ≈0.805–0.820, pooled 0.791/0.802/0.809/0.813. FedAvg beats all local models pooled, trails each local model slightly in-domain.
2. **Flower trajectory** [HISTORICAL — original recipe]: FedAvg pooled-test AUROC per round fluctuates ±0.01 (e.g. seed 42: 0.805…0.830); DP ε=4 (original) peaks at round ~6
   (0.814) then declines monotonically to 0.797 by round 20.
3. **DP loss defect demonstration** [CURRENT]: sample-0 per-sample-gradient norm 193.37 (companions negative) vs 77.60 (positive), rel. diff 0.599 under
   `F.cross_entropy(weight=…)`; 0.000 under `(ce*w[y]).mean()`.
4. **Noise check** [CURRENT] (Opacus): empirical noise std 1.0007 (σC=1.0) and 1.2515 (σC=1.25).
5. **Accountant cross-checks** [CURRENT]: q=1, σ=5, 50 steps: RDP 7.077 vs exact analytic 6.573; hospital-A-like (σ 0.9, q 0.00766, 2600 steps): RDP 3.092 vs PRV 2.760.
6. **Harness parity** [CURRENT] (in-process vs Flower, seed 42, legacy FedAvg): test AUROC rounds 1–4 identical (0.8048, 0.8081, 0.8149, 0.8161); then diverges
   (max per-round difference 0.0196, mean 0.0070; final weights max-abs diff 0.19) — floating-point chaos within the ±0.01 band. Re-checked after GPU edits: unchanged.
7. **Validation-only FedAvg diagnosis** [CURRENT] (`scripts/exp_fl_diagnosis.py`, mean of last 5 rounds, pooled/A/B/C val AUROC): class weights local+all views
   0.818/0.951/0.830/0.838; local+eval-only 0.838/0.961/0.828/0.836; none+all 0.810/0.924/0.829/0.836; none+eval-only 0.832/0.944/0.832/0.835.
   Centralized val: 0.902/0.990/0.836/0.846. ⇒ augmentation hurts; class weights <0.01; within-hospital gaps 0.01–0.03 vs pooled gap 0.06–0.09.
8. **GPU timing** [CURRENT]: one DP round for 3 clients: 3.3 s GPU vs 11.3 s multi-thread CPU vs 49 s single core. GPU memory ≈ 880 MB/worker ⇒ max 3 workers (5 OOM'd).
9. **DP recipe sweep** [CURRENT] (validation only, ε=4, seed 42, 48 runs, `outputs/results/exp_dp_sweep_val.jsonl`): original Adam 1e-3 (legacy loss, local weights) macro 0.8176;
   legacy vs per-sample loss and local vs none weights differ ≤0.001 ⇒ loss defect empirically immaterial; SGD best macro 0.8385 (clip 0.5, lr 0.1, 10 rounds);
   **winner: Adam, lr 3e-4, clip 1.0, 20 rounds, per-sample loss, no class weights, macro val 0.8432** (pooled 0.8564). Three sweep pools deadlocked
   (futex) and were restarted with `maxtasksperchild=1`.
10. **Report phase** [CURRENT for matched / tuned rows; original-recipe DP rows HISTORICAL with the F4/F5 caveats] (`scripts/exp_final_ablation.py`, test; table T1 in `docs/audit_tables.md`):
    | Config | Macro AUROC | A / B / C | Pooled | Sens @ per-hospital 90%-spec thr. |
    |---|---|---|---|---|
    | Centralized | 0.893±0.001 | 0.979/0.838/0.862 | 0.905 | 0.686±0.013 |
    | FedAvg (project) | 0.876±0.006 | 0.946/0.828/0.854 | 0.814 | 0.638±0.020 |
    | FedAvg+SecAgg (orig. run) | 0.874±0.007 | 0.950/0.823/0.849 | 0.819 | 0.635±0.019 |
    | FedAvg DP-matched | 0.869±0.005 | 0.923/0.830/0.853 | 0.824 | 0.603±0.012 |
    | DP ε=1 (orig/tuned) | 0.785/0.796±0.018 | A 0.780/0.785 | 0.791/0.838 | 0.429/0.476 |
    | DP ε=2 | 0.806/0.817±0.013 | A 0.833/0.833 | 0.802/0.845 | 0.475/0.509 |
    | DP ε=4 | 0.819/0.830±0.010 | A 0.864/0.865 | 0.809/0.847 | 0.502/0.538 |
    | DP ε=8 | 0.828/0.839±0.008 | A 0.886/0.886 | 0.813/0.849 | 0.523/0.556 |
    ⇒ federation costs 0.017 macro AUROC; DP ε=4 costs 0.038 (AUROC) and 0.065 (sensitivity) vs matched; tuned recipe +0.011 at every ε;
    Hospital A loses most (±0.03–0.06 seed spread); pooled AUROC can invert the DP comparison (0.847 vs 0.824) while macro does not (0.830 < 0.869).
    At threshold 0.5 the tuned DP model has sens 0.19/spec 0.98 (prior-tracking, no class weights) — low "default-threshold sensitivity" is partly an artefact.
11. **Original loss-attack re-audit** [CURRENT as a measurement — the original report's interpretation is what is superseded] (`scripts/audit_mia_existing.py`, T3): train-vs-test AUROC centralized 0.5084, FedAvg 0.4994, DP ε1/4/8 0.4950/0.4965/0.4970;
    **negative control val-vs-test: centralized 0.5038**, others 0.497–0.503; TPR@1%FPR ≈ 0.010 (centralized 0.016). ⇒ attack had no demonstrated power.
12. **Shadow-model (LiRA) audits** [CURRENT] (T4; ~half-size training data each): non-DP 32 models: global-threshold AUROC 0.5107, LiRA 0.5169, TPR@1%FPR 0.0116, indicative ε_lb 0.13;
    DP ε=4 (32): 0.5009 / 0.4992 (TPR 0.0100, ε_lb 0); ε=1 (16): 0.5017 / 0.4982; ε=8 (16): 0.5019 / 0.4993. Speed-up from caching pooled data (shadow ≈35 s each).
13. **Real Flower SecAgg+ campaign** [CURRENT] (`scripts/run_full_system_campaign.py`, GPU fractions 0.3/client): 20 rounds ≈ 83 s (SecAgg+), 99–109 s (SecAgg+DP).
    Macro AUROC: SecAgg+ matched 0.8653±0.0119 (per seed 0.877/0.853/0.866); **row 6 (SecAgg+DP ε=4) 0.8303±0.0102 (0.8409/0.8205/0.8296) vs in-process DP-only 0.8303±0.0103
    (0.8409/0.8203/0.8296)** — differences ≤ 2×10⁻⁴; sens@spec90 0.538.
14. **Communication** [CURRENT] (clean runs): SecAgg+ per client per round: receives 4.16 MB, sends 2.11 MB (masked update int64 = 2× plain FedAvg's 1.05 MB), 5 messages;
    control messages ≈ 2 KB. Defect: first logs for runs 2–6 landed in the first run's file (simulation SuperLink keeps its first environment) → added `run_id` to records.
15. **Instrumentation bugs found by running** [DEFECT, fixed]: (i) `ConfigRecord.count_bytes` raises IndexError on empty lists in SecAgg+ messages — my mod crashed clients mid-protocol (fixed, tested);
    (ii) Ray hides the GPU from simulated clients — `client-resources-num-gpus=0.3` required (field name found in Flower source).
16. **F11 — accountant reset defect** [DEFECT for the constant 2.12 and the first campaign's logged ε; the post-fix values are CURRENT]: SecAgg+ DP client logged ε = 2.12 in all 20 rounds (cumulative should reach ≈4). Cause: `client_fn` builds a new client per message. Fix:
    persist accountant history in `Context.state` (`src/privacy/accountant_state.py`). After fix: 2.12 → 3.17 (round 10) → 3.992 (round 20) on all seeds; canonical app live smoke (3 rounds): 3.384, 3.728, 3.995.
    First-principles recomputation (`scripts/verify_reported_epsilon.py`): A 3.9955, B 3.9939, C 3.9871 (RDP), mean 3.992178 = logged; PRV 3.52/3.38/3.37.
17. **Earlier Docker run (pre-fix, 2 rounds)** [DEFECT — pre-fix reporting; not a valid ε]: logged ε 3.604 in both rounds (reset); loss fell, loop worked.
18. **Pre-commit review** [CURRENT]: no deletions; no secrets/PII patterns; 53 manifest files verified present and unchanged (size + SHA-256); 15 headline numbers cross-checked against raw JSON
    (one rounding fixed: DP cost 0.038 not 0.039); reproduction script order fixed (SecAgg+ rows before final scoring; ε 1/8 audits added); privacy assumptions documented.
19. **Docker deployment of the committed code** [CURRENT — one seed, CPU, 6 rounds] (`NUM_ROUNDS=6 scripts/run_deployment_full_system.sh`; HEAD `7187c04`; CPU containers; TLS + 3 registered node keys; SecAgg+ + DP ε=4 tuned recipe):
    | Round | Logged ε | Independent ε |
    |---|---|---|
    | 1 | 2.943209 | 2.943209 |
    | 2 | 3.230412 | 3.230412 |
    | 3 | 3.455059 | 3.455059 |
    | 4 | 3.649697 | 3.649697 |
    | 5 | 3.829521 | 3.829521 |
    | 6 | 3.991884 | 3.991884 |
    Strictly increasing; max abs diff 0. 30 distinct `flwr-clientapp` PIDs per hospital (6 rounds × 5 SecAgg+ messages) ⇒ fresh client process per message; 2 long-lived processes
    (`flower-supernode`, `flower-superexec`). Code in all three containers byte-identical to the commit (sha256 of 4 files). No errors. Pooled-test AUROC ≈ 0.82 after 6 rounds (0.29 untrained).
    Evidence (gitignored): `outputs/deployment_lifecycle/` (`run.log`, container logs, `process_poll.log`, `code_hashes_*`, `verification.json`, `RECORD.md`).

## C. Things that were NOT tested (carry forward)
- Full clean-environment install; multi-seed or GPU Docker deployment (containers CPU-only, one 6-round run, single seed).
- Opacus's Poisson-subsampled RDP analysis (taken as given); gradient-inversion attacks; external validation set; per-ε DP re-tuning over 3 seeds.
- ε covers one training run per hospital; many runs/sweeps/validation selection are not composed.
