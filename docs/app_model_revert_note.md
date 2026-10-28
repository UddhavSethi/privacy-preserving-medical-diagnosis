# Deployed app model: revert note

Written 2026-10-28, before switching the public Streamlit app to the row 6 (full-system) checkpoint.

**Model deployed BEFORE this change (revert target)**
- `active_configuration: fedavg_dp_eps4` in `conf/app.yaml`
- Checkpoint `outputs/checkpoints/ablation/dp_eps4.0_seed42.pt` (FedAvg + DP, eps=4, delta=1e-5, original recipe, seed 42; macro AUROC 0.818)
- decision_threshold 0.35, temperature 2.2534, abstention_half_width 0.05, deferral_threshold 0.6728980541229248
- Last commit with this deployed state: `aacdecb` (also the parent of the switch commit)

**To revert:** `git revert <switch-commit>` (or set `active_configuration: fedavg_dp_eps4` in `conf/app.yaml`), then push. The old entry is left untouched in `conf/app.yaml`.
