#!/usr/bin/env bash
# Regenerate the October 2026 audit end to end (docs/audit_2026-10.md). Order matters: model selection
# (validation only) happens before any test split is read. GPU recommended (4 GB is enough with the
# worker counts below). Existing outputs/ are never deleted; scripts skip work already on disk.
set -euo pipefail
cd "$(dirname "$0")/.."

# Inputs that must already exist (all under the gitignored outputs/ and data/ trees; produced by the earlier stages,
# see docs/reproducibility.md): the data partitions and feature caches (data/), the original ablation and baseline
# checkpoints (outputs/checkpoints/{ablation,centralized_baseline,local_baseline}/ from scripts/run_ablation.py,
# scripts/train_centralized.py and the local baselines), and outputs/mlflow state. Nothing below deletes them.

# 0. zero-training audits of the existing checkpoints
uv run python scripts/audit_per_hospital_eval.py
uv run python scripts/audit_mia_existing.py
uv run python scripts/validate_fl_sim_parity.py          # harness vs saved Flower checkpoint (max-diff is not zero: see audit doc)

# 1. validation-only studies and DP recipe selection (no test data read).
#    NOTE: re-running the sweep on a different device/seed order can change near-tied rankings; the recorded choice is
#    outputs/results/selected_dp_config.json. To reproduce the published numbers, SKIP select_dp_config.py and keep that file.
uv run python scripts/exp_fl_diagnosis.py
uv run python scripts/exp_dp_sweep.py --stages ABCDE --device cuda --workers 3
# uv run python scripts/select_dp_config.py              # writes outputs/results/selected_dp_config.json (see note above)

# 2. report phase part 1: trains the in-process rows, scores whatever exists (SecAgg+ rows appear after step 4)
uv run python scripts/exp_final_ablation.py --workers 3

# 3. shadow-model membership inference: non-private + eps 4 (32 shadows each), eps 1 + 8 (16 each)
scripts/run_lira_campaign.sh
scripts/run_lira_campaign_extra.sh

# 4. real SecAgg+ pipeline, 3 seeds: rows 4' and 6 (first campaign), then row 6 again after the accountant fix (_v2)
uv run python -u scripts/run_full_system_campaign.py --rounds 20
uv run python -u scripts/run_full_system_campaign.py --rounds 20 --only row6 --suffix _v2

# 5. report phase part 2: re-score so the SecAgg+ rows are included; independent epsilon recomputation
uv run python scripts/exp_final_ablation.py --workers 1
uv run python scripts/verify_reported_epsilon.py

# 6. tables, checksum manifest, tests
uv run python scripts/make_audit_tables.py
uv run python scripts/make_audit_manifest.py
uv run pytest -q
