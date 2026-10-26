#!/usr/bin/env bash
# Shadow-model (LiRA) membership-inference audit campaign. 32 shadow federated runs per configuration,
# each trained on a random half of (train U val); GPU, 3 workers (4 GB card). See scripts/exp_mia_lira.py.
# The DP recipe is the validation-selected one (outputs/results/selected_dp_config.json): Adam, lr 3e-4,
# clip 1.0, 20 rounds, per-sample loss, no class weights. The non-private arm is the matched FedAvg recipe.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { uv run python scripts/exp_mia_lira.py run --tag "$1" --n-shadow 32 --workers 3 "${@:2}"; uv run python scripts/exp_mia_lira.py analyze --tag "$1"; }
run nondp    --lr 0.001  --rounds 20
run dp_eps4  --lr 0.0003 --rounds 20 --opt adam --clip 1.0 --dp-eps 4
run dp_eps1  --lr 0.0003 --rounds 20 --opt adam --clip 1.0 --dp-eps 1
run dp_eps8  --lr 0.0003 --rounds 20 --opt adam --clip 1.0 --dp-eps 8
