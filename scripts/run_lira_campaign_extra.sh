#!/usr/bin/env bash
# epsilon = 1 and 8 shadow-model audits with 16 shadow models each (the epsilon = 4 and non-private
# audits use 32). Fewer shadows -> fewer in/out observations per example, so these two are noisier;
# stated wherever they are reported. Same recipe as scripts/run_lira_campaign.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { uv run python scripts/exp_mia_lira.py run --tag "$1" --n-shadow 16 --workers 3 "${@:2}"; uv run python scripts/exp_mia_lira.py analyze --tag "$1"; }
run dp_eps1 --lr 0.0003 --rounds 20 --opt adam --clip 1.0 --dp-eps 1
run dp_eps8 --lr 0.0003 --rounds 20 --opt adam --clip 1.0 --dp-eps 8
