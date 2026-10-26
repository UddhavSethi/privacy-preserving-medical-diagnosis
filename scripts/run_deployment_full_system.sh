#!/usr/bin/env bash
# Full system in DEPLOYMENT mode: FedAvg + SecAgg+ + DP (validation-selected recipe, eps=4) over the
# Docker Compose deployment with real TLS and registered-node authentication. Demonstration, not a
# measurement (containers are CPU-only; use NUM_ROUNDS small). Temporarily points
# [tool.flwr.app.components] at the SecAgg+ app pair and restores it on exit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cp pyproject.toml /tmp/pyproject.before_full_system.toml
restore() { cp /tmp/pyproject.before_full_system.toml pyproject.toml; echo "pyproject.toml restored"; }
trap restore EXIT
python3 - <<'PY'
import re
s = open("pyproject.toml").read()
pat = re.compile(r'serverapp = "src\.federated\.server_app(?:_secagg)?:app"\nclientapp = "src\.federated\.client_app(?:_secagg)?:app"')
s, n = pat.subn('serverapp = "src.federated.server_app_secagg:app"\nclientapp = "src.federated.client_app_secagg:app"', s, count=1)
assert n == 1
open("pyproject.toml", "w").write(s)
PY
export NUM_ROUNDS="${NUM_ROUNDS:-2}"
export EXTRA_RUN_CONFIG='dp-enabled=true target-epsilon=4.0 dp-loss-mode="per_sample_mean" dp-optimizer="adam" max-grad-norm=1.0 learning-rate=0.0003 class-weights="none" device="cpu"'
./scripts/run_deployment.sh
