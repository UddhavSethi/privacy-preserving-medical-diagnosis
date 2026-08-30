#!/usr/bin/env bash
# Stage 17 — Docker Compose multi-client deployment (CLAUDE.md section 3.3's
# "Deployment" engine): brings up one SuperLink + three hospital containers over
# real TLS with real Flower node authentication, and runs a real federated round
# using the canonical Stage 13/14/16 app, completely unmodified.
#
# Demonstration only (DG-9, owner-approved 2026-08-30): CPU-only, few rounds by
# default. The measured ablation results come from simulation (Stages 11-15),
# not from this script.
#
# Isolation: uses a project-local FLWR_HOME (.flwr_home/, gitignored) rather
# than the real ~/.flwr/, so running this script never touches the operator's
# own Flower CLI state — matching tests/test_tls_auth.py's own isolation
# pattern.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NUM_ROUNDS="${NUM_ROUNDS:-3}"
CONTROL_API_PORT=9093
FLWR_HOME="$(pwd)/.flwr_home"

echo "=== Generating fresh certificates ==="
./scripts/generate_certs.sh

echo "=== Preparing per-hospital data shards ==="
uv run python scripts/prepare_deployment_shards.py

echo "=== Building images ==="
docker compose -f docker/docker-compose.yml build

# SuperNode auth (Stage 16) requires registering each hospital's public key
# BEFORE that hospital's SuperNode ever attempts to connect — dynamic
# self-registration is disabled once --enable-supernode-auth is set. Bringing
# up every container at once (`docker compose up -d` for all services) races
# hospital SuperNodes starting immediately against the registration loop
# below; found via this script's own first real run, not by inspection: a
# SuperNode whose first activation attempt lands before its registration
# crashes that connection permanently (Flower doesn't retry a hard
# FAILED_PRECONDITION the way it retries a plain connection failure), so the
# container is left "Up" but never actually connected. Fixed by starting the
# SuperLink alone first, registering every hospital, THEN starting the
# hospital containers — eliminating the race rather than trying to out-time it.
echo "=== Starting SuperLink ==="
docker compose -f docker/docker-compose.yml up -d superlink

cleanup() {
    echo
    echo "=== Tearing down containers ==="
    docker compose -f docker/docker-compose.yml down
}
trap cleanup EXIT

echo "=== Waiting for SuperLink Control API ==="
for _ in $(seq 1 30); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${CONTROL_API_PORT}") 2>/dev/null; then
        exec 3<&- 3>&-
        break
    fi
    sleep 1
done

rm -rf "$FLWR_HOME"
mkdir -p "$FLWR_HOME"
cat > "$FLWR_HOME/config.toml" <<EOF
[superlink]
default = "deployment"

[superlink.deployment]
address = "127.0.0.1:${CONTROL_API_PORT}"
root-certificates = "$(pwd)/certs/ca.crt"
EOF
export FLWR_HOME

echo "=== Registering hospitals ==="
for hospital in A B C; do
    uv run flwr supernode register "certs/hospital_${hospital}.pub" deployment
done

echo "=== Starting hospital SuperNodes (registered first, per the fix above) ==="
docker compose -f docker/docker-compose.yml up -d hospital-a hospital-b hospital-c

echo "=== Waiting for SuperNodes to connect and authenticate ==="
sleep 8
uv run flwr supernode list deployment

echo "=== Running ${NUM_ROUNDS}-round federated training over Docker ==="
uv run flwr run . deployment --run-config "num-server-rounds=${NUM_ROUNDS}" --stream

echo
echo "Deployment run complete."
