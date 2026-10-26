#!/usr/bin/env bash
# Evidence collector for the Docker full-system run (docs/deployment_privacy_accounting_evidence.md). Start it BEFORE
#   NUM_ROUNDS=6 scripts/run_deployment_full_system.sh > outputs/deployment_lifecycle/run.log
# It waits for the hospital containers and then records: (1) hospital container logs, followed until they stop (the
# deployment script tears containers down at exit, which would destroy them), (2) sha256 of the code inside each
# container, (3) every 2 s the hospital containers' client processes as seen from the host (`docker top`).
# This is the same logic as the collector used for the recorded run (outputs/deployment_lifecycle/watch.sh); only the
# working directory is derived from the script location instead of being absolute.
set -u
cd "$(dirname "$0")/.." && mkdir -p outputs/deployment_lifecycle && cd outputs/deployment_lifecycle
until docker ps --format '{{.Names}}' | grep -q "^docker-hospital-c-1$"; do sleep 1; done
for h in a b c; do docker logs -f docker-hospital-$h-1 > container_hospital_$h.log 2>&1 & done
sleep 3
for h in a b c; do
  docker exec docker-hospital-$h-1 sh -c 'cd /app && sha256sum src/privacy/accountant_state.py src/federated/client_app_secagg.py src/privacy/dp.py src/federated/comm_accounting.py' > code_hashes_hospital_$h.txt 2>&1
done
while docker ps --format '{{.Names}}' | grep -q "^docker-hospital-"; do
  for h in a b c; do
    docker top docker-hospital-$h-1 -eo pid,lstart,etime,args 2>/dev/null | grep -E "flwr-clientapp|flower-supernode|clientapp" | sed "s/^/$(date +%H:%M:%S) hospital_$h /" >> process_poll.log
  done
  sleep 2
done
wait
