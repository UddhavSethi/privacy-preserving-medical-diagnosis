# Deployment evidence: privacy accounting across rounds with fresh client processes

**Question.** After the accountant-reset fix (`docs/audit_2026-10.md`, F11), does the reported ε accumulate across rounds in the
Docker deployment, where Flower starts a fresh client process per message? **Answer: yes, in one six-round run; this is a
single-seed, CPU-only demonstration, not a measurement.**

## Configuration of the recorded run (2026-10-26)

| Item | Value |
|---|---|
| Code | commit `7187c049307dcf8643831d043a6d830355e919bf`; working tree clean apart from untracked files; images `COPY src` at build time |
| Command | `NUM_ROUNDS=6 scripts/run_deployment_full_system.sh` (swaps in the SecAgg+ app pair, restores `pyproject.toml` on exit) |
| System | 1 SuperLink + 3 hospital containers, per-hospital data shards; **server-side TLS + 3 registered node key pairs** (SuperNode authentication) |
| Layers active | FedAvg + SecAgg+ (3 shares, threshold 2) + DP-SGD |
| DP | target ε = 4.0, δ = 1e-5, Adam lr 3e-4, clip 1.0, `per_sample_mean` loss, no class weights, batch 32, 1 local epoch (validation-selected recipe) |
| Rounds / seed / device | 6 / 42 (run-config default) / CPU (containers are CPU-only, owner decision DG-9) |
| Errors | none (no traceback in the run log or any container log) |

## Per-round reported ε versus independent recomputation

`scripts/verify_deployment_epsilon.py` recomputes the expected ε from the recorded configuration (batch size, local epochs and δ from
`pyproject.toml`; target ε from `run_deployment_full_system.sh`; per-hospital training-set sizes 4,180 / 9,330 / 9,348 from the partition file): noise
multiplier calibrated for the planned 6 rounds, realised sampling rate q′ = 1/⌈N/B⌉, steps r·⌈N/B⌉, a **fresh** Opacus RDP accountant per round, mean over
hospitals. The logged values are only compared afterwards. Calibration sanity: at the planned (q = B/N) setting the calibrated σ gives ε = 3.9939 / 3.9912 / 3.9959 (target 4).

| Round | ε logged by the deployment | ε recomputed | |difference| |
|---|---|---|---|
| 1 | 2.943209 | 2.943209 | 0 |
| 2 | 3.230412 | 3.230412 | 0 |
| 3 | 3.455059 | 3.455059 | 0 |
| 4 | 3.649697 | 3.649697 | 0 |
| 5 | 3.829521 | 3.829521 | 0 |
| 6 | 3.991884 | 3.991884 | 0 |

Strictly increasing; final value below the target 4. For contrast, the earlier pre-fix Docker run (2 rounds) logged 3.604 in **both** rounds (reset every round).

## Fresh-process-per-round finding

Each hospital container runs two long-lived processes (`flower-supernode`, `flower-superexec`); the client code runs in separate `flwr-clientapp`
subprocesses. Sampling the containers from the host (`docker top`, every 1–2 s) saw **30 distinct `flwr-clientapp` PIDs per hospital**
= 6 rounds × 5 SecAgg+ messages: a new client process per message, so an in-memory accountant would have been lost every time. The hash of
`accountant_state.py`, `client_app_secagg.py`, `dp.py` and `comm_accounting.py` inside all three containers equals `git show 7187c04:<path>`.

## What this does and does not show

- Shows: the persisted-accountant fix works with stateless client processes under real TLS/node authentication; reported ε equals an independent recomputation.
- Does not show: more than one seed or run; GPU deployment; that the process sampling caught every short-lived process (the count matches the expected
  message structure exactly, but sampling cannot prove completeness); anything about ε's assumptions (per hospital, per run, sample-level; see `docs/audit_2026-10.md`,
  "The privacy guarantee"). Opacus's Poisson-subsampled RDP analysis is shared with training and is not independently derived here.

## Regenerating the evidence (large raw outputs are gitignored and kept locally)

```bash
mkdir -p outputs/deployment_lifecycle
scripts/collect_deployment_lifecycle.sh &                       # start first: records container logs, code hashes, process samples
NUM_ROUNDS=6 scripts/run_deployment_full_system.sh > outputs/deployment_lifecycle/run.log 2>&1
uv run python scripts/verify_deployment_epsilon.py --rounds 6   # writes outputs/deployment_lifecycle/verification.json
uv run pytest tests/test_verify_deployment_epsilon.py           # pins the recomputation to the table above
```
Needs Docker (CPU is enough; ≈ 75 s per round), generated certificates (`scripts/generate_certs.sh`, done by the run script) and the data shards.
The recorded run used `outputs/deployment_lifecycle/watch.sh`, whose logic `collect_deployment_lifecycle.sh` reproduces (only the working directory differs).

## Local raw evidence (gitignored, preserved) — fingerprints, first 16 hex of SHA-256

| File (`outputs/deployment_lifecycle/`) | bytes | sha256 |
|---|---|---|
| `run.log` | 47216 | `e54d703d322a287c` |
| `process_poll.log` (1,565 samples) | 561166 | `92c1d83235b538d3` |
| `container_hospital_a.log` (b, c alike) | 119357 | `2bbc4e5d325c9910` |
| `code_hashes_hospital_a.txt` (b, c alike) | 459 | `d0f44ca403e6b945` |
| `verification.json` (re-generated with the recorded configuration included) | 2099 | `43ca7feba1d87761` |
