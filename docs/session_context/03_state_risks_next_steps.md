# State, risks, operational gotchas, next steps

## Open risks (what passing tests does NOT establish)
- Stateless-deployment accountant: now verified live in Docker (6 rounds, ε 2.94→3.99, matches recomputation) but single seed, CPU, one run.
- ε semantics: sample-level, per hospital, per run, local-DP-style; excludes validation-based selection/thresholds/temperature, label-count class
  weights in the *original-recipe* rows, example counts sent for FedAvg weighting, and the composition of this study's many runs.
  Original-recipe rows also carry the F4 loss caveat (empirically immaterial).
- Opacus's subsampled-RDP analysis not independently derived; `secure_mode` off.
- Membership inference is weak evidence by construction (chance everywhere except a tiny signal, LiRA AUROC 0.517, without DP); no gradient inversion.
- DP recipe tuned once (ε=4, seed 42) and reused for other ε. Hospital A's DP accuracy has ±0.03–0.06 seed spread.
- No external validation set; simulated hospitals on one machine; 3 clients (SecAgg+ and the aggregate reveal a lot at n=3).
- GPU runs not bit-reproducible; the in-process harness matches Flower for rounds 1–4 then diverges within round-to-round noise.

## Stale deliverables to regenerate (resolved later on 2026-10-26 — see the addendum in 00_INDEX.md)
- `docs/report/PneumoFL_VIT_Project_Report.pdf` and the IEEE manuscript still state pre-audit pooled-AUROC claims ("federation costs ~9 points",
  "did not beat the best single hospital", "misses more than half of pneumonia cases at the default threshold") and "mutually authenticated TLS"
  (the implementation is server TLS + Flower node authentication).

## Pending owner decisions
- (resolved later the same day: applied at the owner's instruction) `CLAUDE.md` edits (six proposals at the end of `docs/audit_2026-10.md`): row 6 now 3-seed in simulation; ε figures 3.37/3.60 are single-round values;
  headline metric = macro AUROC; privacy-attack wording; TLS wording; declare `scipy` (done in pyproject by explicit instruction).
- (resolved later the same day: pushed at the owner's request) Push of the local commits (`7187c04` and later) — never pushed without approval.
- Whether to build GPU Docker images (needs NVIDIA container toolkit; changes the DG-9 CPU-only decision).

## Operational gotchas learned this session
- `pkill -f <pattern>` / `pgrep -f` can match the command line of the shell running it and kill it (exit 144). Match anchored process names or use PIDs.
- Long-lived `multiprocessing` pools deadlocked (futex) three times on this machine → every experiment script uses `Pool(..., maxtasksperchild=1)`.
- GPU workers: ~880 MB each; 3 workers max on the 4 GB card.
- Flower simulation SuperLink outlives a `flwr run` and fixes its environment at first start → env-var config reaches only the first run; records carry `run_id`.
- Simulated clients need `--federation-config "num-supernodes=3 client-resources-num-gpus=0.3"` to see the GPU (`options.backend...` is rejected).
- `scripts/run_full_system_campaign.py` / `run_deployment_full_system.sh` swap `[tool.flwr.app.components]` in `pyproject.toml` and restore it on exit; a hard kill leaves it swapped (restore by hand).
- Never run the full pytest suite while a Flower campaign is running (one test starts its own simulation and uses the same pyproject).
- `uv lock` without `--check` re-resolves and upgrades unrelated packages; edit the lock by hand for a single declared dependency.
- `float32` clipping near 1.0 gives log(0) in membership-inference statistics — compute in float64.
- Flower 1.35 `ConfigRecord.count_bytes` raises on empty lists.

## Next highest-value steps toward publication
1. Owner approves/rejects the `CLAUDE.md` proposals; decide push.
2. Regenerate the VIT report and IEEE manuscript on the macro-AUROC protocol (and fix the TLS wording).
3. Re-tune the DP recipe per ε over 3 seeds; report Hospital A fairness explicitly.
4. Multi-seed Docker/TLS runs (GPU images if the toolkit is installed).
5. Gradient-inversion attack; external validation set; distributed DP discussion.
6. Literature search to support the novelty claim.

## Commits made in this session (in order)
`becf2a3` → amended to `082c8be` (force-pushed) → history rewritten twice to remove Claude attribution (new hashes; the old history survives only in the two
local backup branches) → `b580a0b`, `7fd3970` (hash-reference remap; pushed) → `da8ff1e` (blank page) → `e3b2a77` (cover order; pushed; this is `origin/main`) →
`7187c04` (audit; local only) → `bf59584` (deployment privacy-accounting verification; local only) → the docs commit that adds this directory (local only).
