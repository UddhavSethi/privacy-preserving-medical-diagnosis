"""Check a Docker-deployment run's per-round reported epsilon against an independent recomputation, and report
how many distinct client processes served the hospitals (the fresh-process-per-round question).

What is recomputed and from what. The logged epsilon values are NEVER used to compute the expected ones; they are only
compared against them afterwards. The expected values come from the *recorded configuration*:
  * batch size, local epochs, delta       <- `[tool.flwr.app.config]` in pyproject.toml (the run-config defaults)
  * target epsilon                         <- the EXTRA_RUN_CONFIG recorded in scripts/run_deployment_full_system.sh
  * rounds                                 <- --rounds (the NUM_ROUNDS the run was launched with)
  * per-hospital training-set sizes N      <- data/partitions/hospitals_natural.json
and are then computed as: noise multiplier calibrated for the planned total (R rounds) with the project's calibration
function (a thin wrapper over Opacus's solver), realised sample rate q' = 1/ceil(N/B), steps after round r =
r * ceil(N/B) * local_epochs, a FRESH Opacus RDP accountant per round, mean over the three hospitals (the server logs
the mean of the clients' epsilon_spent). Sanity check: at the planned (q = B/N, floor-steps) the calibrated sigma must
reproduce the target epsilon. Shared with the training code: the calibration wrapper and Opacus's RDP analysis itself.

Usage: uv run python scripts/verify_deployment_epsilon.py --rounds 6 [--dir outputs/deployment_lifecycle]
"""
import argparse, json, math, re, tomllib
from collections import defaultdict
from pathlib import Path

from opacus.accountants import RDPAccountant

from src.privacy.accounting import compute_noise_multiplier, compute_total_steps

REPO = Path(__file__).resolve().parents[1]


def recorded_config() -> dict:
    """Privacy-relevant configuration of the full-system deployment run, read from the files that define it."""
    app = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    wrapper = (REPO / "scripts/run_deployment_full_system.sh").read_text()
    target = float(re.search(r"target-epsilon=([0-9.]+)", wrapper).group(1))
    return {"target_epsilon": target, "delta": float(app["target-delta"]), "batch": int(app["batch-size"]),
            "local_epochs": int(app["local-epochs"])}


def hospital_sizes() -> list[int]:
    part = json.loads((REPO / "data/partitions/hospitals_natural.json").read_text())["hospitals"]
    return [sum(1 for r in recs if r["frozen_split"] == "train") for recs in part.values()]


def expected(rounds: int, cfg: dict | None = None) -> list[float]:
    cfg = cfg or recorded_config()
    target, delta, batch, epochs = cfg["target_epsilon"], cfg["delta"], cfg["batch"], cfg["local_epochs"]
    out = []
    for r in range(1, rounds + 1):
        eps = []
        for n in hospital_sizes():
            sigma = compute_noise_multiplier(target, delta, batch / n, compute_total_steps(n, batch, epochs, rounds))
            ll = math.ceil(n / batch)
            acc = RDPAccountant(); acc.history = [(sigma, 1.0 / ll, ll * epochs * r)]
            eps.append(acc.get_epsilon(delta))
        out.append(sum(eps) / len(eps))
    return out


def calibration_sanity(rounds: int, cfg: dict) -> list[float]:
    """epsilon the calibrated sigma gives at the PLANNED (q = B/N, floor-steps) setting; must be ~ the target."""
    res = []
    for n in hospital_sizes():
        sigma = compute_noise_multiplier(cfg["target_epsilon"], cfg["delta"], cfg["batch"] / n,
                                         compute_total_steps(n, cfg["batch"], cfg["local_epochs"], rounds))
        acc = RDPAccountant()
        acc.history = [(sigma, cfg["batch"] / n, compute_total_steps(n, cfg["batch"], cfg["local_epochs"], rounds))]
        res.append(acc.get_epsilon(cfg["delta"]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--dir", default="outputs/deployment_lifecycle"); a = ap.parse_args()
    d = REPO / a.dir
    text = (d / "run.log").read_text()
    logged = {int(r): float(e) for r, e in re.findall(r"round (\d+): epsilon_spent=([0-9.]+)", text)}
    cfg = recorded_config()
    print("recorded configuration:", cfg, "| rounds:", a.rounds, "| hospital train sizes:", hospital_sizes())
    sane = calibration_sanity(a.rounds, cfg)
    print("calibration sanity (planned q, planned steps) epsilon per hospital:", [round(x, 4) for x in sane])
    assert all(abs(x - cfg["target_epsilon"]) < 0.02 for x in sane), "calibration does not reproduce the target epsilon"
    exp = expected(a.rounds, cfg)
    print(f"{'round':>5s} {'logged (deployment)':>20s} {'independent':>12s} {'abs diff':>10s}")
    ok = len(logged) == a.rounds
    for r in range(1, a.rounds + 1):
        l = logged.get(r); diff = abs(l - exp[r - 1]) if l is not None else float("nan")
        ok &= l is not None and diff < 1e-6
        print(f"{r:5d} {l if l is not None else float('nan'):20.6f} {exp[r-1]:12.6f} {diff:10.2e}")
    increasing = all(logged[r] < logged[r + 1] for r in range(1, a.rounds)) if ok else False
    print("strictly increasing across rounds:", increasing)

    pids = defaultdict(set)
    poll = d / "process_poll.log"
    if poll.exists():
        for line in poll.read_text().splitlines():
            m = re.match(r"\S+ (hospital_\w) +(\d+) .*flwr-clientapp", line)
            if m: pids[m.group(1)].add(m.group(2))
    print("distinct flwr-clientapp PIDs observed per hospital:", {k: len(v) for k, v in sorted(pids.items())})
    (d / "verification.json").write_text(json.dumps({"rounds": a.rounds, "recorded_config": cfg, "logged": logged, "independent": exp,
        "match_within_1e-6": bool(ok), "strictly_increasing": bool(increasing),
        "distinct_clientapp_pids": {k: sorted(v) for k, v in pids.items()}}, indent=2))
    print("RESULT:", "MATCH" if ok and increasing else "MISMATCH")


if __name__ == "__main__":
    main()
