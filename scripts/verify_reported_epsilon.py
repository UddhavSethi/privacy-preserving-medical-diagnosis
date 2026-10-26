"""Recompute the reported epsilon from first principles, independent of any training run.

For each hospital: the noise multiplier is calibrated from (target epsilon, delta, batch/N, planned steps)
exactly as the clients do, then the epsilon actually implied by what Opacus's Poisson loader realises is
recomputed with a FRESH accountant — realised sample rate q' = 1/ceil(N/B) and T = ceil(N/B) * local_epochs *
rounds steps — and compared with the cumulative epsilon the corrected SecAgg+/DP runs logged. The tighter PRV
accountant is shown for reference. This checks the accounting arithmetic and the F11 fix; it does not (and
cannot) check the Poisson-subsampled RDP analysis itself, which is taken from Opacus.

Usage: uv run python scripts/verify_reported_epsilon.py
"""
import json, math, re
from pathlib import Path

from opacus.accountants import PRVAccountant, RDPAccountant

from src.privacy.accounting import compute_noise_multiplier, compute_total_steps

REPO = Path(__file__).resolve().parents[1]
BATCH, ROUNDS, EPOCHS, DELTA, TARGET = 32, 20, 1, 1e-5, 4.0


def main() -> None:
    part = json.loads((REPO / "data/partitions/hospitals_natural.json").read_text())["hospitals"]
    rows, eps_rdp = {}, []
    for h, recs in part.items():
        n = sum(1 for r in recs if r["frozen_split"] == "train")
        sigma = compute_noise_multiplier(TARGET, DELTA, BATCH / n, compute_total_steps(n, BATCH, EPOCHS, ROUNDS))
        loader_len = math.ceil(n / BATCH)
        q, steps = 1.0 / loader_len, loader_len * EPOCHS * ROUNDS
        rdp, prv = RDPAccountant(), PRVAccountant()
        rdp.history, prv.history = [(sigma, q, steps)], [(sigma, q, steps)]
        e_rdp, e_prv = rdp.get_epsilon(DELTA), prv.get_epsilon(DELTA)
        eps_rdp.append(e_rdp)
        rows[h] = {"n_train": n, "sigma": sigma, "q_realised": q, "steps": steps, "eps_rdp": e_rdp, "eps_prv": e_prv}
        print(f"hospital {h}: N={n:5d} sigma={sigma:.4f} q'={q:.6f} T={steps:5d}  eps_RDP={e_rdp:.4f}  eps_PRV={e_prv:.4f}")
    mean_rdp = sum(eps_rdp) / len(eps_rdp)
    logged = []
    for f in sorted((REPO / "outputs/ablation_logs").glob("secagg_dp_eps4_v2_seed*.log")):
        logged.append(float(re.findall(r"round 20: epsilon_spent=([0-9.]+)", f.read_text())[-1]))
    print(f"mean eps_RDP over hospitals (recomputed) = {mean_rdp:.6f}   logged round-20 mean (3 corrected runs) = {logged}")
    assert logged and all(abs(x - mean_rdp) < 1e-6 for x in logged), "logged epsilon does not match the recomputation"
    assert max(eps_rdp) <= TARGET, "an RDP epsilon exceeds the target"
    out = {"hospitals": rows, "mean_eps_rdp": mean_rdp, "logged_round20_mean_eps": logged, "target": TARGET, "delta": DELTA}
    (REPO / "outputs/results/verified_epsilon.json").write_text(json.dumps(out, indent=2))
    print("OK: logged epsilon reproduced from first principles; no hospital exceeds the target")


if __name__ == "__main__":
    main()
