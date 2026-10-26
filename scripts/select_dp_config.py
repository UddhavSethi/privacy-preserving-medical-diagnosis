"""Pick the DP training configuration from the VALIDATION-only sweep (scripts/exp_dp_sweep.py).

Selection rule, fixed before looking at results: highest macro (mean over hospitals) validation
AUROC of the FINAL-round global model (the model that would be released). Ties within 0.002 are
broken in favour of the more conservative configuration: per_sample_mean loss, no class weights,
then fewer rounds. The whole ranking is written next to the choice so the decision is auditable;
no test split is read here.

Usage: uv run python scripts/select_dp_config.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs/results/exp_dp_sweep_val.jsonl"
OUT = REPO / "outputs/results/selected_dp_config.json"


def macro(h):
    return (h["val_auroc_A"] + h["val_auroc_B"] + h["val_auroc_C"]) / 3


def main() -> None:
    rows = [json.loads(l) for l in SRC.read_text().splitlines()]
    ranked = sorted(rows, key=lambda r: -macro(r["history"][-1]))
    top = macro(ranked[0]["history"][-1])
    near = [r for r in ranked if macro(r["history"][-1]) >= top - 0.002]
    near.sort(key=lambda r: (r["loss"] != "per_sample_mean", r["cw"] != "none", r["rounds"]))
    best = near[0]
    keys = ("stage", "seed", "rounds", "lr", "clip", "cw", "loss", "opt")
    OUT.write_text(json.dumps({
        "selection_rule": "max macro val AUROC at final round; ties within 0.002 -> per_sample_mean, cw=none, fewer rounds",
        "selected": {k: best[k] for k in keys},
        "selected_val_macro_auroc": macro(best["history"][-1]),
        "ranking": [{**{k: r[k] for k in keys}, "val_macro_auroc_final": macro(r["history"][-1]),
                     "val_pooled_auroc_final": r["history"][-1]["val_auroc_pooled"],
                     "val_macro_auroc_best_round": max(macro(h) for h in r["history"]),
                     "epsilon_spent": r["epsilon_spent"]} for r in ranked],
    }, indent=2))
    print(f"{'rank':>4s} {'stage':5s} {'opt':5s} {'loss':22s} {'cw':6s} {'C':>5s} {'lr':>7s} {'R':>3s}  val macro / pooled AUROC (final round)  max eps_spent")
    for i, r in enumerate(ranked, 1):
        h = r["history"][-1]
        print(f"{i:4d} {r['stage']:5s} {r['opt']:5s} {r['loss']:22s} {r['cw']:6s} {r['clip']:5.2f} {r['lr']:7.4f} {r['rounds']:3d}  {macro(h):.4f} / {h['val_auroc_pooled']:.4f}"
              f"   {max(r['epsilon_spent'].values()):.3f}")
    print("\nSELECTED:", {k: best[k] for k in keys})


if __name__ == "__main__":
    main()
