"""Audit: like-for-like evaluation of every already-trained checkpoint (no training).

Motivation (docs/audit_2026-10.md, finding F1): ablation row 1 ("local, per-hospital
average") is the mean over hospitals of each local model scored on ITS OWN hospital's
test split, while rows 2-5 are scored on the POOLED test split. Those are different
test distributions (Hospital A = pediatric Kermany, AUROC ~0.985 in-domain), so the
row-1-vs-row-3 comparison is not like-for-like. This script scores every checkpoint
on (a) each hospital's own test split and (b) the pooled test split, so the same
model is compared on the same data. Test labels are used only to report metrics --
no threshold or model choice is made here.

Usage: uv run python scripts/audit_per_hospital_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.metrics import compute_metrics
from src.evaluation.reporting import aggregate_over_seeds
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import load_hospital_features

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT = REPO_ROOT / "outputs" / "checkpoints"
OUT = REPO_ROOT / "outputs" / "results" / "audit_per_hospital_eval.json"
PARTITIONS = {"natural": REPO_ROOT / "data" / "partitions" / "hospitals_natural.json",
              "balanced": REPO_ROOT / "data" / "partitions" / "hospitals_natural_balanced.json"}
HOSPITALS = ["A", "B", "C"]
SEEDS = [42, 123, 2024]


def _probs(ckpt: Path, x: torch.Tensor) -> np.ndarray:
    m = DenseNet121Head()
    m.classifier.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    with torch.no_grad():
        return F.softmax(m.classifier(x), dim=1)[:, 1].numpy()


def main() -> None:
    result: dict = {"prevalence": {}, "scores": {}}
    for regime, ppath in PARTITIONS.items():
        data = {h: load_hospital_features(ppath, h) for h in HOSPITALS}
        pooled_x = torch.cat([data[h].test_features for h in HOSPITALS])
        pooled_y = torch.cat([data[h].test_labels for h in HOSPITALS]).numpy()
        result["prevalence"][regime] = {
            h: {
                s: {"n": int(len(getattr(data[h], f"{s}_labels"))),
                    "pneumonia_fraction": float(getattr(data[h], f"{s}_labels").float().mean())}
                for s in ("train", "val", "test")
            } for h in HOSPITALS
        }

        def score(ckpt_by_seed: dict[int, Path]) -> dict:
            """Per-hospital-test and pooled-test AUROC / sens@90%spec, aggregated over seeds."""
            per_seed = []
            for seed, ck in ckpt_by_seed.items():
                if not ck.exists():
                    continue
                row = {}
                for h in HOSPITALS:
                    m = compute_metrics(data[h].test_labels.numpy(), _probs(ck, data[h].test_features))
                    row[f"auroc_{h}"], row[f"sens@90spec_{h}"] = m.auroc, m.sensitivity_at_target_specificity
                m = compute_metrics(pooled_y, _probs(ck, pooled_x))
                row["auroc_pooled"], row["sens@90spec_pooled"] = m.auroc, m.sensitivity_at_target_specificity
                per_seed.append(row)
            if not per_seed:
                return {}
            return {k: aggregate_over_seeds([r[k] for r in per_seed]).to_dict() for k in per_seed[0]}

        configs: dict[str, dict[int, Path]] = {}
        for h in HOSPITALS:
            configs[f"local_{h}"] = {s: CKPT / "local_baseline" / f"{regime}_{h}_seed{s}.pt" for s in SEEDS}
        configs["centralized"] = {s: CKPT / "centralized_baseline" / f"{regime}_seed{s}.pt" for s in SEEDS}
        configs["fedavg"] = {s: CKPT / "ablation" / f"fedavg_{regime}_seed{s}.pt" for s in SEEDS}
        if regime == "natural":
            configs["fedavg_secagg"] = {s: CKPT / "ablation" / f"secagg_seed{s}.pt" for s in SEEDS}
            for eps in ("1.0", "2.0", "4.0", "8.0"):
                configs[f"fedavg_dp_eps{eps}"] = {s: CKPT / "ablation" / f"dp_eps{eps}_seed{s}.pt" for s in SEEDS}
        result["scores"][regime] = {name: score(c) for name, c in configs.items()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"written: {OUT}\n")
    for regime in PARTITIONS:
        print(f"== {regime}: pneumonia fraction (n) per hospital/split")
        for h in HOSPITALS:
            p = result["prevalence"][regime][h]
            print("  ", h, "  ".join(f"{s}: {p[s]['pneumonia_fraction']:.3f} (n={p[s]['n']})" for s in p))
        print(f"== {regime}: AUROC mean over seeds  [own-hospital test A | B | C | pooled test]")
        for name, r in result["scores"][regime].items():
            if r:
                print(f"  {name:18s} " + " | ".join(f"{r[f'auroc_{k}']['mean']:.4f}" for k in ("A", "B", "C", "pooled")))


if __name__ == "__main__":
    main()
