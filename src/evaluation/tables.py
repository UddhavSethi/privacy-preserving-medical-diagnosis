"""Stage 21 — ablation table generation (CLAUDE.md section 11.1: "the ablation
table is the paper"). Aggregates mean +/- std AUROC over >=3 seeds for every
row, from the authoritative source in each case: Stages 11/12's saved JSON
results for the non-federated baselines (rows 1-2), and MLflow (Stage 20/21's
instrumentation) for the federated rows (3-5) and the Dirichlet supplement —
CLAUDE.md section 12: "a result that is not in MLflow does not exist."
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow

from src.evaluation.reporting import aggregate_over_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "outputs" / "results"
MLFLOW_TRACKING_URI = "sqlite:////mnt/storage/pneumonia-detection/mlruns.db"
MLFLOW_EXPERIMENT = "federated_ablation"


def load_local_baseline_row(regime: str) -> dict:
    """Row 1 (local, single-hospital) — averages the three hospitals' own
    mean AUROC (each already aggregated over >=3 seeds by Stage 11) into one
    headline number; per-hospital detail remains in the saved JSON, since
    there is no single "local" model to report a pooled number for."""
    data = json.loads((RESULTS_DIR / "local_baseline.json").read_text())
    hospital_means = [data[regime][h]["auroc"]["mean"] for h in ("A", "B", "C")]
    hospital_stds = [data[regime][h]["auroc"]["std"] for h in ("A", "B", "C")]
    return {
        "row": "1. Local (per-hospital, averaged)",
        "regime": regime,
        "mean_auroc": sum(hospital_means) / len(hospital_means),
        "std_auroc": sum(hospital_stds) / len(hospital_stds),  # mean of per-hospital stds, not seed-pooled
        "n_seeds": data[regime]["A"]["auroc"]["n_seeds"],
        "per_hospital": {h: data[regime][h]["auroc"]["mean"] for h in ("A", "B", "C")},
    }


def load_centralized_baseline_row(regime: str) -> dict:
    """Row 2 (centralized, privacy-free ceiling) — pooled test AUROC,
    already the single comparable number Stage 12 reports."""
    data = json.loads((RESULTS_DIR / "centralized_baseline.json").read_text())
    pooled = data[regime]["pooled_test"]["auroc"]
    return {
        "row": "2. Centralized (pooled)",
        "regime": regime,
        "mean_auroc": pooled["mean"],
        "std_auroc": pooled["std"],
        "n_seeds": pooled["n_seeds"],
    }


def _placeholder_row(row_label: str) -> dict:
    return {"row": row_label, "mean_auroc": None, "std_auroc": None, "n_seeds": 0}


def _query_runs(filter_string: str) -> list:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        return []
    return client.search_runs(
        [experiment.experiment_id], filter_string=f"attributes.status = 'FINISHED' and {filter_string}"
    )


def _aggregate_runs(row_label: str, runs: list) -> dict:
    """Pulls `final_pooled_test_auroc` from each run (one per seed) and
    aggregates. Returns a placeholder row (mean/std None, n_seeds 0) if no
    matching runs exist yet — the campaign may not have reached that
    configuration, which stays visible in the table rather than silently
    absent."""
    values = [
        r.data.metrics["final_pooled_test_auroc"]
        for r in runs
        if "final_pooled_test_auroc" in r.data.metrics
    ]
    if not values:
        return _placeholder_row(row_label)
    agg = aggregate_over_seeds(values)
    return {"row": row_label, "mean_auroc": agg.mean, "std_auroc": agg.std, "n_seeds": agg.n_seeds}


def query_fedavg_row(row_label: str, partition_path: str) -> dict:
    """Rows 3 and Dirichlet: filters on the exact `partition_path` PARAM
    (logged verbatim by server_app.py) and `dp_enabled = 'False'`, rather
    than matching on run *name* — run-name prefix matching is fragile here
    since row 5's DP runs share row 3's exact name prefix
    ("fedavg_natural_seed42" vs. "fedavg_natural_seed42_dp_eps1.0"), which a
    naive SQL LIKE '...seed%' pattern would incorrectly also match. Param
    filtering sidesteps that ambiguity entirely, found while first drafting
    this module's queries, not by inspection."""
    runs = _query_runs(f"params.partition_path = '{partition_path}' and params.dp_enabled = 'False'")
    return _aggregate_runs(row_label, runs)


def query_dp_row(epsilon: float) -> dict:
    runs = _query_runs(f"params.dp_enabled = 'True' and params.target_epsilon = '{epsilon}'")
    return _aggregate_runs(f"5. FedAvg + DP (epsilon={epsilon})", runs)


def query_secagg_row() -> dict:
    """SecAgg's own run-name prefix ("secagg_seed") is unambiguous — no
    other row's naming scheme starts with it, so a name-based LIKE match is
    safe here (server_app_secagg.py also doesn't log a partition_path or
    dp_enabled param to filter on instead)."""
    runs = _query_runs("tags.mlflow.runName LIKE 'secagg_seed%'")
    return _aggregate_runs("4. FedAvg + SecAgg", runs)


def build_ablation_table() -> list[dict]:
    """Assembles every row currently available (partial results while the
    campaign is still running are fine — n_seeds makes incompleteness
    visible, per Stage 10's own established convention)."""
    rows = []

    for regime in ("natural", "balanced"):
        rows.append(load_local_baseline_row(regime))
        rows.append(load_centralized_baseline_row(regime))

    for regime, filename in [("natural", "hospitals_natural.json"), ("balanced", "hospitals_natural_balanced.json")]:
        partition_path = str(REPO_ROOT / "data" / "partitions" / filename)
        rows.append(query_fedavg_row(f"3. FedAvg ({regime})", partition_path))

    for epsilon in (1.0, 2.0, 4.0, 8.0):
        rows.append(query_dp_row(epsilon))

    rows.append(query_secagg_row())

    for alpha in (0.1, 1.0):
        partition_path = str(REPO_ROOT / "data" / "partitions" / f"hospitals_dirichlet_alpha{alpha}.json")
        rows.append(query_fedavg_row(f"Dirichlet (supplementary, alpha={alpha})", partition_path))

    return rows


def format_markdown_table(rows: list[dict]) -> str:
    lines = ["| Row | Mean AUROC | Std | N seeds |", "|---|---|---|---|"]
    for r in rows:
        mean = f"{r['mean_auroc']:.4f}" if r.get("mean_auroc") is not None else "—"
        std = f"{r['std_auroc']:.4f}" if r.get("std_auroc") is not None else "—"
        lines.append(f"| {r['row']} | {mean} | {std} | {r.get('n_seeds', 0)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    table = build_ablation_table()
    print(format_markdown_table(table))
