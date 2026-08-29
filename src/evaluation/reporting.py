"""Multi-seed result aggregation, serialization, and MLflow logging (Stage 10).
CLAUDE.md section 11.2: report mean +/- std over at least 3 seeds — single-run numbers
are not credible in FL, where run-to-run variance is high.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

MIN_RECOMMENDED_SEEDS = 3


@dataclass(frozen=True)
class AggregatedMetric:
    mean: float
    std: float
    n_seeds: int
    values: list

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate_over_seeds(values: list) -> AggregatedMetric:
    """Does not raise below MIN_RECOMMENDED_SEEDS — a single run during development
    is a legitimate intermediate state — but `n_seeds` is always recorded, so a table
    built from too few seeds is visible in the data, not silently presented as final."""
    arr = np.asarray(values, dtype=float)
    return AggregatedMetric(
        mean=float(np.nanmean(arr)),
        std=float(np.nanstd(arr)),
        n_seeds=len(values),
        values=[float(v) for v in values],
    )


def aggregate_metrics_over_seeds(metrics_per_seed: list) -> dict:
    """`metrics_per_seed`: a list of flat dicts (e.g. `Metrics.to_dict()`), one per
    seed. Returns {metric_name: AggregatedMetric.to_dict()} for every numeric field
    common to all of them."""
    if not metrics_per_seed:
        raise ValueError("metrics_per_seed must be non-empty")

    numeric_keys = [
        k for k, v in metrics_per_seed[0].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    result = {}
    for key in numeric_keys:
        values = [m[key] for m in metrics_per_seed]
        result[key] = aggregate_over_seeds(values).to_dict()
    return result


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def log_metrics_to_mlflow(metrics: dict, prefix: str = "") -> None:
    """Log a flat dict of numeric metrics to the currently active MLflow run."""
    import mlflow

    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mlflow.log_metric(f"{prefix}{key}" if prefix else key, value)
