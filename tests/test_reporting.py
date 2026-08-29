import json

import pytest

from src.evaluation.reporting import (
    aggregate_metrics_over_seeds,
    aggregate_over_seeds,
    load_results,
    save_results,
)


def test_aggregate_over_seeds_mean_and_std():
    agg = aggregate_over_seeds([0.8, 0.85, 0.9])
    assert agg.mean == pytest.approx((0.8 + 0.85 + 0.9) / 3)
    assert agg.n_seeds == 3
    assert agg.std > 0


def test_aggregate_over_seeds_records_n_seeds_even_if_below_recommended():
    # A single run during development is legitimate but must be visibly flagged as n=1,
    # not silently presented as if it were the final multi-seed number.
    agg = aggregate_over_seeds([0.8])
    assert agg.n_seeds == 1
    assert agg.std == pytest.approx(0.0)


def test_aggregate_metrics_over_seeds():
    metrics_per_seed = [
        {"auroc": 0.80, "auprc": 0.70, "threshold": 0.5},
        {"auroc": 0.82, "auprc": 0.72, "threshold": 0.5},
        {"auroc": 0.78, "auprc": 0.68, "threshold": 0.5},
    ]
    result = aggregate_metrics_over_seeds(metrics_per_seed)
    assert result["auroc"]["mean"] == pytest.approx(0.80)
    assert result["auroc"]["n_seeds"] == 3
    assert result["threshold"]["mean"] == pytest.approx(0.5)


def test_aggregate_metrics_over_seeds_empty_raises():
    with pytest.raises(ValueError):
        aggregate_metrics_over_seeds([])


def test_save_and_load_results_round_trip(tmp_path):
    results = {"auroc": {"mean": 0.83, "std": 0.02, "n_seeds": 3}}
    path = tmp_path / "results.json"
    save_results(results, path)

    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk == results

    loaded = load_results(path)
    assert loaded == results


def test_save_results_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "results.json"
    save_results({"a": 1}, path)
    assert path.exists()
