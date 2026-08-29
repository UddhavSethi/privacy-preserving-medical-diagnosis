import statistics

from src.data.partitioning import (
    assert_no_patient_overlap_across_hospitals,
    dirichlet_partition,
    natural_shard_rsna,
    per_client_stats,
)


def _make_records(num_normal_patients=200, num_pneumonia_patients=100, images_per_patient=1):
    records = []
    for i in range(num_normal_patients):
        for j in range(images_per_patient):
            records.append({"patient_id": f"normal-{i}", "label": "Normal", "image": f"{i}-{j}"})
    for i in range(num_pneumonia_patients):
        for j in range(images_per_patient):
            records.append({"patient_id": f"pneumonia-{i}", "label": "Pneumonia", "image": f"{i}-{j}"})
    return records


def test_natural_shard_zero_patient_overlap():
    records = _make_records()
    shards = natural_shard_rsna(records, seed=1000)
    assert_no_patient_overlap_across_hospitals(shards)


def test_natural_shard_deterministic_given_seed():
    records = _make_records()
    a = natural_shard_rsna(records, seed=1000)
    b = natural_shard_rsna(records, seed=1000)
    for name in ("B", "C"):
        assert sorted(r["patient_id"] for r in a[name]) == sorted(r["patient_id"] for r in b[name])


def test_natural_shard_roughly_balanced_and_class_stratified():
    records = _make_records(num_normal_patients=200, num_pneumonia_patients=100)
    shards = natural_shard_rsna(records, seed=1000)
    stats = per_client_stats(shards)
    # Roughly half the patients in each shard.
    assert 130 <= stats["B"]["num_patients"] <= 170
    assert 130 <= stats["C"]["num_patients"] <= 170
    # Class ratio preserved per shard (overall is 2:1 normal:pneumonia).
    for name in ("B", "C"):
        counts = stats[name]["label_counts"]
        ratio = counts["Normal"] / counts["Pneumonia"]
        assert 1.5 < ratio < 2.5


def test_dirichlet_zero_patient_overlap():
    records = _make_records()
    parts = dirichlet_partition(records, num_clients=5, alpha=0.5, seed=2000)
    assert_no_patient_overlap_across_hospitals(parts)


def test_dirichlet_all_patients_preserved():
    records = _make_records()
    total_patients = len({r["patient_id"] for r in records})
    parts = dirichlet_partition(records, num_clients=5, alpha=0.5, seed=2000)
    assigned_patients = sum(len({r["patient_id"] for r in recs}) for recs in parts.values())
    assert assigned_patients == total_patients


def test_dirichlet_deterministic_given_seed():
    records = _make_records()
    a = dirichlet_partition(records, num_clients=4, alpha=0.3, seed=2000)
    b = dirichlet_partition(records, num_clients=4, alpha=0.3, seed=2000)
    for name in a:
        assert sorted(r["patient_id"] for r in a[name]) == sorted(r["patient_id"] for r in b[name])


def test_dirichlet_alpha_sweep_changes_skew():
    """Lower alpha should produce more skewed (higher-variance) per-client label
    proportions than higher alpha — the whole point of the Dirichlet partitioner."""
    records = _make_records(num_normal_patients=300, num_pneumonia_patients=300)

    def pneumonia_fraction_variance(alpha: float) -> float:
        parts = dirichlet_partition(records, num_clients=8, alpha=alpha, seed=2000)
        fractions = []
        for recs in parts.values():
            if not recs:
                continue
            n_pneumonia = sum(1 for r in recs if r["label"] == "Pneumonia")
            fractions.append(n_pneumonia / len(recs))
        return statistics.variance(fractions)

    low_alpha_variance = pneumonia_fraction_variance(0.05)
    high_alpha_variance = pneumonia_fraction_variance(100.0)
    assert low_alpha_variance > high_alpha_variance
