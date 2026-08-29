import pytest

from src.data.splitting import assert_no_patient_overlap, grouped_stratified_split


def _make_records(num_normal_patients=100, num_pneumonia_patients=60, images_per_patient=2):
    records = []
    for i in range(num_normal_patients):
        for j in range(images_per_patient):
            records.append({"patient_id": f"normal-{i}", "label": "Normal", "image": f"{i}-{j}"})
    for i in range(num_pneumonia_patients):
        for j in range(images_per_patient):
            records.append({"patient_id": f"pneumonia-{i}", "label": "Pneumonia", "image": f"{i}-{j}"})
    return records


def test_zero_patient_overlap_across_splits():
    records = _make_records()
    splits = grouped_stratified_split(records, val_frac=0.15, test_frac=0.15, seed=1000)
    assert_no_patient_overlap(splits)  # raises AssertionError on failure


def test_split_is_deterministic_given_seed():
    records = _make_records()
    a = grouped_stratified_split(records, val_frac=0.15, test_frac=0.15, seed=1000)
    b = grouped_stratified_split(records, val_frac=0.15, test_frac=0.15, seed=1000)
    for name in ("train", "val", "test"):
        ids_a = sorted(r["patient_id"] for r in a[name])
        ids_b = sorted(r["patient_id"] for r in b[name])
        assert ids_a == ids_b


def test_different_seeds_produce_different_splits():
    records = _make_records()
    a = grouped_stratified_split(records, val_frac=0.15, test_frac=0.15, seed=1000)
    b = grouped_stratified_split(records, val_frac=0.15, test_frac=0.15, seed=2000)
    ids_a = sorted(r["patient_id"] for r in a["test"])
    ids_b = sorted(r["patient_id"] for r in b["test"])
    assert ids_a != ids_b


def test_class_balance_preserved_per_split():
    records = _make_records(num_normal_patients=200, num_pneumonia_patients=100)
    splits = grouped_stratified_split(records, val_frac=0.2, test_frac=0.2, seed=1000)

    overall_ratio = 200 / 100  # normal : pneumonia patients
    for name, recs in splits.items():
        patients = {r["patient_id"]: r["label"] for r in recs}
        normal = sum(1 for label in patients.values() if label == "Normal")
        pneumonia = sum(1 for label in patients.values() if label == "Pneumonia")
        assert pneumonia > 0 and normal > 0, f"{name} split lost an entire class"
        ratio = normal / pneumonia
        assert overall_ratio * 0.5 < ratio < overall_ratio * 1.5, (
            f"{name} split class ratio {ratio:.2f} diverges too far from overall {overall_ratio:.2f}"
        )


def test_all_records_preserved():
    records = _make_records()
    splits = grouped_stratified_split(records, val_frac=0.15, test_frac=0.15, seed=1000)
    total = sum(len(recs) for recs in splits.values())
    assert total == len(records)


def test_conflicting_label_for_same_patient_raises():
    records = [
        {"patient_id": "p1", "label": "Normal", "image": "a"},
        {"patient_id": "p1", "label": "Pneumonia", "image": "b"},
    ]
    with pytest.raises(ValueError, match="conflicting labels"):
        grouped_stratified_split(records)


def test_invalid_fractions_raise():
    with pytest.raises(ValueError):
        grouped_stratified_split(_make_records(), val_frac=0.6, test_frac=0.5)
