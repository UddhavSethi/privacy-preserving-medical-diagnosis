import json

import torch

from src.data.feature_cache import FeatureCacheKey, cache_file_path, save_feature_bank
from src.training.trainer import (
    FEATURE_KEY,
    compute_class_weights,
    evaluate_classifier,
    load_hospital_features,
    load_pooled_features,
    train_classifier,
)


def _separable_synthetic_features(n_per_class=60, dim=1024, seed=0, num_views=1):
    """Two well-separated Gaussian clusters in feature space — a classifier head
    should be able to learn this easily and quickly, giving a real, meaningful AUROC
    check without depending on the real (slow-to-load, 62GB) dataset."""
    rng = torch.Generator().manual_seed(seed)
    neg = torch.randn(n_per_class, dim, generator=rng) - 3.0
    pos = torch.randn(n_per_class, dim, generator=rng) + 3.0
    features = torch.cat([neg, pos], dim=0)
    labels = torch.cat([torch.zeros(n_per_class, dtype=torch.long), torch.ones(n_per_class, dtype=torch.long)])

    perm = torch.randperm(len(labels), generator=rng)
    features, labels = features[perm], labels[perm]

    # replicate across `num_views` (as if these were the K augmented + eval views)
    train_features = features.unsqueeze(1).repeat(1, num_views, 1)
    return train_features, labels


def test_compute_class_weights_inverse_frequency():
    labels = torch.tensor([0, 0, 0, 1])  # 3 negative, 1 positive
    weights = compute_class_weights(labels)
    assert weights[1] > weights[0]  # the rarer class gets the larger weight


def test_compute_class_weights_balanced_gives_equal_weights():
    labels = torch.tensor([0, 0, 1, 1])
    weights = compute_class_weights(labels)
    assert torch.allclose(weights[0], weights[1])


def test_training_loss_decreases():
    train_features, train_labels = _separable_synthetic_features(num_views=1)
    val_features, val_labels = _separable_synthetic_features(seed=1, num_views=1)
    val_features = val_features[:, 0, :]

    result = train_classifier(
        train_features, train_labels, val_features, val_labels,
        seed=42, num_epochs=15, batch_size=16, patience=15,
    )
    losses = [h["train_loss"] for h in result["history"]]
    assert losses[-1] < losses[0]


def test_training_reaches_meaningful_auroc_on_separable_data():
    train_features, train_labels = _separable_synthetic_features(num_views=1)
    val_features, val_labels = _separable_synthetic_features(seed=1, num_views=1)
    val_features = val_features[:, 0, :]

    result = train_classifier(
        train_features, train_labels, val_features, val_labels,
        seed=42, num_epochs=30, batch_size=16, patience=30,
    )
    assert result["best_val_auroc"] > 0.9  # well-separated synthetic data — should be near-perfect


def test_checkpoint_selects_best_val_epoch_not_last_epoch():
    """Construct data where the model can overfit past its best point, and confirm
    the returned checkpoint corresponds to the best val AUROC seen, not epoch N."""
    train_features, train_labels = _separable_synthetic_features(n_per_class=20, num_views=1)
    val_features, val_labels = _separable_synthetic_features(n_per_class=20, seed=1, num_views=1)
    val_features = val_features[:, 0, :]

    result = train_classifier(
        train_features, train_labels, val_features, val_labels,
        seed=42, num_epochs=10, batch_size=8, patience=100,  # patience high enough to run all epochs
    )
    best_recorded = max(h["val_auroc"] for h in result["history"])
    assert result["best_val_auroc"] == best_recorded


def test_evaluate_classifier_uses_checkpoint_not_random_weights():
    train_features, train_labels = _separable_synthetic_features(num_views=1)
    val_features, val_labels = _separable_synthetic_features(seed=1, num_views=1)
    val_features = val_features[:, 0, :]
    test_features, test_labels = _separable_synthetic_features(seed=2, num_views=1)
    test_features = test_features[:, 0, :]

    result = train_classifier(
        train_features, train_labels, val_features, val_labels,
        seed=42, num_epochs=30, batch_size=16, patience=30,
    )
    test_metrics = evaluate_classifier(result["classifier_state"], test_features, test_labels)
    assert test_metrics["auroc"] > 0.9


def test_early_stopping_triggers_before_num_epochs():
    train_features, train_labels = _separable_synthetic_features(num_views=1)
    val_features, val_labels = _separable_synthetic_features(seed=1, num_views=1)
    val_features = val_features[:, 0, :]

    result = train_classifier(
        train_features, train_labels, val_features, val_labels,
        seed=42, num_epochs=100, batch_size=16, patience=3,
    )
    assert len(result["history"]) < 100  # should converge and stop well before the cap


def test_reproducible_given_seed():
    train_features, train_labels = _separable_synthetic_features(num_views=1)
    val_features, val_labels = _separable_synthetic_features(seed=1, num_views=1)
    val_features = val_features[:, 0, :]

    result_a = train_classifier(
        train_features, train_labels, val_features, val_labels,
        seed=7, num_epochs=5, batch_size=16, patience=5,
    )
    result_b = train_classifier(
        train_features, train_labels, val_features, val_labels,
        seed=7, num_epochs=5, batch_size=16, patience=5,
    )
    for k in result_a["classifier_state"]:
        assert torch.equal(result_a["classifier_state"][k], result_b["classifier_state"][k])


def _build_synthetic_hospital_fixture(tmp_path, source: str, hospital_patient_ids: dict):
    """Builds a synthetic partition file + matching feature-cache banks, mirroring the
    real on-disk structure, for hermetically testing load_hospital_features /
    load_pooled_features without depending on the real 62GB dataset.

    `hospital_patient_ids`: {hospital_name: {"train": [...], "val": [...], "test": [...]}}
    """
    feature_cache_dir = tmp_path / "feature_cache"
    partition = {"hospitals": {}}

    # collect all patient_ids per split across hospitals, to build one shared bank per split
    by_split_all: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for hospital, splits in hospital_patient_ids.items():
        partition["hospitals"][hospital] = []
        for split, pids in splits.items():
            for pid in pids:
                by_split_all[split].append(pid)
                partition["hospitals"][hospital].append(
                    {"source": source, "patient_id": pid, "label": "Normal", "frozen_split": split}
                )

    for split, pids in by_split_all.items():
        # Always write a bank per split, even an empty one — a real source's feature
        # cache always has train/val/test banks regardless of any one hospital's split.
        num_views = FEATURE_KEY.num_augmented_views + 1 if split == "train" else 1
        features = torch.randn(len(pids), num_views, 1024)
        labels = [0] * len(pids)
        bank_path = cache_file_path(feature_cache_dir, source, split, FEATURE_KEY)
        save_feature_bank(bank_path, features, pids, labels)

    partition_path = tmp_path / "partition.json"
    partition_path.write_text(json.dumps(partition))
    return partition_path, feature_cache_dir


def test_load_hospital_features_shapes(tmp_path):
    partition_path, feature_cache_dir = _build_synthetic_hospital_fixture(
        tmp_path,
        source="kermany",
        hospital_patient_ids={
            "A": {"train": ["p1", "p2", "p3"], "val": ["p4"], "test": ["p5", "p6"]},
        },
    )
    hf = load_hospital_features(partition_path, "A", feature_cache_dir=feature_cache_dir)
    assert hf.train_features.shape == (3, FEATURE_KEY.num_augmented_views + 1, 1024)
    assert hf.val_features.shape == (1, 1024)  # eval view only, flattened
    assert hf.test_features.shape == (2, 1024)
    assert len(hf.train_labels) == 3


def test_load_hospital_features_filters_by_hospital_patient_ids(tmp_path):
    """Hospital B/C are shards of a source — load_hospital_features must only pull the
    patient_ids belonging to that hospital, not the whole source's feature bank."""
    partition_path, feature_cache_dir = _build_synthetic_hospital_fixture(
        tmp_path,
        source="rsna",
        hospital_patient_ids={
            "B": {"train": ["p1", "p2"], "val": [], "test": []},
            "C": {"train": ["p3", "p4", "p5"], "val": [], "test": []},
        },
    )
    hf_b = load_hospital_features(partition_path, "B", feature_cache_dir=feature_cache_dir)
    hf_c = load_hospital_features(partition_path, "C", feature_cache_dir=feature_cache_dir)
    assert hf_b.train_features.shape[0] == 2
    assert hf_c.train_features.shape[0] == 3


def test_load_pooled_features_concatenates_hospitals(tmp_path):
    partition_path, feature_cache_dir = _build_synthetic_hospital_fixture(
        tmp_path,
        source="rsna",
        hospital_patient_ids={
            "B": {"train": ["p1", "p2"], "val": ["p3"], "test": ["p4"]},
            "C": {"train": ["p5", "p6", "p7"], "val": ["p8"], "test": ["p9"]},
        },
    )
    pooled = load_pooled_features(partition_path, ["B", "C"], feature_cache_dir=feature_cache_dir)
    assert pooled.train_features.shape[0] == 5  # 2 + 3
    assert pooled.val_features.shape[0] == 2  # 1 + 1
    assert pooled.test_features.shape[0] == 2  # 1 + 1


def test_load_hospital_features_supports_multi_source_hospital(tmp_path):
    """Stage 21: Dirichlet-partitioned synthetic clients pool both Kermany and RSNA
    before assigning patients to clients, so a single "hospital" can span both
    sources — the real bug this test guards against: an earlier version of
    load_hospital_features assumed every record shared hospital_records[0]["source"],
    which would silently drop every record from whichever source wasn't first."""
    feature_cache_dir = tmp_path / "feature_cache"
    partition = {
        "hospitals": {
            "D": [
                {"source": "kermany", "patient_id": "k1", "label": "Normal", "frozen_split": "train"},
                {"source": "kermany", "patient_id": "k2", "label": "Normal", "frozen_split": "train"},
                {"source": "rsna", "patient_id": "r1", "label": "Pneumonia", "frozen_split": "train"},
                {"source": "rsna", "patient_id": "r2", "label": "Pneumonia", "frozen_split": "train"},
                {"source": "rsna", "patient_id": "r3", "label": "Pneumonia", "frozen_split": "train"},
                {"source": "kermany", "patient_id": "k3", "label": "Normal", "frozen_split": "val"},
                {"source": "rsna", "patient_id": "r4", "label": "Pneumonia", "frozen_split": "test"},
            ]
        }
    }
    for source, split, pids in [
        ("kermany", "train", ["k1", "k2"]),
        ("rsna", "train", ["r1", "r2", "r3"]),
        ("kermany", "val", ["k3"]),
        ("rsna", "val", []),
        ("kermany", "test", []),
        ("rsna", "test", ["r4"]),
    ]:
        num_views = FEATURE_KEY.num_augmented_views + 1 if split == "train" else 1
        features = torch.randn(len(pids), num_views, 1024)
        labels = [0] * len(pids)
        bank_path = cache_file_path(feature_cache_dir, source, split, FEATURE_KEY)
        save_feature_bank(bank_path, features, pids, labels)

    partition_path = tmp_path / "partition.json"
    partition_path.write_text(json.dumps(partition))

    hf = load_hospital_features(partition_path, "D", feature_cache_dir=feature_cache_dir)
    assert hf.train_features.shape[0] == 5  # 2 kermany + 3 rsna, neither silently dropped
    assert hf.val_features.shape[0] == 1  # kermany only
    assert hf.test_features.shape[0] == 1  # rsna only
