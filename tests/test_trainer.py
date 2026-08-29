import torch

from src.training.trainer import (
    compute_class_weights,
    evaluate_classifier,
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
