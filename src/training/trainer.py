"""Non-federated head training loop (Stage 11 local baselines, Stage 12 centralized).

Trains only the classifier (Stage 8's frozen backbone never runs during this loop) on
Stage 9's cached pooled features — not raw images — for the ~8.3x per-step speedup
that is what makes a 3-seed x multiple-hospital x multiple-partition-regime sweep
feasible on a 4GB laptop GPU.

Class imbalance is handled explicitly via inverse-frequency loss weighting
(`compute_class_weights`), not left implicit (CLAUDE.md section 7) — chosen over
oversampling because it needs no data duplication and is the standard default for
this kind of imbalance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.feature_cache import FeatureCacheKey, cache_file_path, load_feature_bank
from src.evaluation.metrics import compute_metrics
from src.models.densenet_head import DenseNet121Head
from src.utils.seeding import set_global_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_CACHE_DIR = REPO_ROOT / "data" / "feature_cache"

FEATURE_KEY = FeatureCacheKey(
    image_size=224,
    num_augmented_views=5,
    rotation_degrees=10.0,
    brightness=0.1,
    contrast=0.1,
)


@dataclass
class HospitalFeatures:
    train_features: torch.Tensor  # (N_train, num_views+1, 1024)
    train_labels: torch.Tensor  # (N_train,)
    val_features: torch.Tensor  # (N_val, 1024) — eval-style view only
    val_labels: torch.Tensor
    test_features: torch.Tensor  # (N_test, 1024)
    test_labels: torch.Tensor


def load_hospital_features(
    partition_path: Path, hospital: str, feature_cache_dir: Path = FEATURE_CACHE_DIR
) -> HospitalFeatures:
    """Load one hospital's cached features for train/val/test, filtered from the
    relevant source's full feature bank by patient_id membership (a hospital may be
    only a patient-disjoint shard of a source, e.g. Hospitals B/C are RSNA shards)."""
    partition = json.loads(partition_path.read_text())
    hospital_records = partition["hospitals"][hospital]
    if not hospital_records:
        raise ValueError(f"No records for hospital {hospital} in {partition_path}")
    source = hospital_records[0]["source"]

    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for r in hospital_records:
        by_split[r["frozen_split"]].append(r["patient_id"])

    def _gather(split: str, all_views: bool):
        bank_path = cache_file_path(feature_cache_dir, source, split, FEATURE_KEY)
        bank = load_feature_bank(bank_path)
        id_to_idx = {rid: i for i, rid in enumerate(bank["record_ids"])}
        idx = [id_to_idx[pid] for pid in by_split[split] if pid in id_to_idx]
        feats = bank["features"][idx]  # (n, V, 1024)
        labels = torch.tensor([bank["labels"][i] for i in idx])
        if not all_views:
            feats = feats[:, -1, :]  # the eval-style view is always the last index
        return feats, labels

    train_features, train_labels = _gather("train", all_views=True)
    val_features, val_labels = _gather("val", all_views=False)
    test_features, test_labels = _gather("test", all_views=False)

    return HospitalFeatures(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
    )


def compute_class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts.sum() / (num_classes * counts.clamp(min=1))


def train_classifier(
    train_features: torch.Tensor,  # (N, V, 1024) — V>=1 augmented/eval views
    train_labels: torch.Tensor,  # (N,)
    val_features: torch.Tensor,  # (N_val, 1024)
    val_labels: torch.Tensor,
    seed: int,
    num_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 5,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Trains a fresh `DenseNet121Head`'s classifier on cached features, cycling
    through augmented views per sample per epoch, with early stopping on val AUROC.
    Returns the training history, best val AUROC, and the best classifier state dict
    (for checkpointing) — does NOT evaluate on test; callers do that separately so
    test-set contamination from tuning decisions stays impossible by construction.
    """
    set_global_seed(seed=seed, data_partition_seed=seed, client_sampling_seed=seed)
    generator = torch.Generator().manual_seed(seed)

    model = DenseNet121Head().to(device)
    class_weights = compute_class_weights(train_labels).to(device)
    opt = torch.optim.Adam(model.classifier.parameters(), lr=lr)

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels_np = val_labels.numpy()

    n = train_features.shape[0]
    num_views = train_features.shape[1]

    best_val_auroc = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start : start + batch_size]
            view_idx = torch.randint(0, num_views, (len(batch_idx),), generator=generator)
            x = train_features[batch_idx, view_idx]
            y = train_labels[batch_idx]

            out = model.classifier(x)
            loss = F.cross_entropy(out, y, weight=class_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(batch_idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_probs = F.softmax(model.classifier(val_features), dim=1)[:, 1].cpu().numpy()
        val_metrics = compute_metrics(val_labels_np, val_probs)
        history.append(
            {"epoch": epoch, "train_loss": epoch_loss, "val_auroc": val_metrics.auroc}
        )

        val_auroc = val_metrics.auroc if val_metrics.auroc == val_metrics.auroc else -1.0  # NaN guard
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.classifier.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return {"history": history, "best_val_auroc": best_val_auroc, "classifier_state": best_state}


def evaluate_classifier(
    classifier_state: dict,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Loads a saved classifier state onto a fresh model and reports test metrics."""
    model = DenseNet121Head().to(device)
    model.classifier.load_state_dict(classifier_state)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model.classifier(test_features.to(device)), dim=1)[:, 1].cpu().numpy()
    return compute_metrics(test_labels.numpy(), probs).to_dict()
