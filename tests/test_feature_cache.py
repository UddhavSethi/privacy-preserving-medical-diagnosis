import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.feature_cache import (
    FeatureCacheKey,
    cache_file_path,
    compute_pooled_features,
    load_feature_bank,
    save_feature_bank,
)
from src.data.transforms import build_eval_transform
from src.models.densenet_head import DenseNet121Head


def _tiny_model():
    # pretrained=False avoids a network download in tests; the backbone is still
    # deterministic and frozen either way, which is all these tests need.
    model = DenseNet121Head(pretrained=False)
    model.eval()
    return model


def test_cached_feature_matches_live_forward_pass():
    model = _tiny_model()
    image = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    transform = build_eval_transform(224)

    cached = compute_pooled_features(model, image, transform)

    with torch.no_grad():
        live = model.pooled_features(transform(image).unsqueeze(0)).squeeze(0)

    assert torch.allclose(cached, live, atol=1e-6)


def test_cache_key_hash_differs_across_configs():
    k1 = FeatureCacheKey(image_size=224, num_augmented_views=5, rotation_degrees=10.0, brightness=0.1, contrast=0.1)
    k2 = FeatureCacheKey(image_size=224, num_augmented_views=3, rotation_degrees=10.0, brightness=0.1, contrast=0.1)
    assert k1.hash_suffix() != k2.hash_suffix()

    path1 = cache_file_path(Path("/cache"), "kermany", "train", k1)
    path2 = cache_file_path(Path("/cache"), "kermany", "train", k2)
    assert path1 != path2


def test_feature_bank_save_and_load_round_trip(tmp_path):
    features = torch.randn(3, 2, 1024)
    record_ids = ["p1", "p2", "p3"]
    labels = [0, 1, 0]

    path = tmp_path / "bank.pt"
    save_feature_bank(path, features, record_ids, labels)
    loaded = load_feature_bank(path)

    assert torch.equal(loaded["features"], features)
    assert loaded["record_ids"] == record_ids
    assert loaded["labels"] == labels


def test_head_training_on_cached_feature_matches_live_training_no_augmentation():
    """Stage 9's core correctness claim: with augmentation disabled, training the
    classifier from a cached (eval-style) feature must be mathematically identical to
    training it from a live full-model forward pass, since the frozen backbone is a
    deterministic function of the same input either way."""
    base_model = _tiny_model()
    image = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    eval_transform = build_eval_transform(224)

    # Path A: precompute the feature once, then train only the classifier on it.
    model_a = copy.deepcopy(base_model)
    feature = compute_pooled_features(model_a, image, eval_transform)
    model_a.train()
    torch.manual_seed(42)  # controls Dropout's random mask
    out_a = model_a.classifier(feature.unsqueeze(0))
    loss_a = F.cross_entropy(out_a, torch.tensor([1]))
    loss_a.backward()
    torch.optim.SGD([p for p in model_a.parameters() if p.requires_grad], lr=0.1).step()

    # Path B: the ordinary live forward pass through the full model.
    model_b = copy.deepcopy(base_model)
    model_b.train()
    x = eval_transform(image).unsqueeze(0)
    torch.manual_seed(42)  # same Dropout mask as path A
    out_b = model_b(x)
    loss_b = F.cross_entropy(out_b, torch.tensor([1]))
    loss_b.backward()
    torch.optim.SGD([p for p in model_b.parameters() if p.requires_grad], lr=0.1).step()

    for p_a, p_b in zip(model_a.classifier.parameters(), model_b.classifier.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-5)


def test_cached_feature_training_step_is_faster_than_live_full_model():
    """Soft sanity check, not a strict regression gate (timing is environment-
    dependent): training from a cached feature should be meaningfully faster than a
    live full-model forward+backward pass, which is the entire point of Stage 9."""
    model = _tiny_model()
    image = np.random.default_rng(2).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    eval_transform = build_eval_transform(224)
    feature = compute_pooled_features(model, image, eval_transform)
    x = eval_transform(image).unsqueeze(0)

    n_iters = 20

    model.train()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = model.classifier(feature.unsqueeze(0))
        F.cross_entropy(out, torch.tensor([1])).backward()
        model.zero_grad()
    cached_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = model(x)
        F.cross_entropy(out, torch.tensor([1])).backward()
        model.zero_grad()
    live_time = time.perf_counter() - t0

    assert cached_time < live_time


