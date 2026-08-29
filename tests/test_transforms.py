import numpy as np
import torch

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform, build_train_transform


def test_eval_transform_output_shape_and_dtype():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (300, 250, 3), dtype=np.uint8)
    transform = build_eval_transform(image_size=224)
    out = transform(image)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_normalization_matches_expected_formula():
    # Constant-color image: uint8 255 scales to 1.0, so every normalized channel
    # value should equal (1.0 - mean) / std for that channel.
    image = np.full((224, 224, 3), 255, dtype=np.uint8)
    transform = build_eval_transform(image_size=224)
    out = transform(image)

    for c in range(3):
        expected = (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert torch.allclose(out[c], torch.full_like(out[c], expected), atol=1e-4)


def test_eval_transform_is_deterministic_no_augmentation():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    transform = build_eval_transform(image_size=224)

    out_a = transform(image)
    out_b = transform(image)
    assert torch.equal(out_a, out_b)  # zero randomness at eval


def test_train_transform_reproducible_given_seed():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    transform = build_train_transform(image_size=224)

    torch.manual_seed(999)
    out_a = transform(image)
    torch.manual_seed(999)
    out_b = transform(image)
    assert torch.equal(out_a, out_b)


def test_train_transform_differs_across_seeds():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    transform = build_train_transform(image_size=224)

    torch.manual_seed(1)
    out_a = transform(image)
    torch.manual_seed(2)
    out_b = transform(image)
    assert not torch.equal(out_a, out_b)


def test_train_transform_output_shape_and_dtype():
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (180, 220, 3), dtype=np.uint8)
    transform = build_train_transform(image_size=224)
    out = transform(image)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32
