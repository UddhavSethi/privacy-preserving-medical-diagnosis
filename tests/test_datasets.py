import numpy as np
import torch

from src.data.datasets import ChestXrayDataset, build_dataloader
from src.data.preprocessing import ClaheParams, cache_path_for, save_to_cache
from src.data.transforms import build_eval_transform


def _write_fake_cache_image(cache_dir, source, relative_path, params, size=32, seed=0):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    cache_path = cache_path_for(cache_dir, source, relative_path, params)
    save_to_cache(rgb, cache_path)


def _make_records(n=4):
    return [
        {
            "relative_path": f"img_{i}.jpeg",
            "label": "Normal" if i % 2 == 0 else "Pneumonia",
            "patient_id": f"p{i}",
        }
        for i in range(n)
    ]


def test_dataset_returns_correct_shape_and_label(tmp_path):
    params = ClaheParams()
    records = _make_records(4)
    for r in records:
        _write_fake_cache_image(tmp_path, "kermany", r["relative_path"], params)

    ds = ChestXrayDataset(
        records, source="kermany", transform=build_eval_transform(224),
        cache_dir=tmp_path, clahe_params=params,
    )
    assert len(ds) == 4

    image, label = ds[0]
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert label == 0  # Normal

    _, label1 = ds[1]
    assert label1 == 1  # Pneumonia


def test_dataset_without_transform_returns_raw_array(tmp_path):
    params = ClaheParams()
    records = _make_records(1)
    _write_fake_cache_image(tmp_path, "kermany", records[0]["relative_path"], params, size=16)

    ds = ChestXrayDataset(records, source="kermany", transform=None, cache_dir=tmp_path, clahe_params=params)
    image, _ = ds[0]
    assert isinstance(image, np.ndarray)
    assert image.shape == (16, 16, 3)


def test_dataloader_seeded_shuffle_is_deterministic(tmp_path):
    params = ClaheParams()
    records = _make_records(10)
    for r in records:
        _write_fake_cache_image(tmp_path, "kermany", r["relative_path"], params)

    ds = ChestXrayDataset(
        records, source="kermany", transform=build_eval_transform(224),
        cache_dir=tmp_path, clahe_params=params,
    )

    loader_a = build_dataloader(ds, batch_size=2, shuffle=True, seed=123)
    loader_b = build_dataloader(ds, batch_size=2, shuffle=True, seed=123)

    labels_a = [batch[1].tolist() for batch in loader_a]
    labels_b = [batch[1].tolist() for batch in loader_b]
    assert labels_a == labels_b


def test_dataloader_different_seeds_differ(tmp_path):
    params = ClaheParams()
    records = _make_records(10)
    for r in records:
        _write_fake_cache_image(tmp_path, "kermany", r["relative_path"], params)

    ds = ChestXrayDataset(
        records, source="kermany", transform=build_eval_transform(224),
        cache_dir=tmp_path, clahe_params=params,
    )

    loader_a = build_dataloader(ds, batch_size=10, shuffle=True, seed=1)
    loader_b = build_dataloader(ds, batch_size=10, shuffle=True, seed=2)

    order_a = next(iter(loader_a))[1].tolist()
    order_b = next(iter(loader_b))[1].tolist()
    assert order_a != order_b


def test_dataloader_no_shuffle_preserves_order(tmp_path):
    params = ClaheParams()
    records = _make_records(6)
    for r in records:
        _write_fake_cache_image(tmp_path, "kermany", r["relative_path"], params)

    ds = ChestXrayDataset(
        records, source="kermany", transform=build_eval_transform(224),
        cache_dir=tmp_path, clahe_params=params,
    )
    loader = build_dataloader(ds, batch_size=6, shuffle=False, seed=0)
    labels = next(iter(loader))[1].tolist()
    expected = [0 if i % 2 == 0 else 1 for i in range(6)]
    assert labels == expected
