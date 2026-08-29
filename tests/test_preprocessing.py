from pathlib import Path

import numpy as np
import pytest

from src.data.preprocessing import (
    ClaheParams,
    apply_clahe,
    bgr_to_rgb,
    cache_path_for,
    load_from_cache,
    preprocess_to_rgb,
    save_to_cache,
    window_pixel_array_to_uint8,
)


def test_bgr_to_rgb_swaps_channels_on_asymmetric_image():
    # Distinct values per channel so a swap is numerically detectable — real chest
    # X-ray data (grayscale, R=G=B) would NOT catch this bug, which is exactly why
    # ADR-6 requires an explicit synthetic test rather than trusting real-data checks.
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[..., 0] = 10   # B
    bgr[..., 1] = 20   # G
    bgr[..., 2] = 30   # R

    rgb = bgr_to_rgb(bgr)

    assert np.all(rgb[..., 0] == 30)  # R
    assert np.all(rgb[..., 1] == 20)  # G
    assert np.all(rgb[..., 2] == 10)  # B


def test_bgr_to_rgb_round_trip():
    bgr = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    rgb = bgr_to_rgb(bgr)
    back_to_bgr = bgr_to_rgb(rgb)  # BGR2RGB and RGB2BGR are the same channel-swap op
    assert np.array_equal(bgr, back_to_bgr)


def test_clahe_deterministic_given_fixed_params_and_input():
    rng = np.random.default_rng(42)
    bgr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    params = ClaheParams(clip_limit=2.0, tile_grid_size=(8, 8))

    out_a = apply_clahe(bgr, params)
    out_b = apply_clahe(bgr, params)

    assert np.array_equal(out_a, out_b)  # byte-identical, not just close


def test_clahe_output_is_grayscale_replicated():
    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    out = apply_clahe(bgr, ClaheParams())
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])


def test_window_pixel_array_to_uint8_maps_full_range():
    arr = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    out = window_pixel_array_to_uint8(arr)
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255
    # monotonic: higher input -> higher (or equal) output
    assert out[0, 0] < out[1, 1]


def test_window_pixel_array_zero_dynamic_range_raises():
    arr = np.full((4, 4), 500, dtype=np.uint16)
    with pytest.raises(ValueError, match="dynamic range"):
        window_pixel_array_to_uint8(arr)


def test_cache_path_differs_across_param_sets():
    p1 = cache_path_for(Path("/cache"), "kermany", "train/NORMAL/x.jpeg", ClaheParams(clip_limit=2.0))
    p2 = cache_path_for(Path("/cache"), "kermany", "train/NORMAL/x.jpeg", ClaheParams(clip_limit=4.0))
    assert p1 != p2  # a parameter change must not silently reuse the same cache path


def test_cache_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    bgr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    rgb = preprocess_to_rgb(bgr, ClaheParams())

    cache_path = cache_path_for(tmp_path, "kermany", "train/NORMAL/x.jpeg", ClaheParams())
    assert not cache_path.exists()  # cache miss

    save_to_cache(rgb, cache_path)
    assert cache_path.exists()  # now a cache hit for subsequent runs

    reloaded = load_from_cache(cache_path)
    assert np.array_equal(reloaded, rgb)  # lossless round trip (PNG)
