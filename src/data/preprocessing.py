"""OpenCV CLAHE preprocessing (ADR-6): CLAHE only — everything else (resize, normalize,
augment) stays in torchvision.

Mandatory practices (ADR-6):
  - CLAHE parameters (`clip_limit`, `tile_grid_size`) are fixed and logged.
  - CLAHE output is precomputed/cached to disk (see scripts/build_clahe_cache.py) so it
    is not a per-run source of nondeterminism or a throughput bottleneck.
  - `cv2.imread` returns BGR; ImageNet-pretrained DenseNet121 expects RGB. This
    conversion is made explicit here (`bgr_to_rgb`) and covered by a dedicated test
    using a synthetic asymmetric-color image (tests/test_preprocessing.py) — a swapped
    channel order is numerically invisible on this project's actual data (grayscale
    chest X-rays, R=G=B) and would otherwise go undetected.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pydicom


@dataclass(frozen=True)
class ClaheParams:
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)

    def cache_key_suffix(self) -> str:
        raw = f"clip{self.clip_limit}_tile{self.tile_grid_size[0]}x{self.tile_grid_size[1]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


def load_jpeg_bgr(path: Path) -> np.ndarray:
    """Load a JPEG via OpenCV. `cv2.imread` returns BGR channel order."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2 failed to decode image: {path}")
    return img


def window_pixel_array_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a raw pixel array to uint8 [0, 255].

    Valid on its own (without RescaleSlope/RescaleIntercept) only because RSNA's
    DICOMs were verified in Stage 3 to have neither present — a source that does
    carry them would need that linear rescale applied to `arr` before this call.
    """
    arr = arr.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        raise ValueError("Zero dynamic range in pixel array — cannot window to 8-bit.")
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return (normalized * 255.0).round().astype(np.uint8)


def load_dicom_as_uint8_bgr(path: Path) -> np.ndarray:
    """Load a DICOM and convert to an 8-bit, 3-channel BGR-shaped array.

    The single grayscale channel is replicated three times to a BGR-shaped array so
    the rest of the pipeline (CLAHE, RGB conversion) is identical for both dataset
    sources.
    """
    ds = pydicom.dcmread(path)
    uint8_img = window_pixel_array_to_uint8(ds.pixel_array)
    return cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2BGR)


def apply_clahe(bgr_image: np.ndarray, params: ClaheParams) -> np.ndarray:
    """Apply CLAHE to the grayscale intensity, replicated back to 3 (identical) channels.

    Chest X-rays are grayscale; CLAHE is applied directly to intensity rather than via
    a color-space luminance channel, matching standard chest X-ray CLAHE usage in the
    literature — this project never applies CLAHE to a genuinely-color image.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=params.clip_limit, tileGridSize=params.tile_grid_size)
    enhanced_gray = clahe.apply(gray)
    return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)


def bgr_to_rgb(bgr_image: np.ndarray) -> np.ndarray:
    """Explicit BGR->RGB conversion (ADR-6) — never assume or skip this."""
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def preprocess_to_rgb(bgr_image: np.ndarray, params: ClaheParams) -> np.ndarray:
    """Full CLAHE pipeline: apply CLAHE, then the explicit BGR->RGB conversion."""
    enhanced_bgr = apply_clahe(bgr_image, params)
    return bgr_to_rgb(enhanced_bgr)


def cache_path_for(cache_dir: Path, source: str, relative_path: str, params: ClaheParams) -> Path:
    """Deterministic cache path: cache_dir/source/param-hash/relative_path(.png).

    A parameter change produces a disjoint directory (via `params.cache_key_suffix()`)
    rather than silently overwriting or reusing stale output under the same path — the
    cache-invalidation risk Stage 6 flags.
    """
    safe_rel = Path(relative_path).with_suffix(".png")
    return cache_dir / source / params.cache_key_suffix() / safe_rel


def save_to_cache(rgb_image: np.ndarray, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite expects BGR; convert back for correct on-disk color values.
    bgr_for_write = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(cache_path), bgr_for_write):
        raise IOError(f"cv2 failed to write cache file: {cache_path}")


def load_from_cache(cache_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read cached image: {cache_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
