"""Builds and validates the chest-X-ray input gate (`src/uncertainty/xray_gate.py`,
added 2026-09-02 -- see that module's docstring for the full design rationale
and the real synthetic-only-negatives failure that motivated this approach).

One-time local bootstrap: reads a small set of real, non-X-ray photos already
present on THIS machine (wallpapers, downloaded images -- not a new dataset,
nothing downloaded) as negative examples, mixed with synthetic surrogates for
extra volume, fits a logistic-regression gate against real chest X-ray
features, and saves ONLY the fitted linear weights to
`src/uncertainty/xray_gate_weights.json` -- the committed artifact. The photo
paths below are specific to the machine this was bootstrapped on and are not
expected to exist elsewhere; re-running this script on another machine
requires pointing REAL_PHOTO_PATHS at whatever real, non-medical photos are
locally available there. The app only ever loads the saved weights, never
these source images.

Usage: uv run python scripts/build_xray_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_ood_detector import _synthetic_ood_features  # noqa: E402
from src.data.transforms import build_eval_transform  # noqa: E402
from src.models.densenet_head import DenseNet121Head  # noqa: E402
from src.training.trainer import load_hospital_features  # noqa: E402
from src.uncertainty.xray_gate import fit_xray_gate, predict_is_xray, save_gate_weights  # noqa: E402

PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
WEIGHTS_PATH = REPO_ROOT / "src" / "uncertainty" / "xray_gate_weights.json"
HOSPITALS = ["A", "B", "C"]
IMAGE_SIZE = 224

# One-time local bootstrap set -- real, ordinary photos already on this
# machine (wallpapers, downloads), used only to fit the committed weights
# above, never committed themselves. A handful are deliberately held out
# below to check generalization before trusting the fit.
_PICTURES = Path("/home/uddhav/Pictures")
_DOWNLOADS = Path("/home/uddhav/Downloads")
REAL_PHOTO_PATHS = [
    _DOWNLOADS / "musashi2.jpeg", _DOWNLOADS / "musashi3.jpg", _DOWNLOADS / "_.jpeg",
    _DOWNLOADS / "spiderman, spiderman wallpaper, spiderman brand new day, peter parker.jpeg",
    _DOWNLOADS / "Firefly_Gemini Flash_do 981658.png", _DOWNLOADS / "Spiderman.jpeg",
    _DOWNLOADS / "Spaidr-man.jpeg", _DOWNLOADS / "musashi.jpeg", _DOWNLOADS / "Firefly.jpg",
    _PICTURES / "thorrfin2.png", _PICTURES / "Screenshot.png",
    _PICTURES / "ChatGPT Image Jul 17, 2026, 08_20_24 PM.png", _PICTURES / "Screenshot-2.png",
    _PICTURES / "tody.png", _PICTURES / "Screenshot-1.png", _PICTURES / "Screenshot-3.png",
    _PICTURES / "wallpapers" / "gojo_edit.png", _PICTURES / "wallpapers" / "ss8.png",
    _PICTURES / "wallpapers" / "ss6_clean.png", _PICTURES / "wallpapers" / "ss2.png",
    _PICTURES / "wallpapers" / "spiderman1.jpeg", _PICTURES / "wallpapers" / "ss9_clean.png",
    _PICTURES / "wallpapers" / "ss3.png", _PICTURES / "wallpapers" / "_.jpeg",
    _PICTURES / "wallpapers" / "spiderman2.jpeg", _PICTURES / "wallpapers" / "ss1.png",
    _PICTURES / "wallpapers" / "ss5_clean.png", _PICTURES / "wallpapers" / "ss7_clean.png",
    _PICTURES / "wallpapers" / "ss9.png", _PICTURES / "wallpapers" / "Toji.jpeg",
    _PICTURES / "wallpapers" / "Spiderman_Ubuntu_Wallpaper_1920x1080.jpg",
    _PICTURES / "wallpapers" / "ss3_clean.png", _PICTURES / "wallpapers" / "ss4.png",
    _PICTURES / "wallpapers" / "thorrfin.jpeg", _PICTURES / "wallpapers" / "ss5.png",
    _PICTURES / "wallpapers" / "ss6.png", _PICTURES / "wallpapers" / "ss7.png",
    _PICTURES / "wallpapers" / "strokes.png", _PICTURES / "wallpapers" / "gojo.jpeg",
    _PICTURES / "wallpapers" / "gojo2.jpeg", _PICTURES / "wallpapers" / "batsymbol.jpeg",
]
HELD_OUT_NAMES = {"gojo2.jpeg", "batsymbol.jpeg", "Spiderman.jpeg", "Toji.jpeg", "ss1.png", "musashi.jpeg"}


def _features_for(model: DenseNet121Head, transform, path: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0)
    with torch.no_grad():
        return model.pooled_features(tensor).numpy()[0]


def main() -> None:
    model = DenseNet121Head()
    transform = build_eval_transform(image_size=IMAGE_SIZE)

    print(f"Reading {len(REAL_PHOTO_PATHS)} local real-photo negatives...")
    train_photo_feats, heldout_photo_feats, heldout_names = [], [], []
    for path in REAL_PHOTO_PATHS:
        feats = _features_for(model, transform, path)
        if feats is None:
            print(f"  skip (not found/decode failed): {path}")
            continue
        if path.name in HELD_OUT_NAMES:
            heldout_photo_feats.append(feats)
            heldout_names.append(path.name)
        else:
            train_photo_feats.append(feats)
    train_photo_feats = np.stack(train_photo_feats)
    heldout_photo_feats = np.stack(heldout_photo_feats) if heldout_photo_feats else np.empty((0, 1024))
    print(f"  {len(train_photo_feats)} train, {len(heldout_photo_feats)} held out ({heldout_names})")

    print("Loading real chest X-ray features (all hospitals, train/val/test, both classes)...")
    train_feats, val_feats, test_feats = [], [], []
    for h in HOSPITALS:
        f = load_hospital_features(PARTITION_PATH, h)
        train_feats.append(f.train_features[:, -1, :].numpy())
        val_feats.append(f.val_features.numpy())
        test_feats.append(f.test_features.numpy())
    train_feats = np.concatenate(train_feats, axis=0)
    val_feats = np.concatenate(val_feats, axis=0)
    test_feats = np.concatenate(test_feats, axis=0)
    print(f"  train={len(train_feats)} val={len(val_feats)} test={len(test_feats)}")

    print("Generating synthetic negatives for extra volume (same generator as OPT-5)...")
    synth_neg = np.concatenate([
        _synthetic_ood_features("random_noise", 150, seed=0),
        _synthetic_ood_features("structured_pattern", 150, seed=1),
    ], axis=0)

    neg_train = np.concatenate([synth_neg, train_photo_feats], axis=0)
    print(f"Fitting gate: {len(train_feats)} X-ray positives vs {len(neg_train)} negatives "
          f"({len(synth_neg)} synthetic + {len(train_photo_feats)} real photos)...")
    gate = fit_xray_gate(train_feats, neg_train, seed=42)

    def eval_batch(name: str, features: np.ndarray, expected_is_xray: bool) -> float:
        preds = [predict_is_xray(gate, f).is_xray for f in features]
        acc = float(np.mean([p == expected_is_xray for p in preds])) if len(preds) else float("nan")
        print(f"  {name}: n={len(features)} accuracy={acc:.4f}")
        return acc

    print("\n=== Validation ===")
    eval_batch("real X-ray val (should be True)", val_feats, True)
    eval_batch("real X-ray test (should be True)", test_feats, True)
    eval_batch("HELD-OUT real photos, never trained on (should be False)", heldout_photo_feats, False)

    save_gate_weights(gate, WEIGHTS_PATH)
    print(f"\nSaved gate weights: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
