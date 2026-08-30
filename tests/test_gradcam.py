"""Stage 18 — Grad-CAM. Real trained checkpoint, real CLAHE-cached images, not
mocks — this stage's own flagged risk (ADR-1's frozen backbone vs. Grad-CAM's
need for gradients through that backbone's activations) can only be verified
empirically, per docs/IMPLEMENTATION_PLAN.md's Stage 18 write-up: "must be
verified, not assumed."
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.preprocessing import ClaheParams, cache_path_for, load_from_cache
from src.data.transforms import build_eval_transform
from src.explain.gradcam import (
    NORMAL_CLASS_INDEX,
    PNEUMONIA_CLASS_INDEX,
    compute_gradcam_heatmap,
    generate_overlay,
    get_target_layer,
)
from src.models.densenet_head import DenseNet121Head

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "outputs" / "checkpoints" / "centralized_baseline" / "natural_seed42.pt"
CLAHE_CACHE_DIR = REPO_ROOT / "data" / "clahe_cache"

PNEUMONIA_IMAGE = "train/PNEUMONIA/BACTERIA-1025587-0001.jpeg"
NORMAL_IMAGE = "train/NORMAL/NORMAL-1031320-0001.jpeg"

pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="requires a trained centralized_baseline checkpoint (Stage 12)"
)


def _load_trained_model() -> DenseNet121Head:
    model = DenseNet121Head()
    state = torch.load(CHECKPOINT, weights_only=True)
    model.classifier.load_state_dict(state)
    return model


def _load_real_image(relative_path: str) -> np.ndarray:
    path = cache_path_for(CLAHE_CACHE_DIR, "kermany", relative_path, ClaheParams())
    if not path.exists():
        pytest.skip(f"CLAHE cache image not present: {path}")
    return load_from_cache(path)


def test_target_layer_resolves_to_the_expected_batchnorm():
    model = DenseNet121Head(pretrained=False)
    layer = get_target_layer(model)
    assert layer is model.features.norm5
    assert isinstance(layer, torch.nn.BatchNorm2d)


def test_gradcam_gradients_flow_through_the_frozen_backbone():
    """The core ADR-1 interaction this stage must verify, not assume: even
    though every backbone parameter has requires_grad=False, Grad-CAM must
    still be able to compute a gradient at the target layer's *activation*.
    Uses a real trained checkpoint and the model's own predicted class — not
    an untrained random-init model, whose classifier head has no learned
    signal at all and can legitimately produce a degenerate (all-negative,
    ReLU'd-to-zero) map by pure chance regardless of whether gradients are
    flowing correctly (found via this test's own first, flaky version)."""
    model = _load_trained_model()
    image = _load_real_image(PNEUMONIA_IMAGE)
    input_tensor = build_eval_transform(224)(image).unsqueeze(0)
    with torch.no_grad():
        predicted_class = int(model(input_tensor).argmax(dim=1).item())

    heatmap = compute_gradcam_heatmap(model, input_tensor, target_class=predicted_class)

    assert heatmap.shape == (224, 224)
    assert np.isfinite(heatmap).all()
    assert heatmap.std() > 1e-6, "heatmap has zero variance — gradients likely never reached the target layer"
    assert not np.allclose(heatmap, heatmap.flat[0]), "heatmap is uniform, not class-discriminative"


def test_gradcam_overlay_renders_for_both_classes_on_real_images():
    """Functional rendering claim only (correct shape/dtype/value-range) — NOT
    a non-degeneracy claim for every class. A confidently-classified image
    queried against the class it was confidently *rejected* as can validly
    produce a heatmap that's all (or mostly) zero after Grad-CAM's ReLU: there
    is no localized positive evidence for a class the model is very sure
    isn't present anywhere in the image. That was found empirically here, not
    assumed — see test_gradcam_is_non_degenerate_for_the_predicted_class for
    the actual non-degeneracy claim, which uses the class the model has real
    evidence for."""
    model = _load_trained_model()
    pneumonia_image = _load_real_image(PNEUMONIA_IMAGE)
    normal_image = _load_real_image(NORMAL_IMAGE)

    for image, target_class in [
        (pneumonia_image, PNEUMONIA_CLASS_INDEX),
        (pneumonia_image, NORMAL_CLASS_INDEX),
        (normal_image, PNEUMONIA_CLASS_INDEX),
        (normal_image, NORMAL_CLASS_INDEX),
    ]:
        result = generate_overlay(model, image, target_class=target_class)
        assert result.heatmap.shape == (224, 224)
        assert 0.0 <= result.heatmap.min() and result.heatmap.max() <= 1.0 + 1e-6
        assert result.overlay_rgb.shape == (224, 224, 3)
        assert result.overlay_rgb.dtype == np.uint8


def test_gradcam_is_non_degenerate_for_the_predicted_class():
    """The real non-degeneracy claim (CLAUDE.md/plan's own testing criterion):
    for the class the model actually predicts — where it has real positive
    evidence somewhere in the image — the heatmap must be non-degenerate."""
    model = _load_trained_model()
    image = _load_real_image(PNEUMONIA_IMAGE)

    input_tensor = build_eval_transform(224)(image).unsqueeze(0)
    with torch.no_grad():
        predicted_class = int(model(input_tensor).argmax(dim=1).item())
    assert predicted_class == PNEUMONIA_CLASS_INDEX, "expected the model to correctly classify this real example"

    heatmap = generate_overlay(model, image, target_class=predicted_class).heatmap
    assert heatmap.std() > 1e-6, "non-degenerate heatmap required for the predicted class"


def test_gradcam_is_class_discriminative_on_a_real_pneumonia_case():
    """The two class-target heatmaps for the SAME image should differ if the
    model's prediction genuinely depends on which class is being explained —
    a real (if soft) check that Grad-CAM isn't just returning the same
    generic saliency map regardless of target class."""
    model = _load_trained_model()
    image = _load_real_image(PNEUMONIA_IMAGE)

    pneumonia_heatmap = generate_overlay(model, image, target_class=PNEUMONIA_CLASS_INDEX).heatmap
    normal_heatmap = generate_overlay(model, image, target_class=NORMAL_CLASS_INDEX).heatmap

    assert not np.allclose(pneumonia_heatmap, normal_heatmap, atol=1e-3)


def test_gradcam_does_not_concentrate_purely_on_image_borders():
    """A known chest X-ray failure mode (CLAUDE.md/plan's own flagged risk):
    models sometimes latch onto scanner artifacts or text overlays near the
    image edges rather than the lung fields. Automatable proxy: the outer
    margin (10% border) should not hold a disproportionate share of the
    heatmap's total activation energy for a real pneumonia-positive case."""
    model = _load_trained_model()
    image = _load_real_image(PNEUMONIA_IMAGE)
    heatmap = generate_overlay(model, image, target_class=PNEUMONIA_CLASS_INDEX).heatmap

    h, w = heatmap.shape
    margin_h, margin_w = int(h * 0.1), int(w * 0.1)
    border_mask = np.ones_like(heatmap, dtype=bool)
    border_mask[margin_h : h - margin_h, margin_w : w - margin_w] = False

    total_energy = heatmap.sum()
    border_energy = heatmap[border_mask].sum()
    border_area_fraction = border_mask.sum() / border_mask.size

    if total_energy > 1e-6:
        border_energy_fraction = border_energy / total_energy
        # The border covers ~36% of pixels at a 10% margin; a model NOT
        # latching onto borders should not concentrate meaningfully more of
        # its total activation there than that area's own share.
        assert border_energy_fraction < border_area_fraction + 0.15
