import numpy as np
import pytest

from src.experiments.evaluation_protocol import threshold_for_specificity


def test_threshold_for_specificity_attains_target_on_validation_and_uses_no_positives():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    s = rng.normal(size=5000) + 1.2 * y
    thr = threshold_for_specificity(y, s, 0.9)
    spec = float((s[y == 0] < thr).mean())
    assert 0.9 <= spec <= 0.905
    # positives' scores must not change the threshold (only negatives define specificity)
    s2 = s.copy(); s2[y == 1] += 5.0
    assert threshold_for_specificity(y, s2, 0.9) == pytest.approx(thr)


def test_threshold_is_monotone_in_target_specificity():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000); s = rng.normal(size=2000)
    assert threshold_for_specificity(y, s, 0.8) < threshold_for_specificity(y, s, 0.95)
