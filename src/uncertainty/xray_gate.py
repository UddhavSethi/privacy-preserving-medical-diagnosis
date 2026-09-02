"""Chest X-ray input gate, added 2026-09-02 following a real live finding: the
existing per-hospital IsolationForest OOD detectors (OPT-5, src/uncertainty/
ood_detector.py) are one-class density estimators calibrated on only two
sources -- they answer "does this look like Kermany/RSNA specifically," not
"is this a chest X-ray at all." A real chest X-ray sourced any other way
(different equipment, different export pipeline) can trip the same flag as a
photo of Spider-Man, which made a hard block on that signal unsafe (see
docs/adr1_groupnorm_fallback.md sections 14-15).

A simpler color-based heuristic (mean saturation) was tried first and rejected
before writing any code here: tested directly against real images, a black-
and-white non-X-ray photo (an anime wallpaper) came out MORE grayscale than a
real X-ray, and a moderately colorful non-X-ray photo (a logo wallpaper) came
out about as saturated as the real X-ray -- color alone doesn't separate these
(see the session's own record for the numbers).

This module trains a real, supervised binary classifier: real chest X-ray
pooled features (thousands already available, both classes -- diagnosis is
irrelevant to this question) as positives, vs. non-X-ray negatives.

**A synthetic-only negative class (reusing `scripts/build_ood_detector.py`'s
own `_synthetic_ood_features` -- random pixel noise + structured colored-shape
patterns, run through the REAL frozen backbone) was tried FIRST and measured
to fail**, not assumed to work: held-out accuracy on the synthetic
distributions themselves was a perfect 1.0, but two real non-X-ray photos held
out from training were both misclassified as X-rays (one at 92% confidence) --
synthetic noise/shapes don't share real-photo feature statistics closely
enough for the learned boundary to transfer. Adding a genuinely small number
(35) of real, locally-available photos (already on the training machine --
wallpapers, downloaded images; NOT a new external dataset, no download, no new
dependency) to the negative class, mixed with the synthetic ones for extra
volume, fixed this completely: 6 different real photos held out from THIS
training (never seen at all) were then all correctly classified, all at high
confidence (p_xray < 0.005), while both known real chest X-rays stayed
correctly classified as X-rays (p_xray = 1.0). Small amounts of real photo
diversity generalize; synthetic surrogates alone do not.

**The bootstrap photos are not committed to this repository** -- many are
copyrighted wallpapers/fan art, inappropriate to redistribute in a public
repo, and referencing a personal `~/Pictures` path from committed code isn't
portable to another machine anyway. `save_gate_weights`/`load_gate_weights`
persist only the fitted linear model's ~1,025 floating-point numbers (coef_ +
intercept_) as plain JSON -- reproducible and portable without the source
images, the same way a trained checkpoint is committed without needing its
training images alongside it. See `scripts/build_xray_gate.py` for the
one-time local bootstrap that produced the committed weights.

Logistic regression, not IsolationForest: this task has real negative
examples to learn from (not just one-class density estimation), so a
discriminative classifier draws an actual decision boundary between the two
classes rather than only modeling what's "usual." scikit-learn is already
pinned -- no new dependency.

**Known limitation, stated honestly:** 35 real photos is a small, ad hoc,
non-diverse negative set (mostly wallpapers/fan art), not a curated benchmark.
It generalized cleanly to 6 different held-out real photos, which is real
evidence, not proof against every possible input. Treat this as a genuine
gate for clearly-non-medical images, not a formal guarantee.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class XrayGateResult:
    is_xray: bool
    p_xray: float  # calibrated-ish probability from the logistic model, not a formal calibration

    def to_dict(self) -> dict:
        return asdict(self)


def fit_xray_gate(
    xray_features: np.ndarray, non_xray_features: np.ndarray, seed: int = 42
) -> LogisticRegression:
    """Fits a logistic-regression gate. `xray_features`: real chest X-ray
    pooled features (any hospital, any class). `non_xray_features`: synthetic
    non-X-ray surrogate pooled features (see module docstring)."""
    X = np.concatenate([xray_features, non_xray_features], axis=0)
    y = np.concatenate([
        np.ones(len(xray_features), dtype=int),
        np.zeros(len(non_xray_features), dtype=int),
    ])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(X, y)
    return clf


def predict_is_xray(gate: LogisticRegression, features: np.ndarray, threshold: float = 0.5) -> XrayGateResult:
    """`features`: a single example's pooled feature vector, shape (1024,) or (1, 1024)."""
    features = np.asarray(features).reshape(1, -1)
    p_xray = float(gate.predict_proba(features)[0, 1])
    return XrayGateResult(is_xray=p_xray >= threshold, p_xray=p_xray)


def save_gate_weights(gate: LogisticRegression, path: Path) -> None:
    """Saves only the fitted linear weights (coef_, intercept_, classes_) as
    plain JSON -- a few KB, no training images involved. Deliberately NOT
    pickling the sklearn estimator object (version-fragile, and unnecessary
    for a two-parameter linear model)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "coef": gate.coef_[0].tolist(),
        "intercept": float(gate.intercept_[0]),
        "classes": gate.classes_.tolist(),
    }))


def load_gate_weights(path: Path) -> LogisticRegression:
    """Inverse of `save_gate_weights` -- reconstructs a usable LogisticRegression
    from saved coefficients, no training data or images required at load time."""
    data = json.loads(path.read_text())
    gate = LogisticRegression()
    gate.coef_ = np.array([data["coef"]])
    gate.intercept_ = np.array([data["intercept"]])
    gate.classes_ = np.array(data["classes"])
    gate.n_features_in_ = len(data["coef"])
    return gate
