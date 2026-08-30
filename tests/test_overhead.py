"""Stage 20 — overhead instrumentation. Stage 20's own testing criteria:
measured payload matches the theoretical parameter-count size; measurements
are stable/sane across repeats.
"""
import time

import torch

from src.evaluation.overhead import (
    classifier_payload_size_bytes,
    measure_peak_memory,
    measure_wall_clock,
    theoretical_payload_size_bytes,
)
from src.models.densenet_head import DenseNet121Head


def test_measured_payload_is_close_to_theoretical_for_the_real_classifier_head():
    """ADR-1's real federated payload — the head-only classifier, not the
    frozen backbone. Measured (torch.save, with pickle/header overhead) must
    never be smaller than the raw parameter-count theoretical size, and
    should be close to it (small constant overhead, not a large multiple)."""
    model = DenseNet121Head(pretrained=False)
    state = model.classifier.state_dict()

    measured = classifier_payload_size_bytes(state)
    theoretical = theoretical_payload_size_bytes(state)

    assert measured >= theoretical
    assert measured < theoretical * 1.1, (
        f"measured payload ({measured} bytes) is more than 10% larger than "
        f"the theoretical size ({theoretical} bytes) — unexpectedly large serialization overhead"
    )


def test_theoretical_payload_matches_known_parameter_count():
    """Stage 8's own validated trainable-parameter count (262,914) at fp32."""
    model = DenseNet121Head(pretrained=False)
    state = model.classifier.state_dict()
    total_params = sum(t.numel() for t in state.values())
    assert total_params == 262_914
    assert theoretical_payload_size_bytes(state) == 262_914 * 4


def test_payload_size_scales_with_parameter_count():
    small = {"w": torch.zeros(10)}
    large = {"w": torch.zeros(10_000)}
    assert classifier_payload_size_bytes(large) > classifier_payload_size_bytes(small)


def test_wall_clock_measures_a_real_delay():
    with measure_wall_clock() as timing:
        time.sleep(0.05)
    assert timing.wall_clock_seconds >= 0.04  # allow scheduler slack, never exactly 0.05


def test_wall_clock_is_near_zero_for_a_trivial_block():
    with measure_wall_clock() as timing:
        pass
    assert 0.0 <= timing.wall_clock_seconds < 0.01


def test_peak_memory_reflects_a_real_allocation():
    with measure_peak_memory() as mem_small:
        _ = bytearray(1_000)
    with measure_peak_memory() as mem_large:
        _ = bytearray(10_000_000)
    assert mem_large.peak_bytes > mem_small.peak_bytes
