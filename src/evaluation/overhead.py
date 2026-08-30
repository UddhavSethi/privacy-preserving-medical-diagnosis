"""Stage 20 — overhead instrumentation (CLAUDE.md section 11.2): communication
and compute cost as a first-class measured output, not an afterthought.

Reports two numbers per federated round, per client, attributable separately
to each privacy/security layer by comparing runs with a layer on vs. off
(Stage 14's DP, Stage 15's SecAgg — both config-switchable already):
  - bytes transmitted (the actual serialized classifier-head update, ADR-1's
    head-only federated payload — never the frozen backbone)
  - wall-clock time (local training, measured client-side)
"""
from __future__ import annotations

import io
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass

import torch


def classifier_payload_size_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    """Actual serialized size of a classifier-head update — what really
    crosses the wire, via the same `torch.save` mechanism Flower's
    `ArrayRecord` ultimately serializes tensors through
    (`src/federated/serialization.py`)."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getbuffer().nbytes


def theoretical_payload_size_bytes(state_dict: dict[str, torch.Tensor], bytes_per_param: int = 4) -> int:
    """Raw parameter-count-based size (num_params * 4 bytes for fp32) — the
    lower bound the measured payload should sit close to. This stage's own
    testing criterion: "measured payload matches the theoretical
    parameter-count size" (measured is always somewhat larger, due to
    `torch.save`'s pickle/tensor-header overhead — never smaller)."""
    return sum(t.numel() for t in state_dict.values()) * bytes_per_param


@dataclass
class TimingResult:
    wall_clock_seconds: float = 0.0


@contextmanager
def measure_wall_clock():
    """Usage: `with measure_wall_clock() as t: ...; t.wall_clock_seconds`."""
    result = TimingResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.wall_clock_seconds = time.perf_counter() - start


@dataclass
class MemoryResult:
    peak_bytes: int = 0


@contextmanager
def measure_peak_memory():
    """CPU peak memory via `tracemalloc` — this project trains the
    classifier head on cached pooled features (CPU-friendly by design, Stage
    9), so CPU peak memory is the relevant number, not CUDA memory."""
    tracemalloc.start()
    result = MemoryResult()
    try:
        yield result
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.peak_bytes = peak
