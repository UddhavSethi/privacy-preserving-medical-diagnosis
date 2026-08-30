"""Parameter serialization for federated updates (Stage 13).

Head-only (ADR-1): only the trainable classifier's state dict is ever
serialized/transmitted between client and server — the frozen backbone (~7M
parameters) never crosses a hospital boundary, only the ~263K-parameter head.
"""
from __future__ import annotations

from flwr.app import ArrayRecord


def classifier_state_to_array_record(state_dict: dict) -> ArrayRecord:
    return ArrayRecord(state_dict)


def array_record_to_classifier_state(array_record: ArrayRecord) -> dict:
    return array_record.to_torch_state_dict()
