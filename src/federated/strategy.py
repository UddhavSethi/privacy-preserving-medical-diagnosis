"""FedAvg strategy configuration (Stage 13). CLAUDE.md section 8: strategy is FedAvg —
do not substitute FedProx, FedBN, or anything else without approval. Uses Flower's
built-in `FedAvg` directly rather than a custom subclass.
"""
from __future__ import annotations

from flwr.serverapp.strategy import FedAvg


def build_fedavg_strategy(fraction_evaluate: float, min_available_nodes: int) -> FedAvg:
    return FedAvg(fraction_evaluate=fraction_evaluate, min_available_nodes=min_available_nodes)
