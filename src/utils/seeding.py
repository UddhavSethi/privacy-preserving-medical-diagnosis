"""Global seeding utilities.

The data-partition seed and the client-sampling seed are tracked as distinct values
from the general training seed because they are the two most commonly forgotten
sources of run-to-run variance in federated-learning work (CLAUDE.md section 12).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    seed: int
    data_partition_seed: int
    client_sampling_seed: int
    deterministic: bool


def set_global_seed(
    seed: int,
    data_partition_seed: int,
    client_sampling_seed: int,
    deterministic: bool = True,
    warn_only: bool = False,
) -> SeedState:
    """Seed every source of randomness this project uses and return what was applied."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Must be set before any CUDA context touches cuBLAS; harmless if CUDA is unused.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return SeedState(
        seed=seed,
        data_partition_seed=data_partition_seed,
        client_sampling_seed=client_sampling_seed,
        deterministic=deterministic,
    )


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn`: give each worker a distinct, reproducible seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A seeded `torch.Generator` for `DataLoader(generator=...)`."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
