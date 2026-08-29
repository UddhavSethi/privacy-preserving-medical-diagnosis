import subprocess
import sys
from pathlib import Path

import torch

from src.utils.seeding import make_generator, set_global_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_same_seed_same_process_bit_identical():
    set_global_seed(seed=42, data_partition_seed=1, client_sampling_seed=2)
    a = torch.randn(100)
    set_global_seed(seed=42, data_partition_seed=1, client_sampling_seed=2)
    b = torch.randn(100)
    assert torch.equal(a, b)


def test_different_seed_diverges():
    set_global_seed(seed=42, data_partition_seed=1, client_sampling_seed=2)
    a = torch.randn(100)
    set_global_seed(seed=43, data_partition_seed=1, client_sampling_seed=2)
    b = torch.randn(100)
    assert not torch.equal(a, b)


def test_same_seed_across_processes_bit_identical():
    script = (
        "import torch\n"
        "from src.utils.seeding import set_global_seed\n"
        "set_global_seed(seed=42, data_partition_seed=1, client_sampling_seed=2)\n"
        "print(torch.randn(10).tolist())\n"
    )
    out1 = subprocess.check_output([sys.executable, "-c", script], cwd=REPO_ROOT)
    out2 = subprocess.check_output([sys.executable, "-c", script], cwd=REPO_ROOT)
    assert out1 == out2


def test_generator_is_reproducible():
    g1 = make_generator(7)
    g2 = make_generator(7)
    a = torch.randn(10, generator=g1)
    b = torch.randn(10, generator=g2)
    assert torch.equal(a, b)


def test_seed_state_records_distinct_seeds():
    state = set_global_seed(seed=42, data_partition_seed=1000, client_sampling_seed=2000)
    assert state.seed == 42
    assert state.data_partition_seed == 1000
    assert state.client_sampling_seed == 2000
