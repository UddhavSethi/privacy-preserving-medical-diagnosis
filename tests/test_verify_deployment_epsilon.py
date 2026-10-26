"""The deployment-epsilon verifier recomputes epsilon from the recorded configuration; pin it to the recorded evidence
(docs/deployment_privacy_accounting_evidence.md) and to basic properties, so a change to the calibration, the
accountant, or the configuration files is noticed."""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if not (REPO / "data/partitions/hospitals_natural.json").exists():
    pytest.skip("partition file not available", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("verify_deployment_epsilon", REPO / "scripts/verify_deployment_epsilon.py")
vde = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vde)

# per-round mean epsilon recorded from the six-round Docker deployment run and reproduced by the recomputation
RECORDED_6_ROUNDS = [2.943209, 3.230412, 3.455059, 3.649697, 3.829521, 3.991884]


def test_recorded_configuration_is_read_from_the_defining_files():
    cfg = vde.recorded_config()
    assert cfg == {"target_epsilon": 4.0, "delta": 1e-5, "batch": 32, "local_epochs": 1}


def test_recomputed_epsilon_matches_the_deployment_evidence_and_accumulates():
    exp = vde.expected(6)
    assert all(a < b for a, b in zip(exp, exp[1:]))  # accumulates, never resets
    assert exp == pytest.approx(RECORDED_6_ROUNDS, abs=1e-5)
    assert exp[-1] <= 4.0  # final cumulative epsilon does not exceed the target


def test_calibration_reproduces_the_target_epsilon_at_the_planned_setting():
    cfg = vde.recorded_config()
    assert all(abs(e - cfg["target_epsilon"]) < 0.02 for e in vde.calibration_sanity(6, cfg))


def test_expected_epsilon_depends_on_the_configuration_not_on_any_log(tmp_path):
    base = vde.recorded_config()
    looser = vde.expected(6, {**base, "target_epsilon": 8.0})
    assert looser[-1] > vde.expected(6, base)[-1] + 3  # a different target gives a clearly different answer
