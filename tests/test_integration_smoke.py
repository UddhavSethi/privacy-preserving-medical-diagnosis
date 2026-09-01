"""Stage 22 — the integration smoke test CLAUDE.md section 11.3 explicitly
names: "a 2-client, single-round smoke test that runs end to end." This
project's real, approved architecture is fixed at 3 hospitals (not a
variable N) — `server_app.py`'s own `HOSPITALS` constant, `client_app.py`'s
`PARTITION_TO_HOSPITAL` mapping, and every stage's live validation this
session all commit to exactly 3. A literal 2-client variant would exercise a
shape the real system never actually runs (the production strategy's
`min_available_nodes=len(HOSPITALS)` would simply block it from completing).
This test uses the real 3-hospital setup for one real round instead, which
is the more faithful version of the requirement's own stated purpose: catch
the class of integration bug no unit test can (several were found only by a
real `flwr run` this session — Message-API wiring mismatches, FAB packaging,
absolute-path requirements — never by a unit test alone).

A real subprocess invocation, not a mock — matches the discipline used for
every Flower-protocol claim this session (Stages 13-21).

Isolation: overrides `mlflow-tracking-uri` to a throwaway sqlite file in
`tmp_path` rather than letting it fall through to pyproject.toml's real
default (`mlruns.db`) — found necessary the hard way, not by inspection:
this test's own early runs (before this fix) silently wrote extra
"fedavg_natural_seed42"-named runs into the real campaign's MLflow
experiment, contaminating `src.evaluation.tables`' seed-aggregated query for
that exact configuration (it filters on `partition_path`/`dp_enabled`, not
`seed`, so any additional matching run — regardless of which seed it
claims — gets folded into the average). The stray runs were found and
deleted from the real `mlruns.db` after the fact; this fix stops it from
happening again on every future run of this test (including in CI).
"""
import subprocess
from pathlib import Path

import pytest

from tests.conftest import kill_local_simulation_daemon, kill_process_tree

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = REPO_ROOT / "data" / "partitions" / "hospitals_natural.json"
FEATURE_CACHE_DIR = REPO_ROOT / "data" / "feature_cache"

pytestmark = pytest.mark.skipif(
    not PARTITION_PATH.exists() or not FEATURE_CACHE_DIR.exists(),
    reason="requires the real frozen partition + feature cache (Stages 4-9)",
)


def test_single_round_federated_smoke_end_to_end(tmp_path):
    """Two real, independent robustness gaps found and fixed 2026-09-01 (see
    tests/conftest.py's own module docstring for which one actually caused
    this session's flakiness — it wasn't the one you'd guess):

    1. `Popen` + `kill_process_tree`, not `subprocess.run(..., timeout=...)` —
       the latter only kills the direct child on timeout, orphaning whatever
       daemon tree `flwr run` started.
    2. `kill_local_simulation_daemon()` before AND after — `flwr run`'s local
       SuperLink deliberately detaches into its own session (by design, for
       reuse across runs), so even (1)'s process-group kill can't reach it if
       something does hang. Calling this before the run guards against a
       stale daemon from an earlier, differently-configured invocation
       (exactly what actually broke this test this session — see
       pyproject.toml's `[tool.flwr.app.components]` history); calling it
       after prevents this run's own daemon from doing the same to whatever
       runs next.
    """
    kill_local_simulation_daemon()
    isolated_mlflow_uri = f"sqlite:///{tmp_path / 'smoke_test_mlruns.db'}"
    proc = subprocess.Popen(
        [
            "uv", "run", "flwr", "run", ".",
            "--run-config",
            f'num-server-rounds=1 mlflow-tracking-uri="{isolated_mlflow_uri}"',
            "--federation-config", "num-supernodes=3",
            "--stream",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        try:
            stdout = proc.communicate(timeout=180)[0]
        except subprocess.TimeoutExpired:
            stdout = kill_process_tree(proc)
            pytest.fail(f"flwr run did not complete within 180s (process tree killed):\n{stdout}")

        assert proc.returncode == 0, f"flwr run failed:\n{stdout}"
        assert "pooled_test_auroc" in stdout
        assert "Final global classifier saved" in stdout
    finally:
        kill_local_simulation_daemon()
