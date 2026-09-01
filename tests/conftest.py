"""Shared test helpers.

`kill_process_tree` -- factored out of `test_tls_auth.py` (Stage 16), where it
genuinely works: that test starts `flower-superlink` itself as a direct Popen
child, so killing its process group reaches it. Reused by
`test_integration_smoke.py` for the same "kill whatever this Popen started"
job, but **verified, not assumed, to have a real gap there**: `flwr run .`
(implicit local-simulation federation) spawns its own SuperLink daemon and Ray
cluster in a *separate* session it deliberately detaches into (Flower designs
this daemon to survive the CLI invocation that launched it, for local-dev
reuse across runs) -- `os.killpg` on the invoking process's own group does not
reach it. Confirmed directly: after a real timeout, `kill_process_tree` ran,
`communicate()` returned, and the SuperLink/Ray processes were still alive and
still consuming CPU seconds later.

`kill_local_simulation_daemon` is the fix for that gap -- kills by matching
the actual process signature (`flower-superlink --simulation`, `flwr-
simulation`, and their Ray children) rather than by process-group membership.
Session note for anyone debugging a slow/hung run of this test in the future,
since it's easy to misdiagnose: the actual cause of every real flakiness
incident traced this session was NOT this daemon-lifecycle gap (that's a real
gap, but nothing in this session's evidence shows it caused a timeout on its
own) -- it was `pyproject.toml`'s `[tool.flwr.app.components]` being left
pointed at the expensive raw-image fine-tuning app by an unrelated crashed
script (`scripts/run_federated_finetune_pilot.py`, killed before its own
revert-on-exit `finally` block could run), which the canonical smoke test then
silently ran instead of the cheap cached-feature app it expects. Once that
config was reverted, a clean run took 36s, well inside the timeout. Both
issues are real and both are fixed here; don't re-attribute one to the other.
"""
from __future__ import annotations

import os
import signal
import subprocess


def kill_process_tree(proc: subprocess.Popen, timeout: float = 5.0) -> str:
    """Kills a process AND its descendants that share its process group
    (requires the process to have been started with `start_new_session=True`).
    Real limitation, not a bug in this function: a child that itself starts a
    new session (as Flower's local-simulation SuperLink does) is not reached
    by this -- see `kill_local_simulation_daemon` for that case specifically.
    Returns whatever combined stdout was captured."""
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        return proc.communicate(timeout=timeout)[0] or ""
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        return proc.communicate(timeout=timeout)[0] or ""


def kill_local_simulation_daemon() -> None:
    """Kills any `flower-superlink --simulation` / `flwr-simulation` process
    (and, by extension, the Ray cluster they own) by matching the process
    signature directly -- not by process-group membership, which doesn't
    reach a daemon that has deliberately detached into its own session (see
    module docstring). Safe to call unconditionally: a no-op if nothing
    matches. Call this before a test that needs a genuinely fresh local
    SuperLink, and after, in case this run's own daemon needs clearing too."""
    for pattern in ("flower-superlink.*--simulation", "flwr-simulation"):
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
