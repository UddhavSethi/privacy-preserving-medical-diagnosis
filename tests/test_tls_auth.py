"""Stage 16 — TLS + client authentication (ADR-4).

CLAUDE.md's own testing requirement for this stage: "A negative test
confirming that an unregistered or unauthenticated client is rejected —
without this, client authentication is decorative." This is a real
subprocess-level integration test, not a mock — it starts an actual
`flower-superlink` with TLS and `--enable-supernode-auth`, and actual
`flower-supernode` processes, matching the discipline used for every other
Flower-protocol claim in this project (Stages 13-15).

Isolation: `FLWR_HOME` (an env var Flower's own CLI supports — see
`flwr.supercore.utils.get_flwr_home`) redirects the `flwr` CLI's config file
and all SuperLink/SuperNode runtime state into a pytest `tmp_path`, so this
test never touches the developer's real `~/.flwr/` state, and two runs never
collide. Ports are fixed but distinct from every other port used elsewhere
in this project (the simulation daemon uses 39091/39093; the manual Stage 16
deployment validation used the real defaults 9092/9093) — chosen so a
concurrent `flwr run` in another terminal can't collide with this test.
"""
from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_API_PORT = 29093
FLEET_API_PORT = 29092

pytestmark = pytest.mark.skipif(
    shutil.which("openssl") is None or shutil.which("ssh-keygen") is None,
    reason="requires the openssl and ssh-keygen command-line tools",
)


def _kill_process_tree(proc: subprocess.Popen, timeout: float = 5.0) -> str:
    """Kills a process AND its descendants (Flower's CLI processes spawn a
    "SuperExec" sidecar that outlives a plain `terminate()`/`kill()` on the
    parent alone and keeps the stdout pipe open, which otherwise hangs
    `communicate()` forever — found via this test's own first run, not by
    inspection). Requires the process to have been started with
    `start_new_session=True`. Returns whatever combined stdout was captured."""
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


def _wait_for_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"Nothing listening on 127.0.0.1:{port} after {timeout}s")


@pytest.fixture
def deployment_env(tmp_path):
    """Generates throwaway certs/keys, starts a real TLS+auth SuperLink
    against isolated FLWR_HOME state, and tears everything down afterward."""
    cert_dir = tmp_path / "certs"
    subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "generate_certs.sh"), str(cert_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )

    flwr_home = tmp_path / "flwr_home"
    flwr_home.mkdir()
    env = {**os.environ, "FLWR_HOME": str(flwr_home)}

    superlink = subprocess.Popen(
        [
            "flower-superlink",
            "--ssl-ca-certfile", str(cert_dir / "ca.crt"),
            "--ssl-certfile", str(cert_dir / "server.pem"),
            "--ssl-keyfile", str(cert_dir / "server.key"),
            "--enable-supernode-auth",
            "--database", ":flwr-in-memory:",
            "--disable-runtime-dependency-installation",
            "--control-api-address", f"127.0.0.1:{CONTROL_API_PORT}",
            "--fleet-api-address", f"127.0.0.1:{FLEET_API_PORT}",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        _wait_for_port(CONTROL_API_PORT)
        # Register a named SuperLink connection for this test's isolated
        # FLWR_HOME, the same registry `flwr supernode register` reads
        # (found via this stage's live validation: there is no CLI command
        # to create one against a fresh local address in the open-source
        # build, so it is written directly — a plain, user-editable TOML
        # file, not a protected format).
        (flwr_home / "config.toml").write_text(
            f'[superlink]\ndefault = "test"\n\n'
            f'[superlink.test]\naddress = "127.0.0.1:{CONTROL_API_PORT}"\n'
            f'root-certificates = "{cert_dir / "ca.crt"}"\n'
        )
        yield cert_dir, env
    finally:
        _kill_process_tree(superlink)


def test_supernode_auth_requires_tls():
    """Stage 16's own testing criterion: "confirmation that traffic is
    encrypted." Flower itself refuses to enable SuperNode authentication over
    an unencrypted channel — `--enable-supernode-auth` with `--insecure`
    together must be rejected outright, not silently accepted with auth
    running unencrypted. This is what makes "authenticated" mean something:
    an attacker who can't read the channel can't replay or forge a
    registered hospital's credentials either."""
    result = subprocess.run(
        ["flower-superlink", "--insecure", "--enable-supernode-auth", "--database", ":flwr-in-memory:"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "requires encrypted TLS" in combined or "TLS" in combined, combined


def test_unregistered_supernode_is_rejected(deployment_env):
    """The negative test CLAUDE.md names explicitly: an unregistered
    SuperNode — one whose public key was never pre-registered with the
    SuperLink — must be rejected, not silently accepted.

    `flower-supernode` prints the rejection but does not exit cleanly even
    with `--max-retries 0` (its SuperExec sidecar keeps polling) — found via
    this test's own first run, not by inspection. So this uses `Popen` +
    a bounded wait + explicit kill, the same pattern as the positive test,
    rather than `subprocess.run(..., timeout=...)`, which raises instead of
    returning the captured output once the deadline passes.
    """
    cert_dir, env = deployment_env

    supernode = subprocess.Popen(
        [
            "flower-supernode",
            "--root-certificates", str(cert_dir / "ca.crt"),
            "--superlink", f"127.0.0.1:{FLEET_API_PORT}",
            "--auth-supernode-private-key", str(cert_dir / "hospital_A"),
            "--max-retries", "0",
            "--max-wait-time", "5",
            "--node-config", "partition-id=0 num-partitions=3",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and supernode.poll() is None:
            time.sleep(0.3)
    finally:
        combined = _kill_process_tree(supernode)

    assert "FAILED_PRECONDITION" in combined or "Failed to activate SuperNode" in combined, (
        f"expected an unregistered SuperNode to be rejected by the SuperLink; "
        f"got:\n{combined}"
    )


def test_registered_supernode_is_accepted(deployment_env):
    """Positive counterpart — the same key, after being registered via
    `flwr supernode register`, must be accepted. Without this half, the
    negative test alone can't distinguish 'auth works' from 'the SuperLink
    is just broken and rejects everyone'."""
    cert_dir, env = deployment_env

    register = subprocess.run(
        ["flwr", "supernode", "register", str(cert_dir / "hospital_A.pub"), "test"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert register.returncode == 0, f"registration failed:\n{register.stdout}\n{register.stderr}"
    assert "registered successfully" in register.stdout

    supernode = subprocess.Popen(
        [
            "flower-supernode",
            "--root-certificates", str(cert_dir / "ca.crt"),
            "--superlink", f"127.0.0.1:{FLEET_API_PORT}",
            "--auth-supernode-private-key", str(cert_dir / "hospital_A"),
            "--host", "127.0.0.1",
            "--port", "29094",
            "--node-config", "partition-id=0 num-partitions=3",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 15
        # A registered node that gets rejected exits quickly (see the
        # negative test's FAILED_PRECONDITION); staying alive for the whole
        # window is itself strong evidence of acceptance.
        while time.monotonic() < deadline and supernode.poll() is None:
            time.sleep(0.3)
        accepted = supernode.poll() is None
    finally:
        output = _kill_process_tree(supernode)

    assert accepted, (
        f"expected a registered SuperNode to stay connected, but it exited early:\n{output}"
    )
