"""MLflow tracking helpers.

Every reported result must trace back to a resolved config, a git SHA and a seed set
(CLAUDE.md section 12) — this module makes that automatic rather than optional, so a
result that isn't in MLflow with its full config genuinely doesn't exist.
"""
from __future__ import annotations

import subprocess
from contextlib import contextmanager
from typing import Any, Iterator

import mlflow
from omegaconf import DictConfig, OmegaConf


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def flatten_config(cfg: DictConfig) -> dict[str, Any]:
    """Flatten a resolved OmegaConf config into dotted-key MLflow params."""
    container = OmegaConf.to_container(cfg, resolve=True)
    flat: dict[str, Any] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, sub_value in value.items():
                _walk(f"{prefix}.{key}" if prefix else str(key), sub_value)
        else:
            flat[prefix] = value

    _walk("", container)
    return flat


@contextmanager
def tracked_run(
    cfg: DictConfig, experiment_name: str, run_name: str | None = None
) -> Iterator[None]:
    """Open an MLflow run that logs the fully resolved config and git SHA up front."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten_config(cfg))
        mlflow.set_tag("git_sha", git_sha())
        yield
