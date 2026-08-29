import mlflow
from omegaconf import OmegaConf

from src.utils.mlflow_utils import tracked_run


def _client(tracking_uri: str) -> mlflow.tracking.MlflowClient:
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


def test_tracked_run_logs_fully_resolved_config(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
    cfg = OmegaConf.create(
        {
            "mlflow": {"tracking_uri": tracking_uri, "experiment_name": "test"},
            "seed": 42,
            "model": {"lr": 0.01},
        }
    )

    with tracked_run(cfg, experiment_name="test-exp"):
        run_id = mlflow.active_run().info.run_id

    data = _client(tracking_uri).get_run(run_id).data
    assert data.params["seed"] == "42"
    assert data.params["model.lr"] == "0.01"
    assert "git_sha" in data.tags


def test_config_override_changes_logged_params(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
    cfg1 = OmegaConf.create({"mlflow": {"tracking_uri": tracking_uri}, "seed": 1})
    cfg2 = OmegaConf.create({"mlflow": {"tracking_uri": tracking_uri}, "seed": 2})

    with tracked_run(cfg1, experiment_name="test-exp"):
        run_id1 = mlflow.active_run().info.run_id
    with tracked_run(cfg2, experiment_name="test-exp"):
        run_id2 = mlflow.active_run().info.run_id

    client = _client(tracking_uri)
    assert client.get_run(run_id1).data.params["seed"] == "1"
    assert client.get_run(run_id2).data.params["seed"] == "2"
