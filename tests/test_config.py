from pathlib import Path

from hydra import compose, initialize_config_dir

CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")


def test_top_level_config_composes():
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name="config")
    assert cfg.seed == 42
    assert cfg.data_partition_seed != cfg.client_sampling_seed
    assert cfg.mlflow.tracking_uri
