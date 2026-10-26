import json

import pytest
from types import SimpleNamespace

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict

from src.federated.comm_accounting import byte_counting_mod, message_payload_bytes, summarize_comm_log


def _msg(content: RecordDict, group_id="3"):
    msg = Message(content=content, dst_node_id=1, message_type="train", group_id=group_id)
    return msg


def test_payload_bytes_sums_records_and_is_larger_for_larger_arrays():
    import torch
    small = _msg(RecordDict({"arrays": ArrayRecord({"w": torch.zeros(10)})}))
    big = _msg(RecordDict({"arrays": ArrayRecord({"w": torch.zeros(10_000)}), "cfg": ConfigRecord({"a": 1})}))
    assert 0 < message_payload_bytes(small) < message_payload_bytes(big)
    assert message_payload_bytes(big) >= 40_000  # 10k float32


def test_mod_passes_reply_through_and_logs_only_when_enabled(tmp_path, monkeypatch):
    import torch
    msg = _msg(RecordDict({"cfg": ConfigRecord({"x": 1})}))
    reply = Message(content=RecordDict({"arrays": ArrayRecord({"w": torch.zeros(100)})}), reply_to=msg)
    ctx = Context(run_id=1, node_id=1, node_config={"partition-id": 2}, state=RecordDict(), run_config={})
    log = tmp_path / "comm.jsonl"

    monkeypatch.delenv("PNEUMO_COMM_LOG", raising=False)
    assert byte_counting_mod(msg, ctx, lambda m, c: reply) is reply
    assert not log.exists()

    monkeypatch.setenv("PNEUMO_COMM_LOG", str(log))
    assert byte_counting_mod(msg, ctx, lambda m, c: reply) is reply
    row = json.loads(log.read_text().splitlines()[0])
    assert row["node"] == 2 and row["group_id"] == "3" and row["bytes_in"] > 0 and row["bytes_out"] >= 400
    summary = summarize_comm_log(str(log))
    assert summary[f"run{ctx.run_id}_node2_round3"]["messages"] == 1


def test_empty_list_config_values_do_not_crash_the_counter_and_logging_never_raises(tmp_path, monkeypatch):
    """Regression: Flower 1.35's ConfigRecord.count_bytes raises IndexError on empty-list values, which
    SecAgg+ stage messages contain; the first version of this mod crashed live clients."""
    cfg = ConfigRecord({"stage": "share_keys", "empty": [], "blob": b"x" * 10, "ids": [1, 2, 3]})
    with pytest.raises(IndexError):
        cfg.count_bytes()  # documents the upstream defect this guards against
    msg = _msg(RecordDict({"cfg": cfg}))
    assert message_payload_bytes(msg) >= 10 + 3 * 8

    monkeypatch.setenv("PNEUMO_COMM_LOG", str(tmp_path / "no_such_dir" / "x.jsonl"))  # unwritable path
    ctx = Context(run_id=1, node_id=1, node_config={"partition-id": 0}, state=RecordDict(), run_config={})
    reply = Message(content=RecordDict({"cfg": cfg}), reply_to=msg)
    assert byte_counting_mod(msg, ctx, lambda m, c: reply) is reply  # must not raise
