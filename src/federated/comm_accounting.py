"""Application-layer communication accounting for the SecAgg+ app pair.

`docs/results.md` reports only the model-update payload (`payload_bytes`, ~1.05 MB), which is
what a plain FedAvg client uploads. SecAgg+ adds extra messages per round (key advertisement,
secret-share exchange, masked-vector upload, unmasking shares), and nothing measured them, so
"SecAgg overhead" was reported as "expected to be larger but not instrumented". This mod sits
OUTSIDE `secaggplus_mod` on the client, so it sees every message of every protocol stage, and
sums the serialized size of the records in each (`Record.count_bytes()`), in and out, per round.

What it measures: Flower record payload bytes. It excludes gRPC/HTTP2/TLS framing and
transport compression, so it is a lower bound on wire bytes and is labelled that way.
Logging is opt-in (env var `PNEUMO_COMM_LOG` = path of a JSONL file) so the default code path
is unchanged.
"""
from __future__ import annotations

import json
import os


def _value_bytes(v) -> int:
    """Serialized size of one ConfigRecord value (bool/int/float scalars, str, bytes, or lists of those)."""
    if isinstance(v, (bytes, bytearray)):
        return len(v)
    if isinstance(v, str):
        return len(v.encode("utf-8"))
    if isinstance(v, bool):
        return 1
    if isinstance(v, (int, float)):
        return 8
    if isinstance(v, (list, tuple)):
        return sum(_value_bytes(x) for x in v)  # an empty list is 0 bytes
    return 0


def _record_bytes(rec) -> int:
    """`Record.count_bytes()` where it works; a manual count where Flower's own raises.

    Flower 1.35's `ConfigRecord.count_bytes` indexes `v[0]` for list values and raises IndexError on
    an empty list, which SecAgg+ stage messages contain. Found when this mod, as first written,
    crashed the clients mid-protocol (docs/audit_2026-10.md); instrumentation must never break the run.
    """
    try:
        return int(rec.count_bytes())
    except Exception:
        if hasattr(rec, "items"):
            return int(sum(len(str(k).encode("utf-8")) + _value_bytes(v) for k, v in rec.items()))
        return 0


def message_payload_bytes(msg) -> int:
    """Total serialized size of the records in a Flower message's content (0 for error replies)."""
    if getattr(msg, "has_error", None) and msg.has_error():
        return 0
    return int(sum(_record_bytes(rec) for rec in msg.content.values()))


def byte_counting_mod(msg, ctxt, call_next):
    """Client mod: record bytes received and bytes sent for each message, then pass through."""
    reply = call_next(msg, ctxt)
    path = os.environ.get("PNEUMO_COMM_LOG")
    if path:
        try:
            _log(path, msg, reply, ctxt)
        except Exception:  # accounting must never break a federated round
            pass
    return reply


def _log(path, msg, reply, ctxt) -> None:
    if True:
        record = {
            "run_id": str(ctxt.run_id),  # the simulation's SuperLink process outlives a run, so its environment
            # (and thus PNEUMO_COMM_LOG) is fixed by the FIRST run; the run id is what separates later runs
            "node": ctxt.node_config.get("partition-id"),
            "group_id": msg.metadata.group_id,  # the federated round
            "message_type": msg.metadata.message_type,
            "bytes_in": message_payload_bytes(msg),
            "bytes_out": message_payload_bytes(reply),
        }
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def summarize_comm_log(path: str) -> dict:
    """Per-(run, node, round) totals from a JSONL log written by `byte_counting_mod`. Records written
    before `run_id` existed are grouped under run "?"."""
    rows = [json.loads(line) for line in open(path)]
    by_key: dict = {}
    for r in rows:
        key = (r.get("run_id", "?"), r["node"], r["group_id"])
        agg = by_key.setdefault(key, {"bytes_in": 0, "bytes_out": 0, "messages": 0})
        agg["bytes_in"] += r["bytes_in"]; agg["bytes_out"] += r["bytes_out"]; agg["messages"] += 1
    return {f"run{k[0]}_node{k[1]}_round{k[2]}": v for k, v in sorted(by_key.items(), key=lambda kv: tuple(map(str, kv[0])))}
