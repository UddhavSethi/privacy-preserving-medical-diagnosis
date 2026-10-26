"""Carry the RDP accountant's history across federated rounds when the client process does not.

Found 2026-10 (docs/audit_2026-10.md F11): the SecAgg+ client app builds a NEW client object for
every message, and a deployment-mode hospital may start a fresh ClientApp process per message, so a
PrivacyEngine cached only in memory is silently re-created every round. The training noise is
unaffected (sigma is calibrated once for the TOTAL number of steps over all rounds), but the
*reported* epsilon is then the single-round value (e.g. 2.12 in every round of a 20-round run whose
cumulative epsilon is 4) — a privacy figure that is wrong in the optimistic direction.

The accountant's whole state is its `history`, a list of (noise_multiplier, sample_rate, steps)
tuples, so persisting it is enough. Flower keeps `Context.state` for a node across rounds of a run;
these helpers serialise the history into it (and are pure functions, tested without Flower).
"""
from __future__ import annotations

import json


def history_to_json(privacy_engine) -> str:
    return json.dumps([[float(s), float(q), int(n)] for s, q, n in privacy_engine.accountant.history])


def apply_history_json(privacy_engine, payload: str) -> None:
    """Replace a fresh engine's (empty) history with the one saved after the previous round."""
    privacy_engine.accountant.history = [(float(s), float(q), int(n)) for s, q, n in json.loads(payload)]
