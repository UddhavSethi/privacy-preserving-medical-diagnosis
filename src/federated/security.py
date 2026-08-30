"""Stage 16 — TLS + client authentication (ADR-4) helpers.

Certificate/key generation lives in `scripts/generate_certs.sh` (ADR-4:
"Certificate/key generation must be scripted and committed" — the script,
never the generated `certs/` material itself, which stays gitignored).
`src/federated/client_app.py`/`server_app.py` need no code changes for TLS or
node authentication — both are entirely CLI/deployment-runtime concerns
(`flower-superlink --ssl-*  --enable-supernode-auth`,
`flower-supernode --root-certificates --auth-supernode-private-key`), not
something the app code configures. This module covers the one piece of
ADR-4 that *is* a code-level claim worth testing directly.

gRPC message-length note: ADR-4's own text (and this project's earlier
architecture notes) assume Flower's classic 4MB default message-size
ceiling, requiring explicit override. Verified against the pinned
flwr==1.35.0's actual CLI (`flower-superlink --help`, `flower-supernode
--help`, `flwr run --help`): no flag to override gRPC message size exists at
this version any more — `flwr.common.GRPC_MAX_MESSAGE_LENGTH` is a hardcoded
~2GB constant with no exposed override. The 4MB-default assumption is stale
for this pinned version, not something to silently pretend was "explicitly
configured" — this module instead verifies the real relationship that
matters (our real payload vs. the real limit), per this stage's own testing
criterion ("confirmation that the configured message length exceeds the
actual update size").
"""
from __future__ import annotations

import io

import torch
from flwr.common import GRPC_MAX_MESSAGE_LENGTH

from src.models.densenet_head import DenseNet121Head


def classifier_payload_size_bytes(model: DenseNet121Head | None = None) -> int:
    """Serialized size of the head-only federated payload (ADR-1) — what
    actually crosses the wire once per round, not the full frozen backbone."""
    model = model if model is not None else DenseNet121Head()
    buffer = io.BytesIO()
    torch.save(model.classifier.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def assert_payload_within_message_limit(model: DenseNet121Head | None = None) -> int:
    """Raises if the real classifier-head payload would not fit inside a
    single gRPC message under this pinned Flower version's actual limit.
    Returns the measured payload size in bytes."""
    size = classifier_payload_size_bytes(model)
    if size >= GRPC_MAX_MESSAGE_LENGTH:
        raise AssertionError(
            f"classifier payload ({size} bytes) meets or exceeds "
            f"GRPC_MAX_MESSAGE_LENGTH ({GRPC_MAX_MESSAGE_LENGTH} bytes) — "
            "federated updates would be rejected by gRPC."
        )
    return size
