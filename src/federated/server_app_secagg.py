"""Stage 15 — Flower ServerApp for the SecAgg+ ablation row (ablation row 4:
FedAvg + Secure Aggregation, no DP). See `client_app_secagg.py`'s docstring for
why this is a separate app pair from Stage 13/14's `server_app.py`/`client_app.py`.

SecAgg+ at flwr==1.35.0 is only wired through Flower's legacy
`Strategy`/`ClientManager`/`workflow` API (`flwr.server.workflow.SecAggPlusWorkflow`
+ `DefaultWorkflow`), not the new Message-API `strategy.start(grid=...)` loop
Stages 13/14 use — no ported Message-API equivalent exists at this pinned
version (verified against the installed source; confirmed against Flower's own
`examples/flower-secure-aggregation`, which uses this exact
`LegacyContext`/`DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(...))` pattern
from inside an `@app.main()` function — so the decorator-based `ServerApp`
entry point is preserved, only the strategy/workflow construction differs).

`pyproject.toml`'s `[tool.flwr.app.components]` points at Stage 13/14's
`server_app.py`/`client_app.py` by default (rows 1/2/3/5/6 of the ablation
ladder). To run *this* row (4), `[tool.flwr.app.components]` must be
temporarily pointed at this module and `client_app_secagg.py` instead — see
`docs/SESSION_STATE.md` for the exact swap and how to revert it.

`num_shares=3` / `reconstruction_threshold=2` (all 3 hospitals share pieces of
each other's key, any 2 of 3 can reconstruct — tolerates 1 dropout): an
engineering default for this project's fixed 3-client setup, not a privacy
epsilon-like decision requiring its own gate — CLAUDE.md's threat model already
states collusion up to the SecAgg+ threshold is in scope, and 2-of-3 is the
protocol's natural minimum above trivial (`num_shares` must be an int > 2, or a
majority-style float).

`max_weight=20000` overrides `SecAggPlusWorkflow`'s own default of 1000.0 —
found via this stage's live validation run, not by inspection: the protocol
weights each client's update by `num_examples`, and the default is far below
this project's real per-hospital counts (Hospital B/C's natural shards are
~13,342 train each), which silently triggered the workflow's own "potential
overflow" warning in the weight-quantization math every round.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg as LegacyFedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD

from src.evaluation.metrics import compute_metrics
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import load_pooled_features

HOSPITALS = ["A", "B", "C"]

app = ServerApp()


def _make_legacy_evaluate_fn(partition_path: str, feature_cache_dir: str, reference_keys: list[str]):
    """Old-style `evaluate_fn` (positional ndarrays, not ArrayRecord) — same
    pooled-test-set protocol as Stage 13/14's `server_app.py`, for direct
    comparability of `pooled_test_auroc` across ablation rows."""
    pooled = load_pooled_features(Path(partition_path), HOSPITALS, feature_cache_dir=Path(feature_cache_dir))

    def evaluate_fn(server_round: int, ndarrays: list, config: dict):
        model = DenseNet121Head()
        state = {k: torch.tensor(arr) for k, arr in zip(reference_keys, ndarrays, strict=True)}
        model.classifier.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model.classifier(pooled.test_features), dim=1)[:, 1].numpy()
        m = compute_metrics(pooled.test_labels.numpy(), probs)
        auroc = m.auroc if m.auroc == m.auroc else 0.0
        return 1.0 - auroc, {"pooled_test_auroc": auroc}

    return evaluate_fn


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config["num-server-rounds"])
    num_shares = int(context.run_config["num-shares"])
    reconstruction_threshold = int(context.run_config["reconstruction-threshold"])
    seed = int(context.run_config["seed"])

    torch.manual_seed(seed)
    global_model = DenseNet121Head()
    reference_keys = list(global_model.classifier.state_dict().keys())
    initial_ndarrays = [v.detach().cpu().numpy() for v in global_model.classifier.state_dict().values()]
    parameters = ndarrays_to_parameters(initial_ndarrays)

    strategy = LegacyFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        min_fit_clients=len(HOSPITALS),
        min_evaluate_clients=len(HOSPITALS),
        min_available_clients=len(HOSPITALS),
        initial_parameters=parameters,
        evaluate_fn=_make_legacy_evaluate_fn(
            context.run_config["partition-path"], context.run_config["feature-cache-dir"], reference_keys
        ),
    )

    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    fit_workflow = SecAggPlusWorkflow(
        num_shares=num_shares,
        reconstruction_threshold=reconstruction_threshold,
        max_weight=float(context.run_config["max-weight"]),
    )
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    workflow(grid, legacy_context)

    print("\n=== Run history (SecAgg+, ablation row 4) ===")
    print(legacy_context.history)

    final_ndarrays = legacy_context.state.array_records[MAIN_PARAMS_RECORD].to_numpy_ndarrays()
    final_state = {k: torch.tensor(arr) for k, arr in zip(reference_keys, final_ndarrays, strict=True)}

    out_path = Path(
        context.run_config.get("output-checkpoint", "outputs/checkpoints/federated/secagg_final.pt")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_state, out_path)
    print(f"\nFinal global classifier saved: {out_path}")
