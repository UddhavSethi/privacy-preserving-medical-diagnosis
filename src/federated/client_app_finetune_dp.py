"""ADR-1 fine-tuning x ADR-2 Differential Privacy -- federated ClientApp.

Built for CLAUDE.md's pending decision 4 "clean comparison" follow-up: the existing
DP epsilon sweep (`client_app.py`) is head-only + federated; the existing DP +
fine-tuning result (`scripts/train_centralized_finetune_dp.py`) is fine-tuned +
centralized. Neither isolates the architecture effect while holding topology fixed.
This app holds topology fixed at FEDERATED (this project's actual thesis, not the
centralized ceiling) so it can be compared directly against `client_app.py`'s
existing federated DP sweep at the same epsilon.

Pairs with the EXISTING `server_app_finetune.py`, unmodified -- FedAvg aggregation
over `trainable_state_dict()` arrays is agnostic to how each client computed its
gradients, so only a new ClientApp is needed. Combines two things that already
exist separately:
  - `client_app_finetune.py`'s raw-image, partially-unfrozen-backbone training loop
    (denseblock4 + norm5 + classifier, differential learning rates).
  - `scripts/train_centralized_finetune_dp.py`'s Opacus wrapping approach: the
    frozen prefix runs once outside Opacus on the deterministic eval-style view (no
    augmentation -- Opacus's Poisson-sampling DataLoader doesn't compose with a
    fresh augmented view per epoch, same reason `src/privacy/dp.py` gives for the
    head-only path), and only the FineTuneTail (denseblock4+norm5+classifier) is
    wrapped by `PrivacyEngine.make_private()`.

Frozen-prefix features are computed ONCE per hospital and cached across rounds
(module-level, same simulated-node-process-reuse assumption `client_app.py`'s own
caches rely on) -- the frozen prefix never updates, so recomputing it every round
would be pure waste. The `PrivacyEngine` instance is ALSO cached per hospital across
rounds (mirroring `client_app.py`'s `_privacy_engine_cache`), required for the
accountant's spent-epsilon to accumulate correctly across all `num-server-rounds`
rather than resetting each round.

After training, the updated tail's parameters are read back off `head` itself
(`head.trainable_state_dict()`), not off the Opacus-wrapped tail's own state dict --
`FineTuneTail`'s submodules are the SAME objects as `head.features.denseblock4`/
`norm5`/`classifier` (shared by reference, not copied), and `PrivacyEngine.
make_private()` adds hooks without cloning parameters, so `dp_opt.step()` updates
them in place. This sidesteps any key-prefix mismatch between `FineTuneTail`'s own
state dict (`denseblock4.*`, `norm5.*`, `classifier.*`) and
`trainable_state_dict()`'s naming (`features.denseblock4.*`, `features.norm5.*`,
`classifier.*`), which the server/aggregation side requires.

Scope: single-seed diagnostic (`scripts/run_federated_finetune_dp_pilot.py`), not a
3-seed campaign, and does NOT touch pending decision 3 (whether to scale
fine-tuning into the paper's headline ablation results or the app's default
checkpoint) -- this is a bounded probe answering "does fine-tuning help under DP,
holding federated topology fixed," nothing more.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from src.data.raw_image_dataset import RawImageDataset, records_for
from src.data.transforms import build_eval_transform
from src.evaluation.metrics import compute_metrics
from src.evaluation.overhead import classifier_payload_size_bytes, measure_wall_clock
from src.federated.serialization import array_record_to_classifier_state, classifier_state_to_array_record
from src.models.densenet_head import _FROZEN_PREFIX_NAMES, DenseNet121Head
from src.privacy.accounting import compute_noise_multiplier, compute_total_steps
from src.privacy.dp import make_privacy_engine
from src.training.trainer import compute_class_weights

PARTITION_TO_HOSPITAL = {0: "A", 1: "B", 2: "C"}
IMAGE_SIZE = 224
BACKBONE_LR_FRACTION = 0.1  # matches client_app_finetune.py / the centralized DP script

app = ClientApp()

# Module-level caches: a simulated node's process is reused across rounds within one
# run (same assumption client_app.py's own caches rely on).
_feature_cache: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
_privacy_engine_cache: dict[str, tuple[PrivacyEngine, float]] = {}


class FineTuneTail(nn.Module):
    """denseblock4 + norm5 + pool + classifier -- mirrors
    `scripts/train_centralized_finetune_dp.py`'s class of the same name: the exact
    subset of `DenseNet121Head.forward()` that runs on the frozen prefix's output,
    and the only part of the model Opacus ever sees."""

    def __init__(self, head: DenseNet121Head) -> None:
        super().__init__()
        self.denseblock4 = head.features.denseblock4
        self.norm5 = head.features.norm5
        self.pool = head.pool
        self.classifier = head.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.denseblock4(x)
        x = self.norm5(x)
        x = torch.relu(x)
        x = self.pool(x)
        return self.classifier(x)


def _resolve_config(context: Context, key: str) -> str:
    if key in context.node_config:
        return str(context.node_config[key])
    return str(context.run_config[key])


@torch.no_grad()
def _frozen_prefix_features(head: DenseNet121Head, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    head.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = x
        for name in _FROZEN_PREFIX_NAMES:
            out = getattr(head.features, name)(out)
        feats.append(out.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def _get_hospital_features(
    hospital: str, partition_path: str, clahe_cache_dir: str, split: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (hospital, partition_path, split)
    if key not in _feature_cache:
        partition = json.loads(Path(partition_path).read_text())
        records = records_for(partition, [hospital], split)
        eval_transform = build_eval_transform(image_size=IMAGE_SIZE)
        ds = RawImageDataset(records, eval_transform, Path(clahe_cache_dir))
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        head = DenseNet121Head(fine_tune_last_block=True).to(device)
        _feature_cache[key] = _frozen_prefix_features(head, loader, device)
    return _feature_cache[key]


def _get_privacy_engine(hospital: str, context: Context, dataset_size: int) -> tuple[PrivacyEngine, float]:
    if hospital not in _privacy_engine_cache:
        total_steps = compute_total_steps(
            dataset_size=dataset_size,
            batch_size=int(context.run_config["batch-size"]),
            local_epochs=int(context.run_config["local-epochs"]),
            num_rounds=int(context.run_config["num-server-rounds"]),
        )
        sample_rate = int(context.run_config["batch-size"]) / dataset_size
        noise_multiplier = compute_noise_multiplier(
            target_epsilon=float(context.run_config["target-epsilon"]),
            target_delta=float(context.run_config["target-delta"]),
            sample_rate=sample_rate,
            total_steps=total_steps,
        )
        _privacy_engine_cache[hospital] = (make_privacy_engine(), noise_multiplier)
    return _privacy_engine_cache[hospital]


@app.train()
def train(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    hospital = PARTITION_TO_HOSPITAL[partition_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_features, train_labels = _get_hospital_features(
        hospital, _resolve_config(context, "partition-path"), _resolve_config(context, "clahe-cache-dir"), "train", device
    )

    seed = int(context.run_config["seed"]) + partition_id
    local_epochs = int(context.run_config["local-epochs"])
    lr = float(msg.content["config"]["lr"])
    batch_size = int(context.run_config["batch-size"])
    target_delta = float(context.run_config["target-delta"])
    max_grad_norm = float(context.run_config["max-grad-norm"])

    head = DenseNet121Head(fine_tune_last_block=True).to(device)
    head.load_trainable_state_dict(
        {k: v.to(device) for k, v in array_record_to_classifier_state(msg.content["arrays"]).items()}
    )
    tail = FineTuneTail(head)

    class_weights = compute_class_weights(train_labels).to(device)
    opt = torch.optim.Adam(
        [
            {"params": list(tail.denseblock4.parameters()) + list(tail.norm5.parameters()), "lr": lr * BACKBONE_LR_FRACTION},
            {"params": tail.classifier.parameters(), "lr": lr},
        ]
    )

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, generator=generator)

    privacy_engine, noise_multiplier = _get_privacy_engine(hospital, context, dataset_size=len(train_labels))

    dp_tail, dp_opt, dp_loader = privacy_engine.make_private(
        module=tail, optimizer=opt, data_loader=loader,
        noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
    )

    dp_tail.train()
    total_loss, n_examples = 0.0, 0
    with measure_wall_clock() as timing:
        for _ in range(local_epochs):
            for x, y in dp_loader:
                x, y = x.to(device), y.to(device)
                out = dp_tail(x)
                loss = F.cross_entropy(out, y, weight=class_weights)
                dp_opt.zero_grad()
                loss.backward()
                dp_opt.step()
                total_loss += loss.item() * len(y)
                n_examples += len(y)

    try:
        epsilon_spent = privacy_engine.get_epsilon(delta=target_delta)
    except OverflowError:
        epsilon_spent = float("inf")

    # head's own submodules were updated in place (see module docstring) -- read
    # the correctly-prefixed dict straight off head, not off dp_tail.
    trainable_state_cpu = {k: v.cpu() for k, v in head.trainable_state_dict().items()}

    metrics = {
        "train_loss": total_loss / max(n_examples, 1),
        "num-examples": n_examples,
        "wall_clock_seconds": timing.wall_clock_seconds,
        "payload_bytes": classifier_payload_size_bytes(trainable_state_cpu),
        "epsilon_spent": epsilon_spent,
        "noise_multiplier": noise_multiplier,
    }

    content = RecordDict(
        {
            "arrays": classifier_state_to_array_record(trainable_state_cpu),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    hospital = PARTITION_TO_HOSPITAL[partition_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_features, val_labels = _get_hospital_features(
        hospital, _resolve_config(context, "partition-path"), _resolve_config(context, "clahe-cache-dir"), "val", device
    )

    head = DenseNet121Head(fine_tune_last_block=True).to(device)
    head.load_trainable_state_dict(
        {k: v.to(device) for k, v in array_record_to_classifier_state(msg.content["arrays"]).items()}
    )
    tail = FineTuneTail(head)
    tail.eval()

    with torch.no_grad():
        probs = F.softmax(tail(val_features.to(device)), dim=1)[:, 1].cpu().numpy()
    m = compute_metrics(val_labels.numpy(), probs)
    auroc = m.auroc if m.auroc == m.auroc else 0.0

    content = RecordDict({"metrics": MetricRecord({"val_auroc": auroc, "num-examples": len(val_labels)})})
    return Message(content=content, reply_to=msg)
