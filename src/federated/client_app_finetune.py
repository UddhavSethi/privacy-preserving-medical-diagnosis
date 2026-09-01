"""ADR-1 GroupNorm fallback -- federated fine-tuning ClientApp, added 2026-08-31.

Separate app pair from the canonical `client_app.py`/`server_app.py`, following
this project's own established precedent (Stage 15's `client_app_secagg.py`/
`server_app_secagg.py`, per CLAUDE.md's resolved decision 7): a genuinely different
training mode gets its own app pair rather than a config flag threaded through the
canonical one, since the canonical app's cached-pooled-feature training loop is
architecturally incompatible with a partially-unfrozen backbone (see
`DenseNet121Head`'s `fine_tune_last_block` docstring and
`docs/adr1_groupnorm_fallback.md` for the full rationale).

Same FedAvg, no-DP protocol Stage 13's canonical app runs (`train_local_round`'s
federated-round paradigm: a few local epochs starting from the server-provided
global state, matching this project's own precedent), except:
  - Trains on raw CLAHE-cached images through the partially-unfrozen model
    (`RawImageDataset`), not Stage 9's cached pooled features -- those assume a
    fully frozen backbone.
  - Transmits `DenseNet121Head.trainable_state_dict()` (classifier +
    denseblock4 + norm5, ~9.7MB), not classifier-only (~1MB) -- the federated
    payload grows because there is genuinely more to federate now, but stays
    far under Flower's message-size ceiling (`src/federated/security.py`).
  - Differential learning rates for the backbone tail vs. the head, matching
    the centralized pilot's own choice (`scripts/train_centralized_finetune.py`).

DP-SGD is NOT wired into this app (out of scope for this pilot -- Opacus's
per-sample gradients over a partially-unfrozen backbone on raw images is a
separate, larger integration than this pilot's question, which is "does
fine-tuning help accuracy/Grad-CAM at all").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch.utils.data import DataLoader

from src.data.raw_image_dataset import RawImageDataset, records_for
from src.data.transforms import build_eval_transform, build_train_transform
from src.evaluation.metrics import compute_metrics
from src.evaluation.overhead import classifier_payload_size_bytes, measure_wall_clock
from src.federated.serialization import array_record_to_classifier_state, classifier_state_to_array_record
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import compute_class_weights

PARTITION_TO_HOSPITAL = {0: "A", 1: "B", 2: "C"}
IMAGE_SIZE = 224
BACKBONE_LR_FRACTION = 0.1  # backbone tail LR = head LR * this fraction (matches the centralized pilot's 1e-4 vs 1e-3)

app = ClientApp()

_dataset_cache: dict[tuple[str, str, str], tuple[RawImageDataset, RawImageDataset]] = {}


def _get_datasets(hospital: str, partition_path: str, clahe_cache_dir: str) -> tuple[RawImageDataset, RawImageDataset]:
    key = (hospital, partition_path, clahe_cache_dir)
    if key not in _dataset_cache:
        partition = json.loads(Path(partition_path).read_text())
        train_records = records_for(partition, [hospital], "train")
        val_records = records_for(partition, [hospital], "val")
        train_ds = RawImageDataset(train_records, build_train_transform(image_size=IMAGE_SIZE), Path(clahe_cache_dir))
        val_ds = RawImageDataset(val_records, build_eval_transform(image_size=IMAGE_SIZE), Path(clahe_cache_dir))
        _dataset_cache[key] = (train_ds, val_ds)
    return _dataset_cache[key]


def _resolve_config(context: Context, key: str) -> str:
    if key in context.node_config:
        return str(context.node_config[key])
    return str(context.run_config[key])


@app.train()
def train(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    hospital = PARTITION_TO_HOSPITAL[partition_id]
    train_ds, _ = _get_datasets(
        hospital, _resolve_config(context, "partition-path"), _resolve_config(context, "clahe-cache-dir")
    )
    seed = int(context.run_config["seed"]) + partition_id
    local_epochs = int(context.run_config["local-epochs"])
    lr = float(msg.content["config"]["lr"])
    batch_size = int(context.run_config["batch-size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121Head(fine_tune_last_block=True).to(device)
    model.load_trainable_state_dict({k: v.to(device) for k, v in array_record_to_classifier_state(msg.content["arrays"]).items()})

    backbone_params = list(model.features.denseblock4.parameters()) + list(model.features.norm5.parameters())
    opt = torch.optim.Adam(
        [
            {"params": backbone_params, "lr": lr * BACKBONE_LR_FRACTION},
            {"params": model.classifier.parameters(), "lr": lr},
        ]
    )

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)

    train_labels_t = torch.tensor([train_ds.records[i]["label"] == "Pneumonia" for i in range(len(train_ds))], dtype=torch.long)
    class_weights = compute_class_weights(train_labels_t).to(device)

    model.train()
    total_loss, n_examples = 0.0, 0
    with measure_wall_clock() as timing:
        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = F.cross_entropy(out, y, weight=class_weights)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(y)
                n_examples += len(y)

    trainable_state_cpu = {k: v.cpu() for k, v in model.trainable_state_dict().items()}
    metrics = {
        "train_loss": total_loss / max(n_examples, 1),
        "num-examples": n_examples,
        "wall_clock_seconds": timing.wall_clock_seconds,
        "payload_bytes": classifier_payload_size_bytes(trainable_state_cpu),
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
    _, val_ds = _get_datasets(
        hospital, _resolve_config(context, "partition-path"), _resolve_config(context, "clahe-cache-dir")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121Head(fine_tune_last_block=True).to(device)
    model.load_trainable_state_dict({k: v.to(device) for k, v in array_record_to_classifier_state(msg.content["arrays"]).items()})
    model.eval()

    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())

    m = compute_metrics(np.array(all_labels), np.array(all_probs))
    auroc = m.auroc if m.auroc == m.auroc else 0.0

    content = RecordDict({"metrics": MetricRecord({"val_auroc": auroc, "num-examples": len(all_labels)})})
    return Message(content=content, reply_to=msg)
