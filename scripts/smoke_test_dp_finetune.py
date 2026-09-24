"""ADR-1 x ADR-2 VRAM smoke test: does DP-SGD fit in 4GB VRAM when combined with
the fine-tuned architecture (denseblock4 + norm5 + classifier trainable)?

CLAUDE.md's pending decision 4 (raised 2026-09-22): the 3-seed fine-tuning campaign
and the DP epsilon sweep have never been run together. Fine-tuning raises the
trainable parameter count from ~263K (head-only, `src/privacy/dp.py`'s already-
validated production path) to ~2.4M -- a ~9x increase on exactly the axis ADR-1's
own DP-utility-collapse rationale warns about, and Opacus's per-sample gradients
carry a memory cost beyond a normal backward pass that head-only DP has never come
close to stressing on this machine's 4GB laptop GPU. This script is the scoped
first step named in that pending decision: confirm whether the fine-tuned tail fits
at the project's standard fine-tuning batch size (16, `train_centralized_finetune.py`),
and if not, find a physical batch size (via Opacus's `BatchMemoryManager`) that does.

**This is a smoke test, not an experiment.** It does not produce a reportable
epsilon/accuracy result and touches no checkpoint used elsewhere. If it passes,
the next step (not this script) is a real single-seed, epsilon=4, centralized DP
fine-tuning run.

Why the whole model can't just be handed to Opacus directly: `denseblock1-3` (the
frozen prefix) still contain real BatchNorm -- Opacus's ModuleValidator rejects
that outright regardless of `requires_grad` (this is ADR-1's original, still-true
DP/BatchNorm incompatibility). `src/privacy/dp.py` avoids this by only ever
wrapping `model.classifier`, feeding it precomputed frozen-backbone features.
This script extends exactly that pattern: the frozen prefix
(`DenseNet121Head._FROZEN_PREFIX_NAMES`) runs once, outside Opacus, under
`torch.no_grad()` on the deterministic eval-style view (matching `src/privacy/
dp.py`'s own stated reason -- Opacus's Poisson-sampling DataLoader doesn't compose
with a different random augmented view per epoch). Only `denseblock4 + norm5 +
classifier` (already GroupNorm/Linear -- DP-safe per ADR-1's resolved decision 11)
is wrapped by `PrivacyEngine.make_private()`, taking the frozen prefix's output as
its input tensor.

Usage: uv run python scripts/smoke_test_dp_finetune.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset

from src.data.transforms import build_eval_transform
from src.models.densenet_head import _FROZEN_PREFIX_NAMES, DenseNet121Head
from src.models.freezing import count_trainable_parameters
from src.privacy.accounting import compute_noise_multiplier, compute_total_steps
from src.utils.seeding import set_global_seed

# Sibling-module import, not `scripts.train_centralized_finetune` -- `scripts/` is
# not an installed package, only `src.*` imports resolve from an arbitrary cwd.
# Running this file directly puts its own directory on sys.path[0] (same note as
# `scripts/run_federated_finetune_pilot.py`).
from train_centralized_finetune import PARTITION_PATH, RawImageDataset, _records_for

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "outputs" / "results" / "dp_finetune_vram_smoke_test.json"

SEED = 42
N_SMOKE_EXAMPLES = 256  # small on purpose -- this only needs to exercise memory, not train a real model
BATCH_SIZE = 16  # matches train_centralized_finetune.py's own protocol
NUM_SMOKE_STEPS = 5
TARGET_EPSILON = 4.0  # project default, DG-7
TARGET_DELTA = 1e-5  # project default, DG-7
MAX_GRAD_NORM = 1.0
IMAGE_SIZE = 224

# Escalation ladder if the full logical batch doesn't fit unassisted.
BMM_PHYSICAL_BATCH_SIZES = [8, 4, 2, 1]


class FineTuneTail(nn.Module):
    """denseblock4 + norm5 + pool + classifier, sharing parameters with `head` --
    the exact subset of `DenseNet121Head.forward()` that runs on the frozen prefix's
    *output* rather than raw pixels. This is the only part of the fine-tuned model
    Opacus is ever shown."""

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


@torch.no_grad()
def _frozen_prefix_features(head: DenseNet121Head, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Everything up to (not including) denseblock4 -- deterministic given fixed
    pixels, computed once outside Opacus. Uses the eval-style (non-augmented) view,
    matching `src/privacy/dp.py`'s own documented reason for doing so."""
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


def _run_steps(loader, opt, model, device, num_steps: int) -> int:
    """Runs up to `num_steps` real DP-SGD steps (re-iterating the loader if it runs
    out first). Returns how many steps actually completed."""
    model.train()
    steps_done = 0
    for _ in range(50):  # generous upper bound on re-iterations
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps_done += 1
            if steps_done >= num_steps:
                return steps_done
    return steps_done


def _attempt(label: str, run_fn) -> dict:
    """Runs `run_fn()`, reporting success/failure and peak CUDA memory either way."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"\n--- attempt: {label} ---")
    try:
        steps_done = run_fn()
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  OK -- {steps_done} steps completed, peak CUDA memory = {peak_mb:.0f} MB")
        return {"label": label, "ok": True, "peak_mb": peak_mb, "steps_done": steps_done}
    except torch.cuda.OutOfMemoryError as e:
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  OOM -- peak CUDA memory before failure = {peak_mb:.0f} MB")
        torch.cuda.empty_cache()
        return {"label": label, "ok": False, "peak_mb": peak_mb, "error": str(e)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type != "cuda":
        print("WARNING: no CUDA device found -- this smoke test is only meaningful on the GPU.")

    set_global_seed(seed=SEED, data_partition_seed=SEED, client_sampling_seed=SEED)

    partition = json.loads(PARTITION_PATH.read_text())
    records = _records_for(partition, ["A", "B", "C"], "train")[:N_SMOKE_EXAMPLES]
    print(f"smoke-test records: n={len(records)} (of the real ~22858-example pooled train set)")

    ds = RawImageDataset(records, build_eval_transform(image_size=IMAGE_SIZE))
    raw_loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    head = DenseNet121Head(fine_tune_last_block=True).to(device)
    n_trainable = count_trainable_parameters(FineTuneTail(head))
    print(f"trainable params in the DP-wrapped tail (denseblock4+norm5+classifier): {n_trainable:,}")

    print("computing frozen-prefix features (outside Opacus, no_grad, eval-style view)...")
    t0 = time.monotonic()
    frozen_features, labels = _frozen_prefix_features(head, raw_loader, device)
    print(f"  done in {time.monotonic() - t0:.1f}s -- feature shape {tuple(frozen_features.shape)}")

    tail_dataset = TensorDataset(frozen_features, labels)

    # Real accountant path, not a placeholder -- confirms ADR-2's noise-calibration
    # machinery itself works for this tail, even though this smoke test's own
    # sample_rate/step count aren't the real experiment's.
    sample_rate = BATCH_SIZE / len(tail_dataset)
    total_steps = compute_total_steps(
        dataset_size=len(tail_dataset), batch_size=BATCH_SIZE, local_epochs=1, num_rounds=NUM_SMOKE_STEPS
    )
    noise_multiplier = compute_noise_multiplier(
        target_epsilon=TARGET_EPSILON, target_delta=TARGET_DELTA, sample_rate=sample_rate, total_steps=total_steps
    )
    print(f"noise_multiplier for eps={TARGET_EPSILON}, delta={TARGET_DELTA}: {noise_multiplier:.4f}")

    results: list[dict] = []

    # --- Attempt 1: full logical batch size, no BatchMemoryManager ---
    # Uses `head`'s own submodules directly (safe -- `head` hasn't touched Opacus
    # yet). Later attempts build a fresh `DenseNet121Head` each time instead of
    # reusing these submodules, because `PrivacyEngine.make_private()` wraps and
    # hooks the module it's given IN PLACE (`GradSampleModule`) -- re-wrapping the
    # same already-hooked submodules a second time is not a safe operation Opacus
    # supports, so `head` is treated as single-use once attempt 1 wraps its tail.
    tail = FineTuneTail(head)
    issues = ModuleValidator.validate(tail, strict=False)
    print(f"ModuleValidator issues on the tail module: {issues if issues else 'none'}")

    opt = torch.optim.Adam(tail.parameters(), lr=1e-4)
    loader = DataLoader(tail_dataset, batch_size=BATCH_SIZE, generator=torch.Generator().manual_seed(SEED))
    privacy_engine = PrivacyEngine(accountant="rdp")
    dp_tail, dp_opt, dp_loader = privacy_engine.make_private(
        module=tail, optimizer=opt, data_loader=loader,
        noise_multiplier=noise_multiplier, max_grad_norm=MAX_GRAD_NORM,
    )
    results.append(_attempt(
        f"logical batch_size={BATCH_SIZE}, no BatchMemoryManager",
        lambda: _run_steps(dp_loader, dp_opt, dp_tail, device, NUM_SMOKE_STEPS),
    ))

    # --- Attempt 2 (only if 1 failed): BatchMemoryManager escalation ladder ---
    if not results[0]["ok"]:
        for physical_bs in BMM_PHYSICAL_BATCH_SIZES:
            tail2 = FineTuneTail(DenseNet121Head(fine_tune_last_block=True).to(device))
            tail2.load_state_dict(FineTuneTail(head).state_dict())
            tail2.to(device)
            opt2 = torch.optim.Adam(tail2.parameters(), lr=1e-4)
            loader2 = DataLoader(tail_dataset, batch_size=BATCH_SIZE, generator=torch.Generator().manual_seed(SEED))
            privacy_engine2 = PrivacyEngine(accountant="rdp")
            dp_tail2, dp_opt2, dp_loader2 = privacy_engine2.make_private(
                module=tail2, optimizer=opt2, data_loader=loader2,
                noise_multiplier=noise_multiplier, max_grad_norm=MAX_GRAD_NORM,
            )

            def run_with_bmm(dp_loader2=dp_loader2, dp_opt2=dp_opt2, dp_tail2=dp_tail2, physical_bs=physical_bs):
                with BatchMemoryManager(
                    data_loader=dp_loader2, max_physical_batch_size=physical_bs, optimizer=dp_opt2
                ) as mem_loader:
                    return _run_steps(mem_loader, dp_opt2, dp_tail2, device, NUM_SMOKE_STEPS)

            outcome = _attempt(
                f"logical batch_size={BATCH_SIZE}, BatchMemoryManager physical_batch_size={physical_bs}",
                run_with_bmm,
            )
            results.append(outcome)
            if outcome["ok"]:
                break

    print("\n=== summary ===")
    for r in results:
        status = "OK" if r["ok"] else "OOM"
        print(f"  [{status}] {r['label']} -- peak {r['peak_mb']:.0f} MB")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "n_trainable_params": n_trainable,
        "n_smoke_examples": len(tail_dataset),
        "batch_size": BATCH_SIZE,
        "target_epsilon": TARGET_EPSILON,
        "target_delta": TARGET_DELTA,
        "noise_multiplier": noise_multiplier,
        "gpu_total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 1e6 if device.type == "cuda" else None,
        "attempts": results,
    }, indent=2))
    print(f"\nResults written: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
