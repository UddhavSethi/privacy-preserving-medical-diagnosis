"""DenseNet121 with a frozen ImageNet-pretrained backbone and a small trainable
classifier head (ADR-1 — the load-bearing decision of the whole project).

Dropout placement: **head-only** (a single Dropout layer inside the trainable head,
between its hidden layer and the final classification layer). Chosen over
after-dense-block placement because it is simpler, requires no extra fine-tuning of
backbone-adjacent layers, and keeps the entire backbone/head separation ADR-1 depends
on completely clean — the tradeoff (captures only last-layer uncertainty for MC
Dropout, Stage 19) is accepted. This resolves CLAUDE.md section 14's dropout-placement
pending decision.

**`fine_tune_last_block` (added 2026-08-31, ADR-1's own documented approved
fallback, owner-approved before implementation — see `docs/adr1_groupnorm_fallback.md`
for the full rationale and evaluation).** Debugging a real "the model looks at the
wrong region" complaint traced to a structural cause: with the backbone fully frozen,
the classifier only ever sees a globally-average-pooled 1024-number summary (`pool` /
`pooled_features` below) — no spatial layout survives to reach it, so Grad-CAM's
heatmap (computed from backbone gradients) is at best a reconstruction of what the
classifier's decision correlates with, not what it actually used. Measured effect:
quantitative Grad-CAM localization (`docs/gradcam_localization.md`'s own pointing-game
metric) collapses from an already-weak 18.6% to 0.0% specifically on images the
frozen-backbone model gets wrong. `fine_tune_last_block=True` leaves everything through
`transition3` frozen exactly as before (ADR-1's DP/FL/VRAM reasoning is unaffected
there) but unfreezes `denseblock4` + `norm5` and swaps their BatchNorm for GroupNorm via
Opacus's `ModuleValidator.fix()` — DP-compatible per-sample statistics, ADR-1's own
named fallback — so the model can adapt spatial features to chest X-rays instead of
relying entirely on frozen generic ImageNet channels behind a fixed pooling op. Default
remains `False`: every existing checkpoint (`outputs/checkpoints/**/*.pt`) was trained
against the frozen-only architecture and loads exactly as before with the default.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from opacus.validators import ModuleValidator

from src.models.freezing import freeze_batchnorm, freeze_module

# features submodule names that stay frozen (ADR-1's original scheme) even when
# fine_tune_last_block=True -- only denseblock4 + norm5 (the trainable tail) are
# excluded from this list.
_FROZEN_PREFIX_NAMES = (
    "conv0", "norm0", "relu0", "pool0",
    "denseblock1", "transition1",
    "denseblock2", "transition2",
    "denseblock3", "transition3",
)


class DenseNet121Head(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
        fine_tune_last_block: bool = False,
    ) -> None:
        super().__init__()
        weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = tv_models.densenet121(weights=weights)

        self.features = densenet.features  # ADR-1's backbone
        num_backbone_features = densenet.classifier.in_features  # 1024
        self.fine_tune_last_block = fine_tune_last_block

        if fine_tune_last_block:
            # Freeze the same prefix ADR-1 always froze; denseblock4 + norm5 stay
            # trainable below (requires_grad=True is torchvision's pretrained-load
            # default -- nothing to set here).
            for name in _FROZEN_PREFIX_NAMES:
                submodule = getattr(self.features, name)
                freeze_module(submodule)
                freeze_batchnorm(submodule)
            # ADR-1's own named fallback: BatchNorm -> GroupNorm in the trainable
            # tail only (DP-per-sample-safe; verified via ModuleValidator.validate()
            # returning zero issues -- see docs/adr1_groupnorm_fallback.md).
            self.features.denseblock4 = ModuleValidator.fix(self.features.denseblock4)
            self.features.norm5 = ModuleValidator.fix(self.features.norm5)
        else:
            freeze_module(self.features)
            freeze_batchnorm(self.features)

        # Split into a parameter-free pooling step and the trainable classifier so
        # Stage 9's feature cache can store pooled_features(x) output (1024-dim) and
        # later train only `classifier` on it, without re-running the backbone.
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.classifier = nn.Sequential(
            nn.Linear(num_backbone_features, hidden_dim),
            # inplace=False: Opacus's per-sample-gradient hooks (GradSampleModule) need
            # to track intermediate activations by view; an in-place ReLU here breaks
            # that and raises at backward time — confirmed empirically while validating
            # Stage 8, not a theoretical concern.
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_rate),  # deliberately inserted — see module docstring
            nn.Linear(hidden_dim, num_classes),
        )

    def train(self, mode: bool = True) -> "DenseNet121Head":
        """Override so .train() can never re-enable the frozen prefix's BatchNorm
        training behavior. This permanence (not just calling .eval() once at init) is
        what makes frozen BatchNorm safe throughout an entire training run, including
        whatever a federated client's training loop does with .train(). When
        fine_tune_last_block=True, only the frozen prefix is force-eval'd here --
        denseblock4/norm5 are GroupNorm (no running stats, no per-batch mixing) and
        are safe to leave in train mode along with the classifier."""
        super().train(mode)
        if self.fine_tune_last_block:
            for name in _FROZEN_PREFIX_NAMES:
                getattr(self.features, name).eval()
        else:
            self.features.eval()
        return self

    def trainable_state_dict(self) -> dict:
        """The subset of `state_dict()` that actually varies with training --
        `classifier` always, plus `features.denseblock4`/`features.norm5` when
        `fine_tune_last_block=True`. Added for the federated fine-tuning pilot
        (`src/federated/client_app_finetune.py`): transmitting only this subset,
        not the full model, keeps the federated payload at ~9.7MB (the trainable
        tail) instead of ~28MB (the whole backbone) -- the exact payload-size
        concern ADR-1 originally named, still avoided. NOT used by the default
        (fine_tune_last_block=False) path elsewhere in this project -- that path's
        checkpoint format (`model.classifier.state_dict()`, unprefixed keys) is
        unchanged for backward compatibility with every existing checkpoint."""
        state = {f"classifier.{k}": v for k, v in self.classifier.state_dict().items()}
        if self.fine_tune_last_block:
            state.update({f"features.denseblock4.{k}": v for k, v in self.features.denseblock4.state_dict().items()})
            state.update({f"features.norm5.{k}": v for k, v in self.features.norm5.state_dict().items()})
        return state

    def load_trainable_state_dict(self, state: dict) -> None:
        """Inverse of `trainable_state_dict()`."""
        self.classifier.load_state_dict(
            {k.removeprefix("classifier."): v for k, v in state.items() if k.startswith("classifier.")}
        )
        if self.fine_tune_last_block:
            self.features.denseblock4.load_state_dict(
                {k.removeprefix("features.denseblock4."): v for k, v in state.items() if k.startswith("features.denseblock4.")}
            )
            self.features.norm5.load_state_dict(
                {k.removeprefix("features.norm5."): v for k, v in state.items() if k.startswith("features.norm5.")}
            )

    def pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        """Everything up to but not including the trainable classifier: frozen
        backbone -> ReLU -> pool -> flatten. This is exactly what Stage 9's feature
        cache precomputes and stores (`src/data/feature_cache.py`), since it is a pure
        function of a frozen model — identical input always yields identical output.
        """
        features = self.features(x)
        # DenseNet's own final backbone layer (features.norm5) is a BatchNorm with no
        # trailing activation baked in — replicate torchvision's own forward() here,
        # since self.features is used standalone rather than through the full
        # DenseNet.forward().
        features = torch.relu(features)  # torch.relu is always out-of-place
        return self.pool(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pooled_features(x))
