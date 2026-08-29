"""DenseNet121 with a frozen ImageNet-pretrained backbone and a small trainable
classifier head (ADR-1 — the load-bearing decision of the whole project).

Dropout placement: **head-only** (a single Dropout layer inside the trainable head,
between its hidden layer and the final classification layer). Chosen over
after-dense-block placement because it is simpler, requires no extra fine-tuning of
backbone-adjacent layers, and keeps the entire backbone/head separation ADR-1 depends
on completely clean — the tradeoff (captures only last-layer uncertainty for MC
Dropout, Stage 19) is accepted. This resolves CLAUDE.md section 14's dropout-placement
pending decision.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

from src.models.freezing import freeze_batchnorm, freeze_module


class DenseNet121Head(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = tv_models.densenet121(weights=weights)

        self.features = densenet.features  # frozen backbone (ADR-1)
        num_backbone_features = densenet.classifier.in_features  # 1024

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
        """Override so .train() can never re-enable the frozen backbone's BatchNorm
        training behavior. This permanence (not just calling .eval() once at init) is
        what makes the frozen BatchNorm safe throughout an entire training run,
        including whatever a federated client's training loop does with .train()."""
        super().train(mode)
        self.features.eval()
        return self

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
