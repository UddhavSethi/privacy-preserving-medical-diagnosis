"""OPT-2 — empirical privacy attacks (owner-approved 2026-08-30, Phase 6 priority 2).

CLAUDE.md's central premise — "model updates are not automatically safe... shared
updates can leak information about training images" — is, before this module,
supported only by citation (CLAUDE.md section 15, item 9: "Empirical privacy
leakage is asserted from literature, not demonstrated"). This module demonstrates
it: a loss-based membership inference attack (Yeom et al. 2018, "Privacy Risk in
Machine Learning") measuring whether an adversary who can query the trained
classifier can tell whether a specific patient's record was used in training.

**Scope, deliberately narrower than everything the plan's OPT-2 write-up
mentions**: loss-based/confidence-based membership inference only. Gradient
inversion (reconstructing images directly from gradients) is explicitly NOT
implemented — it is a substantially heavier, more finicky research undertaking
(optimization-based reconstruction, image priors, careful hyperparameter tuning to
even get a legible result) that the plan's own risk note already flags as
"finicky" and marks optional ("and optionally gradient inversion"). Loss-based MIA
is the well-established, easily-defensible baseline for measuring memorization/
privacy leakage and is what this module measures.

**Attack methodology.** "Member" = a training example the model was actually
trained on (this project's pooled train set — for a federated checkpoint, the
union of every hospital's train split, since that is what the aggregated global
model was collectively exposed to over the course of training). "Non-member" = a
held-out example the model never saw during training (the pooled test set). For
each example, the attack score is NEGATIVE per-example cross-entropy loss (higher
score = lower loss = more likely to be a training member — the standard intuition
that a model fits its training data better than unseen data, and DP-SGD's clipping
+ noise exist specifically to suppress this effect). Attack strength is reported as
the AUROC of using this score to distinguish members from non-members: 0.5 = no
membership signal at all (ideal — indistinguishable from a coin flip); 1.0 =
perfect membership inference (total leakage). This is exactly analogous to how
`src/evaluation/metrics.py` already reports classification AUROC — reusing the
same well-understood, threshold-free metric for a different question.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


def per_example_cross_entropy_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """probs: (N, C) softmax probabilities. labels: (N,) integer class indices.
    Returns (N,) per-example loss -log(p_true_class)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(f"probs has {probs.shape[0]} rows but labels has {labels.shape[0]}")
    true_class_probs = probs[np.arange(len(labels)), labels]
    return -np.log(np.clip(true_class_probs, eps, 1.0))


@dataclass(frozen=True)
class MembershipInferenceResult:
    attack_auroc: float  # 0.5 = no leakage signal, 1.0 = perfect membership inference
    mean_member_loss: float
    mean_nonmember_loss: float
    generalization_gap: float  # mean_nonmember_loss - mean_member_loss; 0 = no overfitting signal
    n_members: int
    n_nonmembers: int

    def to_dict(self) -> dict:
        return asdict(self)


def run_membership_inference_attack(
    member_loss: np.ndarray, nonmember_loss: np.ndarray
) -> MembershipInferenceResult:
    """Given per-example losses already computed for the member and non-member
    sets, runs the loss-threshold membership inference attack and reports its
    AUROC. Attack score = -loss (lower loss -> higher score -> more likely member)."""
    member_loss = np.asarray(member_loss, dtype=float)
    nonmember_loss = np.asarray(nonmember_loss, dtype=float)
    if len(member_loss) == 0 or len(nonmember_loss) == 0:
        raise ValueError("both member_loss and nonmember_loss must be non-empty")

    scores = np.concatenate([-member_loss, -nonmember_loss])
    is_member = np.concatenate([np.ones(len(member_loss)), np.zeros(len(nonmember_loss))])
    attack_auroc = float(roc_auc_score(is_member, scores))

    return MembershipInferenceResult(
        attack_auroc=attack_auroc,
        mean_member_loss=float(member_loss.mean()),
        mean_nonmember_loss=float(nonmember_loss.mean()),
        generalization_gap=float(nonmember_loss.mean() - member_loss.mean()),
        n_members=len(member_loss),
        n_nonmembers=len(nonmember_loss),
    )
