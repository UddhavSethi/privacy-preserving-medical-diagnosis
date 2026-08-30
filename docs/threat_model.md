# Threat Model

This document is the write-up form of `CLAUDE.md` §6. It is descriptive, not a new
decision — if it and `CLAUDE.md` ever disagree, `CLAUDE.md` is authoritative.

## Assets and layers

The system has four independent protection layers. They are not interchangeable and
each is independently switchable, which is exactly what the ablation ladder
(`docs/results.md`) measures the cost of.

| Layer | Protects | Against | Mechanism |
|---|---|---|---|
| Federated Learning | Raw patient images | Any party outside the hospital | Images never leave local storage — only a small parameter update is ever transmitted |
| Differential Privacy | Information encoded *inside* an update | Inference/reconstruction attacks on updates | Opacus DP-SGD: per-sample gradient clipping + calibrated Gaussian noise + RDP accountant, reported as (epsilon, delta) |
| Secure Aggregation | An individual hospital's update | The server itself (honest-but-curious) | Flower SecAgg+ masking — server observes only the summed aggregate |
| TLS + client authentication | Messages in transit | Network eavesdropping, tampering, impersonation | gRPC over TLS with per-hospital node authentication |

## Threat actors in scope

- **Honest-but-curious server.** Follows the FedAvg protocol correctly but tries to
  infer information from whatever it receives. Countered by Differential Privacy
  (content of an update) and Secure Aggregation (attribution of an update to a
  specific hospital).
- **Passive network adversary.** Eavesdrops on or tampers with traffic between a
  hospital and the server. Countered by TLS.
- **Unregistered party impersonating a hospital.** Countered by client
  (node) authentication — a connection cannot participate in a round without a
  registered key pair.
- **Client collusion up to the SecAgg+ threshold.** Bounded by the protocol's own
  stated dropout/collusion tolerance, not a project-specific guarantee.

## Explicitly out of scope for this phase

- **Malicious clients** submitting poisoned or Byzantine updates. No defense is
  implemented; see `CLAUDE.md` §16.2.
- **Collusion above the SecAgg+ threshold.**
- **Side-channel and physical attacks**, or compromise of a hospital's own
  infrastructure (its OS, its filesystem, its network segment before the TLS
  boundary).

## Known architectural tensions

These are named as contributions of this project, not flaws to be silently
engineered around:

1. **Secure Aggregation vs. Byzantine detection are directly opposed.** SecAgg exists
   specifically so the server cannot see individual updates; Byzantine-robust
   aggregation requires inspecting individual updates to find outliers. Reconciling
   the two needs specific techniques (robust aggregation over secret shares, or
   zero-knowledge validity proofs) that this project does not implement.
2. **DP noise vs. explainability and calibration.** Grad-CAM heatmap quality and MC
   Dropout confidence calibration plausibly degrade as the DP epsilon tightens.
   Measuring this trade-off (rather than assuming it away) is itself part of this
   project's contribution.
3. **Local DP vs. the accuracy objective.** Each hospital adds its own full DP noise
   independently — this is *local* DP, which has strictly worse utility than
   *central* DP at the same epsilon. See "Local vs. distributed DP" below.

## Local vs. distributed DP — the honest limitation

With each hospital adding full Gaussian noise to its own update before Secure
Aggregation ever runs, the system as built provides **local differential privacy**,
not central DP. This is a real utility cost, not a bookkeeping detail: local DP needs
more noise for the same (epsilon, delta) guarantee than a trusted aggregator adding
noise once to the sum would.

The principled fix is **distributed DP**: each hospital adds only a *share* of the
total required noise, and Secure Aggregation's masking ensures the server only ever
sees the fully-noised sum — recovering central-DP utility with no trusted server.
Implementing this rigorously requires discrete Gaussian or Skellam noise over a finite
field, integrated with the SecAgg protocol itself. That is out of scope for this
prototype (`CLAUDE.md` ADR-2) and is named here as the principled extension, not
silently substituted with a weaker claim.

## Why the backbone is frozen (ties into the threat model)

`CLAUDE.md` ADR-1 freezes DenseNet121's backbone and federates only a small
classifier head. This is a modeling decision, but it is also load-bearing for the
threat model: Opacus's per-sample gradient computation — the mechanism the DP
guarantee depends on — is incompatible with BatchNorm in training mode, because
BatchNorm mixes information across samples in a batch. A frozen backbone with
BatchNorm pinned in `eval()` mode is a fixed affine transform, not a batch-mixing
operation, so DP-SGD applies cleanly to the head parameters actually being trained
and federated.

## What this project does not claim

- Not validated for clinical use; not a medical device.
- Not a real multi-institution deployment — hospitals are simulated
  processes/containers on one machine (`CLAUDE.md` §15, item 1).
- Not empirically demonstrated privacy leakage or its absence — the claim that
  "raw updates can leak patient information" is supported by citation to prior
  work, not by an attack this project ran. An empirical membership-inference /
  gradient-inversion study is a named, approved-in-concept future direction
  (`CLAUDE.md` §16.1) that has not been executed.
