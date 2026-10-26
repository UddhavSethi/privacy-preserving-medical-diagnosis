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
- Not a proof of privacy from the empirical attacks. Membership-inference results
  (`docs/privacy_attack.md`, `docs/audit_2026-10.md`) are evidence against the specific
  attacks run, nothing more: a low attack score does not bound leakage, and the formal
  (epsilon, delta) guarantee is the privacy claim. Gradient inversion has not been run.
- Not mutual TLS. Transport is gRPC over **server-side TLS**; hospitals authenticate with
  Flower node authentication (a per-hospital ECDSA key pair registered with the server, used
  to sign requests). Both parties are authenticated, but the client identity is established at
  the application layer, not by a client certificate in the TLS handshake.
- Not a strong Secure Aggregation guarantee at three clients. SecAgg+ hides one client's
  update from the server only against collusion below the threshold; with `num_shares=3`,
  `reconstruction_threshold=2` the server plus ONE colluding hospital still cannot isolate a
  third hospital's update, but the server plus TWO colluding hospitals trivially can (the
  aggregate minus the two known updates). In-scope collusion is therefore limited to a single
  hospital, and with only three participants the aggregate itself reveals a lot.
- Not a claim that TLS and client authentication were active in every experiment: the
  simulation engine has no network, so the measured ablation rows exercise FedAvg, SecAgg+ and
  DP only; TLS and node authentication are exercised by the Docker Compose deployment
  (`scripts/run_deployment.sh`). `docs/audit_2026-10.md` lists which mechanism was active in
  which run.
- DP is sample-level and per hospital, and its epsilon is the accountant's value for a single
  hospital's own training set. Class weights derived from a hospital's label counts, early
  stopping, threshold and temperature selection on validation data, and the released
  number of training examples are outside the accountant.
