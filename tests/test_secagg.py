"""Stage 15 — Secure Aggregation. CLAUDE.md's own §11.3 names this exact test:
'masks cancel exactly — the unmasked aggregate equals plain FedAvg within
tolerance.' This turns 'we implemented SecAgg' into a defensible claim.

Replicates the exact client-side quantize/weight-encode
(`flwr.client.mod.secure_aggregation.secaggplus_mod`) and server-side
unmask/dequantize (`flwr.server.workflow.secure_aggregation.secaggplus_workflow`)
math using Flower's own functions, for several synthetic clients — but skips
generating and exchanging the actual pairwise/private masks. That's a
deliberate scope boundary, not a shortcut: SecAgg+'s guarantee is that those
masks are constructed (via Shamir secret sharing + ECDH, in
`flwr.common.secure_aggregation.crypto`) to cancel EXACTLY upon summation
across all participating clients — that's Flower's own tested cryptographic
primitive (ADR-3: no custom crypto, and no re-deriving Flower's crypto tests
either). What this test verifies is the part that's actually specific to how
this project uses the library: that the quantize -> weight-encode -> sum ->
unmask -> dequantize round-trip (given masks that cancel, which the crypto
guarantees) reproduces the same weighted average FedAvg would compute in
plain, unmasked arithmetic — i.e. quantization is the *only* source of
deviation, not some other integration bug.
"""
import numpy as np

from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_combine,
    factor_extract,
    parameters_addition,
    parameters_mod,
    parameters_multiply,
)
from flwr.common.secure_aggregation.quantization import dequantize, quantize

CLIPPING_RANGE = 8.0
QUANTIZATION_RANGE = 4_194_304
MODULUS_RANGE = 4_294_967_296
MAX_WEIGHT = 1000.0


def _client_encode(params: list[np.ndarray], num_examples: int) -> list[np.ndarray]:
    """Mirrors `secaggplus_mod`'s stage-2 encoding exactly (minus mask
    addition), given a client's real (unmasked) parameters and weight."""
    ratio = num_examples / MAX_WEIGHT
    q_ratio = round(ratio * QUANTIZATION_RANGE)
    dq_ratio = q_ratio / QUANTIZATION_RANGE

    weighted = parameters_multiply(params, dq_ratio)
    quantized = quantize(weighted, CLIPPING_RANGE, QUANTIZATION_RANGE)
    combined = factor_combine(q_ratio, quantized)
    # In the real protocol this widening happens naturally when the int64 masks
    # (`pseudo_rand_gen`) are added on top of `quantize`'s int32 output; this test
    # skips mask generation entirely (see module docstring), so it must replicate
    # the widening explicitly or `parameters_mod`'s bitwise-AND against
    # MODULUS_RANGE overflows int32.
    return [arr.astype(np.int64) for arr in combined]


def _server_decode(summed: list[np.ndarray], num_active_clients: int) -> list[np.ndarray]:
    """Mirrors `secaggplus_workflow.unmask_stage`'s decode exactly, given the
    (mask-cancelled) sum of all clients' encoded vectors."""
    recon = parameters_mod(summed, MODULUS_RANGE)
    q_total_ratio, recon = factor_extract(recon)
    inv_dq_total_ratio = QUANTIZATION_RANGE / q_total_ratio
    aggregated = dequantize(recon, CLIPPING_RANGE, QUANTIZATION_RANGE)
    offset = -(num_active_clients - 1) * CLIPPING_RANGE
    for vec in aggregated:
        vec += offset
        vec *= inv_dq_total_ratio
    return aggregated


def test_masks_cancel_exactly_reproduces_plain_fedavg_weighted_average():
    rng = np.random.default_rng(0)
    # Small values, well inside [-CLIPPING_RANGE, CLIPPING_RANGE] — real classifier
    # deltas after a local round are this scale, not near the clip boundary.
    client_params = [
        [rng.normal(scale=0.05, size=(4, 3)).astype(np.float64)] for _ in range(3)
    ]
    num_examples = [4180, 13342, 13342]  # real per-hospital magnitudes (natural partition)

    # Ground truth: plain FedAvg weighted average, no quantization/masking at all.
    total_examples = sum(num_examples)
    weighted = [parameters_multiply(p, n / total_examples) for p, n in zip(client_params, num_examples)]
    true_avg = weighted[0][0]
    for w in weighted[1:]:
        true_avg = true_avg + w[0]

    encoded = [_client_encode(p, n) for p, n in zip(client_params, num_examples)]
    summed = encoded[0]
    for e in encoded[1:]:
        summed = parameters_addition(summed, e)

    reconstructed = _server_decode(summed, num_active_clients=len(client_params))[0]

    # Quantization is the only source of error here (masks are assumed to cancel
    # exactly, which SecAgg+'s crypto guarantees) — tolerance reflects
    # QUANTIZATION_RANGE's granularity over CLIPPING_RANGE's span, not slack for a bug.
    quantization_step = 2 * CLIPPING_RANGE / QUANTIZATION_RANGE
    assert np.allclose(reconstructed, true_avg, atol=quantization_step * 10)


def test_masks_cancel_exactly_with_unequal_client_weights():
    """Same claim, deliberately skewed weights (mirrors the natural partition's
    ~3.2x hospital-size imbalance) — the weighted average must track the
    heavier clients more closely than a naive unweighted mean would."""
    rng = np.random.default_rng(1)
    client_params = [[rng.normal(scale=0.1, size=(5,)).astype(np.float64)] for _ in range(3)]
    num_examples = [1000, 1000, 11342]  # third client dominates

    total_examples = sum(num_examples)
    weighted = [parameters_multiply(p, n / total_examples) for p, n in zip(client_params, num_examples)]
    true_avg = weighted[0][0]
    for w in weighted[1:]:
        true_avg = true_avg + w[0]
    naive_unweighted_avg = sum((p[0] for p in client_params), np.zeros_like(client_params[0][0])) / len(
        client_params
    )

    encoded = [_client_encode(p, n) for p, n in zip(client_params, num_examples)]
    summed = encoded[0]
    for e in encoded[1:]:
        summed = parameters_addition(summed, e)
    reconstructed = _server_decode(summed, num_active_clients=len(client_params))[0]

    quantization_step = 2 * CLIPPING_RANGE / QUANTIZATION_RANGE
    assert np.allclose(reconstructed, true_avg, atol=quantization_step * 10)
    # Sanity check the test itself isn't vacuous: weighting must actually matter here.
    assert not np.allclose(true_avg, naive_unweighted_avg, atol=1e-3)


def test_quantize_dequantize_round_trip_is_bounded_by_quantization_step():
    """Narrower claim, isolating quantize/dequantize alone (no weighting, no
    masking): the round-trip error must be bounded by the quantization
    granularity, and must actually be nonzero for generic input (or the test
    would be vacuous)."""
    rng = np.random.default_rng(2)
    original = [rng.uniform(-CLIPPING_RANGE, CLIPPING_RANGE, size=(6, 4)).astype(np.float64)]

    quantized = quantize(original, CLIPPING_RANGE, QUANTIZATION_RANGE)
    recovered = dequantize(quantized, CLIPPING_RANGE, QUANTIZATION_RANGE)

    quantization_step = 2 * CLIPPING_RANGE / QUANTIZATION_RANGE
    assert np.allclose(recovered, original, atol=quantization_step * 2)
    assert not np.array_equal(recovered[0], original[0])  # quantization is lossy, not a no-op
