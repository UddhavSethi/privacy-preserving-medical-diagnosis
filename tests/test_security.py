from src.federated.security import (
    assert_payload_within_message_limit,
    classifier_payload_size_bytes,
)


def test_classifier_payload_is_the_expected_head_only_size():
    """ADR-1's whole premise is that the federated payload is small (head
    only, not the ~7M-parameter backbone). Real measured size should be on
    the order of the plan's own ~1MB estimate (Stage 13's live runs logged
    ArrayRecord sizes around 1.00-1.003 MB), not remotely close to a full
    DenseNet121's ~28MB fp32 footprint."""
    size = classifier_payload_size_bytes()
    assert 100_000 < size < 5_000_000


def test_classifier_payload_fits_well_within_grpc_message_limit():
    """Stage 16's own testing criterion: confirmation that the configured
    message length exceeds the actual update size."""
    size = assert_payload_within_message_limit()
    assert size > 0
