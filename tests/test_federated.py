"""Stage 13 tests that don't require a live Flower run (ClientApp/ServerApp
orchestration is exercised for real via `flwr run`, not simulated in pytest — see
docs/SESSION_STATE.md for the real end-to-end results)."""
import torch

from src.federated.serialization import array_record_to_classifier_state, classifier_state_to_array_record
from src.models.densenet_head import DenseNet121Head
from src.training.trainer import train_local_round


def test_serialization_round_trip_is_lossless():
    model = DenseNet121Head(pretrained=False)
    original_state = model.classifier.state_dict()

    array_record = classifier_state_to_array_record(original_state)
    recovered_state = array_record_to_classifier_state(array_record)

    assert set(recovered_state.keys()) == set(original_state.keys())
    for key in original_state:
        assert torch.equal(original_state[key], recovered_state[key])


def test_serialization_round_trip_preserves_dtype_and_shape():
    model = DenseNet121Head(pretrained=False)
    state = model.classifier.state_dict()
    recovered = array_record_to_classifier_state(classifier_state_to_array_record(state))
    for key in state:
        assert recovered[key].shape == state[key].shape
        assert recovered[key].dtype == state[key].dtype


def test_single_client_round_is_deterministic_and_reproducible():
    """Stage 13's 'single-client FedAvg equals local training' criterion: with only
    one participant, FedAvg's weighted average over one contributor is the identity
    (Flower's own well-tested code, not re-tested here) — what this project's own
    code must get right is that train_local_round (the function client_app.py's
    @app.train() calls) deterministically reproduces the same update given the same
    initial state, data, and hyperparameters, since that's what a single-client round
    reduces to."""
    torch.manual_seed(0)
    base_model = DenseNet121Head(pretrained=False)
    init_state = {k: v.clone() for k, v in base_model.classifier.state_dict().items()}

    features = torch.randn(20, 1, 1024)
    labels = torch.randint(0, 2, (20,))

    def run_once():
        model = DenseNet121Head(pretrained=False)
        model.classifier.load_state_dict({k: v.clone() for k, v in init_state.items()})
        return train_local_round(
            model, features, labels, seed=42, local_epochs=2, lr=0.01, batch_size=8
        )

    result_a = run_once()
    result_b = run_once()

    for key in result_a["classifier_state"]:
        assert torch.equal(result_a["classifier_state"][key], result_b["classifier_state"][key])
    assert result_a["num_examples"] == result_b["num_examples"] == 20
    assert result_a["train_loss"] == result_b["train_loss"]


def test_single_client_round_actually_changes_parameters():
    """A local round must not be a no-op — sanity check that training happened."""
    torch.manual_seed(0)
    model = DenseNet121Head(pretrained=False)
    init_state = {k: v.clone() for k, v in model.classifier.state_dict().items()}

    features = torch.randn(20, 1, 1024)
    labels = torch.randint(0, 2, (20,))
    result = train_local_round(model, features, labels, seed=1, local_epochs=3, lr=0.1, batch_size=8)

    changed = any(
        not torch.equal(init_state[k], result["classifier_state"][k]) for k in init_state
    )
    assert changed
