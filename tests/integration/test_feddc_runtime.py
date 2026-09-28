from __future__ import annotations

from typing import Any

import pytest
import torch

from gecko.engine.coordinator import FederatedCoordinator
from gecko.algorithms.method_config import validate_method_config
from gecko.algorithms.federated.feddc import FedDCStrategy
from gecko.algorithms.federated.feddc import feddc_corrected_state
from gecko.algorithms.federated.feddc import feddc_delta
from gecko.algorithms.federated.feddc import feddc_weighted_state

from tests.helpers import assert_tensor_map_equal
from tests.helpers import make_stream


def _config(*, alpha: float = 0.1, drift_enabled: bool = True) -> dict[str, Any]:
    return {
        "schema": "uefa-method-config",
        "version": 2,
        "name": "feddc_uefa_v1",
        "strategy": {
            "name": "feddc",
            "parameters": {
                "alpha": alpha,
                "drift_enabled": drift_enabled,
            },
        },
        "continual_method": {"name": "Bare", "parameters": {}},
    }


def test_feddc_weighted_state_delta_and_corrected_model_oracles():
    states = {
        0: {"w": torch.tensor([1.0, 3.0])},
        1: {"w": torch.tensor([5.0, 7.0])},
    }
    averaged = feddc_weighted_state(states, {0: 1, 1: 3})
    assert torch.equal(averaged["w"], torch.tensor([4.0, 6.0]))
    delta = feddc_delta(averaged, states[0])
    assert torch.equal(delta["w"], torch.tensor([3.0, 3.0]))
    corrected = feddc_corrected_state(
        states[0], {"w": torch.tensor([0.5, -1.0])}
    )
    assert torch.equal(corrected["w"], torch.tensor([1.5, 2.0]))


def test_feddc_updates_drift_before_corrected_model_aggregation():
    from gecko.engine.protocol import ClientUpload
    from gecko.engine.protocol import ParameterEntry
    from gecko.engine.protocol import ParameterManifest
    from gecko.engine.protocol import RoundContext

    strategy = FedDCStrategy(alpha=0.1)
    strategy.bind_training_protocol(learning_rate=0.5, local_steps=1)
    strategy.initialize(
        {"weight": torch.zeros(1)},
        ParameterManifest((
            ParameterEntry(
                name="weight",
                kind="shared_trainable",
                shape=(1,),
                dtype="torch.float32",
                requires_grad=True,
                wire_eligible=True,
            ),
        )),
        [0, 1],
    )
    context = RoundContext(0, 0, (0, 1), {0: 0, 1: 0})
    result = strategy.aggregate(
        context,
        (
            ClientUpload(0, 0, 1, 1.0, {"weight": torch.tensor([1.0])}),
            ClientUpload(1, 0, 1, 1.0, {"weight": torch.tensor([3.0])}),
        ),
    )

    assert torch.equal(result.shared_state["weight"], torch.tensor([4.0]))
    state = strategy.state_dict()
    assert torch.equal(state["client_drifts"][0]["weight"], torch.tensor([1.0]))
    assert torch.equal(state["client_drifts"][1]["weight"], torch.tensor([3.0]))
    assert torch.equal(
        state["previous_aggregate_delta"]["weight"], torch.tensor([2.0])
    )


@pytest.mark.parametrize("incremental", ["task", "class", "domain"])
def test_feddc_stateful_runtime_tracks_drift_and_split_wire(incremental: str):
    stream = make_stream("NC", incremental, 2, order_profile="synchronized", rounds=1)
    coordinator = FederatedCoordinator(
        stream,
        "feddc",
        "Bare",
        model_name="uefa_gcn",
        model_seed=109,
        method_config=_config(alpha=0.1),
    )
    runtime = coordinator._stateful_runtime
    assert runtime is not None
    assert isinstance(runtime.strategy, FedDCStrategy)

    result = coordinator.run()

    assert (
        result["method_config_resolution"]["support_status"]
        == "implemented_unverified"
    )
    assert result["method_config_benchmark_eligible"] is False
    diagnostics = result["strategy_diagnostics"]
    assert diagnostics["strategy_version"] == "uefa-feddc-strategy-v1"
    assert diagnostics["alpha"] == 0.1
    assert diagnostics["completed_rounds"] == len(result["rounds"])
    assert diagnostics["corrected_model_rule"] == "x_i_plus_updated_h_i"
    assert diagnostics["drift_update_rule"] == "h_i_plus_equals_h_i_plus_x_i_minus_w"
    assert diagnostics["linear_correction_rule"] == "previous_local_minus_previous_aggregate_over_eta_k"
    assert diagnostics["penalty_rule"] == "alpha_over_two_norm_x_i_plus_h_i_minus_w_squared"
    assert diagnostics["gradient_stabilization"] == "finite_norm_clip_v2_fail_closed"
    assert diagnostics["learning_rate"] > 0
    assert diagnostics["local_steps"] == 1
    assert result["personalized_parameter_count"] == 0
    ledger = result["resource_ledger"]
    assert ledger["training_model_uplink_bytes"] > 0
    assert ledger["training_model_downlink_bytes"] > 0
    assert ledger["training_auxiliary_uplink_bytes"] > 0
    assert ledger["client_persistent_bytes"] > 0
    assert ledger["server_persistent_bytes"] > 0
    assert ledger["training_wire_bytes"] > ledger["communication_payload_bytes"]


def test_feddc_objective_matches_penalty_and_linear_correction_equation():
    from gecko.engine.protocol import ParameterEntry
    from gecko.engine.protocol import ParameterManifest

    strategy = FedDCStrategy(alpha=0.2)
    strategy.bind_training_protocol(learning_rate=0.5, local_steps=2)
    strategy.initialize(
        {"weight": torch.zeros(1, 2)},
        ParameterManifest((
            ParameterEntry(
                name="weight",
                kind="shared_trainable",
                shape=(1, 2),
                dtype="torch.float32",
                requires_grad=True,
                wire_eligible=True,
            ),
        )),
        [0],
    )
    strategy._client_drifts[0] = strategy._client_drifts[0].__class__({"weight": torch.tensor([[0.5, -0.5]])})
    strategy._previous_local_deltas[0] = strategy._previous_local_deltas[0].__class__({"weight": torch.tensor([[2.0, 3.0]])})
    strategy._previous_aggregate_delta = strategy._previous_aggregate_delta.__class__({"weight": torch.tensor([[1.0, 1.5]])})
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[10.0, -10.0]]))
    context = type("Context", (), {"client_id": 0})()

    loss = strategy.augment_loss(
        model,
        context,
        model.weight.sum() * 0.0,
        ("weight",),
    )
    loss.backward()

    penalty_gradient = 0.2 * torch.tensor([[10.5, -10.5]])
    correction_gradient = torch.tensor([[1.0, 1.5]])
    expected = penalty_gradient + correction_gradient
    assert torch.allclose(model.weight.grad, expected)


def test_feddc_gradient_transform_rejects_nonfinite_gradients():
    from gecko.engine.protocol import ParameterEntry
    from gecko.engine.protocol import ParameterManifest

    strategy = FedDCStrategy(alpha=0.2)
    strategy.bind_training_protocol(learning_rate=0.1, local_steps=1)
    strategy.initialize(
        {"weight": torch.zeros(1, 2)},
        ParameterManifest((
            ParameterEntry(
                name="weight",
                kind="shared_trainable",
                shape=(1, 2),
                dtype="torch.float32",
                requires_grad=True,
                wire_eligible=True,
            ),
        )),
        [0],
    )
    strategy.gradient_clip_norm = 1.0
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[float("inf"), 1000.0]])
    context = type("Context", (), {"client_id": 0})()

    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        strategy.transform_gradients(model, context, ("weight",))


def test_feddc_finalize_upload_rejects_nonfinite_loss():
    from gecko.engine.protocol import ParameterEntry
    from gecko.engine.protocol import ParameterManifest

    strategy = FedDCStrategy(alpha=0.01)
    shared = {"weight": torch.zeros(1, 2)}
    strategy.initialize(
        shared,
        ParameterManifest((
            ParameterEntry(
                name="weight",
                kind="shared_trainable",
                shape=(1, 2),
                dtype="torch.float32",
                requires_grad=True,
                wire_eligible=True,
            ),
        )),
        [0],
    )
    client = type("Client", (), {"client_id": 0})()
    local_result = type("Result", (), {
        "client_id": 0,
        "global_task_id": 0,
        "shared_state": {"weight": torch.ones(1, 2)},
        "weight": 1,
        "training_loss": float("inf"),
    })()
    context = type("Context", (), {"client_id": 0, "global_task_id": 0})()

    with pytest.raises(ValueError, match="Training loss must be finite"):
        strategy.finalize_upload(client, local_result, context)


def test_feddc_disabled_mechanism_is_explicit_diagnostic_resolution():
    resolution = validate_method_config(
        _config(alpha=0.0, drift_enabled=False),
        problem_type="NC",
        incremental_setting="class",
    )
    assert resolution.support_status == "diagnostic_only_disabled_mechanism"


def test_feddc_checkpoint_resume_is_equivalent_to_uninterrupted_run(tmp_path, monkeypatch):
    stream = make_stream(
        "NC", "class", 2, seed=61, order_profile="synchronized", rounds=2
    )
    baseline = FederatedCoordinator(
        stream,
        "feddc",
        "Bare",
        model_name="uefa_gcn",
        model_seed=113,
        method_config=_config(alpha=0.1),
    )
    baseline_result = baseline.run()

    interrupted = FederatedCoordinator(
        stream,
        "feddc",
        "Bare",
        model_name="uefa_gcn",
        model_seed=113,
        method_config=_config(alpha=0.1),
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
    )
    runtime = interrupted._stateful_runtime
    assert runtime is not None
    original_write = runtime._write_checkpoint

    class SimulatedInterruption(RuntimeError):
        pass

    def write_then_interrupt(checkpoint_id: str) -> None:
        original_write(checkpoint_id)
        if checkpoint_id == "round-00000001":
            raise SimulatedInterruption

    monkeypatch.setattr(runtime, "_write_checkpoint", write_then_interrupt)
    with pytest.raises(SimulatedInterruption):
        interrupted.run()

    resumed = FederatedCoordinator(
        stream,
        "feddc",
        "Bare",
        model_name="uefa_gcn",
        model_seed=113,
        method_config=_config(alpha=0.1),
        resume_from=tmp_path / "round-00000001.manifest.json",
    )
    resumed_result = resumed.run()

    assert torch.allclose(
        baseline_result["A[k,s,t]"],
        resumed_result["A[k,s,t]"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert_tensor_map_equal(
        baseline._stateful_runtime.strategy.state_dict(),
        resumed._stateful_runtime.strategy.state_dict(),
    )
    assert resumed_result["resume_validation"]["identity_matched"] is True
