from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest
import torch

from gecko.engine import runtime as runtime_module
from gecko.engine.coordinator import FederatedCoordinator
from gecko.algorithms.federated.legacy import LegacyStrategyAdapter
from gecko.algorithms.federated.scaffold import ScaffoldStrategy
from gecko.data.datasets.synthetic import build_synthetic_spec
from gecko.data.streams import StreamBuilder

from tests.helpers import assert_tensor_map_equal
from tests.helpers import make_config


def _scaffold_config(
    *, correction_enabled: bool = True, control_updates_enabled: bool = True
) -> dict[str, Any]:
    """Return the canonical YAML-shaped SCAFFOLD method configuration."""

    return {
        "schema": "uefa-method-config",
        "version": 2,
        "name": "scaffold_uefa_adam_v1",
        "strategy": {
            "name": "scaffold",
            "parameters": {
                "control_updates_enabled": control_updates_enabled,
                "correction_enabled": correction_enabled,
            },
        },
        "continual_method": {"name": "Bare", "parameters": {}},
    }


def _fedavg_config() -> dict[str, Any]:
    return {
        "schema": "uefa-method-config",
        "version": 2,
        "name": "legacy_adapter_v1",
        "strategy": {"name": "fedavg", "parameters": {}},
        "continual_method": {"name": "Bare", "parameters": {}},
    }


def _partial_stream(*, seed: int = 211):
    config = make_config(
        "NC",
        "class",
        2,
        seed=seed,
        order_profile="synchronized",
        rounds=2,
    )
    config = replace(
        config,
        training=replace(config.training, participation_fraction=0.5),
    )
    scenario = build_synthetic_spec(
        "NC",
        "class",
        num_tasks=2,
        num_clients=config.partition.num_clients,
        seed=seed,
    )
    return StreamBuilder(config).build(scenario)


def _coordinator(
    stream,
    *,
    strategy: str,
    method_config: dict[str, Any],
    model_seed: int = 223,
    **runtime_options: Any,
) -> FederatedCoordinator:
    return FederatedCoordinator(
        stream,
        strategy,
        "Bare",
        model_name="uefa_gcn",
        model_seed=model_seed,
        method_config=method_config,
        **runtime_options,
    )


def _without_timing(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key
        not in {
            "round_runtime_seconds",
            "evaluation_runtime_seconds",
            "stage_runtime_seconds",
        }
    }


def test_scaffold_runtime_partial_participation_and_split_accounting():
    stream = _partial_stream()
    coordinator = _coordinator(
        stream,
        strategy="scaffold",
        method_config=_scaffold_config(),
    )
    runtime = coordinator._stateful_runtime
    assert runtime is not None
    assert isinstance(runtime.strategy, ScaffoldStrategy)

    result = coordinator.run()
    assert result["personalized_global_performance_gap"] == {
        "applicable": False,
        "status": "not_applicable_shared_only_strategy",
        "macro_personalized_minus_global": None,
        "per_client": {},
    }
    records = result["rounds"]
    assert all(record["participants"] == 1 for record in records)
    for stage in range(stream.scenario.num_tasks):
        assert {
            client_id
            for record in records
            if record["stage"] == stage
            for client_id in record["participant_ids"]
        } == set(coordinator.clients)

    expected_counts = {
        str(client_id): sum(
            client_id in record["participant_ids"] for record in records
        )
        for client_id in coordinator.clients
    }
    assert result["strategy_diagnostics"][
        "client_control_update_counts"
    ] == expected_counts
    final_stage = stream.scenario.num_tasks - 1
    assert runtime.strategy.state_dict()["last_all_client_weights"] == {
        client_id: coordinator._shard_weight(
            stream.shards[client_id][
                stream.orders.global_task(client_id, final_stage)
            ]
        )
        for client_id in coordinator.clients
    }

    model_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in coordinator.global_state.values()
    )
    transmissions = len(records)
    num_clients = len(coordinator.clients)
    num_stages = stream.scenario.num_tasks
    ledger = result["resource_ledger"]
    assert ledger["initialization_model_downlink_bytes"] == num_clients * model_bytes
    assert (
        ledger["initialization_auxiliary_downlink_bytes"]
        == num_clients * model_bytes
    )
    assert ledger["training_model_uplink_bytes"] == transmissions * model_bytes
    assert ledger["training_model_downlink_bytes"] == transmissions * model_bytes
    assert ledger["training_auxiliary_uplink_bytes"] == transmissions * model_bytes
    assert (
        ledger["training_auxiliary_downlink_bytes"]
        == transmissions * model_bytes
    )
    assert ledger["communication_payload_bytes"] == 2 * transmissions * model_bytes
    assert ledger["training_wire_bytes"] == 2 * ledger["communication_payload_bytes"]
    assert ledger["evaluation_sync_bytes"] == num_clients * num_stages * model_bytes


@pytest.mark.gpu
def test_state_delta_l2_keeps_mixed_device_diagnostic_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the mixed-device regression oracle.")
    before = {"weight": torch.tensor([1.0, 2.0])}
    after = {"weight": torch.tensor([2.0, 4.0], device="cuda:0")}

    measured = FederatedCoordinator._state_delta_l2(before, after)

    assert measured == pytest.approx(5.0**0.5)
    assert before["weight"].device.type == "cpu"
    assert after["weight"].device.type == "cuda"


def test_scaffold_runtime_reports_actual_post_local_client_delta():
    stream = _partial_stream(seed=227)
    coordinator = _coordinator(
        stream,
        strategy="scaffold",
        method_config=_scaffold_config(),
        model_seed=229,
    )
    runtime = coordinator._stateful_runtime
    assert runtime is not None
    runtime._initial_broadcast()
    participant = int(stream.participation.trace[0][0][0])
    client = coordinator.clients[participant]
    before = coordinator.parameter_policy.extract(client.model)

    record = runtime._run_round(0, 0)

    after = coordinator.parameter_policy.extract(client.model)
    expected = coordinator._state_delta_l2(before, after)
    assert record["client_parameter_delta_l2"] == {participant: expected}
    assert expected > 0.0


def test_scaffold_both_disabled_delegates_to_exact_fedavg():
    stream = _partial_stream(seed=233)
    fedavg = _coordinator(
        stream,
        strategy="fedavg",
        method_config=_fedavg_config(),
        model_seed=239,
    )
    scaffold = _coordinator(
        stream,
        strategy="scaffold",
        method_config=_scaffold_config(
            correction_enabled=False,
            control_updates_enabled=False,
        ),
        model_seed=239,
    )
    assert isinstance(scaffold._stateful_runtime.strategy, LegacyStrategyAdapter)
    assert scaffold._stateful_runtime.strategy.name == "fedavg"

    fedavg_result = fedavg.run()
    scaffold_result = scaffold.run()

    assert torch.allclose(
        fedavg_result["client_stage_task_matrix"],
        scaffold_result["client_stage_task_matrix"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert fedavg_result["summary"] == scaffold_result["summary"]
    for expected, actual in zip(fedavg_result["rounds"], scaffold_result["rounds"]):
        for key in (
            "stage",
            "round",
            "participant_ids",
            "participant_global_task_ids",
            "raw_aggregation_weights",
            "normalized_aggregation_weights",
            "normalized_aggregation_weight_sum",
            "empty_updates",
            "mean_training_loss",
            "communication_payload_bytes",
            "client_parameter_delta_l2",
            "server_parameter_delta_l2",
        ):
            assert actual[key] == expected[key]
    assert_tensor_map_equal(fedavg.global_state, scaffold.global_state)
    for client_id in fedavg.clients:
        assert_tensor_map_equal(
            fedavg.clients[client_id].model.state_dict(),
            scaffold.clients[client_id].model.state_dict(),
        )
    for counter in (
        "initialization_model_downlink_bytes",
        "training_model_uplink_bytes",
        "training_model_downlink_bytes",
        "communication_payload_bytes",
        "training_wire_bytes",
        "evaluation_sync_bytes",
    ):
        assert (
            scaffold_result["resource_ledger"][counter]
            == fedavg_result["resource_ledger"][counter]
        )
    assert scaffold_result["resource_ledger"][
        "initialization_auxiliary_downlink_bytes"
    ] == 0
    assert scaffold_result["resource_ledger"]["training_auxiliary_uplink_bytes"] == 0
    assert scaffold_result["resource_ledger"][
        "training_auxiliary_downlink_bytes"
    ] == 0


def test_scaffold_runtime_checkpoint_resume_matches_uninterrupted(
    tmp_path, monkeypatch
):
    stream = _partial_stream(seed=241)
    config = _scaffold_config()
    baseline = _coordinator(
        stream,
        strategy="scaffold",
        method_config=config,
        model_seed=251,
    )
    baseline_result = baseline.run()

    interrupted = _coordinator(
        stream,
        strategy="scaffold",
        method_config=config,
        model_seed=251,
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
    )
    interrupted_runtime = interrupted._stateful_runtime
    assert interrupted_runtime is not None
    original_write = interrupted_runtime._write_checkpoint

    class SimulatedInterruption(RuntimeError):
        pass

    def write_then_interrupt(checkpoint_id: str) -> None:
        original_write(checkpoint_id)
        if checkpoint_id == "round-00000001":
            raise SimulatedInterruption

    monkeypatch.setattr(
        interrupted_runtime, "_write_checkpoint", write_then_interrupt
    )
    with pytest.raises(SimulatedInterruption):
        interrupted.run()

    resumed = _coordinator(
        stream,
        strategy="scaffold",
        method_config=config,
        model_seed=251,
        resume_from=tmp_path / "round-00000001.manifest.json",
    )
    resumed_result = resumed.run()

    assert torch.allclose(
        baseline_result["client_stage_task_matrix"],
        resumed_result["client_stage_task_matrix"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert baseline_result["summary"] == resumed_result["summary"]
    assert [_without_timing(record) for record in baseline_result["rounds"]] == [
        _without_timing(record) for record in resumed_result["rounds"]
    ]
    assert [_without_timing(record) for record in baseline_result["stages"]] == [
        _without_timing(record) for record in resumed_result["stages"]
    ]
    assert_tensor_map_equal(baseline.global_state, resumed.global_state)
    for client_id in baseline.clients:
        assert_tensor_map_equal(
            baseline.clients[client_id].model.state_dict(),
            resumed.clients[client_id].model.state_dict(),
        )
    assert runtime_module._safe_equal(
        baseline._stateful_runtime.strategy.state_dict(),
        resumed._stateful_runtime.strategy.state_dict(),
    )
    for counter in (
        "initialization_model_downlink_bytes",
        "initialization_auxiliary_downlink_bytes",
        "training_model_uplink_bytes",
        "training_model_downlink_bytes",
        "training_auxiliary_uplink_bytes",
        "training_auxiliary_downlink_bytes",
        "communication_payload_bytes",
        "training_wire_bytes",
        "evaluation_sync_bytes",
    ):
        assert (
            baseline_result["resource_ledger"][counter]
            == resumed_result["resource_ledger"][counter]
        )


def test_scaffold_controls_ignore_heldout_and_future_label_perturbations():
    original_stream = _partial_stream(seed=263)
    perturbed_stream = copy.deepcopy(original_stream)
    num_classes = perturbed_stream.scenario.num_classes

    heldout_nodes = []
    for tasks in perturbed_stream.evaluation_shards.values():
        for shard in tasks.values():
            if shard.validation_queries.ndim == 1:
                heldout_nodes.append(shard.validation_queries)
            if shard.test_queries.ndim == 1:
                heldout_nodes.append(shard.test_queries)
    heldout = torch.unique(torch.cat(heldout_nodes))
    perturbed_stream.scenario.labels[heldout] = (
        perturbed_stream.scenario.labels[heldout] + 1
    ) % num_classes

    for client_id in perturbed_stream.shards:
        future_task = perturbed_stream.orders.global_task(client_id, 1)
        future_labels = perturbed_stream.shards[client_id][future_task].train_labels
        future_labels.copy_((future_labels + 1) % num_classes)

    original = _coordinator(
        original_stream,
        strategy="scaffold",
        method_config=_scaffold_config(),
        model_seed=269,
    )
    perturbed = _coordinator(
        perturbed_stream,
        strategy="scaffold",
        method_config=_scaffold_config(),
        model_seed=269,
    )
    original_runtime = original._stateful_runtime
    perturbed_runtime = perturbed._stateful_runtime
    assert original_runtime is not None and perturbed_runtime is not None
    original_runtime._initial_broadcast()
    perturbed_runtime._initial_broadcast()

    original_record = original_runtime._run_round(0, 0)
    perturbed_record = perturbed_runtime._run_round(0, 0)

    assert _without_timing(original_record) == _without_timing(perturbed_record)
    assert_tensor_map_equal(original.global_state, perturbed.global_state)
    assert runtime_module._safe_equal(
        original_runtime.strategy.state_dict(),
        perturbed_runtime.strategy.state_dict(),
    )
