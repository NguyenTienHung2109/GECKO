from __future__ import annotations

import hashlib

from typing import Any

import pytest
import torch

from gecko.engine import runtime as runtime_module
from gecko.engine.checkpoint import CheckpointIdentityError
from gecko.engine.checkpoint import CheckpointIntegrityError
from gecko.engine.checkpoint import load_checkpoint
from gecko.engine.checkpoint import save_checkpoint
from gecko.evaluation.evaluator import FederatedEvaluator
from gecko.engine.coordinator import FederatedCoordinator
from gecko.engine.runtime import StatefulFederatedRuntime

from tests.helpers import make_stream


def _method_config(strategy: str = "fedavg") -> dict[str, Any]:
    return {
        "schema": "uefa-method-config",
        "version": 2,
        "name": "legacy_adapter_v1",
        "strategy": {"name": strategy, "parameters": {}},
        "continual_method": {"name": "Bare", "parameters": {}},
    }


def _coordinator(
    stream,
    *,
    model_seed: int = 31,
    method_config: dict[str, Any] | None = None,
) -> FederatedCoordinator:
    return FederatedCoordinator(
        stream,
        "fedavg",
        "Bare",
        model_name="uefa_gcn",
        model_seed=model_seed,
        method_config=method_config,
    )


def _assert_tensor_state_equal(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> None:
    assert first.keys() == second.keys()
    for key in first:
        assert torch.equal(first[key], second[key]), key


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


def test_stateful_runtime_matches_legacy_science_weights_and_model_wire():
    stream = make_stream("NC", "class", 2, order_profile="synchronized", rounds=2)
    legacy = _coordinator(stream)
    legacy_result = legacy.run()
    stateful = _coordinator(stream, method_config=_method_config())
    stateful_result = stateful.run()

    assert stateful_result["result_schema"] == "uefa-run-result-v2"
    assert stateful_result["method_config_resolution"]["name"] == "legacy_adapter_v1"
    assert stateful_result["method_config_benchmark_eligible"] is True
    assert torch.allclose(
        stateful_result["A[k,s,t]"],
        stateful_result["client_stage_task_matrix"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert torch.allclose(
        legacy_result["client_stage_task_matrix"],
        stateful_result["client_stage_task_matrix"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert stateful_result["summary"] == legacy_result["summary"]
    assert (
        stateful_result["communication_payload_bytes"]
        == legacy_result["communication_payload_bytes"]
    )
    for expected, actual in zip(legacy_result["rounds"], stateful_result["rounds"]):
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
    _assert_tensor_state_equal(legacy.global_state, stateful.global_state)

    ledger = stateful_result["resource_ledger"]
    assert ledger["communication_payload_bytes"] == (
        ledger["training_model_uplink_bytes"] + ledger["training_model_downlink_bytes"]
    )
    assert ledger["communication_payload_bytes"] == sum(
        record["communication_payload_bytes"] for record in stateful_result["rounds"]
    )
    model_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in stateful.global_state.values()
    )
    num_clients = len(stateful.clients)
    num_stages = stream.scenario.num_tasks
    assert ledger["initialization_model_downlink_bytes"] == num_clients * model_bytes
    assert ledger["evaluation_sync_bytes"] == (num_clients * num_stages * model_bytes)
    assert stateful_result["training_budget"]["optimizer_state_checkpointed"] is False


def test_round_checkpoint_resume_restores_full_runtime_state_and_rng(
    tmp_path, monkeypatch
):
    stream = make_stream("NC", "class", 2, order_profile="synchronized", rounds=2)
    baseline = _coordinator(stream, model_seed=37)
    baseline_result = StatefulFederatedRuntime(
        baseline, method_config=_method_config()
    ).run()

    interrupted = _coordinator(stream, model_seed=37)
    interrupted_runtime = StatefulFederatedRuntime(
        interrupted,
        method_config=_method_config(),
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
    )
    original_write = interrupted_runtime._write_checkpoint

    class SimulatedInterruption(RuntimeError):
        pass

    def write_then_interrupt(checkpoint_id: str) -> None:
        original_write(checkpoint_id)
        if checkpoint_id == "round-00000001":
            raise SimulatedInterruption

    monkeypatch.setattr(interrupted_runtime, "_write_checkpoint", write_then_interrupt)
    with pytest.raises(SimulatedInterruption):
        interrupted_runtime.run()

    manifest = tmp_path / "round-00000001.manifest.json"
    loaded = load_checkpoint(
        manifest,
        expected_identity=interrupted_runtime.identity,
    )
    assert loaded.cursor == {
        "boundary": "round",
        "next_stage": 0,
        "next_round": 1,
        "stage_participants": [0, 1],
        "global_step": 1,
        "stage_elapsed_seconds": loaded.cursor["stage_elapsed_seconds"],
    }
    assert loaded.state["optimizer_state_saved"] is False
    assert set(loaded.state["clients"]) == set(interrupted.clients)
    for client_state in loaded.state["clients"].values():
        assert {
            "full_model_state",
            "dynamic_model_state",
            "seen_tasks",
            "seen_class_mask",
            "continual_state",
            "method_state",
            "local_model_state",
            "strategy_state",
            "personalized_model_state",
            "topology_overlays",
        } == set(client_state)
    assert "rng" in loaded.state
    assert "stream_content_guard" in loaded.state
    assert "resource_ledger" in loaded.state
    assert "round_records" in loaded.state
    assert "stage_records" in loaded.state

    with pytest.raises(CheckpointIdentityError, match="run_config_digest"):
        StatefulFederatedRuntime(
            _coordinator(stream, model_seed=37),
            method_config=_method_config(),
            resume_from=manifest,
            evaluation_model="shared",
        )

    resumed = _coordinator(stream, model_seed=37)
    resumed_result = StatefulFederatedRuntime(
        resumed,
        method_config=_method_config(),
        resume_from=manifest,
    ).run()

    assert resumed_result["resume_validation"] == {
        "identity_matched": True,
        "diagnostic_override_used": False,
        "benchmark_eligible": True,
        "mismatch_fields": {},
        "diagnostic_override_reason": None,
    }
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
    _assert_tensor_state_equal(baseline.global_state, resumed.global_state)
    for client_id in baseline.clients:
        _assert_tensor_state_equal(
            baseline.clients[client_id].model.state_dict(),
            resumed.clients[client_id].model.state_dict(),
        )
        assert (
            baseline.clients[client_id].state.seen_tasks
            == resumed.clients[client_id].state.seen_tasks
        )
        assert (
            baseline.clients[client_id].algorithm.state
            == resumed.clients[client_id].algorithm.state
        )
    assert (
        resumed_result["checkpoint_records"]
        == interrupted_runtime.checkpoint_records
    )
    assert resumed_result["checkpoint_io_records"] == [
        {
            **interrupted_runtime.checkpoint_records[0],
            "event": "load",
        }
    ]
    assert resumed_result["resource_ledger"]["server_checkpoint_bytes"] == sum(
        record["total_bytes"] for record in resumed_result["checkpoint_records"]
    )
    for counter in (
        "initialization_model_downlink_bytes",
        "training_model_uplink_bytes",
        "training_model_downlink_bytes",
        "training_auxiliary_uplink_bytes",
        "training_auxiliary_downlink_bytes",
        "evaluation_sync_bytes",
        "communication_payload_bytes",
        "training_wire_bytes",
    ):
        assert (
            baseline_result["resource_ledger"][counter]
            == resumed_result["resource_ledger"][counter]
        )


def test_diagnostic_resume_override_is_explicitly_benchmark_ineligible(
    tmp_path, monkeypatch
):
    stream = make_stream("NC", "task", 2, order_profile="synchronized")
    interrupted = _coordinator(stream, model_seed=41)
    runtime = StatefulFederatedRuntime(
        interrupted,
        method_config=_method_config(),
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
    )
    original_write = runtime._write_checkpoint

    class StopAfterCheckpoint(RuntimeError):
        pass

    def stop(checkpoint_id: str) -> None:
        original_write(checkpoint_id)
        if checkpoint_id == "round-00000001":
            raise StopAfterCheckpoint

    monkeypatch.setattr(runtime, "_write_checkpoint", stop)
    with pytest.raises(StopAfterCheckpoint):
        runtime.run()

    resumed = _coordinator(stream, model_seed=41)
    result = StatefulFederatedRuntime(
        resumed,
        method_config=_method_config(),
        resume_from=tmp_path / "round-00000001.manifest.json",
        evaluation_model="shared",
        diagnostic_resume_override="diagnose evaluation selection sensitivity",
    ).run()

    validation = result["resume_validation"]
    assert validation["identity_matched"] is False
    assert validation["diagnostic_override_used"] is True
    assert validation["benchmark_eligible"] is False
    assert "run_config_digest" in validation["mismatch_fields"]
    assert result["benchmark_eligible"] is False


def test_evaluation_model_selection_is_observational_for_future_local_training():
    stream = make_stream("NC", "class", 2, order_profile="synchronized", rounds=2)
    post_local = FederatedCoordinator(
        stream,
        "local_only",
        "Bare",
        model_name="uefa_gcn",
        model_seed=47,
    )
    post_local_result = StatefulFederatedRuntime(
        post_local,
        method_config=_method_config("local_only"),
        evaluation_model="post_local",
    ).run()

    shared_diagnostic = FederatedCoordinator(
        stream,
        "local_only",
        "Bare",
        model_name="uefa_gcn",
        model_seed=47,
    )
    shared_result = StatefulFederatedRuntime(
        shared_diagnostic,
        method_config=_method_config("local_only"),
        evaluation_model="shared",
    ).run()

    assert [
        _without_timing(record) for record in post_local_result["rounds"]
    ] == [_without_timing(record) for record in shared_result["rounds"]]
    for client_id in post_local.clients:
        _assert_tensor_state_equal(
            post_local.clients[client_id].model.state_dict(),
            shared_diagnostic.clients[client_id].model.state_dict(),
        )


def test_resume_identity_ignores_informational_source_tree_digest(tmp_path, monkeypatch):
    stream = make_stream("NC", "class", 2, order_profile="synchronized")
    monkeypatch.setattr(
        runtime_module,
        "_resolve_tracked_source_tree",
        lambda: ("1" * 64, True, ()),
    )
    original = StatefulFederatedRuntime(
        _coordinator(stream, model_seed=53),
        method_config=_method_config(),
        checkpoint_dir=tmp_path,
    )
    original._write_checkpoint("source-tree")

    monkeypatch.setattr(
        runtime_module,
        "_resolve_tracked_source_tree",
        lambda: ("2" * 64, False, ()),
    )
    resumed = StatefulFederatedRuntime(
        _coordinator(stream, model_seed=53),
        method_config=_method_config(),
        resume_from=tmp_path / "source-tree.manifest.json",
    )
    assert resumed.resume_validation.identity_matched
    assert not resumed.resume_validation.diagnostic_override_used


def test_resume_identity_recomputes_complete_stream_content(tmp_path):
    stream = make_stream("NC", "class", 2, order_profile="synchronized")
    reported_fingerprint = "9" * 64
    original = StatefulFederatedRuntime(
        _coordinator(stream, model_seed=59),
        method_config=_method_config(),
        checkpoint_dir=tmp_path,
        stream_scientific_fingerprint=reported_fingerprint,
    )
    original._write_checkpoint("stream-content")

    query_ids = stream.evaluation_shards[0][0].test_query_ids
    assert query_ids.numel() > 0
    query_ids[0].add_(1)
    with pytest.raises(CheckpointIdentityError, match="stream_content_digest"):
        StatefulFederatedRuntime(
            _coordinator(stream, model_seed=59),
            method_config=_method_config(),
            resume_from=tmp_path / "stream-content.manifest.json",
            stream_scientific_fingerprint=reported_fingerprint,
        )


def test_resume_rejects_participant_union_and_query_count_inconsistency(
    tmp_path, monkeypatch
):
    stream = make_stream("NC", "class", 2, order_profile="synchronized", rounds=2)
    runtime = StatefulFederatedRuntime(
        _coordinator(stream, model_seed=61),
        method_config=_method_config(),
        checkpoint_dir=tmp_path,
        checkpoint_every=1,
    )
    original_write = runtime._write_checkpoint

    class StopAfterFirstRound(RuntimeError):
        pass

    def stop(checkpoint_id: str) -> None:
        original_write(checkpoint_id)
        if checkpoint_id == "round-00000001":
            raise StopAfterFirstRound

    monkeypatch.setattr(runtime, "_write_checkpoint", stop)
    with pytest.raises(StopAfterFirstRound):
        runtime.run()
    loaded = load_checkpoint(
        tmp_path / "round-00000001.manifest.json",
        expected_identity=runtime.identity,
    )

    invalid_cursor = dict(loaded.cursor)
    invalid_cursor["stage_participants"] = []
    participant_checkpoint = save_checkpoint(
        tmp_path,
        "invalid-participant-union",
        identity=runtime.identity,
        cursor=invalid_cursor,
        state=loaded.state,
    )
    with pytest.raises(CheckpointIntegrityError, match="participant union"):
        StatefulFederatedRuntime(
            _coordinator(stream, model_seed=61),
            method_config=_method_config(),
            resume_from=participant_checkpoint.manifest_path,
        )

    invalid_state = dict(loaded.state)
    invalid_state["query_counts"] = torch.zeros(1)
    query_checkpoint = save_checkpoint(
        tmp_path,
        "invalid-query-counts",
        identity=runtime.identity,
        cursor=loaded.cursor,
        state=invalid_state,
    )
    with pytest.raises(CheckpointIntegrityError, match="query-count tensor"):
        StatefulFederatedRuntime(
            _coordinator(stream, model_seed=61),
            method_config=_method_config(),
            resume_from=query_checkpoint.manifest_path,
        )


def test_evaluation_failure_restores_full_client_state_in_finally(monkeypatch):
    stream = make_stream("NC", "class", 2, order_profile="synchronized")
    coordinator = FederatedCoordinator(
        stream,
        "local_only",
        "Bare",
        model_name="uefa_gcn",
        model_seed=67,
    )
    runtime = StatefulFederatedRuntime(
        coordinator,
        method_config=_method_config("local_only"),
        evaluation_model="shared",
    )
    client = coordinator.clients[0]
    task_id = int(stream.orders.global_task(0, 0))
    client.mark_seen(task_id, stream.shards[0][task_id].task_class_mask)
    with torch.no_grad():
        for parameter in client.model.parameters():
            parameter.add_(0.25)
    client.model.train()
    client.algorithm.state["evaluation_failure_sentinel"] = torch.tensor([3.0])
    expected_model = {
        key: value.detach().clone()
        for key, value in client.model.state_dict().items()
    }
    expected_method = client.algorithm.save_method_state()
    expected_seen = set(client.state.seen_tasks)
    expected_mask = client.state.seen_class_mask.detach().clone()

    evaluator = FederatedEvaluator(stream)

    class InjectedEvaluationFailure(RuntimeError):
        pass

    def fail(*args, **kwargs):
        raise InjectedEvaluationFailure

    monkeypatch.setattr(evaluator, "evaluate", fail)
    with pytest.raises(InjectedEvaluationFailure):
        runtime._evaluate_stage(evaluator, 0)

    _assert_tensor_state_equal(expected_model, client.model.state_dict())
    assert torch.equal(
        expected_method["evaluation_failure_sentinel"],
        client.algorithm.state["evaluation_failure_sentinel"],
    )
    assert client.state.seen_tasks == expected_seen
    assert torch.equal(client.state.seen_class_mask, expected_mask)
    assert client.model.training is True


def test_untracked_runtime_source_content_and_mode_are_bound(tmp_path):
    source = tmp_path / "src" / "gecko" / "new_runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")

    def identity() -> tuple[str, tuple[str, ...]]:
        digest = hashlib.sha256(b"test-source-base")
        paths = runtime_module._update_untracked_runtime_source_digest(
            digest,
            tmp_path,
            [b"src/gecko/new_runtime.py"],
        )
        return digest.hexdigest(), paths

    first_digest, paths = identity()
    assert paths == ("src/gecko/new_runtime.py",)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second_digest, _ = identity()
    assert second_digest != first_digest
    source.chmod(0o755)
    executable_digest, _ = identity()
    assert executable_digest != second_digest

    with pytest.raises(ValueError, match="Unsafe untracked runtime source"):
        runtime_module._update_untracked_runtime_source_digest(
            hashlib.sha256(),
            tmp_path,
            [b"outside/new_runtime.py"],
        )
