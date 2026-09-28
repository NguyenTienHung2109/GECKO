from __future__ import annotations

from gecko.compat.paths import resolve_source_path

import copy
from pathlib import Path
from typing import Any

import pytest
import torch

from gecko.cli import main as cli_module
from gecko.cli.main import build_parser
from gecko.engine import runtime as runtime_module
from gecko.engine.coordinator import FederatedCoordinator
from gecko.algorithms.method_config import MethodConfigValidationError
from gecko.algorithms.method_config import UnsupportedMethodConfigError
from gecko.algorithms.method_config import resolve_method_config
from gecko.algorithms.method_config import validate_method_config
from gecko.engine.protocol import RoundContext
from gecko.algorithms.federated.legacy import LegacyStrategyAdapter
from gecko.algorithms.federated.scaffold import ScaffoldStrategy
from gecko.algorithms.catalog import MethodRegistry
from gecko.algorithms.continual.twp import TWPAlgorithm

from tests.helpers import assert_tensor_map_equal
from tests.helpers import make_stream


ROOT = Path(__file__).resolve().parents[2]


def _twp_config(
    strategy: str = "local_only",
    *,
    lambda_l: float = 1.0,
    lambda_t: float = 1.0,
    beta: float = 0.0,
) -> dict[str, Any]:
    return {
        "schema": "uefa-method-config",
        "version": 2,
        "name": "twp_uefa_v1",
        "strategy": {"name": strategy, "parameters": {}},
        "continual_method": {
            "name": "TWP",
            "parameters": {
                "lambda_l": lambda_l,
                "lambda_t": lambda_t,
                "beta": beta,
                "middle_layer_index": None,
                "significant_threshold": 1.0e-12,
            },
        },
    }


def _coordinator(
    stream,
    strategy: str = "local_only",
    *,
    config: dict[str, Any] | None = None,
    model_seed: int = 307,
    **options: Any,
) -> FederatedCoordinator:
    return FederatedCoordinator(
        stream,
        strategy,
        "TWP",
        model_name="uefa_gcn",
        model_seed=model_seed,
        method_config=_twp_config(strategy) if config is None else config,
        **options,
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


def test_twp_cli_requires_validated_config_before_any_stream_access(monkeypatch):
    accessed = []

    def forbidden(*args, **kwargs):
        accessed.append((args, kwargs))
        raise AssertionError("stream access must not occur")

    from gecko.workflows import run as run_workflow
    monkeypatch.setattr(run_workflow, "audit_stream", forbidden)
    monkeypatch.setattr(run_workflow, "load_stream", forbidden)
    args = build_parser().parse_args(
        [
            "run",
            "--config",
            str(resolve_source_path(ROOT / "configs" / "uefa_v1" / "synthetic_smoke.yaml")),
            "--cl-algorithm",
            "TWP",
            "--stream",
            "/definitely/missing/stream.pt",
        ]
    )
    with pytest.raises(ValueError, match="requires an explicit validated"):
        args.handler(args)
    assert accessed == []


def test_twp_registry_is_explicit_v2_only_and_forwards_numeric_parameters():
    registry = MethodRegistry()
    assert "TWP" not in registry.all_runnable_names()
    assert "TWP" in registry.cli_names()
    with pytest.raises(ValueError, match="validated explicit v2"):
        registry.create("TWP")
    method = registry.create(
        "TWP",
        explicit_v2_family="twp_uefa_v1",
        lambda_l=2.0,
        lambda_t=3.0,
        beta=0.1,
        middle_layer_index=0,
        significant_threshold=1.0e-9,
    )
    assert isinstance(method, TWPAlgorithm)
    assert method._private_hyperparameters() == {
        "lambda_l": 2.0,
        "lambda_t": 3.0,
        "beta": 0.1,
        "middle_layer_index": 0,
        "significant_threshold": 1.0e-9,
        "topology_max_edges": None,
    }


@pytest.mark.parametrize(
    ("problem", "incremental", "fidelity"),
    [
        ("NC", "task", "faithful_supported_candidate"),
        ("NC", "class", "mechanism_adaptation"),
        ("NC", "domain", "mechanism_adaptation"),
        ("LC", "task", "mechanism_adaptation"),
        ("LC", "class", "mechanism_adaptation"),
        ("LC", "domain", "mechanism_adaptation"),
    ],
)
def test_twp_nc_lc_scenarios_resolve_runnable_but_ineligible(
    problem: str, incremental: str, fidelity: str
):
    resolved = validate_method_config(
        _twp_config(), problem_type=problem, incremental_setting=incremental
    )
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.scientific_fidelity == fidelity


def test_twp_lp_scenario_fails_closed():
    resolved = resolve_method_config(
        _twp_config(), problem_type="LP", incremental_setting="domain"
    )
    assert not resolved.runnable
    assert resolved.support_status == "unsupported_scientific"
    with pytest.raises(UnsupportedMethodConfigError):
        validate_method_config(
            _twp_config(), problem_type="LP", incremental_setting="domain"
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("lambda_l", -1.0),
        ("lambda_t", float("nan")),
        ("beta", True),
        ("middle_layer_index", -1),
        ("significant_threshold", float("inf")),
        ("topology_max_edges", 0),
    ],
)
def test_twp_numeric_method_config_is_exact_and_finite(name: str, value: object):
    config = _twp_config()
    config["continual_method"]["parameters"][name] = value
    with pytest.raises(MethodConfigValidationError):
        resolve_method_config(config)


def test_twp_memory_bounded_topology_is_explicitly_labeled():
    config = _twp_config("fedavg")
    config["continual_method"]["parameters"]["topology_max_edges"] = 200_000
    resolved = validate_method_config(
        config, problem_type="NC", incremental_setting="domain"
    )
    assert resolved.support_status == "implemented_unverified_memory_bounded_topology"
    assert resolved.scientific_fidelity == "memory_bounded_topology_approximation"
    assert dict(resolved.continual_method_parameters)["topology_max_edges"] == 200_000


@pytest.mark.parametrize(
    ("strategy", "strategy_type"),
    [
        ("local_only", LegacyStrategyAdapter),
        ("fedavg", LegacyStrategyAdapter),
        ("fedprox", LegacyStrategyAdapter),
        ("scaffold", ScaffoldStrategy),
    ],
)
def test_coordinator_forwards_twp_parameters_and_maps_strategy(
    strategy: str, strategy_type: type
):
    stream = make_stream("NC", "class", 2, 311, "mild", "synchronized")
    config = _twp_config(strategy, lambda_l=2.0, lambda_t=3.0, beta=0.1)
    coordinator = _coordinator(stream, strategy, config=config)
    runtime = coordinator._stateful_runtime
    assert runtime is not None
    assert isinstance(runtime.strategy, strategy_type)
    if isinstance(runtime.strategy, LegacyStrategyAdapter):
        assert runtime.strategy.name == strategy
    else:
        assert runtime.strategy.correction_enabled
        assert runtime.strategy.control_updates_enabled
    for client in coordinator.clients.values():
        assert isinstance(client.algorithm, TWPAlgorithm)
        assert client.algorithm.lambda_l == 2.0
        assert client.algorithm.lambda_t == 3.0
        assert client.algorithm.beta == 0.1


@pytest.mark.parametrize("strategy", ["fedavg", "scaffold"])
def test_full_strategy_client_receive_preserves_twp_private_state(strategy: str):
    stream = make_stream("NC", "class", 2, 313, "mild", "synchronized")
    coordinator = _coordinator(stream, strategy)
    runtime = coordinator._stateful_runtime
    assert runtime is not None
    client_id = min(coordinator.clients)
    client = coordinator.clients[client_id]
    task_id = int(stream.orders.global_task(client_id, 0))
    context = client.build_method_context(
        stream.shards[client_id][task_id],
        global_task_id=task_id,
        stage_index=0,
        round_index=0,
    )
    client.algorithm.consolidate(client.model, context)
    private_before = copy.deepcopy(client.algorithm.state)
    checksum_before = client.algorithm.private_state_checksum()
    with torch.no_grad():
        for parameter in client.model.parameters():
            parameter.add_(0.5)
    participants = tuple(sorted(coordinator.clients))
    tasks = {value: int(stream.orders.global_task(value, 0)) for value in participants}
    round_context = RoundContext(0, 0, participants, tasks)
    payload = runtime.strategy.prepare_payload(
        round_context, client_id, "initialization"
    )
    assert payload is not None
    runtime.strategy.client_receive(client, payload, context)

    assert runtime_module._safe_equal(private_before, client.algorithm.state)
    assert client.algorithm.private_state_checksum() == checksum_before
    diagnostics = client.algorithm.diagnostics()
    assert diagnostics["pre_broadcast_state_sha256"] == checksum_before
    assert diagnostics["post_broadcast_state_sha256"] == checksum_before


@pytest.mark.parametrize(
    ("problem", "incremental"),
    [
        ("NC", "task"),
        ("NC", "class"),
        ("NC", "domain"),
        ("LC", "task"),
        ("LC", "class"),
        ("LC", "domain"),
    ],
)
def test_twp_end_to_end_exports_round_and_final_diagnostics(
    problem: str, incremental: str
):
    num_tasks = 4 if (problem, incremental) == ("LC", "domain") else 2
    stream = make_stream(problem, incremental, num_tasks, 0, "mild", "synchronized")
    coordinator = _coordinator(stream, "local_only", model_seed=319)
    result = coordinator.run()

    assert result["result_schema"] == "uefa-run-result-v2"
    assert result["method_support_status"].startswith("implemented_unverified")
    assert not result["method_config_benchmark_eligible"]
    assert all(record["client_method_diagnostics"] for record in result["rounds"])
    for client_id, diagnostics in result["client_method_diagnostics"].items():
        assert diagnostics["method"] == "TWP"
        assert diagnostics["num_consolidated_tasks"] == num_tasks
        assert diagnostics["consolidated_task_ids"] == [
            stream.orders.global_task(int(client_id), stage)
            for stage in range(stream.scenario.num_tasks)
        ]


def test_twp_checkpoint_resume_preserves_private_state_and_diagnostics(
    tmp_path, monkeypatch
):
    stream = make_stream("NC", "class", 2, 331, "mild", "synchronized")
    config = _twp_config("fedavg")
    baseline = _coordinator(stream, "fedavg", config=config, model_seed=337)
    baseline_result = baseline.run()

    interrupted = _coordinator(
        stream,
        "fedavg",
        config=config,
        model_seed=337,
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

    resumed = _coordinator(
        stream,
        "fedavg",
        config=config,
        model_seed=337,
        resume_from=tmp_path / "round-00000001.manifest.json",
    )
    resumed_result = resumed.run()

    assert baseline_result["summary"] == resumed_result["summary"]
    assert [_without_timing(value) for value in baseline_result["rounds"]] == [
        _without_timing(value) for value in resumed_result["rounds"]
    ]
    assert (
        baseline_result["client_method_diagnostics"]
        == resumed_result["client_method_diagnostics"]
    )
    assert_tensor_map_equal(baseline.global_state, resumed.global_state)
    for client_id in baseline.clients:
        assert runtime_module._safe_equal(
            baseline.clients[client_id].algorithm.save_method_state(),
            resumed.clients[client_id].algorithm.save_method_state(),
        )


@pytest.mark.parametrize(
    ("strategy", "expected_runtime"),
    [
        ("fedavg", LegacyStrategyAdapter),
        ("fedprox", LegacyStrategyAdapter),
        ("scaffold", ScaffoldStrategy),
    ],
)
def test_twp_federated_compositions_preserve_private_state_and_split_accounting(
    strategy, expected_runtime
):
    stream = make_stream("NC", "class", 2, 347, "mild", "synchronized")
    coordinator = _coordinator(
        stream,
        strategy,
        config=_twp_config(strategy, lambda_l=1.0, lambda_t=1.0, beta=0.0),
        model_seed=349,
    )
    result = coordinator.run()

    runtime = coordinator._stateful_runtime
    assert runtime is not None
    assert isinstance(runtime.strategy, expected_runtime)
    if strategy in {"fedavg", "fedprox"}:
        assert runtime.strategy.name == strategy
    ledger = result["resource_ledger"]
    assert ledger["training_model_uplink_bytes"] > 0
    assert ledger["training_model_downlink_bytes"] > 0
    assert ledger["evaluation_sync_bytes"] > 0
    if strategy == "scaffold":
        assert ledger["training_auxiliary_uplink_bytes"] > 0
        assert ledger["training_auxiliary_downlink_bytes"] > 0
    else:
        assert ledger["training_auxiliary_uplink_bytes"] == 0
        assert ledger["training_auxiliary_downlink_bytes"] == 0

    for diagnostics in result["client_method_diagnostics"].values():
        assert diagnostics["method"] == "TWP"
        assert diagnostics["num_consolidated_tasks"] == 2
        assert (
            diagnostics["pre_broadcast_state_sha256"]
            == (diagnostics["post_broadcast_state_sha256"])
        )
