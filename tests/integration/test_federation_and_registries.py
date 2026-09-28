from __future__ import annotations

from gecko.compat.paths import resolve_source_path

import ast
import builtins
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from gecko.engine import FederatedCoordinator
from gecko.engine import fedprox_penalty
from gecko.engine import weighted_average
from gecko.engine.client import supervised_loss
from gecko.engine import client as client_module
from gecko.evaluation.evaluator import FederatedEvaluator
from gecko.algorithms.catalog import MethodRegistry
from gecko.algorithms.catalog import ORIGINAL_ALGORITHMS
from gecko.models.registry import ModelRegistry
from gecko.types import LocalUpdateResult

from tests.helpers import make_stream


def _update(client_id: int, value: float, weight: int) -> LocalUpdateResult:
    state = {"weight": torch.tensor([value, value + 1], dtype=torch.float32)}
    return LocalUpdateResult(client_id, 0, state, weight, 0.0, 8, ("weight",))


def test_32_fedavg_weighted_average_is_numerically_correct():
    averaged = weighted_average([_update(0, 1.0, 1), _update(1, 4.0, 3)])
    assert torch.allclose(averaged["weight"], torch.tensor([3.25, 4.25]))


def test_lp_positive_anchor_weight_excludes_sampled_negatives():
    stream = make_stream("LP", "domain", 2, order_profile="synchronized")
    positive_counts = {
        client_id: stream.shards[client_id][0].positive_anchor_count
        for client_id in stream.shards
    }
    assert all(
        positive_counts[client_id]
        < stream.shards[client_id][0].supervised_query_count
        for client_id in positive_counts
    )
    configured = replace(
        stream,
        config=replace(
            stream.config,
            training=replace(
                stream.config.training,
                aggregation_weight="current_positive_anchor_count",
            ),
        ),
    )
    result = FederatedCoordinator(
        configured, strategy_name="fedavg", model_name="uefa_gcn"
    ).run()
    assert result["rounds"][0]["raw_aggregation_weights"] == positive_counts


def test_33_fedprox_penalty_is_numerically_correct():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]]))
    server = {"weight": torch.tensor([[1.0, 1.0]])}
    penalty = fedprox_penalty(model, server, ("weight",), mu=0.2)
    assert torch.allclose(penalty, torch.tensor(0.5))


def test_fedprox_mu_zero_uses_the_exact_fedavg_loss_graph(monkeypatch):
    stream = make_stream("NC", "task", 2)
    coordinator = FederatedCoordinator(
        stream, "fedprox", "Bare", model_name="uefa_gcn"
    )
    client = coordinator.clients[0]
    client.config = replace(
        client.config,
        training=replace(client.config.training, fedprox_mu=0.0),
    )
    monkeypatch.setattr(
        client_module,
        "fedprox_penalty",
        lambda *args, **kwargs: pytest.fail("mu=0 must not build a proximal graph"),
    )
    task = stream.orders.global_task(client.client_id, 0)

    update = client.update(
        stream.shards[client.client_id][task],
        global_task_id=task,
        server_state=coordinator.global_state,
        strategy="fedprox",
    )

    assert update.weight > 0
    assert torch.isfinite(torch.tensor(update.training_loss))


def test_34_client_local_algorithm_state_persists_across_rounds_and_stages():
    stream = make_stream("NC", "task", 2, rounds=2)
    coordinator = FederatedCoordinator(stream, "fedavg", "EWC")
    coordinator.run()
    for client in coordinator.clients.values():
        anchors = client.algorithm.state["anchors"]
        assert set(anchors) == set(stream.orders.client_orders[client.client_id])
        assert client.state.continual_state is client.algorithm.state


def test_faithful_lwf_consolidates_once_per_task_not_once_per_round():
    stream = make_stream("NC", "class", 2, rounds=3)
    coordinator = FederatedCoordinator(
        stream, "local_only", "LwF", model_name="uefa_gcn"
    )
    coordinator.run()
    for client in coordinator.clients.values():
        assert client.algorithm.state["consolidated_task_ids"] == tuple(
            stream.orders.client_orders[client.client_id]
        )


def test_server_broadcast_preserves_faithful_method_local_state():
    stream = make_stream("NC", "class", 2, order_profile="synchronized")
    coordinator = FederatedCoordinator(
        stream, "fedavg", "EWC", model_name="uefa_gcn"
    )
    coordinator.run()
    for client in coordinator.clients.values():
        state = client.algorithm.state
        before_anchors = state["anchors"]
        before_fishers = state["fishers"]
        client.load_shared_state(coordinator.global_state)
        assert client.algorithm.state["anchors"] is before_anchors
        assert client.algorithm.state["fishers"] is before_fishers


def test_ergnn_replay_memory_survives_server_broadcast_and_is_owned_local_nodes():
    stream = make_stream("NC", "class", 2, order_profile="synchronized")
    coordinator = FederatedCoordinator(
        stream, "fedavg", "ERGNN", model_name="uefa_gcn"
    )
    coordinator.run()
    for client in coordinator.clients.values():
        memory = client.algorithm.state["buffered_nodes"]
        before = memory.clone()
        assert memory.numel() > 0
        assert int(memory.min()) >= 0
        assert int(memory.max()) < client.graph.node_features.shape[0]
        client.load_shared_state(coordinator.global_state)
        assert torch.equal(client.algorithm.state["buffered_nodes"], before)


@pytest.mark.parametrize(
    ("algorithm", "state_key"),
    (("LwF", "teacher"), ("MAS", "importances")),
)
def test_lwf_and_mas_private_state_survives_server_broadcast(algorithm, state_key):
    stream = make_stream("NC", "class", 2, order_profile="synchronized")
    coordinator = FederatedCoordinator(
        stream, "fedavg", algorithm, model_name="uefa_gcn"
    )
    result = coordinator.run()
    for client in coordinator.clients.values():
        before = client.algorithm.state[state_key]
        client.load_shared_state(coordinator.global_state)
        assert client.algorithm.state[state_key] is before
        assert state_key in result["client_algorithm_local_state_keys"][client.client_id]


def test_ergnn_registry_fails_closed_for_multilabel_nc_and_non_nc_problems():
    registry = MethodRegistry()
    for problem, incremental in (
        ("NC", "domain"),
        ("LC", "class"),
        ("LC", "domain"),
        ("LP", "domain"),
    ):
        with pytest.raises(ValueError, match="scalar node classes"):
            registry.validate("ERGNN", "fedavg", problem, incremental)


@pytest.mark.parametrize("algorithm", ["LwF", "EWC", "MAS", "ERGNN"])
@pytest.mark.parametrize("strategy", ["local_only", "fedavg", "fedprox"])
def test_g12_verified_nc_class_methods_have_exact_supported_status(
    algorithm, strategy
):
    entry = MethodRegistry().validate(algorithm, strategy, "NC", "class")
    assert entry.support_status == "faithful_supported"
    assert entry.scientific_fidelity == "verified"
    assert entry.benchmark_eligible is True
    assert entry.test_coverage == "g12_v1_mechanism_and_real_nc_class_matrix"


def test_round_records_prove_participation_tasks_weights_and_updates():
    stream = make_stream(
        "NC", "task", 2, order_profile="synchronized", rounds=2
    )
    result = FederatedCoordinator(
        stream, "fedavg", "Bare", model_name="uefa_gcn"
    ).run()
    assert result["execution_device"] == "cpu"
    for record in result["rounds"]:
        stage = record["stage"]
        round_id = record["round"]
        expected = list(stream.participation.trace[stage][round_id])
        assert record["participant_ids"] == expected
        assert record["participants"] == len(expected)
        assert set(record["participant_global_task_ids"].values()) == {stage}
        assert record["empty_updates"] == 0
        assert record["normalized_aggregation_weight_sum"] == pytest.approx(1.0)
        assert set(record["raw_aggregation_weights"]) == set(expected)
        assert all(value > 0 for value in record["raw_aggregation_weights"].values())
        assert all(value > 0 for value in record["client_parameter_delta_l2"].values())
        assert record["server_parameter_delta_l2"] > 0


def test_fedavg_evaluation_uses_client_models_with_local_buffers(monkeypatch):
    stream = make_stream("NC", "task", 2, order_profile="synchronized")
    coordinator = FederatedCoordinator(
        stream, "fedavg", "Bare", model_name="uefa_gcn"
    )
    original = FederatedEvaluator.evaluate

    def spy(evaluator, model, client_id, *args, **kwargs):
        assert model is coordinator.clients[client_id].model
        return original(evaluator, model, client_id, *args, **kwargs)

    monkeypatch.setattr(FederatedEvaluator, "evaluate", spy)
    coordinator.run()


@pytest.mark.parametrize(
    "algorithm,state_key",
    [("generic_replay", "memory"), ("generic_weight_isolation", "task_masks")],
)
def test_35_server_broadcast_does_not_overwrite_replay_or_local_masks(algorithm, state_key):
    stream = make_stream("NC", "task", 2)
    coordinator = FederatedCoordinator(
        stream,
        "local_only",
        algorithm,
        allow_experimental_placeholder=True,
    )
    client = coordinator.clients[0]
    client.algorithm.state[state_key] = {"sentinel": torch.tensor([7])}
    before = client.algorithm.state[state_key]
    client.load_shared_state(coordinator.global_state)
    assert client.algorithm.state[state_key] is before
    assert int(before["sentinel"][0]) == 7


def test_36_every_original_begin_algorithm_is_registered():
    registry = MethodRegistry()
    assert set(ORIGINAL_ALGORITHMS) == set(registry.original_names())
    assert set(registry.names()) == {"Bare", "LwF", "EWC", "MAS", "ERGNN"}
    begin_root = resolve_source_path(Path(__file__).resolve().parents[2] / "begin")
    algorithm_root = begin_root / "algorithms"
    if algorithm_root.exists():
        aliases = {
            "bare": "Bare",
            "cat": "CaT",
            "cgnn": "CGNN",
            "ergnn": "ERGNN",
            "ewc": "EWC",
            "gem": "GEM",
            "hat": "HAT",
            "lwf": "LwF",
            "mas": "MAS",
            "packnet": "PackNet",
            "piggyback": "Piggyback",
            "pignn": "PIGNN",
            "twp": "TWP",
        }
        discovered = {
            aliases[path.name]
            for path in algorithm_root.iterdir()
            if path.is_dir() and path.name in aliases
        }
        assert discovered <= set(registry.original_names())


SUPPORTED_COMBINATIONS = tuple(
    entry
    for entry in MethodRegistry().compatibility()
    if entry.runnable
    and entry.support_status
    in {"native", "implemented_unverified", "faithful_supported"}
)


@pytest.mark.parametrize(
    "entry",
    SUPPORTED_COMBINATIONS,
    ids=lambda entry: (
        f"{entry.algorithm}-{entry.server_strategy}-"
        f"{entry.problem_type}-{entry.incremental_setting}"
    ),
)
def test_37_every_supported_algorithm_combination_has_synthetic_smoke(entry):
    num_tasks = 4 if (entry.problem_type, entry.incremental_setting) == ("LC", "domain") else 2
    stream = make_stream(
        entry.problem_type,
        entry.incremental_setting,
        num_tasks,
        order_profile="synchronized",
    )
    result = FederatedCoordinator(
        stream,
        entry.server_strategy,
        entry.algorithm,
    ).run()
    assert result["client_stage_task_matrix"].shape == (
        stream.config.partition.num_clients,
        stream.scenario.num_tasks,
        stream.scenario.num_tasks,
    )


def _model_like_classes(source: Path) -> set[str]:
    module = ".".join(source.with_suffix("").parts[-3:])
    tree = ast.parse(source.read_text(encoding="utf-8"))
    output = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {ast.unparse(base) for base in node.bases}
        if any("Module" in base for base in bases) or node.name.lower().startswith(
            ("gcn", "fullgcn", "adaptive", "progressive")
        ):
            output.add(f"original:{module}.{node.name}")
    return output


def test_38_every_original_graph_network_class_is_in_model_registry():
    registry = ModelRegistry()
    names = set(registry.all_names())
    begin_root = resolve_source_path(Path(__file__).resolve().parents[2] / "begin")
    sources = list((begin_root / "utils").glob("models*.py"))
    sources.append(begin_root / "utils" / "pretraining.py")
    sources.extend((begin_root / "algorithms").rglob("*.py"))
    expected = set()
    for source in sources:
        if source.exists() and source.name != "__init__.py":
            relative = source.relative_to(begin_root.parent)
            module = ".".join(relative.with_suffix("").parts)
            if module == "gecko.algorithms.federated.fedfst" or module.startswith(
                "gecko.algorithms.federated.fedfst."
            ):
                continue
            tree = ast.parse(source.read_text(encoding="utf-8"))
            classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
            model_classes = {
                node.name
                for node in classes
                if any("Module" in ast.unparse(base) for base in node.bases)
                or node.name.lower().startswith(("gcn", "fullgcn", "adaptive", "progressive"))
            }
            changed = True
            while changed:
                changed = False
                for node in classes:
                    bases = {ast.unparse(base).rsplit(".", 1)[-1] for base in node.bases}
                    if node.name not in model_classes and bases.intersection(model_classes):
                        model_classes.add(node.name)
                        changed = True
            expected.update(
                f"original:{module}.{name}" for name in model_classes
            )
    assert expected <= names
    assert not any(
        name.startswith("original:begin.algorithms.fedfst.") for name in names
    )
    assert "uefa_gcn" in names
    assert "begin_gcn" in names
    assert registry.names() == ("begin_gcn", "fedfst_gat")
    assert set(registry.discovered_names()) == {
        name for name in names if name.startswith("original:")
    }


@pytest.mark.parametrize(
    "algorithm",
    ["CGNN", "PackNet", "Piggyback", "HAT", "PIGNN"],
)
def test_original_names_without_faithful_adapters_are_disabled(algorithm):
    registry = MethodRegistry()
    with pytest.raises(ValueError, match="no faithful UEFA adapter"):
        registry.create(algorithm)
    entries = [
        entry for entry in registry.compatibility() if entry.algorithm == algorithm
    ]
    assert entries
    assert all(not entry.runnable for entry in entries)
    assert all(not entry.benchmark_eligible for entry in entries)
    assert all(entry.scientific_fidelity == "false" for entry in entries)


@pytest.mark.parametrize(
    ("algorithm", "family"),
    [
        ("GEM", "gem_uefa_v1"),
        ("TWP", "twp_uefa_v1"),
        ("CaT", "cat_uefa_v1"),
    ],
)
def test_new_continual_adapters_require_explicit_v2_config(algorithm, family):
    registry = MethodRegistry()
    with pytest.raises(ValueError, match="validated explicit v2"):
        registry.create(algorithm)
    assert algorithm in registry.cli_names()
    assert family in {"gem_uefa_v1", "twp_uefa_v1", "cat_uefa_v1"}


def test_generic_mechanisms_require_explicit_opt_in():
    registry = MethodRegistry()
    with pytest.raises(ValueError, match="explicit opt-in"):
        registry.create("generic_replay")
    assert registry.create(
        "generic_replay", allow_experimental_placeholder=True
    ).name == "generic_replay"


def test_joint_oracle_gradient_is_exact_query_count_weighted_shard_gradient():
    torch.manual_seed(7)
    stream = make_stream("NC", "task", 2, order_profile="synchronized")
    coordinator = FederatedCoordinator(
        stream, "joint_oracle", "Bare", model_name="uefa_gcn"
    )
    parameters = tuple(coordinator.global_model.parameters())
    objective = coordinator.joint_oracle_objective(stage=0)
    actual_raw = torch.autograd.grad(objective, parameters, allow_unused=True)
    actual = tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, actual_raw)
    )

    total = sum(
        stream.shards[client_id][stream.orders.global_task(client_id, 0)].supervised_query_count
        for client_id in coordinator.clients
    )
    expected = [torch.zeros_like(parameter) for parameter in parameters]
    for client_id, client in coordinator.clients.items():
        task_id = stream.orders.global_task(client_id, 0)
        shard = stream.shards[client_id][task_id]
        graph = stream.partition.client_graphs[client_id]
        logits = coordinator.global_model.forward_queries(
            graph.node_features,
            graph.edge_index,
            shard.train_queries,
            stream.scenario.problem_type,
        )
        mask = client._class_mask(shard)
        if mask is not None:
            logits = logits.clone()
            logits[..., ~mask] = -1e12
        loss = supervised_loss(logits, shard.train_labels)
        shard_gradients_raw = torch.autograd.grad(
            loss, parameters, allow_unused=True
        )
        shard_gradients = tuple(
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, shard_gradients_raw)
        )
        weight = shard.supervised_query_count / total
        for index, gradient in enumerate(shard_gradients):
            expected[index] += gradient * weight

    for actual_gradient, expected_gradient in zip(actual, expected):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1e-7, rtol=1e-6)


def test_39_wandb_disabled_mode_never_imports_wandb(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "wandb":
            raise AssertionError("disabled mode attempted to import wandb")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    from gecko.tracking import WandbLogger

    logger = WandbLogger(mode="disabled", project="UEFA", directory=tmp_path)
    logger.log({"metric": 1.0})
    logger.finish()
    assert logger.run is None
    assert logger.actual_mode == "disabled"


def test_40_wandb_offline_smoke_run(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_SILENT", "true")
    from gecko.tracking import WandbLogger

    logger = WandbLogger(
        mode="offline",
        project="UEFA",
        config={"raw_graph_uploaded": False},
        run_name="uefa-pytest-offline",
        directory=tmp_path,
    )
    assert logger.run is not None
    assert logger.actual_mode == "offline"
    logger.log({"stage": 0, "accuracy": 0.5}, step=0)
    logger.finish()
