from __future__ import annotations

from gecko.compat.paths import resolve_source_path

import copy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import yaml

from gecko.engine.coordinator import FederatedCoordinator
from gecko.algorithms.method_config import UnsupportedMethodConfigError
from gecko.algorithms.method_config import validate_method_config
from gecko.algorithms.context import ClientMethodContext
from gecko.algorithms.continual.graphkeeper.algorithm import GraphKeeperAlgorithm
from gecko.algorithms.continual.graphkeeper.lp import GraphKeeperLPAlgorithm
from gecko.models.backbones import LegacyGCNAdapter
from gecko.models.backbones import GECKOGraphModel
from gecko.models.capabilities import attach_v2_model_capabilities
from gecko.data.datasets.synthetic import build_synthetic_spec
from gecko.data.streams import StreamBuilder
from tests.helpers import make_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    resolve_source_path(ROOT
    / "configs"
    / "uefa_v1"
    / "methods"
    / "graphkeeper_lp_local_only_lp_domain_v1.yaml")
)


def _method_config(
    *, stage_count: int = 2, strategy: str = "local_only"
) -> dict:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["strategy"]["name"] = strategy
    payload["continual_method"]["parameters"].update(
        {
            "rank": 2,
            "router_projection_dim": 16,
            "dbscan_min_samples": 2,
            "max_cluster_nodes": 32,
            "max_prototypes_per_domain": 4,
            "pretrain_edges": 8,
            "stage_count": stage_count,
        }
    )
    return payload


def _context(*, stage: int = 0, global_task: int | None = None) -> ClientMethodContext:
    features = torch.arange(72, dtype=torch.float32).reshape(12, 6) / 72
    edges = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 3, 5, 7, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 6, 8],
        ],
        dtype=torch.long,
    )
    queries = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7], [0, 4], [1, 6], [2, 8], [3, 10]],
        dtype=torch.long,
    )
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.long)

    def forward(model, node_features, edge_index, values):
        return model.forward_queries(node_features, edge_index, values, "LP")

    def encode(model, node_features, edge_index, layer_index):
        return model.encode_nodes(
            node_features, edge_index, layer_index=layer_index
        )

    return ClientMethodContext(
        client_id=0,
        global_task_id=stage if global_task is None else global_task,
        stage_index=stage,
        round_index=0,
        problem_type="LP",
        incremental_setting="domain",
        train_queries=queries,
        train_labels=labels,
        valid_class_mask=None,
        node_features=features,
        base_edge_index=edges,
        context_edge_index=None,
        forward_queries=forward,
        encode_nodes=encode,
    )


def _model() -> GECKOGraphModel:
    return attach_v2_model_capabilities(
        GECKOGraphModel(
            input_size=6,
            output_size=1,
            hidden_size=8,
            num_layers=2,
            problem_type="LP",
        )
    )


def test_graphkeeper_lp_config_is_explicit_and_fail_closed() -> None:
    payload = _method_config()
    resolved = validate_method_config(
        payload,
        expected_strategy="local_only",
        expected_continual_method="GraphKeeper-LP",
        problem_type="LP",
        incremental_setting="domain",
    )
    assert resolved.runnable
    assert resolved.support_status == "implemented_unverified"
    assert not resolved.benchmark_eligible
    assert resolved.scientific_fidelity == "mechanism_adaptation"

    with pytest.raises(UnsupportedMethodConfigError):
        validate_method_config(
            payload,
            problem_type="NC",
            incremental_setting="domain",
        )
    nc_payload = _method_config()
    nc_payload["continual_method"]["name"] = "GraphKeeper"
    with pytest.raises(UnsupportedMethodConfigError):
        validate_method_config(
            nc_payload,
            problem_type="LP",
            incremental_setting="domain",
        )


def test_python_module_cli_exposes_graphkeeper_lp() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "gecko.cli.main", "run", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "GraphKeeper-LP" in completed.stdout


def test_graphkeeper_lp_recursive_binary_ridge_matches_joint_solution() -> None:
    model = _model()
    algorithm = GraphKeeperLPAlgorithm(
        problem_type="LP",
        incremental_setting="domain",
        client_id=0,
        seed=4,
        rank=2,
        ridge_lambda=0.25,
    )
    first_x = torch.eye(8)[:3]
    second_x = torch.eye(8)[3:6]
    first_y = torch.tensor([1, 0, 1])
    second_y = torch.tensor([0, 1, 0])
    algorithm._update_analytic_classifier(model, first_x, first_y)
    algorithm._update_analytic_classifier(model, second_x, second_y)
    x = torch.cat((first_x, second_x)).double()
    y = torch.cat((first_y, second_y)).double().reshape(-1, 1)
    expected = torch.linalg.solve(
        x.t() @ x + 0.25 * torch.eye(8, dtype=torch.float64),
        x.t() @ y,
    )
    assert algorithm.state["analytic_w"].shape == (8, 1)
    assert torch.allclose(algorithm.state["analytic_w"], expected)


def test_graphkeeper_link_pretraining_matches_official_elementwise_oracle() -> None:
    hidden = torch.tensor(
        [
            [2.0, -1.0, 0.5],
            [3.0, 4.0, -2.0],
            [-2.0, 5.0, 1.5],
            [1.0, -3.0, 2.0],
        ]
    )
    positives = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    negatives = torch.tensor([[0, 2], [3, 3]], dtype=torch.long)

    positive_scores = hidden[positives[0]] * hidden[positives[1]]
    negative_scores = hidden[negatives[0]] * hidden[negatives[1]]
    scores = torch.cat((positive_scores, negative_scores))
    labels = torch.cat(
        (torch.ones_like(positive_scores), torch.zeros_like(negative_scores))
    )
    expected = F.binary_cross_entropy_with_logits(scores, labels)
    observed = GraphKeeperAlgorithm._link_pretraining_loss(
        hidden, positives, negatives
    )

    assert torch.allclose(observed, expected)
    collapsed_scores = scores.sum(dim=1)
    collapsed_labels = torch.cat(
        (
            torch.ones(positive_scores.shape[0]),
            torch.zeros(negative_scores.shape[0]),
        )
    )
    assert not torch.allclose(
        observed,
        F.binary_cross_entropy_with_logits(collapsed_scores, collapsed_labels),
    )


def test_graphkeeper_inter_loss_matches_nearest_euclidean_oracle() -> None:
    current = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    prototypes = (torch.tensor([1.0, 0.0]), torch.tensor([10.0, 0.0]))

    distances = torch.cdist(current, torch.stack(prototypes))
    expected = torch.reciprocal(distances.min(dim=1).values + 1e-8).mean()
    observed = GraphKeeperAlgorithm._inter_domain_disentanglement_loss(
        current, prototypes
    )

    assert torch.allclose(observed, expected)
    previous_wrong_objective = torch.reciprocal(distances.square() + 1e-8).mean()
    assert not torch.allclose(observed, previous_wrong_objective)
    empty = GraphKeeperAlgorithm._inter_domain_disentanglement_loss(current, ())
    assert empty.item() == 0.0


def test_begin_gcn_lp_pure_torch_path_matches_dgl_cpu() -> None:
    context = _context()
    model = LegacyGCNAdapter(
        input_size=6,
        output_size=1,
        hidden_size=8,
        num_layers=2,
        problem_type="LP",
        incremental_type="domain",
    )
    model.eval()
    features = context.node_features
    edges = context.base_edge_index
    queries = context.train_queries
    graph = model._graph(edges, features.shape[0])
    with torch.no_grad():
        expected_hidden = model.original(graph, features)
        actual_hidden = model._torch_encode_nodes(features, edges)
        expected_scores = (
            expected_hidden[queries[:, 0]] * expected_hidden[queries[:, 1]]
        ).sum(dim=1)
        actual_scores = model._torch_forward_queries(
            features, edges, queries, "LP"
        )
    assert torch.allclose(actual_hidden, expected_hidden, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual_scores, expected_scores, atol=1e-6, rtol=1e-6)


def test_graphkeeper_lp_experts_router_and_checkpoint_roundtrip() -> None:
    context = _context()
    model = _model()
    algorithm = GraphKeeperLPAlgorithm(
        problem_type="LP",
        incremental_setting="domain",
        client_id=0,
        seed=9,
        rank=2,
        intra_weight=0.1,
        inter_weight=0.1,
        router_projection_dim=16,
        dbscan_min_samples=2,
        max_cluster_nodes=4,
        max_prototypes_per_domain=4,
        pretrain_edges=8,
        stage_count=2,
    )
    algorithm.before_task(context)
    algorithm.before_round(context)
    logits = context.forward_queries(model)
    base = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, context.train_labels.float()
    )
    loss = algorithm.augment_loss(model, context, logits, base)
    assert torch.isfinite(loss)
    loss.backward()
    algorithm.after_backward(model, context)
    algorithm.after_round(model, context)
    algorithm.consolidate(model, context)

    assert algorithm.state["analytic_w"].shape == (8, 1)
    assert set(algorithm.state["adapters"]) == {"0"}
    assert set(algorithm.state["inter_prototypes"]) == {"0"}
    assert set(algorithm.state["routing_prototypes"]) == {"0"}
    model.eval()
    algorithm.evaluation_topology(context)
    routed = context.forward_queries(model)
    assert routed.shape == context.train_labels.shape
    assert torch.isfinite(routed).all()
    diagnostics = algorithm.diagnostics()
    assert diagnostics["method"] == "GraphKeeper-LP"
    assert (
        diagnostics["method_version"]
        == "uefa-graphkeeper-lp-v3-paper-faithful-objectives-normalized-routing"
    )
    assert diagnostics["problem_type"] == "LP"
    assert diagnostics["prediction_unit"] == "edge_hadamard"
    assert (
        diagnostics["link_pretraining_objective"]
        == "elementwise_endpoint_hadamard_bce"
    )
    assert (
        diagnostics["inter_domain_objective"]
        == "reciprocal_nearest_euclidean_prototype_distance"
    )
    assert (
        diagnostics["router"]
        == "nearest_normalized_strict_local_structural_context_prototype"
    )
    assert diagnostics["num_experts"] == 1
    assert diagnostics["domain_prediction_accuracy"] == 1.0
    assert diagnostics["contrast_query_count"] == 4
    assert diagnostics["contrast_total_queries"] == 8

    restored = GraphKeeperLPAlgorithm(
        problem_type="LP",
        incremental_setting="domain",
        client_id=0,
        seed=9,
        rank=2,
        stage_count=2,
    )
    restored.load_method_state(algorithm.save_method_state())
    assert restored.diagnostics()["num_experts"] == 1
    nc = GraphKeeperAlgorithm(
        problem_type="NC",
        incremental_setting="domain",
        client_id=0,
        seed=9,
        rank=2,
        stage_count=2,
    )
    with pytest.raises(ValueError, match="checkpoint schema"):
        nc.load_method_state(algorithm.save_method_state())

    old_lp_state = copy.deepcopy(algorithm.save_method_state())
    old_lp_state["format"] = "uefa-graphkeeper-lp-private-state-v2"
    with pytest.raises(ValueError, match="checkpoint schema"):
        restored.load_method_state(old_lp_state)

    old_nc_state = nc.save_method_state()
    old_nc_state["format"] = "uefa-graphkeeper-private-state-v2"
    with pytest.raises(ValueError, match="checkpoint schema"):
        nc.load_method_state(old_nc_state)


def test_graphkeeper_lp_reports_multi_expert_routing_without_trivial_votes() -> None:
    algorithm = GraphKeeperLPAlgorithm(
        problem_type="LP",
        incremental_setting="domain",
        client_id=0,
        seed=31,
        rank=2,
        router_projection_dim=4,
        stage_count=2,
    )
    algorithm.state["task_to_stage"] = {"0": 0, "1": 1}
    algorithm.state["routing_prototypes"] = {"0": torch.zeros((2, 4))}
    algorithm._expected_route_stage = 0
    algorithm._expected_global_task = 0
    algorithm._record_route(stage=0, query_count=1001)

    algorithm.state["routing_prototypes"]["1"] = torch.ones((2, 4))
    algorithm._expected_route_stage = 0
    algorithm._expected_global_task = 0
    algorithm._record_route(stage=1, query_count=1001)
    algorithm._expected_route_stage = 1
    algorithm._expected_global_task = 1
    algorithm._record_route(stage=1, query_count=1001)

    diagnostics = algorithm.diagnostics()
    assert diagnostics["domain_prediction_accuracy"] == pytest.approx(2 / 3)
    assert diagnostics["domain_prediction_decisions"] == 3
    assert diagnostics["multi_expert_only_domain_prediction_accuracy"] == 0.5
    assert diagnostics["multi_expert_only_domain_prediction_decisions"] == 2
    assert diagnostics["multi_expert_only_domain_confusion_matrix"] == {
        "0": {"1": 1},
        "1": {"1": 1},
    }


def test_graphkeeper_lp_structural_router_is_orientation_invariant() -> None:
    context = _context()
    algorithm = GraphKeeperLPAlgorithm(
        problem_type="LP",
        incremental_setting="domain",
        client_id=0,
        seed=37,
        rank=2,
        router_projection_dim=32,
        stage_count=2,
    )
    features = context.node_features
    first_edges = context.base_edge_index
    second_edges = torch.tensor(
        [
            [0, 0, 0, 0, 0, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 7, 8, 9, 10, 11],
        ],
        dtype=torch.long,
    )
    queries = context.train_queries
    reversed_queries = queries.flip(1)

    first = algorithm._domain_routing_prototype(
        features, first_edges, queries
    )
    reversed_first = algorithm._domain_routing_prototype(
        features, first_edges, reversed_queries
    )
    second = algorithm._domain_routing_prototype(
        features, second_edges, queries
    )

    assert torch.allclose(first, reversed_first)
    assert not torch.allclose(first, second)
    algorithm.state["routing_prototypes"] = {
        "0": first.detach().clone(),
        "1": second.detach().clone(),
    }
    assert algorithm._route(features, first_edges, reversed_queries)[0] == 0
    assert algorithm._route(features, second_edges, reversed_queries)[0] == 1


def test_graphkeeper_lp_cosine_dbscan_is_scale_invariant() -> None:
    algorithm = GraphKeeperLPAlgorithm(
        problem_type="LP",
        incremental_setting="domain",
        client_id=0,
        seed=43,
        rank=2,
        dbscan_eps=0.02,
        dbscan_min_samples=2,
        max_prototypes_per_domain=4,
        stage_count=2,
    )
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, -0.02],
            [0.0, 1.0],
            [0.01, 0.99],
            [-0.02, 0.98],
        ]
    )

    prototypes, fallback = algorithm._cluster_prototypes(embeddings, stage=0)
    scaled, scaled_fallback = algorithm._cluster_prototypes(
        embeddings * 100.0, stage=0
    )

    assert not fallback and not scaled_fallback
    assert len(prototypes) == len(scaled) == 2
    assert all(
        torch.allclose(F.normalize(left, dim=0), F.normalize(right, dim=0))
        for left, right in zip(prototypes, scaled)
    )


def test_graphkeeper_lp_consolidation_uses_eval_representation_and_restores_mode() -> None:
    context = _context()
    torch.manual_seed(1701)
    first_model = LegacyGCNAdapter(
        input_size=6,
        output_size=1,
        hidden_size=8,
        num_layers=2,
        problem_type="LP",
        incremental_type="domain",
    )
    second_model = copy.deepcopy(first_model)
    first = GraphKeeperLPAlgorithm(
        problem_type="LP",
        incremental_setting="domain",
        client_id=0,
        seed=41,
        rank=2,
        router_projection_dim=8,
        dbscan_min_samples=2,
        stage_count=2,
    )
    second = GraphKeeperLPAlgorithm(
        problem_type="LP",
        incremental_setting="domain",
        client_id=0,
        seed=41,
        rank=2,
        router_projection_dim=8,
        dbscan_min_samples=2,
        stage_count=2,
    )
    first.before_task(context)
    second.before_task(context)
    first_model.train()
    second_model.eval()
    first_running = [
        norm.running_mean.detach().clone() for norm in first_model.original.norms
    ]
    second_running = [
        norm.running_mean.detach().clone() for norm in second_model.original.norms
    ]

    torch.manual_seed(1)
    first.consolidate(first_model, context)
    torch.manual_seed(999)
    second.consolidate(second_model, context)

    assert first_model.training
    assert not second_model.training
    assert all(
        torch.equal(before, norm.running_mean)
        for before, norm in zip(first_running, first_model.original.norms)
    )
    assert all(
        torch.equal(before, norm.running_mean)
        for before, norm in zip(second_running, second_model.original.norms)
    )
    assert torch.allclose(first.state["analytic_w"], second.state["analytic_w"])
    assert torch.allclose(
        first.state["routing_prototypes"]["0"],
        second.state["routing_prototypes"]["0"],
    )
    first_prototypes = first.state["inter_prototypes"]["0"]
    second_prototypes = second.state["inter_prototypes"]["0"]
    assert len(first_prototypes) == len(second_prototypes)
    assert all(
        torch.allclose(left, right)
        for left, right in zip(first_prototypes, second_prototypes)
    )


def test_graphkeeper_lp_synthetic_domain_end_to_end() -> None:
    config = make_config(
        "LP",
        "domain",
        2,
        seed=941,
        order_profile="synchronized",
        num_clients=1,
        rounds=1,
    )
    stream = StreamBuilder(config).build(
        build_synthetic_spec(
            "LP", "domain", num_tasks=2, num_clients=1, seed=941
        )
    )
    result = FederatedCoordinator(
        stream,
        "local_only",
        "GraphKeeper-LP",
        model_name="begin_gcn",
        method_config=_method_config(stage_count=2),
    ).run()
    assert tuple(result["client_stage_task_matrix"].shape) == (1, 2, 2)
    assert result["resolved_model"] == "begin_gcn"
    assert (
        result["method_config_resolution"]["support_status"]
        == "implemented_unverified"
    )
    assert not result["benchmark_eligible"]
    diagnostics = result["client_method_diagnostics"][0]
    assert diagnostics["method"] == "GraphKeeper-LP"
    assert diagnostics["completed_stages"] == [0, 1]
    assert diagnostics["num_experts"] == 2
    assert diagnostics["analytic_parameter_count"] == 8
    assert diagnostics["domain_prediction_decisions"] > 0


@pytest.mark.parametrize("strategy", ["fedavg", "fedprox"])
def test_graphkeeper_lp_federated_backbone_keeps_method_state_private(
    strategy: str,
) -> None:
    config = make_config(
        "LP",
        "domain",
        2,
        seed=0,
        order_profile="synchronized",
        num_clients=2,
        rounds=1,
    )
    config = replace(
        config,
        training=replace(
            config.training,
            local_epochs_per_round=2,
            fedprox_mu=0.01,
        ),
    )
    stream = StreamBuilder(config).build(
        build_synthetic_spec(
            "LP", "domain", num_tasks=2, num_clients=2, seed=0
        )
    )
    coordinator = FederatedCoordinator(
        stream,
        strategy,
        "GraphKeeper-LP",
        model_name="uefa_gcn",
        method_config=_method_config(stage_count=2, strategy=strategy),
    )

    result = coordinator.run()

    assert all(
        record["communication_payload_bytes"] > 0 for record in result["rounds"]
    )
    assert result["strategy_hyperparameters"]["fedprox_mu"] == (
        0.01 if strategy == "fedprox" else None
    )
    assert coordinator.clients[0].algorithm.state is not (
        coordinator.clients[1].algorithm.state
    )
    for client in coordinator.clients.values():
        assert set(client.algorithm.state["adapters"]) == {"0", "1"}
        assert set(client.algorithm.state["routing_prototypes"]) == {"0", "1"}
        assert set(client.algorithm.state["inter_prototypes"]) == {"0", "1"}
        assert "analytic_w" in client.algorithm.state
    private_tokens = ("adapter", "router", "prototype", "analytic", "ridge")
    assert not any(
        token in key.lower()
        for key in coordinator.global_state
        for token in private_tokens
    )
