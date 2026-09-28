from __future__ import annotations

import inspect

import pytest
import torch

from gecko.engine import FederatedCoordinator
from gecko.engine.client import FederatedClient
from gecko.data.splits.common import bitcoin_structural_degree_q4
from gecko.data.scenario import build_lc_spec
from gecko.data.streams import load_stream
from gecko.data.streams import save_stream
from gecko.validation import ScenarioValidationError

from tests.helpers import CASES
from tests.helpers import make_stream
from tests.helpers import positive_key_set


def _stream_with_noncanonical_first_task(problem: str, incremental: str, num_tasks: int):
    for seed in range(20):
        stream = make_stream(problem, incremental, num_tasks, seed=seed)
        if any(order[0] != 0 for order in stream.orders.client_orders.values()):
            return stream
    raise AssertionError("Test setup did not produce a non-canonical client order.")


def test_12_local_stage_maps_to_explicit_global_task_id():
    stream = _stream_with_noncanonical_first_task("NC", "task", 4)
    for client_id, order in stream.orders.client_orders.items():
        for local_stage, expected in enumerate(order):
            assert stream.orders.global_task(client_id, local_stage) == expected


def test_13_seen_task_sets_follow_order_prefix_not_numeric_comparison():
    stream = _stream_with_noncanonical_first_task("NC", "class", 4)
    for client_id, order in stream.orders.client_orders.items():
        for stage in range(stream.scenario.num_tasks):
            assert stream.orders.seen_tasks(client_id, stage) == order[: stage + 1]


def test_14_task_il_mask_uses_global_task_id():
    stream = _stream_with_noncanonical_first_task("NC", "task", 4)
    coordinator = FederatedCoordinator(stream, "local_only", "Bare")
    client_id = next(
        client for client, order in stream.orders.client_orders.items() if order[0] != 0
    )
    global_task = stream.orders.global_task(client_id, 0)
    shard = stream.shards[client_id][global_task]
    assert torch.equal(
        coordinator.clients[client_id]._class_mask(shard),
        stream.scenario.task_masks[global_task],
    )


def test_15_class_il_label_set_is_client_local_and_order_aware():
    stream = _stream_with_noncanonical_first_task("NC", "class", 4)
    coordinator = FederatedCoordinator(stream, "local_only", "Bare")
    client_id = next(iter(coordinator.clients))
    first, second = stream.orders.client_orders[client_id][:2]
    client = coordinator.clients[client_id]
    client.mark_seen(first, stream.shards[client_id][first].task_class_mask)
    actual = client._class_mask(stream.shards[client_id][second])
    expected = stream.scenario.task_masks[[first, second]].any(dim=0)
    assert torch.equal(actual, expected)


def test_16_domain_id_is_absent_from_predictor_inputs():
    stream = make_stream("NC", "domain", 2)
    coordinator = FederatedCoordinator(stream, "fedavg", "Bare")
    client = coordinator.clients[0]
    assert not hasattr(client.scenario, "domains")
    assert "domain" not in inspect.signature(FederatedClient.update).parameters
    coordinator.run()


def test_17_full_ground_truth_is_absent_from_client_objects():
    stream = make_stream("LC", "domain", 4)
    client = FederatedCoordinator(stream, "fedavg", "Bare").clients[0]
    assert not hasattr(client.scenario, "labels")
    assert not hasattr(client.scenario, "query_task_ids")
    assert not hasattr(client.graph, "labels")
    assert not hasattr(client.graph, "node_owner")
    assert not hasattr(client.graph, "local_to_global")
    assert not hasattr(client.graph, "global_to_local")
    assert not hasattr(client.graph, "boundary_mask")


def test_18_future_task_train_labels_are_not_retained_by_client():
    stream = make_stream("NC", "class", 2)
    coordinator = FederatedCoordinator(stream, "local_only", "Bare")
    client = coordinator.clients[0]
    assert not hasattr(client, "shards")
    task = stream.orders.global_task(0, 0)
    shard = stream.shards[0][task]
    client.update(shard, global_task_id=task, server_state=None, strategy="local_only")
    assert not hasattr(client.state, "train_labels")


def test_19_validation_and_test_labels_are_central_only():
    stream = make_stream("LP", "domain", 2)
    shard = stream.shards[0][0]
    assert set(shard.__dict__) == {
        "client_id",
        "global_task_id",
        "train_queries",
        "train_labels",
        "task_class_mask",
        "context_edge_index",
    }
    assert shard.context_edge_index is None
    central = stream.evaluation_shards[0][0]
    assert central.validation_query_ids.numel() > 0
    assert central.test_query_ids.numel() > 0
    assert not hasattr(central, "validation_labels")
    assert not hasattr(central, "test_labels")


def test_21_undirected_reverse_arcs_share_logical_split_and_id():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    spec = build_lc_spec(
        dataset_name="reverse-arcs",
        incremental_type="task",
        metrics=("accuracy",),
        edge_index=edge_index,
        node_features=torch.randn(3, 2),
        raw_labels=torch.tensor([0, 0, 1, 1]),
        raw_task_ids=torch.tensor([0, 0, 1, 1]),
        raw_train_mask=torch.tensor([True, True, False, False]),
        raw_validation_mask=torch.tensor([False, False, True, True]),
        raw_test_mask=torch.tensor([False, False, False, False]),
        num_tasks=2,
        num_classes=2,
        undirected=True,
        task_class_sets={0: torch.tensor([0]), 1: torch.tensor([1])},
    )
    assert spec.query_endpoints.shape[0] == 2
    assert spec.logical_edge_ids[0] == spec.logical_edge_ids[1]
    assert spec.logical_edge_ids[2] == spec.logical_edge_ids[3]
    with pytest.raises(ValueError, match="disagree on split"):
        build_lc_spec(
            dataset_name="bad-reverse-arcs",
            incremental_type="task",
            metrics=("accuracy",),
            edge_index=edge_index,
            node_features=torch.randn(3, 2),
            raw_labels=torch.tensor([0, 0, 1, 1]),
            raw_task_ids=torch.tensor([0, 0, 1, 1]),
            raw_train_mask=torch.tensor([True, False, False, False]),
            raw_validation_mask=torch.tensor([False, True, True, True]),
            raw_test_mask=torch.zeros(4, dtype=torch.bool),
            num_tasks=2,
            num_classes=2,
            undirected=True,
        )


def test_21b_supervised_classification_labels_must_fit_output_vocabulary():
    with pytest.raises(ScenarioValidationError, match="outside"):
        build_lc_spec(
            dataset_name="bad-output-vocabulary",
            incremental_type="domain",
            metrics=("accuracy",),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            node_features=torch.randn(2, 2),
            raw_labels=torch.tensor([1, 1]),
            raw_task_ids=torch.tensor([0, 0]),
            raw_train_mask=torch.tensor([True, True]),
            raw_validation_mask=torch.tensor([False, False]),
            raw_test_mask=torch.tensor([False, False]),
            num_tasks=1,
            num_classes=1,
            undirected=True,
        )


@pytest.mark.parametrize("incremental,num_tasks", [("task", 2), ("class", 2), ("domain", 4)])
def test_22_lc_internal_query_policy_is_enforced(incremental, num_tasks):
    stream = make_stream("LC", incremental, num_tasks)
    for client_id, tasks in stream.shards.items():
        graph = stream.partition.client_graphs[client_id]
        for shard in tasks.values():
            if shard.train_queries.numel():
                assert int(shard.train_queries.min()) >= 0
                assert int(shard.train_queries.max()) < graph.local_to_global.shape[0]
                endpoints = graph.local_to_global[shard.train_queries]
                assert bool((stream.partition.node_owner[endpoints] == client_id).all())


def test_23_bitcoin_structural_domains_are_topology_only():
    signature = inspect.signature(bitcoin_structural_degree_q4)
    assert "labels" not in signature.parameters
    assert "timestamps" not in signature.parameters
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]])
    train = torch.tensor([True, True, True, True, False, False])
    first, _ = bitcoin_structural_degree_q4(edges, train, num_nodes=4)
    second, _ = bitcoin_structural_degree_q4(edges, train, num_nodes=4)
    assert torch.equal(first, second)


def test_24_bitcoin_structural_domain_generation_is_deterministic_and_nonempty():
    edges = torch.tensor([[i % 8, (i * 3 + 1) % 8] for i in range(32)])
    train = torch.arange(32) < 24
    first, first_audit = bitcoin_structural_degree_q4(edges, train, num_nodes=8)
    second, second_audit = bitcoin_structural_degree_q4(edges, train, num_nodes=8)
    assert torch.equal(first, second)
    assert first_audit["training_histogram"] == second_audit["training_histogram"]
    assert all(count > 0 for count in first_audit["training_histogram"])


def test_25_lp_validation_and_test_positives_are_absent_from_contexts():
    stream = make_stream("LP", "domain", 2)
    spec = stream.scenario
    context = {
        (min(source, target), max(source, target))
        for source, target in spec.context_edge_index.t().tolist()
    }
    held_out = spec.metadata["positive_pairs"][spec.metadata["positive_splits"] != 0]
    for source, target in held_out.tolist():
        assert (min(source, target), max(source, target)) not in context
    for graph in stream.partition.client_graphs.values():
        local_context = {
            (min(source, target), max(source, target))
            for source, target in graph.local_to_global[graph.edge_index].t().tolist()
        }
        assert not local_context.intersection(
            {(min(source, target), max(source, target)) for source, target in held_out.tolist()}
        )


def test_26_lp_reverse_heldout_positives_are_absent_from_undirected_context():
    stream = make_stream("LP", "domain", 2)
    spec = stream.scenario
    directed_context = set(map(tuple, spec.context_edge_index.t().tolist()))
    held_out = spec.metadata["positive_pairs"][spec.metadata["positive_splits"] != 0]
    for source, target in held_out.tolist():
        assert (source, target) not in directed_context
        assert (target, source) not in directed_context


def test_27_lp_negatives_are_not_positive_in_any_domain_or_split():
    stream = make_stream("LP", "domain", 2)
    positives = positive_key_set(stream)
    spec = stream.scenario
    for source, target in spec.query_endpoints[spec.labels == 0].tolist():
        assert (min(source, target), max(source, target)) not in positives


def test_28_lp_evaluation_negatives_survive_save_load_exactly(tmp_path):
    stream = make_stream("LP", "domain", 2)
    before = {}
    for task, splits in stream.scenario.query_ids_by_task_split.items():
        for split in ("val", "test"):
            ids = splits[split]
            before[(task, split)] = stream.scenario.query_endpoints[ids][
                stream.scenario.labels[ids] == 0
            ].clone()
    restored = load_stream(save_stream(stream, tmp_path, repository_root=tmp_path))
    for (task, split), expected in before.items():
        ids = restored.scenario.query_ids_by_task_split[task][split]
        actual = restored.scenario.query_endpoints[ids][restored.scenario.labels[ids] == 0]
        assert torch.equal(actual, expected)


def test_29_lp_internal_pair_ownership_artifact_is_valid(tmp_path):
    stream = make_stream("LP", "domain", 2)
    path = save_stream(stream, tmp_path, repository_root=tmp_path)
    pair_owner = torch.load(path / "pair_owner.pt", weights_only=False)
    endpoints = pair_owner["query_endpoints"]
    owners = pair_owner["query_pair_owner"]
    expected_source = stream.partition.node_owner[endpoints[:, 0]]
    expected_target = stream.partition.node_owner[endpoints[:, 1]]
    expected = torch.where(
        expected_source == expected_target,
        expected_source,
        torch.full_like(expected_source, -1),
    )
    assert torch.equal(owners, expected)
