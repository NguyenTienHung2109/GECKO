from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gecko.data.partitioning.base import HConnMetrics
from gecko.data.partitioning.base import build_weighted_logical_topology
from gecko.data.partitioning.base import evaluate_hconn
from gecko.data.partitioning.materialize import materialize_direct_partition
from gecko.types import ScenarioSpec
from tests.helpers import make_config


def _topology(*, backend: str = "auto"):
    return build_weighted_logical_topology(
        torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        num_nodes=4,
        directed=False,
        representation_id="lpt-utils-test",
        backend=backend,
    )


def test_topology_backends_produce_the_same_scientific_object() -> None:
    reference = _topology(backend="python")
    tensor = _topology(backend="tensor")
    assert torch.equal(reference.edge_index, tensor.edge_index)
    assert torch.equal(reference.edge_weights, tensor.edge_weights)
    assert reference.topology_hash == tensor.topology_hash


def test_connectivity_diagnostic_and_serialization() -> None:
    metrics = evaluate_hconn(
        torch.tensor([0, 0, 1, 1]), _topology(), num_clients=2
    )
    assert metrics.h_conn == pytest.approx(1 / 3)
    assert HConnMetrics.from_dict(metrics.to_dict()) == metrics


def test_reverse_arcs_are_one_logical_edge() -> None:
    topology = build_weighted_logical_topology(
        torch.tensor([[0, 1], [1, 0]]),
        num_nodes=2,
        directed=False,
        representation_id="reverse-pair",
    )
    assert topology.logical_edge_count == 1


def test_materializer_builds_only_strict_local_edges() -> None:
    split = {
        0: {
            "train": torch.tensor([0, 1]),
            "val": torch.tensor([2, 3]),
            "test": torch.tensor([4, 5]),
        }
    }
    scenario = ScenarioSpec(
        dataset_name="synthetic",
        problem_type="NC",
        incremental_type="task",
        num_tasks=1,
        num_classes=2,
        num_features=2,
        metrics=("accuracy",),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
        node_features=torch.arange(12, dtype=torch.float32).reshape(6, 2),
        query_ids_by_task_split=split,
        query_task_ids=torch.zeros(6, dtype=torch.long),
        labels=torch.tensor([0, 1, 0, 1, 0, 1]),
        task_class_sets={0: torch.tensor([0, 1])},
        task_masks=torch.ones((1, 2), dtype=torch.bool),
        metadata={"feature_provenance": "provided_node_attributes"},
    )
    config = make_config("NC", "task", 1, seed=0, num_clients=2)
    config = replace(
        config,
        partition=replace(
            config.partition,
            minimum_train_queries_per_client_task=0,
            minimum_validation_queries_per_client_task=0,
            minimum_test_queries_per_client_task=0,
        ),
    )
    partition = materialize_direct_partition(
        scenario,
        torch.tensor([0, 1, 0, 1, 0, 1]),
        config,
        scientific_diagnostics={},
    )
    assert all(
        graph.edge_index.numel() == 0 for graph in partition.client_graphs.values()
    )
