from __future__ import annotations

import torch

from gecko.algorithms.federated.motion.gtmsc import MotionReservoir
from gecko.algorithms.federated.motion.gtmsc import motion_coarsen_graph
from gecko.algorithms.federated.motion.gtmsc import motion_merge_observed_graph
from gecko.algorithms.federated.motion.gtmsc import motion_multi_expert_scores
from gecko.algorithms.federated.motion.gtmsc import motion_topology_features
from gecko.algorithms.federated.motion.gtmsc import motion_update_reservoir


def _bidirectional_edges(pairs: list[tuple[int, int]]) -> torch.Tensor:
    directed = pairs + [(target, source) for source, target in pairs]
    return torch.tensor(directed, dtype=torch.long).t().contiguous()


def test_motion_reservoir_is_bounded_deterministic_and_label_aligned() -> None:
    queries = torch.arange(12, dtype=torch.long)
    labels = queries.remainder(3)
    first = motion_update_reservoir(None, queries, labels, capacity=5, seed=17)
    second = motion_update_reservoir(None, queries, labels, capacity=5, seed=17)
    assert first.seen_samples == 12
    assert first.node_ids.numel() == 5
    assert torch.equal(first.node_ids, second.node_ids)
    assert torch.equal(first.labels, second.labels)
    assert torch.equal(first.labels, first.node_ids.remainder(3))


def test_motion_topology_features_identify_bridge_importance() -> None:
    # Two triangles connected through node 2--3.
    edges = _bidirectional_edges(
        [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
    )
    topology = motion_topology_features(edges, num_nodes=6)
    assert topology.shape == (6, 8)
    assert torch.isfinite(topology).all()
    assert topology[2, 1] > topology[0, 1]
    assert topology[3, 1] > topology[5, 1]


def test_motion_merge_uses_only_current_training_labels() -> None:
    features = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    edges = _bidirectional_edges([(0, 1), (1, 2), (2, 3), (3, 4)])
    memory = motion_merge_observed_graph(
        None,
        node_features=features,
        edge_index=edges,
        train_queries=torch.tensor([0, 4]),
        train_labels=torch.tensor([1, 2]),
        num_classes=3,
    )
    assert memory.node_to_coarse[5] == -1
    assert int(memory.label_histogram.sum()) == 2
    assert int(memory.train_mask.sum()) == 2
    assert set(memory.labels[memory.train_mask].tolist()) == {1, 2}


def test_motion_multi_expert_scores_are_sparse_and_deterministic() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=torch.float32,
    )
    edges = _bidirectional_edges([(0, 1), (1, 2), (2, 3)])
    first = motion_multi_expert_scores(features, edges, expert_select=2)
    second = motion_multi_expert_scores(features, edges, expert_select=2)
    assert torch.equal(first.scores, second.scores)
    assert torch.equal(first.gates, second.gates)
    assert torch.equal((first.gates > 0).sum(dim=1), torch.full((4,), 2))
    assert torch.allclose(first.gates.sum(dim=1), torch.ones(4))


def test_motion_coarsening_preserves_reservoir_mapping_and_feature_means() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [1.0, 0.2], [0.0, 1.0], [0.2, 1.0]],
        dtype=torch.float32,
    )
    edges = _bidirectional_edges([(0, 1), (1, 2), (2, 3)])
    graph = motion_merge_observed_graph(
        None,
        node_features=features,
        edge_index=edges,
        train_queries=torch.arange(4),
        train_labels=torch.tensor([0, 0, 1, 1]),
        num_classes=2,
    )
    result = motion_coarsen_graph(
        graph,
        features,
        protected_raw_nodes=torch.tensor([0, 3]),
        reduction_rate=0.5,
        similarity_threshold=0.0,
    )
    coarse = result.graph
    assert coarse.features.shape == (2, 2)
    assert set(coarse.node_to_coarse[[0, 3]].tolist()) == {0, 1}
    assert torch.equal(torch.bincount(coarse.node_to_coarse), torch.tensor([2, 2]))
    assert torch.allclose(coarse.features[0], torch.tensor([1.0, 0.1]))
    assert torch.allclose(coarse.features[1], torch.tensor([0.1, 1.0]))
    assert coarse.edge_index.shape[0] == 2
    assert int(coarse.edge_index.max()) < coarse.features.shape[0]


def test_motion_reservoir_can_continue_from_checkpoint_state() -> None:
    initial = MotionReservoir(
        node_ids=torch.tensor([1, 2]),
        labels=torch.tensor([0, 1]),
        seen_samples=2,
    )
    updated = motion_update_reservoir(
        initial,
        torch.tensor([3, 4]),
        torch.tensor([1, 0]),
        capacity=3,
        seed=23,
    )
    assert updated.seen_samples == 4
    assert updated.node_ids.numel() == 3
