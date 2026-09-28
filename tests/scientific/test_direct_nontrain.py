from __future__ import annotations

import torch

from gecko.config import PartitionConfig
from gecko.data.partitioning.dirichlet.nontrain import assign_nontrain_nodes_balanced_neighbor_affinity
from gecko.data.partitioning.materialize import rematerialize_partition_context
from gecko.data.datasets.synthetic import build_synthetic_spec
from gecko.types import PartitionResult


def _case(edges=None, tolerance=0.0):
    train = torch.tensor([0, 1, 2, 0, 1, 2, -1, -1, -1, -1, -1, -1])
    edge = torch.tensor(
        edges
        or [[0, 6], [3, 6], [1, 7], [4, 7], [2, 8], [5, 8], [6, 9], [7, 10], [8, 11]],
        dtype=torch.long,
    ).T
    result = assign_nontrain_nodes_balanced_neighbor_affinity(
        train,
        edge,
        torch.ones(edge.shape[1]),
        num_clients=3,
        directed=False,
        client_size_tolerance=tolerance,
        fixed_pass_count=3,
    )
    return train, edge, result


def test_nontrain_assignment_every_node_one_owner():
    _, _, result = _case()
    assert bool((result.complete_owner >= 0).all())
    assert bool((result.complete_owner < 3).all())


def test_nontrain_assignment_train_owners_unchanged():
    train, _, result = _case()
    mask = train >= 0
    assert torch.equal(result.complete_owner[mask], train[mask])


def test_nontrain_assignment_ignores_val_test_labels():
    _, _, first = _case()
    fake_heldout_labels = torch.tensor([999, -3, 17, 42, 8, 1])
    fake_heldout_labels.flip(0)
    _, _, second = _case()
    assert first.frozen_nontrain_owner_hash == second.frozen_nontrain_owner_hash


def test_nontrain_assignment_deterministic():
    assert _case()[-1].complete_owner_hash == _case()[-1].complete_owner_hash


def test_nontrain_assignment_respects_capacity_bounds():
    _, _, result = _case()
    assert result.node_counts.tolist() == [4, 4, 4]
    assert bool((result.node_counts >= result.capacity_lower).all())
    assert bool((result.node_counts <= result.capacity_upper).all())


def test_nontrain_assignment_neighbor_affinity_manual_graph():
    train = torch.tensor([0, 1, -1, -1])
    edges = torch.tensor([[1, 2], [2, 3]], dtype=torch.long).T
    result = assign_nontrain_nodes_balanced_neighbor_affinity(
        train,
        edges,
        torch.ones(2),
        num_clients=2,
        directed=False,
        client_size_tolerance=0.5,
        fixed_pass_count=2,
    )
    assert int(result.complete_owner[2]) == 1


def test_nontrain_owner_hash_fixed_across_hconn_targets():
    result = _case()[-1]
    simulated_targets = {0.2: result.frozen_nontrain_owner_hash, 0.8: result.frozen_nontrain_owner_hash}
    assert len(set(simulated_targets.values())) == 1


def test_nontrain_assignment_does_not_call_community_code(monkeypatch):
    import gecko.data.partitioning.community.micro_communities as communities

    monkeypatch.setattr(communities, "generate_micro_communities", lambda *a, **k: 1 / 0)
    _case()


def test_complete_global_ownership():
    train, _, result = _case()
    assert result.complete_owner.shape == train.shape
    assert result.nontrain_node_ids.numel() == int((train < 0).sum())


def test_persistent_owner_format_compatible_with_existing_materializer():
    spec = build_synthetic_spec("NC", "class", num_tasks=2, num_clients=3, seed=1)
    owner = torch.arange(spec.node_features.shape[0]) % 3
    partition = PartitionResult(owner, {}, torch.empty(0, dtype=torch.long), {})
    rebuilt = rematerialize_partition_context(spec, partition, PartitionConfig(num_clients=3))
    assert torch.equal(rebuilt.node_owner, owner)
    assert set(rebuilt.client_graphs) == {0, 1, 2}
