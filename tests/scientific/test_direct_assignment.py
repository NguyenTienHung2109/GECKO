from __future__ import annotations

import sys

import pytest
import torch

from gecko.data.partitioning.dirichlet.node import direct_assign_train_nodes
from gecko.data.partitioning.dirichlet.quota import ExactDirichletQuotaGenerator


def _case(seed=4):
    labels = torch.tensor([0] * 6 + [1] * 6 + [2] * 6)
    ids = torch.tensor([12, 3, 18, 2, 15, 0, 8, 5, 14, 1, 16, 9, 6, 19, 4, 13, 7, 17])
    quota = ExactDirichletQuotaGenerator().generate(
        [6, 6, 6], 3, 1.0, 2, [6, 6, 6], [[0], [1], [2]], 1
    )
    return ids, labels, quota, direct_assign_train_nodes(
        ids, labels, quota, num_nodes=22, seed=seed
    )


def test_direct_assignment_exact_quota():
    _, _, quota, result = _case()
    assert torch.equal(result.realized_train_class_matrix, quota.integer_quota)
    assert result.exact_quota_verified


def test_direct_assignment_each_train_node_one_owner():
    _, _, _, result = _case()
    assert bool((result.train_owners >= 0).all())
    assert result.train_owners.numel() == torch.unique(result.train_node_ids).numel()


def test_direct_assignment_no_missing_train_node():
    ids, _, _, result = _case()
    assert torch.equal(result.owner_by_global_node[ids], result.train_owners)


def test_direct_assignment_no_duplicate_train_node():
    ids, labels, quota, _ = _case()
    duplicate = ids.clone()
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="unique"):
        direct_assign_train_nodes(duplicate, labels, quota, num_nodes=22, seed=1)


def test_direct_assignment_deterministic_same_seed():
    *_, first = _case(seed=8)
    *_, second = _case(seed=8)
    assert first.owner_hash == second.owner_hash
    assert torch.equal(first.owner_by_global_node, second.owner_by_global_node)


def test_direct_assignment_different_seed_changes_identity_not_quota():
    _, _, quota, first = _case(seed=8)
    *_, second = _case(seed=9)
    assert not torch.equal(first.train_owners, second.train_owners)
    assert torch.equal(first.realized_train_class_matrix, quota.integer_quota)
    assert torch.equal(second.realized_train_class_matrix, quota.integer_quota)


def test_direct_assignment_does_not_call_micro_community_code(monkeypatch):
    import gecko.data.partitioning.community.micro_communities as communities

    monkeypatch.setattr(
        communities,
        "generate_micro_communities",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("called")),
    )
    _case()


def test_direct_assignment_does_not_call_metis(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymetis", None)
    _case()


def test_direct_assignment_does_not_read_val_test_labels():
    ids, labels, quota, first = _case()
    unrelated_heldout_labels = torch.arange(100)
    unrelated_heldout_labels[:] = -999
    second = direct_assign_train_nodes(ids, labels, quota, num_nodes=22, seed=4)
    assert first.owner_hash == second.owner_hash


def test_direct_assignment_owner_hash_deterministic():
    *_, result = _case()
    assert len(result.owner_hash) == 64
    assert result.owner_hash == _case()[-1].owner_hash


def test_direct_assignment_task_counts_exact():
    _, _, quota, result = _case()
    assert torch.equal(result.realized_train_task_matrix, quota.target_task_counts)
