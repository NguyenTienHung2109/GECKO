from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gecko.data.partitioning.dirichlet.quota import ExactDirichletQuotaGenerator
from gecko.data.partitioning.dirichlet.quota import ExactDirichletQuotaInfeasibleError
from gecko.data.partitioning.dirichlet.semantic_target import deterministic_transport_rounding
from gecko.data.datasets.synthetic import build_synthetic_spec


def _quota(seed=3, **changes):
    values = dict(
        train_class_counts=[12, 9, 15, 6],
        num_clients=3,
        alpha_dirichlet=1.0,
        seed=seed,
        client_train_capacities=None,
        task_class_groups=[[0, 1], [2, 3]],
        min_train_support_per_client_task=1,
    )
    values.update(changes)
    return ExactDirichletQuotaGenerator(maximum_retries=32).generate(**values)


def test_exact_quota_class_column_sums():
    quota = _quota()
    assert quota.column_sums.tolist() == [12, 9, 15, 6]


def test_exact_quota_client_row_sums():
    quota = _quota(client_train_capacities=[10, 14, 18])
    assert quota.row_sums.tolist() == [10, 14, 18]


def test_exact_quota_integer_nonnegative():
    quota = _quota()
    assert quota.integer_quota.dtype == torch.long
    assert bool((quota.integer_quota >= 0).all())


def test_exact_quota_same_seed_deterministic():
    first, second = _quota(), _quota()
    assert first.to_dict() == second.to_dict()
    assert first.quota_hash == second.quota_hash


def test_exact_quota_different_seed_changes_target():
    first, second = _quota(seed=1), _quota(seed=2)
    assert not torch.equal(
        first.raw_dirichlet_preferences, second.raw_dirichlet_preferences
    )
    assert first.quota_hash != second.quota_hash


def test_exact_quota_support_retry():
    quota = _quota(
        seed=0,
        train_class_counts=[12, 12],
        task_class_groups=[[0], [1]],
        min_train_support_per_client_task=2,
    )
    assert quota.retry_index >= 0
    assert bool((quota.target_task_counts >= 2).all())


def test_exact_quota_impossible_support_fails_closed():
    with pytest.raises(ExactDirichletQuotaInfeasibleError, match="fixed retry"):
        _quota(
            train_class_counts=[2, 40],
            task_class_groups=[[0], [1]],
            min_train_support_per_client_task=1,
        )


def test_exact_quota_uses_train_labels_only():
    spec = build_synthetic_spec("NC", "class", num_tasks=2, num_clients=2, seed=9)
    first = ExactDirichletQuotaGenerator().generate_from_spec(
        spec,
        num_clients=2,
        alpha_dirichlet=1.0,
        seed=4,
        min_train_support_per_client_task=1,
    )
    labels = spec.labels.clone()
    heldout = torch.cat(
        [
            spec.query_ids_by_task_split[task][split]
            for task in range(spec.num_tasks)
            for split in ("val", "test")
        ]
    )
    labels[heldout] = (labels[heldout] + 1) % spec.num_classes
    second = ExactDirichletQuotaGenerator().generate_from_spec(
        replace(spec, labels=labels),
        num_clients=2,
        alpha_dirichlet=1.0,
        seed=4,
        min_train_support_per_client_task=1,
    )
    assert first.quota_hash == second.quota_hash


def test_exact_quota_transport_rounding_manual_case():
    real = torch.tensor([[1.8, 0.2], [0.2, 1.8]], dtype=torch.float64)
    rounded = deterministic_transport_rounding(
        real, torch.tensor([2, 2]), torch.tensor([2, 2])
    )
    assert rounded.tolist() == [[2, 0], [0, 2]]


def test_exact_quota_tie_break_deterministic():
    real = torch.full((2, 2), 0.5, dtype=torch.float64)
    first = deterministic_transport_rounding(
        real, torch.tensor([1, 1]), torch.tensor([1, 1])
    )
    second = deterministic_transport_rounding(
        real, torch.tensor([1, 1]), torch.tensor([1, 1])
    )
    assert torch.equal(first, second)
    assert first.tolist() == [[0, 1], [1, 0]]


def test_exact_quota_hash_changes_with_scientific_input():
    baseline = _quota()
    assert baseline.quota_hash != _quota(seed=8).quota_hash
    assert baseline.quota_hash != _quota(alpha_dirichlet=0.5).quota_hash
    assert baseline.quota_hash != _quota(train_class_counts=[13, 8, 15, 6]).quota_hash


def test_exact_quota_task_counts_match_class_groups():
    quota = _quota()
    expected = torch.stack(
        (quota.integer_quota[:, [0, 1]].sum(1), quota.integer_quota[:, [2, 3]].sum(1)),
        dim=1,
    )
    assert torch.equal(quota.target_task_counts, expected)
