from __future__ import annotations

from dataclasses import replace

import torch
import pytest

from gecko.data.partitioning.community import NCSoftCommunityConfig
from gecko.data.partitioning.community import brute_force_nc_owner
from gecko.data.partitioning.community import build_nc_soft_community_from_spec
from gecko.data.partitioning.community import build_nc_soft_community_owner
from gecko.data.partitioning.base import build_weighted_logical_topology
from gecko.data.partitioning.dirichlet.quota import ExactDirichletQuota
from gecko.types import ScenarioSpec


def _topology(num_nodes: int, edges: list[tuple[int, int]]):
    return build_weighted_logical_topology(
        torch.tensor(edges, dtype=torch.long).T.contiguous(),
        num_nodes=num_nodes,
        directed=False,
        representation_id="nc-soft-test",
    )


def _quota(values: list[list[int]], *, seed: int = 0) -> ExactDirichletQuota:
    integer = torch.tensor(values, dtype=torch.long)
    rows = integer.sum(1)
    columns = integer.sum(0)
    return ExactDirichletQuota(
        raw_dirichlet_preferences=integer.double(),
        projected_real_quota=integer.double(),
        integer_quota=integer,
        client_train_capacities=rows,
        train_class_counts=columns,
        target_task_counts=rows[:, None],
        task_class_groups=(tuple(range(integer.shape[1])),),
        alpha_dirichlet=1.0,
        seed=seed,
        retry_index=0,
        quota_hash=f"nc-test-quota-{values}",
        capacity_mode="explicit_test",
    )


def _config(num_clients: int = 2, *, seed: int = 0) -> NCSoftCommunityConfig:
    return NCSoftCommunityConfig(
        num_clients=num_clients,
        alpha_dirichlet=1.0,
        seed=seed,
        client_size_tolerance=0.0,
    )


def test_nc_soft_realizes_exact_quota_and_splits_one_microcommunity() -> None:
    topology = _topology(4, [(0, 1), (1, 2), (2, 3)])
    result = build_nc_soft_community_owner(
        topology=topology,
        train_ids=torch.arange(4),
        train_labels=torch.zeros(4, dtype=torch.long),
        quota=_quota([[2], [2]]),
        config=_config(),
        micro_ids=torch.zeros(4, dtype=torch.long),
    )
    realized = torch.bincount(result.owner, minlength=2)
    assert realized.tolist() == [2, 2]
    assert torch.unique(result.owner).numel() == 2
    assert result.diagnostics["maximum_owners_per_microcommunity"] == 2
    assert result.diagnostics["initial_quota_residual"] == 0
    assert result.diagnostics["quota_repair_moves"] == 0


def test_nc_soft_keeps_whole_microcommunities_when_quota_allows() -> None:
    topology = _topology(4, [(0, 1), (2, 3), (1, 2)])
    labels = torch.tensor([0, 1, 0, 1])
    result = build_nc_soft_community_owner(
        topology=topology,
        train_ids=torch.arange(4),
        train_labels=labels,
        quota=_quota([[1, 1], [1, 1]]),
        config=_config(seed=7),
        micro_ids=torch.tensor([0, 0, 1, 1]),
    )
    assert result.owner[0] == result.owner[1]
    assert result.owner[2] == result.owner[3]
    assert result.owner[0] != result.owner[2]
    assert result.diagnostics["community_split_mass"] == 0


def test_nc_soft_is_deterministic_and_micro_ids_ignore_quota() -> None:
    topology = _topology(6, [(0, 1), (1, 2), (3, 4), (4, 5), (2, 3)])
    arguments = dict(
        topology=topology,
        train_ids=torch.arange(6),
        train_labels=torch.tensor([0, 0, 1, 0, 1, 1]),
        quota=_quota([[2, 1], [1, 2]]),
        config=_config(seed=11),
        micro_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
    )
    first = build_nc_soft_community_owner(**arguments)
    second = build_nc_soft_community_owner(**arguments)
    assert torch.equal(first.owner, second.owner)
    assert torch.equal(first.micro_ids, second.micro_ids)
    assert first.diagnostics["owner_hash"] == second.diagnostics["owner_hash"]
    assert first.diagnostics["microcommunity_hash"] == second.diagnostics[
        "microcommunity_hash"
    ]


def test_nc_soft_tiny_fixture_reaches_bruteforce_objective() -> None:
    topology = _topology(4, [(0, 1), (2, 3), (1, 2)])
    micro_ids = torch.tensor([0, 0, 1, 1])
    train_ids = torch.arange(4)
    labels = torch.tensor([0, 1, 0, 1])
    quota = _quota([[1, 1], [1, 1]])
    result = build_nc_soft_community_owner(
        topology=topology,
        train_ids=train_ids,
        train_labels=labels,
        quota=quota,
        config=_config(seed=3),
        micro_ids=micro_ids,
    )
    oracle = brute_force_nc_owner(
        topology=topology,
        micro_ids=micro_ids,
        train_ids=train_ids,
        train_labels=labels,
        quota=quota.integer_quota,
        capacities=result.total_capacities,
    )
    assert oracle.feasible
    assert tuple(result.diagnostics["final_soft_objective"]) == oracle.objective
    assert tuple(result.diagnostics["final_soft_objective"]) <= tuple(
        result.diagnostics["initial_soft_objective"]
    )


def test_nc_soft_heldout_labels_and_features_do_not_change_owner() -> None:
    split = {
        0: {
            "train": torch.tensor([0, 1, 2, 3]),
            "val": torch.tensor([4, 5]),
            "test": torch.tensor([6, 7]),
        }
    }
    spec = ScenarioSpec(
        dataset_name="synthetic-nc-soft",
        problem_type="NC",
        incremental_type="class",
        num_tasks=1,
        num_classes=2,
        num_features=2,
        metrics=("accuracy",),
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long
        ),
        node_features=torch.arange(16, dtype=torch.float32).reshape(8, 2),
        query_ids_by_task_split=split,
        query_task_ids=torch.zeros(8, dtype=torch.long),
        labels=torch.tensor([0, 0, 1, 1, 0, 1, 0, 1]),
        task_class_sets={0: torch.tensor([0, 1])},
    )
    config = replace(
        _config(),
        client_size_tolerance=0.5,
        minimum_train_queries_per_client_task=0,
        minimum_validation_queries_per_client_task=1,
        minimum_test_queries_per_client_task=1,
    )
    micro_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    first = build_nc_soft_community_from_spec(
        spec, config, micro_ids=micro_ids
    )
    labels = spec.labels.clone()
    labels[4:] = 1 - labels[4:]
    changed = replace(
        spec,
        labels=labels,
        node_features=torch.randn_like(spec.node_features),
    )
    second = build_nc_soft_community_from_spec(
        changed, config, micro_ids=micro_ids
    )
    assert torch.equal(first.owner, second.owner)
    assert first.diagnostics["quota_hash"] == second.diagnostics["quota_hash"]
    assert first.diagnostics["microcommunity_hash"] == second.diagnostics[
        "microcommunity_hash"
    ]
    assert first.diagnostics["heldout_support_counts"] == {
        "val": [[1, 1]],
        "test": [[1, 1]],
    }


def test_nc_soft_fails_when_heldout_identity_supply_cannot_cover_clients() -> None:
    topology = _topology(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    with pytest.raises(ValueError, match="held-out support infeasible"):
        build_nc_soft_community_owner(
            topology=topology,
            train_ids=torch.tensor([0, 1]),
            train_labels=torch.zeros(2, dtype=torch.long),
            quota=_quota([[1], [1]]),
            config=replace(
                _config(),
                client_size_tolerance=0.5,
                minimum_validation_queries_per_client_task=1,
                minimum_test_queries_per_client_task=1,
            ),
            micro_ids=torch.zeros(5, dtype=torch.long),
            heldout_query_ids_by_task_split={
                0: {"val": torch.tensor([2]), "test": torch.tensor([3, 4])}
            },
        )


def test_nc_soft_rejects_malformed_quota_without_mutating_inputs() -> None:
    topology = _topology(4, [(0, 1), (1, 2), (2, 3)])
    ids = torch.arange(4)
    original = ids.clone()
    bad = replace(_quota([[2], [2]]), integer_quota=torch.tensor([[3], [3]]))
    try:
        build_nc_soft_community_owner(
            topology=topology,
            train_ids=ids,
            train_labels=torch.zeros(4, dtype=torch.long),
            quota=bad,
            config=_config(),
            micro_ids=torch.zeros(4, dtype=torch.long),
        )
    except ValueError as error:
        assert "every training node" in str(error)
    else:
        raise AssertionError("Malformed NC quota must fail closed.")
    assert torch.equal(ids, original)
