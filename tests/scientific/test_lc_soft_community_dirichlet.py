from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gecko.data.partitioning.dirichlet.link import LCEdgeQueryQuota
from gecko.data.partitioning.dirichlet.link import generate_lc_edge_query_quota
from gecko.data.partitioning.community import LCSoftCommunityConfig
from gecko.data.partitioning.community import LCSoftCommunityError
from gecko.data.partitioning.community import build_lc_soft_community_partition
from gecko.data.partitioning.community import brute_force_lc_partition
from gecko.data.partitioning.base import build_weighted_logical_topology
from gecko.data.transforms import scenario_with_selected_training_queries
from gecko.data.partitioning.diagnostics import partition_diagnostics
from gecko.types import ScenarioSpec
from tests.helpers import make_config


def _quota(values: list[list[int]]) -> LCEdgeQueryQuota:
    integer = torch.tensor(values, dtype=torch.long)
    rows = integer.sum(1)
    columns = integer.sum(0)
    return LCEdgeQueryQuota(
        sampled_dirichlet_proportions=integer.double(),
        raw_targets=integer.double(),
        unconstrained_rounded_targets=integer.clone(),
        capacity_independent_projected_targets=integer.double(),
        integer_quota=integer,
        row_sums=rows,
        column_sums=columns,
        task_counts=rows[:, None],
        alpha_dirichlet=1.0,
        semantic_unit="edge_class",
        seed=0,
        quota_hash=f"lc-test-quota-{values}",
    )


def _scenario(
    num_nodes: int,
    train_edges: list[tuple[int, int]],
    *,
    labels: list[int] | None = None,
    heldout: list[tuple[int, int]] | None = None,
) -> ScenarioSpec:
    # Held-out queries exist only to exercise ScenarioSpec and must not affect
    # the v2 construction.
    heldout = heldout or [(0, max(0, num_nodes - 1)), (max(0, num_nodes - 1), 0)]
    queries = torch.tensor(train_edges + heldout, dtype=torch.long)
    train_count = len(train_edges)
    values = labels or [0] * train_count
    all_labels = torch.tensor(values + [0] * len(heldout), dtype=torch.long)
    context_edges = sorted(set(tuple(sorted(edge)) for edge in train_edges + heldout))
    context = torch.tensor(context_edges, dtype=torch.long).T.contiguous()
    spec = ScenarioSpec(
        dataset_name="synthetic-lc-soft",
        problem_type="LC",
        incremental_type="class",
        num_tasks=1,
        num_classes=max(values, default=0) + 1,
        num_features=2,
        metrics=("accuracy",),
        edge_index=context,
        context_edge_index=context,
        node_features=torch.arange(num_nodes * 2, dtype=torch.float32).reshape(num_nodes, 2),
        query_ids_by_task_split={
            0: {
                "train": torch.arange(train_count),
                "val": torch.tensor([train_count]),
                "test": torch.tensor([train_count + 1]),
            }
        },
        query_task_ids=torch.zeros(train_count + len(heldout), dtype=torch.long),
        labels=all_labels,
        query_endpoints=queries,
        task_class_sets={0: torch.arange(max(values, default=0) + 1)},
        metadata={"feature_provenance": "synthetic"},
    )
    spec.validate()
    return spec


def _config(*, beam_width: int = 16, branch_factor: int = 4) -> LCSoftCommunityConfig:
    return LCSoftCommunityConfig(
        num_clients=2,
        alpha_dirichlet=1.0,
        seed=5,
        supervised_budget_fraction=1.0,
        client_size_tolerance=0.5,
        beam_width=beam_width,
        branch_factor=branch_factor,
        exact_oracle_maximum_nodes=9,
        minimum_validation_queries_per_client_task=0,
        minimum_test_queries_per_client_task=0,
    )


def _assert_exact_and_local(spec, result, expected) -> None:
    realized = torch.tensor(
        [
            [result.selected_by_client_group[c][g].numel() for g in range(spec.num_classes)]
            for c in range(2)
        ]
    )
    assert torch.equal(realized, torch.tensor(expected))
    for client, groups in result.selected_by_client_group.items():
        for ids in groups.values():
            endpoints = spec.query_endpoints[ids]
            assert bool((result.owner[endpoints[:, 0]] == client).all())
            assert bool((result.owner[endpoints[:, 1]] == client).all())


def test_lc_quota_persists_authoritative_realized_manipulation() -> None:
    quota = _quota([[9, 1], [1, 9]])

    diagnostics = quota.manipulation_diagnostics()

    assert diagnostics["dirichlet_realized_population"] == "selected_training_queries"
    assert diagnostics["dirichlet_realized_semantic_unit"] == "edge_class"
    assert diagnostics["dirichlet_integer_quota_client_by_semantic_group"] == [
        [9, 1],
        [1, 9],
    ]
    assert diagnostics["dirichlet_task_counts_client_by_task"] == [[10], [10]]
    assert diagnostics["dirichlet_realized_semantic_js_divergence_mean"] > 0
    assert diagnostics["dirichlet_realized_task_js_divergence_mean"] == 0
    assert diagnostics["dirichlet_group_allocation_js_to_uniform_mean"] > 0
    assert 0 < diagnostics[
        "dirichlet_group_allocation_normalized_entropy_mean"
    ] < 1
    assert diagnostics["dirichlet_group_allocation_max_client_share_mean"] > 0.5


def test_lc_quota_realized_semantic_js_separates_alpha_grid() -> None:
    means = []
    group_means = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        seed_means = []
        seed_group_means = []
        for seed in range(4):
            quota = generate_lc_edge_query_quota(
                [4000] * 6,
                num_clients=10,
                alpha_dirichlet=alpha,
                supervised_budget_fraction=0.05,
                seed=seed,
                task_class_groups=((0, 1), (2, 3), (4, 5)),
                minimum_train_queries_per_client_task=5,
            )
            seed_means.append(
                quota.manipulation_diagnostics()[
                    "dirichlet_realized_semantic_js_divergence_mean"
                ]
            )
            seed_group_means.append(
                quota.manipulation_diagnostics()[
                    "dirichlet_group_allocation_js_to_uniform_mean"
                ]
            )
        means.append(sum(seed_means) / len(seed_means))
        group_means.append(sum(seed_group_means) / len(seed_group_means))

    assert means[0] > means[1] > means[2] > means[3]
    assert group_means[0] > group_means[1] > group_means[2] > group_means[3]


def test_active_train_diagnostics_ignore_inactive_backing_queries() -> None:
    spec = _scenario(
        8,
        [(0, 1), (2, 3), (4, 5), (6, 7)],
        labels=[0, 1, 0, 1],
    )
    active = replace(
        spec,
        query_ids_by_task_split={
            0: {
                "train": torch.tensor([0, 3]),
                "val": torch.tensor([4]),
                "test": torch.tensor([5]),
            }
        },
    )
    owner = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    support = torch.zeros((2, 1, 3), dtype=torch.long)
    first = partition_diagnostics(active, owner, support, num_clients=2)
    changed_labels = active.labels.clone()
    changed_labels[1] = 0
    changed_labels[2] = 1
    changed = partition_diagnostics(
        replace(active, labels=changed_labels), owner, support, num_clients=2
    )

    assert first["active_train_label_js_divergence"] == changed[
        "active_train_label_js_divergence"
    ]
    assert first["active_train_label_js_divergence"] == [
        pytest.approx(0.21576155433883565),
        pytest.approx(0.21576155433883565),
    ]
    assert (
        first["legacy_categorical_divergence_semantics"]
        == "complete_backing_tensor_internal_queries_including_inactive_rows"
    )
    assert set(first["active_label_js_divergence_by_split"]) == {
        "train",
        "val",
        "test",
    }


def test_lc_soft_splits_one_microcommunity_and_realizes_exact_quota() -> None:
    spec = _scenario(8, [(0, 1), (2, 3), (4, 5), (6, 7)])
    result = build_lc_soft_community_partition(
        spec,
        _config(),
        micro_ids=torch.zeros(8, dtype=torch.long),
        quota=_quota([[2], [2]]),
    )
    _assert_exact_and_local(spec, result, [[2], [2]])
    assert result.status == "success"
    assert result.diagnostics["maximum_owners_per_microcommunity"] == 2
    assert result.diagnostics["exact_quota_residual"] == 0
    assert result.diagnostics[
        "dirichlet_integer_quota_client_by_semantic_group"
    ] == [[2], [2]]
    assert result.diagnostics["dirichlet_task_counts_client_by_task"] == [[2], [2]]
    assert result.diagnostics["dirichlet_realized_semantic_js_divergence_mean"] == 0


def test_lc_soft_prefers_whole_microcommunities_when_feasible() -> None:
    spec = _scenario(8, [(0, 1), (2, 3), (4, 5), (6, 7)])
    result = build_lc_soft_community_partition(
        spec,
        _config(),
        micro_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        quota=_quota([[2], [2]]),
    )
    _assert_exact_and_local(spec, result, [[2], [2]])
    assert result.owner[:4].unique().numel() == 1
    assert result.owner[4:].unique().numel() == 1
    assert result.owner[0] != result.owner[4]


def test_lc_soft_shared_endpoint_uses_compatible_alternative() -> None:
    spec = _scenario(5, [(0, 1), (1, 2), (3, 4)])
    result = build_lc_soft_community_partition(
        spec,
        _config(),
        micro_ids=torch.tensor([0, 0, 0, 1, 1]),
        quota=_quota([[1], [1]]),
    )
    _assert_exact_and_local(spec, result, [[1], [1]])
    selected = set(result.selected_query_ids.tolist())
    assert 2 in selected
    assert selected != {0, 1}


def test_lc_soft_shared_endpoint_proven_infeasible() -> None:
    spec = _scenario(3, [(0, 1), (1, 2)])
    with pytest.raises(LCSoftCommunityError) as raised:
        build_lc_soft_community_partition(
            spec,
            _config(),
            micro_ids=torch.zeros(3, dtype=torch.long),
            quota=_quota([[1], [1]]),
        )
    assert raised.value.status == "structurally_infeasible"
    assert raised.value.certificate["tiny_oracle_feasible"] is False


def test_lc_soft_is_invariant_to_heldout_labels() -> None:
    spec = _scenario(6, [(0, 1), (2, 3), (4, 5)], labels=[0, 1, 0])
    arguments = dict(
        config=_config(),
        micro_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        quota=_quota([[1, 0], [0, 1]]),
    )
    first = build_lc_soft_community_partition(spec, **arguments)
    labels = spec.labels.clone()
    labels[-2:] = 1 - labels[-2:]
    changed = replace(spec, labels=labels)
    second = build_lc_soft_community_partition(changed, **arguments)
    assert torch.equal(first.owner, second.owner)
    assert torch.equal(first.selected_query_ids, second.selected_query_ids)
    assert first.diagnostics["microcommunity_hash"] == second.diagnostics[
        "microcommunity_hash"
    ]
    assert first.diagnostics["selected_query_hash"] == second.diagnostics[
        "selected_query_hash"
    ]


def test_lc_soft_jointly_guarantees_val_and_test_internal_support() -> None:
    spec = _scenario(
        8,
        [(0, 1), (2, 3), (4, 5), (6, 7)],
        heldout=[(0, 1), (4, 5), (2, 3), (6, 7)],
    )
    spec = replace(
        spec,
        query_ids_by_task_split={
            0: {
                "train": torch.arange(4),
                "val": torch.tensor([4, 5]),
                "test": torch.tensor([6, 7]),
            }
        },
    )
    config = replace(
        _config(),
        minimum_validation_queries_per_client_task=1,
        minimum_test_queries_per_client_task=1,
    )
    result = build_lc_soft_community_partition(
        spec,
        config,
        micro_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        quota=_quota([[2], [2]]),
    )
    assert result.diagnostics["heldout_support_counts"] == {
        "val": [[1, 1]],
        "test": [[1, 1]],
    }
    assert result.diagnostics["heldout_anchor_count"] == 4


def test_lc_soft_fails_closed_when_heldout_support_is_impossible() -> None:
    spec = _scenario(4, [(0, 1), (2, 3)])
    config = replace(
        _config(),
        minimum_validation_queries_per_client_task=1,
        minimum_test_queries_per_client_task=1,
    )
    with pytest.raises(LCSoftCommunityError) as raised:
        build_lc_soft_community_partition(
            spec, config, micro_ids=torch.zeros(4, dtype=torch.long), quota=_quota([[1], [1]])
        )
    assert raised.value.status == "structurally_infeasible"
    assert raised.value.certificate["stage"] == "heldout_supply"


def test_lc_soft_uses_heldout_endpoints_and_detects_structural_conflict() -> None:
    spec = _scenario(
        4,
        [(0, 1), (2, 3)],
        heldout=[(0, 1), (0, 1), (2, 3), (2, 3)],
    )
    spec = replace(
        spec,
        query_ids_by_task_split={
            0: {
                "train": torch.tensor([0, 1]),
                "val": torch.tensor([2, 3]),
                "test": torch.tensor([4, 5]),
            }
        },
    )
    config = replace(
        _config(),
        minimum_validation_queries_per_client_task=1,
        minimum_test_queries_per_client_task=1,
    )
    with pytest.raises(LCSoftCommunityError) as raised:
        build_lc_soft_community_partition(
            spec,
            config,
            micro_ids=torch.tensor([0, 0, 1, 1]),
            quota=_quota([[1], [1]]),
        )
    assert raised.value.status == "structurally_infeasible"
    assert raised.value.certificate["tiny_oracle_feasible"] is False


def test_lc_soft_is_bitwise_deterministic() -> None:
    spec = _scenario(5, [(0, 1), (1, 2), (3, 4)])
    arguments = dict(
        micro_ids=torch.tensor([0, 0, 0, 1, 1]),
        quota=_quota([[1], [1]]),
    )
    first = build_lc_soft_community_partition(spec, _config(), **arguments)
    second = build_lc_soft_community_partition(spec, _config(), **arguments)
    assert torch.equal(first.owner, second.owner)
    assert torch.equal(first.selected_query_ids, second.selected_query_ids)
    assert first.diagnostics["audit_hash"] == second.diagnostics["audit_hash"]


def test_lc_soft_tiny_fixture_reaches_bruteforce_objective() -> None:
    spec = _scenario(4, [(0, 1), (2, 3)])
    micro_ids = torch.tensor([0, 0, 1, 1])
    quota = _quota([[1], [1]])
    result = build_lc_soft_community_partition(
        spec, _config(), micro_ids=micro_ids, quota=quota
    )
    topology = build_weighted_logical_topology(
        spec.context_edge_index,
        num_nodes=4,
        directed=False,
        representation_id="lc-soft-oracle-test",
    )
    oracle = brute_force_lc_partition(
        topology=topology,
        micro_ids=micro_ids,
        query_ids=spec.query_ids_by_task_split[0]["train"],
        query_endpoints=spec.query_endpoints,
        query_groups=spec.labels[spec.query_ids_by_task_split[0]["train"]],
        quota=quota.integer_quota,
        capacity_lower=1,
        capacity_upper=3,
    )
    assert oracle.feasible
    assert tuple(result.diagnostics["soft_objective"]) == oracle.objective


def test_lc_soft_search_exhausted_is_not_mislabeled_infeasible() -> None:
    spec = _scenario(4, [(0, 1), (0, 2), (0, 3), (1, 2)])
    micro_ids = torch.zeros(4, dtype=torch.long)
    small = replace(
        _config(beam_width=1, branch_factor=1),
        seed=0,
        maximum_search_expansions=1,
    )
    with pytest.raises(LCSoftCommunityError) as raised:
        build_lc_soft_community_partition(
            spec, small, micro_ids=micro_ids, quota=_quota([[1], [1]])
        )
    assert raised.value.status == "search_exhausted"
    assert raised.value.certificate["tiny_oracle_feasible"] is True
    larger = replace(
        small, beam_width=32, branch_factor=4, maximum_search_expansions=100
    )
    result = build_lc_soft_community_partition(
        spec, larger, micro_ids=micro_ids, quota=_quota([[1], [1]])
    )
    assert result.status == "success"


def test_lc_soft_masks_every_unselected_training_label() -> None:
    spec = _scenario(5, [(0, 1), (1, 2), (3, 4)])
    result = build_lc_soft_community_partition(
        spec,
        _config(),
        micro_ids=torch.tensor([0, 0, 0, 1, 1]),
        quota=_quota([[1], [1]]),
    )
    selected_scenario = scenario_with_selected_training_queries(spec, result)
    retained = selected_scenario.query_ids_by_task_split[0]["train"]
    assert torch.equal(torch.sort(retained).values, result.selected_query_ids)
    assert retained.numel() == 2 < spec.query_ids_by_task_split[0]["train"].numel()
    assert (
        selected_scenario.metadata["lc_partition_mode"]
        == "lc_soft_community_exact_edge_dirichlet_v3"
    )
    assert (
        selected_scenario.metadata["lc_stream_version"]
        == "lc_soft_community_selected_query_stream_v3"
    )


def test_lc_soft_accepts_an_exact_zero_query_budget() -> None:
    spec = _scenario(4, [])
    result = build_lc_soft_community_partition(
        spec,
        _config(),
        micro_ids=torch.tensor([0, 0, 1, 1]),
        quota=_quota([[0], [0]]),
    )
    assert result.status == "success"
    assert result.selected_query_ids.numel() == 0
    assert result.diagnostics["exact_quota_residual"] == 0
    assert torch.bincount(result.owner, minlength=2).tolist() == [2, 2]
