from __future__ import annotations

import torch

from gecko.data.partitioning import assignment as assignment_module
from gecko.data.partitioning.assignment import assign_micro_communities
from gecko.types import ScenarioSpec


def test_lc_pair_support_anchors_cross_community_queries_for_every_split(
    monkeypatch,
):
    """LC support must not depend on greedy assignment preserving query pairs."""

    def unexpected_pymetis(*args, **kwargs):
        raise AssertionError("LC assignment must not call in-process PyMetis")

    monkeypatch.setattr(
        assignment_module, "_compressed_metis_assignments", unexpected_pymetis
    )

    endpoints = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], dtype=torch.long
    )
    spec = ScenarioSpec(
        dataset_name="pair-support-fixture",
        problem_type="LC",
        incremental_type="domain",
        num_tasks=1,
        num_classes=2,
        num_features=1,
        metrics=("accuracy",),
        edge_index=endpoints.t().contiguous(),
        node_features=torch.zeros(12, 1),
        query_ids_by_task_split={
            0: {
                "train": torch.tensor([0, 1], dtype=torch.long),
                "val": torch.tensor([2, 3], dtype=torch.long),
                "test": torch.tensor([4, 5], dtype=torch.long),
            }
        },
        query_task_ids=torch.zeros(6, dtype=torch.long),
        labels=torch.zeros(6, dtype=torch.long),
        query_endpoints=endpoints,
    )

    outcome = assign_micro_communities(
        spec,
        torch.arange(12, dtype=torch.long),
        num_clients=2,
        seed=13,
        spatial_profile="easy",
        client_size_tolerance=1.0,
        minimum_support=(1, 1, 1),
        maximum_iterations=1,
        allow_infeasible=False,
    )

    assert not outcome.failures
    assert torch.all(outcome.support >= 1)
    assert outcome.support_anchor_community_count >= 6
