from __future__ import annotations

import pytest
import torch

from gecko.data.partitioning.diagnostics import _components
from gecko.data.partitioning.diagnostics import _global_homophily
from gecko.data.partitioning.diagnostics import _local_homophily
from gecko.types import ScenarioSpec


def _multi_label_spec() -> ScenarioSpec:
    return ScenarioSpec(
        dataset_name="diagnostic-fixture",
        problem_type="NC",
        incremental_type="domain",
        num_tasks=1,
        num_classes=2,
        num_features=1,
        metrics=("rocauc",),
        edge_index=torch.tensor(
            [[0, 0, 1, 2, 3], [1, 2, 3, 3, 0]], dtype=torch.long
        ),
        node_features=torch.zeros(4, 1),
        query_ids_by_task_split={
            0: {
                "train": torch.tensor([0], dtype=torch.long),
                "val": torch.tensor([1], dtype=torch.long),
                "test": torch.tensor([2, 3], dtype=torch.long),
            }
        },
        query_task_ids=torch.zeros(4, dtype=torch.long),
        labels=torch.tensor([[1, 0], [1, 1], [0, 1], [0, 1]], dtype=torch.long),
    )


def _dense_homophily(labels: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> float:
    source_labels = labels[source]
    target_labels = labels[target]
    valid = (source_labels >= 0) & (target_labels >= 0)
    agreement = ((source_labels > 0) == (target_labels > 0)) & valid
    return float((agreement.sum(dim=1).float() / valid.sum(dim=1).clamp_min(1)).mean())


def test_chunked_multi_label_homophily_matches_dense_fixture():
    spec = _multi_label_spec()
    owner = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    source, target = spec.edge_index

    assert _global_homophily(spec) == pytest.approx(
        _dense_homophily(spec.labels, source, target)
    )
    expected_local = []
    for client in range(2):
        internal = (owner[source] == client) & (owner[target] == client)
        expected_local.append(
            None
            if not internal.any()
            else _dense_homophily(spec.labels, source[internal], target[internal])
        )
    assert _local_homophily(spec, owner, 2) == pytest.approx(expected_local)


def test_sparse_component_summary_matches_expected_connectivity():
    edges = torch.tensor([[0, 1, 3], [1, 0, 3]], dtype=torch.long)
    components, largest, isolated = _components(edges, 4)

    assert components == 3
    assert largest == pytest.approx(0.5)
    assert isolated == pytest.approx(0.25)
