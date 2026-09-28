from __future__ import annotations

import pytest
import torch

from gecko.data.scenario import build_nc_spec
from gecko.data.splits.node import select_nc_class_tasks


def _seven_class_spec():
    labels = torch.arange(7, dtype=torch.long).repeat_interleave(3)
    task_ids = labels.clone()
    node_ids = torch.arange(labels.numel(), dtype=torch.long)
    edge_index = torch.stack((node_ids, torch.roll(node_ids, shifts=-1)))
    position_in_class = node_ids.remainder(3)
    return build_nc_spec(
        dataset_name="tiny-seven-class",
        incremental_type="class",
        metrics=("accuracy",),
        edge_index=edge_index,
        node_features=torch.arange(labels.numel() * 2, dtype=torch.float32).reshape(-1, 2),
        labels=labels,
        task_ids=task_ids,
        train_mask=position_in_class == 0,
        validation_mask=position_in_class == 1,
        test_mask=position_in_class == 2,
        num_tasks=7,
        num_classes=7,
        task_class_sets={
            class_id: torch.tensor([class_id], dtype=torch.long)
            for class_id in range(7)
        },
    )


def test_select_six_classes_as_three_two_class_tasks() -> None:
    source = _seven_class_spec()
    paired = select_nc_class_tasks(source, ((0, 1), (2, 3), (4, 5)))

    paired.validate()
    assert paired.num_tasks == 3
    assert paired.num_classes == 6
    assert [paired.task_class_sets[task].tolist() for task in range(3)] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]
    assert paired.task_masks.tolist() == [
        [True, True, False, False, False, False],
        [False, False, True, True, False, False],
        [False, False, False, False, True, True],
    ]

    excluded = source.labels == 6
    assert bool((paired.labels[excluded] == -1).all())
    assert bool((paired.query_task_ids[excluded] == -1).all())
    assert paired.metadata["excluded_original_classes"] == [6]
    assert paired.metadata["excluded_nodes_are_topology_only_context"] is True
    assert torch.equal(paired.edge_index, source.edge_index)
    assert torch.equal(paired.node_features, source.node_features)

    for task_id in range(3):
        for split in ("train", "val", "test"):
            query_ids = paired.query_ids_by_task_split[task_id][split]
            assert query_ids.numel() == 2
            assert set(paired.labels[query_ids].tolist()) == set(
                paired.task_class_sets[task_id].tolist()
            )
            assert not bool(excluded[query_ids].any())


@pytest.mark.parametrize(
    "groups, message",
    [
        (((0, 1), (1, 2)), "exactly one task"),
        (((0, 7),), "outside the source class range"),
        (((), (0, 1)), "nonempty tasks"),
    ],
)
def test_select_nc_class_tasks_rejects_invalid_groups(groups, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        select_nc_class_tasks(_seven_class_spec(), groups)
