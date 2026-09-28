from __future__ import annotations

from dataclasses import replace

import torch

from gecko.data.transforms import apply_balanced_class_task_plan
from gecko.data.transforms import build_balanced_class_task_plan
from gecko.data.scenario import build_nc_spec
from gecko.types import ScenarioSpec


def _nc_source() -> ScenarioSpec:
    train_counts = (9, 8, 7, 6, 5, 4, 1)
    labels: list[int] = []
    split_codes: list[int] = []
    for class_id, count in enumerate(train_counts):
        labels.extend([class_id] * count + [class_id, class_id])
        split_codes.extend([0] * count + [1, 2])
    label_tensor = torch.tensor(labels, dtype=torch.long)
    codes = torch.tensor(split_codes, dtype=torch.long)
    nodes = torch.arange(label_tensor.numel(), dtype=torch.long)
    return build_nc_spec(
        dataset_name="balanced-nc",
        incremental_type="class",
        metrics=("accuracy",),
        edge_index=torch.stack((nodes, torch.roll(nodes, shifts=-1))),
        node_features=torch.arange(nodes.numel() * 2, dtype=torch.float32).reshape(-1, 2),
        labels=label_tensor,
        task_ids=label_tensor,
        train_mask=codes == 0,
        validation_mask=codes == 1,
        test_mask=codes == 2,
        num_tasks=7,
        num_classes=7,
        task_class_sets={
            class_id: torch.tensor([class_id]) for class_id in range(7)
        },
    )


def _lc_source() -> ScenarioSpec:
    labels = torch.arange(6, dtype=torch.long).repeat_interleave(3)
    positions = torch.arange(labels.numel()).remainder(3)
    endpoints = torch.stack(
        (torch.arange(labels.numel()).remainder(8), torch.arange(labels.numel()).add(1).remainder(8)),
        dim=1,
    )
    return ScenarioSpec(
        dataset_name="balanced-lc",
        problem_type="LC",
        incremental_type="task",
        num_tasks=6,
        num_classes=6,
        num_features=2,
        metrics=("accuracy",),
        edge_index=torch.stack((torch.arange(8), torch.roll(torch.arange(8), -1))),
        node_features=torch.arange(16, dtype=torch.float32).reshape(8, 2),
        query_ids_by_task_split={
            class_id: {
                "train": torch.nonzero((labels == class_id) & (positions == 0)).flatten(),
                "val": torch.nonzero((labels == class_id) & (positions == 1)).flatten(),
                "test": torch.nonzero((labels == class_id) & (positions == 2)).flatten(),
            }
            for class_id in range(6)
        },
        query_task_ids=labels.clone(),
        labels=labels,
        query_endpoints=endpoints,
        task_class_sets={
            class_id: torch.tensor([class_id]) for class_id in range(6)
        },
    )


def test_plan_drops_rarest_train_class_and_is_heldout_label_blind() -> None:
    source = _nc_source()
    plan = build_balanced_class_task_plan(source, target_num_tasks=3)

    assert plan.excluded_classes == (6,)
    assert plan.class_groups == ((0, 5), (1, 4), (2, 3))
    assert plan.task_train_counts == (13, 13, 13)
    assert all(len(group) == 2 for group in plan.class_groups)
    assert sorted(value for group in plan.class_groups for value in group) == list(range(6))
    heldout_ids = torch.cat(
        [
            source.query_ids_by_task_split[task][split]
            for task in range(source.num_tasks)
            for split in ("val", "test")
        ]
    )
    changed_labels = source.labels.clone()
    changed_labels[heldout_ids] = torch.roll(changed_labels[heldout_ids], shifts=3)
    changed = replace(source, labels=changed_labels)
    assert build_balanced_class_task_plan(changed, target_num_tasks=3) == plan

    transformed = apply_balanced_class_task_plan(source, plan)
    assert transformed.metadata["class_selection_label_access"] == "train_labels_only"
    assert transformed.metadata["excluded_original_classes"] == [6]
    assert bool((transformed.labels[source.labels == 6] == -1).all())


def test_balanced_transform_supports_lc_task_il_without_resampling() -> None:
    source = _lc_source()
    plan = build_balanced_class_task_plan(source, target_num_tasks=3)
    transformed = apply_balanced_class_task_plan(source, plan)

    transformed.validate()
    assert transformed.problem_type == "LC"
    assert transformed.incremental_type == "task"
    assert transformed.num_tasks == 3
    assert transformed.num_classes == 6
    assert torch.equal(transformed.query_endpoints, source.query_endpoints)
    assert all(
        transformed.query_ids_by_task_split[task][split].numel() == 2
        for task in range(3)
        for split in ("train", "val", "test")
    )
