from __future__ import annotations

"""Train-only class selection and capacity-balanced task construction."""

from dataclasses import dataclass

import torch

from gecko.types import ScenarioSpec


BALANCED_CLASS_TASK_POLICY_VERSION = "drop_rarest_train_lpt_balanced_v1"


@dataclass(frozen=True)
class BalancedClassTaskPlan:
    """A deterministic task plan derived from training labels only."""

    class_groups: tuple[tuple[int, ...], ...]
    excluded_classes: tuple[int, ...]
    train_class_counts: tuple[int, ...]
    task_train_counts: tuple[int, ...]
    classes_per_task: int
    policy_version: str = BALANCED_CLASS_TASK_POLICY_VERSION


def build_balanced_class_task_plan(
    spec: ScenarioSpec,
    *,
    target_num_tasks: int,
) -> BalancedClassTaskPlan:
    """Drop the rarest train classes, then balance class mass across tasks."""

    if spec.problem_type not in {"NC", "LC"}:
        raise ValueError("Balanced class tasks support NC and LC only.")
    if spec.incremental_type not in {"task", "class"}:
        raise ValueError("Balanced class tasks require Task-IL or Class-IL.")
    if spec.labels.ndim != 1:
        raise ValueError("Balanced class tasks require single-label targets.")
    if target_num_tasks < 1 or target_num_tasks > spec.num_classes:
        raise ValueError("target_num_tasks must lie in [1, num_classes].")

    classes_per_task = spec.num_classes // target_num_tasks
    selected_count = target_num_tasks * classes_per_task
    train_ids = torch.cat(
        [
            spec.query_ids_by_task_split[task]["train"]
            for task in range(spec.num_tasks)
        ]
    ).detach().cpu().long()
    if int(torch.unique(train_ids).numel()) != int(train_ids.numel()):
        raise ValueError("Source train query IDs must be globally unique.")
    counts = torch.bincount(
        spec.labels[train_ids].detach().cpu().long(), minlength=spec.num_classes
    )
    drop_count = spec.num_classes - selected_count
    rarest_first = sorted(
        range(spec.num_classes),
        key=lambda class_id: (int(counts[class_id]), class_id),
    )
    excluded = tuple(sorted(rarest_first[:drop_count]))
    excluded_set = set(excluded)
    selected = [
        class_id
        for class_id in range(spec.num_classes)
        if class_id not in excluded_set
    ]

    groups: list[list[int]] = [[] for _ in range(target_num_tasks)]
    task_counts = [0] * target_num_tasks
    for class_id in sorted(selected, key=lambda value: (-int(counts[value]), value)):
        candidates = [
            task
            for task in range(target_num_tasks)
            if len(groups[task]) < classes_per_task
        ]
        task = min(
            candidates,
            key=lambda value: (task_counts[value], len(groups[value]), value),
        )
        groups[task].append(class_id)
        task_counts[task] += int(counts[class_id])
    if any(len(group) != classes_per_task for group in groups):
        raise AssertionError("Balanced class-task construction violated class capacity.")

    return BalancedClassTaskPlan(
        class_groups=tuple(tuple(group) for group in groups),
        excluded_classes=excluded,
        train_class_counts=tuple(int(value) for value in counts.tolist()),
        task_train_counts=tuple(task_counts),
        classes_per_task=classes_per_task,
    )


"""Apply frozen train-balanced class groups to NC or LC scenarios."""

from typing import Dict

import torch

from gecko.types import ScenarioSpec


BALANCED_CLASS_TRANSFORM_VERSION = "balanced_class_task_transform_v1"


def apply_balanced_class_task_plan(
    spec: ScenarioSpec,
    plan: BalancedClassTaskPlan,
) -> ScenarioSpec:
    """Apply a train-only plan without resampling any train/val/test query."""

    if spec.problem_type not in {"NC", "LC"}:
        raise ValueError("Balanced class transform supports NC and LC only.")
    if spec.incremental_type not in {"task", "class"}:
        raise ValueError("Balanced class transform requires Task-IL or Class-IL.")
    groups = plan.class_groups
    flattened = tuple(class_id for group in groups for class_id in group)
    if not groups or any(not group for group in groups):
        raise ValueError("Balanced class groups must be nonempty.")
    if len(set(flattened)) != len(flattened):
        raise ValueError("Every selected class must occur exactly once.")
    if min(flattened) < 0 or max(flattened) >= spec.num_classes:
        raise ValueError("A selected class is outside the source class range.")

    class_remap = {original: remapped for remapped, original in enumerate(flattened)}
    labels = torch.full_like(spec.labels, -1)
    query_task_ids = torch.full_like(spec.query_task_ids, -1)
    for original, remapped in class_remap.items():
        selected = spec.labels == original
        labels[selected] = remapped
    for task, group in enumerate(groups):
        for original in group:
            query_task_ids[spec.labels == original] = task

    split_universes: Dict[str, torch.Tensor] = {}
    for split in ("train", "val", "test"):
        source_ids = torch.cat(
            [
                spec.query_ids_by_task_split[task][split]
                for task in range(spec.num_tasks)
            ]
        ).detach().cpu().long()
        if int(torch.unique(source_ids).numel()) != int(source_ids.numel()):
            raise ValueError(f"Source {split} query IDs must be globally unique.")
        split_universes[split] = source_ids.sort().values

    query_splits: Dict[int, Dict[str, torch.Tensor]] = {}
    remapped_task_classes: Dict[int, torch.Tensor] = {}
    for task, group in enumerate(groups):
        originals = torch.tensor(group, dtype=torch.long)
        remapped_task_classes[task] = torch.tensor(
            [class_remap[original] for original in group], dtype=torch.long
        )
        query_splits[task] = {}
        for split, source_ids in split_universes.items():
            keep = torch.isin(spec.labels[source_ids], originals)
            query_splits[task][split] = source_ids[keep].clone()

    task_masks = torch.zeros(len(groups), len(flattened), dtype=torch.bool)
    for task, class_ids in remapped_task_classes.items():
        task_masks[task, class_ids] = True
    transformed = ScenarioSpec(
        dataset_name=spec.dataset_name,
        problem_type=spec.problem_type,
        incremental_type=spec.incremental_type,
        num_tasks=len(groups),
        num_classes=len(flattened),
        num_features=spec.num_features,
        metrics=spec.metrics,
        edge_index=spec.edge_index.detach().cpu().clone(),
        node_features=spec.node_features.detach().cpu().clone(),
        query_ids_by_task_split=query_splits,
        query_task_ids=query_task_ids.detach().cpu().long().clone(),
        labels=labels.detach().cpu().long().clone(),
        query_endpoints=(
            None
            if spec.query_endpoints is None
            else spec.query_endpoints.detach().cpu().long().clone()
        ),
        task_class_sets=remapped_task_classes,
        domains=None if spec.domains is None else spec.domains.detach().cpu().clone(),
        task_masks=task_masks,
        logical_edge_ids=(
            None
            if spec.logical_edge_ids is None
            else spec.logical_edge_ids.detach().cpu().long().clone()
        ),
        logical_edge_to_raw_edges={
            key: value.detach().cpu().long().clone()
            for key, value in spec.logical_edge_to_raw_edges.items()
        },
        context_edge_index=(
            None
            if spec.context_edge_index is None
            else spec.context_edge_index.detach().cpu().clone()
        ),
        bipartite=spec.bipartite,
        node_types=(
            None if spec.node_types is None else spec.node_types.detach().cpu().clone()
        ),
        metadata={
            **spec.metadata,
            "class_task_transform_version": BALANCED_CLASS_TRANSFORM_VERSION,
            "class_task_selection_policy": plan.policy_version,
            "source_num_tasks": spec.num_tasks,
            "source_num_classes": spec.num_classes,
            "original_class_groups": [list(group) for group in groups],
            "remapped_class_groups": [
                value.tolist() for value in remapped_task_classes.values()
            ],
            "original_to_remapped_class": {
                str(original): remapped for original, remapped in class_remap.items()
            },
            "excluded_original_classes": list(plan.excluded_classes),
            "train_class_counts": list(plan.train_class_counts),
            "task_train_counts": list(plan.task_train_counts),
            "class_selection_label_access": "train_labels_only",
            "excluded_queries_are_topology_only_context": True,
        },
    )
    transformed.validate()
    return transformed


from dataclasses import replace
from typing import Any
import torch
from gecko.types import ScenarioSpec

def scenario_with_selected_training_queries(
    spec: ScenarioSpec,
    result: Any,
) -> ScenarioSpec:
    """Mask excess LC train labels by retaining only frozen selected query IDs."""
    from gecko.data.partitioning.base import _result_versions

    if spec.problem_type != "LC" or spec.query_endpoints is None:
        raise ValueError("Selected-query stream materialization requires an LC scenario.")
    selected = result.selected_query_ids.detach().cpu().long()
    if selected.numel() != torch.unique(selected).numel():
        raise ValueError("Selected LC training query IDs must be unique.")

    original_train = torch.cat(
        [spec.query_ids_by_task_split[task]["train"] for task in range(spec.num_tasks)]
    ).detach().cpu().long()
    selected_membership = torch.zeros(spec.labels.shape[0], dtype=torch.bool)
    selected_membership[selected] = True
    original_membership = torch.zeros_like(selected_membership)
    original_membership[original_train] = True
    if bool((selected_membership & ~original_membership).any()):
        raise ValueError("A selected LC query is not in the original train split.")

    split_map: dict[int, dict[str, torch.Tensor]] = {}
    realized_ids: list[torch.Tensor] = []
    for task in range(spec.num_tasks):
        source = spec.query_ids_by_task_split[task]
        train = source["train"].detach().cpu().long()
        retained = train[selected_membership[train]].clone()
        if retained.numel() and bool((spec.query_task_ids[retained] != task).any()):
            raise ValueError("Selected LC query IDs do not match immutable task IDs.")
        split_map[task] = {
            "train": retained,
            "val": source["val"].detach().cpu().long().clone(),
            "test": source["test"].detach().cpu().long().clone(),
        }
        realized_ids.append(retained)
    realized = torch.sort(torch.cat(realized_ids)).values
    if not torch.equal(realized, torch.sort(selected).values):
        raise AssertionError("Selected LC train split materialization lost query IDs.")

    partition_mode, stream_version = _result_versions(result)
    scenario = replace(
        spec,
        query_ids_by_task_split=split_map,
        metadata={
            **spec.metadata,
            "lc_partition_mode": partition_mode,
            "lc_stream_version": stream_version,
            "lc_quota_hash": result.quota.quota_hash,
            "lc_ownership_hash": result.diagnostics["ownership_hash"],
            "lc_selected_query_hash": result.diagnostics["selected_query_hash"],
            "lc_selected_query_count": int(selected.numel()),
            "lc_unused_internal_label_policy": "central_only_masked_unless_selected",
        },
    )
    scenario.validate()
    return scenario


