from __future__ import annotations

from typing import Dict
from typing import Tuple
import torch
from gecko.types import ScenarioSpec

NC_CLASS_TASK_TRANSFORM_VERSION = "nc_class_subset_tasks_v1"


def select_nc_class_tasks(
    spec: ScenarioSpec,
    class_groups: Tuple[Tuple[int, ...], ...],
) -> ScenarioSpec:
    """Return an NC-Class view containing exactly the requested class tasks.

    The static graph and node features are retained. Selected original class
    IDs are remapped contiguously in group order; nodes from excluded classes
    remain topology-only context with label and task sentinels of ``-1``.
    Existing train/validation/test memberships are preserved for every
    selected query, so this transform never resamples the benchmark split.
    """

    if spec.problem_type.upper() != "NC" or spec.incremental_type.lower() != "class":
        raise ValueError("Class-task selection requires an NC-Class ScenarioSpec.")
    groups = tuple(tuple(int(class_id) for class_id in group) for group in class_groups)
    if not groups or any(not group for group in groups):
        raise ValueError("class_groups must contain nonempty tasks.")
    flattened = tuple(class_id for group in groups for class_id in group)
    if len(set(flattened)) != len(flattened):
        raise ValueError("Each selected class must appear in exactly one task.")
    if min(flattened) < 0 or max(flattened) >= spec.num_classes:
        raise ValueError("A selected class is outside the source class range.")

    class_remap = {original: remapped for remapped, original in enumerate(flattened)}
    labels = torch.full_like(spec.labels, -1)
    query_task_ids = torch.full_like(spec.query_task_ids, -1)
    for original, remapped in class_remap.items():
        selected_nodes = spec.labels == original
        labels[selected_nodes] = remapped
    for task_id, group in enumerate(groups):
        for original in group:
            query_task_ids[spec.labels == original] = task_id

    split_universes: Dict[str, torch.Tensor] = {}
    for split in ("train", "val", "test"):
        source_ids = torch.cat(
            [
                spec.query_ids_by_task_split[task_id][split]
                for task_id in range(spec.num_tasks)
            ]
        ).detach().cpu().long()
        if int(torch.unique(source_ids).numel()) != int(source_ids.numel()):
            raise ValueError(f"Source NC {split} query IDs are not globally unique.")
        split_universes[split] = source_ids.sort().values

    query_splits: Dict[int, Dict[str, torch.Tensor]] = {}
    remapped_task_classes: Dict[int, torch.Tensor] = {}
    for task_id, group in enumerate(groups):
        original_classes = torch.tensor(group, dtype=torch.long)
        remapped_task_classes[task_id] = torch.tensor(
            [class_remap[original] for original in group], dtype=torch.long
        )
        query_splits[task_id] = {}
        for split, source_ids in split_universes.items():
            keep = torch.isin(spec.labels[source_ids], original_classes)
            query_splits[task_id][split] = source_ids[keep].clone()

    task_masks = torch.zeros(len(groups), len(flattened), dtype=torch.bool)
    for task_id, class_ids in remapped_task_classes.items():
        task_masks[task_id, class_ids] = True
    selected_set = set(flattened)
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
        query_endpoints=None,
        task_class_sets=remapped_task_classes,
        domains=None if spec.domains is None else spec.domains.detach().cpu().clone(),
        task_masks=task_masks,
        logical_edge_ids=None,
        logical_edge_to_raw_edges={},
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
            "class_task_transform_version": NC_CLASS_TASK_TRANSFORM_VERSION,
            "source_num_tasks": spec.num_tasks,
            "source_num_classes": spec.num_classes,
            "original_class_groups": [list(group) for group in groups],
            "remapped_class_groups": [
                class_ids.tolist() for class_ids in remapped_task_classes.values()
            ],
            "original_to_remapped_class": {
                str(original): remapped for original, remapped in class_remap.items()
            },
            "excluded_original_classes": [
                class_id
                for class_id in range(spec.num_classes)
                if class_id not in selected_set
            ],
            "excluded_nodes_are_topology_only_context": True,
        },
    )
    transformed.validate()
    return transformed




_RELOCATED_EXPORTS = {'build_nc_spec': ('gecko.data.scenario', 'build_nc_spec')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)
