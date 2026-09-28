from __future__ import annotations

from typing import Dict
from typing import Tuple
import torch
from gecko.types import ScenarioSpec
from gecko.data.splits.common import task_split_indices

def build_nc_spec(
    *,
    dataset_name: str,
    incremental_type: str,
    metrics: Tuple[str, ...],
    edge_index: torch.Tensor,
    node_features: torch.Tensor,
    labels: torch.Tensor,
    task_ids: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_tasks: int,
    num_classes: int,
    task_class_sets: Dict[int, torch.Tensor] | None = None,
    domains: torch.Tensor | None = None,
    metadata: Dict[str, object] | None = None,
) -> ScenarioSpec:
    query_splits = task_split_indices(
        task_ids, train_mask, validation_mask, test_mask, num_tasks
    )
    task_classes = task_class_sets or {}
    task_masks = None
    if task_classes:
        task_masks = torch.zeros(num_tasks, num_classes, dtype=torch.bool)
        for task_id, class_ids in task_classes.items():
            task_masks[task_id, class_ids.long()] = True
    # ScenarioSpec is immutable; one shared CPU edge tensor is sufficient for
    # both the base and context topology.  Cloning both fields doubles more
    # than a GiB on OGBN-Proteins without adding isolation.
    frozen_edges = edge_index.detach().cpu().long().contiguous()
    spec = ScenarioSpec(
        dataset_name=dataset_name,
        problem_type="NC",
        incremental_type=incremental_type,
        num_tasks=num_tasks,
        num_classes=num_classes,
        num_features=int(node_features.shape[-1]),
        metrics=metrics,
        edge_index=frozen_edges,
        node_features=node_features.detach().cpu().clone(),
        query_ids_by_task_split=query_splits,
        query_task_ids=task_ids.detach().cpu().long().clone(),
        labels=labels.detach().cpu().clone(),
        task_class_sets={key: value.detach().cpu().long().clone() for key, value in task_classes.items()},
        domains=None if domains is None else domains.detach().cpu().long().clone(),
        task_masks=task_masks,
        context_edge_index=frozen_edges,
        metadata={
            "feature_provenance": "provided_node_attributes",
            **dict(metadata or {}),
        },
    )
    spec.validate()
    return spec




from typing import Dict
from typing import Tuple
import torch
from gecko.types import ScenarioSpec
from gecko.data.splits.common import canonicalize_logical_edges
from gecko.data.splits.common import task_split_indices

def build_lc_spec(
    *,
    dataset_name: str,
    incremental_type: str,
    metrics: Tuple[str, ...],
    edge_index: torch.Tensor,
    node_features: torch.Tensor,
    raw_labels: torch.Tensor,
    raw_task_ids: torch.Tensor,
    raw_train_mask: torch.Tensor,
    raw_validation_mask: torch.Tensor,
    raw_test_mask: torch.Tensor,
    num_tasks: int,
    num_classes: int,
    undirected: bool,
    task_class_sets: Dict[int, torch.Tensor] | None = None,
    domains: torch.Tensor | None = None,
    metadata: Dict[str, object] | None = None,
) -> ScenarioSpec:
    logical_edges, raw_to_logical, logical_to_raw = canonicalize_logical_edges(
        edge_index, undirected=undirected
    )
    logical_count = logical_edges.shape[0]
    labels = torch.empty(logical_count, dtype=raw_labels.dtype)
    task_ids = torch.empty(logical_count, dtype=torch.long)
    train_mask = torch.zeros(logical_count, dtype=torch.bool)
    validation_mask = torch.zeros(logical_count, dtype=torch.bool)
    test_mask = torch.zeros(logical_count, dtype=torch.bool)
    logical_domains = None if domains is None else torch.empty(logical_count, dtype=torch.long)
    for logical_id, raw_ids in logical_to_raw.items():
        first = int(raw_ids[0])
        if not torch.equal(raw_labels[raw_ids], raw_labels[first].expand_as(raw_labels[raw_ids])):
            raise ValueError(f"Reverse arcs disagree on label for logical edge {logical_id}.")
        if not torch.equal(raw_task_ids[raw_ids], raw_task_ids[first].expand_as(raw_task_ids[raw_ids])):
            raise ValueError(f"Reverse arcs disagree on task for logical edge {logical_id}.")
        split_values = torch.stack(
            [raw_train_mask[raw_ids], raw_validation_mask[raw_ids], raw_test_mask[raw_ids]], dim=1
        ).long()
        if not torch.equal(split_values, split_values[0].expand_as(split_values)):
            raise ValueError(f"Reverse arcs disagree on split for logical edge {logical_id}.")
        labels[logical_id] = raw_labels[first]
        task_ids[logical_id] = raw_task_ids[first]
        train_mask[logical_id] = raw_train_mask[first]
        validation_mask[logical_id] = raw_validation_mask[first]
        test_mask[logical_id] = raw_test_mask[first]
        if logical_domains is not None:
            logical_domains[logical_id] = domains[first]
    query_splits = task_split_indices(
        task_ids, train_mask, validation_mask, test_mask, num_tasks
    )
    task_classes = task_class_sets or {}
    task_masks = None
    if task_classes:
        task_masks = torch.zeros(num_tasks, num_classes, dtype=torch.bool)
        for task_id, class_ids in task_classes.items():
            task_masks[task_id, class_ids.long()] = True
    spec = ScenarioSpec(
        dataset_name=dataset_name,
        problem_type="LC",
        incremental_type=incremental_type,
        num_tasks=num_tasks,
        num_classes=num_classes,
        num_features=int(node_features.shape[-1]),
        metrics=metrics,
        edge_index=edge_index.detach().cpu().clone(),
        node_features=node_features.detach().cpu().clone(),
        query_ids_by_task_split=query_splits,
        query_task_ids=task_ids,
        labels=labels.detach().cpu().clone(),
        query_endpoints=logical_edges,
        task_class_sets={key: value.detach().cpu().long().clone() for key, value in task_classes.items()},
        domains=logical_domains,
        task_masks=task_masks,
        logical_edge_ids=raw_to_logical,
        logical_edge_to_raw_edges=logical_to_raw,
        context_edge_index=edge_index.detach().cpu().clone(),
        metadata={
            "feature_provenance": "provided_node_attributes",
            **dict(metadata or {}),
            "undirected": undirected,
        },
    )
    spec.validate()
    return spec




from typing import Dict
from typing import Tuple
import torch
from gecko.types import ScenarioSpec
from gecko.data.splits.common import canonicalize_logical_edges
from gecko.data.splits.common import validate_negative_pairs

def build_lp_spec(
    *,
    dataset_name: str,
    metrics: Tuple[str, ...],
    node_features: torch.Tensor,
    positive_pairs: torch.Tensor,
    positive_task_ids: torch.Tensor,
    positive_splits: torch.Tensor,
    negative_pairs_by_task_split: Dict[int, Dict[str, torch.Tensor]],
    candidate_group_ids_by_task_split: Dict[int, Dict[str, torch.Tensor]] | None = None,
    num_tasks: int,
    undirected: bool,
    known_positive_pairs: torch.Tensor | None = None,
    context_positive_pairs: torch.Tensor | None = None,
    context_edge_index: torch.Tensor | None = None,
    bipartite: bool = False,
    node_types: torch.Tensor | None = None,
    metadata: Dict[str, object] | None = None,
) -> ScenarioSpec:
    known_positives = (
        positive_pairs if known_positive_pairs is None else known_positive_pairs
    )
    positive_keys = {
        (min(source, target), max(source, target)) if undirected else (source, target)
        for source, target in known_positives.tolist()
    }
    query_endpoints: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    task_ids: list[torch.Tensor] = []
    candidate_group_ids: list[torch.Tensor] = []
    task_splits: Dict[int, Dict[str, torch.Tensor]] = {
        task: {} for task in range(num_tasks)
    }
    cursor = 0
    split_codes = {"train": 0, "val": 1, "test": 2}
    for task in range(num_tasks):
        for split, split_code in split_codes.items():
            positive_mask = (positive_task_ids == task) & (positive_splits == split_code)
            positives = positive_pairs[positive_mask].long()
            negatives = negative_pairs_by_task_split[task][split].long()
            validate_negative_pairs(
                negatives,
                positive_keys,
                undirected=undirected,
                num_nodes=node_features.shape[0],
                node_types=node_types if bipartite else None,
            )
            endpoints = torch.cat([positives, negatives], dim=0)
            count = endpoints.shape[0]
            query_endpoints.append(endpoints)
            labels.append(
                torch.cat(
                    [torch.ones(positives.shape[0]), torch.zeros(negatives.shape[0])]
                ).long()
            )
            task_ids.append(torch.full((count,), task, dtype=torch.long))
            if candidate_group_ids_by_task_split is None:
                groups = torch.full(
                    (count,), task * len(split_codes) + split_code, dtype=torch.long
                )
            else:
                groups = candidate_group_ids_by_task_split[task][split].long()
                if groups.shape != (count,):
                    raise ValueError(
                        f"Candidate groups for task={task} split={split} do not align."
                    )
            candidate_group_ids.append(groups)
            task_splits[task][split] = torch.arange(cursor, cursor + count, dtype=torch.long)
            cursor += count
    all_queries = torch.cat(query_endpoints, dim=0)
    all_labels = torch.cat(labels, dim=0)
    all_task_ids = torch.cat(task_ids, dim=0)
    all_candidate_group_ids = torch.cat(candidate_group_ids, dim=0)
    train_positive_mask = positive_splits == 0
    if context_edge_index is not None:
        if context_edge_index.ndim != 2 or context_edge_index.shape[0] != 2:
            raise ValueError("context_edge_index must have shape [2, num_edges].")
        train_positive_pairs, _, _ = canonicalize_logical_edges(
            context_edge_index, undirected=undirected
        )
    else:
        train_positive_pairs = (
            positive_pairs[train_positive_mask]
            if context_positive_pairs is None
            else context_positive_pairs
        )
    for source, target in train_positive_pairs.tolist():
        key = (min(source, target), max(source, target)) if undirected else (source, target)
        if key not in positive_keys:
            raise ValueError("An LP context edge is absent from the known-positive universe.")
    if context_edge_index is not None:
        context_arcs = context_edge_index.detach().cpu().long().clone()
    else:
        context_arcs = train_positive_pairs.t().contiguous()
        if undirected:
            non_loops = context_arcs[0] != context_arcs[1]
            context_arcs = torch.cat(
                [context_arcs, context_arcs.flip(0)[:, non_loops]], dim=1
            )
    logical_context, raw_to_logical, logical_to_raw = canonicalize_logical_edges(
        context_arcs, undirected=undirected
    )
    held_out = positive_pairs[positive_splits != 0]
    context_keys = {
        (min(source, target), max(source, target)) if undirected else (source, target)
        for source, target in logical_context.tolist()
    }
    for source, target in held_out.tolist():
        key = (min(source, target), max(source, target)) if undirected else (source, target)
        if key in context_keys:
            raise ValueError("LP validation/test positive leaked into the context graph.")
    spec = ScenarioSpec(
        dataset_name=dataset_name,
        problem_type="LP",
        incremental_type="domain",
        num_tasks=num_tasks,
        num_classes=1,
        num_features=int(node_features.shape[-1]),
        metrics=metrics,
        edge_index=context_arcs,
        node_features=node_features.detach().cpu().clone(),
        query_ids_by_task_split=task_splits,
        query_task_ids=all_task_ids,
        labels=all_labels,
        query_endpoints=all_queries,
        domains=all_task_ids.clone(),
        logical_edge_ids=raw_to_logical,
        logical_edge_to_raw_edges=logical_to_raw,
        context_edge_index=context_arcs.clone(),
        bipartite=bipartite,
        node_types=None if node_types is None else node_types.detach().cpu().clone(),
        metadata={
            "feature_provenance": "provided_node_attributes",
            **dict(metadata or {}),
            "undirected": undirected,
            "positive_pairs": positive_pairs.detach().cpu().clone(),
            "known_positive_pairs": known_positives.detach().cpu().clone(),
            "positive_task_ids": positive_task_ids.detach().cpu().clone(),
            "positive_splits": positive_splits.detach().cpu().clone(),
            "candidate_group_ids": all_candidate_group_ids,
        },
    )
    spec.validate()
    return spec


