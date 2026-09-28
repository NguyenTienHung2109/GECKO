"""Pure tensor helpers used by legacy scenario `export_spec` methods."""

from __future__ import annotations

import math
from typing import Dict
from typing import Iterable
from typing import Tuple

import torch


def canonicalize_logical_edges(
    edge_index: torch.Tensor,
    *,
    undirected: bool,
) -> tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
    """Return canonical endpoints, raw-to-logical IDs, and reverse-arc groups."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_raw_edges].")
    groups: Dict[Tuple[int, int], list[int]] = {}
    for raw_id, (source, target) in enumerate(edge_index.t().tolist()):
        key = (min(source, target), max(source, target)) if undirected else (source, target)
        groups.setdefault(key, []).append(raw_id)
    ordered = sorted(groups)
    key_to_logical = {key: logical_id for logical_id, key in enumerate(ordered)}
    raw_to_logical = torch.empty(edge_index.shape[1], dtype=torch.long)
    logical_to_raw: Dict[int, torch.Tensor] = {}
    for key, raw_ids in groups.items():
        logical_id = key_to_logical[key]
        raw_tensor = torch.tensor(sorted(raw_ids), dtype=torch.long)
        logical_to_raw[logical_id] = raw_tensor
        raw_to_logical[raw_tensor] = logical_id
    logical_edges = torch.tensor(ordered, dtype=torch.long).reshape(-1, 2)
    return logical_edges, raw_to_logical, logical_to_raw


def task_split_indices(
    task_ids: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_tasks: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    split_masks = {
        "train": train_mask.bool(),
        "val": validation_mask.bool(),
        "test": test_mask.bool(),
    }
    return {
        task_id: {
            split: torch.nonzero(mask & (task_ids == task_id), as_tuple=True)[0].long()
            for split, mask in split_masks.items()
        }
        for task_id in range(num_tasks)
    }


def infer_undirected_reverse_arcs(edge_index: torch.Tensor) -> bool:
    """Conservatively detect whether every non-loop arc has its reverse."""

    arcs = {(int(source), int(target)) for source, target in edge_index.t().tolist()}
    return all(source == target or (target, source) in arcs for source, target in arcs)


def bitcoin_structural_degree_q4(
    logical_edges: torch.Tensor,
    training_mask: torch.Tensor,
    *,
    num_nodes: int | None = None,
) -> tuple[torch.Tensor, Dict[str, object]]:
    """Assign deterministic topology-only degree-quartile edge domains.

    Quantiles are fitted on training logical edges. Stable rank boundaries use
    logical-edge IDs to resolve ties and guarantee four non-empty training bins
    whenever at least four training edges are available.
    """

    if logical_edges.ndim != 2 or logical_edges.shape[1] != 2:
        raise ValueError("logical_edges must have shape [num_edges, 2].")
    if training_mask.shape != (logical_edges.shape[0],):
        raise ValueError("training_mask must align with logical edges.")
    if int(training_mask.sum()) < 4:
        raise ValueError("bitcoin_structural_degree_q4 needs at least four training edges.")
    inferred_nodes = int(logical_edges.max()) + 1 if logical_edges.numel() else 0
    node_count = inferred_nodes if num_nodes is None else num_nodes
    degree = torch.zeros(node_count, dtype=torch.float64)
    ones = torch.ones(logical_edges.shape[0], dtype=torch.float64)
    degree.index_add_(0, logical_edges[:, 0], ones)
    degree.index_add_(0, logical_edges[:, 1], ones)
    scores = torch.log1p(degree[logical_edges[:, 0]]) + torch.log1p(degree[logical_edges[:, 1]])

    train_ids = torch.nonzero(training_mask, as_tuple=True)[0]
    train_scores = scores[train_ids]
    quantiles = torch.quantile(train_scores, torch.tensor([0.25, 0.50, 0.75], dtype=torch.float64))
    domains = torch.bucketize(scores, quantiles, right=False).long()
    train_histogram = torch.bincount(domains[training_mask], minlength=4)
    fallback = bool((train_histogram == 0).any())
    rank_boundaries: list[tuple[float, int]] = []
    if fallback:
        ordered_train = sorted(
            train_ids.tolist(), key=lambda edge_id: (float(scores[edge_id]), int(edge_id))
        )
        for fraction in (0.25, 0.50, 0.75):
            boundary_position = max(0, min(len(ordered_train) - 1, math.ceil(len(ordered_train) * fraction) - 1))
            boundary_id = ordered_train[boundary_position]
            rank_boundaries.append((float(scores[boundary_id]), int(boundary_id)))
        domains = torch.zeros(logical_edges.shape[0], dtype=torch.long)
        for edge_id, score in enumerate(scores.tolist()):
            domains[edge_id] = sum((float(score), edge_id) > boundary for boundary in rank_boundaries)
        train_histogram = torch.bincount(domains[training_mask], minlength=4)

    audit: Dict[str, object] = {
        "constructor": "bitcoin_structural_degree_q4",
        "constructor_version": 1,
        "score": "log1p(total_degree_u)+log1p(total_degree_v)",
        "quantile_scores": [float(value) for value in quantiles.tolist()],
        "stable_rank_fallback": fallback,
        "rank_boundaries": rank_boundaries,
        "training_histogram": train_histogram.tolist(),
        "all_edge_histogram": torch.bincount(domains, minlength=4).tolist(),
        "scores": scores.float(),
    }
    return domains, audit


def validate_negative_pairs(
    negative_pairs: torch.Tensor,
    positive_pairs: Iterable[Tuple[int, int]],
    *,
    undirected: bool,
    num_nodes: int,
    node_types: torch.Tensor | None = None,
) -> None:
    positives = set(positive_pairs)
    for source, target in negative_pairs.tolist():
        if not 0 <= source < num_nodes or not 0 <= target < num_nodes:
            raise ValueError("A negative pair references a node outside the graph.")
        if source == target:
            raise ValueError("Self-pairs are not valid LP negatives.")
        key = (min(source, target), max(source, target)) if undirected else (source, target)
        if key in positives:
            raise ValueError(f"Negative pair {(source, target)} is a known positive.")
        if node_types is not None and node_types[source] == node_types[target]:
            raise ValueError("A negative pair violates the bipartite node-type constraint.")
