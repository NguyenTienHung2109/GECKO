"""Partition diagnostics persisted with every UEFA stream."""

from __future__ import annotations

import math
from typing import Any
from typing import Dict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import torch

from gecko.types import ScenarioSpec


def _js_divergence(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.double()
    second = second.double()
    first = first / first.sum().clamp_min(1)
    second = second / second.sum().clamp_min(1)
    midpoint = 0.5 * (first + second)

    def kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        valid = left > 0
        return (left[valid] * (left[valid] / right[valid]).log()).sum()

    return float(0.5 * kl(first, midpoint) + 0.5 * kl(second, midpoint))


def _distribution_summary(values: torch.Tensor) -> Dict[str, float | int]:
    values = values.detach().cpu().double().reshape(-1)
    if values.numel() == 0:
        return {
            "count": 0,
            "min": 0,
            "p10": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0,
        }
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "p10": float(torch.quantile(values, 0.10)),
        "median": float(torch.median(values)),
        "p90": float(torch.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def _query_owners(spec: ScenarioSpec, owner: torch.Tensor) -> torch.Tensor:
    if spec.problem_type == "NC":
        return owner
    assert spec.query_endpoints is not None
    source_owner = owner[spec.query_endpoints[:, 0]]
    target_owner = owner[spec.query_endpoints[:, 1]]
    return torch.where(
        source_owner == target_owner,
        source_owner,
        torch.full_like(source_owner, -1),
    )


def _categorical_divergence(
    values: torch.Tensor,
    value_owner: torch.Tensor,
    num_clients: int,
) -> list[float | None]:
    valid = value_owner >= 0
    if values.ndim != 1 or not valid.any():
        return [None for _ in range(num_clients)]
    values = values[valid].long()
    owners = value_owner[valid]
    minimum = int(values.min())
    shifted = values - minimum
    width = int(shifted.max()) + 1
    global_counts = torch.bincount(shifted, minlength=width)
    divergences: list[float | None] = []
    for client in range(num_clients):
        local = shifted[owners == client]
        if local.numel() == 0:
            divergences.append(None)
        else:
            divergences.append(
                _js_divergence(torch.bincount(local, minlength=width), global_counts)
            )
    return divergences


def _degree_divergence(
    degree: torch.Tensor,
    owner: torch.Tensor,
    num_clients: int,
) -> list[float | None]:
    maximum = max(1.0, float(degree.max()))
    global_hist = torch.histc(degree, bins=16, min=0, max=maximum)
    values: list[float | None] = []
    for client in range(num_clients):
        local = degree[owner == client]
        values.append(
            None
            if local.numel() == 0
            else _js_divergence(
                torch.histc(local, bins=16, min=0, max=maximum), global_hist
            )
        )
    return values


_HOMOPHILY_CHUNK_SIZE = 65_536


def _homophily_values(
    labels: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    source_labels = labels[source]
    target_labels = labels[target]
    if labels.ndim == 1:
        return (source_labels == target_labels).to(torch.float64)
    valid = (source_labels >= 0) & (target_labels >= 0)
    agreement = ((source_labels > 0) == (target_labels > 0)) & valid
    return agreement.sum(dim=1, dtype=torch.float64) / valid.sum(
        dim=1, dtype=torch.float64
    ).clamp_min(1)


def _local_homophily(
    spec: ScenarioSpec,
    owner: torch.Tensor,
    num_clients: int,
) -> list[float | None]:
    if spec.problem_type != "NC":
        return [None for _ in range(num_clients)]
    source, target = spec.edge_index
    totals = torch.zeros(num_clients, dtype=torch.float64)
    counts = torch.zeros(num_clients, dtype=torch.long)
    for start in range(0, source.numel(), _HOMOPHILY_CHUNK_SIZE):
        end = min(start + _HOMOPHILY_CHUNK_SIZE, source.numel())
        chunk_source = source[start:end]
        chunk_target = target[start:end]
        source_owner = owner[chunk_source]
        internal = source_owner == owner[chunk_target]
        if not internal.any():
            continue
        values = _homophily_values(spec.labels, chunk_source, chunk_target)
        clients = source_owner[internal]
        totals.index_add_(0, clients, values[internal])
        counts.index_add_(0, clients, torch.ones_like(clients, dtype=torch.long))
    return [
        None if count == 0 else float(totals[client] / count)
        for client, count in enumerate(counts.tolist())
    ]


def _global_homophily(spec: ScenarioSpec) -> float | None:
    if spec.problem_type != "NC" or spec.edge_index.shape[1] == 0:
        return None
    source, target = spec.edge_index
    total = 0.0
    count = 0
    for start in range(0, source.numel(), _HOMOPHILY_CHUNK_SIZE):
        end = min(start + _HOMOPHILY_CHUNK_SIZE, source.numel())
        values = _homophily_values(spec.labels, source[start:end], target[start:end])
        total += float(values.sum())
        count += values.numel()
    return total / max(1, count)


def _components(edge_index: torch.Tensor, num_nodes: int) -> tuple[int, float, float]:
    if num_nodes == 0:
        return 0, 0.0, 0.0
    if edge_index.shape[1] == 0:
        return num_nodes, 1.0 / num_nodes, 1.0
    source = edge_index[0].detach().cpu().numpy()
    target = edge_index[1].detach().cpu().numpy()
    adjacency = coo_matrix(
        (np.ones(source.shape[0], dtype=np.uint8), (source, target)),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    component_count, labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    component_sizes = np.bincount(labels, minlength=component_count)
    source_degree = np.bincount(source, minlength=num_nodes)
    target_degree = np.bincount(target, minlength=num_nodes)
    largest = int(component_sizes.max()) / num_nodes
    isolated = int(np.count_nonzero((source_degree + target_degree) == 0)) / num_nodes
    return int(component_count), largest, isolated


def partition_diagnostics(
    spec: ScenarioSpec,
    owner: torch.Tensor,
    support: torch.Tensor,
    num_clients: int,
) -> Dict[str, Any]:
    source_owner = owner[spec.edge_index[0]]
    target_owner = owner[spec.edge_index[1]]
    cut = source_owner != target_owner
    boundary = torch.zeros(owner.shape[0], dtype=torch.bool)
    if cut.any():
        boundary[spec.edge_index[0, cut]] = True
        boundary[spec.edge_index[1, cut]] = True
    node_counts = torch.bincount(owner, minlength=num_clients)
    internal_edges = torch.zeros(num_clients, dtype=torch.long)
    components, largest_ratios, isolated_ratios = [], [], []
    global_degree = torch.bincount(spec.edge_index[0], minlength=owner.shape[0]).float()
    retained_degree = torch.zeros_like(global_degree)
    for client in range(num_clients):
        mask = (source_owner == client) & (target_owner == client)
        internal_edges[client] = int(mask.sum())
        retained_degree.index_add_(
            0,
            spec.edge_index[0, mask],
            torch.ones(int(mask.sum()), dtype=retained_degree.dtype),
        )
        nodes = torch.nonzero(owner == client, as_tuple=True)[0]
        mapping = torch.full((owner.shape[0],), -1, dtype=torch.long)
        mapping[nodes] = torch.arange(nodes.shape[0])
        local_edges = mapping[spec.edge_index[:, mask]]
        component_count, largest, isolated = _components(local_edges, nodes.shape[0])
        components.append(component_count)
        largest_ratios.append(largest)
        isolated_ratios.append(isolated)
    degree_divergence = _degree_divergence(global_degree, owner, num_clients)
    query_totals = support.sum(dim=(1, 2)).float()
    query_cv = float(query_totals.std(unbiased=False) / query_totals.mean().clamp_min(1))
    query_owner = _query_owners(spec, owner)
    all_query_ids = torch.cat(
        [
            indices
            for splits in spec.query_ids_by_task_split.values()
            for indices in splits.values()
        ]
    )
    active_query_ids_by_split = {
        split: torch.cat(
            [
                spec.query_ids_by_task_split[task][split]
                for task in range(spec.num_tasks)
            ]
        )
        for split in ("train", "val", "test")
    }

    def active_divergence(
        values: torch.Tensor | None, query_ids: torch.Tensor
    ) -> list[float | None]:
        if values is None:
            return [None for _ in range(num_clients)]
        return _categorical_divergence(
            values[query_ids], query_owner[query_ids], num_clients
        )

    internal_query_count = int((query_owner[all_query_ids] >= 0).sum())
    total_query_count = int(all_query_ids.numel())
    active_task_js = active_divergence(spec.query_task_ids, all_query_ids)
    active_label_js = active_divergence(spec.labels, all_query_ids)
    active_domain_js = active_divergence(spec.domains, all_query_ids)
    active_task_js_by_split = {
        split: active_divergence(spec.query_task_ids, query_ids)
        for split, query_ids in active_query_ids_by_split.items()
    }
    active_label_js_by_split = {
        split: active_divergence(spec.labels, query_ids)
        for split, query_ids in active_query_ids_by_split.items()
    }
    active_domain_js_by_split = {
        split: active_divergence(spec.domains, query_ids)
        for split, query_ids in active_query_ids_by_split.items()
    }
    # Preserve these historical fields exactly. They summarize the complete
    # backing tensors and therefore include query rows masked out of an active
    # stream. The active fields below must be used for stream manipulation
    # checks, especially for selected-query LC streams.
    task_js = _categorical_divergence(
        spec.query_task_ids, query_owner, num_clients
    )
    label_js = _categorical_divergence(spec.labels, query_owner, num_clients)
    domain_js = (
        [None for _ in range(num_clients)]
        if spec.domains is None
        else _categorical_divergence(spec.domains, query_owner, num_clients)
    )
    local_homophily = _local_homophily(spec, owner, num_clients)
    global_homophily = _global_homophily(spec)
    diagnostics: Dict[str, Any] = {
        "node_count_per_client": node_counts.tolist(),
        "internal_edge_count_per_client": internal_edges.tolist(),
        "edge_cut_ratio": float(cut.float().mean()) if cut.numel() else 0.0,
        "boundary_node_ratio": float(boundary.float().mean()),
        "connected_component_count_per_client": components,
        "largest_component_ratio_per_client": largest_ratios,
        "isolated_node_ratio_per_client": isolated_ratios,
        "degree_distribution_divergence": degree_divergence,
        "global_homophily": global_homophily,
        "local_homophily": local_homophily,
        "homophily_drift_per_client": [
            None
            if value is None or global_homophily is None
            else value - global_homophily
            for value in local_homophily
        ],
        "label_js_divergence": label_js,
        "domain_js_divergence": domain_js,
        "query_js_divergence": task_js,
        "legacy_categorical_divergence_semantics": (
            "complete_backing_tensor_internal_queries_including_inactive_rows"
        ),
        "active_categorical_divergence_semantics": (
            "query_ids_present_in_query_ids_by_task_split_internal_queries_only"
        ),
        "active_label_js_divergence": active_label_js,
        "active_domain_js_divergence": active_domain_js,
        "active_query_js_divergence": active_task_js,
        "active_label_js_divergence_by_split": active_label_js_by_split,
        "active_domain_js_divergence_by_split": active_domain_js_by_split,
        "active_query_js_divergence_by_split": active_task_js_by_split,
        "active_train_label_js_divergence": active_label_js_by_split["train"],
        "active_train_domain_js_divergence": active_domain_js_by_split["train"],
        "active_train_query_js_divergence": active_task_js_by_split["train"],
        "client_query_volume_coefficient_of_variation": query_cv,
        "client_task_support_matrix": support.tolist(),
        "client_task_support_summary": {
            split: _distribution_summary(support[:, :, split_index])
            for split_index, split in enumerate(("train", "val", "test"))
        },
        "internal_query_coverage": internal_query_count / max(1, total_query_count),
        "retained_degree_ratio_summary": _distribution_summary(
            retained_degree / global_degree.clamp_min(1)
        ),
        "retained_degree_ratio_per_client": [
            _distribution_summary(
                (retained_degree / global_degree.clamp_min(1))[owner == client]
            )
            for client in range(num_clients)
        ],
    }
    if spec.problem_type == "LC":
        diagnostics["lc_internal_query_coverage"] = diagnostics["internal_query_coverage"]
        diagnostics["lc_excluded_cross_client_query_count"] = int(
            (query_owner < 0).sum()
        )
        class_support = []
        for client in range(num_clients):
            labels = spec.labels[query_owner == client].long()
            labels = labels[labels >= 0]
            class_support.append(
                torch.bincount(labels, minlength=spec.num_classes).tolist()
            )
        diagnostics["lc_per_client_edge_class_support"] = class_support
        all_labels = spec.labels[spec.labels >= 0].long()
        internal_labels = spec.labels[query_owner >= 0].long()
        internal_labels = internal_labels[internal_labels >= 0]
        all_histogram = torch.bincount(all_labels, minlength=spec.num_classes)
        internal_histogram = torch.bincount(
            internal_labels, minlength=spec.num_classes
        )
        diagnostics["lc_internal_vs_all_label_js_divergence"] = _js_divergence(
            internal_histogram, all_histogram
        )
        diagnostics["lc_query_coverage_by_task"] = []
        for task_id in range(spec.num_tasks):
            selected = spec.query_task_ids == task_id
            diagnostics["lc_query_coverage_by_task"].append(
                float((query_owner[selected] >= 0).sum() / selected.sum().clamp_min(1))
            )
        diagnostics["lc_query_coverage_by_class"] = []
        for class_id in range(spec.num_classes):
            selected = spec.labels == class_id
            diagnostics["lc_query_coverage_by_class"].append(
                float((query_owner[selected] >= 0).sum() / selected.sum().clamp_min(1))
            )
    if spec.problem_type == "LP":
        positive = spec.labels == 1
        positive_total = int(positive.sum())
        internal_positive = int(((query_owner >= 0) & positive).sum())
        if spec.metadata.get("partition_first_release_protocol", False):
            realized_positive_coverage = float(
                spec.metadata["partition_internal_positive_coverage"]
            )
            diagnostics["lp_internal_positive_coverage"] = realized_positive_coverage
            diagnostics["lp_cross_client_positive_rate"] = (
                1.0 - realized_positive_coverage
            )
            diagnostics["lp_query_positive_pool_count"] = int(
                spec.metadata["partition_query_positive_pool_count"]
            )
            diagnostics["lp_internal_query_positive_count"] = int(
                spec.metadata["partition_internal_positive_count"]
            )
        else:
            diagnostics["lp_internal_positive_coverage"] = (
                internal_positive / max(1, positive_total)
            )
            diagnostics["lp_cross_client_positive_rate"] = float(
                ((query_owner < 0) & positive).sum() / max(1, positive_total)
            )
        diagnostics["lp_internal_candidate_coverage"] = diagnostics["internal_query_coverage"]
        diagnostics["lp_false_negative_validation_count"] = 0
        candidate_counts = []
        negative_counts = []
        for client in range(num_clients):
            for task_id in range(spec.num_tasks):
                for split in ("train", "val", "test"):
                    query_ids = spec.query_ids_by_task_split[task_id][split]
                    internal = query_ids[query_owner[query_ids] == client]
                    candidate_counts.append(int(internal.numel()))
                    negative_counts.append(int((spec.labels[internal] == 0).sum()))
        diagnostics["lp_candidate_count_summary"] = _distribution_summary(
            torch.tensor(candidate_counts)
        )
        diagnostics["lp_negative_candidate_count_summary"] = _distribution_summary(
            torch.tensor(negative_counts)
        )
    return diagnostics
