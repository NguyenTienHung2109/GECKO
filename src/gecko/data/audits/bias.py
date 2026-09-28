"""Read-only coverage, selection-bias, and LP construction provenance audit."""

from __future__ import annotations

from typing import Any

import torch

from gecko.data.audits.policy import AUDIT_POLICY
from gecko.data.audits.policy import AUDIT_POLICY_VERSION
from gecko.data.audits.policy import RELEASE_TIER_OVERRIDES
from gecko.data.audits.policy import SEVERE_QUALITY_WARNING_PREFIXES
from gecko.data.audits.policy import audit_policy_hash
from gecko.data.partitioning.diagnostics import _global_homophily
from gecko.data.partitioning.diagnostics import _local_homophily
from gecko.data.streams.builder import StreamBundle


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().cpu().double().reshape(-1)
    if values.numel() == 0:
        return {
            "count": 0,
            "min": 0.0,
            "p10": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "coefficient_of_variation": 0.0,
        }
    mean = values.mean()
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "p10": float(torch.quantile(values, 0.10)),
        "median": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
        "max": float(values.max()),
        "mean": float(mean),
        "coefficient_of_variation": float(
            values.std(unbiased=False) / mean.abs().clamp_min(1e-12)
        ),
    }


def _js_divergence(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.double() / first.double().sum().clamp_min(1)
    second = second.double() / second.double().sum().clamp_min(1)
    midpoint = 0.5 * (first + second)

    def kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        valid = left > 0
        return (left[valid] * (left[valid] / right[valid]).log()).sum()

    return float(0.5 * kl(first, midpoint) + 0.5 * kl(second, midpoint))


def _query_owner(bundle: StreamBundle) -> torch.Tensor:
    spec = bundle.scenario
    owner = bundle.partition.node_owner
    if spec.problem_type == "NC":
        return owner
    assert spec.query_endpoints is not None
    source = owner[spec.query_endpoints[:, 0]]
    target = owner[spec.query_endpoints[:, 1]]
    return torch.where(source == target, source, torch.full_like(source, -1))


def _degree(spec) -> torch.Tensor:
    return torch.bincount(
        torch.cat((spec.edge_index[0], spec.edge_index[1])),
        minlength=spec.node_features.shape[0],
    ).float()


def _endpoint_degree_summary(spec, ids: torch.Tensor) -> dict[str, float | int]:
    if spec.query_endpoints is None or ids.numel() == 0:
        return _summary(torch.empty(0))
    degree = _degree(spec)
    endpoints = spec.query_endpoints[ids]
    return _summary(0.5 * (degree[endpoints[:, 0]] + degree[endpoints[:, 1]]))


def _support_rows(bundle: StreamBundle, query_owner: torch.Tensor) -> list[dict[str, Any]]:
    spec = bundle.scenario
    rows: list[dict[str, Any]] = []
    for client in range(bundle.config.partition.num_clients):
        for task, splits in spec.query_ids_by_task_split.items():
            for split, ids in splits.items():
                internal = ids[query_owner[ids] == client]
                row: dict[str, Any] = {
                    "client": client,
                    "task": task,
                    "split": split,
                    "queries": int(internal.numel()),
                }
                if spec.problem_type == "LP":
                    row["positives"] = int((spec.labels[internal] == 1).sum())
                    row["negatives"] = int((spec.labels[internal] == 0).sum())
                elif spec.labels.ndim == 1:
                    row["class_support"] = torch.bincount(
                        spec.labels[internal].long(), minlength=spec.num_classes
                    ).tolist()
                if (
                    spec.domains is not None
                    and spec.domains.ndim == 1
                    and spec.domains.shape[0] == spec.labels.shape[0]
                ):
                    row["domain_support"] = {
                        str(int(domain)): int((spec.domains[internal] == domain).sum())
                        for domain in torch.unique(spec.domains).tolist()
                    }
                rows.append(row)
    return rows


def _coverage_rows(bundle: StreamBundle, query_owner: torch.Tensor) -> list[dict[str, Any]]:
    spec = bundle.scenario
    rows: list[dict[str, Any]] = []
    for task, splits in spec.query_ids_by_task_split.items():
        for split, ids in splits.items():
            class_values = (
                torch.unique(spec.labels[ids]).tolist()
                if spec.problem_type != "LP" and spec.labels.ndim == 1
                else [None]
            )
            for class_id in class_values:
                selected = ids if class_id is None else ids[spec.labels[ids] == class_id]
                internal = int((query_owner[selected] >= 0).sum())
                rows.append(
                    {
                        "task": task,
                        "split": split,
                        "class": class_id,
                        "total": int(selected.numel()),
                        "internal": internal,
                        "coverage": internal / max(1, int(selected.numel())),
                    }
                )
    return rows


def _client_class_coverage_rows(
    bundle: StreamBundle, query_owner: torch.Tensor
) -> list[dict[str, Any]]:
    spec = bundle.scenario
    if spec.problem_type != "LC" or spec.labels.ndim != 1:
        return []
    rows = []
    for task, splits in spec.query_ids_by_task_split.items():
        for split, ids in splits.items():
            for class_id in torch.unique(spec.labels[ids]).tolist():
                stratum = ids[spec.labels[ids] == class_id]
                total = int(stratum.numel())
                for client in range(bundle.config.partition.num_clients):
                    internal = int((query_owner[stratum] == client).sum())
                    rows.append(
                        {
                            "client": client,
                            "task": task,
                            "split": split,
                            "class": int(class_id),
                            "original_stratum_queries": total,
                            "client_internal_queries": internal,
                            "coverage": internal / max(1, total),
                        }
                    )
    return rows


def _lc_support_adequacy(
    bundle: StreamBundle,
    support_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    spec = bundle.scenario
    cells = []
    for row in support_rows:
        class_support = row.get("class_support")
        if class_support is None:
            continue
        global_ids = spec.query_ids_by_task_split[row["task"]][row["split"]]
        relevant = sorted(torch.unique(spec.labels[global_ids]).long().tolist())
        relevant_support = [int(class_support[class_id]) for class_id in relevant]
        cells.append(
            {
                "client": row["client"],
                "task": row["task"],
                "split": row["split"],
                "queries": row["queries"],
                "relevant_classes": relevant,
                "relevant_class_support": relevant_support,
                "present_relevant_classes": sum(value > 0 for value in relevant_support),
                "absent_relevant_classes": sum(value == 0 for value in relevant_support),
                "one_query_relevant_classes": sum(value == 1 for value in relevant_support),
                "macro_f1_cell_eligible": (
                    row["queries"] > 1 and all(value > 0 for value in relevant_support)
                ),
            }
        )
    summaries = {}
    for split in ("train", "val", "test"):
        selected = [row for row in cells if row["split"] == split]
        nonzero_class_support = [
            value
            for row in selected
            for value in row["relevant_class_support"]
            if value > 0
        ]
        summaries[split] = {
            "query_support": _summary(
                torch.tensor([row["queries"] for row in selected])
            ),
            "nonzero_class_support": _summary(torch.tensor(nonzero_class_support)),
            "cells": len(selected),
            "zero_query_cells": sum(row["queries"] == 0 for row in selected),
            "one_query_cells": sum(row["queries"] == 1 for row in selected),
            "cells_with_absent_relevant_class": sum(
                row["absent_relevant_classes"] > 0 for row in selected
            ),
            "macro_f1_eligible_cells": sum(
                row["macro_f1_cell_eligible"] for row in selected
            ),
        }
    return {"cells": cells, "summary_by_split": summaries}


def _lc_selection_diagnostics(
    bundle: StreamBundle,
    query_owner: torch.Tensor,
    internal_ids: torch.Tensor,
) -> dict[str, Any]:
    spec = bundle.scenario
    all_ids = torch.arange(spec.labels.shape[0])

    def categorical(values: torch.Tensor) -> dict[str, Any]:
        retained = values[values >= 0].long()
        internal = values[internal_ids].long()
        internal = internal[internal >= 0]
        width = int(retained.max()) + 1 if retained.numel() else 0
        all_histogram = torch.bincount(retained, minlength=width)
        internal_histogram = torch.bincount(
            internal, minlength=width
        )
        return {
            "all_histogram": all_histogram.tolist(),
            "internal_histogram": internal_histogram.tolist(),
            "js_divergence": _js_divergence(all_histogram, internal_histogram),
        }

    endpoints = spec.query_endpoints
    features = spec.node_features.float()
    source = features[endpoints[:, 0]]
    target = features[endpoints[:, 1]]
    cosine = torch.nn.functional.cosine_similarity(source, target, dim=1)
    communities = bundle.partition.micro_community_ids
    same_community = (
        communities[endpoints[:, 0]] == communities[endpoints[:, 1]]
    ).float()
    output = {
        "task_distribution": categorical(spec.query_task_ids),
        "endpoint_feature_cosine": {
            "all": _summary(cosine),
            "internal": _summary(cosine[internal_ids]),
        },
        "same_micro_community_rate": {
            "all": float(same_community.mean()),
            "internal": float(same_community[internal_ids].mean()),
        },
    }
    if spec.domains is not None and spec.domains.shape[0] == all_ids.shape[0]:
        output["domain_distribution"] = categorical(spec.domains)
    return output


def _support_summaries(support_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = {}
    for split in ("train", "val", "test"):
        selected = [row for row in support_rows if row["split"] == split]
        summaries[split] = {
            "queries": _summary(
                torch.tensor([row["queries"] for row in selected], dtype=torch.long)
            ),
            "positives": _summary(
                torch.tensor(
                    [row.get("positives", 0) for row in selected], dtype=torch.long
                )
            ),
        }
    return summaries


def _feature_drift(bundle: StreamBundle) -> dict[str, Any]:
    spec = bundle.scenario
    rows = []
    for client, graph in bundle.partition.client_graphs.items():
        reference = spec.node_features[graph.local_to_global]
        if reference.shape != graph.node_features.shape:
            return {"available": False, "reason": "feature_shapes_do_not_match"}
        difference = (graph.node_features.float() - reference.float()).reshape(-1)
        rows.append(
            {
                "client": client,
                "mean_absolute": float(difference.abs().mean()) if difference.numel() else 0.0,
                "root_mean_square": float(difference.square().mean().sqrt()) if difference.numel() else 0.0,
            }
        )
    return {
        "available": True,
        "reference": "scenario_node_features",
        "per_client": rows,
    }


def _nc_topology_recomputation(bundle: StreamBundle) -> dict[str, Any]:
    spec = bundle.scenario
    if spec.problem_type != "NC":
        return {}
    owner = bundle.partition.node_owner
    source, target = spec.edge_index
    internal = owner[source] == owner[target]
    global_degree = torch.bincount(source, minlength=owner.shape[0]).float()
    retained_degree = torch.bincount(
        source[internal], minlength=owner.shape[0]
    ).float()
    ratio = retained_degree / global_degree.clamp_min(1)

    global_homophily = _global_homophily(spec)
    local_homophily = _local_homophily(
        spec, owner, bundle.config.partition.num_clients
    )
    return {
        "retained_degree_ratio_summary": _summary(ratio),
        "retained_degree_ratio_per_client": [
            _summary(ratio[owner == client])
            for client in range(bundle.config.partition.num_clients)
        ],
        "global_homophily": global_homophily,
        "local_homophily": local_homophily,
        "homophily_drift_per_client": [
            None
            if value is None or global_homophily is None
            else value - global_homophily
            for value in local_homophily
        ],
    }


def _edge_attribute_selection(bundle: StreamBundle, internal_ids: torch.Tensor) -> dict[str, Any]:
    spec = bundle.scenario
    for name in ("edge_features", "scores"):
        values = spec.metadata.get(name)
        if torch.is_tensor(values) and values.shape[0] == spec.labels.shape[0]:
            return {
                "available": True,
                "attribute": name,
                "all": _summary(values.float()),
                "internal": _summary(values[internal_ids].float()),
            }
    return {"available": False, "reason": "no_query_aligned_edge_attribute"}


def _lp_group_audit(bundle: StreamBundle, query_owner: torch.Tensor) -> dict[str, Any]:
    spec = bundle.scenario
    group_ids = spec.metadata["candidate_group_ids"]
    group_rows = []
    negative_sizes = []
    negative_sizes_by_split: dict[str, list[int]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for client in range(bundle.config.partition.num_clients):
        for task, splits in spec.query_ids_by_task_split.items():
            for split, ids in splits.items():
                internal = ids[query_owner[ids] == client]
                for group in torch.unique(group_ids[internal]).tolist():
                    group_query_ids = internal[group_ids[internal] == group]
                    positives = int((spec.labels[group_query_ids] == 1).sum())
                    negatives = int((spec.labels[group_query_ids] == 0).sum())
                    negative_sizes.append(negatives)
                    negative_sizes_by_split[split].append(negatives)
                    group_rows.append(
                        {
                            "client": client,
                            "task": task,
                            "split": split,
                            "group": int(group),
                            "positives": positives,
                            "negatives": negatives,
                            "candidates": int(group_query_ids.numel()),
                        }
                    )
    all_ids = torch.arange(spec.labels.shape[0])
    internal_ids = all_ids[query_owner >= 0]
    negative_ids = internal_ids[spec.labels[internal_ids] == 0]
    degree = _degree(spec)
    endpoints = spec.query_endpoints[negative_ids]
    degree_product = (
        degree[endpoints[:, 0]] * degree[endpoints[:, 1]]
        if endpoints.numel()
        else torch.empty(0)
    )
    diagnostics = bundle.partition.diagnostics
    return {
        "group_rows": group_rows,
        "negative_count_per_group": _summary(torch.tensor(negative_sizes)),
        "negative_count_per_group_by_split": {
            split: _summary(torch.tensor(values))
            for split, values in negative_sizes_by_split.items()
        },
        "negative_degree_product_hardness": _summary(degree_product),
        "partition_provenance": {
            "information_scope": diagnostics.get("lp_partition_information_scope", "legacy_unspecified"),
            "uses_evaluation_positive_endpoints": diagnostics.get(
                "lp_partition_uses_evaluation_positive_endpoints", True
            ),
            "uses_evaluation_candidates": diagnostics.get(
                "lp_partition_uses_evaluation_candidates", True
            ),
            "evaluation_support_role": diagnostics.get(
                "lp_evaluation_support_role", "legacy_unspecified"
            ),
            "support_anchor_query_splits": diagnostics.get(
                "lp_support_anchor_query_splits", ["legacy_unspecified"]
            ),
            "support_anchor_community_count": diagnostics.get(
                "lp_support_anchor_community_count"
            ),
        },
    }


def audit_bias(bundle: StreamBundle) -> dict[str, Any]:
    """Return a deterministic report without mutating the stream."""

    spec = bundle.scenario
    owner = _query_owner(bundle)
    all_ids = torch.arange(spec.labels.shape[0])
    internal_ids = all_ids[owner >= 0]
    support_rows = _support_rows(bundle, owner)
    coverage_rows = _coverage_rows(bundle, owner)
    blockers: list[str] = []
    warnings: list[str] = []
    policy = AUDIT_POLICY
    for row in coverage_rows:
        if (
            row["total"] >= policy["minimum_group_size"]
            and row["coverage"] < policy["catastrophic_group_coverage"]
        ):
            blockers.append(
                "catastrophic_internal_query_underrepresentation:"
                f"task={row['task']},split={row['split']},class={row['class']}"
            )
        elif (
            row["total"] >= policy["minimum_group_size"]
            and row["coverage"] < policy["low_group_coverage_warning"]
        ):
            warnings.append(
                "low_internal_query_coverage:"
                f"task={row['task']},split={row['split']},class={row['class']}"
            )
    diagnostics = bundle.partition.diagnostics
    if diagnostics.get("edge_cut_ratio", 0.0) > policy["high_edge_cut_warning"]:
        warnings.append("high_edge_cut_ratio")
    if diagnostics.get("boundary_node_ratio", 0.0) > policy["high_boundary_node_warning"]:
        warnings.append("high_boundary_node_ratio")
    if any(row["queries"] <= 1 for row in support_rows):
        warnings.append("client_task_split_query_support_at_most_one")
    topology = {
        key: bundle.partition.diagnostics.get(key)
        for key in (
            "edge_cut_ratio",
            "boundary_node_ratio",
            "retained_degree_ratio_summary",
            "retained_degree_ratio_per_client",
            "connected_component_count_per_client",
            "largest_component_ratio_per_client",
            "isolated_node_ratio_per_client",
            "global_homophily",
            "local_homophily",
            "homophily_drift_per_client",
        )
    }
    topology.update(_nc_topology_recomputation(bundle))
    report: dict[str, Any] = {
        "audit_name": "uefa_coverage_bias_audit",
        "audit_version": AUDIT_POLICY_VERSION,
        "audit_policy_hash": audit_policy_hash(),
        "stream_id": bundle.stream_id,
        "policy": policy,
        "problem": spec.problem_type,
        "incremental_setting": spec.incremental_type,
        "topology": topology,
        "query_coverage": {
            "total": int(all_ids.numel()),
            "internal": int(internal_ids.numel()),
            "ratio": int(internal_ids.numel()) / max(1, int(all_ids.numel())),
            "rows_by_task_split_class": coverage_rows,
        },
        "support_rows": support_rows,
        "support_summary_by_split": _support_summaries(support_rows),
        "endpoint_degree": {
            "all_queries": _endpoint_degree_summary(spec, all_ids),
            "internal_queries": _endpoint_degree_summary(spec, internal_ids),
        },
        "feature_drift": _feature_drift(bundle),
        "edge_attribute_selection": _edge_attribute_selection(bundle, internal_ids),
    }
    if spec.problem_type == "LC" and spec.labels.ndim == 1:
        retained_labels = spec.labels[spec.labels >= 0].long()
        all_histogram = torch.bincount(
            retained_labels, minlength=spec.num_classes
        )
        internal_labels = spec.labels[internal_ids].long()
        internal_labels = internal_labels[internal_labels >= 0]
        internal_histogram = torch.bincount(
            internal_labels, minlength=spec.num_classes
        )
        report["label_selection_bias"] = {
            "all_histogram": all_histogram.tolist(),
            "internal_histogram": internal_histogram.tolist(),
            "js_divergence": _js_divergence(all_histogram, internal_histogram),
        }
        heatmap = _client_class_coverage_rows(bundle, owner)
        report["client_task_split_class_coverage"] = heatmap
        report["client_class_support_adequacy"] = {
            "zero_cells": sum(
                row["client_internal_queries"] == 0 for row in heatmap
            ),
            "one_query_cells": sum(
                row["client_internal_queries"] == 1 for row in heatmap
            ),
            "minimum_nonzero": min(
                (
                    row["client_internal_queries"]
                    for row in heatmap
                    if row["client_internal_queries"] > 0
                ),
                default=0,
            ),
        }
        report["lc_support_adequacy"] = _lc_support_adequacy(
            bundle, support_rows
        )
        report["lc_selection_diagnostics"] = _lc_selection_diagnostics(
            bundle, owner, internal_ids
        )
    if (
        spec.domains is not None
        and spec.domains.ndim == 1
        and spec.domains.shape[0] == spec.labels.shape[0]
    ):
        report["domain_selection_bias"] = [
            {
                "domain": int(domain),
                "total": int((spec.domains == domain).sum()),
                "internal": int(((spec.domains == domain) & (owner >= 0)).sum()),
                "coverage": float(
                    ((spec.domains == domain) & (owner >= 0)).sum()
                    / (spec.domains == domain).sum().clamp_min(1)
                ),
            }
            for domain in torch.unique(spec.domains).tolist()
        ]
    if spec.problem_type == "LP":
        lp = _lp_group_audit(bundle, owner)
        positive = spec.labels == 1
        lp["positive_coverage"] = float(
            bundle.partition.diagnostics.get(
                "lp_internal_positive_coverage",
                ((owner >= 0) & positive).sum() / positive.sum().clamp_min(1),
            )
        )
        lp["candidate_coverage"] = float(
            bundle.partition.diagnostics.get(
                "lp_internal_candidate_coverage", (owner >= 0).float().mean()
            )
        )
        report["link_prediction"] = lp
        if lp["partition_provenance"]["uses_evaluation_positive_endpoints"]:
            blockers.append("lp_partition_is_conditioned_on_evaluation_positive_endpoints")
        if lp["partition_provenance"]["uses_evaluation_candidates"]:
            blockers.append("lp_partition_is_conditioned_on_evaluation_candidates")
        if any(row["positives"] == 0 for row in support_rows):
            blockers.append("lp_client_task_split_without_positive")
        group_summary = lp["negative_count_per_group"]
        if (
            group_summary["min"] > 0
            and group_summary["p90"] / group_summary["min"]
            > policy["candidate_group_imbalance_warning"]
        ):
            warnings.append("lp_candidate_group_size_imbalance")
    report["blockers"] = sorted(set(blockers))
    report["warnings"] = sorted(set(warnings))
    severe = any(
        warning.startswith(SEVERE_QUALITY_WARNING_PREFIXES)
        for warning in report["warnings"]
    )
    report["invariant_status"] = "pass"
    report["quality_status"] = (
        "severe_warning" if severe else "warning" if report["warnings"] else "pass"
    )
    benchmark_tier = RELEASE_TIER_OVERRIDES.get(
        (spec.problem_type, spec.incremental_type),
        "core",
    )
    severe_is_declared_stress = severe and benchmark_tier == "stress"
    report["release_status"] = (
        "blocked"
        if report["blockers"]
        else "provisional"
        if severe and not severe_is_declared_stress
        else "eligible"
    )
    report["benchmark_tier"] = benchmark_tier
    report["release_eligible"] = report["release_status"] == "eligible"
    report["benchmark_eligible"] = report["release_eligible"]
    report["leaderboard_ready"] = False
    report["status"] = (
        "blocked"
        if report["release_status"] == "blocked"
        else "passed_with_warnings"
        if report["warnings"]
        else "passed"
    )
    return report
