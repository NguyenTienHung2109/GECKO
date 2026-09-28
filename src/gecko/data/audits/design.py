"""Fail-closed scientific design checks for UEFA stream construction."""

from __future__ import annotations

from typing import Any
from typing import Dict

import torch

from gecko.validation import DatasetDesignError


def _edge_keys(edges: torch.Tensor, *, undirected: bool) -> set[tuple[int, int]]:
    if edges.numel() == 0:
        return set()
    values = edges.t().tolist() if edges.shape[0] == 2 else edges.tolist()
    return {
        (min(source, target), max(source, target))
        if undirected
        else (source, target)
        for source, target in values
    }


def audit_scenario_design(config, spec) -> Dict[str, Any]:
    """Validate preprocessing provenance before partitioning."""

    configured = config.scenario.feature_provenance
    actual = spec.metadata.get("feature_provenance")
    if actual != configured:
        raise DatasetDesignError(
            f"Feature provenance mismatch: config={configured!r}, scenario={actual!r}."
        )
    partition_first_lp = (
        spec.problem_type == "LP"
        and config.scenario.lp_query_split_protocol
        == "partition_first_query_split_v1"
    )
    audit: Dict[str, Any] = {
        "split_protocol": config.scenario.split_protocol,
        "feature_provenance": actual,
        "strict_local_features": actual != "public_global_precomputed",
        "partition_uses_context_graph": not partition_first_lp,
    }
    if actual == "strict_local_derived":
        edge_features = spec.metadata.get("context_edge_features")
        if not torch.is_tensor(edge_features):
            raise DatasetDesignError(
                "strict_local_derived requires context_edge_features in scenario metadata."
            )
        if edge_features.shape[0] != spec.edge_index.shape[1]:
            raise DatasetDesignError(
                "context_edge_features must align with the partition/context edge_index."
            )
        valid = spec.metadata.get("context_edge_feature_valid_mask")
        if valid is not None and (
            not torch.is_tensor(valid) or valid.shape != (spec.edge_index.shape[1],)
        ):
            raise DatasetDesignError(
                "context_edge_feature_valid_mask must align with edge_index."
            )

    if spec.problem_type == "LP":
        if spec.context_edge_index is None or not torch.equal(
            spec.edge_index, spec.context_edge_index
        ):
            raise DatasetDesignError(
                "LP partition edge_index must be exactly the held-out-free context graph."
            )
        undirected = bool(spec.metadata.get("undirected", False))
        context = _edge_keys(spec.context_edge_index, undirected=undirected)
        positives = spec.metadata["positive_pairs"]
        splits = spec.metadata["positive_splits"]
        held_out = _edge_keys(positives[splits != 0], undirected=undirected)
        overlap = context.intersection(held_out)
        if overlap:
            raise DatasetDesignError(
                f"LP context/partition graph contains {len(overlap)} held-out positives."
            )
        groups = spec.metadata.get("candidate_group_ids")
        if not torch.is_tensor(groups) or groups.shape != spec.labels.shape:
            raise DatasetDesignError(
                "LP queries require candidate_group_ids aligned with labels."
            )
        audit["lp_heldout_context_overlap"] = 0
        if partition_first_lp:
            base_pairs = spec.metadata.get("partition_base_positive_pairs")
            query_pool = spec.metadata.get("partition_query_positive_pool_pairs")
            if not torch.is_tensor(base_pairs) or not torch.is_tensor(query_pool):
                raise DatasetDesignError(
                    "Partition-first LP requires explicit base and query-positive pools."
                )
            base_keys = _edge_keys(base_pairs, undirected=undirected)
            query_keys = _edge_keys(query_pool, undirected=undirected)
            partition_overlap = base_keys.intersection(query_keys)
            if partition_overlap:
                raise DatasetDesignError(
                    "Partition-first LP base topology overlaps the query-positive pool."
                )
            if spec.metadata.get("partition_assignment_used_query_endpoints", False):
                raise DatasetDesignError(
                    "Partition-first LP assignment must not use query endpoints."
                )
            if spec.metadata.get("partition_assignment_used_evaluation_data", False):
                raise DatasetDesignError(
                    "Partition-first LP assignment must not use evaluation data."
                )
            audit["lp_partition_query_positive_overlap"] = 0
            audit["lp_partition_base_policy"] = spec.metadata.get(
                "partition_base_policy"
            )
            audit["lp_partition_base_edge_ratio"] = spec.metadata.get(
                "partition_base_edge_ratio"
            )
            audit["lp_partition_first_release_protocol"] = bool(
                spec.metadata.get("partition_first_release_protocol", False)
            )
        audit["lp_candidate_grouping"] = config.scenario.lp_candidate_grouping
        audit["hits_tie_policy"] = config.scenario.hits_tie_policy
        scope = config.partition.lp_partition_information_scope
        evaluation_aware = scope == "all_positive_splits"
        audit["lp_partition_information_scope"] = scope
        audit["lp_partition_uses_evaluation_positive_endpoints"] = evaluation_aware
        audit["lp_partition_uses_evaluation_candidates"] = evaluation_aware
        audit["lp_evaluation_support_role"] = (
            "assignment_constraint"
            if evaluation_aware
            else "post_partition_audit_only"
        )
    return audit


def audit_partition_design(
    config,
    spec,
    partition,
    evaluation_shards,
    scenario_audit: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate realized coverage and LP candidate groups after partitioning."""

    diagnostics = partition.diagnostics
    failures = []
    lc_threshold = config.partition.minimum_lc_internal_query_coverage
    if spec.problem_type == "LC" and lc_threshold is not None:
        realized = float(diagnostics["lc_internal_query_coverage"])
        if realized < lc_threshold:
            failures.append(
                f"LC internal-query coverage={realized:.6f} required={lc_threshold:.6f}"
            )
    if spec.problem_type == "LP":
        candidate_threshold = config.partition.minimum_lp_internal_candidate_coverage
        positive_threshold = config.partition.minimum_lp_internal_positive_coverage
        if candidate_threshold is not None:
            realized = float(diagnostics["lp_internal_candidate_coverage"])
            if realized < candidate_threshold:
                failures.append(
                    "LP internal-candidate coverage="
                    f"{realized:.6f} required={candidate_threshold:.6f}"
                )
        if positive_threshold is not None:
            realized = float(diagnostics["lp_internal_positive_coverage"])
            if realized < positive_threshold:
                failures.append(
                    "LP internal-positive coverage="
                    f"{realized:.6f} required={positive_threshold:.6f}"
                )

        required_k = max(
            [
                int(metric.lower().split("@", 1)[1])
                for metric in spec.metrics
                if metric.lower().startswith("hits@")
            ]
            or [0]
        )
        required_negatives = max(
            required_k, config.scenario.minimum_evaluation_negatives
        )
        group_ids = spec.metadata["candidate_group_ids"]
        for client_id, tasks in evaluation_shards.items():
            for task_id, shard in tasks.items():
                for split, ids in (
                    ("val", shard.validation_query_ids),
                    ("test", shard.test_query_ids),
                ):
                    labels = spec.labels[ids]
                    if not bool((labels == 1).any()):
                        failures.append(
                            f"client={client_id} task={task_id} split={split} "
                            "has no positive query"
                        )
                        continue
                    groups = group_ids[ids]
                    for group in torch.unique(groups[labels == 1]).tolist():
                        negative_count = int(((groups == group) & (labels == 0)).sum())
                        if negative_count < required_negatives:
                            failures.append(
                                f"client={client_id} task={task_id} split={split} "
                                f"candidate_group={group} negatives={negative_count} "
                                f"required={required_negatives}"
                            )
    if failures:
        raise DatasetDesignError(
            "Dataset-design audit failed:\n- " + "\n- ".join(failures)
        )
    return {
        **scenario_audit,
        "lc_internal_query_coverage_threshold": lc_threshold,
        "lp_internal_candidate_coverage_threshold": (
            config.partition.minimum_lp_internal_candidate_coverage
        ),
        "lp_internal_positive_coverage_threshold": (
            config.partition.minimum_lp_internal_positive_coverage
        ),
        "lp_partition_information_scope": (
            config.partition.lp_partition_information_scope
            if spec.problem_type == "LP"
            else None
        ),
        "evaluation_aware_stream_construction": (
            spec.problem_type == "LP"
            and config.partition.lp_partition_information_scope
            == "all_positive_splits"
        ),
        "status": "passed",
    }
