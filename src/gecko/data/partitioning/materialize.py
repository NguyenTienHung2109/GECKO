"""Persistent induced-subgraph ownership implementation."""

from __future__ import annotations

from gecko.data.partitioning.base import DIRECT_FACTORIAL_STREAM_VERSION
from gecko.data.partitioning.base import GRID_VERSION

from dataclasses import dataclass

import torch

from gecko.config import PartitionConfig
from gecko.types import ClientGraph
from gecko.types import PartitionResult
from gecko.types import ScenarioSpec
from gecko.data.partitioning.assignment import _support_tensor
from gecko.data.partitioning.assignment import assign_micro_communities
from gecko.data.partitioning.diagnostics import partition_diagnostics
from gecko.data.partitioning.community.micro_communities import generate_micro_communities


def derive_strict_local_node_features(
    spec: ScenarioSpec,
    global_nodes: torch.Tensor,
    internal_edge_mask: torch.Tensor,
) -> torch.Tensor:
    """Aggregate only visible internal edge features for owned nodes."""

    edge_features = spec.metadata["context_edge_features"]
    valid = spec.metadata.get("context_edge_feature_valid_mask")
    usable = internal_edge_mask.clone()
    if valid is not None:
        usable &= valid.bool()
    global_to_local = torch.full(
        (spec.node_features.shape[0],), -1, dtype=torch.long
    )
    global_to_local[global_nodes] = torch.arange(global_nodes.shape[0])
    local_sources = global_to_local[spec.edge_index[0, usable]]
    aggregate = torch.zeros(
        (global_nodes.shape[0], edge_features.shape[1]),
        dtype=edge_features.dtype,
    )
    counts = torch.zeros(global_nodes.shape[0], dtype=edge_features.dtype)
    if local_sources.numel():
        aggregate.index_add_(0, local_sources, edge_features[usable])
        counts.index_add_(
            0, local_sources, torch.ones_like(local_sources, dtype=edge_features.dtype)
        )
    return aggregate / counts.clamp_min(1).unsqueeze(1)


@dataclass(frozen=True)
class PersistentSubgraphPartitioner:
    """Partition once, retain ownership, and expose strict induced graphs."""

    config: PartitionConfig
    seed: int
    prefer_metis: bool = True

    def partition(self, spec: ScenarioSpec) -> PartitionResult:
        num_nodes = spec.node_features.shape[0]
        target_micro = self.config.micro_partition_multiplier * self.config.num_clients
        micro_ids = generate_micro_communities(
            spec.edge_index,
            num_nodes,
            target_micro,
            self.seed,
            prefer_metis=self.prefer_metis,
        )
        outcome = assign_micro_communities(
            spec,
            micro_ids,
            num_clients=self.config.num_clients,
            seed=self.seed,
            spatial_profile=self.config.spatial_profile,
            client_size_tolerance=self.config.client_size_tolerance,
            minimum_support=(
                self.config.minimum_train_queries_per_client_task,
                self.config.minimum_validation_queries_per_client_task,
                self.config.minimum_test_queries_per_client_task,
            ),
            maximum_iterations=self.config.maximum_assignment_iterations,
            allow_infeasible=self.config.allow_infeasible,
            minimum_internal_candidate_coverage=(
                self.config.minimum_lp_internal_candidate_coverage
                if (
                    spec.problem_type == "LP"
                    and self.config.lp_partition_information_scope
                    == "all_positive_splits"
                )
                else None
            ),
            minimum_internal_query_coverage=(
                self.config.minimum_lc_internal_query_coverage
                if spec.problem_type == "LC"
                else None
            ),
            lp_partition_information_scope=(
                self.config.lp_partition_information_scope
            ),
        )
        owner = outcome.node_owner
        source_owner = owner[spec.edge_index[0]]
        target_owner = owner[spec.edge_index[1]]
        cross = source_owner != target_owner
        boundary = torch.zeros(num_nodes, dtype=torch.bool)
        if cross.any():
            boundary[spec.edge_index[0, cross]] = True
            boundary[spec.edge_index[1, cross]] = True
        client_graphs = {}
        feature_provenance = spec.metadata.get(
            "feature_provenance", "provided_node_attributes"
        )
        for client_id in range(self.config.num_clients):
            global_nodes = torch.nonzero(owner == client_id, as_tuple=True)[0]
            global_to_local_tensor = torch.full((num_nodes,), -1, dtype=torch.long)
            global_to_local_tensor[global_nodes] = torch.arange(global_nodes.shape[0])
            internal = (source_owner == client_id) & (target_owner == client_id)
            local_edges = global_to_local_tensor[spec.edge_index[:, internal]]
            if feature_provenance == "strict_local_derived":
                node_features = derive_strict_local_node_features(
                    spec, global_nodes, internal
                )
            else:
                node_features = spec.node_features[global_nodes].detach().cpu().clone()
            client_graphs[client_id] = ClientGraph(
                client_id=client_id,
                edge_index=local_edges,
                node_features=node_features,
                local_to_global=global_nodes.clone(),
                global_to_local={
                    int(global_id): local_id
                    for local_id, global_id in enumerate(global_nodes.tolist())
                },
                boundary_mask=boundary[global_nodes].clone(),
            )
        diagnostics = partition_diagnostics(
            spec, owner, outcome.support, self.config.num_clients
        )
        diagnostics["assignment_score"] = outcome.score
        diagnostics["assignment_iterations"] = outcome.iterations
        diagnostics["constraint_failures"] = list(outcome.failures)
        diagnostics["atomic_microcommunity_repairs"] = []
        if spec.problem_type == "LP":
            information_scope = self.config.lp_partition_information_scope
            evaluation_aware = information_scope == "all_positive_splits"
            diagnostics["lp_partition_information_scope"] = information_scope
            diagnostics["lp_partition_uses_evaluation_positive_endpoints"] = (
                evaluation_aware
            )
            diagnostics["lp_partition_uses_evaluation_candidates"] = evaluation_aware
            diagnostics["lp_evaluation_support_role"] = (
                "assignment_constraint"
                if evaluation_aware
                else "post_partition_audit_only"
            )
            diagnostics["lp_support_anchor_query_splits"] = (
                ["train", "validation", "test"]
                if evaluation_aware
                else []
                if information_scope == "topology_only"
                else ["train"]
            )
            diagnostics["lp_support_anchor_community_count"] = (
                outcome.support_anchor_community_count
            )
        diagnostics["micro_community_count"] = int(torch.unique(micro_ids).numel())
        diagnostics["spatial_profile"] = self.config.spatial_profile
        topology_only_lp = (
            spec.problem_type == "LP"
            and self.config.lp_partition_information_scope == "topology_only"
        )
        diagnostics["topology_affinity_weight"] = (
            {"easy": 0.5, "mild": 5.0, "hard": 20.0}
            if topology_only_lp
            else {"easy": 20.0, "mild": 5.0, "hard": 0.5}
        )[self.config.spatial_profile]
        diagnostics["assignment_task_volume_l2_divergence"] = (
            outcome.semantic_divergence
        )
        diagnostics["semantic_divergence_target"] = (
            None
            if topology_only_lp
            else {
                "easy": 0.0,
                "mild": 0.08,
                "hard": 0.25,
            }[self.config.spatial_profile]
        )
        diagnostics["spatial_profile_semantics"] = (
            "topology_only_partition_not_query_semantic_controlled"
            if topology_only_lp
            else "task_volume_l2_semantic_controlled"
        )
        diagnostics["semantic_control_metric"] = (
            None if topology_only_lp else "client_task_volume_l2"
        )
        diagnostics["semantic_control_status"] = (
            "not_controlled_topology_only"
            if topology_only_lp
            else "controlled"
        )
        diagnostics["target_alpha"] = diagnostics["topology_affinity_weight"]
        diagnostics["feature_provenance"] = feature_provenance
        diagnostics["support_semantics"] = (
            "positive_queries" if spec.problem_type == "LP" else "all_queries"
        )
        diagnostics["strict_local_derived_feature_cross_client_edge_count"] = 0
        return PartitionResult(owner, client_graphs, micro_ids, diagnostics)


def rematerialize_partition_context(
    spec: ScenarioSpec,
    partition: PartitionResult,
    config: PartitionConfig,
) -> PartitionResult:
    """Rebuild fixed client graphs after post-partition LP query construction."""

    owner = partition.node_owner
    num_nodes = spec.node_features.shape[0]
    source_owner = owner[spec.edge_index[0]]
    target_owner = owner[spec.edge_index[1]]
    cross = source_owner != target_owner
    boundary = torch.zeros(num_nodes, dtype=torch.bool)
    if cross.any():
        boundary[spec.edge_index[0, cross]] = True
        boundary[spec.edge_index[1, cross]] = True
    feature_provenance = spec.metadata.get(
        "feature_provenance", "provided_node_attributes"
    )
    client_graphs = {}
    for client_id in range(config.num_clients):
        global_nodes = torch.nonzero(owner == client_id, as_tuple=True)[0]
        global_to_local_tensor = torch.full((num_nodes,), -1, dtype=torch.long)
        global_to_local_tensor[global_nodes] = torch.arange(global_nodes.shape[0])
        internal = (source_owner == client_id) & (target_owner == client_id)
        local_edges = global_to_local_tensor[spec.edge_index[:, internal]]
        if feature_provenance == "strict_local_derived":
            node_features = derive_strict_local_node_features(
                spec, global_nodes, internal
            )
        else:
            node_features = spec.node_features[global_nodes].detach().cpu().clone()
        client_graphs[client_id] = ClientGraph(
            client_id=client_id,
            edge_index=local_edges,
            node_features=node_features,
            local_to_global=global_nodes.clone(),
            global_to_local={
                int(global_id): local_id
                for local_id, global_id in enumerate(global_nodes.tolist())
            },
            boundary_mask=boundary[global_nodes].clone(),
        )
    support = _support_tensor(spec, owner, config.num_clients)
    diagnostics = partition_diagnostics(spec, owner, support, config.num_clients)
    base = partition.diagnostics
    diagnostics.update(
        {
            "partition_base_edge_cut_ratio": base.get("edge_cut_ratio"),
            "partition_base_boundary_node_ratio": base.get("boundary_node_ratio"),
            "partition_base_internal_edge_count_per_client": base.get(
                "internal_edge_count_per_client"
            ),
            "training_context_edge_cut_ratio": diagnostics["edge_cut_ratio"],
            "assignment_score": base.get("assignment_score"),
            "assignment_iterations": base.get("assignment_iterations"),
            "constraint_failures": base.get("constraint_failures", []),
            "micro_community_count": base.get("micro_community_count"),
            "spatial_profile": base.get("spatial_profile"),
            "target_alpha": base.get("target_alpha"),
            "topology_affinity_weight": base.get("topology_affinity_weight"),
            "semantic_divergence_target": base.get("semantic_divergence_target"),
            "spatial_profile_semantics": base.get("spatial_profile_semantics"),
            "assignment_task_volume_l2_divergence": base.get(
                "assignment_task_volume_l2_divergence"
            ),
            "semantic_control_metric": base.get("semantic_control_metric"),
            "semantic_control_status": base.get("semantic_control_status"),
            "feature_provenance": feature_provenance,
            "support_semantics": "positive_queries",
            "strict_local_derived_feature_cross_client_edge_count": 0,
            "lp_partition_information_scope": "topology_only",
            "lp_partition_uses_evaluation_positive_endpoints": False,
            "lp_partition_uses_evaluation_candidates": False,
            "lp_evaluation_support_role": "post_partition_audit_only",
            "lp_support_anchor_query_splits": [],
            "lp_support_anchor_community_count": 0,
        }
    )
    return PartitionResult(
        owner,
        client_graphs,
        partition.micro_community_ids,
        diagnostics,
    )


from collections.abc import Mapping
from typing import Any
import torch
from gecko.config import GECKOConfig
from gecko.data.partitioning.assignment import _support_tensor
from gecko.data.partitioning.diagnostics import partition_diagnostics

from gecko.types import ClientGraph
from gecko.types import PartitionResult
from gecko.types import ScenarioSpec

def materialize_direct_partition(
    spec: ScenarioSpec,
    owner: torch.Tensor,
    config: GECKOConfig,
    *,
    scientific_diagnostics: Mapping[str, Any],
) -> PartitionResult:
    """Materialize strict induced graphs without invoking community code."""
    from gecko.data.partitioning.base import DIRECT_FACTORIAL_STREAM_VERSION
    from gecko.data.partitioning.base import GRID_VERSION

    values = owner.detach().cpu().long().clone()
    num_nodes = int(spec.node_features.shape[0])
    num_clients = config.partition.num_clients
    if values.shape != (num_nodes,) or bool((values < 0).any()) or bool(
        (values >= num_clients).any()
    ):
        raise ValueError("Direct ownership must assign every node exactly once.")
    sources, targets = spec.edge_index.detach().cpu().long()
    source_owner = values[sources]
    target_owner = values[targets]
    cross = source_owner != target_owner
    boundary = torch.zeros(num_nodes, dtype=torch.bool)
    if bool(cross.any()):
        boundary[sources[cross]] = True
        boundary[targets[cross]] = True
    feature_provenance = spec.metadata.get(
        "feature_provenance", "provided_node_attributes"
    )
    client_graphs: dict[int, ClientGraph] = {}
    for client_id in range(num_clients):
        global_nodes = torch.nonzero(values == client_id, as_tuple=True)[0]
        global_to_local = torch.full((num_nodes,), -1, dtype=torch.long)
        global_to_local[global_nodes] = torch.arange(global_nodes.numel())
        internal = (source_owner == client_id) & (target_owner == client_id)
        local_edges = global_to_local[spec.edge_index[:, internal]]
        if feature_provenance == "strict_local_derived":
            features = derive_strict_local_node_features(spec, global_nodes, internal)
        else:
            features = spec.node_features[global_nodes].detach().cpu().clone()
        client_graphs[client_id] = ClientGraph(
            client_id=client_id,
            edge_index=local_edges.clone(),
            node_features=features,
            local_to_global=global_nodes.clone(),
            global_to_local={
                int(global_id): local_id
                for local_id, global_id in enumerate(global_nodes.tolist())
            },
            boundary_mask=boundary[global_nodes].clone(),
        )
    support = _support_tensor(spec, values, num_clients)
    diagnostics = partition_diagnostics(spec, values, support, num_clients)
    diagnostics.update(
        {
            **dict(scientific_diagnostics),
            "partition_mode": "direct_dirichlet_hconn",
            "direct_stream_version": DIRECT_FACTORIAL_STREAM_VERSION,
            "grid_version": GRID_VERSION,
            "community_generation_used": False,
            "strict_local_derived_feature_cross_client_edge_count": 0,
            "feature_provenance": feature_provenance,
            "constraint_failures": [],
        }
    )
    # PartitionResult keeps this compatibility field for legacy consumers.
    # -1 is an explicit sentinel: no micro-community object exists.
    direct_sentinel = torch.full((num_nodes,), -1, dtype=torch.long)
    return PartitionResult(
        node_owner=values,
        client_graphs=client_graphs,
        micro_community_ids=direct_sentinel,
        diagnostics=diagnostics,
    )


