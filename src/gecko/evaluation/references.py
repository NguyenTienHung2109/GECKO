"""Matched NC-Domain visibility modes for centralized diagnostic references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from gecko.data.streams.builder import StreamBundle


@dataclass(frozen=True)
class ReferenceMode:
    name: str
    topology_scope: str
    feature_scope: str
    semantics: str


REFERENCE_MODES: Dict[str, ReferenceMode] = {
    "strict_local_topology_local_features": ReferenceMode(
        "strict_local_topology_local_features",
        "strict_local",
        "strict_local_derived",
        "centralized_shard_reference",
    ),
    "full_topology_local_features": ReferenceMode(
        "full_topology_local_features",
        "full_global",
        "strict_local_derived",
        "full_topology_same_queries_local_features_reference",
    ),
    "strict_local_topology_global_features": ReferenceMode(
        "strict_local_topology_global_features",
        "strict_local",
        "global_derived",
        "strict_local_topology_global_features_reference",
    ),
    "full_topology_global_features": ReferenceMode(
        "full_topology_global_features",
        "full_global",
        "global_derived",
        "full_information_same_queries_reference",
    ),
}


class NCReferenceView:
    """Map fixed client queries onto one of four matched visibility settings."""

    def __init__(self, stream: StreamBundle, mode: str) -> None:
        if mode not in REFERENCE_MODES:
            raise ValueError(f"Unknown NC reference mode: {mode!r}")
        if (
            stream.scenario.problem_type != "NC"
            or stream.scenario.incremental_type != "domain"
        ):
            raise ValueError("Matched topology/feature references are NC-Domain only.")
        self.stream = stream
        self.mode = REFERENCE_MODES[mode]
        self._stitched_local_features = self._stitch_local_features()
        self._device_graph_inputs: dict[
            tuple[int, str], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    @property
    def uses_full_topology(self) -> bool:
        return self.mode.topology_scope == "full_global"

    def _stitch_local_features(self) -> torch.Tensor:
        scenario = self.stream.scenario
        stitched = torch.empty_like(scenario.node_features)
        assigned = torch.zeros(scenario.node_features.shape[0], dtype=torch.bool)
        for graph in self.stream.partition.client_graphs.values():
            if graph.node_features.shape[1:] != scenario.node_features.shape[1:]:
                raise ValueError("Strict-local and global feature shapes do not match.")
            if assigned[graph.local_to_global].any():
                raise ValueError("Client ownership overlaps while stitching local features.")
            stitched[graph.local_to_global] = graph.node_features
            assigned[graph.local_to_global] = True
        if not assigned.all():
            raise ValueError("Client ownership is incomplete for reference features.")
        return stitched

    def inputs(
        self,
        client_id: int,
        local_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph = self.stream.partition.client_graphs[client_id]
        if self.uses_full_topology:
            features = (
                self._stitched_local_features
                if self.mode.feature_scope == "strict_local_derived"
                else self.stream.scenario.node_features
            )
            queries = graph.local_to_global[local_queries]
            return features, self.stream.scenario.edge_index, queries
        features = (
            graph.node_features
            if self.mode.feature_scope == "strict_local_derived"
            else self.stream.scenario.node_features[graph.local_to_global]
        )
        return features, graph.edge_index, local_queries

    def all_node_inputs(
        self, client_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.uses_full_topology:
            features = (
                self._stitched_local_features
                if self.mode.feature_scope == "strict_local_derived"
                else self.stream.scenario.node_features
            )
            queries = torch.arange(features.shape[0], dtype=torch.long)
            return features, self.stream.scenario.edge_index, queries
        graph = self.stream.partition.client_graphs[client_id]
        return self.inputs(
            client_id,
            torch.arange(graph.node_features.shape[0], dtype=torch.long),
        )

    def device_inputs(
        self,
        client_id: int,
        local_queries: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, edge_index, queries = self.inputs(client_id, local_queries)
        cache_client = -1 if self.uses_full_topology else client_id
        key = (cache_client, str(device))
        if key not in self._device_graph_inputs:
            self._device_graph_inputs[key] = (
                features.to(device),
                edge_index.to(device),
            )
        cached_features, cached_edges = self._device_graph_inputs[key]
        return cached_features, cached_edges, queries.to(device)

    def all_node_device_inputs(
        self,
        client_id: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, edge_index, queries = self.all_node_inputs(client_id)
        cache_client = -1 if self.uses_full_topology else client_id
        key = (cache_client, str(device))
        if key not in self._device_graph_inputs:
            self._device_graph_inputs[key] = (
                features.to(device),
                edge_index.to(device),
            )
        cached_features, cached_edges = self._device_graph_inputs[key]
        return cached_features, cached_edges, queries.to(device)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.mode.name,
            "topology_scope": self.mode.topology_scope,
            "feature_scope": self.mode.feature_scope,
            "semantics": self.mode.semantics,
            "same_supervised_query_set": True,
            "same_task_schedule": True,
            "leaderboard_method": False,
            "benchmark_eligible": False,
        }


def summarize_nc_domain_reference_gaps(
    results: Dict[str, dict],
    *,
    fedavg_result: dict,
    metric: str = "final_average_performance",
) -> dict[str, float | str]:
    """Compute declared causal gaps from protocol-matched result records."""

    required = set(REFERENCE_MODES)
    if set(results) != required:
        raise ValueError(f"Reference results must contain exactly {sorted(required)}.")
    stream_ids = {result["stream_id"] for result in results.values()}
    stream_ids.add(fedavg_result["stream_id"])
    if len(stream_ids) != 1:
        raise ValueError("Reference gaps require one fixed stream/query artifact.")
    protocol_fields = (
        "stream_hash",
        "resolved_model",
        "base_metric",
        "initial_model_state_digest",
        "training_budget",
    )
    all_results = [*results.values(), fedavg_result]
    for field in protocol_fields:
        if any(field not in result for result in all_results):
            raise ValueError(f"Matched reference metadata is missing {field}.")
        values_for_field = {
            str(result[field])
            if not isinstance(result[field], dict)
            else str(sorted(result[field].items()))
            for result in all_results
        }
        if len(values_for_field) != 1:
            raise ValueError(f"Reference runs disagree on matched field {field}.")
    for name, result in results.items():
        reference = result.get("diagnostic_reference")
        if not isinstance(reference, dict) or reference.get("name") != name:
            raise ValueError(f"Result is not the declared reference mode {name}.")
    values = {
        name: float(result["summary"][metric]) for name, result in results.items()
    }
    fedavg = float(fedavg_result["summary"][metric])
    shard_local = values["strict_local_topology_local_features"]
    full_local = values["full_topology_local_features"]
    shard_global = values["strict_local_topology_global_features"]
    full_global = values["full_topology_global_features"]
    return {
        "metric": metric,
        "matched_protocol_verified": True,
        "topology_gap": full_local - shard_local,
        "feature_provenance_gap": full_global - full_local,
        "federation_gap": shard_local - fedavg,
        "feature_gap_at_strict_local_topology": shard_global - shard_local,
        "topology_gap_with_global_features": full_global - shard_global,
        "topology_feature_interaction": (
            (full_global - shard_global) - (full_local - shard_local)
        ),
    }
