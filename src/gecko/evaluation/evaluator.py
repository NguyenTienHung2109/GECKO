"""Central evaluator that never exposes held-out labels to clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Dict
from typing import Iterable
from typing import Set

import torch
from torch import nn

from gecko.data.streams.builder import StreamBundle
from gecko.evaluation.metrics import compute_metrics

if TYPE_CHECKING:
    from gecko.evaluation.references import NCReferenceView


@dataclass
class FederatedEvaluator:
    stream: StreamBundle
    reference_view: "NCReferenceView | None" = None
    class_mask_policy: str = "seen"

    def __post_init__(self) -> None:
        if self.class_mask_policy not in {"seen", "full"}:
            raise ValueError("class_mask_policy must be 'seen' or 'full'.")
        self._reference_logits: Dict[tuple[int, int], torch.Tensor] = {}

    def clear_reference_cache(self) -> None:
        self._reference_logits.clear()


    def _resolve_edge_index(
        self,
        client_id: int,
        default_edge_index: torch.Tensor,
        edge_index_override: torch.Tensor | None,
    ) -> torch.Tensor:
        if edge_index_override is None:
            return default_edge_index
        if self.reference_view is not None:
            raise ValueError(
                "Topology overrides are incompatible with centralized references."
            )
        if not torch.is_tensor(edge_index_override):
            raise TypeError("edge_index_override must be a torch.Tensor.")
        if (
            edge_index_override.dtype != torch.long
            or edge_index_override.ndim != 2
            or edge_index_override.shape[0] != 2
        ):
            raise ValueError(
                "edge_index_override must have int64 shape [2, num_edges]."
            )
        num_nodes = self.stream.partition.client_graphs[
            client_id
        ].node_features.shape[0]
        if edge_index_override.numel() and (
            int(edge_index_override.min()) < 0
            or int(edge_index_override.max()) >= num_nodes
        ):
            raise ValueError(
                "edge_index_override contains an endpoint outside the strict-local graph."
            )
        return edge_index_override.detach().clone()

    def _class_mask(self, seen_tasks: Set[int], task_id: int, final_stage: bool) -> torch.Tensor | None:
        if self.class_mask_policy == "full":
            return None
        spec = self.stream.scenario
        if spec.task_masks is None:
            return None
        if spec.incremental_type == "task":
            return spec.task_masks[task_id]
        if spec.incremental_type == "class":
            if final_stage:
                return torch.ones(spec.num_classes, dtype=torch.bool)
            return spec.task_masks[list(sorted(seen_tasks))].any(dim=0)
        return None

    def evaluate(
        self,
        model: nn.Module,
        client_id: int,
        task_id: int,
        seen_tasks: Set[int],
        *,
        split: str = "test",
        final_stage: bool = False,
        edge_index_override: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        if split not in {"val", "validation", "test"}:
            raise ValueError("Evaluation split must be val/validation or test.")
        shard = self.stream.evaluation_shards[client_id][task_id]
        graph = self.stream.partition.client_graphs[client_id]
        default_edge_index = (
            graph.edge_index
            if shard.context_edge_index is None
            else shard.context_edge_index
        )
        resolved_edge_index = self._resolve_edge_index(
            client_id, default_edge_index, edge_index_override
        )
        evaluation_split = "test" if split == "test" else "validation"
        queries, central_ids = shard.queries(evaluation_split)
        labels = self.stream.scenario.labels[central_ids]
        if labels.shape[0] == 0:
            return {metric: float("nan") for metric in self.stream.scenario.metrics}
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            if self.reference_view is None:
                edge_index = resolved_edge_index
                logits = model.forward_queries(
                    graph.node_features.to(device),
                    edge_index.to(device),
                    queries.to(device),
                    self.stream.scenario.problem_type,
                )
            else:
                cache_client = -1 if self.reference_view.uses_full_topology else client_id
                cache_key = (id(model), cache_client)
                if cache_key not in self._reference_logits:
                    features, edge_index, mapped = (
                        self.reference_view.all_node_device_inputs(client_id, device)
                    )
                    self._reference_logits[cache_key] = model.forward_queries(
                        features,
                        edge_index,
                        mapped,
                        self.stream.scenario.problem_type,
                    )
                _, _, mapped_queries = self.reference_view.device_inputs(
                    client_id, queries, device
                )
                logits = self._reference_logits[cache_key][mapped_queries]
            mask = self._class_mask(seen_tasks, task_id, final_stage)
            if mask is not None and logits.ndim > 1 and logits.shape[-1] == mask.shape[0]:
                logits = logits.clone()
                logits[..., ~mask.to(device)] = -1e12
        logits = logits.detach().cpu()
        labels = labels.detach().cpu()
        candidate_groups = self.stream.scenario.metadata.get("candidate_group_ids")
        if torch.is_tensor(candidate_groups):
            candidate_groups = candidate_groups[central_ids]
        metrics = compute_metrics(
            logits,
            labels,
            self.stream.scenario.metrics,
            candidate_group_ids=candidate_groups,
            hits_tie_policy=str(
                self.stream.scenario.metadata.get(
                    "hits_tie_policy", self.stream.config.scenario.hits_tie_policy
                )
            ),
        )
        if self.stream.scenario.problem_type == "NC":
            boundary = graph.boundary_mask[queries]
            for metric in self.stream.scenario.metrics:
                boundary_value = (
                    compute_metrics(logits[boundary], labels[boundary], (metric,))[metric]
                    if boundary.any()
                    else float("nan")
                )
                interior_value = (
                    compute_metrics(logits[~boundary], labels[~boundary], (metric,))[metric]
                    if (~boundary).any()
                    else float("nan")
                )
                metrics[f"boundary_{metric}"] = boundary_value
                metrics[f"interior_{metric}"] = interior_value
                metrics[f"boundary_gap_{metric}"] = (
                    interior_value - boundary_value
                    if boundary_value == boundary_value and interior_value == interior_value
                    else float("nan")
                )
        return metrics

    def predict_central(
        self,
        model: nn.Module,
        client_id: int,
        task_id: int,
        seen_tasks: Set[int],
        *,
        split: str = "test",
        final_stage: bool = True,
        edge_index_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return central-only predictions for offline stability analysis."""

        if self.reference_view is not None:
            raise ValueError("Central prediction export is for standard client graphs.")
        if split not in {"val", "validation", "test"}:
            raise ValueError("Prediction split must be val/validation or test.")
        shard = self.stream.evaluation_shards[client_id][task_id]
        graph = self.stream.partition.client_graphs[client_id]
        default_edge_index = (
            graph.edge_index
            if shard.context_edge_index is None
            else shard.context_edge_index
        )
        edge_index = self._resolve_edge_index(
            client_id, default_edge_index, edge_index_override
        )
        queries, central_ids = shard.queries(
            "test" if split == "test" else "validation"
        )
        labels = self.stream.scenario.labels[central_ids]
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            logits = model.forward_queries(
                graph.node_features.to(device),
                edge_index.to(device),
                queries.to(device),
                self.stream.scenario.problem_type,
            )
            mask = self._class_mask(seen_tasks, task_id, final_stage)
            if mask is not None and logits.ndim > 1 and logits.shape[-1] == mask.shape[0]:
                logits = logits.clone()
                logits[..., ~mask.to(device)] = -1e12
        return logits.detach().cpu(), labels.detach().cpu(), central_ids.clone()
