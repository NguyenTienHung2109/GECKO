from __future__ import annotations

from gecko.algorithms.continual.graphkeeper.common import GRAPHKEEPER_LP_STATE_FORMAT

import torch
from torch import nn
import torch.nn.functional as F
from gecko.algorithms.continual.graphkeeper.algorithm import GraphKeeperAlgorithm

class GraphKeeperLPAlgorithm(GraphKeeperAlgorithm):
    """LP-Domain adaptation using endpoint-product analytic edge features."""

    name = "GraphKeeper-LP"
    method_version = "uefa-graphkeeper-lp-v4-shared-backbone-consolidation"
    consolidate_first_stage_with_shared_backbone = True
    supported_problem_type = "LP"
    state_format = GRAPHKEEPER_LP_STATE_FORMAT
    prediction_unit = "edge_hadamard"
    router_name = "nearest_normalized_strict_local_structural_context_prototype"

    def __init__(self, *, prediction_head: str = "ridge", **kwargs: object) -> None:
        if prediction_head not in {"ridge", "dot_product"}:
            raise ValueError("GraphKeeper-LP prediction_head must be ridge or dot_product.")
        self.prediction_head = prediction_head
        if prediction_head == "dot_product":
            self.state_format = f"{self.state_format}-dot-product"
            self.method_version = f"{self.method_version}-dot-product"
        super().__init__(**kwargs)

    def _analytic_logits(self, model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        if self.prediction_head == "dot_product":
            return hidden.sum(dim=-1)
        return super()._analytic_logits(model, hidden)

    def _update_analytic_classifier(
        self, model: nn.Module, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> None:
        if self.prediction_head == "ridge":
            super()._update_analytic_classifier(model, embeddings, labels)

    def diagnostics(self) -> dict[str, object]:
        diagnostics = super().diagnostics()
        diagnostics["prediction_head"] = self.prediction_head
        if self.prediction_head == "dot_product":
            diagnostics["preservation_mechanism"] = "frozen_backbone_with_private_experts"
        return diagnostics

    @staticmethod
    def _signed_log1p(values: torch.Tensor) -> torch.Tensor:
        return values.sign() * torch.log1p(values.abs())

    def _project_domain(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Encode strict-local LP structure without relying on one raw feature."""
        from gecko.algorithms.continual.graphkeeper.common import _mean_neighbors

        node_count = int(features.shape[0])
        source, target = edge_index
        out_degree = torch.bincount(source, minlength=node_count).to(features.dtype)
        in_degree = torch.bincount(target, minlength=node_count).to(features.dtype)
        total_degree = out_degree + in_degree
        logged_degree = torch.log1p(total_degree).unsqueeze(1)
        neighbor_degree = _mean_neighbors(logged_degree, edge_index)
        structural = torch.cat(
            (
                self._signed_log1p(features),
                torch.log1p(out_degree).unsqueeze(1),
                torch.log1p(in_degree).unsqueeze(1),
                logged_degree,
                neighbor_degree,
            ),
            dim=1,
        )
        centered = structural - structural.mean(dim=0, keepdim=True)
        scaled = centered / structural.std(
            dim=0, unbiased=False, keepdim=True
        ).clamp_min(1e-6)
        weight = self._ensure_routing_projection(int(scaled.shape[1])).to(
            features.device
        )
        projected = scaled @ weight
        return F.relu(projected + _mean_neighbors(projected, edge_index))

    def _domain_routing_prototype(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Pool task-context structure independently of LP candidate ordering."""

        del queries
        projected = self._project_domain(features, edge_index)
        summary = torch.cat(
            (
                projected.mean(dim=0),
                projected.std(dim=0, unbiased=False),
            )
        )
        return F.normalize(summary, dim=0)

    def _cluster_metric_values(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(embeddings, dim=-1)

    def _cluster_neighbor_mask(self, values: torch.Tensor) -> torch.Tensor:
        cosine_distance = 1.0 - (values @ values.t()).clamp(min=-1.0, max=1.0)
        return cosine_distance <= self.dbscan_eps


__all__ = ["GraphKeeperAlgorithm", "GraphKeeperLPAlgorithm"]


