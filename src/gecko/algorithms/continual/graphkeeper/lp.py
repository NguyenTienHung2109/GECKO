from __future__ import annotations

from gecko.algorithms.continual.graphkeeper.common import GRAPHKEEPER_LP_STATE_FORMAT

import torch
import torch.nn.functional as F
from gecko.algorithms.continual.graphkeeper.algorithm import GraphKeeperAlgorithm

class GraphKeeperLPAlgorithm(GraphKeeperAlgorithm):
    """LP-Domain adaptation using endpoint-product analytic edge features."""

    name = "GraphKeeper-LP"
    method_version = "uefa-graphkeeper-lp-v3-paper-faithful-objectives-normalized-routing"
    supported_problem_type = "LP"
    state_format = GRAPHKEEPER_LP_STATE_FORMAT
    prediction_unit = "edge_hadamard"
    router_name = "nearest_normalized_strict_local_structural_context_prototype"

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


