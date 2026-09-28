"""Experiment-local CUDA compatibility for legacy BeGin NC GraphConv."""

from __future__ import annotations

from typing import Any

import torch


CUDA_COMPAT_VERSION = "begin_gcn_nc_torch_coo_graphconv_both_v1"
_INSTALLED = False


def forward_nc_torch_graphconv(
    original: Any,
    features: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Match DGL GraphConv(norm='both') using native torch COO operations."""

    source, target = edge_index.long()
    num_nodes = features.shape[0]
    h = original.dropout(features)
    for index, conv in enumerate(original.convs):
        out_degree = torch.zeros(num_nodes, dtype=h.dtype, device=h.device)
        in_degree = torch.zeros_like(out_degree)
        ones = torch.ones(source.shape[0], dtype=h.dtype, device=h.device)
        if source.numel():
            out_degree.index_add_(0, source, ones)
            in_degree.index_add_(0, target, ones)
        source_norm = out_degree.clamp_min(1).pow(-0.5).unsqueeze(1)
        target_norm = in_degree.clamp_min(1).pow(-0.5).unsqueeze(1)
        transformed = (h * source_norm) @ conv.weight
        aggregated = torch.zeros(
            (num_nodes, transformed.shape[1]),
            dtype=transformed.dtype,
            device=transformed.device,
        )
        if source.numel():
            aggregated.index_add_(0, target, transformed[source])
        h = aggregated * target_norm
        h = original.norms[index](h)
        h = original.activation(h)
        h = original.dropout(h)
    return original.classifier(h)


def install_nc_cuda_graphconv_compatibility() -> None:
    """Patch only this process; canonical registry and serialized model stay unchanged."""

    global _INSTALLED
    if _INSTALLED:
        return
    from gecko.models.backbones import LegacyGCNAdapter

    legacy = LegacyGCNAdapter.forward_queries

    def forward_queries(
        self: Any,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor,
        problem_type: str,
    ) -> torch.Tensor:
        if problem_type.upper() == "NC" and features.is_cuda:
            return forward_nc_torch_graphconv(
                self.original, features, edge_index
            )[queries]
        return legacy(self, features, edge_index, queries, problem_type)

    LegacyGCNAdapter.forward_queries = forward_queries
    _INSTALLED = True

