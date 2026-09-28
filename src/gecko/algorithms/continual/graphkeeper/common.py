from __future__ import annotations

import torch
from torch import nn

GRAPHKEEPER_STATE_FORMAT = "uefa-graphkeeper-private-state-v3"


GRAPHKEEPER_LP_STATE_FORMAT = "uefa-graphkeeper-lp-private-state-v3"


def _positive_pairs(labels: torch.Tensor) -> torch.Tensor:
    """Return supervised-contrastive positives for scalar or multi-label NC."""

    if labels.ndim == 1:
        return labels[:, None] == labels[None, :]
    positive = labels > 0
    return (positive.float() @ positive.float().t()) > 0


def _mean_neighbors(features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean aggregate strict-local messages without materializing E x D."""

    if not edge_index.numel():
        return torch.zeros_like(features)
    source, target = edge_index
    adjacency = torch.sparse_coo_tensor(
        edge_index.flip(0),
        torch.ones(source.shape[0], device=features.device, dtype=features.dtype),
        (features.shape[0], features.shape[0]),
        device=features.device,
        dtype=features.dtype,
        check_invariants=False,
    )
    aggregate = torch.sparse.mm(adjacency, features)
    degree = torch.bincount(target, minlength=features.shape[0]).to(features.dtype)
    return aggregate / degree.clamp_min(1).unsqueeze(1)


def _symmetric_neighbors(
    features: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """Match DGL ``GraphConv(norm="both")`` before its weight matrix."""

    if not edge_index.numel():
        return torch.zeros_like(features)
    source, target = edge_index
    source_degree = torch.bincount(
        source, minlength=features.shape[0]
    ).to(features.dtype)
    target_degree = torch.bincount(
        target, minlength=features.shape[0]
    ).to(features.dtype)
    values = (
        source_degree[source].clamp_min(1).pow(-0.5)
        * target_degree[target].clamp_min(1).pow(-0.5)
    )
    adjacency = torch.sparse_coo_tensor(
        edge_index.flip(0),
        values,
        (features.shape[0], features.shape[0]),
        device=features.device,
        dtype=features.dtype,
        check_invariants=False,
    )
    return torch.sparse.mm(adjacency, features)


def _output_size(model: nn.Module) -> int:
    from gecko.models.backbones import LegacyGCNAdapter
    from gecko.models.backbones import GECKOGraphModel

    problem = str(getattr(model, "problem_type", "")).upper()
    if problem == "LP" and isinstance(
        model, (GECKOGraphModel, LegacyGCNAdapter)
    ):
        return 1
    if isinstance(model, GECKOGraphModel):
        return int(model.node_head.out_features)
    if isinstance(model, LegacyGCNAdapter) and model.problem_type == "NC":
        classifier = model.original.classifier
        if classifier is None:
            raise RuntimeError("GraphKeeper requires the BeGin NC classifier metadata.")
        return int(classifier.num_outputs)
    raise RuntimeError("GraphKeeper requires a verified NC/LP graph backbone.")


def _query_embeddings(
    hidden: torch.Tensor,
    queries: torch.Tensor,
    problem_type: str,
) -> torch.Tensor:
    """Represent NC nodes or LP endpoint pairs in one analytic feature space."""

    if problem_type == "NC":
        return hidden[queries]
    if problem_type == "LP":
        return hidden[queries[:, 0]] * hidden[queries[:, 1]]
    raise ValueError(f"GraphKeeper has no query representation for {problem_type}.")


