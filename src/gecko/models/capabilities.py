"""Opt-in model capabilities for stateful UEFA v2 methods.

The v1 adapter module is checksum-locked release evidence.  Capabilities are
therefore attached only to explicit v2 model instances; ordinary logits,
parameter names, gradients, and every legacy constructor remain unchanged.
"""

from __future__ import annotations

from types import MethodType
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from gecko.models.backbones import LegacyGCNAdapter
from gecko.models.backbones import GECKOGraphModel


def encode_model_nodes(
    model: nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    layer_index: int | None = None,
) -> torch.Tensor:
    """Return differentiable encoder features for a supported graph model."""

    if isinstance(model, GECKOGraphModel):
        if layer_index is None:
            return model.encode(features, edge_index)
        if not 0 <= layer_index < len(model.layers):
            raise IndexError("layer_index is outside the UEFA graph encoder.")
        hidden = features
        for index, layer in enumerate(model.layers):
            hidden = F.relu(layer(hidden, edge_index))
            if index == layer_index:
                return hidden
        raise AssertionError("unreachable")

    if isinstance(model, LegacyGCNAdapter):
        graph = model._graph(edge_index, features.shape[0])
        encoder = model.original.gcn if model.problem_type == "LC" else model.original
        if layer_index is None:
            return encoder.forward_without_classifier(graph, features)
        if not 0 <= layer_index < int(encoder.n_layers):
            raise IndexError("layer_index is outside the BeGin graph encoder.")
        hidden = encoder.dropout(features)
        for index in range(encoder.n_layers):
            hidden = encoder.convs[index](graph, hidden)
            hidden = encoder.norms[index](hidden)
            hidden = encoder.activation(hidden)
            hidden = encoder.dropout(hidden)
            if index == layer_index:
                return hidden
        raise AssertionError("unreachable")

    encode = getattr(model, "encode", None)
    if callable(encode) and layer_index is None:
        return encode(features, edge_index)
    raise RuntimeError(
        f"{type(model).__name__} has no verified UEFA node-encoding capability."
    )


def project_model_nodes_for_twp(
    model: nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    layer_index: int,
) -> torch.Tensor:
    """Return the paper Eq.-(10) projection ``h^(l-1) W^l``.

    This capability is attached only to explicit-v2 model instances. It
    deliberately stops before the selected layer's graph aggregation,
    normalization, activation, and dropout. For ``MeanGraphLayer``, the
    neighbor/message transform is the graph-convolution ``W``; the separate
    self transform is an architectural residual path and is not part of the
    nonparametric Eq.-(10) score.
    """

    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise TypeError("layer_index must be an integer.")
    if isinstance(model, GECKOGraphModel):
        if not 0 <= layer_index < len(model.layers):
            raise IndexError("layer_index is outside the UEFA graph encoder.")
        hidden = features
        for index, layer in enumerate(model.layers):
            if index == layer_index:
                return layer.neighbor_linear(hidden)
            hidden = F.relu(layer(hidden, edge_index))
        raise AssertionError("unreachable")

    if isinstance(model, LegacyGCNAdapter):
        encoder, resolved_edges = model._encoder_and_edges(
            edge_index, int(features.shape[0])
        )
        if not 0 <= layer_index < int(encoder.n_layers):
            raise IndexError("layer_index is outside the BeGin graph encoder.")
        hidden = encoder.dropout(features)
        for index in range(encoder.n_layers):
            if index == layer_index:
                weight = encoder.convs[index].weight
                if weight is None:
                    raise RuntimeError("The selected BeGin GCN layer has no weight.")
                return torch.matmul(hidden, weight)
            # The canonical fcgl DGL build cannot execute GraphConv's degree
            # checks on CUDA.  The adapter's torch sparse aggregation is the
            # same ``norm='both'`` operation used by the ordinary CUDA forward
            # path and keeps the TWP Eq.-(10) projection differentiable.
            layer = encoder.convs[index]
            weight = layer.weight
            if weight is None:
                raise RuntimeError("The selected BeGin GCN layer has no weight.")
            hidden = model._symmetric_aggregate(hidden, resolved_edges) @ weight
            if layer.bias is not None:
                hidden = hidden + layer.bias
            hidden = encoder.norms[index](hidden)
            hidden = encoder.activation(hidden)
            hidden = encoder.dropout(hidden)
        raise AssertionError("unreachable")

    raise RuntimeError(
        f"{type(model).__name__} has no verified TWP Eq.-(10) projection."
    )


def collect_model_topology_scores(
    model: nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    score_edges: torch.Tensor | None = None,
    layer_index: int | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Collect differentiable nonparametric cosine scores for local arcs."""

    embeddings = F.normalize(
        encode_model_nodes(
            model,
            features,
            edge_index,
            layer_index=layer_index,
        ),
        p=2,
        dim=-1,
        eps=eps,
    )
    arcs = edge_index if score_edges is None else score_edges
    if arcs.dtype != torch.long or arcs.ndim != 2 or arcs.shape[0] != 2:
        raise ValueError("score_edges must have int64 shape [2, num_arcs].")
    if arcs.numel() and (int(arcs.min()) < 0 or int(arcs.max()) >= embeddings.shape[0]):
        raise ValueError("score_edges contains a non-local endpoint.")
    if arcs.numel() == 0:
        return embeddings.new_empty((0,))
    return (embeddings[arcs[0]] * embeddings[arcs[1]]).sum(dim=-1)


def mask_model_output_gradients(
    model: nn.Module,
    class_mask: torch.Tensor,
) -> None:
    """Zero inactive NC/LC output rows after strategy/method corrections.

    This capability is attached only on the explicit v2 path. It identifies
    the verified public NC classifier for each supported model family and
    fails closed for an unknown output head instead of guessing from tensor
    shapes.
    """

    if (
        not torch.is_tensor(class_mask)
        or class_mask.dtype != torch.bool
        or class_mask.ndim != 1
    ):
        raise ValueError("class_mask must be a one-dimensional bool tensor.")
    if isinstance(model, GECKOGraphModel):
        if model.problem_type == "NC":
            output = model.node_head
        elif model.problem_type == "LC":
            output = model.edge_head
        else:
            raise ValueError("Output-gradient class masks require NC or LC.")
    elif isinstance(model, LegacyGCNAdapter):
        if model.problem_type not in {"NC", "LC"}:
            raise ValueError("Output-gradient class masks require NC or LC.")
        output = model.original.classifier.lin
    else:
        raise RuntimeError(
            f"{type(model).__name__} has no verified UEFA output-gradient mask."
        )
    if not isinstance(output, nn.Linear) or output.out_features != class_mask.numel():
        raise RuntimeError("Verified NC/LC output head does not match the class mask.")
    inactive = ~class_mask.to(output.weight.device)
    if output.weight.grad is not None:
        output.weight.grad[inactive] = 0
    if output.bias is not None and output.bias.grad is not None:
        output.bias.grad[inactive] = 0


def export_model_dynamic_state(model: nn.Module) -> dict[str, object]:
    """Export persistent non-state_dict metadata for supported v2 models."""

    if not isinstance(model, LegacyGCNAdapter):
        return {}
    adaptive_outputs = {}
    for name, module in model.named_modules():
        observed = getattr(module, "observed", None)
        if not torch.is_tensor(observed):
            continue
        masks = getattr(module, "output_masks", None)
        adaptive_outputs[name] = {
            "num_outputs": int(module.num_outputs),
            "observed": observed.detach().cpu().clone(),
            "output_masks": (
                None
                if masks is None
                else tuple(mask.detach().cpu().clone() for mask in masks)
            ),
        }
    return {"adaptive_outputs": adaptive_outputs}


def load_model_dynamic_state(
    model: nn.Module,
    state: Mapping[str, object],
) -> None:
    """Restore persistent dynamic metadata and invalidate graph caches."""

    if not isinstance(model, LegacyGCNAdapter):
        if dict(state):
            raise ValueError(f"{type(model).__name__} has no dynamic model state.")
        return
    if set(state) != {"adaptive_outputs"}:
        raise ValueError("Invalid BeGin dynamic checkpoint fields.")
    raw_outputs = state["adaptive_outputs"]
    if not isinstance(raw_outputs, Mapping):
        raise TypeError("adaptive_outputs must be a mapping.")
    modules = {
        name: module
        for name, module in model.named_modules()
        if torch.is_tensor(getattr(module, "observed", None))
    }
    if set(raw_outputs) != set(modules):
        raise ValueError("BeGin dynamic module names do not match the model.")
    for name, module in modules.items():
        payload = raw_outputs[name]
        if not isinstance(payload, Mapping) or set(payload) != {
            "num_outputs",
            "observed",
            "output_masks",
        }:
            raise ValueError(f"Invalid dynamic metadata for module {name!r}.")
        if payload["num_outputs"] != int(module.num_outputs):
            raise ValueError(f"Dynamic output width mismatch for module {name!r}.")
        observed = payload["observed"]
        if (
            not torch.is_tensor(observed)
            or observed.dtype != torch.bool
            or tuple(observed.shape) != (module.num_outputs,)
        ):
            raise ValueError(f"Invalid observed mask for module {name!r}.")
        masks = payload["output_masks"]
        if masks is not None and not isinstance(masks, (list, tuple)):
            raise TypeError(f"Output masks for module {name!r} must be a sequence.")
        restored_masks = None
        if masks is not None:
            restored_masks = []
            for mask in masks:
                if (
                    not torch.is_tensor(mask)
                    or mask.dtype != torch.bool
                    or tuple(mask.shape) != (module.num_outputs,)
                ):
                    raise ValueError(f"Invalid output mask for module {name!r}.")
                restored_masks.append(
                    mask.detach().clone().to(module.lin.weight.device)
                )
        module.observed = observed.detach().clone().to(module.lin.weight.device)
        module.output_masks = restored_masks
    model._cached_graph_key = None
    model._cached_graph = None


def _bound_encode(
    self: nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    layer_index: int | None = None,
) -> torch.Tensor:
    return encode_model_nodes(
        self,
        features,
        edge_index,
        layer_index=layer_index,
    )


def _bound_topology(
    self: nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    score_edges: torch.Tensor | None = None,
    layer_index: int | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    return collect_model_topology_scores(
        self,
        features,
        edge_index,
        score_edges=score_edges,
        layer_index=layer_index,
        eps=eps,
    )


def _bound_twp_projection(
    self: nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    layer_index: int,
) -> torch.Tensor:
    return project_model_nodes_for_twp(
        self,
        features,
        edge_index,
        layer_index=layer_index,
    )


def _bound_export(self: nn.Module) -> dict[str, object]:
    return export_model_dynamic_state(self)


def _bound_load(self: nn.Module, state: Mapping[str, object]) -> None:
    load_model_dynamic_state(self, state)


def _bound_mask_output_gradients(
    self: nn.Module,
    class_mask: torch.Tensor,
) -> None:
    mask_model_output_gradients(self, class_mask)


def attach_v2_model_capabilities(model: nn.Module) -> nn.Module:
    """Attach verified capabilities to one explicit-v2 model instance."""

    if not hasattr(model, "encode_nodes"):
        model.encode_nodes = MethodType(_bound_encode, model)
    if not hasattr(model, "collect_topology_scores"):
        model.collect_topology_scores = MethodType(_bound_topology, model)
    if not hasattr(model, "twp_project_nodes"):
        model.twp_project_nodes = MethodType(_bound_twp_projection, model)
    if not hasattr(model, "export_dynamic_state"):
        model.export_dynamic_state = MethodType(_bound_export, model)
    if not hasattr(model, "load_dynamic_state"):
        model.load_dynamic_state = MethodType(_bound_load, model)
    if not hasattr(model, "mask_output_gradients"):
        model.mask_output_gradients = MethodType(_bound_mask_output_gradients, model)
    return model
