"""Leakage-safe client method context and model capability wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from typing import runtime_checkable

import torch


@runtime_checkable
class ForwardQueriesCapability(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor: ...


@runtime_checkable
class EncodeNodesCapability(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        layer_index: int | None,
    ) -> torch.Tensor: ...


def _owned_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor.")
    return value.detach().clone().contiguous()


def _validate_edge_index(
    edge_index: torch.Tensor, *, num_nodes: int, name: str
) -> None:
    if (
        edge_index.dtype != torch.long
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError(f"{name} must have int64 shape [2, num_edges].")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError(f"{name} contains a non-local endpoint.")


def _validate_queries(
    queries: torch.Tensor, *, problem_type: str, num_nodes: int, name: str
) -> None:
    if queries.dtype != torch.long:
        raise ValueError(f"{name} must use int64 local indices.")
    if problem_type == "NC":
        if queries.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional for NC.")
    elif queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError(f"{name} must have shape [num_queries, 2] for LC/LP.")
    if queries.numel() and (int(queries.min()) < 0 or int(queries.max()) >= num_nodes):
        raise ValueError(f"{name} contains a non-local endpoint.")


@dataclass(frozen=True, init=False)
class ClientMethodContext:
    """Owned current-task data and constrained local-model capabilities.

    The record intentionally has no stream, ownership, global-node mapping,
    central evaluation shard, held-out label, or future-task field. Tensor
    properties return defensive clones, and capability calls validate every
    supplied graph/query against its local feature matrix.
    """

    client_id: int
    global_task_id: int
    stage_index: int
    round_index: int
    problem_type: str
    incremental_setting: str
    _train_queries: torch.Tensor
    _train_labels: torch.Tensor
    _valid_class_mask: torch.Tensor | None
    _node_features: torch.Tensor
    _base_edge_index: torch.Tensor
    _context_edge_index: torch.Tensor | None
    _forward_capability: ForwardQueriesCapability
    _encode_capability: EncodeNodesCapability

    def __init__(
        self,
        *,
        client_id: int,
        global_task_id: int,
        stage_index: int,
        round_index: int,
        problem_type: str,
        incremental_setting: str,
        train_queries: torch.Tensor,
        train_labels: torch.Tensor,
        valid_class_mask: torch.Tensor | None,
        node_features: torch.Tensor,
        base_edge_index: torch.Tensor,
        context_edge_index: torch.Tensor | None,
        forward_queries: ForwardQueriesCapability,
        encode_nodes: EncodeNodesCapability,
    ) -> None:
        problem = str(problem_type).upper()
        incremental = str(incremental_setting).lower()
        if client_id < 0 or global_task_id < 0 or stage_index < 0 or round_index < 0:
            raise ValueError("Client, task, stage, and round IDs must be non-negative.")
        if problem not in {"NC", "LC", "LP"}:
            raise ValueError(f"Unsupported problem type: {problem!r}.")
        if incremental not in {"task", "class", "domain"}:
            raise ValueError(f"Unsupported incremental setting: {incremental!r}.")
        if not callable(forward_queries) or not callable(encode_nodes):
            raise TypeError("Forward and encoding capabilities must be callable.")
        features = _owned_tensor(node_features, name="node_features")
        if features.ndim < 2 or features.shape[0] == 0:
            raise ValueError("node_features must describe at least one local node.")
        base_edges = _owned_tensor(base_edge_index, name="base_edge_index")
        _validate_edge_index(
            base_edges, num_nodes=features.shape[0], name="base_edge_index"
        )
        context_edges = (
            None
            if context_edge_index is None
            else _owned_tensor(context_edge_index, name="context_edge_index")
        )
        if context_edges is not None:
            _validate_edge_index(
                context_edges,
                num_nodes=features.shape[0],
                name="context_edge_index",
            )
        queries = _owned_tensor(train_queries, name="train_queries")
        _validate_queries(
            queries,
            problem_type=problem,
            num_nodes=features.shape[0],
            name="train_queries",
        )
        labels = _owned_tensor(train_labels, name="train_labels")
        if labels.ndim == 0 or labels.shape[0] != queries.shape[0]:
            raise ValueError(
                "Current train labels must align with current train queries."
            )
        class_mask = (
            None
            if valid_class_mask is None
            else _owned_tensor(valid_class_mask, name="valid_class_mask")
        )
        if class_mask is not None and (
            class_mask.dtype != torch.bool or class_mask.ndim != 1
        ):
            raise ValueError("valid_class_mask must be a one-dimensional bool tensor.")

        object.__setattr__(self, "client_id", int(client_id))
        object.__setattr__(self, "global_task_id", int(global_task_id))
        object.__setattr__(self, "stage_index", int(stage_index))
        object.__setattr__(self, "round_index", int(round_index))
        object.__setattr__(self, "problem_type", problem)
        object.__setattr__(self, "incremental_setting", incremental)
        object.__setattr__(self, "_train_queries", queries)
        object.__setattr__(self, "_train_labels", labels)
        object.__setattr__(self, "_valid_class_mask", class_mask)
        object.__setattr__(self, "_node_features", features)
        object.__setattr__(self, "_base_edge_index", base_edges)
        object.__setattr__(self, "_context_edge_index", context_edges)
        object.__setattr__(self, "_forward_capability", forward_queries)
        object.__setattr__(self, "_encode_capability", encode_nodes)

    @property
    def train_queries(self) -> torch.Tensor:
        return self._train_queries.clone()

    @property
    def train_labels(self) -> torch.Tensor:
        return self._train_labels.clone()

    @property
    def valid_class_mask(self) -> torch.Tensor | None:
        return (
            None if self._valid_class_mask is None else self._valid_class_mask.clone()
        )

    @property
    def node_features(self) -> torch.Tensor:
        return self._node_features.clone()

    @property
    def base_edge_index(self) -> torch.Tensor:
        return self._base_edge_index.clone()

    @property
    def context_edge_index(self) -> torch.Tensor | None:
        return (
            None
            if self._context_edge_index is None
            else self._context_edge_index.clone()
        )

    @property
    def effective_edge_index(self) -> torch.Tensor:
        if self._context_edge_index is not None:
            return self._context_edge_index.clone()
        return self._base_edge_index.clone()

    def forward_queries(
        self,
        model: torch.nn.Module,
        queries: torch.Tensor | None = None,
        *,
        node_features: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run logits for the supplied local model after endpoint validation."""

        if not isinstance(model, torch.nn.Module):
            raise TypeError("Capability model must be a torch.nn.Module.")
        features = self.node_features if node_features is None else node_features
        edges = self.effective_edge_index if edge_index is None else edge_index
        values = self.train_queries if queries is None else queries
        if not torch.is_tensor(features) or features.ndim < 2 or features.shape[0] == 0:
            raise ValueError("Capability node_features must describe local nodes.")
        _validate_edge_index(
            edges, num_nodes=features.shape[0], name="capability edge_index"
        )
        _validate_queries(
            values,
            problem_type=self.problem_type,
            num_nodes=features.shape[0],
            name="capability queries",
        )
        return self._forward_capability(model, features, edges, values)

    def encode_nodes(
        self,
        model: torch.nn.Module,
        *,
        node_features: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        layer_index: int | None = None,
    ) -> torch.Tensor:
        """Run the supplied local model's embedding capability safely."""

        if not isinstance(model, torch.nn.Module):
            raise TypeError("Capability model must be a torch.nn.Module.")
        features = self.node_features if node_features is None else node_features
        edges = self.effective_edge_index if edge_index is None else edge_index
        if not torch.is_tensor(features) or features.ndim < 2 or features.shape[0] == 0:
            raise ValueError("Capability node_features must describe local nodes.")
        _validate_edge_index(
            edges, num_nodes=features.shape[0], name="capability edge_index"
        )
        return self._encode_capability(model, features, edges, layer_index)

    def twp_project_nodes(
        self,
        model: torch.nn.Module,
        *,
        node_features: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        layer_index: int,
    ) -> torch.Tensor:
        """Return a safe paper-Eq.-(10) pre-aggregation projection.

        The bound model capability exists only on explicit-v2 model instances;
        legacy models and unknown architectures fail closed.
        """

        if not isinstance(model, torch.nn.Module):
            raise TypeError("Capability model must be a torch.nn.Module.")
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer.")
        if layer_index < 0:
            raise ValueError("layer_index must be non-negative.")
        features = self.node_features if node_features is None else node_features
        edges = self.effective_edge_index if edge_index is None else edge_index
        if not torch.is_tensor(features) or features.ndim < 2 or features.shape[0] == 0:
            raise ValueError("Capability node_features must describe local nodes.")
        _validate_edge_index(
            edges, num_nodes=features.shape[0], name="capability edge_index"
        )
        capability = getattr(model, "twp_project_nodes", None)
        if not callable(capability):
            raise RuntimeError(
                f"{type(model).__name__} has no verified TWP projection capability."
            )
        try:
            model_device = next(model.parameters()).device
        except StopIteration as error:
            raise RuntimeError(
                "TWP projection requires a parameterized model."
            ) from error
        return capability(
            features.to(model_device),
            edges.to(model_device),
            layer_index=layer_index,
        )
