"""Topology-aware Weight Preserving (TWP) for leakage-safe UEFA clients.

The implementation follows equations (3), (6)--(10) of Liu et al.,
"Overcoming Catastrophic Forgetting in Graph Neural Networks" (AAAI 2021).
Unlike the historical reference implementation, loss and topology gradients
are collected independently, importance is gradient magnitude rather than a
squared gradient, and the beta term retains its higher-order graph.
"""

from __future__ import annotations


import hashlib
import math
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Sequence
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from gecko.algorithms.base import ClientContinualAlgorithm


_STATE_VERSION = "uefa-twp-private-state-v1"
_CHECKPOINT_VERSION = "uefa-twp-method-checkpoint-v1"


def _validate_local_edges(edge_index: torch.Tensor, num_nodes: int) -> None:
    if (
        not torch.is_tensor(edge_index)
        or edge_index.dtype != torch.long
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError("edge_index must have int64 shape [2, num_edges].")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError("edge_index contains a non-local endpoint.")


def normalized_nonparametric_attention(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    center_nodes: torch.Tensor | None = None,
    max_retained_edges: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return equation-(10) attention on all or selected local center nodes.

    UEFA COO arcs follow ``source -> target`` message-passing semantics, so the
    target is the center ``i`` and the source is its neighbor ``j``.  The raw
    score is ``z_i.T @ tanh(z_j)`` and softmax normalization is performed over
    the retained incoming neighbors of each center.  The second result holds
    the retained column indices in the input edge tensor.
    """

    if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_nodes, hidden_size].")
    _validate_local_edges(edge_index, embeddings.shape[0])
    if max_retained_edges is not None and (
        isinstance(max_retained_edges, bool)
        or not isinstance(max_retained_edges, int)
        or max_retained_edges < 1
    ):
        raise ValueError("max_retained_edges must be a positive integer or None.")
    device = embeddings.device
    arcs = edge_index.to(device)
    if center_nodes is None:
        retained = torch.arange(arcs.shape[1], device=device, dtype=torch.long)
    else:
        if (
            not torch.is_tensor(center_nodes)
            or center_nodes.dtype != torch.long
            or center_nodes.ndim != 1
        ):
            raise ValueError("center_nodes must be a one-dimensional int64 tensor.")
        centers = center_nodes.to(device)
        if centers.numel() and (
            int(centers.min()) < 0 or int(centers.max()) >= embeddings.shape[0]
        ):
            raise ValueError("center_nodes contains a non-local endpoint.")
        if arcs.shape[1] == 0 or centers.numel() == 0:
            retained = torch.empty(0, dtype=torch.long, device=device)
        else:
            is_center = torch.isin(arcs[1], centers.unique())
            retained = is_center.nonzero(as_tuple=False).reshape(-1)

    if max_retained_edges is not None and retained.numel() > max_retained_edges:
        # The unbounded path remains the exact Eq.-(10) computation.  This
        # explicit large-graph path uses a deterministic, evenly spaced arc
        # subset and is recorded in the method configuration and diagnostics.
        if max_retained_edges == 1:
            positions = torch.zeros(1, device=device, dtype=torch.long)
        else:
            positions = torch.div(
                torch.arange(max_retained_edges, device=device, dtype=torch.long)
                * (retained.numel() - 1),
                max_retained_edges - 1,
                rounding_mode="floor",
            )
        retained = retained[positions]

    if retained.numel() == 0:
        return embeddings.new_empty((0,)), retained

    source = arcs[0, retained]
    target = arcs[1, retained]
    raw = (embeddings[target] * torch.tanh(embeddings[source])).sum(dim=-1)
    maxima = raw.new_full((embeddings.shape[0],), -torch.inf)
    maxima.scatter_reduce_(0, target, raw, reduce="amax", include_self=True)
    exponentials = torch.exp(raw - maxima[target])
    denominators = raw.new_zeros((embeddings.shape[0],))
    denominators.index_add_(0, target, exponentials)
    coefficients = exponentials / denominators[target]
    return coefficients, retained


def topology_attention_squared_norm(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    center_nodes: torch.Tensor | None = None,
    max_retained_edges: int | None = None,
) -> torch.Tensor:
    """Return the squared L2 norm used inside the topology gradient in Eq. (6)."""

    coefficients, _ = normalized_nonparametric_attention(
        embeddings,
        edge_index,
        center_nodes=center_nodes,
        max_retained_edges=max_retained_edges,
    )
    if coefficients.numel() == 0:
        # Keep a differentiable zero connected to the encoder graph.
        return embeddings.sum() * 0.0
    return coefficients.square().sum()


def _supervised_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim > 1:
        valid = labels >= 0
        if not bool(valid.any()):
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(logits[valid], labels[valid].float())
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(
            logits.reshape(-1), labels.float().reshape(-1)
        )
    return F.cross_entropy(logits, labels.long())


def _named_trainable_parameters(
    model: nn.Module,
) -> Tuple[Tuple[str, nn.Parameter], ...]:
    parameters = tuple(
        sorted(
            (
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.is_floating_point()
            ),
            key=lambda item: item[0],
        )
    )
    if not parameters:
        raise ValueError("TWP requires at least one floating trainable parameter.")
    return parameters


def _zeros_like_named(
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
) -> Dict[str, torch.Tensor]:
    return {name: torch.zeros_like(parameter) for name, parameter in named_parameters}


def _gradient_magnitudes(
    objective: torch.Tensor,
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
    *,
    create_graph: bool,
) -> Dict[str, torch.Tensor]:
    if not torch.is_tensor(objective) or objective.ndim != 0:
        raise ValueError("TWP gradient objectives must be scalar tensors.")
    if not objective.requires_grad:
        raise RuntimeError("TWP gradient objective is detached from the model.")
    gradients = torch.autograd.grad(
        objective,
        tuple(parameter for _, parameter in named_parameters),
        allow_unused=True,
        create_graph=create_graph,
        retain_graph=create_graph,
    )
    output: Dict[str, torch.Tensor] = {}
    for (name, parameter), gradient in zip(named_parameters, gradients):
        output[name] = (
            torch.zeros_like(parameter) if gradient is None else gradient.abs()
        )
    return output


def _parameter_metadata(
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
) -> Dict[str, object]:
    return {
        name: {
            "shape": tuple(parameter.shape),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in named_parameters
    }


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    values = tensor.detach().cpu().contiguous()
    return values.view(torch.uint8).numpy().tobytes()


def _update_digest(digest: "hashlib._Hash", value: Any) -> None:
    if torch.is_tensor(value):
        digest.update(b"tensor\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(_tensor_bytes(value))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"tuple\0" if isinstance(value, tuple) else b"list\0")
        for item in value:
            _update_digest(digest, item)
        return
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(value).encode("utf-8"))


class TWPAlgorithm(ClientContinualAlgorithm):
    """Private per-task TWP anchors and differentiable importance penalty."""

    name = "TWP"
    method_version = "uefa-twp-v1"

    def __init__(
        self,
        *,
        lambda_l: float = 10000.0,
        lambda_t: float = 10000.0,
        beta: float = 0.01,
        middle_layer_index: int | None = None,
        significant_threshold: float = 1e-12,
        topology_max_edges: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        for name, value in (
            ("lambda_l", lambda_l),
            ("lambda_t", lambda_t),
            ("beta", beta),
            ("significant_threshold", significant_threshold),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if middle_layer_index is not None and (
            isinstance(middle_layer_index, bool)
            or not isinstance(middle_layer_index, int)
            or middle_layer_index < 0
        ):
            raise ValueError("middle_layer_index must be a non-negative integer.")
        if topology_max_edges is not None and (
            isinstance(topology_max_edges, bool)
            or not isinstance(topology_max_edges, int)
            or topology_max_edges < 1
        ):
            raise ValueError("topology_max_edges must be a positive integer or None.")
        self.lambda_l = float(lambda_l)
        self.lambda_t = float(lambda_t)
        self.beta = float(beta)
        self.middle_layer_index = (
            None if middle_layer_index is None else int(middle_layer_index)
        )
        self.significant_threshold = float(significant_threshold)
        self.topology_max_edges = topology_max_edges
        self._last_diagnostics: Dict[str, object] = {
            "penalty": 0.0,
            "anchor_distance_l2": 0.0,
            "current_importance_l1": 0.0,
            "topology_squared_norm": 0.0,
            "resolved_middle_layer_index": None,
            "topology_retained_edge_count": 0,
            "pre_broadcast_state_sha256": None,
            "post_broadcast_state_sha256": None,
        }
        self.state.update(
            {
                "state_version": _STATE_VERSION,
                "method_hyperparameters": self._private_hyperparameters(),
                "parameter_metadata": {},
                "anchors": {},
                "loss_importance": {},
                "topology_importance": {},
                "consolidated_task_ids": [],
            }
        )

    def _private_hyperparameters(self) -> Dict[str, object]:
        return {
            "lambda_l": self.lambda_l,
            "lambda_t": self.lambda_t,
            "beta": self.beta,
            "middle_layer_index": self.middle_layer_index,
            "significant_threshold": self.significant_threshold,
            "topology_max_edges": self.topology_max_edges,
        }

    def hyperparameters(self) -> Dict[str, object]:
        values = super().hyperparameters()
        values.update(self._private_hyperparameters())
        return values

    @staticmethod
    def _encoder_layer_count(model: nn.Module) -> int | None:
        declared = getattr(model, "twp_num_encoder_layers", None)
        if declared is not None:
            count = int(declared)
            return count if count > 0 else None
        layers = getattr(model, "layers", None)
        if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
            return len(layers)
        original = getattr(model, "original", None)
        encoder = getattr(original, "gcn", original)
        count = getattr(encoder, "n_layers", None)
        if count is not None and int(count) > 0:
            return int(count)
        return None

    def _resolve_middle_layer(self, model: nn.Module) -> int:
        count = self._encoder_layer_count(model)
        if self.middle_layer_index is None:
            if count is None:
                raise RuntimeError(
                    "TWP cannot infer the encoder depth; configure "
                    "middle_layer_index explicitly."
                )
            resolved = (count - 1) // 2
        else:
            resolved = self.middle_layer_index
        if count is not None and resolved >= count:
            raise ValueError(
                f"middle_layer_index={resolved} is outside {count} encoder layers."
            )
        return resolved

    def _topology_objective(
        self,
        model: nn.Module,
        context: Any,
    ) -> torch.Tensor:
        problem = str(context.problem_type).upper()
        if problem not in {"NC", "LC"}:
            raise ValueError("TWP UEFA topology importance supports NC and LC only.")
        resolved = self._resolve_middle_layer(model)
        embeddings = context.twp_project_nodes(model, layer_index=resolved)
        if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
            raise RuntimeError(
                "TWP node projection must return [num_nodes, hidden_size]."
            )
        centers = context.train_queries
        if problem == "LC":
            centers = torch.unique(centers.reshape(-1))
        coefficients, retained = normalized_nonparametric_attention(
            embeddings,
            context.effective_edge_index.to(embeddings.device),
            center_nodes=centers.to(embeddings.device),
            max_retained_edges=self.topology_max_edges,
        )
        objective = (
            embeddings.sum() * 0.0
            if coefficients.numel() == 0
            else coefficients.square().sum()
        )
        self._last_diagnostics["resolved_middle_layer_index"] = resolved
        self._last_diagnostics["topology_retained_edge_count"] = int(
            retained.numel()
        )
        self._last_diagnostics["topology_squared_norm"] = float(
            objective.detach().cpu()
        )
        return objective

    @staticmethod
    def _masked_logits(
        logits: torch.Tensor,
        class_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            class_mask is None
            or logits.ndim < 2
            or logits.shape[-1] != class_mask.shape[0]
        ):
            return logits
        output = logits.clone()
        output[..., ~class_mask.to(logits.device)] = -1e12
        return output

    def _validated_parameter_metadata(
        self,
        named_parameters: Sequence[Tuple[str, nn.Parameter]],
    ) -> Dict[str, object]:
        actual = _parameter_metadata(named_parameters)
        stored = self.state["parameter_metadata"]
        if stored != {} and stored != actual:
            raise ValueError("TWP model parameter metadata changed across tasks.")
        return actual

    def augment_loss(
        self,
        model: nn.Module,
        context: Any,
        logits: torch.Tensor,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Apply equations (7)--(9), retaining the beta second-order graph."""

        del logits  # The supplied base loss is the authoritative masked objective.
        if self.lambda_l == 0.0 and self.lambda_t == 0.0:
            self._last_diagnostics.update(
                {"penalty": 0.0, "current_importance_l1": 0.0}
            )
            return base_loss

        named_parameters = _named_trainable_parameters(model)
        parameter_names = {name for name, _ in named_parameters}
        penalty = base_loss.new_zeros(())
        anchor_distance_square = base_loss.new_zeros(())
        has_penalty = False
        for task_id in self.state["consolidated_task_ids"]:
            anchors = self.state["anchors"][task_id]
            loss_importance = self.state["loss_importance"][task_id]
            topology_importance = self.state["topology_importance"][task_id]
            if set(anchors) != parameter_names:
                raise ValueError("TWP anchor names do not match the current model.")
            for name, parameter in named_parameters:
                displacement = parameter - anchors[name].to(
                    parameter.device, parameter.dtype
                )
                importance = self.lambda_l * loss_importance[name].to(
                    parameter.device, parameter.dtype
                ) + self.lambda_t * topology_importance[name].to(
                    parameter.device, parameter.dtype
                )
                penalty = penalty + (importance * displacement.square()).sum()
                anchor_distance_square = (
                    anchor_distance_square + displacement.square().sum()
                )
            has_penalty = True

        current_importance_l1 = base_loss.new_zeros(())
        has_current_importance = self.beta != 0.0
        if has_current_importance:
            loss_importance_current = (
                _gradient_magnitudes(
                    base_loss,
                    named_parameters,
                    create_graph=True,
                )
                if self.lambda_l != 0.0
                else _zeros_like_named(named_parameters)
            )
            topology_importance_current = (
                _gradient_magnitudes(
                    self._topology_objective(model, context),
                    named_parameters,
                    create_graph=True,
                )
                if self.lambda_t != 0.0
                else _zeros_like_named(named_parameters)
            )
            for name, _ in named_parameters:
                current_importance_l1 = (
                    current_importance_l1
                    + (
                        self.lambda_l * loss_importance_current[name]
                        + self.lambda_t * topology_importance_current[name]
                    ).sum()
                )

        self._last_diagnostics["penalty"] = float(penalty.detach().cpu())
        self._last_diagnostics["anchor_distance_l2"] = math.sqrt(
            float(anchor_distance_square.detach().cpu())
        )
        self._last_diagnostics["current_importance_l1"] = float(
            current_importance_l1.detach().cpu()
        )
        if not has_penalty and not has_current_importance:
            return base_loss
        return base_loss + penalty + self.beta * current_importance_l1

    def consolidate(self, model: nn.Module, context: Any) -> None:
        """Store independent Eq. (3) and Eq. (6) magnitudes for one global task."""

        if str(context.problem_type).upper() not in {"NC", "LC"}:
            raise ValueError("TWP UEFA consolidation supports NC and LC only.")
        task_id = int(context.global_task_id)
        if task_id < 0:
            raise ValueError("global_task_id must be non-negative.")
        if task_id in self.state["consolidated_task_ids"]:
            raise ValueError(f"TWP task {task_id} has already been consolidated.")
        named_parameters = _named_trainable_parameters(model)
        parameter_metadata = self._validated_parameter_metadata(named_parameters)
        previous_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in named_parameters
        }
        try:
            if self.lambda_l != 0.0:
                logits = context.forward_queries(model)
                logits = self._masked_logits(logits, context.valid_class_mask)
                loss_importance = _gradient_magnitudes(
                    _supervised_loss(
                        logits,
                        context.train_labels.to(logits.device),
                    ),
                    named_parameters,
                    create_graph=False,
                )
            else:
                loss_importance = _zeros_like_named(named_parameters)
            if self.lambda_t != 0.0:
                topology_importance = _gradient_magnitudes(
                    self._topology_objective(model, context),
                    named_parameters,
                    create_graph=False,
                )
            else:
                topology_importance = _zeros_like_named(named_parameters)
        finally:
            for name, parameter in named_parameters:
                restored = previous_gradients[name]
                parameter.grad = None if restored is None else restored

        anchors = {
            name: parameter.detach().cpu().clone()
            for name, parameter in named_parameters
        }
        stored_loss = {
            name: value.detach().cpu().clone()
            for name, value in loss_importance.items()
        }
        stored_topology = {
            name: value.detach().cpu().clone()
            for name, value in topology_importance.items()
        }
        for kind, values in (
            ("loss", stored_loss),
            ("topology", stored_topology),
        ):
            for name, value in values.items():
                if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                    raise RuntimeError(
                        f"TWP {kind} importance for {name!r} is invalid."
                    )
        # Commit only after every computation and validation succeeds.
        if self.state["parameter_metadata"] == {}:
            self.state["parameter_metadata"] = parameter_metadata
        self.state["anchors"][task_id] = anchors
        self.state["loss_importance"][task_id] = stored_loss
        self.state["topology_importance"][task_id] = stored_topology
        self.state["consolidated_task_ids"].append(task_id)

    def on_broadcast(self, context: Any, payload: Any) -> None:
        """Broadcasts may alter model parameters but never private TWP state."""

        del context, payload
        checksum = self.private_state_checksum()
        self._last_diagnostics["pre_broadcast_state_sha256"] = checksum
        self._last_diagnostics["post_broadcast_state_sha256"] = checksum
        return None

    def private_state_checksum(self) -> str:
        digest = hashlib.sha256()
        _update_digest(digest, self.state)
        return digest.hexdigest()

    def diagnostics(self) -> Dict[str, object]:
        loss_values = []
        topology_values = []
        significant = 0
        total = 0
        for task_id in self.state["consolidated_task_ids"]:
            for name in self.state["loss_importance"][task_id]:
                loss = self.state["loss_importance"][task_id][name]
                topology = self.state["topology_importance"][task_id][name]
                loss_values.append(loss.reshape(-1).float())
                topology_values.append(topology.reshape(-1).float())
                combined = self.lambda_l * loss + self.lambda_t * topology
                significant += int((combined > self.significant_threshold).sum())
                total += combined.numel()

        def statistics(values: Sequence[torch.Tensor]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "max": 0.0}
            flattened = torch.cat(tuple(values))
            return {
                "mean": float(flattened.mean()),
                "max": float(flattened.max()),
            }

        return {
            "method": self.name,
            "state_sha256": self.private_state_checksum(),
            "consolidated_task_ids": tuple(self.state["consolidated_task_ids"]),
            "num_consolidated_tasks": len(self.state["consolidated_task_ids"]),
            "loss_importance": statistics(loss_values),
            "topology_importance": statistics(topology_values),
            "significant_parameter_fraction": (
                0.0 if total == 0 else significant / total
            ),
            **self._last_diagnostics,
        }

    def _validate_loaded_state(self) -> None:
        required = {
            "state_version",
            "method_hyperparameters",
            "parameter_metadata",
            "anchors",
            "loss_importance",
            "topology_importance",
            "consolidated_task_ids",
        }
        if set(self.state) != required:
            raise ValueError("TWP checkpoint fields do not match the state schema.")
        if self.state["state_version"] != _STATE_VERSION:
            raise ValueError("Unsupported TWP private-state version.")
        if self.state["method_hyperparameters"] != self._private_hyperparameters():
            raise ValueError("TWP checkpoint hyperparameters do not match this method.")
        metadata = self.state["parameter_metadata"]
        if not isinstance(metadata, Mapping):
            raise TypeError("TWP parameter_metadata must be a mapping.")
        task_ids = self.state["consolidated_task_ids"]
        if not isinstance(task_ids, list) or any(
            isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0
            for task_id in task_ids
        ):
            raise ValueError("TWP consolidated_task_ids must be non-negative integers.")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("TWP consolidated_task_ids contains duplicates.")
        task_set = set(task_ids)
        for field in ("anchors", "loss_importance", "topology_importance"):
            values = self.state[field]
            if not isinstance(values, Mapping) or set(values) != task_set:
                raise ValueError(
                    f"TWP {field} task IDs do not match consolidation IDs."
                )
        names = set(metadata)
        if not task_ids and names:
            raise ValueError(
                "Unconsolidated TWP state must not contain parameter metadata."
            )
        for name, specification in metadata.items():
            if not isinstance(name, str) or not isinstance(specification, Mapping):
                raise TypeError("Invalid TWP parameter metadata entry.")
            if set(specification) != {"shape", "dtype"}:
                raise ValueError("Invalid TWP parameter metadata fields.")
            if not isinstance(specification["shape"], tuple) or not all(
                isinstance(size, int) and not isinstance(size, bool) and size >= 0
                for size in specification["shape"]
            ):
                raise ValueError("Invalid TWP parameter shape metadata.")
            if not isinstance(specification["dtype"], str):
                raise TypeError("Invalid TWP parameter dtype metadata.")
        for task_id in task_ids:
            for field in ("anchors", "loss_importance", "topology_importance"):
                tensors = self.state[field][task_id]
                if not isinstance(tensors, Mapping) or set(tensors) != names:
                    raise ValueError(
                        f"TWP {field} parameter names do not match metadata."
                    )
                for name, tensor in tensors.items():
                    specification = metadata[name]
                    if (
                        not torch.is_tensor(tensor)
                        or tuple(tensor.shape) != specification["shape"]
                        or str(tensor.dtype) != specification["dtype"]
                        or tensor.device.type != "cpu"
                        or not bool(torch.isfinite(tensor).all())
                    ):
                        raise ValueError(
                            f"Invalid TWP tensor {field}.{task_id}.{name}."
                        )
                    if field != "anchors" and bool((tensor < 0).any()):
                        raise ValueError("TWP importance tensors must be non-negative.")

    @staticmethod
    def _validate_diagnostics(values: Mapping[str, object]) -> None:
        required = {
            "penalty",
            "anchor_distance_l2",
            "current_importance_l1",
            "topology_squared_norm",
            "resolved_middle_layer_index",
            "topology_retained_edge_count",
            "pre_broadcast_state_sha256",
            "post_broadcast_state_sha256",
        }
        if set(values) != required:
            raise ValueError("TWP checkpoint diagnostic fields do not match.")
        for name in (
            "penalty",
            "anchor_distance_l2",
            "current_importance_l1",
            "topology_squared_norm",
        ):
            value = values[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"TWP diagnostic {name!r} is invalid.")
        layer = values["resolved_middle_layer_index"]
        if layer is not None and (
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
        ):
            raise ValueError("TWP resolved middle-layer diagnostic is invalid.")
        for name in (
            "pre_broadcast_state_sha256",
            "post_broadcast_state_sha256",
        ):
            value = values[name]
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"TWP diagnostic checksum {name!r} is invalid.")

    def save_method_state(self) -> Dict[str, object]:
        """Checkpoint diagnostics separately from broadcast-immutable state."""

        return {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "private_state": super().save_method_state(),
            "diagnostics": dict(self._last_diagnostics),
        }

    def load_method_state(self, state: Mapping[str, object]) -> None:
        """Restore private scientific state and transient diagnostics atomically."""

        previous_state = super().save_method_state()
        previous_diagnostics = dict(self._last_diagnostics)
        try:
            if not isinstance(state, Mapping) or set(state) != {
                "checkpoint_version",
                "private_state",
                "diagnostics",
            }:
                raise ValueError(
                    "TWP checkpoint fields do not match the method schema."
                )
            if state["checkpoint_version"] != _CHECKPOINT_VERSION:
                raise ValueError("Unsupported TWP method-checkpoint version.")
            private_state = state["private_state"]
            diagnostics = state["diagnostics"]
            if not isinstance(private_state, Mapping):
                raise TypeError("TWP private checkpoint state must be a mapping.")
            if not isinstance(diagnostics, Mapping):
                raise TypeError("TWP checkpoint diagnostics must be a mapping.")
            self._validate_diagnostics(diagnostics)
            super().load_method_state(private_state)
            self._validate_loaded_state()
            self._last_diagnostics = dict(diagnostics)
        except Exception:
            super().load_method_state(previous_state)
            self._last_diagnostics = previous_diagnostics
            raise


__all__ = [
    "TWPAlgorithm",
    "normalized_nonparametric_attention",
    "topology_attention_squared_norm",
]
