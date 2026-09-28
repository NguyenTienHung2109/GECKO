"""Strict-local FedGTA paper-oracle primitives.

This module deliberately contains only the deterministic statistics and
personalized aggregation equations used by FedGTA.  It is kept independent of
the coordinator until the complete strategy lifecycle has passed its leakage,
state, and accounting gates.  In particular, unlike the pinned reference
implementation, :func:`propagate_strict_local_labels` accepts no validation or
test labels: only labels for the current local training queries can be clamped.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import torch

from gecko.engine.accounting import ResourceLedger
from gecko.engine.protocol import AggregationResult
from gecko.engine.protocol import BroadcastPayload
from gecko.engine.protocol import BroadcastReason
from gecko.engine.protocol import ClientUpload
from gecko.engine.protocol import EvaluationSelection
from gecko.engine.protocol import FrozenTensorMap
from gecko.engine.protocol import ParameterManifest
from gecko.engine.protocol import RoundContext
from gecko.engine.protocol import TensorState


def _require_finite_float(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def propagate_strict_local_labels(
    *,
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    current_train_nodes: torch.Tensor,
    current_train_labels: torch.Tensor,
    propagation_steps: int = 5,
    propagation_alpha: float = 0.5,
    temperature: float = 20.0,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    """Return FedGTA confidence labels, moment states, and local degrees.

    This is the paper's non-parametric label propagation restricted to one
    client graph. ``current_train_labels`` is the sole label input; all other
    nodes start from model probabilities. As in the reference implementation,
    temperature scaling is applied only to the final distribution used by the
    smoothing-confidence equation. The returned per-hop states use the raw
    softmax and therefore have the paper's ``(k * K) x |Y|`` moment layout.
    """

    if logits.ndim != 2 or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be a finite floating [num_nodes, num_classes] tensor.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
        raise ValueError("edge_index must be a [2, num_edges] torch.long tensor.")
    node_count, class_count = logits.shape
    if node_count == 0 or class_count == 0:
        raise ValueError("logits must contain at least one node and one class.")
    if (
        current_train_nodes.ndim != 1
        or current_train_nodes.dtype != torch.long
        or current_train_labels.ndim != 1
        or current_train_labels.dtype != torch.long
        or current_train_nodes.numel() != current_train_labels.numel()
    ):
        raise ValueError("current train nodes and labels must be paired 1-D torch.long tensors.")
    if current_train_nodes.numel() == 0:
        raise ValueError("FedGTA requires at least one current local training label.")
    if torch.any(current_train_nodes < 0) or torch.any(current_train_nodes >= node_count):
        raise ValueError("current train nodes are outside the strict-local graph.")
    if torch.unique(current_train_nodes).numel() != current_train_nodes.numel():
        raise ValueError("current train nodes must be unique.")
    if torch.any(current_train_labels < 0) or torch.any(current_train_labels >= class_count):
        raise ValueError("current train labels are outside the model class range.")
    if edge_index.numel() and (torch.any(edge_index < 0) or torch.any(edge_index >= node_count)):
        raise ValueError("edge_index contains a non-local endpoint.")
    if not isinstance(propagation_steps, int) or propagation_steps <= 0:
        raise ValueError("propagation_steps must be a positive integer.")
    if not 0.0 <= _require_finite_float(propagation_alpha, name="propagation_alpha") <= 1.0:
        raise ValueError("propagation_alpha must lie in [0, 1].")
    if _require_finite_float(temperature, name="temperature") <= 0.0:
        raise ValueError("temperature must be positive.")

    device = logits.device
    edge_index = edge_index.to(device=device)
    train_nodes = current_train_nodes.to(device=device)
    train_labels = current_train_labels.to(device=device)
    # The reference propagation normalizes an adjacency with self loops, while
    # Eq. (4)'s confidence degree is computed from the original local graph.
    # UEFA client graphs already contain reverse arcs for undirected datasets.
    self_nodes = torch.arange(node_count, device=device, dtype=torch.long)
    raw_source = edge_index[0]
    raw_destination = edge_index[1]
    degrees = torch.bincount(
        raw_destination, minlength=node_count
    ).to(dtype=logits.dtype)
    source = torch.cat((raw_source, self_nodes))
    destination = torch.cat((raw_destination, self_nodes))
    normalized_degree = torch.bincount(
        destination, minlength=node_count
    ).to(dtype=logits.dtype)
    if torch.any(normalized_degree <= 0):  # Self loops make this defensive.
        raise RuntimeError("Self-loop normalized propagation produced a zero degree.")
    norm = (
        normalized_degree[destination] * normalized_degree[source]
    ).rsqrt()

    seed = torch.softmax(logits, dim=1)
    seed = seed.clone()
    seed[train_nodes] = torch.nn.functional.one_hot(
        train_labels, num_classes=class_count
    ).to(dtype=logits.dtype)
    current = seed
    states: list[torch.Tensor] = []
    for _ in range(propagation_steps):
        propagated = torch.zeros_like(current)
        propagated.index_add_(0, destination, current[source] * norm.unsqueeze(1))
        current = float(propagation_alpha) * seed + (
            1.0 - float(propagation_alpha)
        ) * propagated
        current = current.clone()
        current[train_nodes] = seed[train_nodes]
        moment_state = torch.softmax(current, dim=1)
        moment_state = moment_state.clone()
        moment_state[train_nodes] = seed[train_nodes]
        states.append(moment_state)
    confidence_labels = torch.softmax(current / float(temperature), dim=1)
    confidence_labels = confidence_labels.clone()
    confidence_labels[train_nodes] = seed[train_nodes]
    return confidence_labels, tuple(states), degrees


def fedgta_moments(
    propagated_states: Sequence[torch.Tensor],
    *,
    moment_order: int,
    moment_type: str,
) -> torch.Tensor:
    """Compute FedGTA origin, centered, or hybrid moments for every LP hop."""

    if not isinstance(moment_order, int) or moment_order <= 0:
        raise ValueError("moment_order must be a positive integer.")
    if moment_type not in {"origin", "mean", "hybrid"}:
        raise ValueError("moment_type must be one of: origin, mean, hybrid.")
    if not propagated_states:
        raise ValueError("FedGTA requires at least one propagated state.")
    shape: tuple[int, int] | None = None
    origin: list[torch.Tensor] = []
    centered: list[torch.Tensor] = []
    for state in propagated_states:
        if (
            state.ndim != 2
            or not state.is_floating_point()
            or not torch.isfinite(state).all()
        ):
            raise ValueError("propagated states must be finite floating matrices.")
        if shape is None:
            shape = tuple(state.shape)
        elif tuple(state.shape) != shape:
            raise ValueError("all propagated states must share a shape.")
        state_centered = state - state.mean(dim=0, keepdim=True)
        origin.extend(
            torch.mean(state.pow(power), dim=0)
            for power in range(1, moment_order + 1)
        )
        centered.extend(
            torch.mean(state_centered.pow(power), dim=0)
            for power in range(1, moment_order + 1)
        )
    if moment_type == "origin":
        selected = origin
    elif moment_type == "mean":
        selected = centered
    else:
        # This ordering matches compute_moment(..., moment_type="hybrid") in
        # the official repository: all origin moments, then all centered ones.
        selected = origin + centered
    return torch.cat(selected, dim=0)


def fedgta_origin_moments(
    propagated_states: Sequence[torch.Tensor], *, moment_order: int
) -> torch.Tensor:
    """Backward-compatible wrapper for official origin moments."""

    return fedgta_moments(
        propagated_states,
        moment_order=moment_order,
        moment_type="origin",
    )

def fedgta_smoothing_confidence(
    *, propagated_labels: torch.Tensor, degrees: torch.Tensor
) -> torch.Tensor:
    """Compute Eq. (4)'s reversed entropy confidence without label access."""

    if propagated_labels.ndim != 2 or not propagated_labels.is_floating_point():
        raise ValueError("propagated_labels must be a floating [nodes, classes] tensor.")
    if degrees.ndim != 1 or degrees.shape[0] != propagated_labels.shape[0]:
        raise ValueError("degrees must be a per-node vector for propagated_labels.")
    if not torch.isfinite(propagated_labels).all() or not torch.isfinite(degrees).all():
        raise ValueError("FedGTA confidence inputs must be finite.")
    if torch.any(propagated_labels < 0) or torch.any(degrees < 0):
        raise ValueError("FedGTA confidence inputs must be non-negative.")
    probabilities = propagated_labels / propagated_labels.sum(dim=1, keepdim=True).clamp_min(torch.finfo(propagated_labels.dtype).tiny)
    entropy_term = (
        probabilities
        * probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()
    ).sum(dim=1)
    class_count = probabilities.shape[1]
    return torch.sum(
        degrees.to(dtype=probabilities.dtype)
        * (class_count * math.exp(-1.0) + entropy_term)
    )


def fedgta_personalized_weights(
    *,
    moments_by_client: Mapping[int, torch.Tensor],
    confidence_by_client: Mapping[int, torch.Tensor | float],
    similarity_threshold: float,
) -> dict[int, dict[int, torch.Tensor]]:
    """Return Eq. (6)--(7) personalized confidence-normalized weights.

    The mapping is intentionally round-scoped: inactive clients are absent and
    thus cannot participate in a new aggregate.  The caller preserves their
    previous personalized state instead.
    """

    threshold = _require_finite_float(similarity_threshold, name="similarity_threshold")
    if not moments_by_client or set(moments_by_client) != set(confidence_by_client):
        raise ValueError("moments and confidences must cover exactly the active clients.")
    client_ids = tuple(sorted(moments_by_client))
    if any(not isinstance(client_id, int) or client_id < 0 for client_id in client_ids):
        raise ValueError("FedGTA client IDs must be non-negative integers.")
    reference_shape: tuple[int, ...] | None = None
    vectors: dict[int, torch.Tensor] = {}
    confidences: dict[int, torch.Tensor] = {}
    for client_id in client_ids:
        vector = moments_by_client[client_id]
        if vector.ndim != 1 or not vector.is_floating_point() or not torch.isfinite(vector).all():
            raise ValueError("FedGTA moment vectors must be finite floating 1-D tensors.")
        if reference_shape is None:
            reference_shape = tuple(vector.shape)
        elif tuple(vector.shape) != reference_shape:
            raise ValueError("FedGTA moment vectors must share a shape.")
        norm = torch.linalg.vector_norm(vector)
        if norm <= 0:
            raise ValueError("FedGTA moment vectors must have non-zero norm.")
        vectors[client_id] = vector
        confidence = torch.as_tensor(confidence_by_client[client_id], dtype=vector.dtype, device=vector.device)
        if (
            confidence.numel() != 1
            or not torch.isfinite(confidence)
            or confidence <= 0
        ):
            raise ValueError("FedGTA confidences must be positive finite scalars.")
        confidences[client_id] = confidence.reshape(())
    weights: dict[int, dict[int, torch.Tensor]] = {}
    for target in client_ids:
        eligible = [
            source for source in client_ids
            if torch.dot(vectors[target], vectors[source])
            / (torch.linalg.vector_norm(vectors[target]) * torch.linalg.vector_norm(vectors[source]))
            >= threshold
        ]
        if target not in eligible:
            eligible.append(target)
        denominator = torch.stack([confidences[source] for source in eligible]).sum()
        if torch.isclose(denominator, torch.zeros_like(denominator)):
            raise ValueError("FedGTA confidence sum is zero for an aggregation set.")
        weights[target] = {
            source: confidences[source] / denominator for source in sorted(eligible)
        }
    return weights


def fedgta_personalized_aggregate(
    *,
    model_states: Mapping[int, Mapping[str, torch.Tensor]],
    personalized_weights: Mapping[int, Mapping[int, torch.Tensor]],
) -> dict[int, dict[str, torch.Tensor]]:
    """Aggregate full model states with prevalidated personalized weights."""

    if not model_states or set(model_states) != set(personalized_weights):
        raise ValueError("FedGTA states and target weights must cover the same clients.")
    keys = None
    for client_id, state in model_states.items():
        if not state:
            raise ValueError(f"FedGTA model state for client {client_id} is empty.")
        if keys is None:
            keys = tuple(sorted(state))
        elif tuple(sorted(state)) != keys:
            raise ValueError("FedGTA model-state keys must match.")
    assert keys is not None
    output: dict[int, dict[str, torch.Tensor]] = {}
    for target, weights in personalized_weights.items():
        if not weights or not set(weights).issubset(model_states):
            raise ValueError("FedGTA aggregation weights reference an unknown client.")
        if target not in weights:
            raise ValueError("FedGTA target must retain its own local model in the aggregate.")
        output[target] = {}
        for key in keys:
            reference = model_states[target][key]
            if not reference.is_floating_point():
                raise ValueError("FedGTA currently supports floating shared model tensors only.")
            result = torch.zeros_like(reference)
            for source, weight in weights.items():
                tensor = model_states[source][key]
                if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                    raise ValueError("FedGTA model-state tensor shapes and dtypes must match.")
                result.add_(tensor.to(device=result.device), alpha=float(weight))
            output[target][key] = result
    return output

class FedGTAStrategy:
    """Persistent personalized FedGTA state, pending registry promotion.

    The strategy remains unregistered until its strict-local client statistic
    collection and full round lifecycle pass their dedicated gates.  This class
    owns the state identity now so checkpoint and inactive-client invariants do
    not become an afterthought when aggregation is wired in.
    """

    strategy_version = "uefa-fedgta-strategy-v2"
    name = "fedgta"
    aggregates = True
    oracle = False
    uses_proximal_objective = False

    def __init__(
        self,
        *,
        propagation_steps: int = 5,
        propagation_alpha: float = 0.5,
        temperature: float = 20.0,
        moment_order: int = 13,
        moment_type: str = "origin",
        similarity_threshold: float = 0.65,
    ) -> None:
        if not isinstance(propagation_steps, int) or propagation_steps <= 0:
            raise ValueError("FedGTA propagation_steps must be a positive integer.")
        if not isinstance(moment_order, int) or moment_order <= 0:
            raise ValueError("FedGTA moment_order must be a positive integer.")
        if moment_type not in {"origin", "mean", "hybrid"}:
            raise ValueError(
                "FedGTA moment_type must be one of: origin, mean, hybrid."
            )
        if not 0.0 <= _require_finite_float(
            propagation_alpha, name="propagation_alpha"
        ) <= 1.0:
            raise ValueError("FedGTA propagation_alpha must lie in [0, 1].")
        if _require_finite_float(temperature, name="temperature") <= 0.0:
            raise ValueError("FedGTA temperature must be positive.")
        _require_finite_float(similarity_threshold, name="similarity_threshold")
        self.propagation_steps = propagation_steps
        self.propagation_alpha = float(propagation_alpha)
        self.temperature = float(temperature)
        self.moment_order = moment_order
        self.moment_type = moment_type
        self.similarity_threshold = float(similarity_threshold)
        self._manifest: ParameterManifest | None = None
        self._client_ids: tuple[int, ...] = ()
        self._shared_state = FrozenTensorMap()
        self._personalized_states: dict[int, FrozenTensorMap] = {}
        self._last_statistics: dict[int, FrozenTensorMap] = {}
        self._completed_rounds = 0
        self._initialized = False

    def _require_initialized(self) -> None:
        if not self._initialized or self._manifest is None:
            raise RuntimeError("FedGTA has not been initialized.")

    def _validate_state(
        self, state: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self._manifest is None:
            raise RuntimeError("FedGTA parameter manifest is unavailable.")
        expected = tuple(self._manifest.shared_trainable)
        if set(state) != set(expected):
            raise ValueError("FedGTA state keys do not match shared trainables.")
        reference = self._shared_state.materialize()
        output: dict[str, torch.Tensor] = {}
        for key in expected:
            value = state[key]
            expected_value = reference[key]
            if (
                not torch.is_tensor(value)
                or value.shape != expected_value.shape
                or value.dtype != expected_value.dtype
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"FedGTA state tensor {key!r} is invalid.")
            output[key] = value.detach().clone().contiguous()
        return output

    def initialize(
        self,
        shared_state: TensorState,
        parameter_manifest: ParameterManifest,
        client_ids: Iterable[int],
    ) -> None:
        if self._initialized:
            raise RuntimeError("FedGTA cannot be initialized twice.")
        clients = tuple(sorted(int(client_id) for client_id in client_ids))
        if not clients or len(set(clients)) != len(clients) or any(
            client_id < 0 for client_id in clients
        ):
            raise ValueError("FedGTA client IDs must be unique non-negative integers.")
        expected = tuple(parameter_manifest.shared_trainable)
        if not expected or set(shared_state) != set(expected):
            raise ValueError("FedGTA requires every shared trainable parameter.")
        self._manifest = parameter_manifest
        self._client_ids = clients
        self._shared_state = FrozenTensorMap(shared_state)
        initial = self._validate_state(shared_state)
        self._shared_state = FrozenTensorMap(initial)
        self._personalized_states = {
            client_id: FrozenTensorMap(initial) for client_id in clients
        }
        self._initialized = True

    @property
    def shared_state(self) -> dict[str, torch.Tensor]:
        self._require_initialized()
        return self._shared_state.materialize()

    def personalized_state(self, client_id: int) -> dict[str, torch.Tensor]:
        self._require_initialized()
        if client_id not in self._personalized_states:
            raise ValueError("Unknown FedGTA client.")
        return self._personalized_states[client_id].materialize()

    def prepare_payload(
        self, context: RoundContext, client_id: int, reason: BroadcastReason
    ) -> BroadcastPayload:
        self._require_initialized()
        if client_id not in context.participant_ids or client_id not in self._client_ids:
            raise ValueError("FedGTA payload client is not a known participant.")
        model_state = self._personalized_states[client_id]
        resources = ResourceLedger()
        if reason == "initialization":
            resources.add(
                initialization_model_downlink_bytes=model_state.payload_bytes
            )
        elif reason == "training":
            resources.add(training_model_downlink_bytes=model_state.payload_bytes)
        elif reason == "evaluation":
            resources.add(evaluation_sync_bytes=model_state.payload_bytes)
        elif reason != "resume":
            raise ValueError(f"Unknown FedGTA broadcast reason {reason!r}.")
        return BroadcastPayload(
            client_id=client_id,
            round_context=context,
            reason=reason,
            model_state=model_state,
            metadata={
                "strategy_version": self.strategy_version,
                "personalized": True,
            },
            resources=resources,
        )

    def client_receive(
        self, client: Any, payload: BroadcastPayload, method_context: Any
    ) -> None:
        if int(client.client_id) != payload.client_id:
            raise ValueError("FedGTA payload was delivered to the wrong client.")
        client.load_shared_state(payload.model_state.materialize())
        client.algorithm.on_broadcast(method_context, payload)

    def transform_gradients(
        self, model: torch.nn.Module, method_context: Any, shared_keys: tuple[str, ...]
    ) -> None:
        return None

    def select_evaluation(
        self, client_id: int, stage_index: int
    ) -> EvaluationSelection:
        self._require_initialized()
        if client_id not in self._client_ids or stage_index < 0:
            raise ValueError("Unknown FedGTA evaluation client or stage.")
        return EvaluationSelection(
            client_id=client_id,
            source="personalized",
            model_state=self._personalized_states[client_id],
            count_evaluation_sync=True,
        )

    def diagnostics(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "fidelity": "mechanism_adaptation",
            "personalized": True,
            "propagation_steps": self.propagation_steps,
            "propagation_alpha": self.propagation_alpha,
            "temperature": self.temperature,
            "moment_order": self.moment_order,
            "moment_type": self.moment_type,
            "similarity_threshold": self.similarity_threshold,
            "last_aggregation_weights": self._diagnostic_aggregation_weights(),
            "completed_rounds": self._completed_rounds,
            "clients_with_last_known_statistics": sorted(self._last_statistics),
        }

    def _diagnostic_aggregation_weights(self) -> dict[str, dict[str, float]]:
        if not self._last_statistics:
            return {}
        moments = {
            client_id: state["moments"]
            for client_id, state in self._last_statistics.items()
        }
        confidences = {
            client_id: state["confidence"].reshape(())
            for client_id, state in self._last_statistics.items()
        }
        weights = fedgta_personalized_weights(
            moments_by_client=moments,
            confidence_by_client=confidences,
            similarity_threshold=self.similarity_threshold,
        )
        return {
            str(target): {
                str(source): float(weight.detach().cpu())
                for source, weight in source_weights.items()
            }
            for target, source_weights in weights.items()
        }

    def state_dict(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "client_ids": self._client_ids,
            "shared_state": self._shared_state.materialize(),
            "personalized_states": {
                client_id: state.materialize()
                for client_id, state in self._personalized_states.items()
            },
            "last_statistics": {
                client_id: state.materialize()
                for client_id, state in self._last_statistics.items()
            },
            "completed_rounds": self._completed_rounds,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._require_initialized()
        expected_fields = {
            "strategy_version",
            "client_ids",
            "shared_state",
            "personalized_states",
            "last_statistics",
            "completed_rounds",
        }
        if set(state) != expected_fields or state["strategy_version"] != self.strategy_version:
            raise ValueError("FedGTA checkpoint identity mismatch.")
        if tuple(state["client_ids"]) != self._client_ids:
            raise ValueError("FedGTA checkpoint client IDs mismatch.")
        shared = state["shared_state"]
        personalized = state["personalized_states"]
        statistics = state["last_statistics"]
        completed_rounds = state["completed_rounds"]
        if not isinstance(shared, Mapping) or not isinstance(personalized, Mapping):
            raise TypeError("FedGTA checkpoint states must be mappings.")
        if not isinstance(statistics, Mapping):
            raise TypeError("FedGTA checkpoint statistics must be a mapping.")
        if isinstance(completed_rounds, bool) or not isinstance(completed_rounds, int) or completed_rounds < 0:
            raise ValueError("FedGTA checkpoint completed rounds are invalid.")
        if set(personalized) != set(self._client_ids):
            raise ValueError("FedGTA checkpoint personalized clients mismatch.")
        self._shared_state = FrozenTensorMap(self._validate_state(shared))
        self._personalized_states = {
            int(client_id): FrozenTensorMap(self._validate_state(value))
            for client_id, value in personalized.items()
        }
        self._last_statistics = {
            int(client_id): FrozenTensorMap(value)
            for client_id, value in statistics.items()
        }
        self._completed_rounds = completed_rounds

    def _statistics_from_client(
        self, client: Any, context: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.problem_type != "NC":
            raise ValueError("FedGTA initial scope supports node classification only.")
        node_count = int(context.node_features.shape[0])
        queries = torch.arange(
            node_count, dtype=torch.long, device=context.node_features.device
        )
        with torch.no_grad():
            logits = context.forward_queries(client.model, queries)
        if context.valid_class_mask is not None:
            class_mask = context.valid_class_mask.to(logits.device)
            if logits.ndim != 2 or logits.shape[1] != class_mask.numel():
                raise ValueError("FedGTA class mask is incompatible with logits.")
            logits = logits.clone()
            logits[:, ~class_mask] = -1e12
        propagated, states, degrees = propagate_strict_local_labels(
            logits=logits,
            edge_index=context.effective_edge_index.to(logits.device),
            current_train_nodes=context.train_queries.to(logits.device),
            current_train_labels=context.train_labels.to(logits.device),
            propagation_steps=self.propagation_steps,
            propagation_alpha=self.propagation_alpha,
            temperature=self.temperature,
        )
        moments = fedgta_moments(
            states,
            moment_order=self.moment_order,
            moment_type=self.moment_type,
        )
        confidence = fedgta_smoothing_confidence(
            propagated_labels=propagated, degrees=degrees
        )
        return moments, confidence

    def finalize_upload(
        self, client: Any, local_result: Any, method_context: Any
    ) -> ClientUpload:
        self._require_initialized()
        client_id = int(client.client_id)
        if (
            local_result.client_id != client_id
            or local_result.global_task_id != int(method_context.global_task_id)
            or client_id not in self._client_ids
        ):
            raise ValueError("FedGTA local result identity mismatch.")
        model_state = self._validate_state(local_result.shared_state)
        moments, confidence = self._statistics_from_client(client, method_context)
        auxiliary = FrozenTensorMap(
            {
                "moments": moments.detach().clone().contiguous(),
                "confidence": confidence.detach().reshape(1).clone().contiguous(),
            }
        )
        self._last_statistics[client_id] = auxiliary
        resources = ResourceLedger(
            training_model_uplink_bytes=FrozenTensorMap(model_state).payload_bytes,
            training_auxiliary_uplink_bytes=auxiliary.payload_bytes,
        )
        return ClientUpload(
            client_id=client_id,
            global_task_id=local_result.global_task_id,
            weight=local_result.weight,
            training_loss=local_result.training_loss,
            model_state=model_state,
            auxiliary_state=auxiliary,
            diagnostics={
                "strategy_version": self.strategy_version,
                "propagation_steps": self.propagation_steps,
                "moment_order": self.moment_order,
                "confidence": float(confidence.detach().cpu()),
            },
            resources=resources,
        )

    def aggregate(
        self, context: RoundContext, uploads: tuple[ClientUpload, ...]
    ) -> AggregationResult:
        """Apply FedGTA Eq. (6)--(7) to current participants only."""

        self._require_initialized()
        upload_ids = tuple(upload.client_id for upload in uploads)
        if len(set(upload_ids)) != len(upload_ids) or set(upload_ids) != set(
            context.participant_ids
        ):
            raise ValueError("FedGTA requires exactly one upload per participant.")
        if any(
            upload.global_task_id != context.task_for(upload.client_id)
            for upload in uploads
        ):
            raise ValueError("FedGTA upload changed an immutable global task ID.")
        states = {
            upload.client_id: self._validate_state(upload.model_state)
            for upload in uploads
        }
        moments: dict[int, torch.Tensor] = {}
        confidences: dict[int, torch.Tensor] = {}
        resources = ResourceLedger()
        for upload in uploads:
            auxiliary = upload.auxiliary_state
            if set(auxiliary) != {"moments", "confidence"}:
                raise ValueError("FedGTA upload auxiliary state is malformed.")
            moment = auxiliary["moments"]
            confidence = auxiliary["confidence"]
            if moment.ndim != 1 or confidence.numel() != 1:
                raise ValueError("FedGTA upload statistics have invalid shapes.")
            moments[upload.client_id] = moment
            confidences[upload.client_id] = confidence.reshape(())
            self._last_statistics[upload.client_id] = FrozenTensorMap(
                {"moments": moment, "confidence": confidence.reshape(1)}
            )
            resources.merge(upload.resources)
        weights = fedgta_personalized_weights(
            moments_by_client=moments,
            confidence_by_client=confidences,
            similarity_threshold=self.similarity_threshold,
        )
        active_personalized = fedgta_personalized_aggregate(
            model_states=states, personalized_weights=weights
        )
        updated = dict(self._personalized_states)
        updated.update(
            {
                client_id: FrozenTensorMap(state)
                for client_id, state in active_personalized.items()
            }
        )
        # This diagnostic shared state is never selected for FedGTA training or
        # evaluation.  Its unweighted construction avoids presenting it as a
        # generic query-weighted FedAvg aggregate.
        diagnostic_shared = {
            key: torch.stack(
                [active_personalized[client_id][key] for client_id in sorted(active_personalized)]
            ).mean(dim=0)
            for key in next(iter(active_personalized.values()))
        }
        self._personalized_states = updated
        self._shared_state = FrozenTensorMap(diagnostic_shared)
        self._completed_rounds += 1
        return AggregationResult(
            shared_state=self._shared_state,
            personalized_states=self._personalized_states,
            diagnostics={
                "strategy_version": self.strategy_version,
                "aggregation": "thresholded_confidence_weighted_personalized",
                "participants": list(sorted(context.participant_ids)),
                "similarity_threshold": self.similarity_threshold,
            },
            resources=resources,
        )

    def personalize(
        self, context: RoundContext, result: AggregationResult
    ) -> AggregationResult:
        return result
