"""FED-PUB strict-local personalized strategy primitives.

This module implements FED-PUB's random-graph functional embeddings,
similarity-weighted personalized aggregation, and client-private differentiable
sparse masks for UEFA v2.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

from gecko.engine.accounting import ResourceLedger
from gecko.engine.protocol import AggregationResult
from gecko.engine.protocol import BroadcastPayload
from gecko.engine.protocol import BroadcastReason
from gecko.engine.protocol import ClientUpload
from gecko.engine.protocol import EvaluationSelection
from gecko.engine.protocol import FrozenTensorMap
from gecko.engine.protocol import MethodArtifact
from gecko.engine.protocol import ParameterManifest
from gecko.engine.protocol import RoundContext
from gecko.engine.protocol import TensorState


def _finite_positive(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def build_fedpub_proxy_artifact(
    *,
    feature_dim: int,
    seed: int,
    blocks: int = 5,
    nodes_per_block: int = 100,
    p_in: float = 0.1,
    p_out: float = 0.01,
) -> MethodArtifact:
    """Build the deterministic unlabeled SBM proxy graph declared for FED-PUB."""

    if (
        isinstance(feature_dim, bool)
        or not isinstance(feature_dim, int)
        or feature_dim <= 0
    ):
        raise ValueError("feature_dim must be a positive integer.")
    if blocks != 5 or nodes_per_block != 100:
        raise ValueError("FED-PUB proxy must use five 100-node SBM blocks.")
    if float(p_in) != 0.1 or float(p_out) != 0.01:
        raise ValueError("FED-PUB proxy probabilities must be p_in=0.1, p_out=0.01.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("FED-PUB proxy seed must be an integer.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    node_count = blocks * nodes_per_block
    features = torch.randn(node_count, feature_dim, generator=generator)
    block_ids = torch.arange(node_count, dtype=torch.long) // nodes_per_block
    edges: list[tuple[int, int]] = []
    for source in range(node_count):
        for target in range(source + 1, node_count):
            probability = p_in if block_ids[source] == block_ids[target] else p_out
            if torch.rand((), generator=generator).item() < probability:
                edges.append((source, target))
                edges.append((target, source))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return MethodArtifact.build(
        artifact_type="fed-pub-proxy-sbm",
        version="uefa-fed-pub-proxy-v1",
        payload={
            "features": features,
            "edge_index": edge_index,
            "block_ids": block_ids,
        },
        metadata={
            "seed": seed,
            "blocks": blocks,
            "nodes_per_block": nodes_per_block,
            "p_in": p_in,
            "p_out": p_out,
            "labels": False,
            "feature_dim": feature_dim,
        },
        serialized_bytes=features.numel() * features.element_size()
        + edge_index.numel() * edge_index.element_size()
        + block_ids.numel() * block_ids.element_size(),
    )


def fedpub_effective_state(
    state: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Return effective FED-PUB parameters ``theta * mask``."""

    if set(state) != set(masks):
        raise ValueError("FED-PUB model state and mask keys must match.")
    output: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        mask = masks[key]
        if (
            tensor.shape != mask.shape
            or tensor.dtype != mask.dtype
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
            or not torch.isfinite(mask).all()
        ):
            raise ValueError(f"FED-PUB tensor/mask {key!r} is invalid.")
        effective_mask = (
            mask.detach()
            .to(device=tensor.device, dtype=tensor.dtype)
            .clone()
            .contiguous()
        )
        output[key] = tensor.detach().clone().contiguous() * effective_mask
    return output


def fedpub_similarity_weights(
    proxy_embeddings: Mapping[int, torch.Tensor], *, tau: float = 3.0
) -> dict[int, dict[int, torch.Tensor]]:
    """Return stable row-wise softmax similarities among active clients."""

    tau = _finite_positive(tau, name="tau")
    if not proxy_embeddings:
        raise ValueError("FED-PUB requires active proxy embeddings.")
    client_ids = tuple(sorted(int(client_id) for client_id in proxy_embeddings))
    vectors: dict[int, torch.Tensor] = {}
    shape: tuple[int, ...] | None = None
    for client_id in client_ids:
        value = proxy_embeddings[client_id]
        if value.ndim != 1 or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("FED-PUB proxy embeddings must be finite 1-D tensors.")
        if shape is None:
            shape = tuple(value.shape)
        elif tuple(value.shape) != shape:
            raise ValueError("FED-PUB proxy embeddings must share a shape.")
        norm = torch.linalg.vector_norm(value)
        if norm <= 0:
            raise ValueError("FED-PUB proxy embeddings must have non-zero norm.")
        vectors[client_id] = value
    weights: dict[int, dict[int, torch.Tensor]] = {}
    for target in client_ids:
        logits = []
        for source in client_ids:
            score = torch.nn.functional.cosine_similarity(
                vectors[target], vectors[source], dim=0
            )
            logits.append(score * tau)
        raw = torch.stack(logits)
        probs = torch.softmax(raw - raw.max(), dim=0)
        weights[target] = {
            source: probs[index] for index, source in enumerate(client_ids)
        }
    return weights


def fedpub_personalized_aggregate(
    *,
    effective_states: Mapping[int, Mapping[str, torch.Tensor]],
    weights: Mapping[int, Mapping[int, torch.Tensor]],
) -> dict[int, dict[str, torch.Tensor]]:
    """Aggregate effective masked states into one personalized model per target."""

    if not effective_states or set(effective_states) != set(weights):
        raise ValueError("FED-PUB states and weights must cover the same clients.")
    keys: tuple[str, ...] | None = None
    for state in effective_states.values():
        if keys is None:
            keys = tuple(sorted(state))
        elif tuple(sorted(state)) != keys:
            raise ValueError("FED-PUB state keys must match.")
    assert keys is not None
    output: dict[int, dict[str, torch.Tensor]] = {}
    for target, row in weights.items():
        if set(row) != set(effective_states):
            raise ValueError("FED-PUB weights must reference exactly active clients.")
        output[target] = {}
        for key in keys:
            reference = effective_states[target][key]
            result = torch.zeros_like(reference)
            for source, weight in row.items():
                result.add_(effective_states[source][key].to(result.device), alpha=float(weight))
            output[target][key] = result
    return output


class _FedPUBFunctionalModel(nn.Module):
    """Stateless-call wrapper that preserves the public UEFA model API."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor | None,
        problem_type: str,
        encode_only: bool,
    ) -> torch.Tensor:
        if encode_only:
            encode_nodes = getattr(self.model, "encode_nodes", None)
            if not callable(encode_nodes):
                raise RuntimeError("FED-PUB requires a verified node encoder.")
            return encode_nodes(features, edge_index, layer_index=None)
        if queries is None:
            raise ValueError("FED-PUB query forward requires query indices.")
        return self.model.forward_queries(features, edge_index, queries, problem_type)


class FedPUBStrategy:
    """Persistent FED-PUB proxy/mask/personalized aggregation strategy."""

    strategy_version = "uefa-fed-pub-strategy-v3"
    name = "fed_pub"
    aggregates = True
    oracle = False
    uses_proximal_objective = False
    # The official client performs multiple local optimizer updates before an
    # upload.  UEFA graph shards are full-batch and the benchmark budget uses
    # one local epoch, so one update would evaluate ||theta-theta_personal||_2
    # only at the freshly loaded personalized state, where both its value and
    # gradient are identically zero.  Two disclosed inner updates are the
    # minimum faithful adaptation that lets the second update exercise the
    # published proximity term and gives the persistent L1 mask optimizer a
    # real sparsification trajectory.
    local_optimizer_steps_per_epoch = 2

    def __init__(
        self,
        *,
        tau: float = 3.0,
        lambda1: float = 1e-3,
        lambda2: float = 1e-3,
        proxy_seed: int = 0,
    ) -> None:
        self.tau = _finite_positive(tau, name="tau")
        self.lambda1 = _finite_positive(lambda1, name="lambda1")
        self.lambda2 = _finite_positive(lambda2, name="lambda2")
        if isinstance(proxy_seed, bool) or not isinstance(proxy_seed, int):
            raise ValueError("FED-PUB proxy_seed must be an integer.")
        self.proxy_seed = proxy_seed
        self._manifest: ParameterManifest | None = None
        self._client_ids: tuple[int, ...] = ()
        self._shared_state = FrozenTensorMap()
        self._personalized_states: dict[int, FrozenTensorMap] = {}
        self._masks: dict[int, FrozenTensorMap] = {}
        self._last_proxy_embeddings: dict[int, FrozenTensorMap] = {}
        self._last_loss_terms: dict[int, tuple[float, float]] = {}
        self._last_mask_updates: dict[int, tuple[float, float]] = {}
        self._active_masks: dict[int, dict[str, torch.Tensor]] = {}
        self._mask_step_start: dict[int, dict[str, torch.Tensor]] = {}
        self._proxy_artifact: MethodArtifact | None = None
        self._completed_rounds = 0
        self._initialized = False

    def _require_initialized(self) -> None:
        if not self._initialized or self._manifest is None or self._proxy_artifact is None:
            raise RuntimeError("FED-PUB has not been initialized.")

    def _validate_state(self, state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._manifest is None:
            raise RuntimeError("FED-PUB manifest is unavailable.")
        expected = tuple(self._manifest.shared_trainable)
        if set(state) != set(expected):
            raise ValueError("FED-PUB state keys do not match shared trainables.")
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
                raise ValueError(f"FED-PUB state tensor {key!r} is invalid.")
            output[key] = value.detach().clone().contiguous()
        return output

    def initialize(
        self,
        shared_state: TensorState,
        parameter_manifest: ParameterManifest,
        client_ids: Iterable[int],
    ) -> None:
        if self._initialized:
            raise RuntimeError("FED-PUB cannot be initialized twice.")
        clients = tuple(sorted(int(client_id) for client_id in client_ids))
        if not clients or len(set(clients)) != len(clients) or any(client_id < 0 for client_id in clients):
            raise ValueError("FED-PUB client IDs must be unique non-negative integers.")
        expected = tuple(parameter_manifest.shared_trainable)
        if not expected or set(shared_state) != set(expected):
            raise ValueError("FED-PUB requires every shared trainable parameter.")
        first_tensor = next(iter(shared_state.values()))
        if first_tensor.ndim < 2:
            raise ValueError("FED-PUB could not infer a feature dimension from the model.")
        feature_dim = int(first_tensor.shape[0])
        self._manifest = parameter_manifest
        self._client_ids = clients
        initial = {key: value.detach().clone().contiguous() for key, value in shared_state.items()}
        self._shared_state = FrozenTensorMap(initial)
        self._personalized_states = {client_id: FrozenTensorMap(initial) for client_id in clients}
        self._masks = {
            client_id: FrozenTensorMap({key: torch.ones_like(value) for key, value in initial.items()})
            for client_id in clients
        }
        self._proxy_artifact = build_fedpub_proxy_artifact(
            feature_dim=feature_dim, seed=self.proxy_seed
        )
        self._initialized = True

    def trainable_parameters(
        self,
        model: torch.nn.Module,
        method_context: Any,
        shared_keys: tuple[str, ...],
    ) -> tuple[nn.Parameter, ...]:
        """Materialize this client's private mask parameters for local Adam."""

        self._require_initialized()
        client_id = int(method_context.client_id)
        persistent = self._masks[client_id].materialize()
        parameters = dict(model.named_parameters())
        if set(shared_keys) != set(persistent):
            raise ValueError("FED-PUB mask/shared parameter keys changed.")
        active: dict[str, torch.Tensor] = {}
        trainable: list[nn.Parameter] = []
        start: dict[str, torch.Tensor] = {}
        for name in shared_keys:
            value = persistent[name].to(parameters[name].device, parameters[name].dtype)
            start[name] = value.detach().clone()
            if parameters[name].ndim >= 2:
                mask = nn.Parameter(value.detach().clone(), requires_grad=True)
                trainable.append(mask)
                active[name] = mask
            else:
                active[name] = value.detach().clone()
        self._active_masks[client_id] = active
        self._mask_step_start[client_id] = start
        return tuple(trainable)

    def _functional_forward(
        self,
        model: torch.nn.Module,
        *,
        client_id: int,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor | None,
        problem_type: str,
        encode_only: bool,
        prune: bool,
    ) -> torch.Tensor:
        if client_id not in self._active_masks:
            raise RuntimeError("FED-PUB active masks are unavailable for local forward.")
        active = self._active_masks[client_id]
        replacements: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if name in active:
                mask = active[name]
                if prune:
                    mask = mask.masked_fill(mask.abs() < self.lambda1, 0.0)
                replacements[f"model.{name}"] = parameter * mask
        wrapper = _FedPUBFunctionalModel(model)
        return functional_call(
            wrapper,
            replacements,
            (
                features,
                edge_index,
                queries,
                problem_type,
                encode_only,
            ),
            strict=False,
        )

    def forward_queries(
        self,
        model: torch.nn.Module,
        method_context: Any,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Run the local objective through the differentiable masked model."""

        device = next(model.parameters()).device
        return self._functional_forward(
            model,
            client_id=int(method_context.client_id),
            features=method_context.node_features.to(device),
            edge_index=method_context.effective_edge_index.to(device),
            queries=queries.to(device),
            problem_type=str(method_context.problem_type),
            encode_only=False,
            prune=False,
        )

    @property
    def shared_state(self) -> dict[str, torch.Tensor]:
        self._require_initialized()
        return self._shared_state.materialize()

    def prepare_payload(
        self, context: RoundContext, client_id: int, reason: BroadcastReason
    ) -> BroadcastPayload:
        self._require_initialized()
        if client_id not in context.participant_ids or client_id not in self._client_ids:
            raise ValueError("FED-PUB payload client is not a known participant.")
        model_state = self._personalized_states[client_id]
        resources = ResourceLedger()
        if reason == "initialization":
            resources.add(
                initialization_model_downlink_bytes=model_state.payload_bytes,
                initialization_auxiliary_downlink_bytes=self._proxy_artifact.payload.payload_bytes,
                artifact_distribution_bytes=self._proxy_artifact.serialized_bytes,
            )
        elif reason == "training":
            resources.add(training_model_downlink_bytes=model_state.payload_bytes)
        elif reason == "evaluation":
            resources.add(evaluation_sync_bytes=model_state.payload_bytes)
        elif reason != "resume":
            raise ValueError(f"Unknown FED-PUB broadcast reason {reason!r}.")
        return BroadcastPayload(
            client_id=client_id,
            round_context=context,
            reason=reason,
            model_state=model_state,
            auxiliary_state=self._proxy_artifact.payload if reason == "initialization" else FrozenTensorMap(),
            metadata={
                "strategy_version": self.strategy_version,
                "personalized": True,
                "proxy_sha256": self._proxy_artifact.sha256,
            },
            resources=resources,
        )

    def client_receive(self, client: Any, payload: BroadcastPayload, method_context: Any) -> None:
        if int(client.client_id) != payload.client_id:
            raise ValueError("FED-PUB payload was delivered to the wrong client.")
        client.load_shared_state(payload.model_state.materialize())
        client.algorithm.on_broadcast(method_context, payload)

    def augment_loss(
        self,
        model: torch.nn.Module,
        method_context: Any,
        loss: torch.Tensor,
        shared_keys: tuple[str, ...],
    ) -> torch.Tensor:
        """Add FED-PUB mask L1 and personalized-base L2 proximity terms.

        Masks are private optimizer parameters and never enter the wire payload.
        """

        self._require_initialized()
        client_id = int(method_context.client_id)
        if client_id not in self._active_masks:
            raise RuntimeError("FED-PUB active masks are unavailable for loss augmentation.")
        mask_state = self._active_masks[client_id]
        base_state = self._personalized_states[client_id].materialize()
        l1_term = loss.new_zeros(())
        l2_term = loss.new_zeros(())
        named_parameters = dict(model.named_parameters())
        for name in shared_keys:
            parameter = named_parameters[name]
            mask = mask_state[name]
            base = base_state[name].to(device=parameter.device, dtype=parameter.dtype)
            if parameter.ndim >= 2:
                l1_term = l1_term + mask.abs().sum()
            # Match the official client objective: sum one Euclidean norm per
            # model tensor, rather than an unnormalised sum of squared entries.
            l2_term = l2_term + torch.linalg.vector_norm(parameter - base)
        self._last_loss_terms[client_id] = (
            float(l1_term.detach().cpu()),
            float(l2_term.detach().cpu()),
        )
        return loss + self.lambda1 * l1_term + self.lambda2 * l2_term

    def after_backward(
        self,
        model: torch.nn.Module,
        method_context: Any,
        shared_keys: tuple[str, ...],
        learning_rate: float,
    ) -> None:
        """Fail closed on invalid private-mask gradients before optimizer step."""

        self._require_initialized()
        client_id = int(method_context.client_id)
        for name, mask in self._active_masks.get(client_id, {}).items():
            if mask.requires_grad and (
                mask.grad is None or not torch.isfinite(mask.grad).all()
            ):
                raise FloatingPointError(
                    f"FED-PUB mask gradient for {name!r} is missing or non-finite."
                )

    def transform_gradients(
        self, model: torch.nn.Module, method_context: Any, shared_keys: tuple[str, ...]
    ) -> None:
        self._require_initialized()
        return None

    def after_optimizer_step(
        self,
        model: torch.nn.Module,
        method_context: Any,
        shared_keys: tuple[str, ...],
    ) -> None:
        """Persist private masks after the shared reset-per-round Adam step."""

        client_id = int(method_context.client_id)
        active = self._active_masks[client_id]
        start = self._mask_step_start[client_id]
        updated: dict[str, torch.Tensor] = {}
        total_delta = 0.0
        total_entries = 0
        active_entries = 0
        for name in shared_keys:
            mask = active[name].detach()
            if not torch.isfinite(mask).all():
                raise FloatingPointError(f"FED-PUB mask {name!r} became non-finite.")
            delta = (mask - start[name]).abs()
            total_delta += float(delta.sum().cpu())
            total_entries += int(delta.numel())
            active_entries += int((mask.abs() >= self.lambda1).sum().cpu())
            updated[name] = mask.cpu().clone().contiguous()
        self._masks[client_id] = FrozenTensorMap(updated)
        self._last_mask_updates[client_id] = (
            total_delta / max(total_entries, 1),
            active_entries / max(total_entries, 1),
        )

    def _proxy_embedding_from_client(self, client: Any) -> torch.Tensor:
        self._require_initialized()
        assert self._proxy_artifact is not None
        payload = self._proxy_artifact.payload.materialize()
        features = payload["features"]
        edges = payload["edge_index"]
        device = next(client.model.parameters()).device
        with torch.no_grad():
            hidden = self._functional_forward(
                client.model,
                client_id=int(client.client_id),
                features=features.to(device),
                edge_index=edges.to(device),
                queries=None,
                problem_type="NC",
                encode_only=True,
                prune=True,
            )
        return hidden.detach().mean(dim=0).clone().contiguous()

    def finalize_upload(self, client: Any, local_result: Any, method_context: Any) -> ClientUpload:
        self._require_initialized()
        client_id = int(client.client_id)
        if (
            local_result.client_id != client_id
            or local_result.global_task_id != int(method_context.global_task_id)
            or client_id not in self._client_ids
        ):
            raise ValueError("FED-PUB local result identity mismatch.")
        model_state = self._validate_state(local_result.shared_state)
        mask_state = self._masks[client_id].materialize()
        embedding = self._proxy_embedding_from_client(client)
        auxiliary = FrozenTensorMap({"proxy_embedding": embedding})
        self._last_proxy_embeddings[client_id] = auxiliary
        resources = ResourceLedger(
            training_model_uplink_bytes=FrozenTensorMap(model_state).payload_bytes,
            training_auxiliary_uplink_bytes=auxiliary.payload_bytes,
            client_persistent_bytes=FrozenTensorMap(mask_state).payload_bytes,
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
                "proxy_embedding_norm": float(torch.linalg.vector_norm(embedding).detach().cpu()),
            },
            resources=resources,
        )

    def aggregate(self, context: RoundContext, uploads: tuple[ClientUpload, ...]) -> AggregationResult:
        self._require_initialized()
        upload_ids = tuple(upload.client_id for upload in uploads)
        if len(set(upload_ids)) != len(upload_ids) or set(upload_ids) != set(context.participant_ids):
            raise ValueError("FED-PUB requires exactly one upload per participant.")
        if any(upload.global_task_id != context.task_for(upload.client_id) for upload in uploads):
            raise ValueError("FED-PUB upload changed an immutable global task ID.")
        states = {upload.client_id: self._validate_state(upload.model_state) for upload in uploads}
        embeddings: dict[int, torch.Tensor] = {}
        resources = ResourceLedger()
        for upload in uploads:
            auxiliary = upload.auxiliary_state
            if set(auxiliary) != {"proxy_embedding"}:
                raise ValueError("FED-PUB upload auxiliary state is malformed.")
            embedding = auxiliary["proxy_embedding"]
            embeddings[upload.client_id] = embedding
            self._last_proxy_embeddings[upload.client_id] = FrozenTensorMap(
                {"proxy_embedding": embedding}
            )
            resources.merge(upload.resources)
        weights = fedpub_similarity_weights(embeddings, tau=self.tau)
        active_personalized = fedpub_personalized_aggregate(
            effective_states=states, weights=weights
        )
        updated = dict(self._personalized_states)
        updated.update(
            {client_id: FrozenTensorMap(state) for client_id, state in active_personalized.items()}
        )
        total_weight = sum(upload.weight for upload in uploads)
        if total_weight <= 0:
            raise ValueError("FED-PUB aggregate weight must be positive.")
        diagnostic_shared = {
            key: sum(
                (
                    states[upload.client_id][key]
                    * (float(upload.weight) / float(total_weight))
                    for upload in uploads
                ),
                torch.zeros_like(states[uploads[0].client_id][key]),
            )
            for key in next(iter(states.values()))
        }
        self._personalized_states = updated
        self._shared_state = FrozenTensorMap(diagnostic_shared)
        self._completed_rounds += 1
        resources.add(
            personalized_model_bytes=sum(
                state.payload_bytes for state in self._personalized_states.values()
            ),
            method_artifact_bytes=self._proxy_artifact.serialized_bytes,
            server_persistent_bytes=sum(
                state.payload_bytes for state in self._last_proxy_embeddings.values()
            ),
        )
        return AggregationResult(
            shared_state=self._shared_state,
            personalized_states=self._personalized_states,
            diagnostics={
                "strategy_version": self.strategy_version,
                "aggregation": "proxy_similarity_softmax_personalized",
                "participants": list(sorted(context.participant_ids)),
                "tau": self.tau,
                "proxy_sha256": self._proxy_artifact.sha256,
            },
            resources=resources,
        )

    def personalize(self, context: RoundContext, result: AggregationResult) -> AggregationResult:
        return result

    def select_evaluation(self, client_id: int, stage_index: int) -> EvaluationSelection:
        self._require_initialized()
        if client_id not in self._client_ids or stage_index < 0:
            raise ValueError("Unknown FED-PUB evaluation client or stage.")
        return EvaluationSelection(
            client_id=client_id,
            source="personalized",
            model_state=FrozenTensorMap(
                fedpub_effective_state(
                    self._personalized_states[client_id].materialize(),
                    {
                        name: mask.masked_fill(mask.abs() < self.lambda1, 0.0)
                        for name, mask in self._masks[client_id].materialize().items()
                    },
                )
            ),
            count_evaluation_sync=True,
        )

    def diagnostics(self) -> Mapping[str, object]:
        self._require_initialized()
        assert self._proxy_artifact is not None
        return {
            "strategy_version": self.strategy_version,
            "fidelity": "paper_faithful_benchmark_adaptation",
            "personalized": True,
            "tau": self.tau,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "proxy_seed": self.proxy_seed,
            "proxy_sha256": self._proxy_artifact.sha256,
            "proxy_serialized_bytes": self._proxy_artifact.serialized_bytes,
            "mask_optimizer": "private_reset_per_round_adam",
            "local_optimizer_steps_per_epoch": self.local_optimizer_steps_per_epoch,
            "proximity_norm": "sum_per_tensor_l2",
            "mask_forward": "differentiable_weight_times_mask",
            "mask_evaluation_pruning": "absolute_mask_below_lambda1_zero",
            "functional_embedding": "mean_final_encoder_on_unlabeled_proxy",
            "completed_rounds": self._completed_rounds,
            "clients_with_last_known_proxy_embeddings": sorted(self._last_proxy_embeddings),
            "clients_with_last_loss_terms": sorted(self._last_loss_terms),
            "clients_with_last_mask_updates": sorted(self._last_mask_updates),
            "last_loss_terms": {
                f"client-{client_id}": {"mask_l1": values[0], "proximity_l2": values[1]}
                for client_id, values in sorted(self._last_loss_terms.items())
            },
            "last_mask_updates": {
                f"client-{client_id}": {"mean_abs_delta": values[0], "active_fraction": values[1]}
                for client_id, values in sorted(self._last_mask_updates.items())
            },
        }

    def state_dict(self) -> Mapping[str, object]:
        self._require_initialized()
        assert self._proxy_artifact is not None
        return {
            "strategy_version": self.strategy_version,
            "client_ids": self._client_ids,
            "shared_state": self._shared_state.materialize(),
            "personalized_states": {
                client_id: state.materialize()
                for client_id, state in self._personalized_states.items()
            },
            "masks": {
                client_id: state.materialize() for client_id, state in self._masks.items()
            },
            "last_proxy_embeddings": {
                client_id: state.materialize()
                for client_id, state in self._last_proxy_embeddings.items()
            },
            "last_loss_terms": self._last_loss_terms,
            "last_mask_updates": self._last_mask_updates,
            "proxy_artifact": {
                "sha256": self._proxy_artifact.sha256,
                "payload": self._proxy_artifact.payload.materialize(),
                "metadata": self._proxy_artifact.metadata_dict(),
                "serialized_bytes": self._proxy_artifact.serialized_bytes,
            },
            "completed_rounds": self._completed_rounds,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._require_initialized()
        expected = {
            "strategy_version",
            "client_ids",
            "shared_state",
            "personalized_states",
            "masks",
            "last_proxy_embeddings",
            "last_loss_terms",
            "last_mask_updates",
            "proxy_artifact",
            "completed_rounds",
        }
        if set(state) != expected or state["strategy_version"] != self.strategy_version:
            raise ValueError("FED-PUB checkpoint identity mismatch.")
        if tuple(state["client_ids"]) != self._client_ids:
            raise ValueError("FED-PUB checkpoint client IDs mismatch.")
        completed_rounds = state["completed_rounds"]
        if isinstance(completed_rounds, bool) or not isinstance(completed_rounds, int) or completed_rounds < 0:
            raise ValueError("FED-PUB checkpoint completed rounds are invalid.")
        proxy = state["proxy_artifact"]
        if not isinstance(proxy, Mapping):
            raise TypeError("FED-PUB checkpoint proxy artifact must be a mapping.")
        restored_proxy = MethodArtifact(
            artifact_type="fed-pub-proxy-sbm",
            version="uefa-fed-pub-proxy-v1",
            sha256=str(proxy["sha256"]),
            payload=proxy["payload"],
            metadata=proxy["metadata"],
            serialized_bytes=int(proxy["serialized_bytes"]),
        )
        if self._proxy_artifact is None or restored_proxy.sha256 != self._proxy_artifact.sha256:
            raise ValueError("FED-PUB checkpoint proxy artifact mismatch.")
        personalized = state["personalized_states"]
        masks = state["masks"]
        embeddings = state["last_proxy_embeddings"]
        loss_terms = state["last_loss_terms"]
        mask_updates = state["last_mask_updates"]
        if not isinstance(personalized, Mapping) or not isinstance(masks, Mapping) or not isinstance(embeddings, Mapping):
            raise TypeError("FED-PUB checkpoint states must be mappings.")
        if not isinstance(loss_terms, Mapping) or not isinstance(mask_updates, Mapping):
            raise TypeError("FED-PUB checkpoint loss/update terms must be mappings.")
        if set(personalized) != set(self._client_ids) or set(masks) != set(self._client_ids):
            raise ValueError("FED-PUB checkpoint personalized/mask clients mismatch.")
        self._shared_state = FrozenTensorMap(self._validate_state(state["shared_state"]))
        self._personalized_states = {
            int(client_id): FrozenTensorMap(self._validate_state(value))
            for client_id, value in personalized.items()
        }
        self._masks = {
            int(client_id): FrozenTensorMap(self._validate_state(value))
            for client_id, value in masks.items()
        }
        self._last_proxy_embeddings = {
            int(client_id): FrozenTensorMap(value)
            for client_id, value in embeddings.items()
        }
        self._last_loss_terms = {
            int(client_id): (float(value[0]), float(value[1]))
            for client_id, value in loss_terms.items()
        }
        self._last_mask_updates = {
            int(client_id): (float(value[0]), float(value[1]))
            for client_id, value in mask_updates.items()
        }
        self._completed_rounds = completed_rounds
