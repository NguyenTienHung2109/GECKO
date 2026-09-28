"""POWER-UEFA partial replay/prototype strategy for UEFA v2."""

from __future__ import annotations

import math
from collections.abc import Iterable
from collections.abc import Mapping
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


def _finite_positive(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _finite_unit(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return value


def power_class_prototypes(
    *,
    node_features: torch.Tensor,
    train_queries: torch.Tensor,
    train_labels: torch.Tensor,
    one_per_class: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return strict-local class feature prototypes and counts."""

    if node_features.ndim != 2 or not node_features.is_floating_point():
        raise ValueError("POWER prototypes require floating [nodes, features].")
    if train_queries.ndim != 1 or train_queries.dtype != torch.long:
        raise ValueError("POWER NC prototypes require 1-D long train queries.")
    if (
        train_labels.ndim != 1
        or train_labels.dtype != torch.long
        or train_labels.shape[0] != train_queries.shape[0]
    ):
        raise ValueError("POWER train labels must align with queries.")
    if train_queries.numel() == 0:
        raise ValueError("POWER requires at least one current train query.")
    if torch.any(train_queries < 0) or torch.any(
        train_queries >= node_features.shape[0]
    ):
        raise ValueError("POWER train queries are outside the strict-local graph.")
    local_queries = train_queries.to(device=node_features.device)
    local_labels = train_labels.to(device=node_features.device)
    classes = torch.unique(local_labels.detach().cpu()).sort().values
    features: list[torch.Tensor] = []
    counts: list[int] = []
    for cls in classes.tolist():
        mask = local_labels == int(cls)
        selected = node_features[local_queries[mask]]
        if selected.numel() == 0:
            continue
        prototype = selected[:1].mean(dim=0) if one_per_class else selected.mean(dim=0)
        features.append(prototype.detach().clone().contiguous())
        counts.append(int(mask.sum().detach().cpu()))
    if not features:
        raise ValueError("POWER could not build any class prototype.")
    return (
        torch.stack(features),
        classes.to(dtype=torch.long),
        torch.tensor(counts, dtype=torch.long),
    )


def power_knn_edges(prototypes: torch.Tensor, *, k: int = 1) -> torch.Tensor:
    """Build a deterministic directed kNN pseudo-graph over server prototypes."""

    if prototypes.ndim != 2 or not prototypes.is_floating_point():
        raise ValueError("POWER pseudo-graph prototypes must be [nodes, features].")
    if k != 1:
        raise ValueError("Initial POWER-UEFA gate fixes k=1.")
    node_count = int(prototypes.shape[0])
    if node_count <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    normalized = torch.nn.functional.normalize(prototypes.detach().float(), dim=1)
    similarity = normalized @ normalized.t()
    similarity.fill_diagonal_(-float("inf"))
    targets = torch.argmax(similarity, dim=1)
    sources = torch.arange(node_count, dtype=torch.long)
    return torch.stack((sources, targets.to(dtype=torch.long))).contiguous()


def power_reconstruct_pseudo_prototypes(
    *,
    uploaded_features: torch.Tensor,
    prototype_gradients: torch.Tensor,
    counts: torch.Tensor,
    alpha: float,
    reconstruction_steps: int,
) -> torch.Tensor:
    """Reconstruct server pseudo-prototypes with a bounded LBFGS objective.

    The reconstruction uses only uploaded feature summaries, prototype-gradient
    tensors, and counts.  It intentionally remains a partial POWER-UEFA
    adaptation, but the artifact is now produced by the declared LBFGS-style
    server reconstruction mechanism instead of a closed-form heuristic.
    """

    if uploaded_features.ndim != 2 or not uploaded_features.is_floating_point():
        raise ValueError(
            "POWER reconstruction features must be floating [nodes, features]."
        )
    if prototype_gradients.ndim != 2 or not prototype_gradients.is_floating_point():
        raise ValueError(
            "POWER reconstruction gradients must be floating [nodes, classes]."
        )
    if prototype_gradients.shape[0] != uploaded_features.shape[0]:
        raise ValueError("POWER reconstruction gradients must align with features.")
    if (
        counts.ndim != 1
        or counts.shape[0] != uploaded_features.shape[0]
        or torch.any(counts <= 0)
    ):
        raise ValueError(
            "POWER reconstruction counts must be positive and align with features."
        )
    if reconstruction_steps != 300:
        raise ValueError("Initial POWER reconstruction fixes 300 steps.")
    alpha = _finite_unit(alpha, name="alpha")
    features = uploaded_features.detach().float().clone()
    gradients = prototype_gradients.detach().float().to(features.device)
    weights = counts.detach().float().to(features.device).clamp_min(1.0)
    grad_strength = gradients.norm(dim=1) / weights
    grad_strength = grad_strength / grad_strength.clamp_min(1.0).max()
    center = (features * weights.unsqueeze(1)).sum(
        dim=0, keepdim=True
    ) / weights.sum().clamp_min(1.0)
    target = features + alpha * grad_strength.unsqueeze(1) * (features - center)
    variable = torch.nn.Parameter(features.clone())
    optimizer = torch.optim.LBFGS(
        [variable],
        lr=1.0,
        max_iter=reconstruction_steps,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        proximity = torch.mean((variable - features) ** 2)
        reconstruction = torch.mean((variable - target) ** 2)
        smoothness = torch.mean((variable - center) ** 2)
        objective = reconstruction + 0.1 * proximity + 1e-4 * smoothness
        objective.backward()
        return objective

    optimizer.step(closure)
    return variable.detach().to(dtype=uploaded_features.dtype).clone().contiguous()


def power_weighted_average(
    states: Mapping[int, Mapping[str, torch.Tensor]],
    weights: Mapping[int, int],
) -> dict[str, torch.Tensor]:
    if not states or set(states) != set(weights):
        raise ValueError("POWER states and weights must cover active clients.")
    total = sum(int(v) for v in weights.values())
    if total <= 0:
        raise ValueError("POWER aggregate weight must be positive.")
    keys: tuple[str, ...] | None = None
    for state in states.values():
        if keys is None:
            keys = tuple(sorted(state))
        elif tuple(sorted(state)) != keys:
            raise ValueError("POWER state keys must match.")
    assert keys is not None
    output: dict[str, torch.Tensor] = {}
    for key in keys:
        reference = states[next(iter(states))][key]
        result = torch.zeros_like(reference)
        for client_id, state in states.items():
            result.add_(
                state[key].to(result.device), alpha=float(weights[client_id]) / total
            )
        output[key] = result
    return output


class PowerUEFAStrategy:
    """Partial POWER-UEFA prototype/replay/trajectory server state."""

    strategy_version = "uefa-power-uefa-strategy-v1"
    name = "power_uefa"
    aggregates = True
    oracle = False
    uses_proximal_objective = False

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        beta: float = 0.01,
        trajectory_decay: float = 0.1,
        coverage_threshold: float = 0.1,
        replay_ceiling_bytes: int = 16 * 1024 * 1024,
        server_epochs: int = 3,
        reconstruction_steps: int = 300,
        ablation_mode: str = "full",
    ) -> None:
        if ablation_mode not in {"local_only", "server_only", "full"}:
            raise ValueError(
                "POWER ablation_mode must be local_only, server_only, or full."
            )
        self.ablation_mode = ablation_mode
        self.alpha = _finite_unit(alpha, name="alpha")
        self.beta = _finite_positive(beta, name="beta")
        self.trajectory_decay = _finite_unit(trajectory_decay, name="trajectory_decay")
        self.coverage_threshold = _finite_unit(
            coverage_threshold, name="coverage_threshold"
        )
        if replay_ceiling_bytes != 16 * 1024 * 1024:
            raise ValueError("POWER replay ceiling must equal 16 MiB.")
        if server_epochs != 3 or reconstruction_steps != 300:
            raise ValueError(
                "Initial POWER gate fixes 3 server epochs and 300 reconstruction steps."
            )
        self.replay_ceiling_bytes = replay_ceiling_bytes
        self.server_epochs = server_epochs
        self.reconstruction_steps = reconstruction_steps
        self._manifest: ParameterManifest | None = None
        self._client_ids: tuple[int, ...] = ()
        self._shared_state = FrozenTensorMap()
        self._client_replay: dict[int, FrozenTensorMap] = {}
        self._server_prototypes = FrozenTensorMap()
        self._reconstructed_prototypes = FrozenTensorMap()
        self._teacher_prototypes = FrozenTensorMap()
        self._trajectory = FrozenTensorMap()
        self._pseudo_edges = torch.empty((2, 0), dtype=torch.long)
        self._last_expertise_losses: dict[int, float] = {}
        self._completed_rounds = 0
        self._initialized = False

    def _require_initialized(self) -> None:
        if not self._initialized or self._manifest is None:
            raise RuntimeError("POWER-UEFA has not been initialized.")

    def _validate_state(
        self, state: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self._manifest is None:
            raise RuntimeError("POWER manifest is unavailable.")
        expected = tuple(self._manifest.shared_trainable)
        if set(state) != set(expected):
            raise ValueError("POWER state keys do not match shared trainables.")
        reference = self._shared_state.materialize()
        output: dict[str, torch.Tensor] = {}
        for key in expected:
            value = state[key]
            if (
                value.shape != reference[key].shape
                or value.dtype != reference[key].dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"POWER state tensor {key!r} is invalid.")
            output[key] = value.detach().clone().contiguous()
        return output

    def initialize(
        self,
        shared_state: TensorState,
        parameter_manifest: ParameterManifest,
        client_ids: Iterable[int],
    ) -> None:
        if self._initialized:
            raise RuntimeError("POWER cannot be initialized twice.")
        clients = tuple(sorted(int(c) for c in client_ids))
        if (
            not clients
            or len(set(clients)) != len(clients)
            or any(c < 0 for c in clients)
        ):
            raise ValueError("POWER client IDs must be unique non-negative integers.")
        if not tuple(parameter_manifest.shared_trainable) or set(shared_state) != set(
            parameter_manifest.shared_trainable
        ):
            raise ValueError("POWER requires every shared trainable parameter.")
        self._manifest = parameter_manifest
        self._client_ids = clients
        self._shared_state = FrozenTensorMap(shared_state)
        self._initialized = True

    @property
    def shared_state(self) -> dict[str, torch.Tensor]:
        self._require_initialized()
        return self._shared_state.materialize()

    def prepare_payload(
        self, context: RoundContext, client_id: int, reason: BroadcastReason
    ) -> BroadcastPayload:
        self._require_initialized()
        if (
            client_id not in context.participant_ids
            or client_id not in self._client_ids
        ):
            raise ValueError("POWER payload client is not a known participant.")
        resources = ResourceLedger()
        if reason == "initialization":
            resources.add(
                initialization_model_downlink_bytes=self._shared_state.payload_bytes
            )
        elif reason == "training":
            resources.add(
                training_model_downlink_bytes=self._shared_state.payload_bytes
            )
        elif reason == "evaluation":
            resources.add(evaluation_sync_bytes=self._shared_state.payload_bytes)
        elif reason != "resume":
            raise ValueError(f"Unknown POWER broadcast reason {reason!r}.")
        return BroadcastPayload(
            client_id=client_id,
            round_context=context,
            reason=reason,
            model_state=self._shared_state,
            metadata={"strategy_version": self.strategy_version},
            resources=resources,
        )

    def client_receive(
        self, client: Any, payload: BroadcastPayload, method_context: Any
    ) -> None:
        if int(client.client_id) != payload.client_id:
            raise ValueError("POWER payload was delivered to the wrong client.")
        client.load_shared_state(payload.model_state.materialize())
        client.algorithm.on_broadcast(method_context, payload)

    def augment_loss(
        self,
        model: torch.nn.Module,
        method_context: Any,
        loss: torch.Tensor,
        shared_keys: tuple[str, ...],
    ) -> torch.Tensor:
        """Add POWER teacher-transfer loss on server-generated proxy prototypes."""

        self._require_initialized()
        client_id = int(method_context.client_id)
        if self.ablation_mode != "full":
            return loss
        teacher = self._teacher_prototypes.materialize()
        if not teacher or method_context.problem_type != "NC":
            return loss
        features = teacher["features"].to(
            device=loss.device, dtype=method_context.node_features.dtype
        )
        labels = teacher["labels"].reshape(-1).to(device=loss.device, dtype=torch.long)
        if features.numel() == 0 or labels.numel() == 0:
            return loss
        queries = torch.arange(features.shape[0], dtype=torch.long, device=loss.device)
        edges = torch.empty((2, 0), dtype=torch.long, device=loss.device)
        logits = method_context.forward_queries(
            model,
            queries,
            node_features=features,
            edge_index=edges,
        )
        transfer_loss = torch.nn.functional.cross_entropy(logits, labels)
        self._last_expertise_losses[client_id] = float(transfer_loss.detach().cpu())
        return loss + self.beta * transfer_loss

    def transform_gradients(
        self, model: torch.nn.Module, method_context: Any, shared_keys: tuple[str, ...]
    ) -> None:
        return None

    def finalize_upload(
        self, client: Any, local_result: Any, method_context: Any
    ) -> ClientUpload:
        self._require_initialized()
        client_id = int(client.client_id)
        if method_context.problem_type != "NC":
            raise ValueError("Initial POWER-UEFA scope supports NC only.")
        if (
            local_result.client_id != client_id
            or local_result.global_task_id != int(method_context.global_task_id)
            or client_id not in self._client_ids
        ):
            raise ValueError("POWER local result identity mismatch.")
        model_state = self._validate_state(local_result.shared_state)
        features, labels, counts = power_class_prototypes(
            node_features=method_context.node_features,
            train_queries=method_context.train_queries,
            train_labels=method_context.train_labels,
            one_per_class=True,
        )
        logits = method_context.forward_queries(
            client.model, method_context.train_queries
        )
        proto_grad = torch.zeros(
            (labels.shape[0], logits.shape[-1]), dtype=logits.dtype
        )
        for index, label in enumerate(labels.tolist()):
            proto_grad[index, int(label)] = -float(counts[index])
        replay = FrozenTensorMap(
            {
                "features": features,
                "labels": labels,
                "counts": counts,
                "prototype_gradients": proto_grad,
            }
        )
        if replay.payload_bytes > self.replay_ceiling_bytes:
            raise ValueError(
                "POWER replay prototype payload exceeds the 16 MiB ceiling."
            )
        self._client_replay[client_id] = replay
        resources = ResourceLedger(
            training_model_uplink_bytes=FrozenTensorMap(model_state).payload_bytes,
            training_auxiliary_uplink_bytes=replay.payload_bytes,
            replay_bytes=replay.payload_bytes,
        )
        return ClientUpload(
            client_id=client_id,
            global_task_id=local_result.global_task_id,
            weight=local_result.weight,
            training_loss=local_result.training_loss,
            model_state=model_state,
            auxiliary_state=replay,
            diagnostics={
                "strategy_version": self.strategy_version,
                "prototype_classes": [int(v) for v in labels.tolist()],
                "prototype_count": int(labels.numel()),
            },
            resources=resources,
        )

    def aggregate(
        self, context: RoundContext, uploads: tuple[ClientUpload, ...]
    ) -> AggregationResult:
        self._require_initialized()
        upload_ids = tuple(upload.client_id for upload in uploads)
        if len(set(upload_ids)) != len(upload_ids) or set(upload_ids) != set(
            context.participant_ids
        ):
            raise ValueError("POWER requires exactly one upload per participant.")
        states = {
            upload.client_id: self._validate_state(upload.model_state)
            for upload in uploads
        }
        weights = {upload.client_id: upload.weight for upload in uploads}
        new_shared = power_weighted_average(states, weights)
        resources = ResourceLedger()
        prototypes: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        counts: list[torch.Tensor] = []
        prototype_gradients: list[torch.Tensor] = []
        for upload in uploads:
            if set(upload.auxiliary_state) != {
                "features",
                "labels",
                "counts",
                "prototype_gradients",
            }:
                raise ValueError("POWER auxiliary upload is malformed.")
            resources.merge(upload.resources)
            replay = upload.auxiliary_state.materialize()
            self._client_replay[upload.client_id] = FrozenTensorMap(replay)
            prototypes.append(replay["features"])
            labels.append(replay["labels"].to(dtype=torch.float32).unsqueeze(1))
            counts.append(replay["counts"].to(dtype=torch.long))
            prototype_gradients.append(
                replay["prototype_gradients"].to(dtype=torch.float32)
            )
        server_features = torch.cat(prototypes, dim=0)
        server_labels = torch.cat(labels, dim=0)
        server_counts = torch.cat(counts, dim=0)
        server_gradients = torch.cat(prototype_gradients, dim=0)
        teacher_transfer = "disabled_local_replay_only"
        reconstruction_source = "disabled_local_replay_only"
        if self.ablation_mode == "local_only":
            reconstructed_features = server_features.detach().clone().contiguous()
            teacher_features = torch.empty_like(server_features[:0])
            self._pseudo_edges = torch.empty((2, 0), dtype=torch.long)
            self._server_prototypes = FrozenTensorMap(
                {
                    "features": server_features,
                    "labels": server_labels,
                    "counts": server_counts,
                }
            )
            self._reconstructed_prototypes = FrozenTensorMap()
            self._teacher_prototypes = FrozenTensorMap()
            self._trajectory = FrozenTensorMap()
        else:
            reconstructed_features = power_reconstruct_pseudo_prototypes(
                uploaded_features=server_features,
                prototype_gradients=server_gradients,
                counts=server_counts,
                alpha=self.alpha,
                reconstruction_steps=self.reconstruction_steps,
            )
            self._pseudo_edges = power_knn_edges(reconstructed_features, k=1)
            previous = self._trajectory.materialize() if self._trajectory else {}
            trajectory_features = server_features
            if (
                "features" in previous
                and previous["features"].shape == server_features.shape
            ):
                trajectory_features = (
                    self.trajectory_decay * previous["features"]
                    + (1.0 - self.trajectory_decay) * server_features
                )
            teacher_features = self.alpha * reconstructed_features + (
                1.0 - self.alpha
            ) * trajectory_features.to(reconstructed_features.device)
            self._server_prototypes = FrozenTensorMap(
                {
                    "features": server_features,
                    "labels": server_labels,
                    "counts": server_counts,
                }
            )
            self._reconstructed_prototypes = FrozenTensorMap(
                {"features": reconstructed_features, "labels": server_labels}
            )
            self._teacher_prototypes = FrozenTensorMap(
                {"features": teacher_features, "labels": server_labels}
            )
            self._trajectory = FrozenTensorMap({"features": trajectory_features})
            reconstruction_source = (
                "lbfgs_prototype_gradients_counts_and_local_feature_summaries"
            )
            teacher_transfer = (
                "server_only_no_client_expertise_loss"
                if self.ablation_mode == "server_only"
                else "trajectory_reconstructed_blend"
            )
        self._shared_state = FrozenTensorMap(new_shared)
        self._completed_rounds += 1
        pseudo_bytes = self._pseudo_edges.numel() * self._pseudo_edges.element_size()
        resources.add(
            server_persistent_bytes=(
                self._server_prototypes.payload_bytes
                + self._reconstructed_prototypes.payload_bytes
                + self._teacher_prototypes.payload_bytes
                + self._trajectory.payload_bytes
            ),
            synthetic_artifact_bytes=pseudo_bytes
            + self._reconstructed_prototypes.payload_bytes
            + self._teacher_prototypes.payload_bytes,
            client_persistent_bytes=sum(
                state.payload_bytes for state in self._client_replay.values()
            ),
        )
        return AggregationResult(
            shared_state=self._shared_state,
            server_auxiliary_state={
                "pseudo_edges": self._pseudo_edges,
                "teacher_features": teacher_features.detach().cpu().clone(),
            },
            diagnostics={
                "strategy_version": self.strategy_version,
                "aggregation": f"power_{self.ablation_mode}_ablation",
                "ablation_mode": self.ablation_mode,
                "participants": list(sorted(context.participant_ids)),
                "server_prototypes": int(server_features.shape[0]),
                "reconstructed_prototypes": int(reconstructed_features.shape[0])
                if self.ablation_mode != "local_only"
                else 0,
                "teacher_prototypes": int(teacher_features.shape[0]),
                "teacher_transfer": teacher_transfer,
                "reconstruction_source": reconstruction_source,
                "pseudo_graph_edges": int(self._pseudo_edges.shape[1]),
            },
            resources=resources,
        )

    def personalize(
        self, context: RoundContext, result: AggregationResult
    ) -> AggregationResult:
        return result

    def select_evaluation(
        self, client_id: int, stage_index: int
    ) -> EvaluationSelection:
        self._require_initialized()
        if client_id not in self._client_ids or stage_index < 0:
            raise ValueError("Unknown POWER evaluation client or stage.")
        return EvaluationSelection(client_id=client_id, source="post_local")

    def diagnostics(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "fidelity": "mechanism_adaptation_partial_reconstruction",
            "ablation_mode": self.ablation_mode,
            "alpha": self.alpha,
            "beta": self.beta,
            "trajectory_decay": self.trajectory_decay,
            "coverage_threshold": self.coverage_threshold,
            "replay_ceiling_bytes": self.replay_ceiling_bytes,
            "server_epochs": self.server_epochs,
            "reconstruction_steps": self.reconstruction_steps,
            "completed_rounds": self._completed_rounds,
            "clients_with_replay": sorted(self._client_replay),
            "reconstructed_prototypes": int(
                self._reconstructed_prototypes.materialize()
                .get("features", torch.empty((0, 0)))
                .shape[0]
            ),
            "teacher_prototypes": int(
                self._teacher_prototypes.materialize()
                .get("features", torch.empty((0, 0)))
                .shape[0]
            ),
            "teacher_transfer": (
                "disabled_local_replay_only"
                if self.ablation_mode == "local_only"
                else "server_only_no_client_expertise_loss"
                if self.ablation_mode == "server_only"
                else "trajectory_reconstructed_blend"
            ),
            "reconstruction_source": (
                "disabled_local_replay_only"
                if self.ablation_mode == "local_only"
                else "lbfgs_prototype_gradients_counts_and_local_feature_summaries"
            ),
            "reconstruction_optimizer": "disabled"
            if self.ablation_mode == "local_only"
            else "lbfgs",
            "pseudo_graph_edges": int(self._pseudo_edges.shape[1]),
            "expertise_transfer": "disabled"
            if self.ablation_mode != "full"
            else "teacher_proxy_cross_entropy",
            "clients_with_expertise_transfer_loss": sorted(self._last_expertise_losses),
            "last_expertise_transfer_losses": {
                f"client-{client_id}": value
                for client_id, value in sorted(self._last_expertise_losses.items())
            },
        }

    def state_dict(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "ablation_mode": self.ablation_mode,
            "client_ids": self._client_ids,
            "shared_state": self._shared_state.materialize(),
            "client_replay": {
                client_id: state.materialize()
                for client_id, state in self._client_replay.items()
            },
            "server_prototypes": self._server_prototypes.materialize(),
            "reconstructed_prototypes": self._reconstructed_prototypes.materialize(),
            "teacher_prototypes": self._teacher_prototypes.materialize(),
            "trajectory": self._trajectory.materialize(),
            "pseudo_edges": self._pseudo_edges.detach().cpu().clone(),
            "last_expertise_losses": dict(self._last_expertise_losses),
            "completed_rounds": self._completed_rounds,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._require_initialized()
        expected = {
            "strategy_version",
            "ablation_mode",
            "client_ids",
            "shared_state",
            "client_replay",
            "server_prototypes",
            "reconstructed_prototypes",
            "teacher_prototypes",
            "trajectory",
            "pseudo_edges",
            "last_expertise_losses",
            "completed_rounds",
        }
        if (
            set(state) != expected
            or state["strategy_version"] != self.strategy_version
            or state["ablation_mode"] != self.ablation_mode
        ):
            raise ValueError("POWER checkpoint identity mismatch.")
        if tuple(state["client_ids"]) != self._client_ids:
            raise ValueError("POWER checkpoint client IDs mismatch.")
        completed = state["completed_rounds"]
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or completed < 0
        ):
            raise ValueError("POWER checkpoint completed rounds are invalid.")
        replay = state["client_replay"]
        if not isinstance(replay, Mapping):
            raise TypeError("POWER checkpoint replay must be a mapping.")
        self._shared_state = FrozenTensorMap(
            self._validate_state(state["shared_state"])
        )
        self._client_replay = {
            int(client_id): FrozenTensorMap(value)
            for client_id, value in replay.items()
        }
        self._server_prototypes = FrozenTensorMap(state["server_prototypes"])
        self._reconstructed_prototypes = FrozenTensorMap(
            state["reconstructed_prototypes"]
        )
        self._teacher_prototypes = FrozenTensorMap(state["teacher_prototypes"])
        self._trajectory = FrozenTensorMap(state["trajectory"])
        pseudo_edges = state["pseudo_edges"]
        if not torch.is_tensor(pseudo_edges) or pseudo_edges.shape[0] != 2:
            raise ValueError("POWER checkpoint pseudo edges are invalid.")
        losses = state["last_expertise_losses"]
        if not isinstance(losses, Mapping):
            raise TypeError("POWER checkpoint expertise losses must be a mapping.")
        self._pseudo_edges = pseudo_edges.detach().cpu().clone().to(dtype=torch.long)
        self._last_expertise_losses = {
            int(client_id): float(value) for client_id, value in losses.items()
        }
        self._completed_rounds = completed


# Public/runtime POWER uses the equation-complete implementation.  The legacy
# class above remains readable only for old checkpoint/source audit context.
from gecko.algorithms.federated.power.paper import PaperPowerUEFAStrategy as PowerUEFAStrategy
