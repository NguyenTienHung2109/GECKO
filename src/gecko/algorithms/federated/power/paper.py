"""Paper-faithful POWER strategy for NC-Class federation."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import torch
from torch.nn import functional as F

from gecko.task_masks import task_aware_cross_entropy

from gecko.engine.accounting import ResourceLedger
from gecko.algorithms.federated.power.gradient import PowerGradientEncoder
from gecko.algorithms.federated.power.gradient import power_encode_prototype_gradients
from gecko.algorithms.federated.power.gradient import power_gradient_label
from gecko.algorithms.federated.power.gradient import power_reconstruct_from_gradients
from gecko.algorithms.federated.power.oracles import power_append_replay
from gecko.algorithms.federated.power.oracles import power_class_mean_prototypes
from gecko.algorithms.federated.power.oracles import power_local_global_coverage_selection
from gecko.algorithms.federated.power.oracles import power_replay_payload_bytes
from gecko.algorithms.federated.power.server import POWER_BACKTRACK_FACTOR
from gecko.algorithms.federated.power.server import POWER_MAX_BACKTRACK_STEPS
from gecko.algorithms.federated.power.server import power_knn_edges
from gecko.algorithms.federated.power.server import power_server_transfer
from gecko.algorithms.federated.power.server import power_update_trajectory
from gecko.engine.protocol import AggregationResult
from gecko.engine.protocol import BroadcastPayload
from gecko.engine.protocol import BroadcastReason
from gecko.engine.protocol import ClientUpload
from gecko.engine.protocol import EvaluationSelection
from gecko.engine.protocol import FrozenTensorMap
from gecko.engine.protocol import ParameterManifest
from gecko.engine.protocol import RoundContext
from gecko.engine.protocol import TensorState


def _finite_unit(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return normalized


def power_weighted_average(
    states: Mapping[int, Mapping[str, torch.Tensor]],
    weights: Mapping[int, int],
) -> dict[str, torch.Tensor]:
    """FedAvg step that precedes POWER's server knowledge transfer."""

    if not states or set(states) != set(weights):
        raise ValueError("POWER states and weights must cover active clients.")
    total = sum(int(value) for value in weights.values())
    if total <= 0:
        raise ValueError("POWER aggregate weight must be positive.")
    keys = tuple(sorted(next(iter(states.values()))))
    if any(tuple(sorted(state)) != keys for state in states.values()):
        raise ValueError("POWER model-state keys must match.")
    output: dict[str, torch.Tensor] = {}
    for key in keys:
        reference = next(iter(states.values()))[key]
        value = torch.zeros_like(reference)
        for client_id, state in states.items():
            value.add_(state[key].to(value.device), alpha=weights[client_id] / total)
        output[key] = value.contiguous()
    return output


class PaperPowerUEFAStrategy:
    """POWER Eqs. (2)--(14), restricted to leakage-safe NC-Class."""

    strategy_version = "uefa-power-paper-strategy-v5"
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
        samples_per_class: int = 1,
        server_epochs: int = 3,
        server_learning_rate: float = 1e-2,
        reconstruction_steps: int = 300,
        ablation_mode: str = "full",
    ) -> None:
        if ablation_mode not in {"local_only", "server_only", "full"}:
            raise ValueError(
                "POWER ablation_mode must be local_only, server_only, or full."
            )
        if beta <= 0.0 or beta > 1.0 or not math.isfinite(float(beta)):
            raise ValueError("POWER beta must lie in (0, 1].")
        if replay_ceiling_bytes != 16 * 1024 * 1024:
            raise ValueError("POWER replay ceiling must equal 16 MiB.")
        if (
            isinstance(samples_per_class, bool)
            or not isinstance(samples_per_class, int)
            or samples_per_class <= 0
        ):
            raise ValueError("POWER samples_per_class must be a positive integer.")
        if server_epochs != 3 or reconstruction_steps != 300:
            raise ValueError("POWER fixes 3 server epochs and 300 reconstruction steps.")
        if server_learning_rate <= 0.0 or not math.isfinite(
            float(server_learning_rate)
        ):
            raise ValueError("POWER server_learning_rate must be finite and positive.")
        self.ablation_mode = ablation_mode
        self.alpha = _finite_unit(alpha, name="alpha")
        self.beta = float(beta)
        self.trajectory_decay = _finite_unit(
            trajectory_decay, name="trajectory_decay"
        )
        self.coverage_threshold = _finite_unit(
            coverage_threshold, name="coverage_threshold"
        )
        self.replay_ceiling_bytes = int(replay_ceiling_bytes)
        self.samples_per_class = int(samples_per_class)
        self.server_epochs = int(server_epochs)
        self.reconstruction_steps = int(reconstruction_steps)
        self.server_learning_rate = float(server_learning_rate)
        self.server_weight_decay = 5e-4
        self.lbfgs_learning_rate = 1.0

        self._manifest: ParameterManifest | None = None
        self._client_ids: tuple[int, ...] = ()
        self._shared_state = FrozenTensorMap()
        self._model_template: torch.nn.Module | None = None
        self._device = torch.device("cpu")
        self._feature_dim = 0
        self._num_classes = 0
        self._model_seed = 0
        self._encoder: PowerGradientEncoder | None = None
        self._encoder_state = FrozenTensorMap()
        self._server_features = torch.empty((0, 0), dtype=torch.float32)
        self._server_labels = torch.empty((0,), dtype=torch.long)
        self._trajectory = torch.empty((0, 0), dtype=torch.float32)
        self._pseudo_edges = torch.empty((2, 0), dtype=torch.long)
        self._last_local_states: dict[int, FrozenTensorMap] = {}
        self._active_replay: dict[int, dict[str, torch.Tensor]] = {}
        self._task_class_masks: dict[int, torch.Tensor] = {}
        self._task_aware: bool | None = None
        self._last_replay_losses: dict[int, float] = {}
        self._last_server_losses: tuple[float, ...] = ()
        self._last_server_transfer_report: dict[str, object] = {}
        self._last_reconstruction_losses: tuple[tuple[float, float], ...] = ()
        self._trajectory_stages: set[int] = set()
        self._server_changed_rounds = 0
        self._server_zero_signal_rounds = 0
        self._server_rollback_rounds = 0
        self._server_attempted_epochs = 0
        self._server_accepted_epochs = 0
        self._server_proposal_attempts = 0
        self._server_rejected_proposals = 0
        self._server_backtracking_reductions = 0
        self._completed_rounds = 0
        self._initialized = False
        self._bound = False

    @property
    def _local_enabled(self) -> bool:
        return self.ablation_mode in {"local_only", "full"}

    @property
    def _server_enabled(self) -> bool:
        return self.ablation_mode in {"server_only", "full"}

    def bind_runtime(
        self,
        *,
        model_template: torch.nn.Module,
        feature_dim: int,
        num_classes: int,
        model_seed: int,
        device: torch.device | str,
    ) -> None:
        """Bind public model metadata before protocol initialization."""

        if self._bound or self._initialized:
            raise RuntimeError("POWER runtime can only be bound once before initialize.")
        if feature_dim <= 0 or num_classes <= 1:
            raise ValueError("POWER runtime dimensions are invalid.")
        self._model_template = copy.deepcopy(model_template).cpu()
        self._device = torch.device(device)
        self._feature_dim = int(feature_dim)
        self._num_classes = int(num_classes)
        self._model_seed = int(model_seed)
        self._encoder = PowerGradientEncoder(
            self._feature_dim, self._num_classes, seed=self._model_seed
        ).to(self._device)
        self._encoder_state = FrozenTensorMap(
            {
                name: value.detach().cpu().clone()
                for name, value in self._encoder.state_dict().items()
            }
        )
        self._server_features = torch.empty(
            (0, self._feature_dim), dtype=torch.float32
        )
        self._trajectory = torch.empty((0, self._num_classes), dtype=torch.float32)
        self._bound = True

    def _require_initialized(self) -> None:
        if not self._initialized or self._manifest is None:
            raise RuntimeError("POWER has not been initialized.")

    def _validate_state(
        self, state: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self._manifest is None:
            raise RuntimeError("POWER manifest is unavailable.")
        expected = tuple(self._manifest.shared_trainable)
        if set(state) != set(expected):
            raise ValueError("POWER model state keys do not match shared trainables.")
        reference = self._shared_state.materialize()
        output: dict[str, torch.Tensor] = {}
        for key in expected:
            value = state[key]
            if (
                value.shape != reference[key].shape
                or value.dtype != reference[key].dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"POWER model tensor {key!r} is invalid.")
            output[key] = value.detach().cpu().clone().contiguous()
        return output

    def initialize(
        self,
        shared_state: TensorState,
        parameter_manifest: ParameterManifest,
        client_ids: Iterable[int],
    ) -> None:
        if self._initialized or not self._bound:
            raise RuntimeError("POWER must be bound once before initialization.")
        clients = tuple(sorted(int(client) for client in client_ids))
        if not clients or len(set(clients)) != len(clients) or any(
            client < 0 for client in clients
        ):
            raise ValueError("POWER client IDs must be unique non-negative integers.")
        if set(shared_state) != set(parameter_manifest.shared_trainable):
            raise ValueError("POWER requires every shared trainable parameter.")
        self._manifest = parameter_manifest
        self._client_ids = clients
        self._shared_state = FrozenTensorMap(shared_state)
        self._last_local_states = {
            client_id: FrozenTensorMap(shared_state) for client_id in clients
        }
        self._trajectory = torch.zeros(
            (len(clients), self._num_classes), dtype=torch.float32
        )
        self._initialized = True

    @property
    def shared_state(self) -> dict[str, torch.Tensor]:
        self._require_initialized()
        return self._shared_state.materialize()

    @staticmethod
    def _client_state(client: Any) -> dict[str, Any]:
        root = client.state.strategy_state
        state = root.get("power")
        if state is None:
            state = {
                "strategy_version": PaperPowerUEFAStrategy.strategy_version,
                "encoder_state": {},
                "global_model_state": {},
                "replay": {},
            }
            root["power"] = state
        expected = {
            "strategy_version",
            "encoder_state",
            "global_model_state",
            "replay",
        }
        if (
            not isinstance(state, dict)
            or state.get("strategy_version")
            != PaperPowerUEFAStrategy.strategy_version
            or set(state) != expected
        ):
            raise ValueError("POWER client-private state is malformed.")
        return state

    def prepare_payload(
        self, context: RoundContext, client_id: int, reason: BroadcastReason
    ) -> BroadcastPayload:
        self._require_initialized()
        if client_id not in context.participant_ids or client_id not in self._client_ids:
            raise ValueError("POWER payload client is not a known participant.")
        auxiliary = (
            self._encoder_state
            if reason == "initialization" and self._server_enabled
            else FrozenTensorMap()
        )
        resources = ResourceLedger()
        if reason == "initialization":
            resources.add(
                initialization_model_downlink_bytes=self._shared_state.payload_bytes,
                initialization_auxiliary_downlink_bytes=auxiliary.payload_bytes,
            )
        elif reason == "training":
            resources.add(training_model_downlink_bytes=self._shared_state.payload_bytes)
        elif reason == "evaluation":
            resources.add(evaluation_sync_bytes=self._shared_state.payload_bytes)
        elif reason != "resume":
            raise ValueError(f"Unknown POWER broadcast reason {reason!r}.")
        return BroadcastPayload(
            client_id=client_id,
            round_context=context,
            reason=reason,
            model_state=self._shared_state,
            auxiliary_state=auxiliary,
            metadata={
                "strategy_version": self.strategy_version,
                "prototype_encoder": bool(auxiliary),
            },
            resources=resources,
        )

    def client_receive(
        self, client: Any, payload: BroadcastPayload, method_context: Any
    ) -> None:
        if int(client.client_id) != payload.client_id:
            raise ValueError("POWER payload was delivered to the wrong client.")
        client.load_shared_state(payload.model_state.materialize())
        client.algorithm.on_broadcast(method_context, payload)
        is_task = str(method_context.incremental_setting).lower() == "task"
        if self._task_aware is not None and self._task_aware != is_task:
            raise ValueError("POWER cannot mix Class-IL and Task-IL contexts.")
        self._task_aware = is_task
        if is_task:
            mask = method_context.valid_class_mask
            if mask is None or mask.dtype != torch.bool or mask.ndim != 1:
                raise ValueError("POWER Task-IL requires a class mask.")
            task = int(method_context.global_task_id)
            owned = mask.detach().cpu().clone().contiguous()
            previous = self._task_class_masks.get(task)
            if previous is not None and not torch.equal(previous, owned):
                raise ValueError("POWER observed two masks for one global task.")
            self._task_class_masks[task] = owned
        state = self._client_state(client)
        if payload.reason == "initialization" and payload.auxiliary_state:
            state["encoder_state"] = payload.auxiliary_state.materialize()
        if payload.reason == "training":
            state["global_model_state"] = payload.model_state.materialize()
            replay = state["replay"]
            self._active_replay[int(client.client_id)] = (
                {}
                if not replay
                else {
                    "features": replay["features"].detach().cpu().clone(),
                    "labels": replay["labels"].detach().cpu().clone(),
                }
            )

    def augment_loss(
        self,
        model: torch.nn.Module,
        method_context: Any,
        loss: torch.Tensor,
        shared_keys: tuple[str, ...],
    ) -> torch.Tensor:
        self._require_initialized()
        if not self._local_enabled or method_context.problem_type != "NC":
            return loss
        replay = self._active_replay.get(int(method_context.client_id), {})
        if not replay:
            return loss
        features = replay["features"].to(
            device=loss.device, dtype=method_context.node_features.dtype
        )
        labels = replay["labels"].to(device=loss.device, dtype=torch.long)
        queries = torch.arange(features.shape[0], dtype=torch.long, device=loss.device)
        # POWER's upstream replay uses an empty PyG graph, but PyG GATConv adds
        # self-loops internally.  UEFA's dependency-light MeanGraphLayer does
        # not, so a literally empty COO graph would leave every neighbor branch
        # unconstrained by replay and lose graph knowledge across tasks.
        replay_edges = self._isolated_replay_edges(
            features.shape[0], device=loss.device
        )
        logits = method_context.forward_queries(
            model,
            queries,
            node_features=features,
            edge_index=replay_edges,
        )
        if self._task_aware:
            replay_loss = task_aware_cross_entropy(logits, labels, self._task_class_masks)
        else:
            logits = self._mask_replay_logits(logits, method_context.valid_class_mask)
            replay_loss = F.cross_entropy(logits, labels)
        self._last_replay_losses[int(method_context.client_id)] = float(
            replay_loss.detach().cpu()
        )
        return self.beta * loss + (1.0 - self.beta) * replay_loss

    @staticmethod
    def _isolated_replay_edges(
        node_count: int, *, device: torch.device | str
    ) -> torch.Tensor:
        """Match PyG's isolated-node self-loop semantics for POWER replay."""

        if isinstance(node_count, bool) or not isinstance(node_count, int):
            raise TypeError("POWER replay node_count must be an integer.")
        if node_count < 0:
            raise ValueError("POWER replay node_count must be non-negative.")
        nodes = torch.arange(node_count, dtype=torch.long, device=device)
        return torch.stack((nodes, nodes), dim=0)

    @staticmethod
    def _mask_replay_logits(
        logits: torch.Tensor,
        class_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Exclude unseen classes from POWER replay and its encoder gradient."""

        if class_mask is None:
            return logits
        if (
            logits.ndim != 2
            or class_mask.dtype != torch.bool
            or class_mask.ndim != 1
            or logits.shape[1] != class_mask.numel()
        ):
            raise ValueError("POWER replay class mask is incompatible with logits.")
        masked = logits.clone()
        masked[:, ~class_mask.to(logits.device)] = -1e12
        return masked

    def transform_gradients(
        self,
        model: torch.nn.Module,
        method_context: Any,
        shared_keys: tuple[str, ...],
    ) -> None:
        return None

    def _client_encoder(self, state: Mapping[str, Any]) -> PowerGradientEncoder:
        encoded_state = state["encoder_state"]
        if not isinstance(encoded_state, Mapping) or not encoded_state:
            raise ValueError("POWER client did not receive the shared gradient encoder.")
        encoder = PowerGradientEncoder(
            self._feature_dim, self._num_classes, seed=self._model_seed
        ).to(self._device)
        encoder.load_state_dict(
            {name: value.to(self._device) for name, value in encoded_state.items()},
            strict=True,
        )
        return encoder

    @staticmethod
    def _flatten_gradient_upload(
        gradients: Sequence[Sequence[torch.Tensor]], counts: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        payload = {"prototype_counts": counts.detach().cpu().to(dtype=torch.long)}
        for prototype_index, gradient_tuple in enumerate(gradients):
            for gradient_index, value in enumerate(gradient_tuple):
                payload[
                    f"prototype_{prototype_index:04d}_gradient_{gradient_index:02d}"
                ] = value.detach().cpu().clone().contiguous()
        return payload

    def _parse_gradient_upload(
        self, state: Mapping[str, torch.Tensor]
    ) -> tuple[tuple[tuple[torch.Tensor, ...], ...], torch.Tensor]:
        if self._encoder is None or "prototype_counts" not in state:
            raise ValueError("POWER gradient upload has no prototype counts.")
        counts = state["prototype_counts"]
        if (
            counts.ndim != 1
            or counts.dtype != torch.long
            or counts.numel() == 0
            or torch.any(counts <= 0)
        ):
            raise ValueError("POWER prototype counts are invalid.")
        shapes = [parameter.shape for parameter in self._encoder.parameters()]
        expected = {"prototype_counts"}
        gradients: list[tuple[torch.Tensor, ...]] = []
        for prototype_index in range(counts.numel()):
            values = []
            for gradient_index, shape in enumerate(shapes):
                name = (
                    f"prototype_{prototype_index:04d}_gradient_{gradient_index:02d}"
                )
                expected.add(name)
                if name not in state or state[name].shape != shape:
                    raise ValueError("POWER prototype-gradient structure is invalid.")
                values.append(state[name].detach().cpu().clone())
            gradients.append(tuple(values))
        if set(state) != expected:
            raise ValueError("POWER gradient upload contains unexpected fields.")
        return tuple(gradients), counts.detach().cpu().clone()

    def finalize_upload(
        self, client: Any, local_result: Any, method_context: Any
    ) -> ClientUpload:
        self._require_initialized()
        client_id = int(client.client_id)
        if method_context.problem_type != "NC":
            raise ValueError("POWER supports NC only.")
        if (
            local_result.client_id != client_id
            or local_result.global_task_id != int(method_context.global_task_id)
            or client_id not in self._client_ids
        ):
            raise ValueError("POWER local-result identity mismatch.")
        model_state = self._validate_state(local_result.shared_state)
        auxiliary: dict[str, torch.Tensor] = {}
        prototype_count = 0
        if self._server_enabled and int(method_context.round_index) == 0:
            prototypes, classes, counts = power_class_mean_prototypes(
                node_features=method_context.node_features,
                train_queries=method_context.train_queries,
                train_labels=method_context.train_labels,
            )
            encoder = self._client_encoder(self._client_state(client))
            gradients = power_encode_prototype_gradients(
                encoder=encoder,
                prototypes=prototypes.to(self._device),
                class_ids=classes,
            )
            auxiliary = self._flatten_gradient_upload(gradients, counts)
            prototype_count = len(gradients)
        frozen_auxiliary = FrozenTensorMap(auxiliary)
        resources = ResourceLedger(
            training_model_uplink_bytes=FrozenTensorMap(model_state).payload_bytes,
            training_auxiliary_uplink_bytes=frozen_auxiliary.payload_bytes,
        )
        return ClientUpload(
            client_id=client_id,
            global_task_id=local_result.global_task_id,
            weight=local_result.weight,
            training_loss=local_result.training_loss,
            model_state=model_state,
            auxiliary_state=frozen_auxiliary,
            diagnostics={
                "strategy_version": self.strategy_version,
                "prototype_gradient_count": prototype_count,
                "raw_prototype_uploaded": False,
                "raw_label_uploaded": False,
            },
            resources=resources,
        )

    def _append_reconstructions(
        self, context: RoundContext, uploads: Sequence[ClientUpload]
    ) -> None:
        if int(context.stage_index) in self._trajectory_stages:
            raise RuntimeError("POWER trajectory was already updated for this stage.")
        if self._encoder is None:
            raise RuntimeError("POWER server gradient encoder is unavailable.")
        current_counts = torch.zeros_like(self._trajectory)
        new_features: list[torch.Tensor] = []
        new_labels: list[int] = []
        loss_pairs: list[tuple[float, float]] = []
        rows = {client_id: index for index, client_id in enumerate(self._client_ids)}
        for upload in uploads:
            gradients, counts = self._parse_gradient_upload(
                upload.auxiliary_state.materialize()
            )
            for prototype_index, (gradient_tuple, count) in enumerate(
                zip(gradients, counts.tolist(), strict=True)
            ):
                label = power_gradient_label(
                    gradient_tuple, output_dim=self._num_classes
                )
                current_counts[rows[upload.client_id], label] += float(count)
                reconstructed, initial_loss, final_loss = power_reconstruct_from_gradients(
                    encoder=self._encoder,
                    target_gradients=gradient_tuple,
                    class_id=label,
                    steps=self.reconstruction_steps,
                    seed=(
                        self._model_seed
                        + int(context.stage_index) * 100_003
                        + int(upload.client_id) * 1_009
                        + prototype_index
                    ),
                    lbfgs_learning_rate=self.lbfgs_learning_rate,
                )
                new_features.append(reconstructed.squeeze(0))
                new_labels.append(label)
                loss_pairs.append((initial_loss, final_loss))
        if not new_features:
            raise ValueError("POWER first round produced no prototype gradients.")
        stacked = torch.stack(new_features).to(dtype=torch.float32)
        labels = torch.tensor(new_labels, dtype=torch.long)
        self._server_features = torch.cat((self._server_features, stacked), dim=0)
        self._server_labels = torch.cat((self._server_labels, labels), dim=0)
        self._trajectory = power_update_trajectory(
            previous=self._trajectory,
            current_counts=current_counts,
            decay=self.trajectory_decay,
        )
        self._trajectory_stages.add(int(context.stage_index))
        self._last_reconstruction_losses = tuple(loss_pairs)

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
        averaged = power_weighted_average(states, weights)
        resources = ResourceLedger()
        for upload in uploads:
            resources.merge(upload.resources)
            self._last_local_states[upload.client_id] = FrozenTensorMap(
                states[upload.client_id]
            )
        if self._server_enabled and int(context.round_index) == 0:
            self._append_reconstructions(context, uploads)

        new_shared = averaged
        self._last_server_losses = ()
        self._last_server_transfer_report = {}
        if self._server_enabled and self._server_features.shape[0] > 0:
            if self._model_template is None:
                raise RuntimeError("POWER model template is unavailable.")
            self._pseudo_edges = power_knn_edges(self._server_features, k=1)
            local_states = [
                self._last_local_states[client_id].materialize()
                for client_id in self._client_ids
            ]
            (
                new_shared,
                self._last_server_losses,
                transfer_report,
            ) = power_server_transfer(
                model_template=self._model_template,
                averaged_state=averaged,
                local_states=local_states,
                features=self._server_features,
                labels=self._server_labels,
                edge_index=self._pseudo_edges,
                trajectory=self._trajectory,
                epochs=self.server_epochs,
                learning_rate=self.server_learning_rate,
                weight_decay=self.server_weight_decay,
                device=self._device,
                task_class_masks=self._task_class_masks if self._task_aware else None,
            )
            self._last_server_transfer_report = transfer_report.as_dict()
            self._server_attempted_epochs += transfer_report.attempted_epochs
            self._server_accepted_epochs += transfer_report.accepted_epochs
            self._server_proposal_attempts += transfer_report.proposal_attempts
            self._server_rejected_proposals += transfer_report.rejected_proposals
            self._server_backtracking_reductions += (
                transfer_report.backtracking_reductions
            )
            changed = any(
                not torch.equal(new_shared[name], averaged[name]) for name in averaged
            )
            if transfer_report.disposition == "changed":
                if not changed or transfer_report.accepted_epochs <= 0:
                    raise RuntimeError("POWER changed disposition has no parameter update.")
                self._server_changed_rounds += 1
            elif transfer_report.disposition == "zero_signal":
                if changed:
                    raise RuntimeError("POWER zero-signal transfer changed parameters.")
                self._server_zero_signal_rounds += 1
            elif transfer_report.disposition == "stalled":
                if changed:
                    raise RuntimeError("POWER stalled transfer changed parameters.")
                self._server_rollback_rounds += 1
            else:
                raise RuntimeError("POWER server transfer disposition is invalid.")
        else:
            self._pseudo_edges = torch.empty((2, 0), dtype=torch.long)

        self._shared_state = FrozenTensorMap(new_shared)
        self._completed_rounds += 1
        synthetic_bytes = (
            self._server_features.numel() * self._server_features.element_size()
            + self._server_labels.numel() * self._server_labels.element_size()
            + self._pseudo_edges.numel() * self._pseudo_edges.element_size()
        )
        if self._server_enabled:
            resources.add(synthetic_artifact_bytes=synthetic_bytes)
        return AggregationResult(
            shared_state=self._shared_state,
            server_auxiliary_state={
                "pseudo_features": self._server_features,
                "pseudo_labels": self._server_labels,
                "pseudo_edges": self._pseudo_edges,
            }
            if self._server_enabled
            else {},
            diagnostics={
                "strategy_version": self.strategy_version,
                "aggregation": "fedavg_then_power_eq14"
                if self._server_enabled
                else "fedavg_local_replay_ablation",
                "participants": list(sorted(context.participant_ids)),
                "prototype_gradients_uploaded": bool(
                    self._server_enabled and int(context.round_index) == 0
                ),
                "pseudo_prototypes": int(self._server_features.shape[0]),
                "pseudo_graph_edges": int(self._pseudo_edges.shape[1]),
                "server_epochs": self.server_epochs if self._server_enabled else 0,
                "server_objective_trace_points": len(self._last_server_losses),
                "server_transfer_report": dict(self._last_server_transfer_report),
            },
            resources=resources,
        )

    def personalize(
        self, context: RoundContext, result: AggregationResult
    ) -> AggregationResult:
        return result

    def consolidate_client(self, client: Any, method_context: Any) -> None:
        """Run POWER Eqs. (2)--(4) once after the final local round."""

        self._require_initialized()
        if not self._local_enabled or method_context.train_labels.numel() == 0:
            return
        state = self._client_state(client)
        global_state = state["global_model_state"]
        if not isinstance(global_state, Mapping) or not global_state:
            raise RuntimeError("POWER has no final-round global model snapshot.")
        global_model = copy.deepcopy(client.model)
        client.parameter_policy.load(global_model, global_state)
        local_mode = client.model.training
        client.model.eval()
        global_model.eval()
        with torch.no_grad():
            local_embeddings = method_context.encode_nodes(client.model)
            global_embeddings = method_context.encode_nodes(global_model)
        client.model.train(local_mode)
        selected_queries, selected_labels = power_local_global_coverage_selection(
            local_embeddings=local_embeddings,
            global_embeddings=global_embeddings,
            train_queries=method_context.train_queries,
            train_labels=method_context.train_labels,
            alpha=self.alpha,
            coverage_threshold=self.coverage_threshold,
            samples_per_class=self.samples_per_class,
        )
        state["replay"] = power_append_replay(
            replay_state=state["replay"],
            node_features=method_context.node_features,
            selected_queries=selected_queries,
            selected_labels=selected_labels,
            ceiling_bytes=self.replay_ceiling_bytes,
        )
        self._active_replay[int(client.client_id)] = {
            key: value.detach().cpu().clone() for key, value in state["replay"].items()
        }

    def client_replay_payload_bytes(self, client: Any) -> int:
        state = self._client_state(client)
        replay = state["replay"]
        return 0 if not replay else power_replay_payload_bytes(replay)

    def select_evaluation(
        self, client_id: int, stage_index: int
    ) -> EvaluationSelection:
        self._require_initialized()
        if client_id not in self._client_ids or stage_index < 0:
            raise ValueError("Unknown POWER evaluation client or stage.")
        return EvaluationSelection(client_id=client_id, source="post_local")

    def diagnostics(self) -> Mapping[str, object]:
        self._require_initialized()
        recon_losses = self._last_reconstruction_losses
        return {
            "strategy_version": self.strategy_version,
            "fidelity": "paper_equations_2_through_14",
            "paper_server_loss": "kl_global_to_local",
            "official_code_server_loss": "mean_absolute_logits_discrepancy",
            "ablation_mode": self.ablation_mode,
            "alpha": self.alpha,
            "beta": self.beta,
            "trajectory_decay": self.trajectory_decay,
            "coverage_threshold": self.coverage_threshold,
            "replay_ceiling_bytes": self.replay_ceiling_bytes,
            "samples_per_class": self.samples_per_class,
            "server_epochs": self.server_epochs,
            "server_learning_rate": self.server_learning_rate,
            "server_weight_decay": self.server_weight_decay,
            "server_optimization_mode": "eval_deterministic",
            "server_acceptance_rule": "strict_scale_aware_kl_descent",
            "server_backtrack_factor": POWER_BACKTRACK_FACTOR,
            "server_max_backtrack_steps": POWER_MAX_BACKTRACK_STEPS,
            "reconstruction_steps": self.reconstruction_steps,
            "gradient_encoder_initialization": "torch_linear_default_seeded",
            "lbfgs_max_iter_per_outer_step": 1,
            "completed_rounds": self._completed_rounds,
            "pseudo_prototypes": int(self._server_features.shape[0]),
            "pseudo_graph_edges": int(self._pseudo_edges.shape[1]),
            "trajectory_stages": sorted(self._trajectory_stages),
            "server_transfer_losses": list(self._last_server_losses),
            "last_server_transfer_report": dict(
                self._last_server_transfer_report
            ),
            "server_changed_rounds": self._server_changed_rounds,
            "server_zero_signal_rounds": self._server_zero_signal_rounds,
            "server_rollback_rounds": self._server_rollback_rounds,
            "server_attempted_epochs": self._server_attempted_epochs,
            "server_accepted_epochs": self._server_accepted_epochs,
            "server_proposal_attempts": self._server_proposal_attempts,
            "server_rejected_proposals": self._server_rejected_proposals,
            "server_backtracking_reductions": self._server_backtracking_reductions,
            "reconstruction_initial_loss_mean": (
                sum(pair[0] for pair in recon_losses) / len(recon_losses)
                if recon_losses
                else None
            ),
            "reconstruction_final_loss_mean": (
                sum(pair[1] for pair in recon_losses) / len(recon_losses)
                if recon_losses
                else None
            ),
            "clients_with_replay_loss": sorted(self._last_replay_losses),
            "last_replay_losses": {
                f"client-{client_id}": value
                for client_id, value in sorted(self._last_replay_losses.items())
            },
            "gradient_upload_contains_raw_prototypes": False,
            "gradient_upload_contains_raw_labels": False,
            "task_aware": bool(self._task_aware),
            "task_class_masks": sorted(self._task_class_masks),
        }

    def state_dict(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "ablation_mode": self.ablation_mode,
            "client_ids": self._client_ids,
            "shared_state": self._shared_state.materialize(),
            "encoder_state": self._encoder_state.materialize(),
            "server_features": self._server_features.detach().cpu().clone(),
            "server_labels": self._server_labels.detach().cpu().clone(),
            "trajectory": self._trajectory.detach().cpu().clone(),
            "pseudo_edges": self._pseudo_edges.detach().cpu().clone(),
            "last_local_states": {
                client_id: state.materialize()
                for client_id, state in self._last_local_states.items()
            },
            "last_replay_losses": dict(self._last_replay_losses),
            "last_server_losses": self._last_server_losses,
            "last_server_transfer_report": dict(
                self._last_server_transfer_report
            ),
            "last_reconstruction_losses": self._last_reconstruction_losses,
            "trajectory_stages": tuple(sorted(self._trajectory_stages)),
            "server_changed_rounds": self._server_changed_rounds,
            "server_zero_signal_rounds": self._server_zero_signal_rounds,
            "server_rollback_rounds": self._server_rollback_rounds,
            "server_attempted_epochs": self._server_attempted_epochs,
            "server_accepted_epochs": self._server_accepted_epochs,
            "server_proposal_attempts": self._server_proposal_attempts,
            "server_rejected_proposals": self._server_rejected_proposals,
            "server_backtracking_reductions": self._server_backtracking_reductions,
            "completed_rounds": self._completed_rounds,
            "task_aware": self._task_aware,
            "task_class_masks": {
                task: mask.detach().cpu().clone()
                for task, mask in self._task_class_masks.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._require_initialized()
        expected = set(self.state_dict())
        if (
            set(state) != expected
            or state["strategy_version"] != self.strategy_version
            or state["ablation_mode"] != self.ablation_mode
            or tuple(state["client_ids"]) != self._client_ids
        ):
            raise ValueError("POWER checkpoint identity mismatch.")
        self._shared_state = FrozenTensorMap(
            self._validate_state(state["shared_state"])
        )
        self._encoder_state = FrozenTensorMap(state["encoder_state"])
        if self._encoder is None:
            raise RuntimeError("POWER encoder is unavailable during restore.")
        self._encoder.load_state_dict(
            {
                name: value.to(self._device)
                for name, value in self._encoder_state.materialize().items()
            },
            strict=True,
        )
        self._server_features = state["server_features"].detach().cpu().clone()
        self._server_labels = state["server_labels"].detach().cpu().clone().long()
        self._trajectory = state["trajectory"].detach().cpu().clone()
        self._pseudo_edges = state["pseudo_edges"].detach().cpu().clone().long()
        local_states = state["last_local_states"]
        if not isinstance(local_states, Mapping) or set(local_states) != set(
            self._client_ids
        ):
            raise ValueError("POWER checkpoint local-model states are invalid.")
        self._last_local_states = {
            int(client_id): FrozenTensorMap(value)
            for client_id, value in local_states.items()
        }
        self._last_replay_losses = {
            int(client_id): float(value)
            for client_id, value in state["last_replay_losses"].items()
        }
        self._last_server_losses = tuple(
            float(value) for value in state["last_server_losses"]
        )
        report = state["last_server_transfer_report"]
        if not isinstance(report, Mapping):
            raise ValueError("POWER checkpoint server-transfer report is invalid.")
        self._last_server_transfer_report = dict(report)
        self._last_reconstruction_losses = tuple(
            (float(pair[0]), float(pair[1]))
            for pair in state["last_reconstruction_losses"]
        )
        self._trajectory_stages = {
            int(value) for value in state["trajectory_stages"]
        }
        for name in (
            "server_changed_rounds",
            "server_zero_signal_rounds",
            "server_rollback_rounds",
            "server_attempted_epochs",
            "server_accepted_epochs",
            "server_proposal_attempts",
            "server_rejected_proposals",
            "server_backtracking_reductions",
        ):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"POWER checkpoint {name} is invalid.")
            setattr(self, f"_{name}", value)
        completed = state["completed_rounds"]
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise ValueError("POWER checkpoint completed rounds are invalid.")
        self._completed_rounds = completed
        task_aware = state["task_aware"]
        if task_aware not in {None, True, False}:
            raise ValueError("POWER checkpoint task-aware flag is invalid.")
        masks = state["task_class_masks"]
        if not isinstance(masks, Mapping):
            raise ValueError("POWER checkpoint task masks are invalid.")
        self._task_aware = task_aware
        self._task_class_masks = {
            int(task): mask.detach().cpu().clone().bool()
            for task, mask in masks.items()
        }


PowerUEFAStrategy = PaperPowerUEFAStrategy
