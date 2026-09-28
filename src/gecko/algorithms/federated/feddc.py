"""FedDC persistent-drift strategy primitives for UEFA v2."""

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


def _finite_nonnegative(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def feddc_weighted_state(
    states: Mapping[int, Mapping[str, torch.Tensor]],
    weights: Mapping[int, int],
) -> dict[str, torch.Tensor]:
    """Return a query-weighted state average with exact key/shape checks."""

    if not states or set(states) != set(weights):
        raise ValueError("FedDC states and weights must cover the same clients.")
    total = sum(int(value) for value in weights.values())
    if total <= 0:
        raise ValueError("FedDC aggregate weight must be positive.")
    keys: tuple[str, ...] | None = None
    for state in states.values():
        if keys is None:
            keys = tuple(sorted(state))
        elif tuple(sorted(state)) != keys:
            raise ValueError("FedDC state keys must match.")
    assert keys is not None
    output: dict[str, torch.Tensor] = {}
    for key in keys:
        reference = states[next(iter(states))][key]
        result = torch.zeros_like(reference)
        for client_id, state in states.items():
            tensor = state[key]
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise ValueError("FedDC tensor shapes and dtypes must match.")
            result.add_(tensor.to(result.device), alpha=float(weights[client_id]) / float(total))
        output[key] = result
    return output


def feddc_delta(
    local_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return ``local - reference`` for each shared tensor."""

    if set(local_state) != set(reference_state):
        raise ValueError("FedDC delta states must share keys.")
    return {
        key: local_state[key].detach().clone().contiguous()
        - reference_state[key].detach().clone().contiguous()
        for key in sorted(local_state)
    }


def feddc_corrected_state(
    local_state: Mapping[str, torch.Tensor],
    drift_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the accepted-paper corrected local model ``x_i + h_i``.

    ``h_i`` is the persistent local drift carried from previous rounds.  The
    current round's drift update is applied only after the corrected states and
    aggregate delta are computed, so inactive clients retain last-known state.
    """

    if set(local_state) != set(drift_state):
        raise ValueError("FedDC corrected state keys must match drift keys.")
    return {
        key: local_state[key].detach().clone().contiguous()
        + drift_state[key].detach().clone().contiguous()
        for key in sorted(local_state)
    }


class FedDCStrategy:
    """Persistent drift-correction strategy, benchmark-ineligible until gated."""

    strategy_version = "uefa-feddc-strategy-v1"
    name = "feddc"
    aggregates = True
    oracle = False
    uses_proximal_objective = False

    def __init__(self, *, alpha: float = 0.1, drift_enabled: bool = True) -> None:
        self.alpha = _finite_nonnegative(alpha, name="alpha")
        self.drift_enabled = bool(drift_enabled)
        self._manifest: ParameterManifest | None = None
        self._client_ids: tuple[int, ...] = ()
        self._shared_state = FrozenTensorMap()
        self._client_drifts: dict[int, FrozenTensorMap] = {}
        self._previous_local_deltas: dict[int, FrozenTensorMap] = {}
        self._previous_aggregate_delta = FrozenTensorMap()
        self.gradient_clip_norm = 10.0
        self._learning_rate: float | None = None
        self._local_steps: int | None = None
        self._completed_rounds = 0
        self._initialized = False

    def bind_training_protocol(self, *, learning_rate: float, local_steps: int) -> None:
        """Bind the frozen ``eta`` and ``K`` used by FedDC's Eq. (4)."""

        rate = float(learning_rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("FedDC learning_rate must be finite and positive.")
        if isinstance(local_steps, bool) or not isinstance(local_steps, int) or local_steps <= 0:
            raise ValueError("FedDC local_steps must be a positive integer.")
        self._learning_rate = rate
        self._local_steps = local_steps

    def _require_initialized(self) -> None:
        if not self._initialized or self._manifest is None:
            raise RuntimeError("FedDC has not been initialized.")

    def _validate_state(self, state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._manifest is None:
            raise RuntimeError("FedDC parameter manifest is unavailable.")
        expected = tuple(self._manifest.shared_trainable)
        if set(state) != set(expected):
            raise ValueError("FedDC state keys do not match shared trainables.")
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
                raise ValueError(f"FedDC state tensor {key!r} is invalid.")
            output[key] = value.detach().clone().contiguous()
        return output

    def initialize(
        self,
        shared_state: TensorState,
        parameter_manifest: ParameterManifest,
        client_ids: Iterable[int],
    ) -> None:
        if self._initialized:
            raise RuntimeError("FedDC cannot be initialized twice.")
        clients = tuple(sorted(int(client_id) for client_id in client_ids))
        if not clients or len(set(clients)) != len(clients) or any(client_id < 0 for client_id in clients):
            raise ValueError("FedDC client IDs must be unique non-negative integers.")
        if not tuple(parameter_manifest.shared_trainable) or set(shared_state) != set(parameter_manifest.shared_trainable):
            raise ValueError("FedDC requires every shared trainable parameter.")
        self._manifest = parameter_manifest
        self._client_ids = clients
        initial = {key: value.detach().clone().contiguous() for key, value in shared_state.items()}
        zeros = {key: torch.zeros_like(value) for key, value in initial.items()}
        self._shared_state = FrozenTensorMap(initial)
        self._client_drifts = {client_id: FrozenTensorMap(zeros) for client_id in clients}
        self._previous_local_deltas = {client_id: FrozenTensorMap(zeros) for client_id in clients}
        self._previous_aggregate_delta = FrozenTensorMap(zeros)
        self._initialized = True

    @property
    def shared_state(self) -> dict[str, torch.Tensor]:
        self._require_initialized()
        return self._shared_state.materialize()

    def prepare_payload(
        self, context: RoundContext, client_id: int, reason: BroadcastReason
    ) -> BroadcastPayload:
        self._require_initialized()
        if client_id not in context.participant_ids or client_id not in self._client_ids:
            raise ValueError("FedDC payload client is not a known participant.")
        model_state = self._shared_state
        resources = ResourceLedger()
        if reason == "initialization":
            resources.add(initialization_model_downlink_bytes=model_state.payload_bytes)
        elif reason == "training":
            resources.add(training_model_downlink_bytes=model_state.payload_bytes)
        elif reason == "evaluation":
            resources.add(evaluation_sync_bytes=model_state.payload_bytes)
        elif reason != "resume":
            raise ValueError(f"Unknown FedDC broadcast reason {reason!r}.")
        return BroadcastPayload(
            client_id=client_id,
            round_context=context,
            reason=reason,
            model_state=model_state,
            metadata={
                "strategy_version": self.strategy_version,
                "alpha": self.alpha,
                "drift_enabled": self.drift_enabled,
            },
            resources=resources,
        )

    def client_receive(self, client: Any, payload: BroadcastPayload, method_context: Any) -> None:
        if int(client.client_id) != payload.client_id:
            raise ValueError("FedDC payload was delivered to the wrong client.")
        client.load_shared_state(payload.model_state.materialize())
        client.algorithm.on_broadcast(method_context, payload)

    def augment_loss(
        self,
        model: torch.nn.Module,
        method_context: Any,
        loss: torch.Tensor,
        shared_keys: tuple[str, ...],
    ) -> torch.Tensor:
        """Add FedDC's drift penalty and previous-update correction (Eq. 4)."""

        self._require_initialized()
        if self._learning_rate is None or self._local_steps is None:
            raise RuntimeError("FedDC training protocol has not been bound.")
        if not self.drift_enabled:
            return loss
        client_id = int(method_context.client_id)
        drift = self._client_drifts[client_id].materialize()
        anchor = self._shared_state.materialize()
        previous_local = self._previous_local_deltas[client_id].materialize()
        previous_global = self._previous_aggregate_delta.materialize()
        penalty = loss.new_zeros(())
        correction = loss.new_zeros(())
        beta = 1.0 / (self._learning_rate * self._local_steps)
        parameters = dict(model.named_parameters())
        for name in shared_keys:
            parameter = parameters[name]
            local_drift = drift[name].to(parameter.device, parameter.dtype)
            global_parameter = anchor[name].to(parameter.device, parameter.dtype)
            local_update = previous_local[name].to(parameter.device, parameter.dtype)
            global_update = previous_global[name].to(parameter.device, parameter.dtype)
            penalty = penalty + torch.sum(
                (parameter + local_drift - global_parameter) ** 2
            )
            correction = correction + torch.sum(
                parameter * (local_update - global_update)
            )
        return loss + 0.5 * self.alpha * penalty + beta * correction

    def transform_gradients(
        self, model: torch.nn.Module, method_context: Any, shared_keys: tuple[str, ...]
    ) -> None:
        self._require_initialized()
        if not self.drift_enabled:
            return None
        corrected_parameters: list[torch.nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if name not in shared_keys or parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    f"FedDC produced a non-finite gradient for {name!r}."
                )
            corrected_parameters.append(parameter)
        if corrected_parameters:
            torch.nn.utils.clip_grad_norm_(corrected_parameters, max_norm=self.gradient_clip_norm)
        return None

    def finalize_upload(self, client: Any, local_result: Any, method_context: Any) -> ClientUpload:
        self._require_initialized()
        client_id = int(client.client_id)
        if (
            local_result.client_id != client_id
            or local_result.global_task_id != int(method_context.global_task_id)
            or client_id not in self._client_ids
        ):
            raise ValueError("FedDC local result identity mismatch.")
        model_state = self._validate_state(local_result.shared_state)
        delta = feddc_delta(model_state, self._shared_state.materialize())
        auxiliary = FrozenTensorMap({"delta." + key: value for key, value in delta.items()})
        resources = ResourceLedger(
            training_model_uplink_bytes=FrozenTensorMap(model_state).payload_bytes,
            training_auxiliary_uplink_bytes=auxiliary.payload_bytes,
        )
        return ClientUpload(
            client_id=client_id,
            global_task_id=local_result.global_task_id,
            weight=local_result.weight,
            training_loss=float(local_result.training_loss),
            model_state=model_state,
            auxiliary_state=auxiliary,
            diagnostics={
                "strategy_version": self.strategy_version,
                "alpha": self.alpha,
                "drift_enabled": self.drift_enabled,
            },
            resources=resources,
        )

    def aggregate(self, context: RoundContext, uploads: tuple[ClientUpload, ...]) -> AggregationResult:
        self._require_initialized()
        upload_ids = tuple(upload.client_id for upload in uploads)
        if len(set(upload_ids)) != len(upload_ids) or set(upload_ids) != set(context.participant_ids):
            raise ValueError("FedDC requires exactly one upload per participant.")
        if any(upload.global_task_id != context.task_for(upload.client_id) for upload in uploads):
            raise ValueError("FedDC upload changed an immutable global task ID.")
        old_shared = self._shared_state.materialize()
        states = {upload.client_id: self._validate_state(upload.model_state) for upload in uploads}
        weights = {upload.client_id: upload.weight for upload in uploads}
        resources = ResourceLedger()
        updated_drifts = dict(self._client_drifts)
        local_deltas: dict[int, dict[str, torch.Tensor]] = {}
        for upload in uploads:
            resources.merge(upload.resources)
            local_delta = feddc_delta(states[upload.client_id], old_shared)
            local_deltas[upload.client_id] = local_delta
            self._previous_local_deltas[upload.client_id] = FrozenTensorMap(local_delta)
            if self.drift_enabled:
                previous = self._client_drifts[upload.client_id].materialize()
                updated_drifts[upload.client_id] = FrozenTensorMap(
                    {
                        key: previous[key] + local_delta[key]
                        for key in previous
                    }
                )
        self._client_drifts = updated_drifts
        corrected_states = {
            client_id: feddc_corrected_state(
                state, self._client_drifts[client_id].materialize()
            )
            for client_id, state in states.items()
        }
        new_shared = feddc_weighted_state(corrected_states, weights)
        aggregate_local_delta = feddc_weighted_state(local_deltas, weights)
        self._previous_aggregate_delta = FrozenTensorMap(aggregate_local_delta)
        self._shared_state = FrozenTensorMap(new_shared)
        self._completed_rounds += 1
        resources.add(
            server_persistent_bytes=self._previous_aggregate_delta.payload_bytes,
            client_persistent_bytes=sum(state.payload_bytes for state in self._client_drifts.values())
            + sum(state.payload_bytes for state in self._previous_local_deltas.values()),
        )
        return AggregationResult(
            shared_state=self._shared_state,
            diagnostics={
                "strategy_version": self.strategy_version,
                "aggregation": "accepted_paper_corrected_model_weighted_average",
                "participants": list(sorted(context.participant_ids)),
                "alpha": self.alpha,
                "drift_enabled": self.drift_enabled,
            },
            resources=resources,
        )

    def personalize(self, context: RoundContext, result: AggregationResult) -> AggregationResult:
        return result

    def select_evaluation(self, client_id: int, stage_index: int) -> EvaluationSelection:
        self._require_initialized()
        if client_id not in self._client_ids or stage_index < 0:
            raise ValueError("Unknown FedDC evaluation client or stage.")
        return EvaluationSelection(
            client_id=client_id,
            source="shared",
            model_state=self._shared_state,
            count_evaluation_sync=True,
        )

    def diagnostics(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "fidelity": "paper_faithful_benchmark_adaptation",
            "alpha": self.alpha,
            "drift_enabled": self.drift_enabled,
            "completed_rounds": self._completed_rounds,
            "clients_with_drift": sorted(self._client_drifts),
            "corrected_model_rule": "x_i_plus_updated_h_i",
            "drift_update_rule": "h_i_plus_equals_h_i_plus_x_i_minus_w",
            "linear_correction_rule": "previous_local_minus_previous_aggregate_over_eta_k",
            "penalty_rule": "alpha_over_two_norm_x_i_plus_h_i_minus_w_squared",
            "gradient_stabilization": "finite_norm_clip_v2_fail_closed",
            "gradient_clip_norm": self.gradient_clip_norm,
            "learning_rate": self._learning_rate,
            "local_steps": self._local_steps,
        }

    def state_dict(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "client_ids": self._client_ids,
            "shared_state": self._shared_state.materialize(),
            "client_drifts": {
                client_id: state.materialize() for client_id, state in self._client_drifts.items()
            },
            "previous_local_deltas": {
                client_id: state.materialize()
                for client_id, state in self._previous_local_deltas.items()
            },
            "previous_aggregate_delta": self._previous_aggregate_delta.materialize(),
            "completed_rounds": self._completed_rounds,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._require_initialized()
        expected = {
            "strategy_version",
            "client_ids",
            "shared_state",
            "client_drifts",
            "previous_local_deltas",
            "previous_aggregate_delta",
            "completed_rounds",
        }
        if set(state) != expected or state["strategy_version"] != self.strategy_version:
            raise ValueError("FedDC checkpoint identity mismatch.")
        if tuple(state["client_ids"]) != self._client_ids:
            raise ValueError("FedDC checkpoint client IDs mismatch.")
        completed = state["completed_rounds"]
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise ValueError("FedDC checkpoint completed rounds are invalid.")
        drifts = state["client_drifts"]
        previous = state["previous_local_deltas"]
        if not isinstance(drifts, Mapping) or not isinstance(previous, Mapping):
            raise TypeError("FedDC checkpoint drift states must be mappings.")
        if set(drifts) != set(self._client_ids) or set(previous) != set(self._client_ids):
            raise ValueError("FedDC checkpoint client drift IDs mismatch.")
        self._shared_state = FrozenTensorMap(self._validate_state(state["shared_state"]))
        self._client_drifts = {
            int(client_id): FrozenTensorMap(self._validate_state(value))
            for client_id, value in drifts.items()
        }
        self._previous_local_deltas = {
            int(client_id): FrozenTensorMap(self._validate_state(value))
            for client_id, value in previous.items()
        }
        self._previous_aggregate_delta = FrozenTensorMap(
            self._validate_state(state["previous_aggregate_delta"])
        )
        self._completed_rounds = completed
