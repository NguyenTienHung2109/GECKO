"""Stateful SCAFFOLD strategy and its paper-equation diagnostic oracle.

The benchmark strategy in this module is an explicitly disclosed UEFA
adaptation: local optimization remains UEFA's reset-per-round Adam, model
deltas use current-query weights, the server step size is one, and Option I is
used for client-control updates.  The scalar diagnostic at the bottom of this
file is deliberately separate and implements the unweighted SGD equations in
Algorithm 1 of the SCAFFOLD paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import Tuple

import torch
from torch import nn

from gecko.types import LocalUpdateResult
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


_CLIENT_STATE_KEY = "scaffold_uefa_adam_v1"
_VARIANT = "scaffold_uefa_adam_v1"
_VERSION = "uefa-scaffold-strategy-v1"


def _clone_state(values: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().contiguous().clone()
        for key, value in values.items()
    }


def _state_payload_bytes(values: Mapping[str, torch.Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in values.values())


def _tensor_is_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all()) if value.is_floating_point() else True


def _state_l2(values: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for value in values.values():
        total += float(value.detach().double().square().sum())
    return math.sqrt(total)


def _states_equal(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> bool:
    return set(first) == set(second) and all(
        first[key].shape == second[key].shape
        and first[key].dtype == second[key].dtype
        and torch.equal(first[key].detach().cpu(), second[key].detach().cpu())
        for key in first
    )


@dataclass
class _RoundInput:
    round_key: Tuple[int, int]
    global_task_id: int
    base_model: Dict[str, torch.Tensor]
    server_control: Dict[str, torch.Tensor]
    old_client_control: Dict[str, torch.Tensor]
    client: Any | None = None


@dataclass
class _PendingCommit:
    client: Any
    old_control: Dict[str, torch.Tensor]
    new_control: Dict[str, torch.Tensor]
    expected_upload: ClientUpload


class ScaffoldStrategy:
    """UEFA's stateful Adam adaptation of SCAFFOLD Algorithm 1.

    The server owns ``c`` and a last-known mirror of every private ``c_i``.
    A client's authoritative private copy lives under
    ``client.state.strategy_state['scaffold_uefa_adam_v1']``.  Upload
    finalization stages updates only; all server and client controls are
    committed together after every upload has been validated.
    """

    name = "scaffold"
    aggregates = True
    oracle = False
    uses_proximal_objective = False
    strategy_version = _VERSION
    variant = _VARIANT
    control_option = "option_i"
    server_learning_rate = 1.0

    def __init__(
        self,
        *,
        correction_enabled: bool = True,
        control_updates_enabled: bool = True,
    ) -> None:
        if not isinstance(correction_enabled, bool) or not isinstance(
            control_updates_enabled, bool
        ):
            raise TypeError("SCAFFOLD mechanism flags must be booleans.")
        self.correction_enabled = correction_enabled
        self.control_updates_enabled = control_updates_enabled
        self._shared_state = FrozenTensorMap()
        self._server_control = FrozenTensorMap()
        self._client_controls: Dict[int, FrozenTensorMap] = {}
        self._client_update_counts: Dict[int, int] = {}
        self._client_ids: Tuple[int, ...] = ()
        self._shared_keys: Tuple[str, ...] = ()
        self._manifest: ParameterManifest | None = None
        self._round_context: RoundContext | None = None
        self._all_client_weights: Dict[int, int] = {}
        self._last_all_client_weights: Dict[int, int] = {}
        self._round_inputs: Dict[int, _RoundInput] = {}
        self._pending: Dict[int, _PendingCommit] = {}
        self._completed_rounds = 0
        self._initialized = False

    @property
    def is_exact_fedavg_degeneration(self) -> bool:
        """Whether the strategy takes the exact legacy-FedAvg fast path."""

        return not self.correction_enabled and not self.control_updates_enabled

    @property
    def legacy_delegate_name(self) -> str | None:
        """Expose the exact degeneration so a runtime may delegate directly."""

        return "fedavg" if self.is_exact_fedavg_degeneration else None

    def _require_initialized(self) -> None:
        if not self._initialized or self._manifest is None:
            raise RuntimeError("SCAFFOLD has not been initialized.")

    def _validate_state(
        self, values: Mapping[str, torch.Tensor], *, label: str
    ) -> Dict[str, torch.Tensor]:
        if set(values) != set(self._shared_keys):
            raise ValueError(f"{label} keys do not match shared trainable parameters.")
        reference = self._shared_state.materialize()
        output: Dict[str, torch.Tensor] = {}
        for key in self._shared_keys:
            value = values[key]
            expected = reference[key]
            if not torch.is_tensor(value):
                raise TypeError(f"{label} element {key!r} is not a tensor.")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"{label} tensor identity mismatch for {key!r}.")
            if not value.is_floating_point() or not _tensor_is_finite(value):
                raise ValueError(f"{label} tensor {key!r} must be finite floating point.")
            output[key] = value.detach().cpu().contiguous().clone()
        return output

    def initialize(
        self,
        shared_state: TensorState,
        parameter_manifest: ParameterManifest,
        client_ids: Iterable[int],
    ) -> None:
        if self._initialized:
            raise RuntimeError("SCAFFOLD cannot be initialized twice.")
        clients = tuple(sorted(int(value) for value in client_ids))
        if not clients:
            raise ValueError("SCAFFOLD requires at least one client.")
        if len(set(clients)) != len(clients) or any(value < 0 for value in clients):
            raise ValueError("SCAFFOLD client IDs must be unique and non-negative.")
        shared_keys = tuple(parameter_manifest.shared_trainable)
        if not shared_keys or set(shared_state) != set(shared_keys):
            raise ValueError(
                "Initial SCAFFOLD state must contain every shared trainable parameter."
            )
        owned = _clone_state(shared_state)
        for key in shared_keys:
            value = owned[key]
            if not value.is_floating_point() or not _tensor_is_finite(value):
                raise ValueError(
                    f"Shared SCAFFOLD parameter {key!r} must be finite floating point."
                )
        zero = {key: torch.zeros_like(owned[key]) for key in shared_keys}
        self._shared_state = FrozenTensorMap(owned)
        self._server_control = FrozenTensorMap(zero)
        self._client_controls = {
            client_id: FrozenTensorMap(zero) for client_id in clients
        }
        self._client_update_counts = {client_id: 0 for client_id in clients}
        self._client_ids = clients
        self._shared_keys = shared_keys
        self._manifest = parameter_manifest
        self._last_all_client_weights = {client_id: 1 for client_id in clients}
        self._initialized = True

    @property
    def shared_state(self) -> Dict[str, torch.Tensor]:
        self._require_initialized()
        return self._shared_state.materialize()

    @property
    def server_control(self) -> Dict[str, torch.Tensor]:
        self._require_initialized()
        return self._server_control.materialize()

    def client_control(self, client_id: int) -> Dict[str, torch.Tensor]:
        self._require_initialized()
        if client_id not in self._client_controls:
            raise ValueError("Unknown SCAFFOLD client.")
        return self._client_controls[client_id].materialize()

    def _weighted_control(
        self,
        controls: Mapping[int, Mapping[str, torch.Tensor]],
        weights: Mapping[int, int],
    ) -> Dict[str, torch.Tensor]:
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("All-client SCAFFOLD control weight must be positive.")
        reference = self._shared_state.materialize()
        output: Dict[str, torch.Tensor] = {}
        for key in self._shared_keys:
            accumulator = torch.zeros_like(reference[key], dtype=torch.float64)
            for client_id in self._client_ids:
                accumulator.add_(
                    controls[client_id][key].detach().cpu().double(),
                    alpha=weights[client_id] / total,
                )
            output[key] = accumulator.to(reference[key].dtype)
        return output

    def prepare_round(
        self,
        context: RoundContext,
        all_client_weights: Mapping[int, int],
    ) -> None:
        """Freeze all-client query weights and recompute ``c`` before send."""

        self._require_initialized()
        if self._round_context is not None or self._round_inputs or self._pending:
            raise RuntimeError("The previous SCAFFOLD round is still active.")
        if set(all_client_weights) != set(self._client_ids):
            raise ValueError("All-client control weights must cover the full population.")
        weights: Dict[int, int] = {}
        for client_id in self._client_ids:
            value = all_client_weights[client_id]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(
                    "SCAFFOLD all-client weights must be non-negative integers."
                )
            weights[client_id] = value
        if sum(weights.values()) <= 0:
            raise ValueError("At least one all-client SCAFFOLD weight must be positive.")
        if not set(context.participant_ids).issubset(self._client_ids):
            raise ValueError("Round context contains an unknown SCAFFOLD client.")
        if any(weights[client_id] <= 0 for client_id in context.participant_ids):
            raise ValueError("Every active SCAFFOLD client must have positive weight.")
        controls = {
            client_id: state.materialize()
            for client_id, state in self._client_controls.items()
        }
        self._server_control = FrozenTensorMap(
            self._weighted_control(controls, weights)
        )
        self._round_context = context
        self._all_client_weights = weights

    @staticmethod
    def _context_key(context: Any) -> Tuple[int, int]:
        try:
            stage = int(context.stage_index)
            round_index = int(context.round_index)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("SCAFFOLD method context has no valid round identity.") from error
        return stage, round_index

    def prepare_payload(
        self,
        context: RoundContext,
        client_id: int,
        reason: BroadcastReason,
    ) -> BroadcastPayload:
        self._require_initialized()
        if client_id not in context.participant_ids or client_id not in self._client_ids:
            raise ValueError("SCAFFOLD payload client is not a round participant.")
        model = self._shared_state.materialize()
        auxiliary: Dict[str, torch.Tensor] = {}
        resources = ResourceLedger()
        model_bytes = _state_payload_bytes(model)
        if reason == "training":
            if self._round_context is None or self._round_context != context:
                raise RuntimeError(
                    "prepare_round(context, all_client_weights) must precede training payloads."
                )
            if client_id in self._round_inputs:
                raise RuntimeError("Duplicate SCAFFOLD training payload for one client.")
            old_control = self._client_controls[client_id].materialize()
            server_control = self._server_control.materialize()
            if self.correction_enabled:
                auxiliary = server_control
            resources.add(training_model_downlink_bytes=model_bytes)
            if auxiliary:
                resources.add(
                    training_auxiliary_downlink_bytes=_state_payload_bytes(auxiliary)
                )
            self._round_inputs[client_id] = _RoundInput(
                round_key=(context.stage_index, context.round_index),
                global_task_id=context.task_for(client_id),
                base_model=_clone_state(model),
                server_control=_clone_state(server_control),
                old_client_control=_clone_state(old_control),
            )
        elif reason == "initialization":
            if self.correction_enabled:
                auxiliary = self._server_control.materialize()
            resources.add(initialization_model_downlink_bytes=model_bytes)
            if auxiliary:
                resources.add(
                    initialization_auxiliary_downlink_bytes=_state_payload_bytes(
                        auxiliary
                    )
                )
        elif reason == "evaluation":
            resources.add(evaluation_sync_bytes=model_bytes)
        elif reason == "resume":
            if self.correction_enabled:
                auxiliary = self._server_control.materialize()
            resources.add(initialization_model_downlink_bytes=model_bytes)
            if auxiliary:
                resources.add(
                    initialization_auxiliary_downlink_bytes=_state_payload_bytes(
                        auxiliary
                    )
                )
        else:
            raise ValueError(f"Unknown SCAFFOLD broadcast reason {reason!r}.")
        return BroadcastPayload(
            client_id=client_id,
            round_context=context,
            reason=reason,
            model_state=model,
            auxiliary_state=auxiliary,
            metadata={
                "strategy_version": self.strategy_version,
                "variant": self.variant,
                "control_option": self.control_option,
                "correction_enabled": self.correction_enabled,
                "control_updates_enabled": self.control_updates_enabled,
            },
            resources=resources,
        )

    def _client_record(
        self, client: Any, *, create: bool
    ) -> Mapping[str, object]:
        if not hasattr(client, "state") or not hasattr(client.state, "strategy_state"):
            raise TypeError("SCAFFOLD client has no typed strategy state.")
        container = client.state.strategy_state
        if not isinstance(container, dict):
            raise TypeError("Client strategy state must be a dictionary.")
        if _CLIENT_STATE_KEY not in container:
            if not create:
                raise ValueError("Client-private SCAFFOLD control is missing.")
            client_id = int(client.client_id)
            container[_CLIENT_STATE_KEY] = {
                "strategy_version": self.strategy_version,
                "variant": self.variant,
                "control": self._client_controls[client_id].materialize(),
                "update_count": self._client_update_counts[client_id],
            }
        record = container[_CLIENT_STATE_KEY]
        if not isinstance(record, Mapping) or set(record) != {
            "strategy_version",
            "variant",
            "control",
            "update_count",
        }:
            raise ValueError("Client-private SCAFFOLD state schema mismatch.")
        if (
            record["strategy_version"] != self.strategy_version
            or record["variant"] != self.variant
        ):
            raise ValueError("Client-private SCAFFOLD state identity mismatch.")
        update_count = record["update_count"]
        if (
            isinstance(update_count, bool)
            or not isinstance(update_count, int)
            or update_count < 0
        ):
            raise ValueError("Client-private SCAFFOLD update count is invalid.")
        if not isinstance(record["control"], Mapping):
            raise TypeError("Client-private SCAFFOLD control must be a mapping.")
        return record

    def _read_client_control(
        self, client: Any, *, create: bool
    ) -> Dict[str, torch.Tensor]:
        record = self._client_record(client, create=create)
        return self._validate_state(
            record["control"], label="Client-private SCAFFOLD control"
        )

    def client_receive(
        self,
        client: Any,
        payload: BroadcastPayload,
        method_context: Any,
    ) -> None:
        self._require_initialized()
        if payload.client_id != int(client.client_id):
            raise ValueError("SCAFFOLD payload was delivered to the wrong client.")
        expected_model = self._shared_state.materialize()
        if not _states_equal(payload.model_state.materialize(), expected_model):
            raise ValueError("SCAFFOLD payload model does not match server state.")
        private = self._read_client_control(client, create=True)
        mirror = self._client_controls[payload.client_id].materialize()
        if not _states_equal(private, mirror):
            raise ValueError("Client-private SCAFFOLD control differs from server mirror.")
        if payload.reason == "training":
            item = self._round_inputs.get(payload.client_id)
            if item is None or item.round_key != self._context_key(method_context):
                raise ValueError("SCAFFOLD training payload round identity mismatch.")
            if int(method_context.global_task_id) != item.global_task_id:
                raise ValueError("SCAFFOLD immutable global task identity mismatch.")
            expected_auxiliary = (
                item.server_control if self.correction_enabled else {}
            )
            if not _states_equal(
                payload.auxiliary_state.materialize(), expected_auxiliary
            ):
                raise ValueError("SCAFFOLD payload control differs from round snapshot.")
            item.client = client
        client.load_shared_state(payload.model_state.materialize())
        client.algorithm.on_broadcast(method_context, payload)

    def transform_gradients(
        self,
        model: nn.Module,
        method_context: Any,
        shared_keys: Tuple[str, ...],
    ) -> None:
        self._require_initialized()
        if set(shared_keys) != set(self._shared_keys):
            raise ValueError("SCAFFOLD gradient keys do not match the parameter manifest.")
        if not self.correction_enabled:
            return
        client_id = int(method_context.client_id)
        item = self._round_inputs.get(client_id)
        if (
            item is None
            or item.client is None
            or item.round_key != self._context_key(method_context)
        ):
            raise RuntimeError("SCAFFOLD gradients were requested without a received payload.")
        parameters = dict(model.named_parameters())
        for key in self._shared_keys:
            if key not in parameters:
                raise ValueError(f"Shared SCAFFOLD parameter {key!r} is missing.")
            parameter = parameters[key]
            correction = (
                item.server_control[key] - item.old_client_control[key]
            ).to(parameter)
            if not _tensor_is_finite(correction):
                raise ValueError("SCAFFOLD gradient correction is non-finite.")
            if parameter.grad is None:
                parameter.grad = correction.clone()
            else:
                parameter.grad.add_(correction)
            if not _tensor_is_finite(parameter.grad):
                raise ValueError("Corrected SCAFFOLD gradient is non-finite.")

    def _option_i_control(
        self,
        client: Any,
        method_context: Any,
        item: _RoundInput,
        local_shared: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        gradient_capability = getattr(client, "control_gradient_at_shared", None)
        if not callable(gradient_capability):
            raise RuntimeError(
                "SCAFFOLD Option I requires client.control_gradient_at_shared."
            )
        values = gradient_capability(
            method_context,
            _clone_state(item.base_model),
            self._shared_keys,
        )
        if not isinstance(values, Mapping):
            raise TypeError("SCAFFOLD Option-I control gradient must be a mapping.")
        new_control = self._validate_state(
            values, label="SCAFFOLD Option-I control gradient"
        )
        if hasattr(client, "parameter_policy"):
            after = client.parameter_policy.extract(client.model)
            if not _states_equal(after, local_shared):
                raise RuntimeError(
                    "control_gradient_at_shared changed the post-local client model."
                )
        return new_control

    def finalize_upload(
        self,
        client: Any,
        local_result: LocalUpdateResult,
        method_context: Any,
    ) -> ClientUpload:
        self._require_initialized()
        client_id = int(client.client_id)
        if local_result.client_id != client_id:
            raise ValueError("SCAFFOLD local result belongs to another client.")
        if local_result.global_task_id != int(method_context.global_task_id):
            raise ValueError("SCAFFOLD upload changed the immutable global task ID.")
        item = self._round_inputs.get(client_id)
        if (
            item is None
            or item.client is not client
            or item.round_key != self._context_key(method_context)
        ):
            raise RuntimeError("SCAFFOLD upload has no matching received payload.")
        if client_id in self._pending:
            raise RuntimeError("Duplicate SCAFFOLD upload finalization.")
        if set(local_result.shareable_keys) != set(self._shared_keys):
            raise ValueError("SCAFFOLD upload shareable-key identity mismatch.")
        local_shared = self._validate_state(
            local_result.shared_state, label="SCAFFOLD local model"
        )
        if local_result.communication_bytes != _state_payload_bytes(local_shared):
            raise ValueError("SCAFFOLD local model byte count is inconsistent.")
        if self._all_client_weights.get(client_id) != local_result.weight:
            raise ValueError("SCAFFOLD upload weight differs from the frozen query weight.")
        private = self._read_client_control(client, create=False)
        if not _states_equal(private, item.old_client_control):
            raise ValueError("Client-private SCAFFOLD control changed during local training.")

        model_delta = {
            key: local_shared[key] - item.base_model[key]
            for key in self._shared_keys
        }
        if self.control_updates_enabled:
            new_control = self._option_i_control(
                client, method_context, item, local_shared
            )
        else:
            new_control = _clone_state(item.old_client_control)
        control_delta = {
            key: new_control[key] - item.old_client_control[key]
            for key in self._shared_keys
        }

        exact = self.is_exact_fedavg_degeneration
        wire_model = local_shared if exact else model_delta
        wire_control = control_delta if self.control_updates_enabled else {}
        resources = ResourceLedger(
            training_model_uplink_bytes=_state_payload_bytes(wire_model),
            training_auxiliary_uplink_bytes=_state_payload_bytes(wire_control),
        )
        upload = ClientUpload(
            client_id=client_id,
            global_task_id=local_result.global_task_id,
            weight=local_result.weight,
            training_loss=local_result.training_loss,
            model_state=wire_model,
            model_state_semantics="full_shared_state" if exact else "delta",
            auxiliary_state=wire_control,
            diagnostics={
                "strategy_version": self.strategy_version,
                "variant": self.variant,
                "control_option": self.control_option,
                "model_delta_l2": _state_l2(model_delta),
                "control_delta_l2": _state_l2(control_delta),
                "exact_fedavg_degeneration": exact,
            },
            resources=resources,
        )
        self._pending[client_id] = _PendingCommit(
            client=client,
            old_control=_clone_state(item.old_client_control),
            new_control=_clone_state(new_control),
            expected_upload=upload,
        )
        return upload

    def _validate_upload(
        self,
        context: RoundContext,
        upload: ClientUpload,
        pending: _PendingCommit,
    ) -> None:
        if upload.global_task_id != context.task_for(upload.client_id):
            raise ValueError("SCAFFOLD upload has the wrong immutable global task ID.")
        if upload.weight != self._all_client_weights[upload.client_id]:
            raise ValueError("SCAFFOLD active model weight changed within the round.")
        expected = pending.expected_upload
        if (
            upload.model_state_semantics != expected.model_state_semantics
            or upload.weight != expected.weight
            or upload.training_loss != expected.training_loss
            or not _states_equal(
                upload.model_state.materialize(), expected.model_state.materialize()
            )
            or not _states_equal(
                upload.auxiliary_state.materialize(),
                expected.auxiliary_state.materialize(),
            )
            or upload.resources.to_dict() != expected.resources.to_dict()
        ):
            raise ValueError("SCAFFOLD upload differs from its atomically staged value.")

    def _aggregate_model(
        self, uploads: Sequence[ClientUpload]
    ) -> Dict[str, torch.Tensor]:
        total = sum(upload.weight for upload in uploads)
        if total <= 0:
            raise ValueError("SCAFFOLD has no positive active model weight.")
        base = self._shared_state.materialize()
        output: Dict[str, torch.Tensor] = {}
        exact = self.is_exact_fedavg_degeneration
        for key in self._shared_keys:
            if exact:
                accumulator = torch.zeros_like(base[key], dtype=torch.float64)
                for upload in uploads:
                    accumulator.add_(
                        upload.model_state[key].double(),
                        alpha=upload.weight / total,
                    )
            else:
                accumulator = base[key].double()
                for upload in uploads:
                    accumulator.add_(
                        upload.model_state[key].double(),
                        alpha=self.server_learning_rate * upload.weight / total,
                    )
            output[key] = accumulator.to(base[key].dtype)
            if not _tensor_is_finite(output[key]):
                raise ValueError("SCAFFOLD produced a non-finite shared model.")
        return output

    def _write_client_control(
        self, client: Any, control: Mapping[str, torch.Tensor], update_count: int
    ) -> None:
        record = self._client_record(client, create=False)
        record["control"] = _clone_state(control)
        record["update_count"] = update_count

    def aggregate(
        self,
        context: RoundContext,
        uploads: Tuple[ClientUpload, ...],
    ) -> AggregationResult:
        self._require_initialized()
        if self._round_context is None or self._round_context != context:
            raise ValueError("SCAFFOLD aggregation round identity mismatch.")
        client_ids = tuple(upload.client_id for upload in uploads)
        if (
            len(set(client_ids)) != len(client_ids)
            or set(client_ids) != set(context.participant_ids)
            or set(self._pending) != set(context.participant_ids)
        ):
            raise ValueError("SCAFFOLD requires exactly one upload per participant.")
        ordered = tuple(sorted(uploads, key=lambda upload: upload.client_id))

        # Validate every wire object and every private control before computing or
        # committing any persistent update.
        for upload in ordered:
            pending = self._pending[upload.client_id]
            self._validate_upload(context, upload, pending)
            current_private = self._read_client_control(
                pending.client, create=False
            )
            if not _states_equal(current_private, pending.old_control):
                raise ValueError(
                    "Client-private SCAFFOLD control changed before aggregation."
                )

        new_shared = self._aggregate_model(ordered)
        new_controls = {
            client_id: state.materialize()
            for client_id, state in self._client_controls.items()
        }
        new_counts = dict(self._client_update_counts)
        if self.control_updates_enabled:
            for upload in ordered:
                pending = self._pending[upload.client_id]
                new_controls[upload.client_id] = _clone_state(pending.new_control)
                new_counts[upload.client_id] += 1
        new_server_control = self._weighted_control(
            new_controls, self._all_client_weights
        )
        identity_control = self._weighted_control(
            new_controls, self._all_client_weights
        )
        if not _states_equal(new_server_control, identity_control):
            raise RuntimeError("SCAFFOLD server-control identity was not preserved.")

        resources = ResourceLedger()
        for upload in ordered:
            resources.merge(upload.resources)

        # The assignments below cannot execute until all validation and tensor
        # arithmetic have succeeded.  Inactive client records are never touched.
        self._shared_state = FrozenTensorMap(new_shared)
        self._server_control = FrozenTensorMap(new_server_control)
        self._client_controls = {
            client_id: FrozenTensorMap(control)
            for client_id, control in new_controls.items()
        }
        self._client_update_counts = new_counts
        for upload in ordered:
            pending = self._pending[upload.client_id]
            self._write_client_control(
                pending.client,
                new_controls[upload.client_id],
                new_counts[upload.client_id],
            )
        self._completed_rounds += 1
        self._last_all_client_weights = dict(self._all_client_weights)
        self._round_context = None
        self._all_client_weights = {}
        self._round_inputs = {}
        self._pending = {}

        return AggregationResult(
            shared_state=self._shared_state,
            server_auxiliary_state=(
                self._server_control
                if self.correction_enabled or self.control_updates_enabled
                else FrozenTensorMap()
            ),
            diagnostics={
                "strategy_version": self.strategy_version,
                "variant": self.variant,
                "participants": list(sorted(client_ids)),
                "control_identity": "weighted_last_known_clients",
                "server_control_l2": _state_l2(new_server_control),
                "exact_fedavg_degeneration": self.is_exact_fedavg_degeneration,
            },
            resources=resources,
        )

    def personalize(
        self,
        context: RoundContext,
        result: AggregationResult,
    ) -> AggregationResult:
        return result

    def select_evaluation(
        self, client_id: int, stage_index: int
    ) -> EvaluationSelection:
        self._require_initialized()
        if client_id not in self._client_ids or stage_index < 0:
            raise ValueError("Unknown SCAFFOLD evaluation client or stage.")
        return EvaluationSelection(
            client_id=client_id,
            source="post_broadcast",
            model_state=self._shared_state,
            count_evaluation_sync=True,
        )

    def diagnostics(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "variant": self.variant,
            "fidelity": "mechanism_adaptation",
            "optimizer": "uefa_reset_per_round_adam",
            "control_option": self.control_option,
            "server_learning_rate": self.server_learning_rate,
            "correction_enabled": self.correction_enabled,
            "control_updates_enabled": self.control_updates_enabled,
            "exact_fedavg_degeneration": self.is_exact_fedavg_degeneration,
            "completed_rounds": self._completed_rounds,
            "server_control_l2": _state_l2(self._server_control.materialize()),
            "client_control_l2": {
                str(client_id): _state_l2(control.materialize())
                for client_id, control in self._client_controls.items()
            },
            "client_control_update_counts": {
                str(client_id): count
                for client_id, count in self._client_update_counts.items()
            },
        }

    def state_dict(self) -> Mapping[str, object]:
        self._require_initialized()
        if self._round_context is not None or self._round_inputs or self._pending:
            raise RuntimeError("SCAFFOLD can be checkpointed only at a round boundary.")
        return {
            "strategy_version": self.strategy_version,
            "name": self.name,
            "variant": self.variant,
            "control_option": self.control_option,
            "server_learning_rate": self.server_learning_rate,
            "correction_enabled": self.correction_enabled,
            "control_updates_enabled": self.control_updates_enabled,
            "client_ids": self._client_ids,
            "shared_keys": self._shared_keys,
            "shared_state": self._shared_state.materialize(),
            "server_control": self._server_control.materialize(),
            "client_controls": {
                client_id: control.materialize()
                for client_id, control in self._client_controls.items()
            },
            "client_control_update_counts": dict(self._client_update_counts),
            "last_all_client_weights": dict(self._last_all_client_weights),
            "completed_rounds": self._completed_rounds,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._require_initialized()
        expected = {
            "strategy_version",
            "name",
            "variant",
            "control_option",
            "server_learning_rate",
            "correction_enabled",
            "control_updates_enabled",
            "client_ids",
            "shared_keys",
            "shared_state",
            "server_control",
            "client_controls",
            "client_control_update_counts",
            "last_all_client_weights",
            "completed_rounds",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("SCAFFOLD checkpoint fields do not match.")
        if (
            state["strategy_version"] != self.strategy_version
            or state["name"] != self.name
            or state["variant"] != self.variant
            or state["control_option"] != self.control_option
            or state["server_learning_rate"] != self.server_learning_rate
            or state["correction_enabled"] != self.correction_enabled
            or state["control_updates_enabled"] != self.control_updates_enabled
        ):
            raise ValueError("SCAFFOLD checkpoint configuration identity mismatch.")
        if (
            tuple(state["client_ids"]) != self._client_ids
            or tuple(state["shared_keys"]) != self._shared_keys
        ):
            raise ValueError("SCAFFOLD checkpoint population/parameter identity mismatch.")
        shared = state["shared_state"]
        server_control = state["server_control"]
        client_controls = state["client_controls"]
        counts = state["client_control_update_counts"]
        last_weights = state["last_all_client_weights"]
        completed = state["completed_rounds"]
        if not isinstance(shared, Mapping) or not isinstance(server_control, Mapping):
            raise TypeError("SCAFFOLD checkpoint tensor states must be mappings.")
        if not isinstance(client_controls, Mapping) or set(client_controls) != set(
            self._client_ids
        ):
            raise ValueError("SCAFFOLD checkpoint client controls mismatch.")
        if not isinstance(counts, Mapping) or set(counts) != set(self._client_ids):
            raise ValueError("SCAFFOLD checkpoint client control counts mismatch.")
        if not isinstance(last_weights, Mapping) or set(last_weights) != set(
            self._client_ids
        ):
            raise ValueError("SCAFFOLD checkpoint all-client weights mismatch.")
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise ValueError("SCAFFOLD checkpoint completed-round count is invalid.")
        restored_shared = self._validate_state(shared, label="Checkpoint shared model")
        # Control validation must use the restored shared tensor identity. Shapes
        # and dtypes are already invariant across a valid shared checkpoint.
        restored_server = self._validate_state(
            server_control, label="Checkpoint server control"
        )
        restored_clients: Dict[int, Dict[str, torch.Tensor]] = {}
        restored_counts: Dict[int, int] = {}
        restored_weights: Dict[int, int] = {}
        for client_id in self._client_ids:
            weight = last_weights[client_id]
            if isinstance(weight, bool) or not isinstance(weight, int) or weight < 0:
                raise ValueError("Checkpoint all-client control weight is invalid.")
            restored_weights[client_id] = weight
        if sum(restored_weights.values()) <= 0:
            raise ValueError("Checkpoint all-client control weight must be positive.")
        for client_id in self._client_ids:
            if not isinstance(client_controls[client_id], Mapping):
                raise TypeError("Checkpoint client control must be a tensor mapping.")
            restored_clients[client_id] = self._validate_state(
                client_controls[client_id],
                label=f"Checkpoint client {client_id} control",
            )
            count = counts[client_id]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("Checkpoint client control count is invalid.")
            restored_counts[client_id] = count
        expected_server = self._weighted_control(restored_clients, restored_weights)
        if not _states_equal(restored_server, expected_server):
            raise ValueError(
                "Checkpoint server control violates the weighted last-known "
                "client-control identity."
            )
        self._shared_state = FrozenTensorMap(restored_shared)
        self._server_control = FrozenTensorMap(restored_server)
        self._client_controls = {
            client_id: FrozenTensorMap(control)
            for client_id, control in restored_clients.items()
        }
        self._client_update_counts = restored_counts
        self._last_all_client_weights = restored_weights
        self._completed_rounds = completed
        self._round_context = None
        self._all_client_weights = {}
        self._round_inputs = {}
        self._pending = {}


@dataclass(frozen=True)
class ScalarScaffoldDiagnosticResult:
    """One exact, diagnostic-only scalar SCAFFOLD Algorithm-1 round."""

    diagnostic_only: bool
    control_option: Literal["option_i", "option_ii"]
    server_model: float
    server_control: float
    client_controls: Tuple[Tuple[int, float], ...]
    local_models: Tuple[Tuple[int, float], ...]
    model_deltas: Tuple[Tuple[int, float], ...]
    control_deltas: Tuple[Tuple[int, float], ...]

    def client_control(self, client_id: int) -> float:
        return dict(self.client_controls)[client_id]


def scaffold_scalar_sgd_diagnostic_round(
    *,
    server_model: float,
    server_control: float,
    client_controls: Mapping[int, float],
    local_gradient_steps: Mapping[int, Sequence[float]],
    local_learning_rate: float,
    server_learning_rate: float = 1.0,
    control_option: Literal["option_i", "option_ii"] = "option_i",
    option_i_server_gradients: Mapping[int, float] | None = None,
) -> ScalarScaffoldDiagnosticResult:
    """Evaluate the paper's unweighted scalar-SGD equations exactly.

    ``local_gradient_steps[i][k]`` is the scalar stochastic gradient used for
    client ``i`` at local step ``k``.  Option I additionally requires the
    gradient of client ``i`` at the broadcast server model.  This helper is a
    behavioral oracle only and must never be reported as a UEFA benchmark run.
    """

    scalars = (server_model, server_control, local_learning_rate, server_learning_rate)
    if any(not math.isfinite(float(value)) for value in scalars):
        raise ValueError("Scalar SCAFFOLD inputs must be finite.")
    if local_learning_rate <= 0 or server_learning_rate <= 0:
        raise ValueError("Scalar SCAFFOLD learning rates must be positive.")
    controls = {int(key): float(value) for key, value in client_controls.items()}
    if not controls or any(key < 0 for key in controls):
        raise ValueError("Scalar SCAFFOLD requires non-negative client IDs.")
    if any(not math.isfinite(value) for value in controls.values()):
        raise ValueError("Scalar client controls must be finite.")
    mean_control = sum(controls.values()) / len(controls)
    if not math.isclose(server_control, mean_control, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Scalar SCAFFOLD server control must equal mean client control.")
    participants = tuple(sorted(int(key) for key in local_gradient_steps))
    if not participants or not set(participants).issubset(controls):
        raise ValueError("Scalar SCAFFOLD participants must be registered clients.")
    steps = {
        client_id: tuple(float(value) for value in local_gradient_steps[client_id])
        for client_id in participants
    }
    lengths = {len(values) for values in steps.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise ValueError("Scalar SCAFFOLD requires one common positive local-step count.")
    if any(not math.isfinite(value) for values in steps.values() for value in values):
        raise ValueError("Scalar SCAFFOLD local gradients must be finite.")
    if control_option not in {"option_i", "option_ii"}:
        raise ValueError("Scalar SCAFFOLD control option must be option_i or option_ii.")
    if control_option == "option_i":
        if option_i_server_gradients is None or set(option_i_server_gradients) != set(
            participants
        ):
            raise ValueError("Option I requires one server-point gradient per participant.")
        option_i = {
            int(key): float(value) for key, value in option_i_server_gradients.items()
        }
        if any(not math.isfinite(value) for value in option_i.values()):
            raise ValueError("Option-I server-point gradients must be finite.")
    else:
        if option_i_server_gradients is not None:
            raise ValueError("Option II must not receive Option-I gradients.")
        option_i = {}

    local_models: Dict[int, float] = {}
    model_deltas: Dict[int, float] = {}
    control_deltas: Dict[int, float] = {}
    new_controls = dict(controls)
    steps_per_client = next(iter(lengths))
    for client_id in participants:
        local_model = float(server_model)
        for gradient in steps[client_id]:
            local_model -= local_learning_rate * (
                gradient - controls[client_id] + server_control
            )
        local_models[client_id] = local_model
        model_deltas[client_id] = local_model - server_model
        if control_option == "option_i":
            updated_control = option_i[client_id]
        else:
            updated_control = (
                controls[client_id]
                - server_control
                + (server_model - local_model)
                / (steps_per_client * local_learning_rate)
            )
        new_controls[client_id] = updated_control
        control_deltas[client_id] = updated_control - controls[client_id]

    updated_model = server_model + server_learning_rate * (
        sum(model_deltas.values()) / len(participants)
    )
    updated_server_control = server_control + (
        sum(control_deltas.values()) / len(controls)
    )
    if not math.isclose(
        updated_server_control,
        sum(new_controls.values()) / len(new_controls),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Scalar SCAFFOLD control identity was not preserved.")
    return ScalarScaffoldDiagnosticResult(
        diagnostic_only=True,
        control_option=control_option,
        server_model=updated_model,
        server_control=updated_server_control,
        client_controls=tuple(sorted(new_controls.items())),
        local_models=tuple(sorted(local_models.items())),
        model_deltas=tuple(sorted(model_deltas.items())),
        control_deltas=tuple(sorted(control_deltas.items())),
    )
