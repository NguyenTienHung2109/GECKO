"""Stateful compatibility adapter for the unchanged UEFA v1 strategies."""

from __future__ import annotations

from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Tuple

import torch
from torch import nn

from gecko.types import LocalUpdateResult
from gecko.engine.accounting import ResourceLedger
from gecko.engine.aggregation import weighted_average
from gecko.engine.protocol import AggregationResult
from gecko.engine.protocol import BroadcastPayload
from gecko.engine.protocol import BroadcastReason
from gecko.engine.protocol import ClientUpload
from gecko.engine.protocol import EvaluationSelection
from gecko.engine.protocol import FrozenTensorMap
from gecko.engine.protocol import ParameterManifest
from gecko.engine.protocol import RoundContext
from gecko.engine.protocol import TensorState
from gecko.algorithms.federated.catalog import STRATEGIES


class LegacyStrategyAdapter:
    """Expose LocalOnly/FedAvg/FedProx through the stateful v2 protocol.

    The adapter does not activate itself. The coordinator uses it only for an
    explicit v2 method configuration; the historical run path remains intact.
    """

    strategy_version = "uefa-legacy-strategy-adapter-v1"

    def __init__(self, strategy_name: str) -> None:
        if strategy_name not in {"local_only", "fedavg", "fedprox"}:
            raise ValueError(
                "LegacyStrategyAdapter supports local_only, fedavg, and fedprox."
            )
        descriptor = STRATEGIES[strategy_name]
        self.name = descriptor.name
        self.aggregates = descriptor.aggregates
        self.oracle = False
        self.uses_proximal_objective = descriptor.uses_proximal_objective
        self._shared_state = FrozenTensorMap()
        self._manifest: ParameterManifest | None = None
        self._client_ids: Tuple[int, ...] = ()
        self._initialized = False

    def initialize(
        self,
        shared_state: TensorState,
        parameter_manifest: ParameterManifest,
        client_ids: Iterable[int],
    ) -> None:
        clients = tuple(sorted(int(value) for value in client_ids))
        if len(set(clients)) != len(clients) or any(value < 0 for value in clients):
            raise ValueError("Strategy client IDs must be unique and non-negative.")
        if set(shared_state) != set(parameter_manifest.shared_trainable):
            raise ValueError(
                "Initial shared state does not match the parameter manifest."
            )
        self._shared_state = FrozenTensorMap(shared_state)
        self._manifest = parameter_manifest
        self._client_ids = clients
        self._initialized = True

    def _require_initialized(self) -> None:
        if not self._initialized or self._manifest is None:
            raise RuntimeError("Strategy adapter has not been initialized.")

    @property
    def shared_state(self) -> dict[str, torch.Tensor]:
        self._require_initialized()
        return self._shared_state.materialize()

    def prepare_payload(
        self,
        context: RoundContext,
        client_id: int,
        reason: BroadcastReason,
    ) -> BroadcastPayload | None:
        self._require_initialized()
        if client_id not in context.participant_ids:
            raise ValueError("Payload client is absent from the round context.")
        if not self.aggregates and reason != "initialization":
            return None
        payload_bytes = self._shared_state.payload_bytes
        resources = ResourceLedger()
        if reason == "initialization":
            resources.add(initialization_model_downlink_bytes=payload_bytes)
        elif reason == "training":
            resources.add(training_model_downlink_bytes=payload_bytes)
        elif reason == "evaluation":
            resources.add(evaluation_sync_bytes=payload_bytes)
        return BroadcastPayload(
            client_id=client_id,
            round_context=context,
            reason=reason,
            model_state=self._shared_state,
            resources=resources,
            metadata={"strategy_version": self.strategy_version},
        )

    def client_receive(
        self,
        client: Any,
        payload: BroadcastPayload,
        method_context: Any,
    ) -> None:
        if payload.client_id != client.client_id:
            raise ValueError("Broadcast payload was delivered to the wrong client.")
        client.load_shared_state(payload.model_state.materialize())
        client.algorithm.on_broadcast(method_context, payload)

    def transform_gradients(
        self,
        model: nn.Module,
        method_context: Any,
        shared_keys: Tuple[str, ...],
    ) -> None:
        return None

    def finalize_upload(
        self,
        client: Any,
        local_result: LocalUpdateResult,
        method_context: Any,
    ) -> ClientUpload:
        if local_result.client_id != client.client_id:
            raise ValueError("Local result belongs to a different client.")
        resources = ResourceLedger()
        if self.aggregates:
            resources.add(training_model_uplink_bytes=local_result.communication_bytes)
        return ClientUpload(
            client_id=local_result.client_id,
            global_task_id=local_result.global_task_id,
            weight=local_result.weight,
            training_loss=local_result.training_loss,
            model_state=local_result.shared_state,
            diagnostics={"shareable_keys": list(local_result.shareable_keys)},
            resources=resources,
        )

    def aggregate(
        self,
        context: RoundContext,
        uploads: Tuple[ClientUpload, ...],
    ) -> AggregationResult:
        self._require_initialized()
        if tuple(upload.client_id for upload in uploads) != context.participant_ids:
            raise ValueError("Upload order/identity does not match the round context.")
        if any(
            upload.global_task_id != context.task_for(upload.client_id)
            for upload in uploads
        ):
            raise ValueError("An upload reported the wrong immutable global task ID.")
        resources = ResourceLedger()
        for upload in uploads:
            resources.merge(upload.resources)
        if self.aggregates:
            legacy_updates = [
                LocalUpdateResult(
                    upload.client_id,
                    upload.global_task_id,
                    upload.model_state.materialize(),
                    upload.weight,
                    upload.training_loss,
                    upload.model_state.payload_bytes,
                    tuple(upload.model_state),
                )
                for upload in uploads
            ]
            self._shared_state = FrozenTensorMap(weighted_average(legacy_updates))
        return AggregationResult(
            shared_state=self._shared_state,
            diagnostics={"strategy_version": self.strategy_version},
            resources=resources,
        )

    def personalize(
        self,
        context: RoundContext,
        result: AggregationResult,
    ) -> AggregationResult:
        return result

    def select_evaluation(
        self,
        client_id: int,
        stage_index: int,
    ) -> EvaluationSelection:
        self._require_initialized()
        if client_id not in self._client_ids:
            raise ValueError("Unknown evaluation client.")
        if self.aggregates:
            return EvaluationSelection(
                client_id=client_id,
                source="post_broadcast",
                model_state=self._shared_state,
                count_evaluation_sync=True,
            )
        return EvaluationSelection(client_id=client_id, source="post_local")

    def diagnostics(self) -> Mapping[str, object]:
        return {
            "strategy_version": self.strategy_version,
            "legacy_strategy": self.name,
            "stateful_v2_adapter": True,
        }

    def state_dict(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "name": self.name,
            "client_ids": self._client_ids,
            "shared_state": self._shared_state.materialize(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._require_initialized()
        if set(state) != {
            "strategy_version",
            "name",
            "client_ids",
            "shared_state",
        }:
            raise ValueError("Legacy strategy checkpoint fields do not match.")
        if (
            state["strategy_version"] != self.strategy_version
            or state["name"] != self.name
        ):
            raise ValueError("Legacy strategy checkpoint identity mismatch.")
        if tuple(state["client_ids"]) != self._client_ids:
            raise ValueError("Legacy strategy checkpoint client IDs mismatch.")
        shared_state = state["shared_state"]
        if not isinstance(shared_state, Mapping):
            raise TypeError("Legacy strategy shared state must be a mapping.")
        if set(shared_state) != set(self._manifest.shared_trainable):
            raise ValueError("Legacy strategy shared keys mismatch.")
        self._shared_state = FrozenTensorMap(shared_state)
