"""Typed, immutable records for stateful UEFA federation strategies.

This module is deliberately independent of the coordinator and stream builder.
It defines wire/state contracts only; importing it cannot expose central stream
objects or change the legacy execution path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import Mapping
from typing import Protocol
from typing import Tuple
from typing import runtime_checkable

import torch
from torch import nn

from gecko.engine.accounting import ResourceLedger


TensorState = Mapping[str, torch.Tensor]
BroadcastReason = Literal["initialization", "training", "evaluation", "resume"]
ModelStateSemantics = Literal["full_shared_state", "delta"]
EvaluationModelSource = Literal[
    "shared", "post_broadcast", "post_local", "personalized"
]
ParameterKind = Literal[
    "shared_trainable", "local_trainable", "local_buffer", "dynamic_metadata"
]


def _owned_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor.")
    return value.detach().clone().contiguous()


class FrozenTensorMap(Mapping[str, torch.Tensor]):
    """An owned tensor mapping whose public reads return defensive clones."""

    __slots__ = ("_items", "_locked")

    def __init__(self, values: TensorState | None = None) -> None:
        values = {} if values is None else values
        if not isinstance(values, Mapping):
            raise TypeError("Tensor state must be a mapping.")
        prepared = []
        for key, value in sorted(values.items()):
            if not isinstance(key, str) or not key:
                raise ValueError("Tensor-state keys must be non-empty strings.")
            prepared.append((key, _owned_tensor(value, name=f"tensor state {key!r}")))
        object.__setattr__(self, "_items", tuple(prepared))
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("FrozenTensorMap is immutable.")
        object.__setattr__(self, name, value)

    def __getitem__(self, key: str) -> torch.Tensor:
        for candidate, value in self._items:
            if candidate == key:
                return value.clone()
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def materialize(self) -> Dict[str, torch.Tensor]:
        """Return an independently owned ordinary dictionary."""

        return {key: value.clone() for key, value in self._items}

    @property
    def payload_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for _, value in self._items)

    def __repr__(self) -> str:
        return (
            f"FrozenTensorMap(keys={tuple(self)!r}, payload_bytes={self.payload_bytes})"
        )


def _freeze_metadata(value: object, *, path: str = "metadata") -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        items = []
        for key, item in sorted(value.items()):
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings.")
            items.append((key, _freeze_metadata(item, path=f"{path}.{key}")))
        return tuple(items)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_metadata(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain only JSON-compatible values; got {type(value).__name__}."
    )


def _thaw_metadata(value: object) -> object:
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {key: _thaw_metadata(item) for key, item in value}
        return [_thaw_metadata(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class RoundContext:
    """Central round metadata with no graph, label, or ownership payload."""

    stage_index: int
    round_index: int
    participant_ids: Tuple[int, ...]
    client_global_task_ids: Tuple[Tuple[int, int], ...] | Mapping[int, int]

    def __post_init__(self) -> None:
        participants = tuple(int(value) for value in self.participant_ids)
        if self.stage_index < 0 or self.round_index < 0:
            raise ValueError("Stage and round indices must be non-negative.")
        if len(set(participants)) != len(participants) or any(
            value < 0 for value in participants
        ):
            raise ValueError("Participant IDs must be unique non-negative integers.")
        raw_tasks = (
            self.client_global_task_ids.items()
            if isinstance(self.client_global_task_ids, Mapping)
            else self.client_global_task_ids
        )
        task_map = tuple((int(client), int(task)) for client, task in raw_tasks)
        if len({client for client, _ in task_map}) != len(task_map):
            raise ValueError("Each participant may have only one global task ID.")
        if {client for client, _ in task_map} != set(participants):
            raise ValueError(
                "Task mapping must contain exactly the round participants."
            )
        if any(task < 0 for _, task in task_map):
            raise ValueError("Global task IDs must be non-negative.")
        ordered_tasks = tuple(
            (client, dict(task_map)[client]) for client in participants
        )
        object.__setattr__(self, "participant_ids", participants)
        object.__setattr__(self, "client_global_task_ids", ordered_tasks)

    def task_for(self, client_id: int) -> int:
        for candidate, task_id in self.client_global_task_ids:
            if candidate == client_id:
                return task_id
        raise KeyError(client_id)


# ResourceLedger is defined once in federation.accounting.


@dataclass(frozen=True)
class BroadcastPayload:
    client_id: int
    round_context: RoundContext
    reason: BroadcastReason
    model_state: FrozenTensorMap | TensorState = field(default_factory=FrozenTensorMap)
    auxiliary_state: FrozenTensorMap | TensorState = field(
        default_factory=FrozenTensorMap
    )
    metadata: object = field(default_factory=dict)
    resources: ResourceLedger = field(default_factory=ResourceLedger)

    def __post_init__(self) -> None:
        if (
            self.client_id < 0
            or self.client_id not in self.round_context.participant_ids
        ):
            raise ValueError(
                "Broadcast client must be a non-negative round participant."
            )
        if self.reason not in {"initialization", "training", "evaluation", "resume"}:
            raise ValueError(f"Unknown broadcast reason: {self.reason!r}.")
        object.__setattr__(self, "model_state", FrozenTensorMap(self.model_state))
        object.__setattr__(
            self, "auxiliary_state", FrozenTensorMap(self.auxiliary_state)
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def metadata_dict(self) -> Dict[str, object]:
        return dict(_thaw_metadata(self.metadata))


@dataclass(frozen=True)
class ClientUpload:
    client_id: int
    global_task_id: int
    weight: int
    training_loss: float
    model_state: FrozenTensorMap | TensorState
    model_state_semantics: ModelStateSemantics = "full_shared_state"
    auxiliary_state: FrozenTensorMap | TensorState = field(
        default_factory=FrozenTensorMap
    )
    diagnostics: object = field(default_factory=dict)
    resources: ResourceLedger = field(default_factory=ResourceLedger)

    def __post_init__(self) -> None:
        if self.client_id < 0 or self.global_task_id < 0:
            raise ValueError("Client and global task IDs must be non-negative.")
        if self.weight < 0:
            raise ValueError("Upload weight must be non-negative.")
        if not math.isfinite(float(self.training_loss)):
            raise ValueError("Training loss must be finite.")
        if self.model_state_semantics not in {"full_shared_state", "delta"}:
            raise ValueError("Unknown model-state semantics.")
        object.__setattr__(self, "model_state", FrozenTensorMap(self.model_state))
        object.__setattr__(
            self, "auxiliary_state", FrozenTensorMap(self.auxiliary_state)
        )
        object.__setattr__(
            self, "diagnostics", _freeze_metadata(self.diagnostics, path="diagnostics")
        )

    def diagnostics_dict(self) -> Dict[str, object]:
        return dict(_thaw_metadata(self.diagnostics))


@dataclass(frozen=True)
class AggregationResult:
    shared_state: FrozenTensorMap | TensorState
    personalized_states: (
        Tuple[Tuple[int, FrozenTensorMap | TensorState], ...]
        | Mapping[int, TensorState]
    ) = ()
    server_auxiliary_state: FrozenTensorMap | TensorState = field(
        default_factory=FrozenTensorMap
    )
    diagnostics: object = field(default_factory=dict)
    resources: ResourceLedger = field(default_factory=ResourceLedger)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shared_state", FrozenTensorMap(self.shared_state))
        raw = (
            self.personalized_states.items()
            if isinstance(self.personalized_states, Mapping)
            else self.personalized_states
        )
        personalized = tuple(
            (int(client_id), FrozenTensorMap(state)) for client_id, state in raw
        )
        if len({client_id for client_id, _ in personalized}) != len(personalized):
            raise ValueError("Personalized client states must have unique IDs.")
        if any(client_id < 0 for client_id, _ in personalized):
            raise ValueError("Personalized client IDs must be non-negative.")
        object.__setattr__(self, "personalized_states", tuple(sorted(personalized)))
        object.__setattr__(
            self, "server_auxiliary_state", FrozenTensorMap(self.server_auxiliary_state)
        )
        object.__setattr__(
            self, "diagnostics", _freeze_metadata(self.diagnostics, path="diagnostics")
        )

    def personalized_state_for(self, client_id: int) -> FrozenTensorMap | None:
        for candidate, state in self.personalized_states:
            if candidate == client_id:
                return FrozenTensorMap(state)
        return None


@dataclass(frozen=True)
class EvaluationSelection:
    client_id: int
    source: EvaluationModelSource
    model_state: FrozenTensorMap | TensorState | None = None
    count_evaluation_sync: bool = False
    topology_overlay_id: str | None = None

    def __post_init__(self) -> None:
        if self.client_id < 0:
            raise ValueError("Evaluation client ID must be non-negative.")
        if self.source not in {
            "shared",
            "post_broadcast",
            "post_local",
            "personalized",
        }:
            raise ValueError(f"Unknown evaluation source: {self.source!r}.")
        requires_state = self.source in {"shared", "post_broadcast", "personalized"}
        if requires_state != (self.model_state is not None):
            raise ValueError(
                f"Evaluation source {self.source!r} "
                f"{'requires' if requires_state else 'must not carry'} model state."
            )
        if self.model_state is not None:
            object.__setattr__(self, "model_state", FrozenTensorMap(self.model_state))
        if self.topology_overlay_id is not None and not self.topology_overlay_id:
            raise ValueError("Topology overlay ID cannot be empty.")


@dataclass(frozen=True)
class ParameterEntry:
    name: str
    kind: ParameterKind
    shape: Tuple[int, ...] = ()
    dtype: str = "python"
    requires_grad: bool = False
    wire_eligible: bool = False
    checkpoint_persistent: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Parameter-manifest names cannot be empty.")
        if self.kind not in {
            "shared_trainable",
            "local_trainable",
            "local_buffer",
            "dynamic_metadata",
        }:
            raise ValueError(f"Unknown parameter kind: {self.kind!r}.")
        if any(int(dimension) < 0 for dimension in self.shape):
            raise ValueError("Parameter dimensions must be non-negative.")
        if self.kind == "shared_trainable" and not self.wire_eligible:
            raise ValueError("Shared trainable parameters must be wire eligible.")
        if self.kind != "shared_trainable" and self.wire_eligible:
            raise ValueError(
                "Only shared trainable parameters are wire eligible by default."
            )


@dataclass(frozen=True)
class ParameterManifest:
    entries: Tuple[ParameterEntry, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        names = [entry.name for entry in entries]
        if len(set(names)) != len(names):
            raise ValueError("Parameter manifest categories must be disjoint by name.")
        object.__setattr__(self, "entries", entries)

    def names(self, kind: ParameterKind) -> Tuple[str, ...]:
        return tuple(entry.name for entry in self.entries if entry.kind == kind)

    @property
    def shared_trainable(self) -> Tuple[str, ...]:
        return self.names("shared_trainable")

    @property
    def local_trainable(self) -> Tuple[str, ...]:
        return self.names("local_trainable")

    @property
    def local_buffers(self) -> Tuple[str, ...]:
        return self.names("local_buffer")

    @property
    def dynamic_metadata(self) -> Tuple[str, ...]:
        return self.names("dynamic_metadata")


def _artifact_digest(
    *, artifact_type: str, version: str, tensors: FrozenTensorMap, metadata: object
) -> str:
    digest = hashlib.sha256()
    digest.update(artifact_type.encode("utf-8"))
    digest.update(b"\0")
    digest.update(version.encode("utf-8"))
    digest.update(b"\0")
    for key, value in tensors._items:
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(
        json.dumps(
            _thaw_metadata(metadata), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    return digest.hexdigest()


@dataclass(frozen=True)
class MethodArtifact:
    artifact_type: str
    version: str
    sha256: str
    payload: FrozenTensorMap | TensorState = field(default_factory=FrozenTensorMap)
    metadata: object = field(default_factory=dict)
    serialized_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.artifact_type or not self.version:
            raise ValueError("Method artifacts require a type and version.")
        if len(self.sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.sha256
        ):
            raise ValueError("Method artifact sha256 must be lowercase hexadecimal.")
        if self.serialized_bytes < 0:
            raise ValueError("Serialized artifact size must be non-negative.")
        payload = FrozenTensorMap(self.payload)
        metadata = _freeze_metadata(self.metadata)
        expected = _artifact_digest(
            artifact_type=self.artifact_type,
            version=self.version,
            tensors=payload,
            metadata=metadata,
        )
        if expected != self.sha256:
            raise ValueError(
                "Method artifact digest does not match its payload and metadata."
            )
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def build(
        cls,
        *,
        artifact_type: str,
        version: str,
        payload: TensorState | None = None,
        metadata: Mapping[str, object] | None = None,
        serialized_bytes: int = 0,
    ) -> "MethodArtifact":
        tensors = FrozenTensorMap(payload)
        frozen_metadata = _freeze_metadata({} if metadata is None else metadata)
        return cls(
            artifact_type=artifact_type,
            version=version,
            sha256=_artifact_digest(
                artifact_type=artifact_type,
                version=version,
                tensors=tensors,
                metadata=frozen_metadata,
            ),
            payload=tensors,
            metadata=frozen_metadata,
            serialized_bytes=serialized_bytes,
        )

    def metadata_dict(self) -> Dict[str, object]:
        """Return a JSON-compatible copy of artifact metadata."""

        return dict(_thaw_metadata(self.metadata))


@runtime_checkable
class StatefulStrategyProtocol(Protocol):
    """Behavioral surface implemented by stateful server strategies."""

    name: str
    aggregates: bool
    oracle: bool

    def initialize(
        self,
        shared_state: TensorState,
        parameter_manifest: ParameterManifest,
        client_ids: Iterable[int],
    ) -> None: ...

    def prepare_payload(
        self, context: RoundContext, client_id: int, reason: BroadcastReason
    ) -> BroadcastPayload | None: ...

    def client_receive(
        self, client: Any, payload: BroadcastPayload, method_context: Any
    ) -> None: ...

    def transform_gradients(
        self, model: nn.Module, method_context: Any, shared_keys: Tuple[str, ...]
    ) -> None: ...

    def finalize_upload(
        self, client: Any, local_result: Any, method_context: Any
    ) -> ClientUpload: ...

    def aggregate(
        self, context: RoundContext, uploads: Tuple[ClientUpload, ...]
    ) -> AggregationResult: ...

    def personalize(
        self, context: RoundContext, result: AggregationResult
    ) -> AggregationResult: ...

    def select_evaluation(
        self, client_id: int, stage_index: int
    ) -> EvaluationSelection: ...

    def diagnostics(self) -> Mapping[str, object]: ...

    def state_dict(self) -> Mapping[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...
