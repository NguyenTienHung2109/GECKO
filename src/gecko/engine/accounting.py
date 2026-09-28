"""Exact resource accounting for stateful federated methods."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from typing import Any
from typing import Mapping

import torch


def _checked_bytes(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer byte count.")
    if value < 0:
        raise ValueError(
            f"{name} cannot be negative; byte counts must be non-negative."
        )
    return value


def tensor_payload_bytes(value: Any, *, _seen: set[int] | None = None) -> int:
    """Count unique raw tensor payload bytes in a nested safe-state value."""

    seen = set() if _seen is None else _seen
    if torch.is_tensor(value):
        identifier = id(value)
        if identifier in seen:
            return 0
        seen.add(identifier)
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(tensor_payload_bytes(item, _seen=seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_payload_bytes(item, _seen=seen) for item in value)
    return 0


@dataclass
class ResourceLedger:
    """Split wire and persistent-state counters for one run or event.

    ``communication_payload_bytes`` intentionally retains the UEFA v1
    definition: training model uplink plus training model downlink only.
    Auxiliary method traffic and evaluation synchronization are reported by
    separate counters and never change that legacy value.
    """

    initialization_model_downlink_bytes: int = 0
    initialization_auxiliary_downlink_bytes: int = 0
    training_model_uplink_bytes: int = 0
    training_model_downlink_bytes: int = 0
    training_auxiliary_uplink_bytes: int = 0
    training_auxiliary_downlink_bytes: int = 0
    evaluation_sync_bytes: int = 0
    artifact_distribution_bytes: int = 0
    replay_bytes: int = 0
    method_artifact_bytes: int = 0
    client_persistent_bytes: int = 0
    server_persistent_bytes: int = 0
    topology_overlay_bytes: int = 0
    synthetic_artifact_bytes: int = 0
    personalized_model_bytes: int = 0
    client_checkpoint_bytes: int = 0
    server_checkpoint_bytes: int = 0

    def __post_init__(self) -> None:
        for item in fields(self):
            _checked_bytes(item.name, getattr(self, item.name))

    @property
    def communication_payload_bytes(self) -> int:
        """Return the unchanged UEFA v1 training-model wire total."""

        return self.training_model_uplink_bytes + self.training_model_downlink_bytes

    @property
    def training_wire_bytes(self) -> int:
        return (
            self.communication_payload_bytes
            + self.training_auxiliary_uplink_bytes
            + self.training_auxiliary_downlink_bytes
        )

    @property
    def evaluation_wire_bytes(self) -> int:
        return self.evaluation_sync_bytes

    @property
    def total_wire_bytes(self) -> int:
        return (
            self.initialization_model_downlink_bytes
            + self.initialization_auxiliary_downlink_bytes
            + self.training_wire_bytes
            + self.evaluation_wire_bytes
            + self.artifact_distribution_bytes
        )

    @property
    def persistent_state_bytes(self) -> int:
        """Return logical persistent client, server, replay, and artifact bytes."""

        return (
            self.client_persistent_bytes
            + self.server_persistent_bytes
            + self.replay_bytes
            + self.method_artifact_bytes
            + self.topology_overlay_bytes
            + self.synthetic_artifact_bytes
            + self.personalized_model_bytes
        )

    @property
    def checkpoint_serialized_bytes(self) -> int:
        """Return actual safe-checkpoint bytes, separate from logical state."""

        return self.client_checkpoint_bytes + self.server_checkpoint_bytes

    def add(self, **increments: int) -> None:
        """Add non-negative byte counts to named base counters."""

        valid = {item.name for item in fields(self)}
        unknown = set(increments) - valid
        if unknown:
            raise KeyError(f"Unknown resource counters: {sorted(unknown)}")
        checked = {
            name: _checked_bytes(name, value) for name, value in increments.items()
        }
        for name, value in checked.items():
            setattr(self, name, getattr(self, name) + value)

    def merge(self, other: "ResourceLedger") -> None:
        """Accumulate another ledger without including derived totals."""

        if not isinstance(other, ResourceLedger):
            raise TypeError("ResourceLedger.merge requires another ResourceLedger.")
        self.add(**asdict(other))

    def to_dict(self) -> dict[str, int]:
        """Return base counters and explicitly named derived totals."""

        return {
            **asdict(self),
            "communication_payload_bytes": self.communication_payload_bytes,
            "training_wire_bytes": self.training_wire_bytes,
            "evaluation_wire_bytes": self.evaluation_wire_bytes,
            "total_wire_bytes": self.total_wire_bytes,
            "persistent_state_bytes": self.persistent_state_bytes,
            "checkpoint_serialized_bytes": self.checkpoint_serialized_bytes,
        }

    def as_dict(self, *, include_derived: bool = True) -> dict[str, int]:
        """Compatibility rendering for immutable protocol records."""

        if include_derived:
            return self.to_dict()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResourceLedger":
        """Restore a ledger and verify any supplied derived totals."""

        if not isinstance(value, Mapping):
            raise TypeError("Resource ledger payload must be a mapping.")
        base_names = {item.name for item in fields(cls)}
        derived_names = {
            "communication_payload_bytes",
            "training_wire_bytes",
            "evaluation_wire_bytes",
            "total_wire_bytes",
            "persistent_state_bytes",
            "checkpoint_serialized_bytes",
        }
        unknown = set(value) - base_names - derived_names
        if unknown:
            raise ValueError(f"Unknown resource ledger fields: {sorted(unknown)}")
        missing = base_names - set(value)
        if missing:
            raise ValueError(f"Missing resource ledger fields: {sorted(missing)}")
        ledger = cls(**{name: value[name] for name in base_names})
        rendered = ledger.to_dict()
        for name in derived_names & set(value):
            if value[name] != rendered[name]:
                raise ValueError(f"Resource ledger derived total mismatch: {name}")
        return ledger
