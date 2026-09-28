from __future__ import annotations

from gecko.algorithms.continual.dslr.records import DSLR_REPLAY_CEILING_BYTES

from gecko.algorithms.continual.dslr.records import DSLR_REPLAY_CEILING_BYTES

from gecko.algorithms.continual.dslr.records import DSLR_REPLAY_CEILING_BYTES

from typing import Dict
from typing import Iterable
from typing import Sequence
from typing import Tuple
import torch

def _payload_tensor(snapshot: DSLRReplaySnapshot) -> torch.Tensor:
    from gecko.algorithms.continual.dslr.records import DSLRReplaySnapshot
    from gecko.algorithms.continual.dslr.records import serialize_dslr_snapshot
    return torch.frombuffer(
        bytearray(serialize_dslr_snapshot(snapshot)), dtype=torch.uint8
    ).clone()


class DSLRReplayStore:
    """Exact-byte replay store with deterministic whole-snapshot clipping."""

    def __init__(
        self,
        *,
        client_id: int,
        ceiling_bytes: int = DSLR_REPLAY_CEILING_BYTES,
    ) -> None:
        from gecko.algorithms.continual.dslr.records import DSLRReplaySnapshot
        from gecko.algorithms.continual.dslr.records import DSLR_REPLAY_CEILING_BYTES
        from gecko.algorithms.continual.dslr.records import _validate_nonnegative_int
        from gecko.algorithms.continual.dslr.records import _validate_positive_int
        self.client_id = _validate_nonnegative_int(client_id, name="client_id")
        self.ceiling_bytes = _validate_positive_int(ceiling_bytes, name="ceiling_bytes")
        self._payloads: list[torch.Tensor] = []
        self._snapshots_cache: Tuple[DSLRReplaySnapshot, ...] = ()

    @property
    def used_bytes(self) -> int:
        return sum(int(payload.numel()) for payload in self._payloads)

    def snapshots(self) -> Tuple[DSLRReplaySnapshot, ...]:
        from gecko.algorithms.continual.dslr.records import DSLRReplaySnapshot
        from gecko.algorithms.continual.dslr.records import deserialize_dslr_snapshot
        if len(self._snapshots_cache) != len(self._payloads):
            self._snapshots_cache = tuple(
                deserialize_dslr_snapshot(payload) for payload in self._payloads
            )
        return self._snapshots_cache

    def payloads(self) -> Tuple[torch.Tensor, ...]:
        return tuple(payload.clone() for payload in self._payloads)

    def replace(
        self,
        snapshots: Sequence[DSLRReplaySnapshot],
        *,
        seen_classes: Iterable[int],
        requested_count: int,
    ) -> Dict[str, int]:
        from gecko.algorithms.continual.dslr.records import DSLRReplaySnapshot
        from gecko.algorithms.continual.dslr.records import _validate_nonnegative_int
        from gecko.algorithms.continual.dslr.records import _validate_positive_int
        requested = _validate_positive_int(requested_count, name="requested_count")
        classes = tuple(
            sorted(
                {
                    _validate_nonnegative_int(value, name="seen_class")
                    for value in seen_classes
                }
            )
        )
        if not classes:
            raise ValueError("DSLR replay replacement requires seen classes.")
        if len(snapshots) != requested:
            raise ValueError(
                "DSLR selected snapshot count must equal its requested target."
            )
        by_class: Dict[int, list[DSLRReplaySnapshot]] = {
            class_id: [] for class_id in classes
        }
        seen_roots: set[int] = set()
        for snapshot in snapshots:
            if not isinstance(snapshot, DSLRReplaySnapshot):
                raise TypeError("DSLR replay store accepts DSLRReplaySnapshot values.")
            if snapshot.client_id != self.client_id:
                raise ValueError("DSLR snapshot belongs to another client.")
            if snapshot.class_id not in by_class:
                raise ValueError("DSLR snapshot has an unobserved class.")
            if snapshot.source_local_index in seen_roots:
                raise ValueError("DSLR snapshot roots must be unique.")
            seen_roots.add(snapshot.source_local_index)
            by_class[snapshot.class_id].append(snapshot)
        missing = [class_id for class_id, values in by_class.items() if not values]
        if missing:
            raise ValueError(
                f"DSLR selected snapshots omit seen classes {missing}; configuration fails."
            )
        mandatory = [by_class[class_id][0] for class_id in classes]
        mandatory_ids = {snapshot.snapshot_id for snapshot in mandatory}
        extras = [
            snapshot
            for snapshot in snapshots
            if snapshot.snapshot_id not in mandatory_ids
        ]
        ordered = mandatory + extras
        payloads: list[torch.Tensor] = []
        retained_snapshots: list[DSLRReplaySnapshot] = []
        used = 0
        for position, snapshot in enumerate(ordered):
            payload = _payload_tensor(snapshot)
            if used + payload.numel() > self.ceiling_bytes:
                if position < len(mandatory):
                    raise ValueError(
                        "DSLR replay ceiling cannot retain one complete snapshot "
                        "per seen class; configuration fails."
                    )
                continue
            payloads.append(payload)
            retained_snapshots.append(snapshot)
            used += int(payload.numel())
        represented = {snapshot.class_id for snapshot in retained_snapshots}
        if represented != set(classes):
            raise RuntimeError("DSLR byte clipping lost a mandatory seen class.")
        self._payloads = payloads
        self._snapshots_cache = tuple(retained_snapshots)
        return {
            "requested_replay_nodes": requested,
            "achieved_replay_nodes": len(payloads),
            "replay_payload_bytes": used,
            "replay_ceiling_bytes": self.ceiling_bytes,
        }

    def load_payloads(self, payloads: Sequence[torch.Tensor]) -> None:
        from gecko.algorithms.continual.dslr.records import DSLRReplaySnapshot
        from gecko.algorithms.continual.dslr.records import _owned_tensor
        from gecko.algorithms.continual.dslr.records import deserialize_dslr_snapshot
        previous = self._payloads
        previous_cache = self._snapshots_cache
        try:
            checked: list[torch.Tensor] = []
            checked_snapshots: list[DSLRReplaySnapshot] = []
            roots: set[int] = set()
            used = 0
            for payload in payloads:
                owned = _owned_tensor(payload, name="replay payload")
                if owned.dtype != torch.uint8 or owned.ndim != 1:
                    raise ValueError(
                        "DSLR replay payload must be one-dimensional uint8."
                    )
                snapshot = deserialize_dslr_snapshot(owned)
                if snapshot.client_id != self.client_id:
                    raise ValueError("DSLR replay payload belongs to another client.")
                if snapshot.source_local_index in roots:
                    raise ValueError("DSLR replay payloads contain duplicate roots.")
                roots.add(snapshot.source_local_index)
                used += int(owned.numel())
                checked.append(owned)
                checked_snapshots.append(snapshot)
            if used > self.ceiling_bytes:
                raise ValueError("DSLR replay payloads exceed the configured ceiling.")
            self._payloads = checked
            self._snapshots_cache = tuple(checked_snapshots)
        except Exception:
            self._payloads = previous
            self._snapshots_cache = previous_cache
            raise


