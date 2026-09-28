from __future__ import annotations

from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT
from gecko.algorithms.continual.ssm.records import SSM_STORE_FORMAT

from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT

from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT

from typing import Dict
from typing import Iterable
from typing import Mapping
from typing import Sequence
from typing import Tuple
import torch

def _payload_tensor(record: SSMRecord) -> torch.Tensor:
    from gecko.algorithms.continual.ssm.records import SSMRecord
    from gecko.algorithms.continual.ssm.records import serialize_ssm_record
    return torch.frombuffer(
        bytearray(serialize_ssm_record(record)), dtype=torch.uint8
    ).clone()


def _clone_primitive_tree(value: object) -> object:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_clone_primitive_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_primitive_tree(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _clone_primitive_tree(item) for key, item in value.items()}
    raise TypeError(f"Unsupported SSM state value: {type(value).__name__}.")


class SSMReplayStore:
    """Stage-partitioned immutable replay payload store."""

    def __init__(
        self,
        *,
        client_id: int,
        total_ceiling_bytes: int = SSM_REPLAY_CEILING_BYTES,
        stage_count: int = SSM_STAGE_COUNT,
    ) -> None:
        from gecko.algorithms.continual.ssm.records import SSMRecord
        from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
        from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT
        from gecko.algorithms.continual.ssm.records import _validate_nonnegative_int
        self.client_id = _validate_nonnegative_int(client_id, name="client_id")
        if (
            isinstance(total_ceiling_bytes, bool)
            or not isinstance(total_ceiling_bytes, int)
            or total_ceiling_bytes <= 0
        ):
            raise ValueError("total_ceiling_bytes must be a positive integer.")
        if (
            isinstance(stage_count, bool)
            or not isinstance(stage_count, int)
            or stage_count <= 0
        ):
            raise ValueError("stage_count must be a positive integer.")
        self.total_ceiling_bytes = int(total_ceiling_bytes)
        self.stage_count = int(stage_count)
        self._payloads: list[torch.Tensor] = []
        self._allocations: Dict[int, Dict[str, object]] = {}
        self._records_cache: Tuple[SSMRecord, ...] = ()

    @property
    def stage_slice_bytes(self) -> int:
        return self.total_ceiling_bytes // self.stage_count

    @property
    def permanently_unassigned_bytes(self) -> int:
        return self.total_ceiling_bytes - self.stage_slice_bytes * self.stage_count

    @property
    def used_bytes(self) -> int:
        return sum(int(payload.numel()) for payload in self._payloads)

    @property
    def safe_checkpoint_bytes(self) -> int:
        """Exact logical bytes of the uint8 tensors stored in safe checkpoints."""

        return self.used_bytes

    def records(self) -> Tuple[SSMRecord, ...]:
        from gecko.algorithms.continual.ssm.records import SSMRecord
        from gecko.algorithms.continual.ssm.records import deserialize_ssm_record
        if len(self._records_cache) != len(self._payloads):
            self._records_cache = tuple(
                deserialize_ssm_record(payload) for payload in self._payloads
            )
        return self._records_cache

    def allocations(self) -> Dict[int, Dict[str, object]]:
        cloned = _clone_primitive_tree(self._allocations)
        assert isinstance(cloned, dict)
        return cloned

    def reserve_stage(
        self,
        *,
        stage_index: int,
        global_task_id: int,
        observed_classes: Iterable[int],
        candidate_records: Iterable[SSMRecord],
    ) -> Dict[str, object]:
        from gecko.algorithms.continual.ssm.records import SSMRecord
        from gecko.algorithms.continual.ssm.records import _validate_nonnegative_int
        stage = _validate_nonnegative_int(stage_index, name="stage_index")
        task = _validate_nonnegative_int(global_task_id, name="global_task_id")
        if stage >= self.stage_count:
            raise ValueError(
                f"stage_index={stage} exceeds the immutable {self.stage_count}-stage allocation."
            )
        if stage in self._allocations:
            raise ValueError(f"SSM stage {stage} has already been finalized.")
        if self._allocations and stage < max(self._allocations):
            raise ValueError(
                "SSM cannot finalize an older skipped stage; its reserved slice "
                "is permanently unused after later participation."
            )
        if any(
            allocation["global_task_id"] == task
            for allocation in self._allocations.values()
        ):
            raise ValueError(f"SSM global task {task} has already been finalized.")
        classes = tuple(
            sorted(
                {
                    _validate_nonnegative_int(value, name="observed_class")
                    for value in observed_classes
                }
            )
        )
        if not classes:
            raise ValueError("An arrived SSM stage must observe at least one class.")
        grouped: Dict[int, list[SSMRecord]] = {class_id: [] for class_id in classes}
        candidate_ids: set[str] = set()
        candidate_roots: set[tuple[int, int]] = set()
        for record in candidate_records:
            if not isinstance(record, SSMRecord):
                raise TypeError("candidate_records must contain SSMRecord values.")
            if (
                record.client_id != self.client_id
                or record.stage_index != stage
                or record.global_task_id != task
            ):
                raise ValueError(
                    "SSM candidate record identity does not match its stage."
                )
            if record.class_id not in grouped:
                raise ValueError("SSM candidate record belongs to an unobserved class.")
            if record.record_id in candidate_ids:
                raise ValueError("SSM candidate records must be unique.")
            root_key = (record.class_id, record.source_root_index)
            if root_key in candidate_roots:
                raise ValueError("SSM permits only one candidate per class/root.")
            candidate_ids.add(record.record_id)
            candidate_roots.add(root_key)
            grouped[record.class_id].append(record)

        class_quota = self.stage_slice_bytes // len(classes)
        class_remainder = self.stage_slice_bytes - class_quota * len(classes)
        staged_payloads: list[torch.Tensor] = []
        class_allocations: Dict[int, Dict[str, object]] = {}
        for class_id in classes:
            candidates = sorted(
                grouped[class_id],
                key=lambda record: (record.source_root_index, record.record_id),
            )
            remaining = class_quota
            inserted: list[SSMRecord] = []
            rejected = 0
            for record in candidates:
                payload = _payload_tensor(record)
                payload_size = int(payload.numel())
                if payload_size <= remaining:
                    staged_payloads.append(payload)
                    inserted.append(record)
                    remaining -= payload_size
                else:
                    # No partial record and no transfer of this class's unused slice.
                    rejected += 1
            used = class_quota - remaining
            class_allocations[class_id] = {
                "quota_bytes": class_quota,
                "used_bytes": used,
                "unused_bytes": remaining,
                "candidate_count": len(candidates),
                "inserted_count": len(inserted),
                "rejected_count": rejected,
                "record_ids": [record.record_id for record in inserted],
            }
        used = sum(
            int(allocation["used_bytes"]) for allocation in class_allocations.values()
        )
        allocation: Dict[str, object] = {
            "global_task_id": task,
            "observed_classes": list(classes),
            "stage_slice_bytes": self.stage_slice_bytes,
            "class_quota_bytes": class_quota,
            "class_unassigned_remainder_bytes": class_remainder,
            "used_bytes": used,
            "unused_bytes": self.stage_slice_bytes - used,
            "class_allocations": class_allocations,
            "record_ids": [
                record_id
                for class_id in classes
                for record_id in class_allocations[class_id]["record_ids"]
            ],
        }
        if used != sum(int(payload.numel()) for payload in staged_payloads):
            raise RuntimeError("Internal SSM serialized-byte accounting mismatch.")
        self._payloads.extend(staged_payloads)
        self._records_cache = self._records_cache + tuple(
            record
            for class_id in classes
            for record in sorted(
                grouped[class_id],
                key=lambda value: (value.source_root_index, value.record_id),
            )
            if record.record_id
            in set(class_allocations[class_id]["record_ids"])
        )
        self._allocations[stage] = allocation
        return self.allocations()[stage]

    def to_state(self) -> Dict[str, object]:
        from gecko.algorithms.continual.ssm.records import SSM_STORE_FORMAT
        return {
            "format": SSM_STORE_FORMAT,
            "client_id": self.client_id,
            "total_ceiling_bytes": self.total_ceiling_bytes,
            "stage_count": self.stage_count,
            "payloads": [payload.clone() for payload in self._payloads],
            "allocations": self.allocations(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "SSMReplayStore":
        from gecko.algorithms.continual.ssm.records import SSMRecord
        from gecko.algorithms.continual.ssm.records import SSM_STORE_FORMAT
        from gecko.algorithms.continual.ssm.records import _payload_bytes
        from gecko.algorithms.continual.ssm.records import _validate_nonnegative_int
        from gecko.algorithms.continual.ssm.records import deserialize_ssm_record
        expected = {
            "format",
            "client_id",
            "total_ceiling_bytes",
            "stage_count",
            "payloads",
            "allocations",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError(
                "SSM replay-store checkpoint fields do not match the schema."
            )
        if state["format"] != SSM_STORE_FORMAT:
            raise ValueError("Unsupported SSM replay-store checkpoint format.")
        store = cls(
            client_id=state["client_id"],
            total_ceiling_bytes=state["total_ceiling_bytes"],
            stage_count=state["stage_count"],
        )
        payloads = state["payloads"]
        allocations = state["allocations"]
        if not isinstance(payloads, list) or not isinstance(allocations, Mapping):
            raise ValueError("Malformed SSM replay-store checkpoint containers.")
        store._payloads = []
        records: list[SSMRecord] = []
        for payload in payloads:
            if not torch.is_tensor(payload):
                raise ValueError(
                    "SSM replay payload checkpoint entries must be tensors."
                )
            raw = _payload_bytes(payload)
            if len(raw) > store.total_ceiling_bytes:
                raise ValueError("One SSM replay record exceeds the client ceiling.")
            record = deserialize_ssm_record(raw)
            if record.client_id != store.client_id:
                raise ValueError("SSM replay payload belongs to another client.")
            store._payloads.append(
                torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
            )
            records.append(record)
        if len({record.record_id for record in records}) != len(records):
            raise ValueError("SSM replay-store checkpoint contains duplicate records.")
        if records != sorted(
            records,
            key=lambda record: (
                record.stage_index,
                record.class_id,
                record.source_root_index,
                record.record_id,
            ),
        ):
            raise ValueError("SSM replay payload order is not canonical.")
        parsed_allocations: Dict[int, Dict[str, object]] = {}
        for raw_stage, raw_allocation in allocations.items():
            stage = _validate_nonnegative_int(raw_stage, name="allocation stage")
            if stage >= store.stage_count or not isinstance(raw_allocation, Mapping):
                raise ValueError("Malformed SSM stage allocation checkpoint.")
            allocation = _clone_primitive_tree(raw_allocation)
            assert isinstance(allocation, dict)
            parsed_allocations[stage] = allocation
        store._allocations = parsed_allocations
        store._validate_loaded_state(records)
        store._records_cache = tuple(records)
        return store

    def _validate_loaded_state(self, records: Sequence[SSMRecord]) -> None:
        from gecko.algorithms.continual.ssm.records import SSMRecord
        from gecko.algorithms.continual.ssm.records import _validate_nonnegative_int
        by_id = {record.record_id: record for record in records}
        referenced: list[str] = []
        tasks: set[int] = set()
        for stage, allocation in sorted(self._allocations.items()):
            expected_fields = {
                "global_task_id",
                "observed_classes",
                "stage_slice_bytes",
                "class_quota_bytes",
                "class_unassigned_remainder_bytes",
                "used_bytes",
                "unused_bytes",
                "class_allocations",
                "record_ids",
            }
            if set(allocation) != expected_fields:
                raise ValueError("SSM stage allocation fields do not match the schema.")
            task = _validate_nonnegative_int(
                allocation["global_task_id"], name="global_task_id"
            )
            if task in tasks:
                raise ValueError("SSM checkpoint repeats an immutable global task.")
            tasks.add(task)
            classes_raw = allocation["observed_classes"]
            if not isinstance(classes_raw, list):
                raise ValueError("SSM observed_classes must be a list.")
            classes = tuple(
                _validate_nonnegative_int(value, name="observed_class")
                for value in classes_raw
            )
            if not classes or classes != tuple(sorted(set(classes))):
                raise ValueError("SSM observed classes are not canonical.")
            class_quota = self.stage_slice_bytes // len(classes)
            class_remainder = self.stage_slice_bytes - class_quota * len(classes)
            if (
                allocation["stage_slice_bytes"] != self.stage_slice_bytes
                or allocation["class_quota_bytes"] != class_quota
                or allocation["class_unassigned_remainder_bytes"] != class_remainder
            ):
                raise ValueError("SSM immutable slice accounting mismatch.")
            class_allocations = allocation["class_allocations"]
            if not isinstance(class_allocations, Mapping) or set(
                class_allocations
            ) != set(classes):
                raise ValueError("SSM class allocation keys mismatch observed classes.")
            stage_ids: list[str] = []
            stage_used = 0
            for class_id in classes:
                item = class_allocations[class_id]
                fields = {
                    "quota_bytes",
                    "used_bytes",
                    "unused_bytes",
                    "candidate_count",
                    "inserted_count",
                    "rejected_count",
                    "record_ids",
                }
                if not isinstance(item, Mapping) or set(item) != fields:
                    raise ValueError(
                        "SSM class allocation fields do not match the schema."
                    )
                ids = item["record_ids"]
                if not isinstance(ids, list) or any(
                    not isinstance(record_id, str) for record_id in ids
                ):
                    raise ValueError("Malformed SSM class record IDs.")
                selected: list[SSMRecord] = []
                for record_id in ids:
                    record = by_id.get(record_id)
                    if record is None:
                        raise ValueError("SSM allocation references an unknown record.")
                    if (
                        record.stage_index != stage
                        or record.global_task_id != task
                        or record.class_id != class_id
                    ):
                        raise ValueError("SSM allocation/record identity mismatch.")
                    selected.append(record)
                if selected != sorted(
                    selected,
                    key=lambda record: (record.source_root_index, record.record_id),
                ):
                    raise ValueError("SSM class record order is not canonical.")
                used = sum(record.serialized_bytes for record in selected)
                inserted = len(selected)
                candidate_count = _validate_nonnegative_int(
                    item["candidate_count"], name="candidate_count"
                )
                rejected = _validate_nonnegative_int(
                    item["rejected_count"], name="rejected_count"
                )
                if (
                    item["quota_bytes"] != class_quota
                    or item["used_bytes"] != used
                    or item["unused_bytes"] != class_quota - used
                    or item["inserted_count"] != inserted
                    or candidate_count != inserted + rejected
                    or used > class_quota
                ):
                    raise ValueError("SSM class serialized-byte accounting mismatch.")
                stage_ids.extend(ids)
                stage_used += used
            if (
                allocation["record_ids"] != stage_ids
                or allocation["used_bytes"] != stage_used
                or allocation["unused_bytes"] != self.stage_slice_bytes - stage_used
            ):
                raise ValueError("SSM stage serialized-byte accounting mismatch.")
            referenced.extend(stage_ids)
        if referenced != [record.record_id for record in records]:
            raise ValueError("SSM replay records and allocation references differ.")
        if self.used_bytes > self.total_ceiling_bytes:
            raise ValueError("SSM replay-store checkpoint exceeds its client ceiling.")


