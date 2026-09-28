"""Fail-closed supply audits for soft-community Dirichlet construction."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Literal

import torch

from gecko.types import ScenarioSpec


SOFT_COMMUNITY_SUPPLY_AUDIT_VERSION = "soft_community_supply_audit_v1"
SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class TaskSplitSupply:
    """Central-only supply counts for one immutable task and split."""

    task: int
    split: SplitName
    available: int
    required: int
    shortage: int
    task_classes: tuple[int, ...]
    class_counts: tuple[tuple[int, int], ...]
    classes_below_client_count: tuple[tuple[int, int], ...]
    zero_sample_classes: tuple[int, ...]


@dataclass(frozen=True)
class SoftCommunitySupplyReport:
    """Serializable feasibility certificate which never changes allocation."""

    status: Literal["feasible", "infeasible_supply"]
    problem_type: str
    dataset_name: str
    num_clients: int
    minimums: dict[str, int]
    rows: tuple[TaskSplitSupply, ...]
    failed_rows: tuple[TaskSplitSupply, ...]
    unassigned_label_classes: tuple[int, ...]
    unassigned_label_counts: tuple[tuple[int, int], ...]
    label_access_policy: str
    version: str = SOFT_COMMUNITY_SUPPLY_AUDIT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible audit payload."""

        return asdict(self)


def _task_classes(spec: ScenarioSpec, task: int) -> tuple[int, ...]:
    configured = spec.task_class_sets.get(task)
    if configured is not None:
        return tuple(sorted(int(value) for value in configured.tolist()))
    train_ids = spec.query_ids_by_task_split[task]["train"]
    if spec.labels.ndim != 1:
        return ()
    return tuple(
        sorted(int(value) for value in torch.unique(spec.labels[train_ids]).tolist())
    )


def audit_soft_community_supply(
    spec: ScenarioSpec,
    *,
    num_clients: int,
    minimum_train_queries_per_client_task: int,
    minimum_validation_queries_per_client_task: int,
    minimum_test_queries_per_client_task: int,
) -> SoftCommunitySupplyReport:
    """Audit raw query supply without relaxing any scientific constraint.

    Validation/test labels are read only for this central post-hoc diagnostic.
    They do not enter quota generation, ownership, Hconn, or Cload optimization.
    """

    if spec.problem_type not in {"NC", "LC"} or spec.labels.ndim != 1:
        raise ValueError("Soft-community supply audit requires single-label NC or LC.")
    if num_clients < 1:
        raise ValueError("num_clients must be positive.")
    minimums = {
        "train": int(minimum_train_queries_per_client_task),
        "val": int(minimum_validation_queries_per_client_task),
        "test": int(minimum_test_queries_per_client_task),
    }
    if min(minimums.values()) < 0:
        raise ValueError("Supply-audit minima cannot be negative.")

    rows: list[TaskSplitSupply] = []
    for task in range(spec.num_tasks):
        classes = _task_classes(spec, task)
        for split in ("train", "val", "test"):
            ids = spec.query_ids_by_task_split[task][split]
            counts = torch.bincount(
                spec.labels[ids].detach().cpu().long(),
                minlength=max(spec.num_classes, max(classes, default=-1) + 1),
            )
            class_counts = tuple((class_id, int(counts[class_id])) for class_id in classes)
            available = int(ids.numel())
            required = num_clients * minimums[split]
            rows.append(
                TaskSplitSupply(
                    task=task,
                    split=split,
                    available=available,
                    required=required,
                    shortage=max(0, required - available),
                    task_classes=classes,
                    class_counts=class_counts,
                    classes_below_client_count=tuple(
                        (class_id, count)
                        for class_id, count in class_counts
                        if count < num_clients
                    ),
                    zero_sample_classes=tuple(
                        class_id for class_id, count in class_counts if count == 0
                    ),
                )
            )
    failed = tuple(row for row in rows if row.shortage > 0)
    assigned_classes = {class_id for row in rows for class_id in row.task_classes}
    valid_labels = spec.labels[spec.labels >= 0].detach().cpu().long()
    global_counts = torch.bincount(valid_labels, minlength=spec.num_classes)
    unassigned = tuple(
        class_id
        for class_id in range(spec.num_classes)
        if class_id not in assigned_classes and int(global_counts[class_id]) > 0
    )
    return SoftCommunitySupplyReport(
        status="infeasible_supply" if failed else "feasible",
        problem_type=spec.problem_type,
        dataset_name=spec.dataset_name,
        num_clients=num_clients,
        minimums=minimums,
        rows=tuple(rows),
        failed_rows=failed,
        unassigned_label_classes=unassigned,
        unassigned_label_counts=tuple(
            (class_id, int(global_counts[class_id])) for class_id in unassigned
        ),
        label_access_policy=(
            "central_posthoc_diagnostic_only_not_used_by_partition_or_exposed_to_clients"
        ),
    )
