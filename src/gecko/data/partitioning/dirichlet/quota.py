"""Exact train-class quotas for the experimental direct-node partition path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any
from typing import Sequence

import torch

from gecko.reproducibility import torch_generator
from gecko.types import ScenarioSpec
from gecko.data.partitioning.dirichlet.semantic_target import SemanticTargetInfeasibleError
from gecko.data.partitioning.dirichlet.semantic_target import _client_capacities
from gecko.data.partitioning.dirichlet.semantic_target import _project_preferences
from gecko.data.partitioning.dirichlet.semantic_target import deterministic_transport_rounding


EXACT_DIRICHLET_QUOTA_VERSION = "direct_exact_balanced_dirichlet_quota_v1"


class ExactDirichletQuotaInfeasibleError(RuntimeError):
    """Raised when no fixed-retry exact quota satisfies train-task support."""


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ExactDirichletQuota:
    """An exact integer client-by-class transportation quota."""

    raw_dirichlet_preferences: torch.Tensor
    projected_real_quota: torch.Tensor
    integer_quota: torch.Tensor
    client_train_capacities: torch.Tensor
    train_class_counts: torch.Tensor
    target_task_counts: torch.Tensor
    task_class_groups: tuple[tuple[int, ...], ...]
    alpha_dirichlet: float
    seed: int
    retry_index: int
    quota_hash: str
    capacity_mode: str
    minimum_support_mode: str = "rejection_sampling"
    policy_version: str = EXACT_DIRICHLET_QUOTA_VERSION

    @property
    def row_sums(self) -> torch.Tensor:
        return self.integer_quota.sum(1)

    @property
    def column_sums(self) -> torch.Tensor:
        return self.integer_quota.sum(0)

    def scientific_payload(self) -> dict[str, Any]:
        payload = {
            "policy_version": self.policy_version,
            "alpha_dirichlet": self.alpha_dirichlet,
            "seed": self.seed,
            "retry_index": self.retry_index,
            "capacity_mode": self.capacity_mode,
            "train_class_counts": self.train_class_counts.tolist(),
            "client_train_capacities": self.client_train_capacities.tolist(),
            "task_class_groups": [list(group) for group in self.task_class_groups],
            "raw_dirichlet_preferences": self.raw_dirichlet_preferences.tolist(),
            "projected_real_quota": self.projected_real_quota.tolist(),
            "integer_quota": self.integer_quota.tolist(),
            "target_task_counts": self.target_task_counts.tolist(),
        }
        if self.minimum_support_mode != "rejection_sampling":
            payload["minimum_support_mode"] = self.minimum_support_mode
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {**self.scientific_payload(), "quota_hash": self.quota_hash}


class ExactDirichletQuotaGenerator:
    """Generate an exact balanced Dirichlet quota with fixed deterministic retries."""

    def __init__(self, *, maximum_retries: int = 64) -> None:
        if maximum_retries < 1:
            raise ValueError("maximum_retries must be positive.")
        self.maximum_retries = int(maximum_retries)

    def generate(
        self,
        train_class_counts: Sequence[int],
        num_clients: int,
        alpha_dirichlet: float,
        seed: int,
        client_train_capacities: Sequence[int] | None,
        task_class_groups: Sequence[Sequence[int]],
        min_train_support_per_client_task: int,
        reserve_minimum_support: bool = False,
    ) -> ExactDirichletQuota:
        counts = torch.tensor(train_class_counts, dtype=torch.long)
        if counts.ndim != 1 or counts.numel() < 1 or bool((counts < 0).any()):
            raise ValueError("train_class_counts must be a nonnegative vector.")
        if num_clients < 1 or not math.isfinite(alpha_dirichlet) or alpha_dirichlet <= 0:
            raise ValueError("num_clients and alpha_dirichlet must be positive.")
        groups = tuple(tuple(int(value) for value in group) for group in task_class_groups)
        if not groups or any(not group for group in groups):
            raise ValueError("task_class_groups must be nonempty.")
        if any(value < 0 or value >= counts.numel() for group in groups for value in group):
            raise ValueError("A task class group references an invalid class.")
        if min_train_support_per_client_task < 0:
            raise ValueError("Minimum train support cannot be negative.")
        if client_train_capacities is None:
            capacities, capacity_mode = _client_capacities(
                int(counts.sum()), num_clients, None
            )
        else:
            capacities = torch.tensor(client_train_capacities, dtype=torch.long)
            if capacities.shape != (num_clients,) or bool((capacities < 0).any()):
                raise ValueError("client_train_capacities must be a nonnegative client vector.")
            if int(capacities.sum()) != int(counts.sum()):
                raise ValueError("Client capacities must sum to the train-node total.")
            capacity_mode = "explicit_exact_capacities"
        for retry in range(self.maximum_retries):
            generator = torch_generator(
                seed,
                EXACT_DIRICHLET_QUOTA_VERSION,
                "retry",
                retry,
                num_clients,
                counts.tolist(),
                f"{alpha_dirichlet:.17g}",
            )
            concentration = torch.full(
                (counts.numel(), num_clients),
                alpha_dirichlet,
                dtype=torch.float64,
            )
            gamma = torch._standard_gamma(concentration, generator=generator)
            theta = gamma / gamma.sum(1, keepdim=True).clamp_min(1e-300)
            raw = theta.transpose(0, 1) * counts.double().unsqueeze(0)
            try:
                if reserve_minimum_support and min_train_support_per_client_task:
                    flattened = [value for group in groups for value in group]
                    if len(flattened) != len(set(flattened)):
                        raise ExactDirichletQuotaInfeasibleError(
                            "Minimum-support reservation requires disjoint task groups."
                        )
                    required_per_client = (
                        len(groups) * min_train_support_per_client_task
                    )
                    if bool((capacities < required_per_client).any()):
                        raise ExactDirichletQuotaInfeasibleError(
                            "A client capacity is below the reserved task-support floor."
                        )
                    base = torch.zeros_like(raw, dtype=torch.long)
                    remaining_counts = counts.clone()
                    for group in groups:
                        required = num_clients * min_train_support_per_client_task
                        if int(counts[list(group)].sum()) < required:
                            raise ExactDirichletQuotaInfeasibleError(
                                "A task does not supply its reserved client support floor."
                            )
                        for slot in range(min_train_support_per_client_task):
                            client_order = sorted(
                                range(num_clients),
                                key=lambda client: (
                                    int(base[client, list(group)].sum()),
                                    (client - retry - slot) % num_clients,
                                ),
                            )
                            for client in client_order:
                                candidates = [
                                    class_id
                                    for class_id in group
                                    if int(remaining_counts[class_id]) > 0
                                ]
                                if not candidates:
                                    raise ExactDirichletQuotaInfeasibleError(
                                        "Reserved task support exhausted its class supply."
                                    )
                                class_id = min(
                                    candidates,
                                    key=lambda value: (
                                        -float(raw[client, value]), value
                                    ),
                                )
                                base[client, class_id] += 1
                                remaining_counts[class_id] -= 1
                    remaining_capacities = capacities - base.sum(1)
                    if int(remaining_counts.sum()) == 0:
                        residual_projected = torch.zeros_like(raw)
                        residual_integer = torch.zeros_like(base)
                    else:
                        residual_projected = _project_preferences(
                            raw, remaining_capacities, remaining_counts
                        )
                        residual_integer = deterministic_transport_rounding(
                            residual_projected,
                            remaining_capacities,
                            remaining_counts,
                        )
                    projected = base.double() + residual_projected
                    integer = base + residual_integer
                else:
                    projected = _project_preferences(raw, capacities, counts)
                    integer = deterministic_transport_rounding(
                        projected, capacities, counts
                    )
            except SemanticTargetInfeasibleError as error:
                raise ExactDirichletQuotaInfeasibleError(str(error)) from error
            task_counts = torch.stack(
                [integer[:, torch.tensor(group)].sum(1) for group in groups],
                dim=1,
            )
            if bool((task_counts < min_train_support_per_client_task).any()):
                continue
            payload = {
                "policy_version": EXACT_DIRICHLET_QUOTA_VERSION,
                "alpha_dirichlet": float(alpha_dirichlet),
                "seed": int(seed),
                "retry_index": retry,
                "capacity_mode": capacity_mode,
                "train_class_counts": counts.tolist(),
                "client_train_capacities": capacities.tolist(),
                "task_class_groups": [list(group) for group in groups],
                "raw_dirichlet_preferences": raw.tolist(),
                "projected_real_quota": projected.tolist(),
                "integer_quota": integer.tolist(),
                "target_task_counts": task_counts.tolist(),
            }
            if reserve_minimum_support:
                payload["minimum_support_mode"] = "deterministic_reservation"
            return ExactDirichletQuota(
                raw_dirichlet_preferences=raw.clone(),
                projected_real_quota=projected.clone(),
                integer_quota=integer.clone(),
                client_train_capacities=capacities.clone(),
                train_class_counts=counts.clone(),
                target_task_counts=task_counts.clone(),
                task_class_groups=groups,
                alpha_dirichlet=float(alpha_dirichlet),
                seed=int(seed),
                retry_index=retry,
                quota_hash=_hash(payload),
                capacity_mode=capacity_mode,
                minimum_support_mode=(
                    "deterministic_reservation"
                    if reserve_minimum_support
                    else "rejection_sampling"
                ),
            )
        raise ExactDirichletQuotaInfeasibleError(
            "No exact Dirichlet quota satisfies train-task support within "
            f"the fixed retry budget ({self.maximum_retries})."
        )

    def generate_from_spec(
        self,
        spec: ScenarioSpec,
        *,
        num_clients: int,
        alpha_dirichlet: float,
        seed: int,
        client_train_capacities: Sequence[int] | None = None,
        min_train_support_per_client_task: int,
        reserve_minimum_support: bool = False,
    ) -> ExactDirichletQuota:
        if spec.problem_type != "NC" or spec.labels.ndim != 1:
            raise ValueError("Direct exact quotas currently require single-label NC.")
        train_ids = torch.cat(
            [spec.query_ids_by_task_split[task]["train"] for task in range(spec.num_tasks)]
        ).long()
        counts = torch.bincount(
            spec.labels[train_ids].long(), minlength=spec.num_classes
        )
        groups = [
            tuple(sorted(int(value) for value in spec.task_class_sets[task]))
            for task in range(spec.num_tasks)
        ]
        return self.generate(
            counts.tolist(),
            num_clients,
            alpha_dirichlet,
            seed,
            client_train_capacities,
            groups,
            min_train_support_per_client_task,
            reserve_minimum_support,
        )
