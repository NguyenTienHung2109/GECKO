"""Frozen exact Dirichlet quotas over LC training edge queries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any
from typing import Sequence

import torch

from gecko.data.partitioning.dirichlet.semantic_target import _client_capacities
from gecko.data.partitioning.dirichlet.semantic_target import _project_preferences
from gecko.data.partitioning.dirichlet.semantic_target import deterministic_transport_rounding
from gecko.reproducibility import torch_generator


LC_EDGE_QUERY_QUOTA_VERSION = "lc_exact_edge_query_dirichlet_quota_v1"


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _js_divergence(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return natural-log Jensen--Shannon divergence for count vectors."""

    first = first.detach().cpu().double()
    second = second.detach().cpu().double()
    first = first / first.sum().clamp_min(1)
    second = second / second.sum().clamp_min(1)
    midpoint = 0.5 * (first + second)

    def kl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        valid = left > 0
        return (left[valid] * (left[valid] / right[valid]).log()).sum()

    return float(0.5 * kl(first, midpoint) + 0.5 * kl(second, midpoint))


def _client_js_divergence(counts: torch.Tensor) -> list[float | None]:
    """Compare every realized client row with the realized global margin."""

    values = counts.detach().cpu().long()
    if values.ndim != 2:
        raise ValueError("LC realized count diagnostics require a matrix.")
    global_counts = values.sum(0)
    return [
        None
        if int(row.sum()) == 0
        else _js_divergence(row, global_counts)
        for row in values
    ]


def _group_allocation_diagnostics(counts: torch.Tensor) -> dict[str, Any]:
    """Measure how strongly each semantic group is concentrated by client."""

    values = counts.detach().cpu().double()
    if values.ndim != 2:
        raise ValueError("LC realized count diagnostics require a matrix.")
    num_clients = int(values.shape[0])
    uniform = torch.ones(num_clients, dtype=torch.float64)
    js_values: list[float | None] = []
    normalized_entropy: list[float | None] = []
    maximum_share: list[float | None] = []
    entropy_scale = math.log(num_clients) if num_clients > 1 else 1.0
    for group in range(values.shape[1]):
        allocation = values[:, group]
        total = float(allocation.sum())
        if total <= 0:
            js_values.append(None)
            normalized_entropy.append(None)
            maximum_share.append(None)
            continue
        probabilities = allocation / total
        positive = probabilities > 0
        entropy = -float(
            (probabilities[positive] * probabilities[positive].log()).sum()
        )
        js_values.append(_js_divergence(allocation, uniform))
        normalized_entropy.append(
            entropy / entropy_scale if num_clients > 1 else 1.0
        )
        maximum_share.append(float(probabilities.max()))
    return {
        "dirichlet_group_allocation_js_to_uniform": js_values,
        "dirichlet_group_allocation_js_to_uniform_mean": _finite_mean(js_values),
        "dirichlet_group_allocation_normalized_entropy": normalized_entropy,
        "dirichlet_group_allocation_normalized_entropy_mean": _finite_mean(
            normalized_entropy
        ),
        "dirichlet_group_allocation_max_client_share": maximum_share,
        "dirichlet_group_allocation_max_client_share_mean": _finite_mean(
            maximum_share
        ),
    }


def _finite_mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else sum(finite) / len(finite)


def _column_round(raw: torch.Tensor, column_sums: torch.Tensor) -> torch.Tensor:
    """Round each semantic column exactly without imposing client row margins."""

    floor = torch.floor(raw).long()
    result = floor.clone()
    for group in range(raw.shape[1]):
        missing = int(column_sums[group] - result[:, group].sum())
        order = sorted(
            range(raw.shape[0]),
            key=lambda client: (
                -float(raw[client, group] - floor[client, group]),
                client,
            ),
        )
        for client in order[:missing]:
            result[client, group] += 1
    return result


def _task_counts(
    quota: torch.Tensor, task_class_groups: Sequence[Sequence[int]]
) -> torch.Tensor:
    return torch.stack(
        [
            quota[:, torch.tensor(tuple(group), dtype=torch.long)].sum(1)
            for group in task_class_groups
        ],
        dim=1,
    )


def _enforce_task_lower_bounds(
    quota: torch.Tensor,
    projected: torch.Tensor,
    task_class_groups: Sequence[Sequence[int]],
    minimum: int,
) -> torch.Tensor:
    """Use deterministic 2x2 transportation cycles to enforce task support."""

    values = quota.clone()
    groups = tuple(tuple(int(value) for value in group) for group in task_class_groups)
    if minimum == 0:
        return values
    class_to_task = {
        class_id: task_id
        for task_id, group in enumerate(groups)
        for class_id in group
    }
    if len(class_to_task) != sum(len(group) for group in groups):
        raise ValueError("LC task class groups must be disjoint.")
    maximum_steps = int(values.sum()) * max(1, len(groups))
    for _ in range(maximum_steps + 1):
        counts = _task_counts(values, groups)
        missing = torch.nonzero(counts < minimum, as_tuple=False)
        if missing.numel() == 0:
            return values
        client, task = min(
            ((int(row[0]), int(row[1])) for row in missing.tolist()),
            key=lambda pair: (int(counts[pair[0], pair[1]]), pair[1], pair[0]),
        )
        candidates: list[tuple[tuple[float, int, int, int], int, int, int]] = []
        for wanted_class in groups[task]:
            for donor in range(values.shape[0]):
                if donor == client or values[donor, wanted_class] <= 0:
                    continue
                donor_task = class_to_task[wanted_class]
                if counts[donor, donor_task] <= minimum:
                    continue
                for returned_class in range(values.shape[1]):
                    returned_task = class_to_task.get(returned_class)
                    if (
                        returned_task == task
                        or values[client, returned_class] <= 0
                    ):
                        continue
                    if returned_task is not None and counts[client, returned_task] <= minimum:
                        continue
                    before = (
                        abs(float(values[client, wanted_class] - projected[client, wanted_class]))
                        + abs(float(values[donor, wanted_class] - projected[donor, wanted_class]))
                        + abs(float(values[client, returned_class] - projected[client, returned_class]))
                        + abs(float(values[donor, returned_class] - projected[donor, returned_class]))
                    )
                    after = (
                        abs(float(values[client, wanted_class] + 1 - projected[client, wanted_class]))
                        + abs(float(values[donor, wanted_class] - 1 - projected[donor, wanted_class]))
                        + abs(float(values[client, returned_class] - 1 - projected[client, returned_class]))
                        + abs(float(values[donor, returned_class] + 1 - projected[donor, returned_class]))
                    )
                    candidates.append(
                        (
                            (after - before, wanted_class, returned_class, donor),
                            donor,
                            wanted_class,
                            returned_class,
                        )
                    )
        if not candidates:
            raise ValueError(
                "Frozen LC quota cannot satisfy minimum client-task support without "
                "changing its row or semantic-group margins."
            )
        _, donor, wanted_class, returned_class = min(candidates)
        values[client, wanted_class] += 1
        values[donor, wanted_class] -= 1
        values[client, returned_class] -= 1
        values[donor, returned_class] += 1
    raise AssertionError("LC quota task-support projection exceeded its finite bound.")


@dataclass(frozen=True)
class LCEdgeQueryQuota:
    """A frozen client-by-semantic-group supervised edge-query quota."""

    sampled_dirichlet_proportions: torch.Tensor
    raw_targets: torch.Tensor
    unconstrained_rounded_targets: torch.Tensor
    capacity_independent_projected_targets: torch.Tensor
    integer_quota: torch.Tensor
    row_sums: torch.Tensor
    column_sums: torch.Tensor
    task_counts: torch.Tensor
    alpha_dirichlet: float
    semantic_unit: str
    seed: int
    quota_hash: str
    policy_version: str = LC_EDGE_QUERY_QUOTA_VERSION

    def manipulation_diagnostics(self) -> dict[str, Any]:
        """Describe the realized Dirichlet intervention after exact projection."""

        semantic_js = _client_js_divergence(self.integer_quota)
        task_js = _client_js_divergence(self.task_counts)
        return {
            "dirichlet_realized_population": "selected_training_queries",
            "dirichlet_realized_semantic_unit": self.semantic_unit,
            "dirichlet_realized_reference_distribution": (
                "global_selected_training_query_distribution"
            ),
            "dirichlet_integer_quota_client_by_semantic_group": (
                self.integer_quota.tolist()
            ),
            "dirichlet_task_counts_client_by_task": self.task_counts.tolist(),
            "dirichlet_realized_semantic_js_divergence": semantic_js,
            "dirichlet_realized_semantic_js_divergence_mean": _finite_mean(
                semantic_js
            ),
            "dirichlet_realized_task_js_divergence": task_js,
            "dirichlet_realized_task_js_divergence_mean": _finite_mean(task_js),
            **_group_allocation_diagnostics(self.integer_quota),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "semantic_unit": self.semantic_unit,
            "alpha_dirichlet": self.alpha_dirichlet,
            "seed": self.seed,
            "sampled_dirichlet_proportions": self.sampled_dirichlet_proportions.tolist(),
            "raw_targets": self.raw_targets.tolist(),
            "unconstrained_rounded_targets": self.unconstrained_rounded_targets.tolist(),
            "capacity_independent_projected_targets": (
                self.capacity_independent_projected_targets.tolist()
            ),
            "integer_quota": self.integer_quota.tolist(),
            "row_sums": self.row_sums.tolist(),
            "column_sums": self.column_sums.tolist(),
            "task_counts": self.task_counts.tolist(),
            "quota_hash": self.quota_hash,
        }


def generate_lc_edge_query_quota(
    train_group_counts: Sequence[int],
    *,
    num_clients: int,
    alpha_dirichlet: float,
    supervised_budget_fraction: float,
    seed: int,
    task_class_groups: Sequence[Sequence[int]],
    minimum_train_queries_per_client_task: int,
    semantic_unit: str = "edge_class",
) -> LCEdgeQueryQuota:
    """Draw once, then project exactly to fixed row and semantic-group margins."""

    counts = torch.tensor(tuple(train_group_counts), dtype=torch.long)
    if semantic_unit not in {"edge_class", "immutable_domain_task_id"}:
        raise ValueError(
            "LC v1 supports semantic_unit='edge_class' or "
            "'immutable_domain_task_id' only."
        )
    if counts.ndim != 1 or counts.numel() < 1 or bool((counts <= 0).any()):
        raise ValueError("Every LC semantic group must have positive train supply.")
    if num_clients < 1 or not math.isfinite(alpha_dirichlet) or alpha_dirichlet <= 0:
        raise ValueError("num_clients and alpha_dirichlet must be positive.")
    if not 0 < supervised_budget_fraction <= 1:
        raise ValueError("supervised_budget_fraction must lie in (0, 1].")
    if minimum_train_queries_per_client_task < 0:
        raise ValueError("Minimum train support cannot be negative.")
    column_sums = torch.floor(counts.double() * supervised_budget_fraction).long()
    column_sums = torch.minimum(column_sums.clamp_min(1), counts)
    groups = tuple(tuple(int(value) for value in group) for group in task_class_groups)
    for group in groups:
        if int(column_sums[torch.tensor(group)].sum()) < (
            num_clients * minimum_train_queries_per_client_task
        ):
            raise ValueError(
                "A semantic task budget cannot satisfy minimum client-task support."
            )

    generator = torch_generator(
        seed,
        LC_EDGE_QUERY_QUOTA_VERSION,
        num_clients,
        counts.tolist(),
        f"{alpha_dirichlet:.17g}",
        f"{supervised_budget_fraction:.17g}",
    )
    concentration = torch.full(
        (counts.numel(), num_clients), alpha_dirichlet, dtype=torch.float64
    )
    gamma = torch._standard_gamma(concentration, generator=generator)
    proportions = (gamma / gamma.sum(1, keepdim=True).clamp_min(1e-300)).T
    raw = proportions * column_sums.double().unsqueeze(0)
    unconstrained = _column_round(raw, column_sums)
    row_sums, _ = _client_capacities(int(column_sums.sum()), num_clients, None)
    projected = _project_preferences(raw, row_sums, column_sums)
    integer = deterministic_transport_rounding(projected, row_sums, column_sums)
    integer = _enforce_task_lower_bounds(
        integer,
        projected,
        groups,
        minimum_train_queries_per_client_task,
    )
    task_counts = _task_counts(integer, groups)
    if not torch.equal(integer.sum(0), column_sums) or not torch.equal(
        integer.sum(1), row_sums
    ):
        raise AssertionError("LC quota projection changed a frozen margin.")
    payload = {
        "policy_version": LC_EDGE_QUERY_QUOTA_VERSION,
        "semantic_unit": semantic_unit,
        "alpha_dirichlet": float(alpha_dirichlet),
        "supervised_budget_fraction": float(supervised_budget_fraction),
        "seed": int(seed),
        "train_group_counts": counts.tolist(),
        "sampled_dirichlet_proportions": proportions.tolist(),
        "raw_targets": raw.tolist(),
        "unconstrained_rounded_targets": unconstrained.tolist(),
        "capacity_independent_projected_targets": projected.tolist(),
        "integer_quota": integer.tolist(),
        "row_sums": row_sums.tolist(),
        "column_sums": column_sums.tolist(),
        "task_counts": task_counts.tolist(),
    }
    return LCEdgeQueryQuota(
        sampled_dirichlet_proportions=proportions,
        raw_targets=raw,
        unconstrained_rounded_targets=unconstrained,
        capacity_independent_projected_targets=projected,
        integer_quota=integer,
        row_sums=row_sums,
        column_sums=column_sums,
        task_counts=task_counts,
        alpha_dirichlet=float(alpha_dirichlet),
        semantic_unit=semantic_unit,
        seed=int(seed),
        quota_hash=_payload_hash(payload),
    )
