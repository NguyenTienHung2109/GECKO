"""Frozen train-only balanced Dirichlet semantic targets."""

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


SEMANTIC_TARGET_POLICY_VERSION = "balanced_dirichlet_transport_target_v1"


class SemanticTargetInfeasibleError(RuntimeError):
    """Raised when no target satisfies exact margins and train-task support."""


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SemanticTarget:
    """One immutable-by-convention integer class transportation target."""

    raw_preference_matrix: torch.Tensor
    projected_real_matrix: torch.Tensor
    integer_target_matrix: torch.Tensor
    client_capacities: torch.Tensor
    train_class_counts: torch.Tensor
    target_task_counts: torch.Tensor
    task_class_groups: tuple[tuple[int, ...], ...]
    alpha: float
    seed: int
    retry_index: int
    capacity_mode: str
    target_hash: str
    policy_version: str = SEMANTIC_TARGET_POLICY_VERSION

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "alpha": self.alpha,
            "seed": self.seed,
            "retry_index": self.retry_index,
            "capacity_mode": self.capacity_mode,
            "train_class_counts": self.train_class_counts.tolist(),
            "client_capacities": self.client_capacities.tolist(),
            "task_class_groups": [list(group) for group in self.task_class_groups],
            "raw_preference_matrix": self.raw_preference_matrix.tolist(),
            "projected_real_matrix": self.projected_real_matrix.tolist(),
            "integer_target_matrix": self.integer_target_matrix.tolist(),
            "target_task_counts": self.target_task_counts.tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.scientific_payload(), "target_hash": self.target_hash}


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: float
    order: int


def deterministic_transport_rounding(
    real_matrix: torch.Tensor,
    row_margins: torch.Tensor,
    column_margins: torch.Tensor,
) -> torch.Tensor:
    """Round a real transportation matrix with exact margins and stable ties."""

    real = real_matrix.detach().cpu().double()
    rows = row_margins.detach().cpu().long()
    columns = column_margins.detach().cpu().long()
    if real.ndim != 2 or real.shape != (rows.numel(), columns.numel()):
        raise ValueError("Transportation matrix and margins do not align.")
    if bool((real < -1e-12).any()) or bool((rows < 0).any()) or bool((columns < 0).any()):
        raise ValueError("Transportation inputs must be nonnegative.")
    if int(rows.sum()) != int(columns.sum()):
        raise ValueError("Transportation margins must have equal total mass.")
    floor = torch.floor(real + 1e-12).long()
    row_remaining = rows - floor.sum(1)
    column_remaining = columns - floor.sum(0)
    if bool((row_remaining < 0).any()) or bool((column_remaining < 0).any()):
        raise ValueError("Floor matrix exceeds a requested margin.")
    remaining = int(row_remaining.sum())
    if remaining != int(column_remaining.sum()):
        raise ValueError("Residual transportation margins disagree.")
    if remaining == 0:
        return floor

    row_count, column_count = real.shape
    source = 0
    row_offset = 1
    column_offset = row_offset + row_count
    sink = column_offset + column_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    edge_order = 0

    def add_edge(left: int, right: int, capacity: int, cost: float) -> int:
        nonlocal edge_order
        forward_index = len(graph[left])
        reverse_index = len(graph[right])
        graph[left].append(
            _FlowEdge(right, reverse_index, capacity, cost, edge_order)
        )
        edge_order += 1
        graph[right].append(
            _FlowEdge(left, forward_index, 0, -cost, edge_order)
        )
        edge_order += 1
        return forward_index

    for row in range(row_count):
        add_edge(source, row_offset + row, int(row_remaining[row]), 0.0)
    cell_edges: dict[tuple[int, int], int] = {}
    fractional = real - floor.double()
    for row in range(row_count):
        for column in range(column_count):
            cell_edges[(row, column)] = add_edge(
                row_offset + row,
                column_offset + column,
                1,
                1.0 - 2.0 * float(fractional[row, column]),
            )
    for column in range(column_count):
        add_edge(
            column_offset + column,
            sink,
            int(column_remaining[column]),
            0.0,
        )

    tolerance = 1e-14
    for _ in range(remaining):
        distances = [math.inf] * len(graph)
        path_keys: list[tuple[int, ...] | None] = [None] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source] = 0.0
        path_keys[source] = ()
        for _pass in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                if not math.isfinite(distances[node]):
                    continue
                assert path_keys[node] is not None
                for index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    proposal = distances[node] + edge.cost
                    proposal_key = (*path_keys[node], edge.order)
                    current_key = path_keys[edge.to]
                    if proposal < distances[edge.to] - tolerance or (
                        abs(proposal - distances[edge.to]) <= tolerance
                        and (current_key is None or proposal_key < current_key)
                    ):
                        distances[edge.to] = proposal
                        path_keys[edge.to] = proposal_key
                        previous[edge.to] = (node, index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            raise SemanticTargetInfeasibleError(
                "Transportation rounding has no residual augmenting path."
            )
        node = sink
        while node != source:
            parent, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent

    result = floor.clone()
    for (row, column), index in cell_edges.items():
        edge = graph[row_offset + row][index]
        result[row, column] += 1 - edge.capacity
    if not torch.equal(result.sum(1), rows) or not torch.equal(result.sum(0), columns):
        raise AssertionError("Deterministic transportation rounding lost a margin.")
    return result


def _client_capacities(
    total: int,
    num_clients: int,
    weights: Sequence[float] | None,
) -> tuple[torch.Tensor, str]:
    if weights is None:
        base, remainder = divmod(total, num_clients)
        capacities = torch.full((num_clients,), base, dtype=torch.long)
        capacities[:remainder] += 1
        return capacities, "equal_floor_remainder"
    values = torch.tensor(weights, dtype=torch.float64)
    if values.shape != (num_clients,) or bool((values < 0).any()) or float(values.sum()) <= 0:
        raise ValueError("Capacity weights must be a nonnegative client vector.")
    real = values / values.sum() * total
    capacities = torch.floor(real).long()
    remainder = total - int(capacities.sum())
    order = sorted(
        range(num_clients),
        key=lambda client: (-float(real[client] - capacities[client]), client),
    )
    for client in order[:remainder]:
        capacities[client] += 1
    return capacities, "configured_capacity_weights"


def _project_preferences(
    preferences: torch.Tensor,
    row_margins: torch.Tensor,
    column_margins: torch.Tensor,
    *,
    tolerance: float = 1e-10,
    maximum_iterations: int = 20_000,
) -> torch.Tensor:
    matrix = preferences.detach().cpu().double().clamp_min(1e-300)
    rows = row_margins.double()
    columns = column_margins.double()
    for _ in range(maximum_iterations):
        matrix *= (columns / matrix.sum(0).clamp_min(1e-300)).unsqueeze(0)
        matrix *= (rows / matrix.sum(1).clamp_min(1e-300)).unsqueeze(1)
        error = max(
            float((matrix.sum(0) - columns).abs().max()),
            float((matrix.sum(1) - rows).abs().max()),
        )
        if error <= tolerance:
            return matrix
    raise SemanticTargetInfeasibleError(
        "Iterative proportional fitting did not converge within its fixed budget."
    )


class BalancedDirichletTargetGenerator:
    """Generate one exact-margin train-class target with deterministic retries."""

    def __init__(self, *, maximum_retries: int = 64) -> None:
        if maximum_retries < 1:
            raise ValueError("maximum_retries must be positive.")
        self.maximum_retries = int(maximum_retries)

    def generate(
        self,
        train_class_counts: Sequence[int],
        num_clients: int,
        alpha: float,
        seed: int,
        client_capacity_weights: Sequence[float] | None,
        task_class_groups: Sequence[Sequence[int]],
        min_train_support_per_client_task: int,
    ) -> SemanticTarget:
        counts = torch.tensor(train_class_counts, dtype=torch.long)
        if counts.ndim != 1 or counts.numel() < 1 or bool((counts < 0).any()):
            raise ValueError("Train class counts must be a nonnegative vector.")
        if num_clients < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("num_clients and alpha must be positive.")
        groups = tuple(tuple(int(value) for value in group) for group in task_class_groups)
        if not groups or any(not group for group in groups):
            raise ValueError("Task class groups must be nonempty.")
        if any(value < 0 or value >= counts.numel() for group in groups for value in group):
            raise ValueError("Task class group contains an invalid class index.")
        if min_train_support_per_client_task < 0:
            raise ValueError("Minimum train support cannot be negative.")
        capacities, capacity_mode = _client_capacities(
            int(counts.sum()), num_clients, client_capacity_weights
        )
        for retry in range(self.maximum_retries):
            generator = torch_generator(
                seed,
                SEMANTIC_TARGET_POLICY_VERSION,
                "retry",
                retry,
                num_clients,
                counts.tolist(),
                f"{alpha:.17g}",
            )
            concentration = torch.full(
                (counts.numel(), num_clients), alpha, dtype=torch.float64
            )
            gamma = torch._standard_gamma(concentration, generator=generator)
            theta = gamma / gamma.sum(1, keepdim=True).clamp_min(1e-300)
            raw = theta.transpose(0, 1) * counts.double().unsqueeze(0)
            projected = _project_preferences(raw, capacities, counts)
            integer = deterministic_transport_rounding(projected, capacities, counts)
            task_counts = torch.stack(
                [integer[:, torch.tensor(group)].sum(1) for group in groups],
                dim=1,
            )
            if bool((task_counts < min_train_support_per_client_task).any()):
                continue
            payload = {
                "policy_version": SEMANTIC_TARGET_POLICY_VERSION,
                "alpha": float(alpha),
                "seed": int(seed),
                "retry_index": retry,
                "capacity_mode": capacity_mode,
                "train_class_counts": counts.tolist(),
                "client_capacities": capacities.tolist(),
                "task_class_groups": [list(group) for group in groups],
                "raw_preference_matrix": raw.tolist(),
                "projected_real_matrix": projected.tolist(),
                "integer_target_matrix": integer.tolist(),
                "target_task_counts": task_counts.tolist(),
            }
            return SemanticTarget(
                raw_preference_matrix=raw.clone(),
                projected_real_matrix=projected.clone(),
                integer_target_matrix=integer.clone(),
                client_capacities=capacities.clone(),
                train_class_counts=counts.clone(),
                target_task_counts=task_counts.clone(),
                task_class_groups=groups,
                alpha=float(alpha),
                seed=int(seed),
                retry_index=retry,
                capacity_mode=capacity_mode,
                target_hash=_canonical_hash(payload),
            )
        raise SemanticTargetInfeasibleError(
            "No balanced Dirichlet target satisfies train-task support within "
            f"the fixed retry budget ({self.maximum_retries})."
        )

    def generate_from_spec(
        self,
        spec: ScenarioSpec,
        *,
        num_clients: int,
        alpha: float,
        seed: int,
        client_capacity_weights: Sequence[float] | None = None,
        min_train_support_per_client_task: int,
    ) -> SemanticTarget:
        if spec.problem_type != "NC" or spec.labels.ndim != 1:
            raise ValueError("Common train-class targets require single-label NC.")
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
            alpha,
            seed,
            client_capacity_weights,
            groups,
            min_train_support_per_client_task,
        )
