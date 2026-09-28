"""Balanced bounded blockwise client task orders."""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple

import torch

from gecko.reproducibility import python_random
from gecko.types import OrderPlan


def _block_size(profile: str, num_tasks: int) -> int:
    return {
        "synchronized": 1,
        "binary_mismatch": num_tasks,
        "mild": min(2, num_tasks),
        "hard": min(4, num_tasks),
        "unconstrained": num_tasks,
    }[profile]


def _permutations(block: Sequence[int], limit: int, seed: int) -> List[Tuple[int, ...]]:
    if len(block) <= 7:
        values = list(itertools.permutations(block))
    else:
        rng = python_random(seed, "large-block")
        values = []
        seen = set()
        while len(values) < limit:
            candidate = list(block)
            rng.shuffle(candidate)
            value = tuple(candidate)
            if value not in seen:
                seen.add(value)
                values.append(value)
    rng = python_random(seed, "permutation-order")
    rng.shuffle(values)
    return values[: max(1, limit)]


def _kendall_distance(first: Sequence[int], second: Sequence[int]) -> float:
    if len(first) < 2:
        return 0.0
    position = {value: index for index, value in enumerate(second)}
    inversions = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            inversions += position[first[left]] > position[first[right]]
    return inversions / (len(first) * (len(first) - 1) / 2)


def generate_client_orders(
    num_clients: int,
    num_tasks: int,
    profile: str,
    seed: int,
) -> OrderPlan:
    """Generate deterministic cohort-balanced blockwise permutations."""

    if profile == "binary_mismatch" and num_tasks != 2:
        raise ValueError("binary_mismatch requires exactly two tasks.")
    canonical = tuple(range(num_tasks))
    block_size = _block_size(profile, num_tasks)
    blocks = [canonical[start : start + block_size] for start in range(0, num_tasks, block_size)]
    block_permutations = [
        _permutations(block, max(num_clients, 1), seed + block_index)
        for block_index, block in enumerate(blocks)
    ]
    client_orders: Dict[int, Tuple[int, ...]] = {}
    cohort_assignments: Dict[int, int] = {}
    for client in range(num_clients):
        order: list[int] = []
        for block_index, permutations in enumerate(block_permutations):
            permutation_index = (client + block_index) % len(permutations)
            order.extend(permutations[permutation_index])
        client_orders[client] = tuple(order)
        cohort_assignments[client] = client % max(1, len(block_permutations[0]))
    inverse = {
        client: {task: stage for stage, task in enumerate(order)}
        for client, order in client_orders.items()
    }
    pairwise = [
        _kendall_distance(client_orders[left], client_orders[right])
        for left in range(num_clients)
        for right in range(left + 1, num_clients)
    ]
    reference = [_kendall_distance(order, canonical) for order in client_orders.values()]
    active_counts: Dict[int, Dict[int, int]] = {}
    active_entropy: Dict[int, float] = {}
    exposure: Dict[int, int] = {}
    exposed = set()
    for stage in range(num_tasks):
        counts = Counter(order[stage] for order in client_orders.values())
        active_counts[stage] = dict(sorted(counts.items()))
        probabilities = [count / num_clients for count in counts.values()]
        active_entropy[stage] = -sum(p * math.log(p) for p in probabilities if p > 0)
        exposed.update(counts)
        exposure[stage] = len(exposed)
    maximum_displacement = max(
        abs(stage - task)
        for order in client_orders.values()
        for stage, task in enumerate(order)
    )
    diagnostics = {
        "core_leaderboard": profile != "unconstrained",
        "normalized_pairwise_kendall_distance": sum(pairwise) / max(1, len(pairwise)),
        "normalized_distance_from_reference": sum(reference) / max(1, len(reference)),
        "maximum_task_displacement": maximum_displacement,
        "active_task_counts_by_stage": active_counts,
        "active_task_entropy_by_stage": active_entropy,
        "global_exposure_curve": exposure,
    }
    return OrderPlan(
        canonical_order=canonical,
        client_orders=client_orders,
        inverse_client_orders=inverse,
        cohort_assignments=cohort_assignments,
        block_size=block_size,
        seed=seed,
        diagnostics=diagnostics,
    )
