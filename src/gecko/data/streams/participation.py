"""Method-independent deterministic participation traces."""

from __future__ import annotations

import math
from typing import Dict
from typing import Tuple

from gecko.reproducibility import python_random
from gecko.types import ParticipationPlan


def generate_participation(
    *,
    num_clients: int,
    num_stages: int,
    rounds_per_stage: int,
    fraction: float,
    seed: int,
) -> ParticipationPlan:
    participants_per_round = max(1, min(num_clients, int(round(num_clients * fraction))))
    trace: Dict[int, Dict[int, Tuple[int, ...]]] = {}
    for stage in range(num_stages):
        rng = python_random(seed, "participation", stage)
        coverage_order = list(range(num_clients))
        rng.shuffle(coverage_order)
        cursor = 0
        trace[stage] = {}
        for round_id in range(rounds_per_stage):
            selected: list[int] = []
            while cursor < num_clients and len(selected) < participants_per_round:
                selected.append(coverage_order[cursor])
                cursor += 1
            remaining = [client for client in range(num_clients) if client not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[: participants_per_round - len(selected)])
            trace[stage][round_id] = tuple(sorted(selected))
        if rounds_per_stage * participants_per_round >= num_clients:
            covered = {client for participants in trace[stage].values() for client in participants}
            if len(covered) != num_clients:
                raise RuntimeError(f"Participation coverage failed at stage {stage}.")
    realized = participants_per_round / num_clients
    return ParticipationPlan(trace, realized, rounds_per_stage, seed)
