"""Deterministic primitives for the ``order_support_v1`` diagnostic campaign.

The functions in this module are deliberately independent from the training
driver.  They make the campaign's schedule, donor intervention algebra, and
summary estimands small enough to exercise with numerical unit tests.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Sequence

import numpy as np
import torch

from gecko.types import OrderPlan
from gecko.types import ParticipationPlan


CAMPAIGN_ID = "order_support_v1"
NUM_CLIENTS = 10
NUM_TASKS = 8
ROUNDS_PER_STAGE = 50
PARTICIPANTS_PER_ROUND = 5
PROBE_ROUNDS_1BASED = (10, 25, 40)
DONOR_SCALES = (0.0, 0.25, 0.5, 1.0)
ZERO_NORM_THRESHOLD = 1e-12


def canonical_json_hash(value: Any) -> str:
    """Hash a JSON-compatible value with the campaign canonical encoding."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_seed(*parts: object) -> int:
    """Resolve an isolated 63-bit seed from explicit metadata."""

    digest = hashlib.sha256(
        json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def generate_balanced_participation(seed_index: int) -> ParticipationPlan:
    """Create the pre-registered 25-per-client schedule for all eight stages."""

    trace: dict[int, dict[int, tuple[int, ...]]] = {}
    seed = deterministic_seed(CAMPAIGN_ID, "balanced-participation", seed_index)
    for stage in range(NUM_TASKS):
        rng = random.Random(deterministic_seed(seed, stage))
        rows: list[tuple[int, ...]] = []
        for _ in range(25):
            permutation = list(range(NUM_CLIENTS))
            rng.shuffle(permutation)
            rows.append(tuple(sorted(permutation[:PARTICIPANTS_PER_ROUND])))
            rows.append(tuple(sorted(permutation[PARTICIPANTS_PER_ROUND:])))
        rng.shuffle(rows)
        trace[stage] = {round_index: row for round_index, row in enumerate(rows)}
    plan = ParticipationPlan(
        trace=trace,
        fraction=PARTICIPANTS_PER_ROUND / NUM_CLIENTS,
        rounds_per_stage=ROUNDS_PER_STAGE,
        seed=seed,
    )
    validate_balanced_participation(plan)
    return plan


def participation_payload(plan: ParticipationPlan) -> dict[str, object]:
    """Return the stable, serializable participation record used for hashing."""

    return {
        "profile": "balanced_diagnostic_v1",
        "seed": int(plan.seed),
        "fraction": float(plan.fraction),
        "rounds_per_stage": int(plan.rounds_per_stage),
        "trace": {
            str(stage): {
                str(round_index): list(clients)
                for round_index, clients in sorted(rounds.items())
            }
            for stage, rounds in sorted(plan.trace.items())
        },
    }


def validate_balanced_participation(plan: ParticipationPlan) -> None:
    """Fail closed unless the complete 8x50 balanced contract is satisfied."""

    if set(plan.trace) != set(range(NUM_TASKS)):
        raise ValueError("Balanced trace must contain exactly eight stages.")
    if plan.rounds_per_stage != ROUNDS_PER_STAGE:
        raise ValueError("Balanced trace must contain 50 rounds per stage.")
    for stage, rounds in sorted(plan.trace.items()):
        if set(rounds) != set(range(ROUNDS_PER_STAGE)):
            raise ValueError(f"Stage {stage} does not contain rounds 0..49.")
        counts = [0] * NUM_CLIENTS
        for participants in rounds.values():
            if len(participants) != PARTICIPANTS_PER_ROUND:
                raise ValueError("Every balanced round must contain five clients.")
            if len(set(participants)) != PARTICIPANTS_PER_ROUND:
                raise ValueError("A balanced round contains a duplicate client.")
            if any(client < 0 or client >= NUM_CLIENTS for client in participants):
                raise ValueError("A balanced round contains an invalid client ID.")
            for client in participants:
                counts[client] += 1
        if counts != [25] * NUM_CLIENTS:
            raise ValueError(
                f"Stage {stage} participation counts are {counts}, expected 25 each."
            )


def validate_block_orders(orders: OrderPlan) -> None:
    """Check one appearance per task and prohibit cross-block permutations."""

    if orders.canonical_order != tuple(range(NUM_TASKS)):
        raise ValueError("Campaign requires immutable global task IDs 0..7.")
    for client, order in sorted(orders.client_orders.items()):
        if sorted(order) != list(range(NUM_TASKS)):
            raise ValueError(f"Client {client} does not encounter every task once.")
        if set(order[:4]) != set(range(4)) or set(order[4:]) != set(range(4, 8)):
            raise ValueError(f"Client {client} crosses the pre-registered task blocks.")
        for task, stage in orders.inverse_client_orders[client].items():
            if order[stage] != task:
                raise ValueError("Order and inverse order disagree.")


def tau_1based(orders: OrderPlan, client: int, task: int) -> int:
    """Return the one-based first-exposure stage for a client-task pair."""

    return int(orders.inverse_client_orders[client][task]) + 1


@dataclass(frozen=True)
class ProbeSelection:
    """Metadata-only selection outcome for one fixed pre-round probe slot."""

    status: str
    reason: str | None
    eligible_tasks: tuple[int, ...]
    chosen_task: int | None
    eligible_receivers: tuple[int, ...]
    chosen_receivers: tuple[int, ...]
    task_selection_probability: float | None
    receiver_inclusion_probability: float | None


def select_probe_case(
    *,
    alpha: float,
    seed_index: int,
    stage_1based: int,
    round_1based: int,
    participant_tasks: Mapping[int, int],
    orders: OrderPlan,
    valid_validation_pairs: set[tuple[int, int]],
    max_receivers: int = 2,
) -> ProbeSelection:
    """Select one target and at most two old receivers without outcome access."""

    eligible_by_task: dict[int, tuple[int, ...]] = {}
    for task in sorted(set(int(value) for value in participant_tasks.values())):
        receivers = tuple(
            client
            for client in sorted(orders.client_orders)
            if tau_1based(orders, client, task) < stage_1based
            and (client, task) in valid_validation_pairs
        )
        if receivers:
            eligible_by_task[task] = receivers
    eligible_tasks = tuple(sorted(eligible_by_task))
    if not eligible_tasks:
        return ProbeSelection(
            status="structural_missing",
            reason="no_active_task_with_old_validation_receiver",
            eligible_tasks=(),
            chosen_task=None,
            eligible_receivers=(),
            chosen_receivers=(),
            task_selection_probability=None,
            receiver_inclusion_probability=None,
        )
    rng = random.Random(
        deterministic_seed(
            CAMPAIGN_ID,
            "probe-selection",
            float(alpha),
            seed_index,
            stage_1based,
            round_1based,
        )
    )
    task = eligible_tasks[rng.randrange(len(eligible_tasks))]
    receivers = list(eligible_by_task[task])
    rng.shuffle(receivers)
    chosen = tuple(sorted(receivers[: min(max_receivers, len(receivers))]))
    return ProbeSelection(
        status="eligible",
        reason=None,
        eligible_tasks=eligible_tasks,
        chosen_task=task,
        eligible_receivers=tuple(sorted(eligible_by_task[task])),
        chosen_receivers=chosen,
        task_selection_probability=1.0 / len(eligible_tasks),
        receiver_inclusion_probability=min(1.0, max_receivers / len(receivers)),
    )


def old_support_metrics(
    *,
    stage_1based: int,
    task_mass: Mapping[int, float],
    orders: OrderPlan,
    valid_pairs: set[tuple[int, int]],
) -> dict[str, object]:
    """Compute schedule exposure over pairs satisfying strict ``tau < stage``."""

    old_pairs = [
        (client, task)
        for client, task in sorted(valid_pairs)
        if tau_1based(orders, client, task) < stage_1based
    ]
    if not old_pairs:
        return {
            "old_support_coverage": None,
            "old_support_mass": None,
            "old_support_numerator": None,
            "old_support_denominator": 0,
            "old_support_reason": "no_old_pairs_at_stage_1",
        }
    supported = sum(float(task_mass.get(task, 0.0)) > 0.0 for _, task in old_pairs)
    mass = sum(float(task_mass.get(task, 0.0)) for _, task in old_pairs)
    return {
        "old_support_coverage": supported / len(old_pairs),
        "old_support_mass": mass / len(old_pairs),
        "old_support_numerator": supported,
        "old_support_denominator": len(old_pairs),
        "old_support_reason": None,
    }


TensorState = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class RNGSnapshot:
    """All process RNG states that can affect one local learner update."""

    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...] | None


def capture_rng_state() -> RNGSnapshot:
    """Capture Python, NumPy, CPU Torch, and every visible CUDA RNG stream."""

    return RNGSnapshot(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=(
            tuple(value.clone() for value in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else None
        ),
    )


def restore_rng_state(snapshot: RNGSnapshot) -> None:
    """Restore an RNG snapshot without consuming a random value."""

    random.setstate(snapshot.python)
    np.random.set_state(snapshot.numpy)
    torch.set_rng_state(snapshot.torch_cpu)
    if snapshot.torch_cuda is not None:
        torch.cuda.set_rng_state_all(list(snapshot.torch_cuda))


@contextmanager
def preserved_rng_state(snapshot: RNGSnapshot | None = None) -> Iterator[None]:
    """Run observer work without changing the learner's RNG trajectory."""

    live = capture_rng_state()
    if snapshot is not None:
        restore_rng_state(snapshot)
    try:
        yield
    finally:
        restore_rng_state(live)


def tensor_state_hash(state: TensorState) -> str:
    """Hash tensor names, dtypes, shapes, and exact CPU bytes."""

    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_delta(after: TensorState, before: TensorState) -> dict[str, torch.Tensor]:
    """Return ``after - before`` while enforcing exact parameter identity."""

    if set(after) != set(before):
        raise ValueError("Tensor-state keys do not align.")
    output: dict[str, torch.Tensor] = {}
    for key in sorted(before):
        if after[key].shape != before[key].shape:
            raise ValueError(f"Tensor-state shape differs for {key}.")
        output[key] = after[key].detach().clone() - before[key].detach().clone()
    return output


def weighted_state_sum(
    states: Sequence[tuple[float, TensorState]],
    *,
    reference: TensorState,
) -> dict[str, torch.Tensor]:
    """Sum weighted displacements without normalizing or changing FP32 scale."""

    keys = set(reference)
    output = {
        key: torch.zeros_like(reference[key], dtype=torch.float64) for key in sorted(keys)
    }
    for weight, state in states:
        if set(state) != keys or not math.isfinite(float(weight)):
            raise ValueError("A weighted state has invalid keys or weight.")
        for key in sorted(keys):
            output[key].add_(state[key].to(dtype=torch.float64), alpha=float(weight))
    return {key: value.to(reference[key].dtype) for key, value in output.items()}


def add_state(base: TensorState, displacement: TensorState) -> dict[str, torch.Tensor]:
    """Add one displacement to a state with FP64 accumulation and base dtype output."""

    if set(base) != set(displacement):
        raise ValueError("Base and displacement keys do not align.")
    return {
        key: (
            base[key].to(dtype=torch.float64)
            + displacement[key].to(dtype=torch.float64)
        ).to(base[key].dtype)
        for key in sorted(base)
    }


def state_l2(state: TensorState) -> float:
    """Compute a global displacement norm in FP64."""

    squared = sum(
        float(value.detach().to(device="cpu", dtype=torch.float64).square().sum())
        for value in state.values()
    )
    return math.sqrt(squared)


def scale_state(state: TensorState, scale: float) -> dict[str, torch.Tensor]:
    """Scale a displacement exactly once; used only for offline branches."""

    if not math.isfinite(scale):
        raise ValueError("State scale must be finite.")
    return {key: value.detach().clone() * scale for key, value in state.items()}


def norm_match_state(
    alternative: TensorState,
    target: TensorState,
    *,
    threshold: float = ZERO_NORM_THRESHOLD,
) -> tuple[dict[str, torch.Tensor] | None, float | None, str]:
    """Match only the aggregate donor norm, with pre-registered zero handling."""

    target_norm = state_l2(target)
    alternative_norm = state_l2(alternative)
    if not math.isfinite(target_norm) or not math.isfinite(alternative_norm):
        return None, None, "nonfinite_norm"
    if target_norm <= threshold:
        return None, None, "zero_donor_norm"
    if alternative_norm <= threshold:
        return None, None, "zero_alternative_norm"
    scale = target_norm / alternative_norm
    return scale_state(alternative, scale), scale, "valid"


def recompose_fedavg(
    *,
    theta_pre: TensorState,
    client_deltas: Mapping[int, TensorState],
    normalized_weights: Mapping[int, float],
    donor_ids: Iterable[int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build full, drop, and donor contributions without donor renormalization."""

    if set(client_deltas) != set(normalized_weights):
        raise ValueError("Client delta and aggregation-weight identities differ.")
    donors = set(int(value) for value in donor_ids)
    if not donors or not donors.issubset(client_deltas):
        raise ValueError("Donor IDs must be a nonempty subset of active clients.")
    donor_delta = weighted_state_sum(
        [(normalized_weights[j], client_deltas[j]) for j in sorted(donors)],
        reference=theta_pre,
    )
    rest_delta = weighted_state_sum(
        [
            (normalized_weights[j], client_deltas[j])
            for j in sorted(client_deltas)
            if j not in donors
        ],
        reference=theta_pre,
    )
    theta_drop = add_state(theta_pre, rest_delta)
    theta_full = add_state(theta_drop, donor_delta)
    return theta_full, theta_drop, donor_delta


def summarize_official(
    records: Sequence[Mapping[str, Any]], orders: OrderPlan
) -> dict[str, float | int]:
    """Compute P, AF, and acquisition from fraction-valued official test cells."""

    by_pair: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    for record in records:
        if record.get("split") != "test" or record.get("mask") != "official":
            continue
        value = record.get("accuracy")
        if value is not None and math.isfinite(float(value)):
            by_pair[(int(record["client_id"]), int(record["task_id"]))][
                int(record["stage"])
            ] = float(value)
    expected = {
        (client, task)
        for client in orders.client_orders
        for task in range(NUM_TASKS)
    }
    if set(by_pair) != expected:
        missing = sorted(expected - set(by_pair))
        raise ValueError(f"Official metrics are missing client-task pairs: {missing}.")
    final_values: list[float] = []
    forgetting: list[float] = []
    acquisition: list[float] = []
    last_task_forgetting: list[float] = []
    for client, task in sorted(expected):
        tau = tau_1based(orders, client, task)
        history = by_pair[(client, task)]
        required = set(range(tau, NUM_TASKS + 1))
        if not required.issubset(history):
            raise ValueError("Official history is incomplete after first exposure.")
        final = history[NUM_TASKS]
        term = max(history[stage] for stage in sorted(required)) - final
        final_values.append(final)
        forgetting.append(term)
        acquisition.append(history[tau])
        if tau == NUM_TASKS:
            last_task_forgetting.append(term)
    if any(abs(value) > 1e-12 for value in last_task_forgetting):
        raise ValueError("Last-stage task forgetting must be exactly zero.")
    return {
        "P_off_pp": 100.0 * float(np.mean(final_values)),
        "AF_off_pp": 100.0 * float(np.mean(forgetting)),
        "Acq_off_pp": 100.0 * float(np.mean(acquisition)),
        "pair_count": len(expected),
    }


def summarize_fixed40(
    records: Sequence[Mapping[str, Any]], orders: OrderPlan
) -> dict[str, float | int]:
    """Compute fixed-head acquisition, block retirement, and age profiles."""

    cells: dict[tuple[int, int, int], float] = {}
    for record in records:
        if record.get("split") == "test" and record.get("mask") == "fixed40":
            value = record.get("accuracy")
            if value is not None and math.isfinite(float(value)):
                cells[(int(record["client_id"]), int(record["stage"]), int(record["task_id"]))] = float(value)
    expected = NUM_CLIENTS * (NUM_TASKS + 1) * NUM_TASKS
    if len(cells) != expected:
        raise ValueError(f"Fixed-40 test grid has {len(cells)} cells, expected {expected}.")
    acquisition = [
        cells[(client, tau_1based(orders, client, task), task)]
        for client in range(NUM_CLIENTS)
        for task in range(NUM_TASKS)
    ]
    block_stage_4 = [
        cells[(client, 4, task)]
        for client in range(NUM_CLIENTS)
        for task in range(4)
    ]
    block_stage_8 = [
        cells[(client, 8, task)]
        for client in range(NUM_CLIENTS)
        for task in range(4)
    ]
    output: dict[str, float | int] = {
        "Acq40_pp": 100.0 * float(np.mean(acquisition)),
        "B40_stage4_pp": 100.0 * float(np.mean(block_stage_4)),
        "B40_stage8_pp": 100.0 * float(np.mean(block_stage_8)),
        "Retired_block_drop_pp": 100.0
        * (float(np.mean(block_stage_4)) - float(np.mean(block_stage_8))),
        "pair_count": NUM_CLIENTS * NUM_TASKS,
    }
    for age in range(5):
        absolute: list[float] = []
        drops: list[float] = []
        for client in range(NUM_CLIENTS):
            for task in range(4):
                tau = tau_1based(orders, client, task)
                start = cells[(client, tau, task)]
                current = cells[(client, tau + age, task)]
                absolute.append(current)
                drops.append(start - current)
        output[f"R40_age{age}_pp"] = 100.0 * float(np.mean(absolute))
        output[f"Age_drop40_age{age}_pp"] = 100.0 * float(np.mean(drops))
    return output


def sample_mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Return mean and sample SD; require the pre-registered three seeds."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("A completed three-seed summary requires three finite values.")
    return float(array.mean()), float(array.std(ddof=1))


def multioutput_gram_inner(
    left: torch.Tensor, feature_gram: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """Return ``tr(left.T @ H @ right)`` without a Kronecker Hessian."""

    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("Multi-output directions must have identical [d,c] shape.")
    if feature_gram.shape != (left.shape[0], left.shape[0]):
        raise ValueError("Feature Gram must have shape [d,d], not [d*c,d*c].")
    return torch.sum(left * (feature_gram @ right))


def enumerate_assignment_indices(tasks: int = 4, clients: int = 5) -> np.ndarray:
    """Enumerate the complete independent assignment population, coincidences included."""

    return np.asarray(list(itertools.product(range(tasks), repeat=clients)), dtype=np.int64)
