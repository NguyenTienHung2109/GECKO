"""Joint edge-query and endpoint ownership for soft-community LC."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any
from typing import Literal

import torch

from gecko.data.partitioning.base import WeightedLogicalTopology
from gecko.data.partitioning.base import build_weighted_logical_topology
from gecko.types import ScenarioSpec
from gecko.data.partitioning.dirichlet.link import LCEdgeQueryQuota
from gecko.data.partitioning.dirichlet.link import generate_lc_edge_query_quota
from gecko.data.partitioning.community.common import MicroCommunityResult
from gecko.data.partitioning.community.common import SoftMicroCommunityConfig
from gecko.data.partitioning.community.common import balanced_total_capacities
from gecko.data.partitioning.community.common import build_microcommunities
from gecko.data.partitioning.community.common import community_diagnostics
from gecko.data.partitioning.community.common import complete_owner_soft
from gecko.data.partitioning.community.common import micro_result_from_ids
from gecko.data.partitioning.community.common import objective_tuple
from gecko.data.partitioning.community.common import payload_hash
from gecko.data.partitioning.community.common import stable_int
from gecko.data.partitioning.community.common import tensor_hash
from gecko.data.partitioning.community.oracle import brute_force_lc_partition
from gecko.data.partitioning.community.oracle import edge_community_dispersion


LC_SOFT_COMMUNITY_VERSION = "lc_soft_community_exact_edge_dirichlet_v3"
LC_SOFT_COMMUNITY_STREAM_VERSION = "lc_soft_community_selected_query_stream_v3"
LCSoftStatus = Literal[
    "success",
    "quota_infeasible",
    "structurally_infeasible",
    "search_exhausted",
    "heldout_support_rejected",
]


class LCSoftCommunityError(RuntimeError):
    """Fail-closed LC construction error with a scientific status."""

    def __init__(
        self,
        status: LCSoftStatus,
        message: str,
        *,
        certificate: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.certificate = certificate or {}


@dataclass(frozen=True)
class LCSoftCommunityConfig:
    """Configuration for joint soft-community LC construction."""

    num_clients: int
    alpha_dirichlet: float
    seed: int
    supervised_budget_fraction: float = 0.05
    client_size_tolerance: float = 0.20
    minimum_train_queries_per_client_task: int = 1
    minimum_validation_queries_per_client_task: int = 1
    minimum_test_queries_per_client_task: int = 1
    maximum_microcommunity_nodes: int = 4096
    louvain_resolution: float = 1.0
    beam_width: int = 16
    branch_factor: int = 4
    maximum_search_expansions: int = 200_000
    exact_oracle_maximum_nodes: int = 9

    def validate(self) -> None:
        if self.num_clients < 1:
            raise ValueError("num_clients must be positive.")
        if not math.isfinite(self.alpha_dirichlet) or self.alpha_dirichlet <= 0:
            raise ValueError("alpha_dirichlet must be positive and finite.")
        if not 0 < self.supervised_budget_fraction <= 1:
            raise ValueError("supervised_budget_fraction must lie in (0, 1].")
        if not 0 <= self.client_size_tolerance < 1:
            raise ValueError("client_size_tolerance must lie in [0, 1).")
        if min(self.beam_width, self.branch_factor, self.maximum_search_expansions) < 1:
            raise ValueError("LC search budgets must be positive.")
        if self.exact_oracle_maximum_nodes < 1:
            raise ValueError("exact_oracle_maximum_nodes must be positive.")
        if min(
            self.minimum_train_queries_per_client_task,
            self.minimum_validation_queries_per_client_task,
            self.minimum_test_queries_per_client_task,
        ) < 0:
            raise ValueError("Minimum query support cannot be negative.")
        SoftMicroCommunityConfig(
            seed=self.seed,
            maximum_microcommunity_nodes=self.maximum_microcommunity_nodes,
            louvain_resolution=self.louvain_resolution,
        ).validate()


@dataclass(frozen=True)
class LCSoftCommunityResult:
    """Complete LC ownership and exact selected-query allocation."""

    status: LCSoftStatus
    owner: torch.Tensor
    micro_ids: torch.Tensor
    quota: LCEdgeQueryQuota
    selected_query_ids: torch.Tensor
    selected_by_client_group: dict[int, dict[int, torch.Tensor]]
    total_capacities: torch.Tensor
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _BeamState:
    owner: torch.Tensor
    remaining: torch.Tensor
    selected: frozenset[int]
    assignments: tuple[tuple[int, int, int], ...]
    heldout_remaining: torch.Tensor
    heldout_assignments: tuple[tuple[int, int, int, int], ...]
    locked_counts: torch.Tensor


def _training_queries(
    spec: ScenarioSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_ids = torch.cat(
        [spec.query_ids_by_task_split[task]["train"] for task in range(spec.num_tasks)]
    ).detach().cpu().long()
    if train_ids.numel() != torch.unique(train_ids).numel():
        raise ValueError("LC training query IDs must be unique.")
    return (
        train_ids,
        spec.query_endpoints[train_ids].detach().cpu().long(),
        spec.labels[train_ids].detach().cpu().long(),
    )


def _selected_mapping(
    assignments: tuple[tuple[int, int, int], ...],
    *,
    num_clients: int,
    num_groups: int,
) -> dict[int, dict[int, torch.Tensor]]:
    values: dict[int, dict[int, list[int]]] = {
        client: {group: [] for group in range(num_groups)}
        for client in range(num_clients)
    }
    for client, group, query_id in assignments:
        values[client][group].append(query_id)
    return {
        client: {
            group: torch.tensor(sorted(ids), dtype=torch.long)
            for group, ids in groups.items()
        }
        for client, groups in values.items()
    }


def _candidate_rank(
    state: _BeamState,
    *,
    client: int,
    query_id: int,
    endpoints: torch.Tensor,
    micro_ids: torch.Tensor,
    micro_mass: torch.Tensor,
    tie_value: int,
) -> tuple[object, ...]:
    left, right = map(int, endpoints[query_id].tolist())
    micros = {int(micro_ids[left]), int(micro_ids[right])}
    new_micro_owners = 0
    community_mass = 0
    for micro in micros:
        current = micro_mass[micro]
        community_mass += int(current[client])
        new_micro_owners += int(
            bool((current > 0).any()) and int(current[client]) == 0
        )
    new_nodes = int(state.owner[left] < 0) + int(state.owner[right] < 0 and right != left)
    return (
        new_micro_owners,
        -community_mass,
        int(micro_ids[left] != micro_ids[right]),
        new_nodes,
        tie_value,
        query_id,
    )


def _compatible_candidates(
    state: _BeamState,
    *,
    client: int,
    candidates: list[int],
    endpoints: torch.Tensor,
    capacity: int,
) -> list[int]:
    result = []
    for query_id in candidates:
        if query_id in state.selected:
            continue
        left, right = map(int, endpoints[query_id].tolist())
        if int(state.owner[left]) not in {-1, client} or int(state.owner[right]) not in {-1, client}:
            continue
        new_nodes = int(state.owner[left] < 0) + int(
            right != left and state.owner[right] < 0
        )
        if int(state.locked_counts[client]) + new_nodes > capacity:
            continue
        result.append(query_id)
    return result


def _partial_state_score(
    state: _BeamState,
    micro_ids: torch.Tensor,
    topology: WeightedLogicalTopology,
    *,
    num_clients: int,
    seed: int,
) -> tuple[object, ...]:
    assigned = state.owner >= 0
    num_micros = int(micro_ids.max()) + 1
    counts = torch.zeros((num_micros, num_clients), dtype=torch.long)
    if bool(assigned.any()):
        encoded = micro_ids[assigned] * num_clients + state.owner[assigned]
        counts = torch.bincount(
            encoded, minlength=num_micros * num_clients
        ).reshape(num_micros, num_clients)
    sizes = counts.sum(1)
    split_mass = int((sizes - counts.max(1).values).sum())
    owner_excess = int(torch.clamp((counts > 0).sum(1) - 1, min=0).sum())
    edges = topology.edge_index
    both = assigned[edges[0]] & assigned[edges[1]] if edges.numel() else torch.zeros(0, dtype=torch.bool)
    cut = 0.0
    if bool(both.any()):
        crossing = state.owner[edges[0, both]] != state.owner[edges[1, both]]
        cut = float(topology.edge_weights[both][crossing].sum())
    return (
        split_mass,
        owner_excess,
        cut,
        int(state.locked_counts.max() - state.locked_counts.min()),
        stable_int(seed, "state", tuple(sorted(state.selected))),
    )


def _beam_allocate(
    *,
    topology: WeightedLogicalTopology,
    micro_ids: torch.Tensor,
    train_ids: torch.Tensor,
    train_endpoints: torch.Tensor,
    train_groups: torch.Tensor,
    query_endpoints: torch.Tensor,
    heldout_ids: dict[int, dict[str, torch.Tensor]],
    quota: torch.Tensor,
    capacities: torch.Tensor,
    config: LCSoftCommunityConfig,
) -> tuple[_BeamState | None, dict[str, Any]]:
    """Reserve selected-query endpoints with deterministic bounded beam search."""

    # All constraints use query identities/endpoints. Only train_groups came
    # from labels; held-out labels never enter this function.
    all_constraint_ids = [train_ids] + [
        heldout_ids[task][split]
        for task in sorted(heldout_ids)
        for split in ("val", "test")
    ]
    constraint_ids = torch.cat(all_constraint_ids)
    maximum_query_id = int(constraint_ids.max()) if constraint_ids.numel() else -1
    endpoint_by_id = query_endpoints.detach().cpu().long()
    by_group = {
        group: train_ids[train_groups == group].tolist()
        for group in range(quota.shape[1])
    }
    tie_values = torch.empty(
        (config.num_clients, maximum_query_id + 1), dtype=torch.int64
    )
    # Keep only 63 bits so values fit a signed torch integer.
    for client in range(config.num_clients):
        for query_id in constraint_ids.tolist():
            tie_values[client, query_id] = stable_int(
                LC_SOFT_COMMUNITY_VERSION, config.seed, client, query_id
            ) % (2**63 - 1)
    initial = _BeamState(
        owner=torch.full((topology.num_nodes,), -1, dtype=torch.long),
        remaining=quota.clone(),
        selected=frozenset(),
        assignments=(),
        heldout_remaining=torch.tensor(
            [
                [
                    [
                        config.minimum_validation_queries_per_client_task,
                        config.minimum_test_queries_per_client_task,
                    ]
                    for _ in sorted(heldout_ids)
                ]
                for _ in range(config.num_clients)
            ],
            dtype=torch.long,
        ),
        heldout_assignments=(),
        locked_counts=torch.zeros(config.num_clients, dtype=torch.long),
    )
    states = [initial]
    expansions = 0
    pruned = 0
    maximum_frontier = 1
    target = int(quota.sum() + initial.heldout_remaining.sum())
    for _ in range(target):
        next_states: list[_BeamState] = []
        for state in states:
            train_cells = [
                ("train", client, group, -1, by_group[group])
                for client in range(config.num_clients)
                for group in range(quota.shape[1])
                if int(state.remaining[client, group]) > 0
            ]
            heldout_cells = [
                (
                    "heldout",
                    client,
                    task,
                    split_index,
                    heldout_ids[task][("val", "test")[split_index]].tolist(),
                )
                for client in range(config.num_clients)
                for task in sorted(heldout_ids)
                for split_index in range(2)
                if int(state.heldout_remaining[client, task, split_index]) > 0
            ]
            cells = train_cells + heldout_cells
            if not cells:
                next_states.append(state)
                continue
            # Use static supply scarcity to choose one cell, then perform the
            # expensive endpoint-compatibility scan only for that cell. This
            # keeps each expansion linear in one query pool rather than in all
            # client/group/task/split pools.
            def cell_rank(
                cell: tuple[str, int, int, int, list[int]],
            ) -> tuple[object, ...]:
                kind, client, index, split_index, pool = cell
                remaining_count = int(
                    state.remaining[client, index]
                    if kind == "train"
                    else state.heldout_remaining[client, index, split_index]
                )
                return (
                    len(pool) - remaining_count,
                    -remaining_count,
                    kind,
                    index,
                    split_index,
                    client,
                )

            kind, client, index, split_index, pool = min(cells, key=cell_rank)
            compatible = _compatible_candidates(
                state,
                client=client,
                candidates=pool,
                endpoints=endpoint_by_id,
                capacity=int(capacities[client]),
            )
            micro_mass = torch.zeros(
                (int(micro_ids.max()) + 1, config.num_clients), dtype=torch.long
            )
            assigned = state.owner >= 0
            if bool(assigned.any()):
                encoded = (
                    micro_ids[assigned] * config.num_clients + state.owner[assigned]
                )
                micro_mass.view(-1).index_add_(
                    0, encoded, torch.ones(encoded.numel(), dtype=torch.long)
                )
            compatible = heapq.nsmallest(
                config.branch_factor,
                compatible,
                key=lambda query_id: _candidate_rank(
                    state,
                    client=client,
                    query_id=query_id,
                    endpoints=endpoint_by_id,
                    micro_ids=micro_ids,
                    micro_mass=micro_mass,
                    tie_value=int(tie_values[client, query_id]),
                ),
            )
            for query_id in compatible:
                if expansions >= config.maximum_search_expansions:
                    break
                owner = state.owner.clone()
                counts = state.locked_counts.clone()
                left, right = map(int, endpoint_by_id[query_id].tolist())
                for node in {left, right}:
                    if int(owner[node]) < 0:
                        owner[node] = client
                        counts[client] += 1
                remaining = state.remaining.clone()
                heldout_remaining = state.heldout_remaining.clone()
                assignments = state.assignments
                heldout_assignments = state.heldout_assignments
                if kind == "train":
                    remaining[client, index] -= 1
                    assignments = assignments + ((client, index, query_id),)
                else:
                    heldout_remaining[client, index, split_index] -= 1
                    heldout_assignments = heldout_assignments + (
                        (client, index, split_index, query_id),
                    )
                next_states.append(
                    _BeamState(
                        owner=owner,
                        remaining=remaining,
                        selected=state.selected | {query_id},
                        assignments=assignments,
                        heldout_remaining=heldout_remaining,
                        heldout_assignments=heldout_assignments,
                        locked_counts=counts,
                    )
                )
                expansions += 1
            if expansions >= config.maximum_search_expansions:
                break
        if not next_states or expansions >= config.maximum_search_expansions:
            states = next_states
            break
        next_states.sort(
            key=lambda state: _partial_state_score(
                state,
                micro_ids,
                topology,
                num_clients=config.num_clients,
                seed=config.seed,
            )
        )
        pruned += max(0, len(next_states) - config.beam_width)
        states = next_states[: config.beam_width]
        maximum_frontier = max(maximum_frontier, len(states))
    complete = [
        state
        for state in states
        if int(state.remaining.sum() + state.heldout_remaining.sum()) == 0
    ]
    best = min(
        complete,
        key=lambda state: _partial_state_score(
            state,
            micro_ids,
            topology,
            num_clients=config.num_clients,
            seed=config.seed,
        ),
        default=None,
    )
    return best, {
        "beam_width": config.beam_width,
        "branch_factor": config.branch_factor,
        "search_expansions": expansions,
        "search_pruned_states": pruned,
        "maximum_search_frontier": maximum_frontier,
        "maximum_search_expansions": config.maximum_search_expansions,
        "best_remaining_quota": (
            int(
                min(
                    (
                        state.remaining.sum() + state.heldout_remaining.sum()
                        for state in states
                    ),
                    default=quota.sum() + initial.heldout_remaining.sum(),
                )
            )
        ),
    }


def _capacity_matrix(
    owner: torch.Tensor,
    train_endpoints: torch.Tensor,
    train_groups: torch.Tensor,
    *,
    num_clients: int,
    num_groups: int,
) -> torch.Tensor:
    result = torch.zeros((num_clients, num_groups), dtype=torch.long)
    internal = owner[train_endpoints[:, 0]] == owner[train_endpoints[:, 1]]
    if bool(internal.any()):
        clients = owner[train_endpoints[internal, 0]]
        encoded = clients * num_groups + train_groups[internal]
        result.view(-1).index_add_(
            0, encoded, torch.ones(encoded.numel(), dtype=torch.long)
        )
    return result


def _heldout_support_counts(
    owner: torch.Tensor,
    query_endpoints: torch.Tensor,
    heldout_ids: dict[int, dict[str, torch.Tensor]],
    *,
    num_clients: int,
) -> dict[str, list[list[int]]]:
    """Audit structural support from held-out IDs/endpoints without labels."""

    result: dict[str, list[list[int]]] = {"val": [], "test": []}
    for split in ("val", "test"):
        for task in sorted(heldout_ids):
            endpoints = query_endpoints[heldout_ids[task][split]]
            internal = owner[endpoints[:, 0]].clone()
            internal[owner[endpoints[:, 0]] != owner[endpoints[:, 1]]] = -1
            result[split].append(
                torch.bincount(
                    internal[internal >= 0], minlength=num_clients
                ).tolist()
            )
    return result


def build_lc_soft_community_partition(
    spec: ScenarioSpec,
    config: LCSoftCommunityConfig,
    *,
    micro_ids: torch.Tensor | None = None,
    quota: LCEdgeQueryQuota | None = None,
) -> LCSoftCommunityResult:
    """Build a leakage-safe joint edge/endpoint soft-community LC partition."""

    config.validate()
    if spec.problem_type != "LC" or spec.query_endpoints is None or spec.labels.ndim != 1:
        raise ValueError("LC soft-community v3 requires single-label LC.")
    context = spec.context_edge_index if spec.context_edge_index is not None else spec.edge_index
    topology = build_weighted_logical_topology(
        context,
        num_nodes=int(spec.node_features.shape[0]),
        directed=False,
        representation_id="lc_soft_community_symmetrized_context_v3",
    )
    train_ids, train_endpoints, train_groups = _training_queries(spec)
    heldout_ids = {
        task: {
            split: spec.query_ids_by_task_split[task][split].detach().cpu().long()
            for split in ("val", "test")
        }
        for task in range(spec.num_tasks)
    }
    all_query_ids = torch.cat(
        [train_ids]
        + [
            heldout_ids[task][split]
            for task in range(spec.num_tasks)
            for split in ("val", "test")
        ]
    )
    if all_query_ids.numel() != torch.unique(all_query_ids).numel():
        raise ValueError("LC train/val/test query IDs must be globally disjoint.")
    constrained_endpoints = spec.query_endpoints[all_query_ids]
    if constrained_endpoints.numel() and (
        int(constrained_endpoints.min()) < 0
        or int(constrained_endpoints.max()) >= topology.num_nodes
    ):
        raise ValueError("An LC constrained query endpoint is outside the topology.")
    for task in range(spec.num_tasks):
        for split, minimum in (
            ("val", config.minimum_validation_queries_per_client_task),
            ("test", config.minimum_test_queries_per_client_task),
        ):
            available = int(heldout_ids[task][split].numel())
            if available < config.num_clients * minimum:
                raise LCSoftCommunityError(
                    "structurally_infeasible",
                    f"LC held-out support infeasible: task={task} split={split} "
                    f"has {available} queries but requires at least "
                    f"{config.num_clients * minimum} distinct queries.",
                    certificate={
                        "stage": "heldout_supply",
                        "task": task,
                        "split": split,
                        "available": available,
                        "required": config.num_clients * minimum,
                    },
                )
    if train_groups.numel() and (
        int(train_groups.min()) < 0 or int(train_groups.max()) >= spec.num_classes
    ):
        raise ValueError("An LC training label is outside the semantic universe.")
    if spec.incremental_type == "domain":
        train_quota_groups = spec.query_task_ids[train_ids].detach().cpu().long()
        num_quota_groups = spec.num_tasks
        task_groups = tuple((task,) for task in range(spec.num_tasks))
        quota_semantic_unit = "immutable_domain_task_id"
        if bool((train_quota_groups < 0).any()):
            raise ValueError("An LC domain training query has no immutable task ID.")
    else:
        train_quota_groups = train_groups
        num_quota_groups = spec.num_classes
        task_groups = tuple(
            tuple(sorted(int(value) for value in spec.task_class_sets[task].tolist()))
            for task in range(spec.num_tasks)
        )
        quota_semantic_unit = "edge_class"
    if quota is None:
        counts = torch.bincount(train_quota_groups, minlength=num_quota_groups)
        try:
            quota = generate_lc_edge_query_quota(
                counts.tolist(),
                num_clients=config.num_clients,
                alpha_dirichlet=config.alpha_dirichlet,
                supervised_budget_fraction=config.supervised_budget_fraction,
                seed=config.seed,
                task_class_groups=task_groups,
                minimum_train_queries_per_client_task=(
                    config.minimum_train_queries_per_client_task
                ),
                semantic_unit=quota_semantic_unit,
            )
        except ValueError as error:
            raise LCSoftCommunityError(
                "quota_infeasible", str(error), certificate={"stage": "quota"}
            ) from error
    if quota.integer_quota.shape != (config.num_clients, num_quota_groups):
        raise ValueError(
            "LC quota shape does not match clients and semantic groups."
        )
    if bool((quota.integer_quota < 0).any()):
        raise ValueError("LC quota cannot contain negative query counts.")
    supply = torch.bincount(train_quota_groups, minlength=num_quota_groups)
    if bool((quota.integer_quota.sum(0) > supply).any()):
        raise ValueError("LC quota requests more queries than a semantic group supplies.")
    micro: MicroCommunityResult = (
        build_microcommunities(
            topology,
            SoftMicroCommunityConfig(
                seed=config.seed,
                maximum_microcommunity_nodes=config.maximum_microcommunity_nodes,
                louvain_resolution=config.louvain_resolution,
            ),
        )
        if micro_ids is None
        else micro_result_from_ids(micro_ids, num_nodes=topology.num_nodes)
    )
    capacities = balanced_total_capacities(
        topology.num_nodes,
        torch.zeros(config.num_clients, dtype=torch.long),
        client_size_tolerance=config.client_size_tolerance,
    )
    state, search = _beam_allocate(
        topology=topology,
        micro_ids=micro.micro_ids,
        train_ids=train_ids,
        train_endpoints=train_endpoints,
        train_groups=train_quota_groups,
        query_endpoints=spec.query_endpoints,
        heldout_ids=heldout_ids,
        quota=quota.integer_quota,
        capacities=capacities,
        config=config,
    )
    if state is None:
        certificate = {
            "stage": "joint_edge_endpoint_search",
            "quota_hash": quota.quota_hash,
            "microcommunity_hash": micro.diagnostics["microcommunity_hash"],
            **search,
        }
        if topology.num_nodes <= config.exact_oracle_maximum_nodes:
            ideal = topology.num_nodes / config.num_clients
            lower = math.floor((1.0 - config.client_size_tolerance) * ideal)
            upper = math.ceil((1.0 + config.client_size_tolerance) * ideal)
            oracle = brute_force_lc_partition(
                topology=topology,
                micro_ids=micro.micro_ids,
                query_ids=train_ids,
                query_endpoints=spec.query_endpoints,
                query_groups=train_quota_groups,
                quota=quota.integer_quota,
                heldout_query_ids_by_task_split=heldout_ids,
                minimum_validation_queries_per_client_task=(
                    config.minimum_validation_queries_per_client_task
                ),
                minimum_test_queries_per_client_task=(
                    config.minimum_test_queries_per_client_task
                ),
                capacity_lower=lower,
                capacity_upper=upper,
                maximum_nodes=config.exact_oracle_maximum_nodes,
            )
            status: LCSoftStatus = (
                "search_exhausted" if oracle.feasible else "structurally_infeasible"
            )
            certificate["tiny_oracle_feasible"] = oracle.feasible
        else:
            status = "search_exhausted"
        raise LCSoftCommunityError(
            status,
            "LC soft-community joint allocation did not produce an exact feasible state.",
            certificate=certificate,
        )
    owner = complete_owner_soft(
        state.owner,
        micro.micro_ids,
        topology,
        capacities,
        seed=config.seed,
    )
    selected = _selected_mapping(
        state.assignments,
        num_clients=config.num_clients,
        num_groups=num_quota_groups,
    )
    selected_ids = torch.tensor(
        sorted(query_id for _, _, query_id in state.assignments), dtype=torch.long
    )
    selected_counts = torch.tensor(
        [
            [selected[client][group].numel() for group in range(num_quota_groups)]
            for client in range(config.num_clients)
        ],
        dtype=torch.long,
    )
    if not torch.equal(selected_counts, quota.integer_quota):
        raise AssertionError("LC joint allocator changed the exact selected-query quota.")
    strict_local = all(
        int(owner[spec.query_endpoints[query_id, 0]]) == client
        and int(owner[spec.query_endpoints[query_id, 1]]) == client
        for client in selected
        for group in selected[client]
        for query_id in selected[client][group].tolist()
    )
    if not strict_local or selected_ids.numel() != torch.unique(selected_ids).numel():
        raise AssertionError("LC selected queries are not unique and strict-local.")
    heldout_support = _heldout_support_counts(
        owner,
        spec.query_endpoints,
        heldout_ids,
        num_clients=config.num_clients,
    )
    for split, minimum in (
        ("val", config.minimum_validation_queries_per_client_task),
        ("test", config.minimum_test_queries_per_client_task),
    ):
        if any(count < minimum for row in heldout_support[split] for count in row):
            raise AssertionError("LC completion violated locked held-out support.")
    structure = community_diagnostics(
        owner,
        micro.micro_ids,
        topology,
        num_clients=config.num_clients,
    )
    dispersion = edge_community_dispersion(
        selected,
        spec.query_endpoints,
        micro.micro_ids,
        num_clients=config.num_clients,
    )
    capacity = _capacity_matrix(
        owner,
        train_endpoints,
        train_quota_groups,
        num_clients=config.num_clients,
        num_groups=num_quota_groups,
    )
    selected_hash = tensor_hash(selected_ids)
    heldout_flat = torch.cat(
        [
            heldout_ids[task][split]
            for task in range(spec.num_tasks)
            for split in ("val", "test")
        ]
    )
    heldout_identity_hash = payload_hash(
        {
            str(task): {
                split: heldout_ids[task][split].tolist()
                for split in ("val", "test")
            }
            for task in range(spec.num_tasks)
        }
    )
    heldout_endpoint_hash = tensor_hash(spec.query_endpoints[heldout_flat])
    audit_payload = {
        "version": LC_SOFT_COMMUNITY_VERSION,
        "topology_hash": topology.topology_hash,
        "microcommunity_hash": micro.diagnostics["microcommunity_hash"],
        "quota_hash": quota.quota_hash,
        "owner_hash": tensor_hash(owner),
        "selected_query_hash": selected_hash,
        "heldout_query_identity_hash": heldout_identity_hash,
        "heldout_endpoint_hash": heldout_endpoint_hash,
    }
    diagnostics = {
        "policy_version": LC_SOFT_COMMUNITY_VERSION,
        "stream_version": LC_SOFT_COMMUNITY_STREAM_VERSION,
        "status": "success",
        "allocation_mode": "joint_edge_endpoint_soft_community",
        "community_generation_role": "topology_only_soft_cohesion_prior",
        "label_access_policy": "train_query_labels_only_heldout_ids_endpoints_no_heldout_labels",
        "heldout_constraint_role": "immutable_query_identity_and_endpoints_only",
        "minimum_validation_queries_per_client_task": config.minimum_validation_queries_per_client_task,
        "minimum_test_queries_per_client_task": config.minimum_test_queries_per_client_task,
        "heldout_support_counts": heldout_support,
        "heldout_anchor_count": len(state.heldout_assignments),
        "heldout_query_identity_hash": heldout_identity_hash,
        "heldout_endpoint_hash": heldout_endpoint_hash,
        "topology_hash": topology.topology_hash,
        "quota_hash": quota.quota_hash,
        "ownership_hash": tensor_hash(owner),
        "owner_hash": tensor_hash(owner),
        "selected_query_hash": selected_hash,
        "selected_query_count": int(selected_ids.numel()),
        "selected_edge_community_dispersion": dispersion,
        "dirichlet_semantic_unit": quota_semantic_unit,
        "dirichlet_semantic_group_count": num_quota_groups,
        **quota.manipulation_diagnostics(),
        "capacity_after_assignment": capacity.tolist(),
        "capacity_slack": (capacity - quota.integer_quota).tolist(),
        "exact_quota_residual": int(torch.abs(selected_counts - quota.integer_quota).sum()),
        "strict_local_selected_query_check": strict_local,
        "total_capacities": capacities.tolist(),
        "soft_objective": [dispersion, *objective_tuple(structure)],
        "audit_hash": payload_hash(audit_payload),
        **micro.diagnostics,
        **search,
        **structure,
    }
    return LCSoftCommunityResult(
        status="success",
        owner=owner,
        micro_ids=micro.micro_ids,
        quota=quota,
        selected_query_ids=selected_ids,
        selected_by_client_group=selected,
        total_capacities=capacities,
        diagnostics=diagnostics,
    )
