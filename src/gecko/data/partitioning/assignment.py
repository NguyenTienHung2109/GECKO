"""Atomic micro-community to client assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from gecko.reproducibility import torch_generator
from gecko.types import ScenarioSpec
from gecko.validation import PartitionInfeasibleError


@dataclass(frozen=True)
class AssignmentOutcome:
    node_owner: torch.Tensor
    support: torch.Tensor
    score: float
    failures: Tuple[str, ...]
    iterations: int
    support_anchor_community_count: int = 0
    semantic_divergence: float | None = None


def _query_owners(spec: ScenarioSpec, node_owner: torch.Tensor) -> torch.Tensor:
    if spec.problem_type == "NC":
        return node_owner
    assert spec.query_endpoints is not None
    source_owner = node_owner[spec.query_endpoints[:, 0]]
    target_owner = node_owner[spec.query_endpoints[:, 1]]
    return torch.where(source_owner == target_owner, source_owner, -torch.ones_like(source_owner))


def _support_tensor(
    spec: ScenarioSpec,
    node_owner: torch.Tensor,
    num_clients: int,
) -> torch.Tensor:
    support = torch.zeros(num_clients, spec.num_tasks, 3, dtype=torch.long)
    query_owner = _query_owners(spec, node_owner)
    split_index = {"train": 0, "val": 1, "test": 2}
    for task_id, splits in spec.query_ids_by_task_split.items():
        for split, query_ids in splits.items():
            if spec.problem_type == "LP":
                query_ids = query_ids[spec.labels[query_ids] == 1]
            owners = query_owner[query_ids]
            internal = owners >= 0
            if internal.any():
                support[:, task_id, split_index[split]] = torch.bincount(
                    owners[internal], minlength=num_clients
                )
    return support


def _compressed_query_support(
    spec: ScenarioSpec,
    micro_index: torch.Tensor,
    num_communities: int,
    *,
    lp_positive_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left_parts = []
    right_parts = []
    task_parts = []
    split_parts = []
    split_index = {"train": 0, "val": 1, "test": 2}
    for task_id, splits in spec.query_ids_by_task_split.items():
        for split, query_ids in splits.items():
            if spec.problem_type == "LP" and lp_positive_only:
                query_ids = query_ids[spec.labels[query_ids] == 1]
            if spec.problem_type == "NC":
                left = micro_index[query_ids]
                right = left
            else:
                assert spec.query_endpoints is not None
                endpoints = spec.query_endpoints[query_ids]
                left = micro_index[endpoints[:, 0]]
                right = micro_index[endpoints[:, 1]]
            left_parts.append(left)
            right_parts.append(right)
            task_parts.append(torch.full_like(left, task_id))
            split_parts.append(torch.full_like(left, split_index[split]))
    left = torch.cat(left_parts)
    right = torch.cat(right_parts)
    tasks = torch.cat(task_parts)
    splits = torch.cat(split_parts)
    encoded = (((left * num_communities + right) * spec.num_tasks + tasks) * 3 + splits)
    unique, counts = torch.unique(encoded, sorted=True, return_counts=True)
    splits = unique.remainder(3)
    unique = unique.div(3, rounding_mode="floor")
    tasks = unique.remainder(spec.num_tasks)
    unique = unique.div(spec.num_tasks, rounding_mode="floor")
    right = unique.remainder(num_communities)
    left = unique.div(num_communities, rounding_mode="floor")
    return left, right, tasks, splits, counts


def _support_from_communities(
    community_owner: torch.Tensor,
    compressed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_clients: int,
    num_tasks: int,
) -> torch.Tensor:
    left, right, tasks, splits, counts = compressed
    left_owner = community_owner[left]
    right_owner = community_owner[right]
    internal = left_owner == right_owner
    encoded = (left_owner[internal] * num_tasks + tasks[internal]) * 3 + splits[internal]
    values = torch.zeros(num_clients * num_tasks * 3, dtype=torch.long)
    values.index_add_(0, encoded, counts[internal])
    return values.reshape(num_clients, num_tasks, 3)


def _compressed_edge_counts(
    edge_index: torch.Tensor,
    micro_index: torch.Tensor,
    num_communities: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = (
        micro_index[edge_index[0]] * num_communities
        + micro_index[edge_index[1]]
    )
    unique, counts = torch.unique(encoded, sorted=True, return_counts=True)
    right = unique.remainder(num_communities)
    left = unique.div(num_communities, rounding_mode="floor")
    return left, right, counts


def _compressed_affinity(
    compressed: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ],
    num_communities: int,
) -> torch.Tensor:
    left, right, _, _, counts = compressed
    affinity = torch.zeros(
        num_communities * num_communities, dtype=torch.float64
    )
    affinity.index_add_(
        0,
        left * num_communities + right,
        counts.to(torch.float64),
    )
    return affinity.reshape(num_communities, num_communities)


def _community_semantic_counts(
    compressed: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ],
    num_communities: int,
    num_tasks: int,
) -> torch.Tensor:
    """Build an endpoint-level task/domain proxy without exposing labels."""

    left, right, tasks, _, counts = compressed
    values = torch.zeros(num_communities * num_tasks, dtype=torch.float64)
    values.index_add_(
        0,
        left * num_tasks + tasks,
        counts.to(torch.float64),
    )
    distinct = right != left
    values.index_add_(
        0,
        right[distinct] * num_tasks + tasks[distinct],
        counts[distinct].to(torch.float64),
    )
    return values.reshape(num_communities, num_tasks)


def _select_splits(
    compressed: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ],
    allowed: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = torch.zeros_like(compressed[3], dtype=torch.bool)
    for split in allowed:
        selected |= compressed[3] == split
    return tuple(value[selected] for value in compressed)  # type: ignore[return-value]


def _rebalance_preferred_assignment(
    anchored_owner: torch.Tensor,
    preferred_owner: torch.Tensor,
    community_sizes: torch.Tensor,
    affinity: torch.Tensor,
    *,
    lower_size: float,
    upper_size: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Preserve a topology assignment while repairing size after LP anchors."""

    owner = preferred_owner.clone()
    locked = anchored_owner >= 0
    owner[locked] = anchored_owner[locked]
    num_clients = int(owner.max()) + 1
    sizes = torch.zeros(num_clients, dtype=torch.long)
    sizes.index_add_(0, owner, community_sizes)
    tie_order = torch.randperm(owner.shape[0], generator=generator)
    tie_rank = torch.empty(owner.shape[0], dtype=torch.long)
    tie_rank[tie_order] = torch.arange(owner.shape[0])

    for _ in range(owner.shape[0] * 2):
        under = torch.nonzero(sizes.float() < lower_size, as_tuple=True)[0]
        over = torch.nonzero(sizes.float() > upper_size, as_tuple=True)[0]
        if under.numel() == 0 and over.numel() == 0:
            return owner
        target = (
            int(under[torch.argmin(sizes[under])])
            if under.numel()
            else int(torch.argmin(sizes))
        )
        donors = over if over.numel() else torch.nonzero(
            sizes.float() > lower_size, as_tuple=True
        )[0]
        candidates = torch.nonzero(
            (~locked) & torch.isin(owner, donors), as_tuple=True
        )[0]
        ranked = []
        for community in candidates.tolist():
            donor = int(owner[community])
            volume = int(community_sizes[community])
            if float(sizes[donor] - volume) < lower_size:
                continue
            if float(sizes[target] + volume) > upper_size:
                continue
            donor_affinity = float(
                affinity[community, owner == donor].sum()
                + affinity[owner == donor, community].sum()
            )
            target_affinity = float(
                affinity[community, owner == target].sum()
                + affinity[owner == target, community].sum()
            )
            ranked.append(
                (
                    target_affinity - donor_affinity,
                    -abs((float(sizes[target]) + volume) - lower_size),
                    -int(tie_rank[community]),
                    community,
                )
            )
        if not ranked:
            break
        chosen = max(ranked)[-1]
        donor = int(owner[chosen])
        owner[chosen] = target
        sizes[donor] -= community_sizes[chosen]
        sizes[target] += community_sizes[chosen]
    return owner


def _failures(
    sizes: torch.Tensor,
    support: torch.Tensor,
    *,
    tolerance: float,
    minimum_support: Tuple[int, int, int],
) -> Tuple[str, ...]:
    failures: list[str] = []
    num_clients = support.shape[0]
    target = float(sizes.sum()) / num_clients
    lower = target * (1 - tolerance)
    upper = target * (1 + tolerance)
    for client, size in enumerate(sizes.tolist()):
        if size < lower or size > upper:
            failures.append(
                f"client={client} node_count={size} outside [{lower:.2f}, {upper:.2f}]"
            )
    split_names = ("train", "val", "test")
    for client in range(num_clients):
        for task in range(support.shape[1]):
            for split_index, minimum in enumerate(minimum_support):
                actual = int(support[client, task, split_index])
                if actual < minimum:
                    failures.append(
                        f"client={client} task={task} split={split_names[split_index]} "
                        f"support={actual} required={minimum}"
                    )
    return tuple(failures)


def _edge_cut_ratio(edge_index: torch.Tensor, owner: torch.Tensor) -> float:
    if edge_index.shape[1] == 0:
        return 0.0
    return float((owner[edge_index[0]] != owner[edge_index[1]]).float().mean())


def _compressed_edge_cut_ratio(
    community_owner: torch.Tensor,
    compressed_edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    left, right, counts = compressed_edges
    cut = community_owner[left] != community_owner[right]
    return float(counts[cut].sum() / counts.sum().clamp_min(1))


def _compressed_metis_assignments(
    compressed_edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    community_sizes: torch.Tensor,
    *,
    num_clients: int,
    seed: int,
    attempts: int,
) -> list[torch.Tensor]:
    try:
        import pymetis  # type: ignore
    except ImportError:
        return []
    left, right, counts = compressed_edges
    non_self = left != right
    left = left[non_self]
    right = right[non_self]
    counts = counts[non_self]
    if left.numel() == 0:
        return []
    num_communities = community_sizes.shape[0]
    row_counts = torch.bincount(left, minlength=num_communities)
    xadj = torch.zeros(num_communities + 1, dtype=torch.long)
    xadj[1:] = row_counts.cumsum(0)
    assignments = []
    for attempt in range(attempts):
        options = pymetis.Options(seed=(seed + attempt) % (2**31))
        _, membership = pymetis.part_graph(
            num_clients,
            xadj=xadj.numpy(),
            adjncy=right.numpy(),
            vweights=community_sizes.numpy(),
            eweights=counts.numpy(),
            recursive=True,
            options=options,
        )
        owner = torch.as_tensor(membership, dtype=torch.long)
        if owner.shape != (num_communities,) or bool((owner < 0).any()):
            continue
        ordered_clients = sorted(
            range(num_clients),
            key=lambda client: (
                int(torch.nonzero(owner == client)[0])
                if bool((owner == client).any())
                else num_communities + client
            ),
        )
        remap = torch.empty(num_clients, dtype=torch.long)
        remap[torch.tensor(ordered_clients)] = torch.arange(num_clients)
        assignments.append(remap[owner])
    return assignments


def _seed_atomic_support(
    compressed_support: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ],
    community_sizes: torch.Tensor,
    *,
    num_clients: int,
    num_tasks: int,
    minimum_support: Tuple[int, int, int],
    generator: torch.Generator,
    preferred_owner: torch.Tensor | None = None,
) -> torch.Tensor:
    left, right, tasks, splits, counts = compressed_support
    atomic = left == right
    left = left[atomic]
    tasks = tasks[atomic]
    splits = splits[atomic]
    counts = counts[atomic]
    num_communities = community_sizes.shape[0]
    flat = torch.zeros(num_communities * num_tasks * 3, dtype=torch.long)
    encoded = (left * num_tasks + tasks) * 3 + splits
    flat.index_add_(0, encoded, counts)
    support = flat.reshape(num_communities, num_tasks, 3)
    supporting_counts = (support > 0).sum(dim=0)
    rarity = supporting_counts.clamp_min(1).to(torch.float64).reciprocal()
    requirements = sorted(
        (
            (int(supporting_counts[task, split]), task, split)
            for task in range(num_tasks)
            for split in range(3)
            if minimum_support[split] > 0
        )
    )
    owner = torch.full((num_communities,), -1, dtype=torch.long)
    client_support = torch.zeros(num_clients, num_tasks, 3, dtype=torch.long)
    client_order = torch.randperm(num_clients, generator=generator).tolist()
    tie_order = torch.randperm(num_communities, generator=generator)
    tie_rank = torch.empty(num_communities, dtype=torch.long)
    tie_rank[tie_order] = torch.arange(num_communities)
    minimum = torch.tensor(minimum_support, dtype=torch.long).view(1, 3)
    for _, task, split in requirements:
        required = minimum_support[split]
        for client in client_order:
            while int(client_support[client, task, split]) < required:
                available = torch.nonzero(
                    (owner < 0) & (support[:, task, split] > 0),
                    as_tuple=True,
                )[0]
                if available.numel() == 0:
                    raise PartitionInfeasibleError(
                        "Atomic support anchoring exhausted eligible communities.",
                        [
                            f"client={client} task={task} split={split} "
                            f"support={int(client_support[client, task, split])} "
                            f"required={required}"
                        ],
                    )
                deficits = (minimum - client_support[client]).clamp_min(0)
                candidates = []
                for community in available.tolist():
                    gain = torch.minimum(support[community], deficits)
                    weighted_gain = float((gain.to(torch.float64) * rarity).sum())
                    candidates.append(
                        (
                            int(
                                preferred_owner is not None
                                and preferred_owner[community] == client
                            ),
                            weighted_gain,
                            int(support[community, task, split]),
                            -int(community_sizes[community]),
                            -int(tie_rank[community]),
                            community,
                        )
                    )
                chosen = max(candidates)[-1]
                owner[chosen] = client
                client_support[client] += support[chosen]
    return owner


def _seed_pair_support(
    compressed_support: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ],
    community_sizes: torch.Tensor,
    *,
    num_clients: int,
    num_tasks: int,
    minimum_support: Tuple[int, int, int],
    generator: torch.Generator,
    preferred_owner: torch.Tensor | None = None,
) -> torch.Tensor:
    """Co-locate query endpoint communities to guarantee dense pair support."""

    left, right, tasks, splits, counts = compressed_support
    owner = torch.full((community_sizes.shape[0],), -1, dtype=torch.long)
    pair_indices = {
        (task, split): torch.nonzero(
            (tasks == task) & (splits == split), as_tuple=True
        )[0]
        for task in range(num_tasks)
        for split in range(3)
    }
    requirements = sorted(
        (
            (int(counts[pair_indices[task, split]].sum()), task, split)
            for task in range(num_tasks)
            for split in range(3)
            if minimum_support[split] > 0
        )
    )
    client_order = torch.randperm(num_clients, generator=generator).tolist()
    edge_order = torch.randperm(left.shape[0], generator=generator)
    edge_rank = torch.empty(left.shape[0], dtype=torch.long)
    edge_rank[edge_order] = torch.arange(left.shape[0])
    claimed = torch.zeros(left.shape[0], dtype=torch.bool)
    client_support = torch.zeros(num_clients, num_tasks, 3, dtype=torch.long)

    for _, task, split in requirements:
        required = minimum_support[split]
        for client in client_order:
            while int(client_support[client, task, split]) < required:
                eligible = pair_indices[task, split]
                compatible = (
                    ~claimed[eligible]
                    & ((owner[left[eligible]] == -1) | (owner[left[eligible]] == client))
                    & ((owner[right[eligible]] == -1) | (owner[right[eligible]] == client))
                )
                candidates = eligible[compatible]
                if candidates.numel() == 0:
                    raise PartitionInfeasibleError(
                        "Pair-support anchoring exhausted eligible communities.",
                        [
                            f"client={client} task={task} split={split} "
                            f"support={int(client_support[client, task, split])} "
                            f"required={required}"
                        ],
                    )
                left_unassigned = owner[left[candidates]] < 0
                right_unassigned = owner[right[candidates]] < 0
                distinct = left[candidates] != right[candidates]
                new_endpoint_count = left_unassigned.to(torch.long) + (
                    right_unassigned & distinct
                ).to(torch.long)
                candidates = candidates[
                    new_endpoint_count == new_endpoint_count.min()
                ]
                left_unassigned = owner[left[candidates]] < 0
                right_unassigned = owner[right[candidates]] < 0
                distinct = left[candidates] != right[candidates]
                new_volume = (
                    community_sizes[left[candidates]] * left_unassigned.to(torch.long)
                    + community_sizes[right[candidates]]
                    * (right_unassigned & distinct).to(torch.long)
                )
                candidates = candidates[new_volume == new_volume.min()]
                candidates = candidates[counts[candidates] == counts[candidates].max()]
                chosen = int(candidates[edge_rank[candidates].argmin()])
                owner[left[chosen]] = client
                owner[right[chosen]] = client
                claimed[chosen] = True
                client_support[client, task, split] += counts[chosen]
    return owner


def assign_micro_communities(
    spec: ScenarioSpec,
    micro_ids: torch.Tensor,
    *,
    num_clients: int,
    seed: int,
    spatial_profile: str,
    client_size_tolerance: float,
    minimum_support: Tuple[int, int, int],
    maximum_iterations: int,
    allow_infeasible: bool,
    minimum_internal_candidate_coverage: float | None = None,
    minimum_internal_query_coverage: float | None = None,
    lp_partition_information_scope: str = "all_positive_splits",
) -> AssignmentOutcome:
    """Search deterministic atomic assignments and retain the best feasible one."""

    communities = torch.unique(micro_ids, sorted=True)
    if len(communities) < num_clients:
        raise PartitionInfeasibleError(
            "There are fewer atomic micro-communities than clients.",
            [f"micro_communities={len(communities)} clients={num_clients}"],
        )
    community_remap = torch.full(
        (int(communities.max()) + 1,), -1, dtype=torch.long
    )
    community_remap[communities] = torch.arange(len(communities))
    micro_index = community_remap[micro_ids]
    community_sizes = torch.bincount(micro_index, minlength=len(communities))
    complete_support = _compressed_query_support(
        spec, micro_index, len(communities)
    )
    complete_candidates = (
        _compressed_query_support(
            spec,
            micro_index,
            len(communities),
            lp_positive_only=False,
        )
        if spec.problem_type == "LP"
        else complete_support
    )
    train_only_lp = (
        spec.problem_type == "LP"
        and lp_partition_information_scope == "topology_and_train_queries"
    )
    topology_only_lp = (
        spec.problem_type == "LP"
        and lp_partition_information_scope == "topology_only"
    )
    compressed_support = (
        _select_splits(complete_support, ())
        if topology_only_lp
        else _select_splits(complete_support, (0,))
        if train_only_lp
        else complete_support
    )
    compressed_candidates = (
        _select_splits(complete_candidates, ())
        if topology_only_lp
        else _select_splits(complete_candidates, (0,))
        if train_only_lp
        else complete_candidates
    )
    objective_minimum_support = (
        (0, 0, 0)
        if topology_only_lp
        else (minimum_support[0], 0, 0)
        if train_only_lp
        else minimum_support
    )
    support_left, _, support_tasks, support_splits, support_counts = compressed_support
    feasibility_failures = []
    split_names = ("train", "val", "test")
    for task_id in range(spec.num_tasks):
        for split_index, minimum in enumerate(objective_minimum_support):
            if minimum <= 0:
                continue
            selected = (support_tasks == task_id) & (support_splits == split_index)
            if spec.problem_type == "NC":
                supporting_communities = int(
                    torch.unique(support_left[selected]).numel()
                )
                if supporting_communities < num_clients:
                    feasibility_failures.append(
                        f"task={task_id} split={split_names[split_index]} "
                        f"supporting_micro_communities={supporting_communities} "
                        f"clients={num_clients}"
                    )
            elif spec.problem_type == "LP":
                available = int(support_counts[selected].sum())
                required = num_clients * minimum
                if available < required:
                    feasibility_failures.append(
                        f"task={task_id} split={split_names[split_index]} "
                        f"positive_queries={available} required={required}"
                    )
    if feasibility_failures:
        raise PartitionInfeasibleError(
            "Atomic micro-community support is insufficient for every client.",
            feasibility_failures,
        )
    atomic_lp_seed = False
    if spec.problem_type == "LP":
        support_right = compressed_support[1]
        atomic = support_left == support_right
        atomic_lp_seed = all(
            int(
                torch.unique(
                    support_left[
                        atomic
                        & (support_tasks == task_id)
                        & (support_splits == split_index)
                    ]
                ).numel()
            )
            >= num_clients
            for task_id in range(spec.num_tasks)
            for split_index, minimum in enumerate(objective_minimum_support)
            if minimum > 0
        )
    compressed_edges = _compressed_edge_counts(
        spec.edge_index, micro_index, len(communities)
    )
    topology_only_affinity = {"easy": 0.5, "mild": 5.0, "hard": 20.0}
    semantic_affinity = {"easy": 20.0, "mild": 5.0, "hard": 0.5}
    topology_affinity_weight = (
        topology_only_affinity[spatial_profile]
        if topology_only_lp
        else semantic_affinity[spatial_profile]
    )
    semantic_divergence_target = {
        "easy": 0.0,
        "mild": 0.08,
        "hard": 0.25,
    }[spatial_profile]
    edge_left, edge_right, edge_counts = compressed_edges
    affinity_flat = torch.zeros(
        len(communities) * len(communities), dtype=torch.float64
    )
    affinity_flat.index_add_(
        0,
        edge_left * len(communities) + edge_right,
        edge_counts.to(torch.float64),
    )
    community_affinity = affinity_flat.reshape(len(communities), len(communities))
    candidate_affinity = _compressed_affinity(
        compressed_candidates, len(communities)
    )
    community_semantic = _community_semantic_counts(
        compressed_support, len(communities), spec.num_tasks
    )
    global_semantic = community_semantic.sum(dim=0)
    global_semantic = global_semantic / global_semantic.sum().clamp_min(1)
    community_strength = community_affinity.sum(dim=1).clamp_min(1.0)
    target_size = float(community_sizes.sum()) / num_clients
    lower_size = target_size * (1 - client_size_tolerance)
    upper_size = target_size * (1 + client_size_tolerance)
    best: AssignmentOutcome | None = None
    preferred_owner: torch.Tensor | None = None

    def evaluate(
        indexed_owner: torch.Tensor,
        attempt_count: int,
        support_anchor_community_count: int = 0,
    ) -> AssignmentOutcome:
        client_sizes = torch.zeros(num_clients, dtype=torch.long)
        client_sizes.index_add_(0, indexed_owner, community_sizes)
        support = _support_from_communities(
            indexed_owner,
            compressed_support,
            num_clients=num_clients,
            num_tasks=spec.num_tasks,
        )
        failures = list(_failures(
            client_sizes,
            support,
            tolerance=client_size_tolerance,
            minimum_support=objective_minimum_support,
        ))
        candidate_left, candidate_right, _, _, candidate_counts = compressed_candidates
        candidate_internal = indexed_owner[candidate_left] == indexed_owner[candidate_right]
        candidate_coverage = float(
            candidate_counts[candidate_internal].sum()
            / candidate_counts.sum().clamp_min(1)
        )
        if (
            minimum_internal_candidate_coverage is not None
            and candidate_coverage < minimum_internal_candidate_coverage
        ):
            failures.append(
                f"internal_candidate_coverage={candidate_coverage:.6f} "
                f"required={minimum_internal_candidate_coverage:.6f}"
            )
        if (
            minimum_internal_query_coverage is not None
            and candidate_coverage < minimum_internal_query_coverage
        ):
            failures.append(
                f"internal_query_coverage={candidate_coverage:.6f} "
                f"required={minimum_internal_query_coverage:.6f}"
            )
        semantic_divergence = None
        semantic_penalty = 0.0
        if not topology_only_lp:
            task_volume = support.sum(dim=2).float()
            normalized = task_volume / task_volume.sum(dim=1, keepdim=True).clamp_min(1)
            global_distribution = task_volume.sum(dim=0)
            global_distribution = (
                global_distribution / global_distribution.sum().clamp_min(1)
            )
            semantic_divergence = float(
                ((normalized - global_distribution.unsqueeze(0)) ** 2)
                .sum(dim=1)
                .mean()
            )
            semantic_penalty = abs(
                semantic_divergence - semantic_divergence_target
            ) * 1_000.0
        normalized_size_variance = float(
            client_sizes.float().var(unbiased=False) / max(1.0, target_size**2)
        )
        coverage_penalty = (
            (1.0 - candidate_coverage) * 1_000.0
            if (
                minimum_internal_candidate_coverage is not None
                or minimum_internal_query_coverage is not None
            )
            else 0.0
        )
        score = (
            len(failures) * 1_000_000_000.0
            + normalized_size_variance * 10.0
            + _compressed_edge_cut_ratio(indexed_owner, compressed_edges) * 100.0
            + coverage_penalty
            + semantic_penalty
        )
        complete_realized_support = _support_from_communities(
            indexed_owner,
            complete_support,
            num_clients=num_clients,
            num_tasks=spec.num_tasks,
        )
        return AssignmentOutcome(
            indexed_owner[micro_index],
            complete_realized_support,
            score,
            tuple(failures),
            attempt_count,
            support_anchor_community_count,
            semantic_divergence,
        )

    # PyMetis runs in-process, so a native crash cannot be recovered by this
    # generator.  Compressed LC/LP query graphs can exercise that failure mode;
    # their deterministic support-first search below is the authoritative
    # proposal mechanism.  Keep the topology proposal only for small, easy NC
    # graphs, where node ownership has no paired-query support constraint.
    if (
        spec.problem_type == "NC"
        and spatial_profile == "easy"
        and len(communities) <= 200
    ):
        metis_attempts = min(20, max(1, maximum_iterations))
        for attempt, indexed_owner in enumerate(
            _compressed_metis_assignments(
                compressed_edges,
                community_sizes,
                num_clients=num_clients,
                seed=seed,
                attempts=metis_attempts,
            ),
            start=1,
        ):
            candidate = evaluate(indexed_owner, attempt)
            if best is None or candidate.score < best.score:
                best = candidate
                preferred_owner = indexed_owner.clone()
        # Keep the best topology proposal as a candidate and as a preferred
        # support anchor, then continue searching for the requested semantic mix.

    iterations = max(1, maximum_iterations)
    last_seed_failures: Tuple[str, ...] = ()
    for iteration in range(iterations):
        generator = torch_generator(seed, "assignment", iteration)
        order = communities[torch.randperm(len(communities), generator=generator)].tolist()
        if spec.problem_type == "NC":
            indexed_owner = _seed_atomic_support(
                compressed_support,
                community_sizes,
                num_clients=num_clients,
                num_tasks=spec.num_tasks,
                minimum_support=objective_minimum_support,
                generator=generator,
                preferred_owner=None,
            )
        elif spec.problem_type in {"LC", "LP"}:
            try:
                seed_function = (
                    _seed_atomic_support if atomic_lp_seed else _seed_pair_support
                )
                indexed_owner = seed_function(
                    compressed_support,
                    community_sizes,
                    num_clients=num_clients,
                    num_tasks=spec.num_tasks,
                    minimum_support=objective_minimum_support,
                    generator=generator,
                    preferred_owner=preferred_owner,
                )
            except PartitionInfeasibleError as error:
                last_seed_failures = error.failures
                continue
            support_anchor_community_count = int((indexed_owner >= 0).sum())
            if atomic_lp_seed and preferred_owner is not None:
                indexed_owner = _rebalance_preferred_assignment(
                    indexed_owner,
                    preferred_owner,
                    community_sizes,
                    community_affinity,
                    lower_size=lower_size,
                    upper_size=upper_size,
                    generator=generator,
                )
        else:
            indexed_owner = torch.full((len(communities),), -1, dtype=torch.long)
        client_sizes = torch.zeros(num_clients, dtype=torch.long)
        assigned = indexed_owner >= 0
        if assigned.any():
            client_sizes.index_add_(
                0, indexed_owner[assigned], community_sizes[assigned]
            )
        affinity_to_client = torch.zeros(
            len(communities), num_clients, dtype=torch.float64
        )
        semantic_by_client = torch.zeros(
            num_clients, spec.num_tasks, dtype=torch.float64
        )
        for community_index in torch.nonzero(assigned, as_tuple=True)[0].tolist():
            affinity_to_client[:, indexed_owner[community_index]] += (
                community_affinity[:, community_index]
            )
            semantic_by_client[indexed_owner[community_index]] += (
                community_semantic[community_index]
            )
        client_ties = torch.randperm(num_clients, generator=generator).tolist()
        tie_rank = {client: rank for rank, client in enumerate(client_ties)}
        for community in order:
            community_index = int(community_remap[int(community)])
            if indexed_owner[community_index] >= 0:
                continue
            candidate_scores = []
            underfilled = [
                client
                for client in range(num_clients)
                if float(client_sizes[client]) < lower_size
            ]
            eligible_clients = underfilled or list(range(num_clients))
            for client in eligible_clients:
                projected_size = float(
                    client_sizes[client] + community_sizes[community_index]
                )
                overflow = max(0.0, projected_size - upper_size)
                affinity = float(
                    affinity_to_client[community_index, client]
                    / community_strength[community_index]
                )
                projected_semantic = (
                    semantic_by_client[client] + community_semantic[community_index]
                )
                projected_semantic = projected_semantic / projected_semantic.sum().clamp_min(1)
                semantic_divergence = float(
                    ((projected_semantic - global_semantic) ** 2).sum()
                )
                semantic_error = (
                    0.0
                    if topology_only_lp
                    else abs(semantic_divergence - semantic_divergence_target)
                )
                objective = (
                    projected_size / target_size
                    - topology_affinity_weight * affinity
                    + semantic_error * 50.0
                )
                candidate_scores.append(
                    (overflow, objective, tie_rank[client], client)
                )
            candidates = sorted(candidate_scores)
            chosen = candidates[0][-1]
            indexed_owner[community_index] = chosen
            client_sizes[chosen] += community_sizes[community_index]
            affinity_to_client[:, chosen] += community_affinity[:, community_index]
            semantic_by_client[chosen] += community_semantic[community_index]
        candidate = evaluate(
            indexed_owner,
            iteration + 1,
            support_anchor_community_count=(
                support_anchor_community_count
                if spec.problem_type in {"LC", "LP"}
                else 0
            ),
        )
        if best is None or candidate.score < best.score:
            best = candidate
            # LC and LP use support-first seeds, so retain their best compressed
            # assignment for the same deterministic swap refinement as NC.
            preferred_owner = indexed_owner.clone()
        if not candidate.failures and iteration >= min(20, iterations - 1):
            break
    if (
        preferred_owner is not None
        and not topology_only_lp
        and spatial_profile in {"mild", "hard"}
    ):
        refined_owner = preferred_owner.clone()
        refined = evaluate(refined_owner, iterations + 1)
        for refinement in range(iterations):
            generator = torch_generator(
                seed, "semantic-swap-refinement", spatial_profile, refinement
            )
            order = torch.randperm(len(communities), generator=generator).tolist()
            pair: tuple[int, int] | None = None
            for left_index, left_community in enumerate(order):
                for right_community in order[left_index + 1 :]:
                    if (
                        refined_owner[left_community]
                        != refined_owner[right_community]
                    ):
                        pair = (left_community, right_community)
                        break
                if pair is not None:
                    break
            if pair is None:
                break
            proposal = refined_owner.clone()
            left_community, right_community = pair
            proposal[left_community], proposal[right_community] = (
                refined_owner[right_community],
                refined_owner[left_community],
            )
            candidate = evaluate(
                proposal, iterations + refinement + 2
            )
            if candidate.score < refined.score:
                refined_owner = proposal
                refined = candidate
                if best is None or candidate.score < best.score:
                    best = candidate

    if best is None:
        raise PartitionInfeasibleError(
            "No LP positive-pair support seed satisfies dense support.",
            last_seed_failures,
        )
    if best.failures and not allow_infeasible:
        raise PartitionInfeasibleError(
            "No atomic micro-community assignment satisfies dense support.",
            best.failures,
        )
    return best
