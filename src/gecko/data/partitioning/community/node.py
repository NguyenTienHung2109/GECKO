"""Quota-first soft-community ownership for node classification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from typing import Mapping

import torch

from gecko.data.partitioning.base import WeightedLogicalTopology
from gecko.data.partitioning.base import build_weighted_logical_topology
from gecko.data.partitioning.dirichlet.quota import ExactDirichletQuota
from gecko.data.partitioning.dirichlet.quota import ExactDirichletQuotaGenerator
from gecko.types import ScenarioSpec
from gecko.data.partitioning.community.common import MicroCommunityResult
from gecko.data.partitioning.community.common import SoftMicroCommunityConfig
from gecko.data.partitioning.community.common import balanced_total_capacities
from gecko.data.partitioning.community.common import build_microcommunities
from gecko.data.partitioning.community.common import community_diagnostics
from gecko.data.partitioning.community.common import complete_owner_soft
from gecko.data.partitioning.community.common import initial_neighbor_owner_mass
from gecko.data.partitioning.community.common import micro_result_from_ids
from gecko.data.partitioning.community.common import objective_tuple
from gecko.data.partitioning.community.common import payload_hash
from gecko.data.partitioning.community.common import stable_int
from gecko.data.partitioning.community.common import tensor_hash


NC_SOFT_COMMUNITY_VERSION = "nc_soft_community_exact_node_dirichlet_v3"


@dataclass(frozen=True)
class NCSoftCommunityConfig:
    """Configuration for quota-first NC soft-community ownership."""

    num_clients: int
    alpha_dirichlet: float
    seed: int
    client_size_tolerance: float = 0.20
    minimum_train_queries_per_client_task: int = 1
    minimum_validation_queries_per_client_task: int = 1
    minimum_test_queries_per_client_task: int = 1
    maximum_microcommunity_nodes: int = 4096
    louvain_resolution: float = 1.0
    maximum_refinement_swaps: int = 16
    refinement_candidate_limit: int = 4
    quota_maximum_retries: int = 64
    reserve_minimum_train_support: bool = False

    def validate(self) -> None:
        if self.num_clients < 1:
            raise ValueError("num_clients must be positive.")
        if not math.isfinite(self.alpha_dirichlet) or self.alpha_dirichlet <= 0:
            raise ValueError("alpha_dirichlet must be positive and finite.")
        if not 0 <= self.client_size_tolerance < 1:
            raise ValueError("client_size_tolerance must lie in [0, 1).")
        if self.minimum_train_queries_per_client_task < 0:
            raise ValueError("Minimum train support cannot be negative.")
        if min(
            self.minimum_validation_queries_per_client_task,
            self.minimum_test_queries_per_client_task,
        ) < 0:
            raise ValueError("Minimum held-out support cannot be negative.")
        if min(self.maximum_refinement_swaps, self.refinement_candidate_limit) < 0:
            raise ValueError("NC refinement budgets cannot be negative.")
        if self.quota_maximum_retries < 1:
            raise ValueError("quota_maximum_retries must be positive.")
        SoftMicroCommunityConfig(
            seed=self.seed,
            maximum_microcommunity_nodes=self.maximum_microcommunity_nodes,
            louvain_resolution=self.louvain_resolution,
        ).validate()


@dataclass(frozen=True)
class NCSoftCommunityResult:
    """Complete NC ownership with exact semantic quota."""

    owner: torch.Tensor
    micro_ids: torch.Tensor
    quota: ExactDirichletQuota
    total_capacities: torch.Tensor
    diagnostics: dict[str, Any]


def _heldout_support_counts(
    owner: torch.Tensor,
    heldout_ids: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    num_clients: int,
) -> dict[str, list[list[int]]]:
    """Count held-out NC queries using identities only, never their labels."""

    result: dict[str, list[list[int]]] = {}
    for split in ("val", "test"):
        rows: list[list[int]] = []
        for task in sorted(heldout_ids):
            ids = heldout_ids[task][split]
            rows.append(torch.bincount(owner[ids], minlength=num_clients).tolist())
        result[split] = rows
    return result


def _reserve_heldout_nodes(
    partial: torch.Tensor,
    micro_ids: torch.Tensor,
    topology: WeightedLogicalTopology,
    heldout_ids: Mapping[int, Mapping[str, torch.Tensor]],
    capacities: torch.Tensor,
    *,
    config: NCSoftCommunityConfig,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Lock label-blind held-out anchors while treating community as preference."""

    owner = partial.clone()
    trace: list[dict[str, Any]] = []
    counts = torch.bincount(owner[owner >= 0], minlength=config.num_clients)
    num_micros = int(micro_ids.max()) + 1
    micro_mass = torch.zeros((num_micros, config.num_clients), dtype=torch.long)
    assigned = owner >= 0
    if bool(assigned.any()):
        encoded = micro_ids[assigned] * config.num_clients + owner[assigned]
        micro_mass.view(-1).index_add_(
            0, encoded, torch.ones(int(assigned.sum()), dtype=torch.long)
        )
    neighbor_mass = initial_neighbor_owner_mass(
        owner, topology, num_clients=config.num_clients
    )
    required = {
        "val": config.minimum_validation_queries_per_client_task,
        "test": config.minimum_test_queries_per_client_task,
    }
    for task in sorted(heldout_ids):
        for split in ("val", "test"):
            pool = heldout_ids[task][split].tolist()
            pool_tensor = torch.tensor(pool, dtype=torch.long)
            minimum = required[split]
            if len(pool) < config.num_clients * minimum:
                raise ValueError(
                    f"NC held-out support infeasible: task={task} split={split} "
                    f"has {len(pool)} queries but requires {config.num_clients * minimum}."
                )
            for client in range(config.num_clients):
                for _ in range(minimum):
                    candidates = pool_tensor[owner[pool_tensor] < 0]
                    if (
                        candidates.numel() == 0
                        or int(counts[client]) >= int(capacities[client])
                    ):
                        raise ValueError(
                            "NC held-out support is structurally infeasible under "
                            f"client capacities (client={client}, task={task}, split={split})."
                        )
                    affinity = neighbor_mass[candidates, client]
                    candidates = candidates[affinity == affinity.max()]
                    community_affinity = micro_mass[micro_ids[candidates], client]
                    candidates = candidates[
                        community_affinity == community_affinity.max()
                    ]
                    node = min(
                        candidates.tolist(),
                        key=lambda value: stable_int(
                            NC_SOFT_COMMUNITY_VERSION,
                            config.seed,
                            "heldout",
                            task,
                            split,
                            client,
                            value,
                        ),
                    )
                    owner[node] = client
                    counts[client] += 1
                    micro_mass[int(micro_ids[node]), client] += 1
                    neighbors = topology.incident_neighbors[node]
                    if neighbors.numel():
                        neighbor_mass[:, client].index_add_(
                            0, neighbors, topology.incident_weights[node]
                        )
                    trace.append(
                        {"client": client, "task": task, "split": split, "query_id": node}
                    )
    return owner, trace


def _realized_quota(
    owner: torch.Tensor,
    train_ids: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    num_clients: int,
    num_classes: int,
) -> torch.Tensor:
    return torch.bincount(
        owner[train_ids] * num_classes + train_labels,
        minlength=num_clients * num_classes,
    ).reshape(num_clients, num_classes)


def _node_order(
    nodes: torch.Tensor,
    topology: WeightedLogicalTopology,
    micro_ids: torch.Tensor,
    *,
    seed: int,
    client: int,
) -> list[int]:
    """Prefer internally well-connected nodes, then use a stable tie-break."""

    selected = set(int(node) for node in nodes.tolist())
    ranked = []
    for node in selected:
        internal_weight = sum(
            float(weight)
            for neighbor, weight in zip(
                topology.incident_neighbors[node].tolist(),
                topology.incident_weights[node].tolist(),
            )
            if neighbor in selected and micro_ids[neighbor] == micro_ids[node]
        )
        ranked.append(
            (-internal_weight, stable_int(NC_SOFT_COMMUNITY_VERSION, seed, node, client), node)
        )
    return [node for _, _, node in sorted(ranked)]


def _allocate_training_nodes(
    topology: WeightedLogicalTopology,
    micro_ids: torch.Tensor,
    train_ids: torch.Tensor,
    train_labels: torch.Tensor,
    quota: ExactDirichletQuota,
    *,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pack community-class supplies directly into exact client quotas."""

    num_clients, num_classes = quota.integer_quota.shape
    owner = torch.full((topology.num_nodes,), -1, dtype=torch.long)
    remaining = quota.integer_quota.detach().cpu().long().clone()
    num_micros = int(micro_ids.max()) + 1
    micro_mass = torch.zeros((num_micros, num_clients), dtype=torch.long)
    allocation_steps: list[dict[str, Any]] = []

    micro_rows: list[tuple[tuple[int, int, int], int]] = []
    for micro in range(num_micros):
        selected = train_ids[micro_ids[train_ids] == micro]
        histogram = torch.bincount(
            train_labels[micro_ids[train_ids] == micro], minlength=num_classes
        )
        diversity = int((histogram > 0).sum())
        micro_rows.append(((-diversity, -int(selected.numel()), micro), micro))

    for _, micro in sorted(micro_rows):
        group_nodes = train_ids[micro_ids[train_ids] == micro]
        if group_nodes.numel() == 0:
            continue
        by_class = {
            class_id: group_nodes[train_labels[micro_ids[train_ids] == micro] == class_id]
            for class_id in range(num_classes)
        }
        group_remaining = torch.tensor(
            [nodes.numel() for nodes in by_class.values()], dtype=torch.long
        )
        ordered_nodes: dict[tuple[int, int], list[int]] = {}
        while int(group_remaining.sum()) > 0:
            whole_fit = [
                client
                for client in range(num_clients)
                if bool((remaining[client] >= group_remaining).all())
            ]
            candidates = whole_fit or [
                client
                for client in range(num_clients)
                if bool(((remaining[client] > 0) & (group_remaining > 0)).any())
            ]
            if not candidates:
                raise AssertionError("NC quota-first packing exhausted client demand.")
            client = min(
                candidates,
                key=lambda value: (
                    0 if value in whole_fit else 1,
                    -int(micro_mass[micro, value]),
                    -int(torch.minimum(remaining[value], group_remaining).sum()),
                    stable_int(NC_SOFT_COMMUNITY_VERSION, seed, micro, value),
                ),
            )
            amount = (
                group_remaining.clone()
                if client in whole_fit
                else torch.minimum(group_remaining, remaining[client])
            )
            if int(amount.sum()) == 0:
                raise AssertionError("NC soft packing made no progress.")
            assigned_count = 0
            for class_id in range(num_classes):
                take = int(amount[class_id])
                if take == 0:
                    continue
                key = (class_id, client)
                if key not in ordered_nodes:
                    ordered_nodes[key] = _node_order(
                        by_class[class_id],
                        topology,
                        micro_ids,
                        seed=seed,
                        client=client,
                    )
                available = [
                    node
                    for node in ordered_nodes[key]
                    if int(owner[node]) < 0
                ]
                if len(available) < take:
                    # A different client-specific ordering may overlap already assigned
                    # nodes; fall back to the remaining canonical IDs.
                    available = sorted(
                        node
                        for node in by_class[class_id].tolist()
                        if int(owner[node]) < 0
                    )
                chosen = torch.tensor(available[:take], dtype=torch.long)
                owner[chosen] = client
                assigned_count += take
            remaining[client] -= amount
            group_remaining -= amount
            micro_mass[micro, client] += assigned_count
            allocation_steps.append(
                {
                    "microcommunity": micro,
                    "client": client,
                    "class_counts": amount.tolist(),
                    "whole_remainder_fit": client in whole_fit,
                }
            )
    if bool((remaining != 0).any()) or bool((owner[train_ids] < 0).any()):
        raise AssertionError("NC soft allocation did not realize the frozen quota.")
    realized = _realized_quota(
        owner,
        train_ids,
        train_labels,
        num_clients=num_clients,
        num_classes=num_classes,
    )
    if not torch.equal(realized, quota.integer_quota):
        raise AssertionError("NC quota-first allocation realized an incorrect matrix.")
    return owner, {
        "allocation_mode": "quota_first_soft_community",
        "allocation_step_count": len(allocation_steps),
        "allocation_steps": allocation_steps,
        "initial_quota_residual": 0,
        "quota_repair_moves": 0,
        "initial_train_class_matrix": realized.tolist(),
    }


def _local_swap_delta(
    owner: torch.Tensor,
    left: int,
    right: int,
    micro_ids: torch.Tensor,
    topology: WeightedLogicalTopology,
    *,
    num_clients: int,
) -> tuple[int, int, float]:
    """Return exact soft-objective delta for one quota-preserving swap."""

    left_owner, right_owner = int(owner[left]), int(owner[right])
    if left_owner == right_owner:
        return (0, 0, 0.0)
    affected_micros = {int(micro_ids[left]), int(micro_ids[right])}
    split_before = split_after = excess_before = excess_after = 0
    for micro in affected_micros:
        counts = torch.bincount(owner[micro_ids == micro], minlength=num_clients)
        size = int(counts.sum())
        split_before += size - int(counts.max())
        excess_before += max(0, int((counts > 0).sum()) - 1)
        after = counts.clone()
        if int(micro_ids[left]) == micro:
            after[left_owner] -= 1
            after[right_owner] += 1
        if int(micro_ids[right]) == micro:
            after[right_owner] -= 1
            after[left_owner] += 1
        split_after += size - int(after.max())
        excess_after += max(0, int((after > 0).sum()) - 1)
    edge_ids = set(topology.incident_edge_ids[left].tolist())
    edge_ids.update(topology.incident_edge_ids[right].tolist())
    cut_before = cut_after = 0.0
    for edge_id in edge_ids:
        source, target = map(int, topology.edge_index[:, edge_id].tolist())
        source_before, target_before = int(owner[source]), int(owner[target])
        source_after = (
            right_owner if source == left else left_owner if source == right else source_before
        )
        target_after = (
            right_owner if target == left else left_owner if target == right else target_before
        )
        weight = float(topology.edge_weights[edge_id])
        cut_before += weight * int(source_before != target_before)
        cut_after += weight * int(source_after != target_after)
    return (
        split_after - split_before,
        excess_after - excess_before,
        cut_after - cut_before,
    )


def _refine_train_swaps(
    owner: torch.Tensor,
    train_ids: torch.Tensor,
    train_labels: torch.Tensor,
    micro_ids: torch.Tensor,
    topology: WeightedLogicalTopology,
    *,
    num_clients: int,
    seed: int,
    maximum_swaps: int,
    candidate_limit: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Improve cohesion with same-class swaps that preserve every hard margin."""

    values = owner.clone()
    trace: list[dict[str, Any]] = []
    if maximum_swaps == 0 or candidate_limit == 0:
        return values, trace
    for _ in range(maximum_swaps):
        best: tuple[tuple[object, ...], int, int, tuple[int, int, float]] | None = None
        for class_id in torch.unique(train_labels, sorted=True).tolist():
            class_nodes = train_ids[train_labels == class_id]
            by_client: dict[int, list[int]] = {}
            for client in range(num_clients):
                candidates = class_nodes[values[class_nodes] == client].tolist()
                candidates.sort(
                    key=lambda node: (
                        -sum(
                            float(weight)
                            for neighbor, weight in zip(
                                topology.incident_neighbors[node].tolist(),
                                topology.incident_weights[node].tolist(),
                            )
                            if int(values[neighbor]) != client
                        ),
                        stable_int(NC_SOFT_COMMUNITY_VERSION, seed, "swap", node),
                    )
                )
                by_client[client] = candidates[:candidate_limit]
            for left_client in range(num_clients):
                for right_client in range(left_client + 1, num_clients):
                    for left in by_client[left_client]:
                        for right in by_client[right_client]:
                            delta = _local_swap_delta(
                                values,
                                left,
                                right,
                                micro_ids,
                                topology,
                                num_clients=num_clients,
                            )
                            if delta >= (0, 0, 0.0):
                                continue
                            rank: tuple[object, ...] = (
                                *delta,
                                stable_int(seed, "swap-pair", left, right),
                            )
                            if best is None or rank < best[0]:
                                best = (rank, left, right, delta)
        if best is None:
            break
        _, left, right, delta = best
        left_owner, right_owner = int(values[left]), int(values[right])
        values[left], values[right] = right_owner, left_owner
        trace.append(
            {
                "left": left,
                "right": right,
                "class": int(train_labels[train_ids == left][0]),
                "objective_delta": list(delta),
            }
        )
    return values, trace


def build_nc_soft_community_owner(
    *,
    topology: WeightedLogicalTopology,
    train_ids: torch.Tensor,
    train_labels: torch.Tensor,
    quota: ExactDirichletQuota,
    config: NCSoftCommunityConfig,
    micro_ids: torch.Tensor | None = None,
    micro_result: MicroCommunityResult | None = None,
    heldout_query_ids_by_task_split: Mapping[int, Mapping[str, torch.Tensor]] | None = None,
) -> NCSoftCommunityResult:
    """Build NC ownership from exact quota with finite community preference."""

    config.validate()
    ids = train_ids.detach().cpu().long().flatten()
    labels = train_labels.detach().cpu().long().flatten()
    if ids.shape != labels.shape or ids.numel() != torch.unique(ids).numel():
        raise ValueError("NC training IDs and labels must be aligned and unique.")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= topology.num_nodes):
        raise ValueError("An NC training node is outside the topology.")
    if quota.integer_quota.shape[0] != config.num_clients:
        raise ValueError("NC quota client count does not match the configuration.")
    if quota.integer_quota.ndim != 2 or bool((quota.integer_quota < 0).any()):
        raise ValueError("NC quota must be a nonnegative client-by-class matrix.")
    if int(quota.integer_quota.sum()) != ids.numel():
        raise ValueError("NC quota must allocate every training node exactly once.")
    num_classes = quota.integer_quota.shape[1]
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError("An NC training label is outside the quota class universe.")
    supply = torch.bincount(labels, minlength=num_classes)
    if not torch.equal(quota.integer_quota.sum(0), supply):
        raise ValueError("NC quota class margins do not match the training labels.")
    heldout = {
        int(task): {
            split: values.detach().cpu().long().flatten()
            for split, values in splits.items()
            if split in {"val", "test"}
        }
        for task, splits in (heldout_query_ids_by_task_split or {}).items()
    }
    for task, splits in heldout.items():
        if set(splits) != {"val", "test"}:
            raise ValueError(f"NC held-out task {task} must define val and test IDs.")
    all_heldout = [heldout[t][s] for t in sorted(heldout) for s in ("val", "test")]
    flat_heldout = torch.cat(all_heldout) if all_heldout else torch.empty(0, dtype=torch.long)
    if flat_heldout.numel():
        if int(flat_heldout.min()) < 0 or int(flat_heldout.max()) >= topology.num_nodes:
            raise ValueError("An NC held-out query ID is outside the topology.")
        combined = torch.cat((ids, flat_heldout))
        if combined.numel() != torch.unique(combined).numel():
            raise ValueError("NC train/val/test query IDs must be globally disjoint.")
    for task in heldout:
        for split, minimum in (
            ("val", config.minimum_validation_queries_per_client_task),
            ("test", config.minimum_test_queries_per_client_task),
        ):
            available = int(heldout[task][split].numel())
            required = config.num_clients * minimum
            if available < required:
                raise ValueError(
                    f"NC held-out support infeasible: task={task} split={split} "
                    f"has {available} queries but requires {required}."
                )
    if micro_ids is not None and micro_result is not None:
        raise ValueError("Provide either micro_ids or micro_result, not both.")
    micro = (
        micro_result
        if micro_result is not None
        else build_microcommunities(
            topology,
            SoftMicroCommunityConfig(
                seed=config.seed,
                maximum_microcommunity_nodes=config.maximum_microcommunity_nodes,
                louvain_resolution=config.louvain_resolution,
            ),
        )
        if micro_ids is None
        else micro_result_from_ids(
            micro_ids,
            num_nodes=topology.num_nodes,
        )
    )
    partial, allocation = _allocate_training_nodes(
        topology,
        micro.micro_ids,
        ids,
        labels,
        quota,
        seed=config.seed,
    )
    train_counts = torch.bincount(partial[ids], minlength=config.num_clients)
    heldout_minimum_per_client = len(heldout) * (
        config.minimum_validation_queries_per_client_task
        + config.minimum_test_queries_per_client_task
    )
    capacities = balanced_total_capacities(
        topology.num_nodes,
        train_counts + heldout_minimum_per_client,
        client_size_tolerance=config.client_size_tolerance,
    )
    partial, heldout_trace = _reserve_heldout_nodes(
        partial,
        micro.micro_ids,
        topology,
        heldout,
        capacities,
        config=config,
    )
    initial_owner = complete_owner_soft(
        partial,
        micro.micro_ids,
        topology,
        capacities,
        seed=config.seed,
    )
    initial_structure = community_diagnostics(
        initial_owner,
        micro.micro_ids,
        topology,
        num_clients=config.num_clients,
    )
    owner, refinement_trace = _refine_train_swaps(
        initial_owner,
        ids,
        labels,
        micro.micro_ids,
        topology,
        num_clients=config.num_clients,
        seed=config.seed,
        maximum_swaps=config.maximum_refinement_swaps,
        candidate_limit=config.refinement_candidate_limit,
    )
    realized = _realized_quota(
        owner,
        ids,
        labels,
        num_clients=config.num_clients,
        num_classes=quota.integer_quota.shape[1],
    )
    if not torch.equal(realized, quota.integer_quota):
        raise AssertionError("NC completion changed the frozen training quota.")
    heldout_support = _heldout_support_counts(
        owner, heldout, num_clients=config.num_clients
    )
    for split, minimum in (
        ("val", config.minimum_validation_queries_per_client_task),
        ("test", config.minimum_test_queries_per_client_task),
    ):
        if any(count < minimum for row in heldout_support[split] for count in row):
            raise AssertionError("NC completion violated locked held-out support.")
    structure = community_diagnostics(
        owner,
        micro.micro_ids,
        topology,
        num_clients=config.num_clients,
    )
    heldout_identity_hash = payload_hash(
        {
            str(task): {
                split: heldout[task][split].tolist()
                for split in ("val", "test")
            }
            for task in sorted(heldout)
        }
    )
    audit_payload = {
        "version": NC_SOFT_COMMUNITY_VERSION,
        "topology_hash": topology.topology_hash,
        "microcommunity_hash": micro.diagnostics["microcommunity_hash"],
        "quota_hash": quota.quota_hash,
        "owner_hash": tensor_hash(owner),
        "heldout_query_identity_hash": heldout_identity_hash,
    }
    diagnostics = {
        "policy_version": NC_SOFT_COMMUNITY_VERSION,
        "label_access_policy": "train_labels_only_heldout_ids_no_heldout_labels",
        "heldout_constraint_role": "immutable_query_identity_only",
        "minimum_validation_queries_per_client_task": config.minimum_validation_queries_per_client_task,
        "minimum_test_queries_per_client_task": config.minimum_test_queries_per_client_task,
        "heldout_support_counts": heldout_support,
        "heldout_anchor_count": len(heldout_trace),
        "heldout_query_identity_hash": heldout_identity_hash,
        "topology_hash": topology.topology_hash,
        "quota_hash": quota.quota_hash,
        "owner_hash": tensor_hash(owner),
        "train_owner_hash": tensor_hash(owner[ids]),
        "total_capacities": capacities.tolist(),
        "final_train_class_matrix": realized.tolist(),
        "initial_soft_objective": list(objective_tuple(initial_structure)),
        "final_soft_objective": list(objective_tuple(structure)),
        "refinement_moves": len(refinement_trace),
        "refinement_trace": refinement_trace,
        "audit_hash": payload_hash(audit_payload),
        **micro.diagnostics,
        **allocation,
        **structure,
    }
    return NCSoftCommunityResult(
        owner=owner,
        micro_ids=micro.micro_ids,
        quota=quota,
        total_capacities=capacities,
        diagnostics=diagnostics,
    )


def build_nc_soft_community_from_spec(
    spec: ScenarioSpec,
    config: NCSoftCommunityConfig,
    *,
    micro_ids: torch.Tensor | None = None,
) -> NCSoftCommunityResult:
    """Build v3 NC ownership with label-blind held-out support constraints."""

    if spec.problem_type != "NC" or spec.labels.ndim != 1:
        raise ValueError("NC soft-community v3 requires single-label NC.")
    topology = build_weighted_logical_topology(
        spec.edge_index,
        num_nodes=int(spec.node_features.shape[0]),
        directed=False,
        representation_id="nc_soft_community_topology_v3",
    )
    train_ids = torch.cat(
        [spec.query_ids_by_task_split[task]["train"] for task in range(spec.num_tasks)]
    ).detach().cpu().long()
    quota = ExactDirichletQuotaGenerator(
        maximum_retries=config.quota_maximum_retries
    ).generate_from_spec(
        spec,
        num_clients=config.num_clients,
        alpha_dirichlet=config.alpha_dirichlet,
        seed=config.seed,
        min_train_support_per_client_task=(
            config.minimum_train_queries_per_client_task
        ),
        reserve_minimum_support=config.reserve_minimum_train_support,
    )
    return build_nc_soft_community_owner(
        topology=topology,
        train_ids=train_ids,
        train_labels=spec.labels[train_ids],
        quota=quota,
        config=config,
        micro_ids=micro_ids,
        heldout_query_ids_by_task_split={
            task: {
                "val": spec.query_ids_by_task_split[task]["val"],
                "test": spec.query_ids_by_task_split[task]["test"],
            }
            for task in range(spec.num_tasks)
        },
    )
