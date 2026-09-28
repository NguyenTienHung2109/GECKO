"""Exhaustive reference oracles for tiny soft-community fixtures."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Iterable
from typing import Mapping

import torch

from gecko.data.partitioning.base import WeightedLogicalTopology
from gecko.data.partitioning.community.common import community_diagnostics
from gecko.data.partitioning.community.common import objective_tuple


@dataclass(frozen=True)
class TinyOracleResult:
    """Exact feasibility and optimum for a deliberately tiny fixture."""

    feasible: bool
    owner: torch.Tensor | None
    selected_by_client_group: dict[int, dict[int, torch.Tensor]] | None
    objective: tuple[object, ...] | None


def _realized_nc(
    owner: torch.Tensor,
    train_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_clients: int,
    num_classes: int,
) -> torch.Tensor:
    return torch.bincount(
        owner[train_ids] * num_classes + labels,
        minlength=num_clients * num_classes,
    ).reshape(num_clients, num_classes)


def brute_force_nc_owner(
    *,
    topology: WeightedLogicalTopology,
    micro_ids: torch.Tensor,
    train_ids: torch.Tensor,
    train_labels: torch.Tensor,
    quota: torch.Tensor,
    capacities: torch.Tensor,
    heldout_query_ids_by_task_split: Mapping[int, Mapping[str, torch.Tensor]] | None = None,
    minimum_validation_queries_per_client_task: int = 0,
    minimum_test_queries_per_client_task: int = 0,
    maximum_nodes: int = 10,
) -> TinyOracleResult:
    """Enumerate every tiny NC owner and return the exact soft optimum."""

    if topology.num_nodes > maximum_nodes:
        raise ValueError("NC brute-force oracle is restricted to tiny graphs.")
    num_clients, num_classes = quota.shape
    best: tuple[tuple[object, ...], torch.Tensor] | None = None
    for raw in itertools.product(range(num_clients), repeat=topology.num_nodes):
        owner = torch.tensor(raw, dtype=torch.long)
        if not torch.equal(torch.bincount(owner, minlength=num_clients), capacities):
            continue
        if not torch.equal(
            _realized_nc(
                owner,
                train_ids,
                train_labels,
                num_clients=num_clients,
                num_classes=num_classes,
            ),
            quota,
        ):
            continue
        heldout = heldout_query_ids_by_task_split or {}
        support_ok = True
        for task in heldout:
            for split, minimum in (
                ("val", minimum_validation_queries_per_client_task),
                ("test", minimum_test_queries_per_client_task),
            ):
                counts = torch.bincount(
                    owner[heldout[task][split]], minlength=num_clients
                )
                support_ok &= bool((counts >= minimum).all())
        if not support_ok:
            continue
        diagnostics = community_diagnostics(
            owner, micro_ids, topology, num_clients=num_clients
        )
        score: tuple[object, ...] = (*objective_tuple(diagnostics), tuple(raw))
        if best is None or score < best[0]:
            best = (score, owner)
    if best is None:
        return TinyOracleResult(False, None, None, None)
    return TinyOracleResult(True, best[1], None, best[0][:-1])


def edge_community_dispersion(
    selected_by_client_group: dict[int, dict[int, torch.Tensor]],
    endpoints: torch.Tensor,
    micro_ids: torch.Tensor,
    *,
    num_clients: int,
) -> int:
    """Measure selected-edge endpoint mass outside each micro's dominant client."""

    num_micros = int(micro_ids.max()) + 1
    mass = torch.zeros((num_micros, num_clients), dtype=torch.long)
    for client, groups in selected_by_client_group.items():
        for query_ids in groups.values():
            if query_ids.numel() == 0:
                continue
            edge_endpoints = endpoints[query_ids]
            mass[:, client].index_add_(
                0,
                micro_ids[edge_endpoints[:, 0]],
                torch.ones(query_ids.numel(), dtype=torch.long),
            )
            mass[:, client].index_add_(
                0,
                micro_ids[edge_endpoints[:, 1]],
                torch.ones(query_ids.numel(), dtype=torch.long),
            )
    return int((mass.sum(1) - mass.max(1).values).sum())


def _selection_products(
    owner: torch.Tensor,
    query_ids: torch.Tensor,
    endpoints: torch.Tensor,
    groups: torch.Tensor,
    quota: torch.Tensor,
) -> Iterable[dict[int, dict[int, torch.Tensor]]]:
    cells: list[tuple[int, int, list[tuple[int, ...]]]] = []
    for client in range(quota.shape[0]):
        for group in range(quota.shape[1]):
            required = int(quota[client, group])
            candidates = query_ids[
                (groups == group)
                & (owner[endpoints[:, 0]] == client)
                & (owner[endpoints[:, 1]] == client)
            ].tolist()
            if len(candidates) < required:
                return
            cells.append(
                (client, group, list(itertools.combinations(candidates, required)))
            )
    for choices in itertools.product(*(cell[2] for cell in cells)):
        selected = {
            client: {
                group: torch.empty(0, dtype=torch.long)
                for group in range(quota.shape[1])
            }
            for client in range(quota.shape[0])
        }
        for (client, group, _), ids in zip(cells, choices):
            selected[client][group] = torch.tensor(ids, dtype=torch.long)
        yield selected


def brute_force_lc_partition(
    *,
    topology: WeightedLogicalTopology,
    micro_ids: torch.Tensor,
    query_ids: torch.Tensor,
    query_endpoints: torch.Tensor,
    query_groups: torch.Tensor,
    quota: torch.Tensor,
    capacity_lower: int,
    capacity_upper: int,
    heldout_query_ids_by_task_split: Mapping[int, Mapping[str, torch.Tensor]] | None = None,
    minimum_validation_queries_per_client_task: int = 0,
    minimum_test_queries_per_client_task: int = 0,
    maximum_nodes: int = 9,
) -> TinyOracleResult:
    """Enumerate tiny LC owners and query choices exactly."""

    if topology.num_nodes > maximum_nodes:
        raise ValueError("LC brute-force oracle is restricted to tiny graphs.")
    num_clients = quota.shape[0]
    endpoints = query_endpoints[query_ids]
    best: tuple[
        tuple[object, ...], torch.Tensor, dict[int, dict[int, torch.Tensor]]
    ] | None = None
    for raw in itertools.product(range(num_clients), repeat=topology.num_nodes):
        owner = torch.tensor(raw, dtype=torch.long)
        counts = torch.bincount(owner, minlength=num_clients)
        if bool((counts < capacity_lower).any()) or bool((counts > capacity_upper).any()):
            continue
        heldout = heldout_query_ids_by_task_split or {}
        support_ok = True
        for task in heldout:
            for split, minimum in (
                ("val", minimum_validation_queries_per_client_task),
                ("test", minimum_test_queries_per_client_task),
            ):
                heldout_endpoints = query_endpoints[heldout[task][split]]
                internal = owner[heldout_endpoints[:, 0]].clone()
                internal[
                    owner[heldout_endpoints[:, 0]] != owner[heldout_endpoints[:, 1]]
                ] = -1
                internal_counts = torch.bincount(
                    internal[internal >= 0], minlength=num_clients
                )
                support_ok &= bool((internal_counts >= minimum).all())
        if not support_ok:
            continue
        node_diagnostics = community_diagnostics(
            owner, micro_ids, topology, num_clients=num_clients
        )
        for selected in _selection_products(
            owner,
            query_ids,
            endpoints,
            query_groups,
            quota,
        ):
            dispersion = edge_community_dispersion(
                selected,
                query_endpoints,
                micro_ids,
                num_clients=num_clients,
            )
            selected_tuple = tuple(
                tuple(selected[client][group].tolist())
                for client in range(num_clients)
                for group in range(quota.shape[1])
            )
            score: tuple[object, ...] = (
                dispersion,
                *objective_tuple(node_diagnostics),
                tuple(raw),
                selected_tuple,
            )
            if best is None or score < best[0]:
                best = (score, owner, selected)
    if best is None:
        return TinyOracleResult(False, None, None, None)
    return TinyOracleResult(True, best[1], best[2], best[0][:-2])
