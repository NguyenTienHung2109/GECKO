"""Graph-only deterministic ownership for non-training nodes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch


DIRECT_NONTRAIN_POLICY_VERSION = "balanced_neighbor_affinity_v1"


@dataclass(frozen=True)
class DirectGlobalAssignment:
    complete_owner: torch.Tensor
    train_owner_hash: str
    frozen_nontrain_owner_hash: str
    complete_owner_hash: str
    nontrain_node_ids: torch.Tensor
    node_counts: torch.Tensor
    capacity_lower: int
    capacity_upper: int
    assignment_passes: int
    policy_version: str = DIRECT_NONTRAIN_POLICY_VERSION


def owner_content_hash(owner: torch.Tensor) -> str:
    values = owner.detach().cpu().long().contiguous()
    return hashlib.sha256(values.numpy().tobytes()).hexdigest()


def _subset_hash(node_ids: torch.Tensor, owners: torch.Tensor) -> str:
    payload = {
        "version": DIRECT_NONTRAIN_POLICY_VERSION,
        "node_ids": node_ids.tolist(),
        "owners": owners.tolist(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _adjacency(
    logical_edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    *,
    num_nodes: int,
    directed: bool,
) -> list[list[tuple[int, float]]]:
    edges = logical_edge_index.detach().cpu().long()
    weights = edge_weights.detach().cpu().double()
    if edges.ndim != 2 or edges.shape[0] != 2 or weights.shape != (edges.shape[1],):
        raise ValueError("Logical edges and weights must align.")
    if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("Logical edge weights must be finite and nonnegative.")
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for index, (left, right) in enumerate(edges.T.tolist()):
        weight = float(weights[index])
        adjacency[left].append((right, weight))
        if not directed:
            adjacency[right].append((left, weight))
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: item[0])
    return adjacency


def assign_nontrain_nodes_balanced_neighbor_affinity(
    train_owner_by_global_node: torch.Tensor,
    logical_edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    *,
    num_clients: int,
    directed: bool,
    client_size_tolerance: float,
    fixed_pass_count: int = 3,
) -> DirectGlobalAssignment:
    """Complete ownership without reading any non-training label."""

    initial = train_owner_by_global_node.detach().cpu().long().clone()
    num_nodes = initial.numel()
    if num_clients < 1 or not 0 <= client_size_tolerance < 1:
        raise ValueError("Invalid client count or size tolerance.")
    if fixed_pass_count < 1:
        raise ValueError("fixed_pass_count must be positive.")
    train_mask = initial >= 0
    if bool((initial[train_mask] >= num_clients).any()):
        raise ValueError("A training owner is outside the client range.")
    nontrain = torch.nonzero(~train_mask, as_tuple=False).flatten()
    train_ids = torch.nonzero(train_mask, as_tuple=False).flatten()
    train_hash = _subset_hash(train_ids, initial[train_ids])
    owner = initial.clone()
    counts = torch.bincount(owner[train_mask], minlength=num_clients)
    ideal = num_nodes / num_clients
    lower = math.floor((1.0 - client_size_tolerance) * ideal)
    upper = math.ceil((1.0 + client_size_tolerance) * ideal)
    if bool((counts > upper).any()):
        raise ValueError("Training ownership already exceeds total-node capacity upper bound.")
    adjacency = _adjacency(
        logical_edge_index,
        edge_weights,
        num_nodes=num_nodes,
        directed=directed,
    )
    pending = nontrain.tolist()
    passes_used = 0
    for pass_index in range(fixed_pass_count):
        passes_used = pass_index + 1
        next_pending = []
        for node in pending:
            affinity = torch.zeros(num_clients, dtype=torch.float64)
            for neighbor, weight in adjacency[node]:
                client = int(owner[neighbor])
                if client >= 0:
                    affinity[client] += weight
            available = [client for client in range(num_clients) if counts[client] < upper]
            if not available:
                raise ValueError("No client has capacity for a non-training node.")
            if float(affinity.max()) == 0.0 and pass_index + 1 < fixed_pass_count:
                next_pending.append(node)
                continue
            client = min(
                available,
                key=lambda candidate: (
                    -float(affinity[candidate]),
                    int(counts[candidate]),
                    candidate,
                    node,
                ),
            )
            owner[node] = client
            counts[client] += 1
        pending = next_pending
        if not pending:
            break
    if pending:
        raise AssertionError("Fixed non-training assignment passes left nodes pending.")

    # Deterministic graph-only repair of lower capacity bounds, if necessary.
    for _ in range(fixed_pass_count):
        recipients = [client for client in range(num_clients) if counts[client] < lower]
        if not recipients:
            break
        changed = False
        for recipient in recipients:
            while counts[recipient] < lower:
                choices = []
                for node in nontrain.tolist():
                    donor = int(owner[node])
                    if donor == recipient or counts[donor] <= lower:
                        continue
                    affinity = torch.zeros(num_clients, dtype=torch.float64)
                    for neighbor, weight in adjacency[node]:
                        client = int(owner[neighbor])
                        if client >= 0:
                            affinity[client] += weight
                    loss = float(affinity[donor] - affinity[recipient])
                    choices.append((loss, node, donor))
                if not choices:
                    raise ValueError("Cannot satisfy total-node lower capacity without moving train nodes.")
                _, node, donor = min(choices)
                owner[node] = recipient
                counts[donor] -= 1
                counts[recipient] += 1
                changed = True
        if not changed:
            break
    if bool((counts < lower).any()) or bool((counts > upper).any()):
        raise ValueError("Graph-only non-training assignment violates total-node capacity bounds.")
    if not torch.equal(owner[train_mask], initial[train_mask]):
        raise AssertionError("Non-training assignment changed training ownership.")
    if bool((owner < 0).any()) or bool((owner >= num_clients).any()):
        raise AssertionError("Complete ownership contains an invalid owner.")
    return DirectGlobalAssignment(
        complete_owner=owner,
        train_owner_hash=train_hash,
        frozen_nontrain_owner_hash=_subset_hash(nontrain, owner[nontrain]),
        complete_owner_hash=owner_content_hash(owner),
        nontrain_node_ids=nontrain,
        node_counts=counts,
        capacity_lower=lower,
        capacity_upper=upper,
        assignment_passes=passes_used,
    )
