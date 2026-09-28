"""Shared primitives for quota-first soft-community ownership."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any
from typing import Mapping
from typing import Sequence

import networkx as nx
import torch

from gecko.data.partitioning.base import CSRRows
from gecko.data.partitioning.base import WeightedLogicalTopology
from gecko.data.partitioning.base import evaluate_hconn


SOFT_COMMUNITY_MICRO_VERSION = "soft_community_louvain_metis_micro_v2"


def payload_hash(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for a JSON scientific payload."""

    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and canonical CPU bytes."""

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def stable_int(*parts: object) -> int:
    """Map deterministic tie-break material to one unsigned integer."""

    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


@dataclass(frozen=True)
class SoftMicroCommunityConfig:
    """Topology-only microcommunity construction parameters."""

    seed: int
    maximum_microcommunity_nodes: int = 4096
    louvain_resolution: float = 1.0

    def validate(self) -> None:
        if self.maximum_microcommunity_nodes < 1:
            raise ValueError("maximum_microcommunity_nodes must be positive.")
        if not math.isfinite(self.louvain_resolution) or self.louvain_resolution <= 0:
            raise ValueError("louvain_resolution must be positive and finite.")


@dataclass(frozen=True)
class MicroCommunityResult:
    """One immutable topology-only microcommunity map."""

    micro_ids: torch.Tensor
    communities: tuple[tuple[int, ...], ...]
    diagnostics: dict[str, Any]


def canonicalize_micro_ids(
    micro_ids: torch.Tensor, *, num_nodes: int
) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
    """Canonicalize arbitrary community IDs by their sorted node tuples."""

    values = micro_ids.detach().cpu().long().flatten()
    if values.shape != (num_nodes,) or bool((values < 0).any()):
        raise ValueError("micro_ids must contain one nonnegative ID per node.")
    communities = [
        tuple(torch.nonzero(values == raw, as_tuple=False).flatten().tolist())
        for raw in torch.unique(values, sorted=True).tolist()
    ]
    if any(not nodes for nodes in communities):
        raise AssertionError("A canonical microcommunity cannot be empty.")
    communities.sort(key=lambda nodes: (nodes[0], len(nodes), nodes))
    canonical = torch.full((num_nodes,), -1, dtype=torch.long)
    for community_id, nodes in enumerate(communities):
        canonical[torch.tensor(nodes, dtype=torch.long)] = community_id
    if bool((canonical < 0).any()):
        raise AssertionError("Microcommunities do not cover every node.")
    return canonical, tuple(communities)


def micro_result_from_ids(
    micro_ids: torch.Tensor,
    *,
    num_nodes: int,
    provenance: str = "injected_test_fixture",
) -> MicroCommunityResult:
    """Build a validated microcommunity result from precomputed IDs."""

    canonical, communities = canonicalize_micro_ids(
        micro_ids, num_nodes=num_nodes
    )
    digest = tensor_hash(canonical)
    return MicroCommunityResult(
        micro_ids=canonical,
        communities=communities,
        diagnostics={
            "microcommunity_version": SOFT_COMMUNITY_MICRO_VERSION,
            "microcommunity_backend": provenance,
            "microcommunity_count": len(communities),
            "largest_microcommunity_size": max(map(len, communities), default=0),
            "microcommunity_hash": digest,
            "label_access_policy": "topology_only",
        },
    )


def _weighted_graph(topology: WeightedLogicalTopology) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(topology.num_nodes))
    graph.add_weighted_edges_from(
        (int(left), int(right), float(weight))
        for (left, right), weight in zip(
            topology.edge_index.T.tolist(), topology.edge_weights.tolist()
        )
    )
    return graph


def _split_connected_atom(
    nodes: Sequence[int],
    graph: nx.Graph,
    *,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    try:
        import pymetis  # type: ignore
    except ImportError as error:  # pragma: no cover - canonical env ships it
        raise RuntimeError(
            "Soft-community v2 requires PyMetis; no fallback may change the partition."
        ) from error
    ordered = tuple(sorted(int(node) for node in nodes))
    local = {node: index for index, node in enumerate(ordered)}
    adjacency = [
        sorted(local[n] for n in graph.neighbors(node) if n in local)
        for node in ordered
    ]
    _, membership = pymetis.part_graph(
        2,
        adjacency=adjacency,
        recursive=True,
        options=pymetis.Options(seed=seed % (2**31), contig=True),
    )
    children: list[tuple[int, ...]] = []
    for part in (0, 1):
        selected = [ordered[index] for index, value in enumerate(membership) if value == part]
        if not selected:
            raise RuntimeError("PyMetis returned an empty soft-community child.")
        children.extend(
            tuple(sorted(component))
            for component in nx.connected_components(graph.subgraph(selected))
        )
    return tuple(children)


def build_microcommunities(
    topology: WeightedLogicalTopology,
    config: SoftMicroCommunityConfig,
) -> MicroCommunityResult:
    """Generate connected topology-only micros without assigning any client."""

    config.validate()
    graph = _weighted_graph(topology)
    # Louvain's modularity objective is undefined when the graph has no edge
    # mass.  In that case singleton micros are the only topology-derived,
    # deterministic answer and require no backend-specific fallback.
    raw = (
        [{node} for node in range(topology.num_nodes)]
        if graph.number_of_edges() == 0
        else nx.community.louvain_communities(
            graph,
            seed=config.seed,
            resolution=config.louvain_resolution,
            weight="weight",
        )
    )
    pending: list[tuple[int, ...]] = []
    for community in raw:
        pending.extend(
            tuple(sorted(component))
            for component in nx.connected_components(graph.subgraph(community))
        )
    finished: list[tuple[int, ...]] = []
    split_index = 0
    while pending:
        nodes = pending.pop(0)
        if len(nodes) <= config.maximum_microcommunity_nodes or len(nodes) == 1:
            finished.append(nodes)
            continue
        children = _split_connected_atom(
            nodes,
            graph,
            seed=config.seed + 1_000_003 * split_index,
        )
        split_index += 1
        pending.extend(sorted(children, key=lambda value: (value[0], len(value))))
    ids = torch.full((topology.num_nodes,), -1, dtype=torch.long)
    for raw_id, nodes in enumerate(finished):
        ids[torch.tensor(nodes, dtype=torch.long)] = raw_id
    result = micro_result_from_ids(
        ids,
        num_nodes=topology.num_nodes,
        provenance=(
            "topology_singletons_edgeless"
            if graph.number_of_edges() == 0
            else "networkx_louvain_recursive_pymetis"
        ),
    )
    diagnostics = {
        **result.diagnostics,
        "topology_hash": topology.topology_hash,
        "community_seed": config.seed,
        "louvain_resolution": config.louvain_resolution,
        "maximum_microcommunity_nodes": config.maximum_microcommunity_nodes,
        "recursive_metis_split_count": split_index,
    }
    return MicroCommunityResult(
        micro_ids=result.micro_ids,
        communities=result.communities,
        diagnostics=diagnostics,
    )


def build_large_metis_microcommunities(
    topology: WeightedLogicalTopology,
    config: SoftMicroCommunityConfig,
) -> MicroCommunityResult:
    """Generate bounded topology-only micros without a NetworkX edge mirror.

    PyMetis consumes the compact CSR directly.  The target is deliberately
    half the configured maximum so ordinary balance variation remains below
    the hard size bound; an oversized result is retried with more partitions
    and fails closed after three deterministic attempts.
    """

    config.validate()
    if not isinstance(topology.incident_neighbors, CSRRows):
        raise TypeError("Large METIS construction requires compact CSR neighbors.")
    try:
        import pymetis  # type: ignore
    except ImportError as error:  # pragma: no cover - canonical env ships it
        raise RuntimeError("Large soft-community construction requires PyMetis.") from error
    neighbors = topology.incident_neighbors
    target_size = max(1, config.maximum_microcommunity_nodes // 2)
    partition_count = max(2, math.ceil(topology.num_nodes / target_size))
    membership: list[int] | None = None
    largest = topology.num_nodes
    attempts = 0
    for attempts in range(1, 4):
        _, raw_membership = pymetis.part_graph(
            partition_count,
            xadj=neighbors.offsets.numpy(),
            adjncy=neighbors.values.numpy(),
            recursive=True,
            options=pymetis.Options(
                seed=config.seed % (2**31),
                contig=True,
            ),
        )
        candidate = torch.tensor(raw_membership, dtype=torch.long)
        sizes = torch.bincount(candidate, minlength=partition_count)
        largest = int(sizes.max()) if sizes.numel() else 0
        membership = raw_membership
        if largest <= config.maximum_microcommunity_nodes:
            break
        partition_count = max(
            partition_count + 1,
            math.ceil(partition_count * largest / config.maximum_microcommunity_nodes),
        )
    if membership is None or largest > config.maximum_microcommunity_nodes:
        raise RuntimeError(
            "Direct METIS could not satisfy maximum_microcommunity_nodes "
            f"after {attempts} attempts (largest={largest})."
        )
    result = micro_result_from_ids(
        torch.tensor(membership, dtype=torch.long),
        num_nodes=topology.num_nodes,
        provenance="pymetis_direct_csr_large_graph",
    )
    return MicroCommunityResult(
        micro_ids=result.micro_ids,
        communities=result.communities,
        diagnostics={
            **result.diagnostics,
            "topology_hash": topology.topology_hash,
            "community_seed": config.seed,
            "louvain_resolution": None,
            "maximum_microcommunity_nodes": config.maximum_microcommunity_nodes,
            "recursive_metis_split_count": attempts,
            "direct_metis_partition_count": len(result.communities),
            "large_graph_backend_reason": "avoid_networkx_OE_python_object_mirror",
        },
    )


def balanced_total_capacities(
    num_nodes: int,
    minimum_counts: torch.Tensor,
    *,
    client_size_tolerance: float,
) -> torch.Tensor:
    """Create exact balanced total-node capacities above fixed assignments."""

    minimum = minimum_counts.detach().cpu().long().flatten()
    num_clients = minimum.numel()
    if num_clients < 1 or num_nodes < int(minimum.sum()):
        raise ValueError("Fixed ownership exceeds the total node supply.")
    if not 0 <= client_size_tolerance < 1:
        raise ValueError("client_size_tolerance must lie in [0, 1).")
    ideal = num_nodes / num_clients
    lower = math.floor((1.0 - client_size_tolerance) * ideal)
    upper = math.ceil((1.0 + client_size_tolerance) * ideal)
    if bool((minimum > upper).any()):
        raise ValueError("Fixed ownership exceeds a client upper capacity.")
    capacities = minimum.clone()
    for _ in range(num_nodes - int(capacities.sum())):
        available = torch.nonzero(capacities < upper, as_tuple=False).flatten()
        if available.numel() == 0:
            raise ValueError("No exact total-node capacity vector satisfies the upper bound.")
        client = min(available.tolist(), key=lambda value: (int(capacities[value]), value))
        capacities[client] += 1
    if bool((capacities < lower).any()) or int(capacities.sum()) != num_nodes:
        raise ValueError("No exact total-node capacity vector satisfies the lower bound.")
    return capacities


def initial_neighbor_owner_mass(
    owner: torch.Tensor,
    topology: WeightedLogicalTopology,
    *,
    num_clients: int,
) -> torch.Tensor:
    """Return exact assigned-neighbor weight for every node/client pair."""

    values = owner.detach().cpu().long()
    edges = topology.edge_index
    weights = topology.edge_weights
    result = torch.zeros(
        (topology.num_nodes, num_clients), dtype=torch.float64
    )
    flat = result.view(-1)
    left_owner = values[edges[0]]
    right_owner = values[edges[1]]
    right_assigned = right_owner >= 0
    if bool(right_assigned.any()):
        encoded = edges[0, right_assigned] * num_clients + right_owner[right_assigned]
        flat.index_add_(0, encoded, weights[right_assigned])
    left_assigned = left_owner >= 0
    if bool(left_assigned.any()):
        encoded = edges[1, left_assigned] * num_clients + left_owner[left_assigned]
        flat.index_add_(0, encoded, weights[left_assigned])
    return result


def complete_owner_soft(
    partial_owner: torch.Tensor,
    micro_ids: torch.Tensor,
    topology: WeightedLogicalTopology,
    capacities: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Assign free nodes individually using community and neighbor affinity."""

    owner = partial_owner.detach().cpu().long().clone()
    micros, _ = canonicalize_micro_ids(micro_ids, num_nodes=topology.num_nodes)
    caps = capacities.detach().cpu().long().flatten()
    num_clients = caps.numel()
    fixed = owner >= 0
    if owner.shape != (topology.num_nodes,) or bool((owner[fixed] >= num_clients).any()):
        raise ValueError("partial_owner is invalid.")
    counts = torch.bincount(owner[fixed], minlength=num_clients)
    if bool((counts > caps).any()) or int(caps.sum()) != topology.num_nodes:
        raise ValueError("Partial ownership is incompatible with total capacities.")
    num_micros = int(micros.max()) + 1
    micro_mass = torch.zeros((num_micros, num_clients), dtype=torch.long)
    if bool(fixed.any()):
        encoded = micros[fixed] * num_clients + owner[fixed]
        micro_mass.view(-1).index_add_(
            0, encoded, torch.ones(int(fixed.sum()), dtype=torch.long)
        )
    neighbor_owner_mass = initial_neighbor_owner_mass(
        owner, topology, num_clients=num_clients
    )
    pending = torch.nonzero(~fixed, as_tuple=False).flatten().tolist()
    pending.sort(key=lambda node: (int(micros[node]), stable_int(seed, "free", node)))
    for _ in range(3):
        deferred: list[int] = []
        for node in pending:
            available = [client for client in range(num_clients) if counts[client] < caps[client]]
            if not available:
                raise ValueError("No client has capacity for a free node.")
            neighbor_mass = neighbor_owner_mass[node]
            community = int(micros[node])
            if (
                int(micro_mass[community].sum()) == 0
                and float(neighbor_mass.max()) == 0.0
                and len(pending) != 1
            ):
                deferred.append(node)
                continue
            client = min(
                available,
                key=lambda candidate: (
                    -int(micro_mass[community, candidate]),
                    -float(neighbor_mass[candidate]),
                    int(counts[candidate]),
                    stable_int(seed, "client", node, candidate),
                ),
            )
            owner[node] = client
            counts[client] += 1
            micro_mass[community, client] += 1
            neighbors = topology.incident_neighbors[node]
            if neighbors.numel():
                neighbor_owner_mass[:, client].index_add_(
                    0, neighbors, topology.incident_weights[node]
                )
        pending = deferred
        if not pending:
            break
    for node in pending:
        available = [client for client in range(num_clients) if counts[client] < caps[client]]
        client = min(
            available,
            key=lambda candidate: (
                -int(micro_mass[int(micros[node]), candidate]),
                int(counts[candidate]),
                stable_int(seed, "fallback", node, candidate),
            ),
        )
        owner[node] = client
        counts[client] += 1
        micro_mass[int(micros[node]), client] += 1
        neighbors = topology.incident_neighbors[node]
        if neighbors.numel():
            neighbor_owner_mass[:, client].index_add_(
                0, neighbors, topology.incident_weights[node]
            )
    if bool((owner < 0).any()) or not torch.equal(counts, caps):
        raise AssertionError("Soft completion did not realize exact capacities.")
    return owner


def community_diagnostics(
    owner: torch.Tensor,
    micro_ids: torch.Tensor,
    topology: WeightedLogicalTopology,
    *,
    num_clients: int,
) -> dict[str, Any]:
    """Measure finite community dispersion and topology cut."""

    values = owner.detach().cpu().long()
    micros, communities = canonicalize_micro_ids(
        micro_ids, num_nodes=topology.num_nodes
    )
    owner_counts: list[int] = []
    retained = 0
    split_mass = 0
    owner_excess = 0
    for community_id, nodes in enumerate(communities):
        counts = torch.bincount(
            values[micros == community_id], minlength=num_clients
        )
        used = int((counts > 0).sum())
        owner_counts.append(used)
        majority = int(counts.max())
        retained += majority
        split_mass += len(nodes) - majority
        owner_excess += max(0, used - 1)
    connectivity = evaluate_hconn(values, topology, num_clients=num_clients)
    return {
        "community_split_mass": split_mass,
        "community_fragment_excess": owner_excess,
        "fragmented_microcommunity_count": sum(value > 1 for value in owner_counts),
        "maximum_owners_per_microcommunity": max(owner_counts, default=0),
        "weighted_microcommunity_purity": retained / max(1, topology.num_nodes),
        "microcommunity_owner_counts": owner_counts,
        "context_cut_edge_weight": connectivity.cut_edge_weight,
        "h_conn": connectivity.h_conn,
    }


def objective_tuple(diagnostics: Mapping[str, Any]) -> tuple[int, int, float]:
    """Return the frozen lexicographic node-community objective."""

    return (
        int(diagnostics["community_split_mass"]),
        int(diagnostics["community_fragment_excess"]),
        float(diagnostics["context_cut_edge_weight"]),
    )
