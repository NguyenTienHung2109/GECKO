from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any
from typing import Literal
import torch
from gecko.types import PartitionResult

GRID_VERSION = "nc_class_dirichlet_order_v2"


DIRECT_FACTORIAL_STREAM_VERSION = "direct_factorial_strict_local_stream_v1"


LC_COMMUNITY_VERSION = "lc_community_exact_edge_query_dirichlet_v1"


LC_COMMUNITY_STREAM_VERSION = "lc_community_selected_query_stream_v1"


DIRECT_HCONN_VERSION = "weighted_logical_edge_cut_direct_v1"


class CSRRows(Sequence[torch.Tensor]):
    """Read-only row views over one compact CSR value tensor."""

    def __init__(self, values: torch.Tensor, offsets: torch.Tensor) -> None:
        self._values = values
        self._offsets = offsets

    def __len__(self) -> int:
        return int(self._offsets.numel()) - 1

    @property
    def values(self) -> torch.Tensor:
        """Return the immutable compact value tensor."""

        return self._values

    @property
    def offsets(self) -> torch.Tensor:
        """Return the immutable CSR offsets."""

        return self._offsets

    def __getitem__(
        self, index: int | slice
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = int(self._offsets[index])
        stop = int(self._offsets[index + 1])
        return self._values[start:stop]

    def __iter__(self) -> Iterator[torch.Tensor]:
        for index in range(len(self)):
            yield self[index]


class ImplicitUnitCSRRows(Sequence[torch.Tensor]):
    """CSR rows whose every stored value is the exact scalar one.

    Large unweighted graphs do not need an additional float64 tensor parallel
    to their adjacency.  Rows are materialized only when an algorithm asks for
    one node's incident weights.
    """

    def __init__(self, offsets: torch.Tensor) -> None:
        self._offsets = offsets.detach().cpu().long().contiguous()

    def __len__(self) -> int:
        return int(self._offsets.numel()) - 1

    def __getitem__(
        self, index: int | slice
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        count = int(self._offsets[index + 1] - self._offsets[index])
        return torch.ones(count, dtype=torch.float64)


class EmptyCSRRows(Sequence[torch.Tensor]):
    """Fixed-length sequence of empty int64 rows."""

    def __init__(self, num_rows: int) -> None:
        self._num_rows = int(num_rows)

    def __len__(self) -> int:
        return self._num_rows

    def __getitem__(
        self, index: int | slice
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return torch.empty(0, dtype=torch.long)


class UnitPairWeights(Mapping[tuple[int, int], float]):
    """Unit-weight pair lookup over lexicographically sorted logical edges."""

    def __init__(self, edges: torch.Tensor, source_offsets: torch.Tensor) -> None:
        self._edges = edges
        self._source_offsets = source_offsets.detach().cpu().long().contiguous()

    def __len__(self) -> int:
        return int(self._edges.shape[1])

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for index in range(len(self)):
            yield (int(self._edges[0, index]), int(self._edges[1, index]))

    def __getitem__(self, pair: tuple[int, int]) -> float:
        source, target = min(pair), max(pair)
        if source < 0 or source + 1 >= self._source_offsets.numel():
            raise KeyError(pair)
        start = int(self._source_offsets[source])
        stop = int(self._source_offsets[source + 1])
        row = self._edges[1, start:stop]
        position = int(torch.searchsorted(row, target))
        if position >= row.numel() or int(row[position]) != target:
            raise KeyError(pair)
        return 1.0


class SortedPairWeights(Mapping[tuple[int, int], float]):
    """Compact immutable mapping backed by sorted pair keys and weights."""

    def __init__(self, keys: torch.Tensor, weights: torch.Tensor, num_nodes: int) -> None:
        self._keys = keys.detach().cpu().long().contiguous()
        self._weights = weights.detach().cpu().double().contiguous()
        self._num_nodes = int(num_nodes)

    def __len__(self) -> int:
        return int(self._keys.numel())

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for key in self._keys.tolist():
            yield (key // self._num_nodes, key % self._num_nodes)

    def __getitem__(self, pair: tuple[int, int]) -> float:
        key = min(pair) * self._num_nodes + max(pair)
        position = int(torch.searchsorted(self._keys, key))
        if position >= len(self) or int(self._keys[position]) != key:
            raise KeyError(pair)
        return float(self._weights[position])


@dataclass(frozen=True)
class WeightedLogicalTopology:
    edge_index: torch.Tensor
    edge_weights: torch.Tensor
    num_nodes: int
    directed: bool
    representation_id: str
    topology_hash: str
    raw_edge_count: int
    logical_edge_count: int
    total_edge_weight: float
    incident_edge_ids: Sequence[torch.Tensor]
    incident_neighbors: Sequence[torch.Tensor]
    incident_weights: Sequence[torch.Tensor]
    pair_incident_weight: Mapping[tuple[int, int], float]
    version: str = DIRECT_HCONN_VERSION


@dataclass(frozen=True)
class HConnMetrics:
    h_conn: float
    cut_edge_weight: float
    total_edge_weight: float
    boundary_node_ratio: float
    average_retained_degree_ratio: float
    per_client_cut_exposure: list[float | None]
    per_client_retained_degree_ratio: list[float | None]
    version: str = DIRECT_HCONN_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HConnMetrics":
        if payload.get("version") != DIRECT_HCONN_VERSION:
            raise ValueError("Unsupported H_conn metric version.")
        return cls(**payload)


def build_weighted_logical_topology(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    directed: bool,
    edge_weights: torch.Tensor | None = None,
    representation_id: str,
    backend: Literal["auto", "python", "tensor", "cuda"] = "auto",
) -> WeightedLogicalTopology:
    """Canonicalize loop-free logical edges and validate duplicate weights."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if backend not in {"auto", "python", "tensor", "cuda"}:
        raise ValueError(f"Unsupported topology canonicalization backend: {backend}.")
    if backend == "auto":
        # CUDA wins decisively once launch overhead is amortized.  Compact CPU
        # tensor canonicalization is faster for small fixtures and retains the
        # same exact scientific object on hosts without CUDA.
        backend = (
            "cuda"
            if torch.cuda.is_available() and edge_index.shape[-1] >= 100_000
            else "tensor"
        )
    if backend != "python":
        return _build_tensor_logical_topology(
            edge_index,
            num_nodes=num_nodes,
            directed=directed,
            edge_weights=edge_weights,
            representation_id=representation_id,
            device=torch.device("cuda" if backend == "cuda" else "cpu"),
        )

    edges = edge_index.detach().cpu().long()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes):
        raise ValueError("edge_index references a node outside the graph.")
    weights = (
        torch.ones(edges.shape[1], dtype=torch.float64)
        if edge_weights is None
        else edge_weights.detach().cpu().double().flatten()
    )
    if weights.shape != (edges.shape[1],):
        raise ValueError("edge_weights must align with raw edges.")
    if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("edge weights must be finite and nonnegative.")
    grouped: dict[tuple[int, int], float] = {}
    for index, (source, target) in enumerate(edges.T.tolist()):
        if source == target:
            continue
        key = (source, target) if directed else (min(source, target), max(source, target))
        weight = float(weights[index])
        previous = grouped.get(key)
        if previous is not None and abs(previous - weight) > 1e-12:
            raise ValueError("Duplicate/reverse logical arcs have inconsistent weights.")
        grouped[key] = weight
    keys = sorted(grouped)
    logical_edges = (
        torch.tensor(keys, dtype=torch.long).T.contiguous()
        if keys
        else torch.empty((2, 0), dtype=torch.long)
    )
    logical_weights = torch.tensor(
        [grouped[key] for key in keys], dtype=torch.float64
    )
    edge_ids: list[list[int]] = [[] for _ in range(num_nodes)]
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    incident_weights: list[list[float]] = [[] for _ in range(num_nodes)]
    pair_incident_weight: dict[tuple[int, int], float] = {}
    for edge_id, (source, target) in enumerate(keys):
        weight = grouped[(source, target)]
        edge_ids[source].append(edge_id)
        neighbors[source].append(target)
        incident_weights[source].append(weight)
        edge_ids[target].append(edge_id)
        neighbors[target].append(source)
        incident_weights[target].append(weight)
        pair = (min(source, target), max(source, target))
        pair_incident_weight[pair] = pair_incident_weight.get(pair, 0.0) + weight
    digest = hashlib.sha256()
    digest.update(str(num_nodes).encode("ascii"))
    digest.update(str(int(directed)).encode("ascii"))
    digest.update(representation_id.encode("utf-8"))
    digest.update(logical_edges.numpy().tobytes())
    digest.update(logical_weights.numpy().tobytes())
    return WeightedLogicalTopology(
        edge_index=logical_edges,
        edge_weights=logical_weights,
        num_nodes=num_nodes,
        directed=directed,
        representation_id=representation_id,
        topology_hash=digest.hexdigest(),
        raw_edge_count=edges.shape[1],
        logical_edge_count=len(keys),
        total_edge_weight=float(logical_weights.sum()),
        incident_edge_ids=tuple(torch.tensor(row, dtype=torch.long) for row in edge_ids),
        incident_neighbors=tuple(torch.tensor(row, dtype=torch.long) for row in neighbors),
        incident_weights=tuple(torch.tensor(row, dtype=torch.float64) for row in incident_weights),
        pair_incident_weight=pair_incident_weight,
    )


def _digest_tensor_chunks(
    digest: "hashlib._Hash", value: torch.Tensor, *, elements: int = 1_048_576
) -> None:
    """Update a digest without allocating one bytes object for a huge tensor."""

    flat = value.detach().cpu().contiguous().view(-1)
    for start in range(0, flat.numel(), elements):
        digest.update(flat[start : start + elements].numpy().tobytes())


def build_large_bidirected_simple_topology(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    representation_id: str,
    device: torch.device | None = None,
) -> WeightedLogicalTopology:
    """Build an exact unweighted topology from a simple bidirected arc list.

    The ordinary canonicalizer materializes several float64 and 2E incidence
    arrays at once.  This path validates that every non-loop arc is unique and
    has exactly one reverse, then stores one sorted logical edge plus a compact
    neighbor CSR.  Incident edge IDs are intentionally omitted; callers must
    use algorithms that do not request edge-ID-local refinement.
    """

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    selected_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    edges = edge_index.detach().to(device=selected_device, dtype=torch.long)
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes):
        raise ValueError("edge_index references a node outside the graph.")
    raw_edge_count = int(edges.shape[1])
    nonloop = edges[0] != edges[1]
    sources = edges[0, nonloop]
    targets = edges[1, nonloop]
    directed_keys = sources * num_nodes + targets
    order = torch.argsort(directed_keys)
    directed_keys = directed_keys[order]
    if directed_keys.numel() and bool((directed_keys[1:] == directed_keys[:-1]).any()):
        raise ValueError("Large bidirected topology contains duplicate directed arcs.")
    sorted_sources = torch.div(directed_keys, num_nodes, rounding_mode="floor")
    sorted_targets = torch.remainder(directed_keys, num_nodes)
    reverse_keys = sorted_targets * num_nodes + sorted_sources
    reverse_positions = torch.searchsorted(directed_keys, reverse_keys)
    reverse_positions = reverse_positions.clamp_max(max(0, directed_keys.numel() - 1))
    if directed_keys.numel() and not bool(
        (directed_keys[reverse_positions] == reverse_keys).all()
    ):
        raise ValueError("Large topology is not exactly bidirected.")

    counts = torch.bincount(sorted_sources, minlength=num_nodes)
    offsets = torch.zeros(num_nodes + 1, dtype=torch.long, device=selected_device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    offsets_cpu = offsets.cpu().contiguous()
    neighbors_cpu = sorted_targets.cpu().contiguous()
    logical = sorted_sources < sorted_targets
    logical_edges = torch.stack(
        (sorted_sources[logical], sorted_targets[logical])
    ).cpu().contiguous()
    logical_count = int(logical_edges.shape[1])
    if 2 * logical_count != int(directed_keys.numel()):
        raise ValueError("Bidirected topology did not reduce to one edge per pair.")
    logical_weights = torch.ones(logical_count, dtype=torch.float64)
    logical_source_counts = torch.bincount(
        logical_edges[0], minlength=num_nodes
    )
    logical_source_offsets = torch.zeros(num_nodes + 1, dtype=torch.long)
    logical_source_offsets[1:] = torch.cumsum(logical_source_counts, dim=0)

    digest = hashlib.sha256()
    digest.update(str(num_nodes).encode("ascii"))
    digest.update(b"0")
    digest.update(representation_id.encode("utf-8"))
    _digest_tensor_chunks(digest, logical_edges)
    _digest_tensor_chunks(digest, logical_weights)
    topology = WeightedLogicalTopology(
        edge_index=logical_edges,
        edge_weights=logical_weights,
        num_nodes=num_nodes,
        directed=False,
        representation_id=representation_id,
        topology_hash=digest.hexdigest(),
        raw_edge_count=raw_edge_count,
        logical_edge_count=logical_count,
        total_edge_weight=float(logical_count),
        incident_edge_ids=EmptyCSRRows(num_nodes),
        incident_neighbors=CSRRows(neighbors_cpu, offsets_cpu),
        incident_weights=ImplicitUnitCSRRows(offsets_cpu),
        pair_incident_weight=UnitPairWeights(
            logical_edges, logical_source_offsets
        ),
    )
    del (
        edges,
        nonloop,
        sources,
        targets,
        order,
        directed_keys,
        sorted_sources,
        sorted_targets,
        reverse_keys,
        reverse_positions,
        counts,
        offsets,
        logical,
    )
    if selected_device.type == "cuda":
        torch.cuda.empty_cache()
    return topology


def _build_tensor_logical_topology(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    directed: bool,
    edge_weights: torch.Tensor | None,
    representation_id: str,
    device: torch.device,
) -> WeightedLogicalTopology:
    """Tensor sort/unique canonicalizer with compact CSR incidence storage."""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA topology canonicalization requires an available GPU.")
    edges = edge_index.detach().to(device=device, dtype=torch.long)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= num_nodes):
        raise ValueError("edge_index references a node outside the graph.")
    weights = (
        torch.ones(edges.shape[1], dtype=torch.float64, device=device)
        if edge_weights is None
        else edge_weights.detach().to(device=device, dtype=torch.float64).flatten()
    )
    if weights.shape != (edges.shape[1],):
        raise ValueError("edge_weights must align with raw edges.")
    if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("edge weights must be finite and nonnegative.")
    raw_edge_count = int(edges.shape[1])
    nonloop = edges[0] != edges[1]
    source = edges[0, nonloop]
    target = edges[1, nonloop]
    weights = weights[nonloop]
    if not directed:
        source, target = torch.minimum(source, target), torch.maximum(source, target)
    keys = source * num_nodes + target
    order = torch.argsort(keys, stable=True)
    keys = keys[order]
    weights = weights[order]
    if keys.numel():
        duplicate = keys[1:] == keys[:-1]
        if bool((duplicate & ((weights[1:] - weights[:-1]).abs() > 1e-12)).any()):
            raise ValueError("Duplicate/reverse logical arcs have inconsistent weights.")
        first = torch.ones(keys.numel(), dtype=torch.bool, device=device)
        first[1:] = ~duplicate
        keys = keys[first]
        weights = weights[first]
        del duplicate, first
    logical_edges_device = torch.stack(
        (torch.div(keys, num_nodes, rounding_mode="floor"), torch.remainder(keys, num_nodes))
    )
    logical_edges = logical_edges_device.cpu().contiguous()
    logical_weights = weights.cpu().contiguous()
    edge_count = int(keys.numel())
    del edges, source, target, order, nonloop
    if device.type == "cuda":
        torch.cuda.empty_cache()

    edge_ids = torch.arange(edge_count, dtype=torch.long, device=device)
    incidence_sources = torch.cat((logical_edges_device[0], logical_edges_device[1]))
    incidence_neighbors = torch.cat((logical_edges_device[1], logical_edges_device[0]))
    incidence_edge_ids = torch.cat((edge_ids, edge_ids))
    incidence_weights = torch.cat((weights, weights))
    incidence_keys = incidence_sources * (edge_count + 1) + incidence_edge_ids
    incidence_order = torch.argsort(incidence_keys, stable=True)
    counts = torch.bincount(incidence_sources, minlength=num_nodes)
    offsets = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    offsets_cpu = offsets.cpu()
    edge_rows = CSRRows(incidence_edge_ids[incidence_order].cpu(), offsets_cpu)
    neighbor_rows = CSRRows(incidence_neighbors[incidence_order].cpu(), offsets_cpu)
    weight_rows = CSRRows(incidence_weights[incidence_order].cpu(), offsets_cpu)

    if not directed:
        pair_keys = keys
        pair_weights = weights
    else:
        pair_keys = (
            torch.minimum(logical_edges_device[0], logical_edges_device[1])
            * num_nodes
            + torch.maximum(logical_edges_device[0], logical_edges_device[1])
        )
        pair_order = torch.argsort(pair_keys, stable=True)
        pair_keys = pair_keys[pair_order]
        pair_weights = weights[pair_order]
        if pair_keys.numel():
            pair_first = torch.ones(
                pair_keys.numel(), dtype=torch.bool, device=device
            )
            pair_first[1:] = pair_keys[1:] != pair_keys[:-1]
            group = torch.cumsum(pair_first.long(), dim=0) - 1
            aggregated = torch.zeros(
                int(group[-1]) + 1, dtype=torch.float64, device=device
            )
            aggregated.index_add_(0, group, pair_weights)
            pair_keys = pair_keys[pair_first]
            pair_weights = aggregated
    pair_incident_weight = SortedPairWeights(pair_keys, pair_weights, num_nodes)

    digest = hashlib.sha256()
    digest.update(str(num_nodes).encode("ascii"))
    digest.update(str(int(directed)).encode("ascii"))
    digest.update(representation_id.encode("utf-8"))
    digest.update(logical_edges.numpy().tobytes())
    digest.update(logical_weights.numpy().tobytes())
    return WeightedLogicalTopology(
        edge_index=logical_edges,
        edge_weights=logical_weights,
        num_nodes=num_nodes,
        directed=directed,
        representation_id=representation_id,
        topology_hash=digest.hexdigest(),
        raw_edge_count=raw_edge_count,
        logical_edge_count=edge_count,
        total_edge_weight=float(logical_weights.sum()),
        incident_edge_ids=edge_rows,
        incident_neighbors=neighbor_rows,
        incident_weights=weight_rows,
        pair_incident_weight=pair_incident_weight,
    )


def neighbor_owner_weight_counts(
    owner: torch.Tensor,
    topology: WeightedLogicalTopology,
    *,
    num_clients: int,
) -> torch.Tensor:
    """Aggregate incident edge weight by the current owner of each neighbor."""

    values = owner.detach().cpu().long()
    if values.shape != (topology.num_nodes,):
        raise ValueError("Owner vector must align with topology nodes.")
    result = torch.zeros(topology.num_nodes, num_clients, dtype=torch.float64)
    for node in range(topology.num_nodes):
        neighbors = topology.incident_neighbors[node]
        if neighbors.numel():
            result[node].index_add_(
                0, values[neighbors], topology.incident_weights[node]
            )
    return result


def evaluate_hconn(
    owner: torch.Tensor,
    topology: WeightedLogicalTopology,
    *,
    num_clients: int,
) -> HConnMetrics:
    values = owner.detach().cpu().long()
    if values.shape != (topology.num_nodes,) or bool((values < 0).any()) or bool((values >= num_clients).any()):
        raise ValueError("Owner vector is incomplete or invalid.")
    edges = topology.edge_index
    cut = values[edges[0]] != values[edges[1]] if edges.numel() else torch.zeros(0, dtype=torch.bool)
    cut_weight = float(topology.edge_weights[cut].sum())
    boundary = torch.zeros(topology.num_nodes, dtype=torch.bool)
    if bool(cut.any()):
        boundary[edges[0, cut]] = True
        boundary[edges[1, cut]] = True
    degree = torch.zeros(topology.num_nodes, dtype=torch.float64)
    retained = torch.zeros_like(degree)
    for node in range(topology.num_nodes):
        degree[node] = topology.incident_weights[node].sum()
    internal = ~cut
    if edges.numel() and bool(internal.any()):
        degree_weights = topology.edge_weights[internal]
        retained.index_add_(0, edges[0, internal], degree_weights)
        retained.index_add_(0, edges[1, internal], degree_weights)
    retained_ratio = retained / degree.clamp_min(1.0)
    cut_incident = torch.zeros(topology.num_nodes, dtype=torch.float64)
    if edges.numel() and bool(cut.any()):
        cut_weights = topology.edge_weights[cut]
        cut_incident.index_add_(0, edges[0, cut], cut_weights)
        cut_incident.index_add_(0, edges[1, cut], cut_weights)
    exposure = []
    retained_client = []
    for client in range(num_clients):
        selected = values == client
        volume = float(degree[selected].sum())
        exposure.append(
            float(cut_incident[selected].sum()) / volume if volume > 0 else None
        )
        retained_client.append(
            float(retained_ratio[selected].mean()) if bool(selected.any()) else None
        )
    return HConnMetrics(
        h_conn=cut_weight / max(topology.total_edge_weight, 1e-300),
        cut_edge_weight=cut_weight,
        total_edge_weight=topology.total_edge_weight,
        boundary_node_ratio=float(boundary.double().mean()),
        average_retained_degree_ratio=float(retained_ratio.mean()),
        per_client_cut_exposure=exposure,
        per_client_retained_degree_ratio=retained_client,
    )


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def strict_local_graph_hash(partition: PartitionResult) -> str:
    digest = hashlib.sha256()
    for client_id, graph in sorted(partition.client_graphs.items()):
        digest.update(str(client_id).encode("ascii"))
        for tensor in (graph.local_to_global, graph.edge_index, graph.boundary_mask):
            digest.update(tensor_sha256(tensor).encode("ascii"))
    return digest.hexdigest()


def query_shard_hash(shards: Mapping[int, Mapping[int, Any]], evaluations: Mapping[int, Mapping[int, Any]]) -> str:
    digest = hashlib.sha256()
    for client_id in sorted(shards):
        for task_id in sorted(shards[client_id]):
            train = shards[client_id][task_id]
            heldout = evaluations[client_id][task_id]
            digest.update(f"{client_id}:{task_id}".encode("ascii"))
            for tensor in (
                train.train_queries,
                train.train_labels,
                heldout.train_query_ids,
                heldout.validation_queries,
                heldout.validation_query_ids,
                heldout.test_queries,
                heldout.test_query_ids,
            ):
                digest.update(tensor_sha256(tensor).encode("ascii"))
    return digest.hexdigest()


def participation_hash(participation: Any) -> str:
    return payload_sha256(
        {
            "trace": {
                str(stage): {
                    str(round_id): list(clients)
                    for round_id, clients in sorted(rounds.items())
                }
                for stage, rounds in sorted(participation.trace.items())
            },
            "fraction": participation.fraction,
            "rounds_per_stage": participation.rounds_per_stage,
            "seed": participation.seed,
        }
    )


def order_hash(order: Any) -> str:
    return payload_sha256(
        {
            "canonical_order": list(order.canonical_order),
            "client_orders": {
                str(client): list(tasks)
                for client, tasks in sorted(order.client_orders.items())
            },
            "seed": order.seed,
            "block_size": order.block_size,
        }
    )


def _result_versions(result: Any) -> tuple[str, str]:
    """Resolve versioned v1/v2 metadata without changing frozen v1 output."""

    partition_mode = str(
        result.diagnostics.get(
            "policy_version",
            result.diagnostics.get("construction_version", LC_COMMUNITY_VERSION),
        )
    )
    stream_version = str(
        result.diagnostics.get("stream_version", LC_COMMUNITY_STREAM_VERSION)
    )
    return partition_mode, stream_version




_RELOCATED_EXPORTS = {'materialize_direct_partition': ('gecko.data.partitioning.materialize', 'materialize_direct_partition'), 'scenario_with_selected_training_queries': ('gecko.data.transforms', 'scenario_with_selected_training_queries')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)
