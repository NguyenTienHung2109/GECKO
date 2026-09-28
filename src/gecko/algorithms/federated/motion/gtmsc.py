"""Leakage-safe client-side G-TMSC graph memory for MOTION.

The implementation preserves the public MOTION mechanisms—dynamic graph
merging, reservoir replay, multi-expert node scoring, similarity-guided
coarsening, and dual node mappings—while accepting only strict-local topology,
features, and current training labels.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable
from typing import Mapping
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MotionReservoir:
    """Persistent Algorithm-R sample of labeled local node identifiers."""

    node_ids: torch.Tensor
    labels: torch.Tensor
    seen_samples: int


@dataclass(frozen=True)
class MotionGraphMemory:
    """One client's coarsened graph and original-to-coarse traceability map."""

    features: torch.Tensor
    edge_index: torch.Tensor
    label_histogram: torch.Tensor
    node_to_coarse: torch.Tensor

    @property
    def train_mask(self) -> torch.Tensor:
        return self.label_histogram.sum(dim=1) > 0

    @property
    def labels(self) -> torch.Tensor:
        return self.label_histogram.argmax(dim=1).to(dtype=torch.long)

    @property
    def payload_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.features,
                self.edge_index,
                self.label_histogram,
                self.node_to_coarse,
            )
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "features": self.features.detach().clone().contiguous(),
            "edge_index": self.edge_index.detach().clone().contiguous(),
            "label_histogram": self.label_histogram.detach().clone().contiguous(),
            "node_to_coarse": self.node_to_coarse.detach().clone().contiguous(),
        }


@dataclass(frozen=True)
class MotionExpertScores:
    """Observable multi-expert scoring evidence used by G-TMSC."""

    scores: torch.Tensor
    gates: torch.Tensor
    topology_features: torch.Tensor
    subgraph_features: torch.Tensor
    similarity_features: torch.Tensor


@dataclass(frozen=True)
class MotionCoarseningResult:
    graph: MotionGraphMemory
    kept_nodes: torch.Tensor
    coarse_assignment: torch.Tensor
    expert_scores: MotionExpertScores


def _validate_edge_index(
    edge_index: torch.Tensor, *, num_nodes: int, name: str = "edge_index"
) -> torch.Tensor:
    if (
        not torch.is_tensor(edge_index)
        or edge_index.dtype != torch.long
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError(f"{name} must have int64 shape [2, edges].")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError(f"{name} contains an invalid local endpoint.")
    return edge_index.detach().clone().contiguous()


def _coalesce_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    encoded = edge_index[0] * num_nodes + edge_index[1]
    encoded = torch.unique(encoded, sorted=True)
    return torch.stack(
        (encoded.div(num_nodes, rounding_mode="floor"), encoded.remainder(num_nodes))
    ).to(dtype=torch.long).contiguous()


def _dense_undirected_adjacency(
    edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Materialize one small strict-local adjacency on the input device."""

    adjacency = torch.zeros(
        (num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device
    )
    if edge_index.numel():
        source, target = edge_index
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0
    adjacency.fill_diagonal_(0.0)
    return adjacency


def _l2_columns(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    return F.normalize(values, p=2, dim=0)


def _sampled_brandes_gpu(
    adjacency: torch.Tensor, sources: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched unweighted Brandes and shortest paths on one tensor device."""

    num_nodes = int(adjacency.shape[0])
    source_count = int(sources.numel())
    distances = torch.full(
        (source_count, num_nodes),
        -1,
        dtype=torch.long,
        device=adjacency.device,
    )
    paths = torch.zeros(
        (source_count, num_nodes), dtype=torch.float32, device=adjacency.device
    )
    rows = torch.arange(source_count, device=adjacency.device)
    distances[rows, sources] = 0
    paths[rows, sources] = 1.0
    frontier = torch.zeros_like(paths, dtype=torch.bool)
    frontier[rows, sources] = True
    maximum_depth = 0
    for depth in range(num_nodes):
        next_paths = (paths * frontier).matmul(adjacency)
        next_frontier = (next_paths > 0) & (distances < 0)
        if not bool(next_frontier.any()):
            break
        distances[next_frontier] = depth + 1
        paths = torch.where(next_frontier, next_paths, paths)
        frontier = next_frontier
        maximum_depth = depth + 1

    edge_pairs = torch.nonzero(adjacency > 0, as_tuple=False)
    source_nodes = edge_pairs[:, 0]
    target_nodes = edge_pairs[:, 1]
    dependency = torch.zeros_like(paths)
    expanded_sources = source_nodes.unsqueeze(0).expand(source_count, -1)
    for depth in range(maximum_depth - 1, -1, -1):
        valid = (distances[:, source_nodes] == depth) & (
            distances[:, target_nodes] == depth + 1
        )
        contribution = (
            paths[:, source_nodes]
            / paths[:, target_nodes].clamp_min(1e-12)
            * (1.0 + dependency[:, target_nodes])
            * valid
        )
        accumulated = torch.zeros_like(dependency)
        accumulated.scatter_add_(1, expanded_sources, contribution)
        dependency.add_(accumulated)
    dependency[rows, sources] = 0.0
    scale = 0.5 * num_nodes / max(1, source_count)
    return (dependency.sum(dim=0) * scale).float(), distances


def motion_topology_features(
    edge_index: torch.Tensor, *, num_nodes: int
) -> torch.Tensor:
    """Return the eight G-TMSC topology features on the tensor device."""

    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes <= 0:
        raise ValueError("num_nodes must be a positive integer.")
    edges = _validate_edge_index(edge_index, num_nodes=num_nodes)
    adjacency = _dense_undirected_adjacency(edges, num_nodes)
    degree = adjacency.sum(dim=1)
    degree_normalized = degree / degree.max().clamp_min(1.0)

    source_count = num_nodes if num_nodes <= 256 else min(64, num_nodes)
    sources = torch.linspace(
        0, num_nodes - 1, steps=source_count, device=edges.device
    ).round().long().unique(sorted=True)
    betweenness, distances = _sampled_brandes_gpu(adjacency, sources)

    common_neighbors = adjacency.matmul(adjacency)
    twice_triangles = (common_neighbors * adjacency).sum(dim=1)
    clustering = twice_triangles / (degree * (degree - 1.0)).clamp_min(1.0)
    clustering = torch.where(degree >= 2, clustering, torch.zeros_like(clustering))

    neighbor_degree_sum = adjacency.matmul(degree)
    neighbor_degree_square_sum = adjacency.matmul(degree.square())
    mean_neighbor_degree = neighbor_degree_sum / degree.clamp_min(1.0)
    variance = (
        neighbor_degree_square_sum / degree.clamp_min(1.0)
        - mean_neighbor_degree.square()
    ).clamp_min(0.0)
    heterogeneity = variance.sqrt() / mean_neighbor_degree.clamp_min(1e-12)
    heterogeneity = torch.where(
        degree > 1, heterogeneity, torch.zeros_like(heterogeneity)
    )

    eigenvector = torch.full(
        (num_nodes,),
        1.0 / math.sqrt(num_nodes),
        device=edges.device,
        dtype=torch.float32,
    )
    for _ in range(64):
        updated = adjacency.matmul(eigenvector)
        updated = updated / updated.norm().clamp_min(1e-12)
        converged = bool(torch.max(torch.abs(updated - eigenvector)) <= 1e-6)
        eigenvector = updated
        if converged:
            break

    valid_distances = distances > 0
    distance_sum = torch.where(
        valid_distances, distances, torch.zeros_like(distances)
    ).sum(dim=0).float()
    reachable = valid_distances.sum(dim=0).float()
    closeness = torch.where(
        distance_sum > 0, reachable / distance_sum.clamp_min(1.0), 0.0
    )

    pagerank = torch.full(
        (num_nodes,), 1.0 / num_nodes, device=edges.device, dtype=torch.float32
    )
    damping = 0.85
    for _ in range(64):
        contributions = pagerank / degree.clamp_min(1.0)
        dangling = pagerank[degree == 0].sum() / num_nodes
        updated = (1.0 - damping) / num_nodes + damping * (
            adjacency.t().matmul(contributions) + dangling
        )
        converged = bool(torch.max(torch.abs(updated - pagerank)) <= 1e-8)
        pagerank = updated
        if converged:
            break

    features = torch.stack(
        (
            degree_normalized,
            betweenness,
            clustering,
            eigenvector,
            closeness,
            pagerank,
            heterogeneity,
            clustering.clone(),
        ),
        dim=1,
    )
    return _l2_columns(features).contiguous()


def motion_subgraph_features(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    hop_sizes: Iterable[int] = (1, 2),
) -> torch.Tensor:
    """Compute tensorized 1/2-hop ego statistics for every local node."""

    edges = _validate_edge_index(edge_index, num_nodes=num_nodes)
    hops = tuple(int(value) for value in hop_sizes)
    if not hops or any(value <= 0 for value in hops):
        raise ValueError("MOTION hop sizes must be positive integers.")
    adjacency = _dense_undirected_adjacency(edges, num_nodes)
    topology = motion_topology_features(edges, num_nodes=num_nodes)
    local_clustering = topology[:, 2]
    visited = torch.eye(num_nodes, dtype=torch.bool, device=edges.device)
    frontier = visited.clone()
    masks: dict[int, torch.Tensor] = {}
    for depth in range(1, max(hops) + 1):
        next_frontier = frontier.float().matmul(adjacency) > 0
        next_frontier &= ~visited
        visited |= next_frontier
        frontier = next_frontier
        if depth in hops:
            masks[depth] = visited.clone()

    per_hop: list[torch.Tensor] = []
    for hop in hops:
        mask = masks[hop].float()
        size = mask.sum(dim=1).clamp_min(1.0)
        internal_twice = (mask.matmul(adjacency) * mask).sum(dim=1)
        edge_count = 0.5 * internal_twice
        average_degree = 2.0 * edge_count / size
        density = torch.where(
            size > 1,
            2.0 * edge_count / (size * (size - 1.0)).clamp_min(1.0),
            0.0,
        )
        clustering = mask.matmul(local_clustering) / size
        diameter = torch.where(
            size > 1,
            torch.full_like(size, float(2 * hop)),
            torch.zeros_like(size),
        )
        per_hop.append(
            torch.stack((average_degree, clustering, diameter, density), dim=1)
        )
    return _l2_columns(torch.stack(per_hop).mean(dim=0)).contiguous()


def motion_degree_embeddings(
    edge_index: torch.Tensor, *, num_nodes: int, embedding_dim: int = 8
) -> torch.Tensor:
    """Return sinusoidal position-aware degree embeddings on-device."""

    if embedding_dim <= 0 or embedding_dim % 2:
        raise ValueError("MOTION degree embedding dimension must be positive and even.")
    edges = _validate_edge_index(edge_index, num_nodes=num_nodes)
    degrees = _dense_undirected_adjacency(edges, num_nodes).sum(dim=1)
    positions = torch.arange(
        0, embedding_dim, 2, dtype=torch.float32, device=edges.device
    )
    divisors = torch.exp(-math.log(10000.0) * positions / embedding_dim)
    output = torch.zeros(
        (num_nodes, embedding_dim), dtype=torch.float32, device=edges.device
    )
    output[:, 0::2] = torch.sin(degrees.unsqueeze(1) * divisors)
    output[:, 1::2] = torch.cos(degrees.unsqueeze(1) * divisors)
    return _l2_columns(output).contiguous()


def motion_similarity_features(
    node_features: torch.Tensor, *, max_reference_nodes: int = 256
) -> torch.Tensor:
    """Return MMD, diagonal Mahalanobis, Pearson, and cosine statistics."""

    if (
        not torch.is_tensor(node_features)
        or not node_features.is_floating_point()
        or node_features.ndim != 2
        or node_features.shape[0] == 0
    ):
        raise ValueError("MOTION similarity features require floating [nodes, features].")
    if not bool(torch.isfinite(node_features).all()):
        raise ValueError("MOTION node features must be finite.")
    features = node_features.detach().float()
    num_nodes, dimension = features.shape
    reference_count = min(max_reference_nodes, num_nodes)
    indices = (
        torch.linspace(
            0, num_nodes - 1, steps=reference_count, device=features.device
        )
        .round()
        .long()
    )
    indices = torch.unique(indices, sorted=True)
    reference = features[indices]

    squared = torch.cdist(reference, reference).square() / max(1, dimension)
    reference_kernel_mean = torch.exp(-squared).mean()
    mmd_parts: list[torch.Tensor] = []
    cosine_parts: list[torch.Tensor] = []
    normalized_reference = F.normalize(reference, p=2, dim=1)
    for start in range(0, num_nodes, 512):
        chunk = features[start : start + 512]
        kernel = torch.exp(
            -torch.cdist(chunk, reference).square() / max(1, dimension)
        )
        mmd_parts.append((kernel.mean(dim=1) - reference_kernel_mean).abs())
        cosine_parts.append(
            (F.normalize(chunk, p=2, dim=1) @ normalized_reference.t()).mean(dim=1)
        )
    mmd = torch.cat(mmd_parts)
    average_cosine = torch.cat(cosine_parts)

    mean = features.mean(dim=0)
    variance = features.var(dim=0, unbiased=False).clamp_min(1e-5)
    mahalanobis = torch.sqrt(((features - mean).square() / variance).mean(dim=1))

    important_count = min(50, num_nodes)
    important = torch.argsort(features.norm(dim=1), descending=True)[:important_count]
    centered = features - features.mean(dim=1, keepdim=True)
    centered = F.normalize(centered, p=2, dim=1)
    pearson = (centered @ centered[important].t()).abs().mean(dim=1)
    output = torch.stack((mmd, mahalanobis, pearson, average_cosine), dim=1)
    return _l2_columns(output).contiguous()


def motion_multi_expert_scores(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    k_list: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    expert_select: int = 2,
    use_node_positional: bool = True,
    use_node_mmd: bool = True,
    use_node_mahalanobis: bool = True,
) -> MotionExpertScores:
    """Score nodes with sparse topology/subgraph/semantic experts."""

    if node_features.ndim != 2 or not node_features.is_floating_point():
        raise ValueError("MOTION experts require floating [nodes, features].")
    num_nodes = int(node_features.shape[0])
    edges = _validate_edge_index(edge_index, num_nodes=num_nodes)
    ratios = tuple(float(value) for value in k_list)
    if (
        not ratios
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in ratios)
        or isinstance(expert_select, bool)
        or not isinstance(expert_select, int)
        or not 1 <= expert_select <= len(ratios)
    ):
        raise ValueError("MOTION expert controls are invalid.")

    topology = motion_topology_features(edges, num_nodes=num_nodes)
    subgraph = motion_subgraph_features(edges, num_nodes=num_nodes)
    similarity = motion_similarity_features(node_features)
    positional = (
        motion_degree_embeddings(edges, num_nodes=num_nodes)
        if use_node_positional
        else torch.zeros((num_nodes, 8))
    )
    structural_core = (
        topology[:, 0]
        + topology[:, 1]
        + topology[:, 3]
        + topology[:, 4]
        + topology[:, 5]
        + subgraph[:, 0]
        + subgraph[:, 3]
    ) / 7.0
    semantic_outlier = (1.0 - similarity[:, 3]).clamp_min(0.0)
    if use_node_mmd:
        semantic_outlier = semantic_outlier + similarity[:, 0]
    if use_node_mahalanobis:
        semantic_outlier = semantic_outlier + similarity[:, 1]
    semantic_outlier = semantic_outlier + (1.0 - similarity[:, 2]).clamp_min(0.0)
    positional_signal = positional.abs().mean(dim=1)

    expert_columns = []
    for ratio in ratios:
        expert_columns.append(
            (1.0 - ratio) * structural_core
            + ratio * semantic_outlier
            + 0.25 * subgraph.mean(dim=1)
            + 0.10 * positional_signal
        )
    expert_outputs = torch.stack(expert_columns, dim=1)
    top_values, top_indices = torch.topk(
        expert_outputs, k=expert_select, dim=1, largest=True, sorted=True
    )
    top_gates = torch.softmax(top_values / 0.5, dim=1)
    gates = torch.zeros_like(expert_outputs)
    gates.scatter_(1, top_indices, top_gates)
    node_scores = (gates * expert_outputs).mean(dim=1)
    final_scores = 0.6 * node_scores + 0.1 * semantic_outlier
    if not bool(torch.isfinite(final_scores).all()):
        raise RuntimeError("MOTION expert scoring produced non-finite values.")
    return MotionExpertScores(
        scores=final_scores.contiguous(),
        gates=gates.contiguous(),
        topology_features=topology,
        subgraph_features=subgraph,
        similarity_features=similarity,
    )


def motion_update_reservoir(
    reservoir: MotionReservoir | None,
    train_queries: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    capacity: int,
    seed: int,
) -> MotionReservoir:
    """Apply deterministic Algorithm-R sampling on the query tensor device."""

    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("MOTION reservoir capacity must be a positive integer.")
    if (
        train_queries.dtype != torch.long
        or train_queries.ndim != 1
        or train_labels.dtype != torch.long
        or train_labels.ndim != 1
        or train_labels.shape != train_queries.shape
    ):
        raise ValueError("MOTION reservoir requires aligned 1-D long queries/labels.")
    device = train_queries.device
    labels_input = train_labels.to(device)
    if reservoir is None:
        node_ids = torch.empty(0, dtype=torch.long, device=device)
        labels = torch.empty(0, dtype=torch.long, device=device)
        seen = 0
    else:
        if (
            reservoir.node_ids.dtype != torch.long
            or reservoir.labels.dtype != torch.long
            or reservoir.node_ids.shape != reservoir.labels.shape
            or reservoir.node_ids.ndim != 1
            or reservoir.seen_samples < reservoir.node_ids.numel()
        ):
            raise ValueError("Existing MOTION reservoir is malformed.")
        node_ids = reservoir.node_ids.detach().to(device).clone()
        labels = reservoir.labels.detach().to(device).clone()
        seen = int(reservoir.seen_samples)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) % (2**63 - 1))
    for index in range(int(train_queries.numel())):
        seen += 1
        node = train_queries[index]
        label = labels_input[index]
        if node_ids.numel() < capacity:
            node_ids = torch.cat((node_ids, node.reshape(1)))
            labels = torch.cat((labels, label.reshape(1)))
            continue
        candidate = torch.randint(
            0, seen, (1,), generator=generator, device=device
        )
        if bool(candidate < capacity):
            node_ids[candidate] = node
            labels[candidate] = label
    return MotionReservoir(
        node_ids=node_ids.contiguous(),
        labels=labels.contiguous(),
        seen_samples=seen,
    )


def motion_memory_from_state(state: Mapping[str, torch.Tensor]) -> MotionGraphMemory:
    expected = {"features", "edge_index", "label_histogram", "node_to_coarse"}
    if set(state) != expected:
        raise ValueError("MOTION graph-memory state fields do not match.")
    memory = MotionGraphMemory(
        features=state["features"].detach().clone().contiguous(),
        edge_index=state["edge_index"].detach().clone().contiguous(),
        label_histogram=state["label_histogram"].detach().clone().contiguous(),
        node_to_coarse=state["node_to_coarse"].detach().clone().contiguous(),
    )
    _validate_memory(memory)
    return memory


def _validate_memory(memory: MotionGraphMemory) -> None:
    if (
        memory.features.ndim != 2
        or not memory.features.is_floating_point()
        or memory.features.shape[0] == 0
        or memory.label_histogram.ndim != 2
        or memory.label_histogram.shape[0] != memory.features.shape[0]
        or not memory.label_histogram.is_floating_point()
        or memory.node_to_coarse.dtype != torch.long
        or memory.node_to_coarse.ndim != 1
    ):
        raise ValueError("MOTION graph memory tensors are malformed.")
    _validate_edge_index(
        memory.edge_index, num_nodes=memory.features.shape[0], name="memory edges"
    )
    assigned = memory.node_to_coarse >= 0
    if assigned.any() and int(memory.node_to_coarse[assigned].max()) >= memory.features.shape[0]:
        raise ValueError("MOTION original-to-coarse mapping is invalid.")
    if not bool(torch.isfinite(memory.features).all()) or not bool(
        torch.isfinite(memory.label_histogram).all()
    ):
        raise ValueError("MOTION graph memory must be finite.")
    if bool((memory.label_histogram < 0).any()):
        raise ValueError("MOTION label histograms cannot be negative.")


def motion_merge_observed_graph(
    memory: MotionGraphMemory | None,
    *,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    train_queries: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
) -> MotionGraphMemory:
    """Merge strict-local observations on the feature tensor device."""

    if (
        node_features.ndim != 2
        or not node_features.is_floating_point()
        or node_features.shape[0] == 0
        or not bool(torch.isfinite(node_features).all())
    ):
        raise ValueError("MOTION merge requires finite floating local node features.")
    device = node_features.device
    num_nodes = int(node_features.shape[0])
    edges = _validate_edge_index(
        edge_index.to(device), num_nodes=num_nodes
    )
    queries = train_queries.to(device)
    labels = train_labels.to(device)
    if (
        queries.dtype != torch.long
        or queries.ndim != 1
        or labels.dtype != torch.long
        or labels.ndim != 1
        or queries.shape != labels.shape
        or (queries.numel() and (int(queries.min()) < 0 or int(queries.max()) >= num_nodes))
        or (labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes))
    ):
        raise ValueError("MOTION merge requires valid current NC train queries/labels.")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 1:
        raise ValueError("MOTION merge requires at least two classes.")

    observed = torch.unique(torch.cat((edges.reshape(-1), queries)), sorted=True)
    if memory is None:
        base_features = torch.empty(
            (0, node_features.shape[1]), dtype=node_features.dtype, device=device
        )
        base_histogram = torch.empty(
            (0, num_classes), dtype=torch.float32, device=device
        )
        base_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        node_to_work = torch.full(
            (num_nodes,), -1, dtype=torch.long, device=device
        )
    else:
        _validate_memory(memory)
        if (
            memory.features.shape[1] != node_features.shape[1]
            or memory.label_histogram.shape[1] != num_classes
            or memory.node_to_coarse.shape[0] != num_nodes
        ):
            raise ValueError("MOTION prior memory is incompatible with the local graph.")
        base_features = memory.features.detach().to(device).clone()
        base_histogram = memory.label_histogram.detach().to(device).clone().float()
        base_edges = memory.edge_index.detach().to(device).clone()
        node_to_work = memory.node_to_coarse.detach().to(device).clone()

    new_nodes = observed[node_to_work[observed] < 0]
    if new_nodes.numel():
        start_index = base_features.shape[0]
        node_to_work[new_nodes] = torch.arange(
            start_index, start_index + new_nodes.numel(), device=device
        )
        features = torch.cat((base_features, node_features.detach()[new_nodes]), dim=0)
        histogram = torch.cat(
            (
                base_histogram,
                torch.zeros(
                    (new_nodes.numel(), num_classes),
                    dtype=torch.float32,
                    device=device,
                ),
            ),
            dim=0,
        )
    else:
        features = base_features
        histogram = base_histogram
    if features.shape[0] == 0:
        raise ValueError("MOTION cannot construct an empty observed graph.")

    coarse_queries = node_to_work[queries]
    if bool((coarse_queries < 0).any()):
        raise RuntimeError("MOTION train node was not added to the observed graph.")
    histogram.index_put_(
        (coarse_queries, labels),
        torch.ones_like(labels, dtype=histogram.dtype),
        accumulate=True,
    )

    mapped_source = node_to_work[edges[0]]
    mapped_target = node_to_work[edges[1]]
    valid = (mapped_source >= 0) & (mapped_target >= 0)
    current_edges = torch.stack((mapped_source[valid], mapped_target[valid]))
    combined_edges = _coalesce_edges(
        torch.cat((base_edges, current_edges), dim=1), features.shape[0]
    )
    result = MotionGraphMemory(
        features=features.contiguous(),
        edge_index=combined_edges,
        label_histogram=histogram.contiguous(),
        node_to_coarse=node_to_work.contiguous(),
    )
    _validate_memory(result)
    return result


def motion_coarsen_graph(
    graph: MotionGraphMemory,
    hidden_features: torch.Tensor,
    *,
    protected_raw_nodes: torch.Tensor,
    reduction_rate: float = 0.5,
    k_list: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    expert_select: int = 2,
    use_node_positional: bool = True,
    use_node_mmd: bool = True,
    use_node_mahalanobis: bool = True,
    similarity_threshold: float = 0.7,
) -> MotionCoarseningResult:
    """Coarsen on-device while forcing reservoir representatives to survive."""

    _validate_memory(graph)
    device = hidden_features.device
    if (
        hidden_features.ndim != 2
        or not hidden_features.is_floating_point()
        or hidden_features.shape[0] != graph.features.shape[0]
        or not bool(torch.isfinite(hidden_features).all())
    ):
        raise ValueError("MOTION hidden features must align with graph memory nodes.")
    reduction_rate = float(reduction_rate)
    if not math.isfinite(reduction_rate) or not 0.0 < reduction_rate < 1.0:
        raise ValueError("MOTION reduction_rate must lie in (0, 1).")
    similarity_threshold = float(similarity_threshold)
    if not math.isfinite(similarity_threshold) or not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("MOTION similarity_threshold must lie in [-1, 1].")
    protected_raw_nodes = protected_raw_nodes.to(device)
    if protected_raw_nodes.dtype != torch.long or protected_raw_nodes.ndim != 1:
        raise ValueError("MOTION protected raw nodes must be a 1-D long tensor.")
    if protected_raw_nodes.numel() and (
        int(protected_raw_nodes.min()) < 0
        or int(protected_raw_nodes.max()) >= graph.node_to_coarse.shape[0]
    ):
        raise ValueError("MOTION protected raw node is outside the local graph.")

    graph_edges = graph.edge_index.detach().to(device)
    scores = motion_multi_expert_scores(
        hidden_features.detach(),
        graph_edges,
        k_list=k_list,
        expert_select=expert_select,
        use_node_positional=use_node_positional,
        use_node_mmd=use_node_mmd,
        use_node_mahalanobis=use_node_mahalanobis,
    )
    raw_mapping = graph.node_to_coarse.detach().to(device)
    protected = raw_mapping[protected_raw_nodes]
    protected = torch.unique(protected[protected >= 0], sorted=True)
    node_count = int(graph.features.shape[0])
    keep_count = max(1, int(node_count * (1.0 - reduction_rate)))
    keep_count = min(node_count, max(keep_count, int(protected.numel())))
    protected_mask = torch.zeros(node_count, dtype=torch.bool, device=device)
    protected_mask[protected] = True
    ranked = torch.argsort(scores.scores, descending=True, stable=True)
    remaining = ranked[~protected_mask[ranked]]
    kept = torch.cat((protected, remaining[: keep_count - protected.numel()]))
    kept = torch.sort(kept).values.long()

    normalized = F.normalize(hidden_features.detach().float(), p=2, dim=1)
    similarity = normalized.matmul(normalized[kept].t())
    assignment_to_kept = similarity.argmax(dim=1)
    assignment_to_kept[kept] = torch.arange(kept.numel(), device=device)
    maximum_similarity = similarity.max(dim=1).values
    structurally_unmatched = maximum_similarity < similarity_threshold
    if bool(structurally_unmatched.any()):
        topology = scores.topology_features
        topology_similarity = F.normalize(topology, p=2, dim=1).matmul(
            F.normalize(topology[kept], p=2, dim=1).t()
        )
        assignment_to_kept[structurally_unmatched] = topology_similarity[
            structurally_unmatched
        ].argmax(dim=1)
        assignment_to_kept[kept] = torch.arange(kept.numel(), device=device)

    coarse_count = int(kept.numel())
    counts = torch.bincount(assignment_to_kept, minlength=coarse_count).float()
    graph_features = graph.features.detach().to(device)
    coarse_features = torch.zeros(
        (coarse_count, graph_features.shape[1]),
        dtype=graph_features.dtype,
        device=device,
    )
    coarse_features.index_add_(0, assignment_to_kept, graph_features)
    coarse_features /= counts.unsqueeze(1).clamp_min(1.0)
    graph_histogram = graph.label_histogram.detach().to(device).float()
    coarse_histogram = torch.zeros(
        (coarse_count, graph_histogram.shape[1]),
        dtype=torch.float32,
        device=device,
    )
    coarse_histogram.index_add_(0, assignment_to_kept, graph_histogram)

    remapped_edges = assignment_to_kept[graph_edges]
    non_self = remapped_edges[0] != remapped_edges[1]
    remapped_edges = remapped_edges[:, non_self]
    if remapped_edges.numel() == 0:
        remapped_edges = torch.tensor(
            [[0], [0]], dtype=torch.long, device=device
        )
    remapped_edges = _coalesce_edges(remapped_edges, coarse_count)
    node_to_coarse = raw_mapping.clone()
    assigned_raw = node_to_coarse >= 0
    node_to_coarse[assigned_raw] = assignment_to_kept[node_to_coarse[assigned_raw]]
    result_graph = MotionGraphMemory(
        features=coarse_features.contiguous(),
        edge_index=remapped_edges,
        label_histogram=coarse_histogram.contiguous(),
        node_to_coarse=node_to_coarse.contiguous(),
    )
    _validate_memory(result_graph)
    return MotionCoarseningResult(
        graph=result_graph,
        kept_nodes=kept,
        coarse_assignment=assignment_to_kept.contiguous(),
        expert_scores=scores,
    )
