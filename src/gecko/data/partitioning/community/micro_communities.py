"""Topology-preserving deterministic micro-community generation."""

from __future__ import annotations


import logging
import math
from collections import deque

import torch

from gecko.reproducibility import torch_generator

LOGGER = logging.getLogger(__name__)
_NATIVE_DGL_METIS_EDGE_THRESHOLD = 10_000_000


def _metis_assignment(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_partitions: int,
    seed: int,
) -> torch.Tensor | None:
    try:
        source = edge_index[0].detach().cpu().long()
        target = edge_index[1].detach().cpu().long()
        if edge_index.shape[1] >= _NATIVE_DGL_METIS_EDGE_THRESHOLD:
            import dgl

            graph = dgl.graph((source, target), num_nodes=num_nodes)
            assignment = dgl.metis_partition_assignment(
                graph,
                num_partitions,
                mode="recursive",
                objtype="cut",
            ).detach().cpu().long()
        else:
            import pymetis  # type: ignore

            non_self = source != target
            source = source[non_self]
            target = target[non_self]
            undirected_source = torch.cat([source, target])
            undirected_target = torch.cat([target, source])
            encoded = torch.unique(
                undirected_source * num_nodes + undirected_target,
                sorted=True,
            )
            csr_source = encoded.div(num_nodes, rounding_mode="floor")
            csr_target = encoded.remainder(num_nodes)
            counts = torch.bincount(csr_source, minlength=num_nodes)
            xadj = torch.zeros(num_nodes + 1, dtype=torch.long)
            xadj[1:] = counts.cumsum(0)
            options = pymetis.Options(seed=seed % (2**31))
            _, membership = pymetis.part_graph(
                num_partitions,
                xadj=xadj.numpy(),
                adjncy=csr_target.numpy(),
                recursive=True,
                options=options,
            )
            assignment = torch.as_tensor(membership, dtype=torch.long)
        if assignment.shape != (num_nodes,) or bool((assignment < 0).any()):
            raise RuntimeError("METIS returned an invalid node assignment.")
        parts = torch.unique(assignment, sorted=True)
        ordered_parts = sorted(
            parts.tolist(),
            key=lambda part: int(torch.nonzero(assignment == part)[0]),
        )
        remap = torch.full((int(parts.max()) + 1,), -1, dtype=torch.long)
        remap[torch.tensor(ordered_parts)] = torch.arange(len(ordered_parts))
        return remap[assignment]
    except (ImportError, RuntimeError, ValueError) as exc:
        LOGGER.info("METIS unavailable; using deterministic BFS fallback: %s", exc)
        return None


def _bfs_assignment(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_partitions: int,
    seed: int,
) -> torch.Tensor:
    adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    for source, target in edge_index.t().tolist():
        if source == target:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    generator = torch_generator(seed, "micro-bfs")
    tie_break = torch.randperm(num_nodes, generator=generator).tolist()
    tie_rank = {node: rank for rank, node in enumerate(tie_break)}
    target_size = max(1, math.ceil(num_nodes / num_partitions))
    assignment = torch.full((num_nodes,), -1, dtype=torch.long)
    community = 0
    for start in sorted(range(num_nodes), key=lambda node: (len(adjacency[node]), tie_rank[node])):
        if assignment[start] >= 0:
            continue
        queue: deque[int] = deque([start])
        while queue:
            if community >= num_partitions:
                community = num_partitions - 1
            current_batch: list[int] = []
            while queue and len(current_batch) < target_size:
                node = queue.popleft()
                if assignment[node] >= 0:
                    continue
                assignment[node] = community
                current_batch.append(node)
                neighbors = sorted(adjacency[node], key=lambda item: tie_rank[item])
                queue.extend(neighbor for neighbor in neighbors if assignment[neighbor] < 0)
            if current_batch:
                community += 1
        if community >= num_partitions:
            break
    unassigned = torch.nonzero(assignment < 0, as_tuple=True)[0]
    if unassigned.numel():
        sizes = torch.bincount(assignment[assignment >= 0], minlength=num_partitions)
        for node in unassigned.tolist():
            client = int(torch.argmin(sizes))
            assignment[node] = client
            sizes[client] += 1
    unique = torch.unique(assignment, sorted=True)
    remap = {int(old): new for new, old in enumerate(unique.tolist())}
    return torch.tensor([remap[int(value)] for value in assignment], dtype=torch.long)


def generate_micro_communities(
    edge_index: torch.Tensor,
    num_nodes: int,
    target_partitions: int,
    seed: int,
    *,
    prefer_metis: bool = True,
) -> torch.Tensor:
    """Generate one immutable micro-community ID per global node."""

    partitions = max(1, min(target_partitions, num_nodes))
    if prefer_metis:
        metis = _metis_assignment(edge_index, num_nodes, partitions, seed)
        if metis is not None:
            return metis
    return _bfs_assignment(edge_index, num_nodes, partitions, seed)
