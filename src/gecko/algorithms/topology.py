"""Method-owned strict-local topology overlays.

Overlays contain local arc indices only. They fingerprint, but never retain or
mutate, the immutable stream/client edge tensor on which they were created.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable
from typing import Tuple

import torch


def _validate_edge_index(
    edge_index: torch.Tensor, *, num_nodes: int, name: str
) -> None:
    if not torch.is_tensor(edge_index):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if (
        edge_index.dtype != torch.long
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError(f"{name} must have int64 shape [2, num_edges].")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError(f"{name} contains an endpoint outside the strict-local graph.")


def edge_index_sha256(edge_index: torch.Tensor, *, num_nodes: int) -> str:
    """Hash an exact local COO tensor independently of its device."""

    _validate_edge_index(edge_index, num_nodes=num_nodes, name="edge_index")
    tensor = edge_index.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(num_nodes).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {
        (int(source), int(target))
        for source, target in edge_index.detach().cpu().t().tolist()
    }


def _owned_normalized_edges(
    edge_index: torch.Tensor | None,
    *,
    num_nodes: int,
    name: str,
    undirected: bool,
) -> torch.Tensor:
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)
    _validate_edge_index(edge_index, num_nodes=num_nodes, name=name)
    arcs = _edge_set(edge_index)
    if undirected:
        arcs |= {(target, source) for source, target in arcs}
    if not arcs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(sorted(arcs), dtype=torch.long).t().contiguous()


def _overlay_digest(
    *,
    client_id: int,
    global_task_id: int,
    method_name: str,
    base_edge_sha256: str,
    added: torch.Tensor,
    deleted: torch.Tensor,
    undirected: bool,
) -> str:
    digest = hashlib.sha256()
    for value in (
        str(client_id),
        str(global_task_id),
        method_name,
        base_edge_sha256,
        "undirected" if undirected else "directed",
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    digest.update(added.numpy().tobytes())
    digest.update(deleted.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, init=False)
class TopologyOverlay:
    """Validated additions/deletions over one exact strict-local graph.

    Added and deleted arcs are deduplicated and sorted. For an undirected
    overlay each non-self arc is represented in both orientations, so reverse
    arcs remain one logical operation while materialization remains COO-safe.
    """

    client_id: int
    global_task_id: int
    method_name: str
    num_nodes: int
    base_edge_sha256: str
    overlay_id: str
    undirected: bool
    _added_edge_index: torch.Tensor
    _deleted_edge_index: torch.Tensor

    def __init__(
        self,
        *,
        client_id: int,
        global_task_id: int,
        method_name: str,
        num_nodes: int,
        base_edge_index: torch.Tensor,
        added_edge_index: torch.Tensor | None = None,
        deleted_edge_index: torch.Tensor | None = None,
        undirected: bool = False,
    ) -> None:
        if client_id < 0 or global_task_id < 0 or num_nodes <= 0:
            raise ValueError(
                "Client/task IDs must be non-negative and num_nodes positive."
            )
        if not method_name:
            raise ValueError("Topology overlay method_name cannot be empty.")
        _validate_edge_index(
            base_edge_index, num_nodes=num_nodes, name="base_edge_index"
        )
        base = base_edge_index.detach().cpu().clone().contiguous()
        base_arcs = _edge_set(base)
        if undirected and any(
            (target, source) not in base_arcs for source, target in base_arcs
        ):
            raise ValueError("An undirected base graph must contain every reverse arc.")
        added = _owned_normalized_edges(
            added_edge_index,
            num_nodes=num_nodes,
            name="added_edge_index",
            undirected=undirected,
        )
        deleted = _owned_normalized_edges(
            deleted_edge_index,
            num_nodes=num_nodes,
            name="deleted_edge_index",
            undirected=undirected,
        )
        added_arcs = _edge_set(added)
        deleted_arcs = _edge_set(deleted)
        overlap = added_arcs & deleted_arcs
        if overlap:
            raise ValueError(
                f"The same local arcs cannot be added and deleted: {sorted(overlap)}"
            )
        already_present = added_arcs & base_arcs
        if already_present:
            raise ValueError(
                f"Added arcs already exist in the base graph: {sorted(already_present)}"
            )
        absent = deleted_arcs - base_arcs
        if absent:
            raise ValueError(
                f"Deleted arcs are absent from the base graph: {sorted(absent)}"
            )
        base_digest = edge_index_sha256(base, num_nodes=num_nodes)
        overlay_digest = _overlay_digest(
            client_id=client_id,
            global_task_id=global_task_id,
            method_name=method_name,
            base_edge_sha256=base_digest,
            added=added,
            deleted=deleted,
            undirected=undirected,
        )
        object.__setattr__(self, "client_id", int(client_id))
        object.__setattr__(self, "global_task_id", int(global_task_id))
        object.__setattr__(self, "method_name", method_name)
        object.__setattr__(self, "num_nodes", int(num_nodes))
        object.__setattr__(self, "base_edge_sha256", base_digest)
        object.__setattr__(self, "overlay_id", overlay_digest)
        object.__setattr__(self, "undirected", bool(undirected))
        object.__setattr__(self, "_added_edge_index", added)
        object.__setattr__(self, "_deleted_edge_index", deleted)

    @property
    def added_edge_index(self) -> torch.Tensor:
        return self._added_edge_index.clone()

    @property
    def deleted_edge_index(self) -> torch.Tensor:
        return self._deleted_edge_index.clone()

    @property
    def payload_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self._added_edge_index, self._deleted_edge_index)
        )

    def apply(self, base_edge_index: torch.Tensor) -> torch.Tensor:
        """Materialize a new COO tensor without modifying the supplied base."""

        _validate_edge_index(
            base_edge_index, num_nodes=self.num_nodes, name="base_edge_index"
        )
        if (
            edge_index_sha256(base_edge_index, num_nodes=self.num_nodes)
            != self.base_edge_sha256
        ):
            raise ValueError("Topology overlay base-edge fingerprint mismatch.")
        deleted = _edge_set(self._deleted_edge_index)
        kept = [
            (int(source), int(target))
            for source, target in base_edge_index.detach().cpu().t().tolist()
            if (int(source), int(target)) not in deleted
        ]
        kept.extend(sorted(_edge_set(self._added_edge_index)))
        if not kept:
            materialized = torch.empty((2, 0), dtype=torch.long)
        else:
            materialized = torch.tensor(kept, dtype=torch.long).t().contiguous()
        return materialized.to(base_edge_index.device)
