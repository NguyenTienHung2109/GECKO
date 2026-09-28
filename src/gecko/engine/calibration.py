"""Explicit, artifact-paired Class-IL calibration diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import hashlib
from typing import Iterable

import torch

from gecko.engine.client import FederatedClient
from gecko.types import ClientTaskShard


@dataclass(frozen=True)
class MemoryExample:
    task: int
    index: int
    label: int
    query: tuple[int, ...]


def training_examples(shard: ClientTaskShard) -> list[MemoryExample]:
    """Identify training interactions independently of duplicate endpoint pairs."""
    return [MemoryExample(shard.global_task_id, i, int(label),
                          tuple(query.reshape(-1).tolist()))
            for i, (query, label) in enumerate(zip(shard.train_queries, shard.train_labels))]


def rebalance_memory(previous: list[MemoryExample], current: list[MemoryExample],
                     classes: Iterable[int], *, seed: int, client: int,
                     stage: int, capacity: int = 100) -> list[MemoryExample]:
    """Balance retained training IDs, with deterministic shortage redistribution."""
    classes = sorted(set(int(c) for c in classes))
    if not classes or capacity < 0:
        raise ValueError('A nonempty class mask and nonnegative capacity are required.')
    candidates = {(e.task, e.index): e for e in previous}
    for example in current:
        key = (example.task, example.index)
        if key in candidates and candidates[key] != example:
            raise ValueError('Conflicting labels or coordinates for one training ID.')
        candidates[key] = example
    if any(e.label not in classes for e in candidates.values()):
        raise ValueError('Memory contains a class outside the seen-class mask.')
    pools = {c: sorted((e for e in candidates.values() if e.label == c),
                       key=lambda e: (e.task, e.index)) for c in classes}
    quota = {c: min(len(pools[c]), capacity // len(classes) + (i < capacity % len(classes)))
             for i, c in enumerate(classes)}
    remaining = min(capacity, len(candidates)) - sum(quota.values())
    while remaining:
        for c in classes:
            if quota[c] < len(pools[c]) and remaining:
                quota[c] += 1
                remaining -= 1
    digest = hashlib.sha256(f'uefa-simple-replay-v1:{seed}:{client}:{stage}'.encode()).digest()
    rng = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % (2**63))
    retained = []
    for c in classes:
        indices = torch.randperm(len(pools[c]), generator=rng)[:quota[c]].tolist()
        retained.extend(pools[c][i] for i in indices)
    return sorted(retained, key=lambda e: (e.task, e.index))


def prior_class(labels: torch.Tensor, mask: torch.Tensor) -> int:
    """Return a train-frequency winner, resolving ties by canonical class ID."""
    allowed = mask.nonzero().flatten()
    if not len(allowed):
        raise ValueError('Prior requires a nonempty output mask.')
    counts = torch.bincount(labels.long().cpu(), minlength=mask.numel())
    return int(allowed[counts[allowed].argmax()])


class CalibrationClient(FederatedClient):
    """Use Bare's exact local optimizer on an explicitly supplied training batch."""

    def set_training_batch(self, queries: torch.Tensor, labels: torch.Tensor,
                           mask: torch.Tensor) -> None:
        self.calibration_batch = (queries.clone(), labels.clone(), mask.clone())

    def build_method_context(self, shard: ClientTaskShard, *, global_task_id: int,
                             stage_index: int, round_index: int):
        queries, labels, mask = self.calibration_batch
        diagnostic_shard = replace(shard, train_queries=queries, train_labels=labels,
                                   task_class_mask=mask)
        return super().build_method_context(
            diagnostic_shard, global_task_id=global_task_id,
            stage_index=stage_index, round_index=round_index)
