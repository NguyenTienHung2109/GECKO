"""Fail-closed task-head objectives shared by NC Task-IL adaptations."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def task_aware_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_class_masks: Mapping[int, torch.Tensor],
) -> torch.Tensor:
    """Average CE while selecting exactly one immutable task head per label."""

    if logits.ndim != 2 or labels.dtype != torch.long or labels.ndim != 1:
        raise ValueError("Task-aware CE requires [samples, classes] logits and int64 labels.")
    if logits.shape[0] != labels.shape[0] or labels.numel() == 0:
        raise ValueError("Task-aware CE requires aligned non-empty samples.")
    covered = torch.zeros(labels.numel(), dtype=torch.long, device=logits.device)
    terms: list[tuple[torch.Tensor, int]] = []
    for task_id, raw_mask in sorted(task_class_masks.items()):
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
            raise ValueError("Task class-mask IDs must be non-negative integers.")
        if (
            not torch.is_tensor(raw_mask)
            or raw_mask.dtype != torch.bool
            or raw_mask.ndim != 1
            or raw_mask.numel() != logits.shape[1]
            or not bool(raw_mask.any())
        ):
            raise ValueError("Task class mask does not align with logits.")
        mask = raw_mask.to(logits.device)
        if int(labels.min()) < 0 or int(labels.max()) >= mask.numel():
            raise ValueError("Task-aware CE label lies outside the model output.")
        positions = mask[labels].nonzero(as_tuple=False).reshape(-1)
        if positions.numel():
            covered[positions] += 1
            masked = logits[positions].masked_fill(~mask, -1e12)
            terms.append((F.cross_entropy(masked, labels[positions]), int(positions.numel())))
    if not bool((covered == 1).all()):
        raise ValueError("Replay labels are not uniquely covered by task heads.")
    return sum(loss * count for loss, count in terms) / labels.numel()


__all__ = ["task_aware_cross_entropy"]
