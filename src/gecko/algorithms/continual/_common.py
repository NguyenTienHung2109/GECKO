from __future__ import annotations

import torch
import torch.nn.functional as F

def _classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim > 1:
        valid = labels >= 0
        return F.binary_cross_entropy_with_logits(logits[valid], labels[valid].float())
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(logits.reshape(-1), labels.float().reshape(-1))
    return F.cross_entropy(logits, labels.long())


def _masked_logits(
    logits: torch.Tensor, class_mask: torch.Tensor | None
) -> torch.Tensor:
    if (
        class_mask is None
        or logits.ndim < 2
        or logits.shape[-1] != class_mask.shape[0]
    ):
        return logits
    masked = logits.clone()
    masked[..., ~class_mask.to(logits.device)] = -1e12
    return masked


