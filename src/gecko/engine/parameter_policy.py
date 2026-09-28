"""Explicit shared/local parameter and buffer policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from typing import Tuple

import torch
from torch import nn

from gecko.validation import AggregationError


@dataclass(frozen=True)
class SharedParameterPolicy:
    """Share trainable floating parameters and retain all buffers locally."""

    local_prefixes: Tuple[str, ...] = ("local_", "task_masks", "packnet_masks")
    batch_norm_policy: str = "local_retention"

    def shareable_keys(self, model: nn.Module) -> Tuple[str, ...]:
        keys = []
        for name, parameter in model.named_parameters():
            if any(name.startswith(prefix) or f".{prefix}" in name for prefix in self.local_prefixes):
                continue
            if not parameter.is_floating_point():
                raise AggregationError(f"Trainable parameter {name} is not floating point.")
            keys.append(name)
        return tuple(keys)

    def extract(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        parameters = dict(model.named_parameters())
        return {
            # Keep runtime state beside the model. Wire accounting is logical
            # and does not require staging every round through host memory.
            key: parameters[key].detach().clone().contiguous()
            for key in self.shareable_keys(model)
        }

    def load(self, model: nn.Module, shared_state: Dict[str, torch.Tensor]) -> None:
        parameters = dict(model.named_parameters())
        expected = set(self.shareable_keys(model))
        if set(shared_state) != expected:
            missing = sorted(expected - set(shared_state))
            extra = sorted(set(shared_state) - expected)
            raise AggregationError(f"Shared-state key mismatch; missing={missing}, extra={extra}")
        with torch.no_grad():
            for key, source in shared_state.items():
                target = parameters[key]
                if source.shape != target.shape or source.dtype != target.dtype:
                    raise AggregationError(
                        f"Incompatible shared parameter {key}: source={source.shape}/{source.dtype}, "
                        f"target={target.shape}/{target.dtype}"
                    )
                target.copy_(source.to(target.device))
