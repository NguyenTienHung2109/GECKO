"""Numerically checked FedAvg and FedProx primitives."""

from __future__ import annotations

from typing import Dict
from typing import Iterable
from typing import Sequence

import torch
from torch import nn

from gecko.types import LocalUpdateResult
from gecko.validation import AggregationError


def weighted_average(updates: Sequence[LocalUpdateResult]) -> Dict[str, torch.Tensor]:
    usable = sorted(
        (update for update in updates if update.weight > 0),
        key=lambda update: update.client_id,
    )
    if not usable:
        raise AggregationError("No positive-weight client update is available.")
    reference_keys = set(usable[0].shared_state)
    total_weight = sum(update.weight for update in usable)
    output: Dict[str, torch.Tensor] = {}
    for update in usable:
        if set(update.shared_state) != reference_keys:
            raise AggregationError("Client shared-state keys do not align.")
    for key in sorted(reference_keys):
        reference = usable[0].shared_state[key]
        if not reference.is_floating_point():
            raise AggregationError(f"Refusing to average non-floating state element {key}.")
        accumulator = torch.zeros_like(reference, dtype=torch.float64)
        for update in usable:
            tensor = update.shared_state[key]
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise AggregationError(f"Client tensor mismatch for {key}.")
            accumulator.add_(tensor.double(), alpha=update.weight / total_weight)
        output[key] = accumulator.to(reference.dtype)
    return output


def fedprox_penalty(
    model: nn.Module,
    server_state: Dict[str, torch.Tensor],
    shareable_keys: Iterable[str],
    mu: float,
) -> torch.Tensor:
    parameters = dict(model.named_parameters())
    penalty = next(model.parameters()).sum() * 0.0
    for key in shareable_keys:
        if key not in server_state or key not in parameters:
            raise AggregationError(f"FedProx state is missing shared parameter {key}.")
        parameter = parameters[key]
        reference = server_state[key].to(parameter)
        if reference.shape != parameter.shape:
            raise AggregationError(f"FedProx shape mismatch for {key}.")
        penalty = penalty + (parameter - reference).square().sum()
    return 0.5 * mu * penalty
