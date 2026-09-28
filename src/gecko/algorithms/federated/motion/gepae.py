"""Server-side G-EPAE parameter compatibility for MOTION.

This is a clean-room adaptation of the public MOTION parameter-conflict
balancing rule.  It operates only on model deltas and never receives client
graphs, features, or labels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def _ratio(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1).")
    return value


def motion_minmax_normalize(
    values: torch.Tensor, *, dim: int
) -> torch.Tensor:
    """Min-max normalize a floating tensor with constant slices mapped to zero."""

    if not torch.is_tensor(values) or not values.is_floating_point():
        raise TypeError("MOTION normalization requires a floating tensor.")
    if values.numel() == 0:
        raise ValueError("MOTION normalization requires a non-empty tensor.")
    if dim < -values.ndim or dim >= values.ndim:
        raise ValueError("MOTION normalization dimension is out of range.")
    minimum = values.amin(dim=dim, keepdim=True)
    maximum = values.amax(dim=dim, keepdim=True)
    denominator = (maximum - minimum).clamp_min(1e-12)
    return (values - minimum) / denominator


def motion_percentile_clamp(
    values: torch.Tensor,
    *,
    min_ratio: float,
    max_ratio: float,
) -> torch.Tensor:
    """Clamp a vector or each matrix row to index-based MOTION quantiles."""

    if not torch.is_tensor(values) or not values.is_floating_point():
        raise TypeError("MOTION clamping requires a floating tensor.")
    if values.ndim not in {1, 2} or values.numel() == 0:
        raise ValueError("MOTION clamping requires a non-empty vector or matrix.")
    min_ratio = _ratio(min_ratio, name="min_ratio")
    max_ratio = _ratio(max_ratio, name="max_ratio")
    if min_ratio + max_ratio >= 1.0:
        raise ValueError("MOTION clamp ratios must sum to less than one.")

    width = int(values.shape[-1])
    lower_index = min(width - 1, int(width * min_ratio))
    upper_index = max(0, int(width * (1.0 - max_ratio) - 1))
    if lower_index > upper_index:
        raise ValueError("MOTION clamp quantiles are reversed.")
    sorted_values = values.sort(dim=-1).values
    if values.ndim == 1:
        lower = sorted_values[lower_index]
        upper = sorted_values[upper_index]
    else:
        lower = sorted_values[:, lower_index].unsqueeze(1)
        upper = sorted_values[:, upper_index].unsqueeze(1)
    return torch.minimum(torch.maximum(values, lower), upper)


def motion_pcb_merge(
    task_vectors: torch.Tensor,
    *,
    pcb_ratio: float = 0.1,
    pcb_min_ratio: float = 0.0001,
    pcb_max_ratio: float = 0.0001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge weighted client deltas with MOTION's compatibility balance rule.

    Returns the merged delta, magnitude-clamped client deltas, and the
    per-client/per-parameter compatibility scales.
    """

    if (
        not torch.is_tensor(task_vectors)
        or not task_vectors.is_floating_point()
        or task_vectors.ndim != 2
        or task_vectors.shape[0] == 0
        or task_vectors.shape[1] == 0
    ):
        raise ValueError(
            "MOTION task vectors must be a non-empty floating [clients, parameters] tensor."
        )
    if not bool(torch.isfinite(task_vectors).all()):
        raise ValueError("MOTION task vectors must be finite.")
    pcb_ratio = _ratio(pcb_ratio, name="pcb_ratio")
    if pcb_ratio <= 0.0:
        raise ValueError("pcb_ratio must be positive.")
    pcb_min_ratio = _ratio(pcb_min_ratio, name="pcb_min_ratio")
    pcb_max_ratio = _ratio(pcb_max_ratio, name="pcb_max_ratio")
    if pcb_min_ratio + pcb_max_ratio >= 1.0:
        raise ValueError("MOTION magnitude clamp ratios must sum to less than one.")

    client_count = int(task_vectors.shape[0])
    absolute = motion_percentile_clamp(
        task_vectors.abs(),
        min_ratio=pcb_min_ratio,
        max_ratio=pcb_max_ratio,
    )
    clamped = task_vectors.sign() * absolute
    self_compatibility = motion_minmax_normalize(absolute, dim=1).square()
    self_activation = torch.exp(client_count * self_compatibility)
    cross_compatibility = torch.tanh(
        task_vectors * task_vectors.sum(dim=0, keepdim=True)
    )
    compatibility = self_activation * cross_compatibility
    selected = motion_percentile_clamp(
        compatibility,
        min_ratio=1.0 - pcb_ratio,
        max_ratio=0.0,
    )
    scales = motion_minmax_normalize(selected, dim=1)
    denominator = scales.sum(dim=0).clamp_min(1e-12)
    merged = (clamped * scales).sum(dim=0) / denominator
    if not bool(torch.isfinite(merged).all()):
        raise RuntimeError("MOTION compatibility merge produced a non-finite delta.")
    return merged, clamped, scales


def motion_gepae_aggregate(
    global_state: Mapping[str, torch.Tensor],
    client_states: Mapping[int, Mapping[str, torch.Tensor]],
    client_weights: Mapping[int, int],
    *,
    pcb_ratio: float = 0.1,
    pcb_min_ratio: float = 0.0001,
    pcb_max_ratio: float = 0.0001,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Apply G-EPAE to full shared states and return state plus scale matrix."""

    if (
        not global_state
        or not client_states
        or set(client_states) != set(client_weights)
    ):
        raise ValueError(
            "MOTION aggregation requires aligned global/client states and weights."
        )
    keys = tuple(sorted(global_state))
    if any(tuple(sorted(state)) != keys for state in client_states.values()):
        raise ValueError("MOTION client state keys must match the global state.")
    total_weight = sum(int(weight) for weight in client_weights.values())
    if total_weight <= 0 or any(
        isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0
        for weight in client_weights.values()
    ):
        raise ValueError("MOTION client weights must be positive integers.")

    weighted_vectors: list[torch.Tensor] = []
    for client_id in sorted(client_states):
        pieces: list[torch.Tensor] = []
        for key in keys:
            reference = global_state[key]
            value = client_states[client_id][key]
            if (
                not torch.is_tensor(reference)
                or not reference.is_floating_point()
                or value.shape != reference.shape
                or value.dtype != reference.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"MOTION state tensor {key!r} is invalid.")
            pieces.append((value.to(reference.device) - reference).reshape(-1))
        weight = float(client_weights[client_id]) / total_weight
        weighted_vectors.append(torch.cat(pieces) * weight)

    merged, _, scales = motion_pcb_merge(
        torch.stack(weighted_vectors),
        pcb_ratio=pcb_ratio,
        pcb_min_ratio=pcb_min_ratio,
        pcb_max_ratio=pcb_max_ratio,
    )
    output: dict[str, torch.Tensor] = {}
    offset = 0
    for key in keys:
        reference = global_state[key]
        size = reference.numel()
        update = merged[offset : offset + size].reshape(reference.shape)
        output[key] = (reference + update).detach().clone().contiguous()
        offset += size
    if offset != merged.numel():
        raise RuntimeError("MOTION flattened update did not consume every parameter.")
    return output, scales.detach().clone().contiguous()
