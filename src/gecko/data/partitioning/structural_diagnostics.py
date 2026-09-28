"""Degree-only structural heterogeneity on the global pre-partition graph."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import math
from typing import Any
from typing import Literal

import torch

from gecko.data.partitioning.base import WeightedLogicalTopology


DIRECT_HSTR_VERSION = "global_degree_role_js_v1"
JSLogPolicy = Literal["base2", "natural_normalized"]


@dataclass(frozen=True)
class DegreeRoleDefinition:
    num_bins: int
    bin_boundaries: torch.Tensor
    role_ids: torch.Tensor
    global_degree: torch.Tensor
    global_role_histogram: torch.Tensor
    quantile_policy: str
    topology_hash: str
    version: str = DIRECT_HSTR_VERSION


@dataclass(frozen=True)
class HStrMetrics:
    h_str: float
    per_client_js: list[float | None]
    per_client_role_histograms: list[list[float] | None]
    global_role_histogram: list[float]
    bin_boundaries: list[float]
    num_bins: int
    js_log_policy: str
    normalization_policy: str
    topology_hash: str
    version: str = DIRECT_HSTR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_degree_roles(
    topology: WeightedLogicalTopology, *, num_bins: int = 3
) -> DegreeRoleDefinition:
    """Assign tied global log-degrees through deterministic quantile boundaries."""

    if num_bins < 2:
        raise ValueError("num_bins must be at least two.")
    degree = torch.tensor(
        [float(weights.sum()) for weights in topology.incident_weights],
        dtype=torch.float64,
    )
    score = torch.log1p(degree)
    if score.numel() == 0:
        boundaries = torch.empty(num_bins - 1, dtype=torch.float64)
        roles = torch.empty(0, dtype=torch.long)
    else:
        quantiles = torch.arange(1, num_bins, dtype=torch.float64) / num_bins
        boundaries = torch.quantile(score, quantiles, interpolation="linear")
        # left-closed tie policy: equal values always share the lowest eligible bin.
        roles = torch.bucketize(score.contiguous(), boundaries.contiguous(), right=False)
    histogram = torch.bincount(roles, minlength=num_bins).double()
    histogram = histogram / max(float(histogram.sum()), 1.0)
    return DegreeRoleDefinition(
        num_bins=num_bins,
        bin_boundaries=boundaries,
        role_ids=roles,
        global_degree=degree,
        global_role_histogram=histogram,
        quantile_policy="global_linear_quantiles_left_tie_v1",
        topology_hash=topology.topology_hash,
    )


def _js(client: torch.Tensor, global_hist: torch.Tensor, policy: JSLogPolicy) -> float:
    midpoint = 0.5 * (client + global_hist)
    client_mask = client > 0
    global_mask = global_hist > 0
    value = 0.5 * (
        torch.sum(client[client_mask] * torch.log(client[client_mask] / midpoint[client_mask]))
        + torch.sum(
            global_hist[global_mask]
            * torch.log(global_hist[global_mask] / midpoint[global_mask])
        )
    )
    normalized = float(value / math.log(2.0))
    if policy not in {"base2", "natural_normalized"}:
        raise ValueError("Unknown JS log policy.")
    return normalized


def evaluate_hstr(
    owner: torch.Tensor,
    roles: DegreeRoleDefinition,
    *,
    num_clients: int,
    js_log_policy: JSLogPolicy = "base2",
) -> HStrMetrics:
    values = owner.detach().cpu().long().flatten()
    if values.shape != roles.role_ids.shape or bool((values < 0).any()) or bool((values >= num_clients).any()):
        raise ValueError("Owner vector is incomplete or incompatible with roles.")
    histograms: list[list[float] | None] = []
    divergences: list[float | None] = []
    for client in range(num_clients):
        selected = values == client
        if not bool(selected.any()):
            histograms.append(None)
            divergences.append(None)
            continue
        counts = torch.bincount(
            roles.role_ids[selected], minlength=roles.num_bins
        ).double()
        histogram = counts / float(counts.sum())
        histograms.append(histogram.tolist())
        divergences.append(_js(histogram, roles.global_role_histogram, js_log_policy))
    if any(value is None for value in divergences):
        raise ValueError("H_str requires every client to own at least one node.")
    h_str = sum(float(value) for value in divergences) / num_clients
    if not -1e-12 <= h_str <= 1.0 + 1e-12:
        raise AssertionError("Normalized JS divergence left [0, 1].")
    return HStrMetrics(
        h_str=max(0.0, min(1.0, h_str)),
        per_client_js=divergences,
        per_client_role_histograms=histograms,
        global_role_histogram=roles.global_role_histogram.tolist(),
        bin_boundaries=roles.bin_boundaries.tolist(),
        num_bins=roles.num_bins,
        js_log_policy=js_log_policy,
        normalization_policy="natural_log_divided_by_log_2",
        topology_hash=roles.topology_hash,
    )


def hstr_pair_swap_delta(
    owner: torch.Tensor,
    roles: DegreeRoleDefinition,
    *,
    num_clients: int,
    first: int,
    second: int,
) -> float:
    """Compute exact H_str delta by updating only the two affected histograms."""

    values = owner.detach().cpu().long()
    first_owner, second_owner = int(values[first]), int(values[second])
    if first_owner == second_owner:
        return 0.0
    affected_before = 0.0
    affected_after = 0.0
    for client in (first_owner, second_owner):
        selected_roles = roles.role_ids[values == client]
        counts = torch.bincount(selected_roles, minlength=roles.num_bins).double()
        affected_before += _js(counts / counts.sum(), roles.global_role_histogram, "base2")
        after = counts.clone()
        outgoing = first if client == first_owner else second
        incoming = second if client == first_owner else first
        after[int(roles.role_ids[outgoing])] -= 1
        after[int(roles.role_ids[incoming])] += 1
        affected_after += _js(after / after.sum(), roles.global_role_histogram, "base2")
    return (affected_after - affected_before) / num_clients
