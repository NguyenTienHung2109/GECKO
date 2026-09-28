"""Paper-faithful POWER trajectory and server-transfer primitives."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


POWER_ZERO_KL_TOLERANCE = 1e-7
POWER_ACCEPTANCE_ABSOLUTE_TOLERANCE = 1e-12
POWER_ACCEPTANCE_RELATIVE_TOLERANCE = 1e-8
POWER_BACKTRACK_FACTOR = 0.5
POWER_MAX_BACKTRACK_STEPS = 12


@dataclass(frozen=True)
class PowerServerTransferReport:
    """Auditable outcome of one deterministic POWER server transfer."""

    disposition: Literal["changed", "zero_signal", "stalled"]
    optimization_mode: str
    acceptance_rule: str
    initial_objective: float
    final_objective: float
    attempted_epochs: int
    accepted_epochs: int
    proposal_attempts: int
    rejected_proposals: int
    backtracking_reductions: int
    accepted_learning_rates: tuple[float, ...]
    exhausted_backtracking: bool

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation for result/checkpoint evidence."""

        return {
            "disposition": self.disposition,
            "optimization_mode": self.optimization_mode,
            "acceptance_rule": self.acceptance_rule,
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "attempted_epochs": self.attempted_epochs,
            "accepted_epochs": self.accepted_epochs,
            "proposal_attempts": self.proposal_attempts,
            "rejected_proposals": self.rejected_proposals,
            "backtracking_reductions": self.backtracking_reductions,
            "accepted_learning_rates": list(self.accepted_learning_rates),
            "exhausted_backtracking": self.exhausted_backtracking,
        }


def power_knn_edges(prototypes: torch.Tensor, *, k: int = 1) -> torch.Tensor:
    """Build Eq. (13)'s deterministic, loop-free symmetric KNN graph."""

    if (
        prototypes.ndim != 2
        or not prototypes.is_floating_point()
        or not torch.isfinite(prototypes).all()
    ):
        raise ValueError("POWER pseudo prototypes must be a finite matrix.")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("POWER KNN k must be positive.")
    node_count = int(prototypes.shape[0])
    if node_count <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    effective_k = min(k, node_count - 1)
    similarity = torch.sigmoid(prototypes.float() @ prototypes.float().t())
    similarity.fill_diagonal_(-float("inf"))
    neighbors = torch.topk(
        similarity, k=effective_k, dim=1, largest=True, sorted=True
    ).indices
    sources = (
        torch.arange(node_count, device=neighbors.device)
        .unsqueeze(1)
        .expand_as(neighbors)
        .reshape(-1)
    )
    directed = torch.stack((sources, neighbors.reshape(-1)), dim=0)
    symmetric = torch.cat((directed, directed.flip(0)), dim=1)
    pairs = torch.unique(symmetric.t().cpu(), dim=0, sorted=True)
    return pairs.t().to(dtype=torch.long).contiguous()


def power_update_trajectory(
    *,
    previous: torch.Tensor | None,
    current_counts: torch.Tensor,
    decay: float,
) -> torch.Tensor:
    """Update Eq. (12)'s cumulative, exponentially decayed label counts."""

    if (
        current_counts.ndim != 2
        or not current_counts.is_floating_point()
        or torch.any(current_counts < 0)
        or not torch.isfinite(current_counts).all()
    ):
        raise ValueError("POWER current trajectory counts must be non-negative.")
    if not 0.0 <= float(decay) <= 1.0:
        raise ValueError("POWER trajectory decay must lie in [0, 1].")
    if previous is None:
        return current_counts.detach().cpu().clone().contiguous()
    if (
        previous.shape != current_counts.shape
        or not previous.is_floating_point()
        or torch.any(previous < 0)
        or not torch.isfinite(previous).all()
    ):
        raise ValueError("POWER previous trajectory is invalid.")
    return (
        float(decay) * previous.detach().cpu()
        + current_counts.detach().cpu()
    ).contiguous()


def power_normalize_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    """Normalize client expertise independently for each class in Eq. (14)."""

    if (
        trajectory.ndim != 2
        or not trajectory.is_floating_point()
        or torch.any(trajectory < 0)
        or not torch.isfinite(trajectory).all()
    ):
        raise ValueError("POWER trajectory must be a non-negative matrix.")
    totals = trajectory.sum(dim=0, keepdim=True)
    return torch.where(totals > 0, trajectory / totals.clamp_min(1e-12), 0.0)


def _model_logits(
    model: nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    queries = torch.arange(features.shape[0], dtype=torch.long, device=features.device)
    forward_queries = getattr(model, "forward_queries", None)
    if not callable(forward_queries):
        raise TypeError("POWER server model must expose forward_queries.")
    logits = forward_queries(features, edge_index, queries, "NC")
    if logits.ndim != 2 or logits.shape[0] != features.shape[0]:
        raise ValueError("POWER server model returned invalid NC logits.")
    return logits


def _load_shared_parameters(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    named = dict(model.named_parameters())
    if set(state) != set(named):
        raise ValueError("POWER shared state does not match model parameters.")
    with torch.no_grad():
        for name, parameter in named.items():
            value = state[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f"POWER shared tensor {name!r} is incompatible.")
            parameter.copy_(value.to(parameter.device))


def power_extract_shared_parameters(
    model: nn.Module, names: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Extract the requested model parameters as CPU tensors."""

    named = dict(model.named_parameters())
    if set(names) != set(named):
        raise ValueError("POWER requested state does not match model parameters.")
    return {
        name: named[name].detach().cpu().clone().contiguous() for name in names
    }


def power_trajectory_kl_loss(
    *,
    global_model: nn.Module,
    local_models: Sequence[nn.Module],
    features: torch.Tensor,
    labels: torch.Tensor,
    edge_index: torch.Tensor,
    normalized_trajectory: torch.Tensor,
    task_class_masks: Mapping[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute Eq. (14): class/client-weighted KL(global || local)."""

    if labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("POWER pseudo labels must be one-dimensional int64.")
    if labels.shape[0] != features.shape[0] or labels.numel() == 0:
        raise ValueError("POWER pseudo labels must align with non-empty features.")
    if normalized_trajectory.ndim != 2 or normalized_trajectory.shape[0] != len(
        local_models
    ):
        raise ValueError("POWER normalized trajectory/client shape is invalid.")
    if torch.any(labels < 0) or torch.any(labels >= normalized_trajectory.shape[1]):
        raise ValueError("POWER pseudo label lies outside trajectory classes.")

    global_logits = _model_logits(global_model, features, edge_index)
    if task_class_masks:
        per_node_masks = []
        for label in labels.tolist():
            candidates = [
                mask
                for mask in task_class_masks.values()
                if int(label) < mask.numel() and bool(mask[int(label)])
            ]
            if len(candidates) != 1:
                raise ValueError("POWER Task-IL label does not identify exactly one task head.")
            per_node_masks.append(candidates[0])
        node_masks = torch.stack(per_node_masks).to(global_logits.device)
        if node_masks.shape != global_logits.shape:
            raise ValueError("POWER Task-IL masks do not align with server logits.")
        global_logits = global_logits.masked_fill(~node_masks, -1e12)
    else:
        node_masks = None
    global_log_probabilities = torch.log_softmax(global_logits, dim=1)
    global_probabilities = global_log_probabilities.exp()
    objective = torch.zeros(
        (), dtype=global_logits.dtype, device=global_logits.device
    )
    for client_index, local_model in enumerate(local_models):
        with torch.no_grad():
            local_logits = _model_logits(local_model, features, edge_index)
            if node_masks is not None:
                local_logits = local_logits.masked_fill(~node_masks, -1e12)
            local_log_probabilities = torch.log_softmax(local_logits, dim=1)
        per_node_kl = torch.sum(
            global_probabilities
            * (global_log_probabilities - local_log_probabilities),
            dim=1,
        )
        node_weights = normalized_trajectory[client_index].to(
            device=labels.device, dtype=per_node_kl.dtype
        )[labels]
        objective = objective + torch.sum(node_weights * per_node_kl)
    return objective


def power_server_transfer(
    *,
    model_template: nn.Module,
    averaged_state: Mapping[str, torch.Tensor],
    local_states: Sequence[Mapping[str, torch.Tensor]],
    features: torch.Tensor,
    labels: torch.Tensor,
    edge_index: torch.Tensor,
    trajectory: torch.Tensor,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device | str,
    task_class_masks: Mapping[int, torch.Tensor] | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    tuple[float, ...],
    PowerServerTransferReport,
]:
    """Fine-tune the FedAvg model using POWER's Eq. (14) server objective."""

    if epochs <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("POWER server optimizer controls are invalid.")
    target_device = torch.device(device)
    global_model = copy.deepcopy(model_template).to(target_device)
    _load_shared_parameters(global_model, averaged_state)
    local_models: list[nn.Module] = []
    for state in local_states:
        local_model = copy.deepcopy(model_template).to(target_device)
        _load_shared_parameters(local_model, state)
        local_model.eval()
        for parameter in local_model.parameters():
            parameter.requires_grad_(False)
        local_models.append(local_model)

    server_features = features.to(target_device)
    server_labels = labels.to(target_device, dtype=torch.long)
    server_edges = edge_index.to(target_device, dtype=torch.long)
    normalized = power_normalize_trajectory(trajectory).to(target_device)
    # POWER's public code optimizes a stochastic train-mode discrepancy, while
    # the paper specifies the trajectory-weighted KL in Eq. (14).  A train-mode
    # gradient followed by an eval-mode acceptance check compares two different
    # objectives whenever the backbone contains Dropout or BatchNorm.  Keep the
    # paper objective deterministic here: eval mode still permits parameter
    # gradients, disables Dropout, and leaves client-local normalization buffers
    # untouched.
    global_model.eval()
    zero_signal_probe = power_trajectory_kl_loss(
        global_model=global_model,
        local_models=local_models,
        features=server_features,
        labels=server_labels,
        edge_index=server_edges,
        normalized_trajectory=normalized,
        task_class_masks=task_class_masks,
    )
    if not torch.isfinite(zero_signal_probe):
        raise FloatingPointError("POWER server transfer loss became non-finite.")
    initial_value = float(zero_signal_probe.detach().cpu())
    if initial_value < -POWER_ZERO_KL_TOLERANCE:
        raise FloatingPointError("POWER trajectory KL became materially negative.")
    if abs(initial_value) <= POWER_ZERO_KL_TOLERANCE:
        reported_value = max(0.0, initial_value)
        trace = tuple(reported_value for _ in range(int(epochs) + 1))
        report = PowerServerTransferReport(
            disposition="zero_signal",
            optimization_mode="eval_deterministic",
            acceptance_rule="strict_scale_aware_kl_descent",
            initial_objective=reported_value,
            final_objective=reported_value,
            attempted_epochs=0,
            accepted_epochs=0,
            proposal_attempts=0,
            rejected_proposals=0,
            backtracking_reductions=0,
            accepted_learning_rates=(),
            exhausted_backtracking=False,
        )
        return (
            power_extract_shared_parameters(global_model, tuple(averaged_state)),
            trace,
            report,
        )

    optimizer = torch.optim.Adam(
        global_model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    losses: list[float] = [initial_value]
    current_value = initial_value
    attempted_epochs = 0
    accepted_epochs = 0
    proposal_attempts = 0
    rejected_proposals = 0
    backtracking_reductions = 0
    accepted_learning_rates: list[float] = []
    exhausted_backtracking = False
    for epoch in range(int(epochs)):
        attempted_epochs += 1
        global_model.eval()
        optimizer.zero_grad(set_to_none=True)
        objective = power_trajectory_kl_loss(
            global_model=global_model,
            local_models=local_models,
            features=server_features,
            labels=server_labels,
            edge_index=server_edges,
            normalized_trajectory=normalized,
            task_class_masks=task_class_masks,
        )
        if not torch.isfinite(objective):
            raise FloatingPointError("POWER server transfer loss became non-finite.")
        objective.backward()
        gradient_square_norm = torch.zeros((), device=target_device)
        for parameter in global_model.parameters():
            if parameter.grad is not None:
                gradient_square_norm = gradient_square_norm + torch.sum(
                    parameter.grad.detach().float().square()
                )
        if not torch.isfinite(gradient_square_norm):
            raise FloatingPointError("POWER server gradient became non-finite.")

        before_model = {
            name: value.detach().clone()
            for name, value in global_model.state_dict().items()
        }
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        accepted = False
        for reduction in range(POWER_MAX_BACKTRACK_STEPS + 1):
            if reduction:
                global_model.load_state_dict(before_model, strict=True)
                optimizer.load_state_dict(before_optimizer)
            candidate_learning_rate = float(learning_rate) * (
                POWER_BACKTRACK_FACTOR**reduction
            )
            for group in optimizer.param_groups:
                group["lr"] = candidate_learning_rate
            optimizer.step()
            proposal_attempts += 1
            global_model.eval()
            with torch.no_grad():
                candidate_objective = power_trajectory_kl_loss(
                    global_model=global_model,
                    local_models=local_models,
                    features=server_features,
                    labels=server_labels,
                    edge_index=server_edges,
                    normalized_trajectory=normalized,
                    task_class_masks=task_class_masks,
                )
            if torch.isfinite(candidate_objective):
                candidate_value = float(candidate_objective.cpu())
                required_decrease = max(
                    POWER_ACCEPTANCE_ABSOLUTE_TOLERANCE,
                    abs(current_value) * POWER_ACCEPTANCE_RELATIVE_TOLERANCE,
                )
                if candidate_value < current_value - required_decrease:
                    current_value = candidate_value
                    losses.append(candidate_value)
                    accepted_epochs += 1
                    backtracking_reductions += reduction
                    accepted_learning_rates.append(candidate_learning_rate)
                    accepted = True
                    break
            rejected_proposals += 1
        if not accepted:
            global_model.load_state_dict(before_model, strict=True)
            optimizer.load_state_dict(before_optimizer)
            exhausted_backtracking = True
            losses.extend(current_value for _ in range(int(epochs) - epoch))
            break
    if len(losses) < int(epochs) + 1:
        losses.extend(current_value for _ in range(int(epochs) + 1 - len(losses)))
    disposition: Literal["changed", "zero_signal", "stalled"] = (
        "changed" if accepted_epochs else "stalled"
    )
    report = PowerServerTransferReport(
        disposition=disposition,
        optimization_mode="eval_deterministic",
        acceptance_rule="strict_scale_aware_kl_descent",
        initial_objective=initial_value,
        final_objective=current_value,
        attempted_epochs=attempted_epochs,
        accepted_epochs=accepted_epochs,
        proposal_attempts=proposal_attempts,
        rejected_proposals=rejected_proposals,
        backtracking_reductions=backtracking_reductions,
        accepted_learning_rates=tuple(accepted_learning_rates),
        exhausted_backtracking=exhausted_backtracking,
    )
    return (
        power_extract_shared_parameters(global_model, tuple(averaged_state)),
        tuple(losses),
        report,
    )
