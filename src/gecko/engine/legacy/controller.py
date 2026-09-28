"""Independent epoch controller for the BeGin compatibility diagnostic."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass(frozen=True)
class EpochDecision:
    improved: bool
    continue_training: bool
    learning_rate: float
    best_value: float


class LegacyEpochController:
    """Mirror BeGin's validation scheduler and best-checkpoint contract.

    This controller is used only by the versioned compatibility diagnostic. It
    deliberately does not participate in UEFA's authoritative federated loop,
    where validation labels remain central and the frozen training protocol is
    fixed-epoch.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        *,
        objective: Literal["min", "max"],
    ) -> None:
        if objective not in {"min", "max"}:
            raise ValueError("Legacy validation objective must be 'min' or 'max'.")
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.objective = objective
        self.best_value = float("inf") if objective == "min" else float("-inf")
        self.best_state: dict[str, torch.Tensor] | None = None

    def step(self, model: nn.Module, validation_value: float) -> EpochDecision:
        improved = (
            validation_value < self.best_value
            if self.objective == "min"
            else validation_value > self.best_value
        )
        if improved:
            self.best_value = float(validation_value)
            self.best_state = copy.deepcopy(model.state_dict())
        self.scheduler.step(float(validation_value))
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        minimum = float(self.scheduler.min_lrs[0])
        continue_training = not (-1e-9 < learning_rate - minimum < 1e-9)
        return EpochDecision(
            improved=improved,
            continue_training=continue_training,
            learning_rate=learning_rate,
            best_value=self.best_value,
        )

    def restore(self, model: nn.Module) -> None:
        if self.best_state is None:
            raise RuntimeError("Cannot restore before observing a validation value.")
        model.load_state_dict(self.best_state)
