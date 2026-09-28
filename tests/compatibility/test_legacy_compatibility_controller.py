from __future__ import annotations

import pytest
import torch
from torch import nn

from gecko.compat import LegacyEpochController


@pytest.mark.parametrize(
    ("objective", "values"),
    [
        ("min", [3.0, 2.0, 2.5, 2.6]),
        ("max", [0.1, 0.4, 0.3, 0.2]),
    ],
)
def test_controller_forces_early_stop_and_restores_competitive_checkpoint(
    objective, values
):
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=objective, factor=0.5, patience=0, min_lr=0.25
    )
    controller = LegacyEpochController(
        optimizer, scheduler, objective=objective
    )
    decisions = []
    for epoch, value in enumerate(values):
        with torch.no_grad():
            model.weight.fill_(epoch + 1.0)
        decisions.append(controller.step(model, value))

    assert [decision.learning_rate for decision in decisions] == [1.0, 1.0, 0.5, 0.25]
    assert decisions[-1].continue_training is False
    controller.restore(model)
    assert model.weight.item() == pytest.approx(2.0)


def test_controller_rejects_restore_without_validation_checkpoint():
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    controller = LegacyEpochController(optimizer, scheduler, objective="min")
    with pytest.raises(RuntimeError, match="before observing"):
        controller.restore(model)
