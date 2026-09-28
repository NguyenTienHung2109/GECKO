from __future__ import annotations

import pytest
import torch
from torch import nn

import gecko.algorithms.federated.power.server as power_server_module
from gecko.algorithms.federated.power.server import POWER_ZERO_KL_TOLERANCE
from gecko.algorithms.federated.power.server import power_knn_edges
from gecko.algorithms.federated.power.server import power_normalize_trajectory
from gecko.algorithms.federated.power.server import power_server_transfer
from gecko.algorithms.federated.power.server import power_trajectory_kl_loss
from gecko.algorithms.federated.power.server import power_update_trajectory


class _TinyNC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward_queries(self, features, edge_index, queries, problem_type):
        assert problem_type == "NC"
        return self.linear(features)[queries]


class _DropoutNC(nn.Module):
    observed_training_modes: list[bool] = []

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        self.dropout = nn.Dropout(p=0.75)

    def forward_queries(self, features, edge_index, queries, problem_type):
        assert problem_type == "NC"
        type(self).observed_training_modes.append(bool(self.training))
        return self.dropout(self.linear(features))[queries]


def _state(weight):
    return {"linear.weight": torch.tensor(weight, dtype=torch.float32)}


def test_power_eq13_knn_is_loop_free_symmetric_and_deterministic():
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    edges = power_knn_edges(features, k=1)
    pairs = {tuple(pair) for pair in edges.t().tolist()}
    assert all(source != target for source, target in pairs)
    assert all((target, source) in pairs for source, target in pairs)
    assert torch.equal(edges, power_knn_edges(features, k=1))


def test_power_eq12_trajectory_is_cumulative_decay_and_class_normalized():
    previous = torch.tensor([[4.0, 0.0], [0.0, 2.0]])
    current = torch.tensor([[0.0, 3.0], [5.0, 0.0]])
    trajectory = power_update_trajectory(
        previous=previous, current_counts=current, decay=0.25
    )
    assert torch.equal(trajectory, torch.tensor([[1.0, 3.0], [5.0, 0.5]]))
    normalized = power_normalize_trajectory(trajectory)
    assert torch.allclose(normalized.sum(dim=0), torch.ones(2))
    assert torch.allclose(normalized[:, 0], torch.tensor([1.0 / 6.0, 5.0 / 6.0]))


def test_power_eq14_is_kl_global_to_local_and_server_optimizes_it():
    template = _TinyNC()
    averaged = _state([[0.0, 0.0], [0.0, 0.0]])
    local_states = (
        _state([[4.0, 0.0], [-4.0, 0.0]]),
        _state([[0.0, -4.0], [0.0, 4.0]]),
    )
    global_model = _TinyNC()
    global_model.load_state_dict(averaged)
    local_models = [_TinyNC(), _TinyNC()]
    for model, state in zip(local_models, local_states, strict=True):
        model.load_state_dict(state)
    features = torch.eye(2)
    labels = torch.tensor([0, 1])
    edges = torch.tensor([[0, 1], [1, 0]])
    trajectory = torch.tensor([[5.0, 0.0], [0.0, 7.0]])
    initial = float(
        power_trajectory_kl_loss(
            global_model=global_model,
            local_models=local_models,
            features=features,
            labels=labels,
            edge_index=edges,
            normalized_trajectory=power_normalize_trajectory(trajectory),
        )
    )
    trained, losses, report = power_server_transfer(
        model_template=template,
        averaged_state=averaged,
        local_states=local_states,
        features=features,
        labels=labels,
        edge_index=edges,
        trajectory=trajectory,
        epochs=20,
        learning_rate=0.1,
        weight_decay=0.0,
        device="cpu",
    )
    assert len(losses) == 21
    assert losses[0] == initial
    assert losses[-1] < losses[0]
    assert report.disposition == "changed"
    assert report.accepted_epochs == 20
    assert trained["linear.weight"][0, 0] > 0
    assert trained["linear.weight"][1, 1] > 0


def test_power_server_backtracks_an_initially_kl_increasing_adam_step():
    template = _TinyNC()
    averaged = _state([[0.0, 0.0], [0.0, 0.0]])
    local_states = (
        _state([[0.1, 0.0], [-0.1, 0.0]]),
        _state([[0.0, -0.1], [0.0, 0.1]]),
    )
    features = torch.eye(2)
    labels = torch.tensor([0, 1])
    edges = torch.tensor([[0, 1], [1, 0]])
    trajectory = torch.tensor([[5.0, 0.0], [0.0, 7.0]])

    trained, losses, report = power_server_transfer(
        model_template=template,
        averaged_state=averaged,
        local_states=local_states,
        features=features,
        labels=labels,
        edge_index=edges,
        trajectory=trajectory,
        epochs=3,
        learning_rate=100.0,
        weight_decay=5e-4,
        device="cpu",
    )

    assert len(losses) == 4
    assert losses[-1] < losses[0]
    assert all(current <= previous for previous, current in zip(losses, losses[1:]))
    assert report.disposition == "changed"
    assert report.accepted_epochs == 3
    assert report.rejected_proposals > 0
    assert report.backtracking_reductions > 0
    assert not torch.equal(trained["linear.weight"], averaged["linear.weight"])


def test_power_server_uses_task_masks_for_post_step_acceptance(monkeypatch):
    template = _TinyNC()
    averaged = _state([[0.0, 0.0], [0.0, 0.0]])
    local_states = (
        _state([[4.0, 0.0], [-4.0, 0.0]]),
        _state([[0.0, -4.0], [0.0, 4.0]]),
    )
    features = torch.eye(2)
    labels = torch.tensor([0, 1])
    edges = torch.tensor([[0, 1], [1, 0]])
    trajectory = torch.tensor([[5.0, 0.0], [0.0, 7.0]])
    task_masks = {0: torch.ones(2, dtype=torch.bool)}
    observed_masks = []
    original = power_server_module.power_trajectory_kl_loss

    def recording_loss(**kwargs):
        observed_masks.append(kwargs.get("task_class_masks"))
        return original(**kwargs)

    monkeypatch.setattr(
        power_server_module, "power_trajectory_kl_loss", recording_loss
    )
    power_server_transfer(
        model_template=template,
        averaged_state=averaged,
        local_states=local_states,
        features=features,
        labels=labels,
        edge_index=edges,
        trajectory=trajectory,
        epochs=2,
        learning_rate=0.1,
        weight_decay=0.0,
        device="cpu",
        task_class_masks=task_masks,
    )

    assert len(observed_masks) >= 3
    assert all(value is task_masks for value in observed_masks)


def test_power_server_rolls_back_when_kl_does_not_measurably_decrease():
    template = _TinyNC()
    averaged = _state([[0.0, 0.0], [0.0, 0.0]])
    local_states = (
        _state([[4.0, 0.0], [-4.0, 0.0]]),
        _state([[0.0, -4.0], [0.0, 4.0]]),
    )
    features = torch.eye(2)
    labels = torch.tensor([0, 1])
    edges = torch.tensor([[0, 1], [1, 0]])
    trajectory = torch.tensor([[5.0, 0.0], [0.0, 7.0]])

    trained, losses, report = power_server_transfer(
        model_template=template,
        averaged_state=averaged,
        local_states=local_states,
        features=features,
        labels=labels,
        edge_index=edges,
        trajectory=trajectory,
        epochs=3,
        learning_rate=1e-12,
        weight_decay=5e-4,
        device="cpu",
    )

    assert losses == pytest.approx((losses[0], losses[0], losses[0], losses[0]))
    assert report.disposition == "stalled"
    assert report.accepted_epochs == 0
    assert report.exhausted_backtracking is True
    assert torch.equal(trained["linear.weight"], averaged["linear.weight"])


def test_power_zero_kl_server_transfer_is_a_noop_with_weight_decay():
    template = _TinyNC()
    aligned = _state([[1.5, -0.5], [-0.25, 0.75]])
    nearly_aligned = _state([[1.5001, -0.5], [-0.25, 0.75]])
    features = torch.eye(2)
    labels = torch.tensor([0, 1])
    edges = torch.tensor([[0, 1], [1, 0]])
    trajectory = torch.ones((2, 2))

    global_model = _TinyNC()
    global_model.load_state_dict(aligned)
    local_models = [_TinyNC(), _TinyNC()]
    local_models[0].load_state_dict(aligned)
    local_models[1].load_state_dict(nearly_aligned)
    initial = power_trajectory_kl_loss(
        global_model=global_model,
        local_models=local_models,
        features=features,
        labels=labels,
        edge_index=edges,
        normalized_trajectory=power_normalize_trajectory(trajectory),
    )
    assert 0.0 < abs(float(initial)) <= POWER_ZERO_KL_TOLERANCE

    trained, losses, report = power_server_transfer(
        model_template=template,
        averaged_state=aligned,
        local_states=(aligned, nearly_aligned),
        features=features,
        labels=labels,
        edge_index=edges,
        trajectory=trajectory,
        epochs=3,
        learning_rate=0.01,
        weight_decay=5e-4,
        device="cpu",
    )

    assert len(losses) == 4
    assert losses == pytest.approx((losses[0],) * 4, abs=1e-12)
    assert report.disposition == "zero_signal"
    assert report.attempted_epochs == 0
    assert torch.equal(trained["linear.weight"], aligned["linear.weight"])


def test_power_server_transfer_is_deterministic_and_never_enables_dropout():
    template = _DropoutNC()
    averaged = _state([[0.0, 0.0], [0.0, 0.0]])
    local_states = (
        _state([[4.0, 0.0], [-4.0, 0.0]]),
        _state([[0.0, -4.0], [0.0, 4.0]]),
    )
    arguments = {
        "model_template": template,
        "averaged_state": averaged,
        "local_states": local_states,
        "features": torch.eye(2),
        "labels": torch.tensor([0, 1]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "trajectory": torch.tensor([[5.0, 0.0], [0.0, 7.0]]),
        "epochs": 3,
        "learning_rate": 0.1,
        "weight_decay": 5e-4,
        "device": "cpu",
    }

    _DropoutNC.observed_training_modes = []
    first_state, first_losses, first_report = power_server_transfer(**arguments)
    second_state, second_losses, second_report = power_server_transfer(**arguments)

    assert _DropoutNC.observed_training_modes
    assert not any(_DropoutNC.observed_training_modes)
    assert first_losses == second_losses
    assert first_report == second_report
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)
