from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from gecko.algorithms.continual.ewc import EWCAlgorithm
from gecko.algorithms.continual.lwf import LwFAlgorithm
from gecko.algorithms.continual.lwf import LwFClassILAlgorithm
from gecko.algorithms.continual.mas import MASAlgorithm
from gecko.algorithms.continual.ergnn import ERGNNAlgorithm
from gecko.algorithms.catalog import MethodRegistry


def _forward(model: nn.Module, queries: torch.Tensor) -> torch.Tensor:
    return model(queries)


def _model() -> nn.Module:
    model = nn.Linear(2, 3, bias=True)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]])
        )
        model.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    return model


def test_lwf_matches_original_class_il_distillation_formula_and_task_boundary():
    model = _model()
    queries = torch.tensor([[1.0, 0.5], [-0.5, 2.0]])
    labels = torch.tensor([0, 1])
    mask = torch.tensor([True, True, False])
    algorithm = LwFAlgorithm(regularization=1.7, temperature=2.0)
    algorithm.after_task(model, _forward, queries, labels, 0, mask)
    teacher_state = copy.deepcopy(algorithm.state["teacher"])

    with torch.no_grad():
        model.weight.add_(0.15)
    logits = model(queries)
    actual = algorithm.additional_loss(model, _forward, queries, logits, labels, 1)

    teacher = copy.deepcopy(model)
    teacher.load_state_dict(teacher_state)
    teacher_probabilities = F.softmax(teacher(queries)[..., mask] / 2.0, dim=-1)
    expected = -1.7 * (
        teacher_probabilities * F.log_softmax(logits[..., mask] / 2.0, dim=-1)
    ).sum(dim=-1).mean()
    assert torch.allclose(actual, expected)

    algorithm.after_update(model, queries, labels, 1)
    assert algorithm.state["consolidated_task_ids"] == (0,)
    for key in teacher_state:
        assert torch.equal(algorithm.state["teacher"][key], teacher_state[key])


def test_lwf_matches_original_binary_lp_distillation_formula():
    model = nn.Linear(2, 1, bias=True)
    queries = torch.tensor([[1.0, 0.5], [-0.5, 2.0]])
    labels = torch.tensor([1.0, 0.0])
    algorithm = LwFAlgorithm(
        regularization=1.3,
        temperature=2.0,
        problem_type="LP",
        incremental_setting="domain",
    )
    algorithm.after_task(model, _forward, queries, labels, 0)
    teacher_state = copy.deepcopy(algorithm.state["teacher"])
    with torch.no_grad():
        model.weight.add_(0.2)
    logits = model(queries).reshape(-1)
    actual = algorithm.additional_loss(model, _forward, queries, logits, labels, 1)

    teacher = copy.deepcopy(model)
    teacher.load_state_dict(teacher_state)
    teacher_scores = teacher(queries).reshape(-1)
    scale = 4.0
    expected = -1.3 * (
        torch.sigmoid(teacher_scores / scale) * F.logsigmoid(logits / scale)
        + torch.sigmoid(-teacher_scores / scale) * F.logsigmoid(-logits / scale)
    ).mean()
    assert torch.allclose(actual, expected)


def test_lwf_domain_multilabel_uses_temperature_scaled_bernoulli_distillation():
    model = _model()
    queries = torch.tensor([[1.0, 0.5], [-0.5, 2.0]])
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    algorithm = LwFAlgorithm(
        regularization=1.3,
        temperature=2.0,
        problem_type="NC",
        incremental_setting="domain",
    )
    algorithm.after_task(model, _forward, queries, labels, 0)
    teacher_state = copy.deepcopy(algorithm.state["teacher"])
    with torch.no_grad():
        model.weight.add_(0.2)
    logits = model(queries)
    actual = algorithm.additional_loss(model, _forward, queries, logits, labels, 1)

    teacher = copy.deepcopy(model)
    teacher.load_state_dict(teacher_state)
    targets = torch.sigmoid(teacher(queries) / 2.0)
    expected = 1.3 * 4.0 * F.binary_cross_entropy_with_logits(
        logits / 2.0,
        targets,
    )
    assert torch.allclose(actual, expected)


def test_registry_dispatches_only_explicit_class_il_lwf_to_single_head_variant():
    registry = MethodRegistry()
    class_il = registry.create(
        "LwF",
        problem_type="NC",
        incremental_setting="class",
        class_il_single_head=True,
    )
    assert isinstance(class_il, LwFClassILAlgorithm)
    with pytest.raises(ValueError, match="only for LwF on NC-Class"):
        registry.create(
            "LwF",
            problem_type="NC",
            incremental_setting="task",
            class_il_single_head=True,
        )


def test_ergnn_replay_bytes_count_exact_node_and_label_tensors():
    algorithm = ERGNNAlgorithm(problem_type="NC", incremental_setting="class")
    algorithm.state["buffered_nodes"] = torch.tensor([2, 7, 11], dtype=torch.long)
    algorithm.state["buffered_labels"] = torch.tensor([0, 1, 2], dtype=torch.long)
    assert algorithm.replay_payload_bytes() == 6 * torch.tensor([], dtype=torch.long).element_size()


def test_lwf_task_il_distills_each_previous_task_mask():
    model = _model()
    queries = torch.tensor([[1.0, 0.5], [-0.5, 2.0]])
    labels = torch.tensor([0, 1])
    first_mask = torch.tensor([True, True, False])
    second_mask = torch.tensor([False, True, True])
    algorithm = LwFAlgorithm(
        temperature=2.0,
        problem_type="NC",
        incremental_setting="task",
    )
    algorithm.after_task(model, _forward, queries, labels, 0, first_mask)
    algorithm.after_task(model, _forward, queries, labels, 1, second_mask)
    teacher_state = copy.deepcopy(algorithm.state["teacher"])
    with torch.no_grad():
        model.bias.add_(torch.tensor([0.2, -0.1, 0.3]))
    logits = model(queries)
    actual = algorithm.additional_loss(model, _forward, queries, logits, labels, 2)

    teacher = copy.deepcopy(model)
    teacher.load_state_dict(teacher_state)
    expected = logits.sum() * 0.0
    for mask in (first_mask, second_mask):
        expected = expected - (
            F.softmax(teacher(queries)[..., mask] / 2.0, dim=-1)
            * F.log_softmax(logits[..., mask] / 2.0, dim=-1)
        ).sum(dim=-1).mean()
    assert torch.allclose(actual, expected)


def test_lwf_zero_distillation_weight_reduces_exactly_to_bare_objective():
    model = _model()
    queries = torch.tensor([[1.0, 0.5], [-0.5, 2.0]])
    labels = torch.tensor([0, 1])
    mask = torch.tensor([True, True, False])
    algorithm = LwFAlgorithm(regularization=0.0, incremental_setting="class")
    algorithm.after_task(model, _forward, queries, labels, 0, mask)
    with torch.no_grad():
        model.weight.add_(0.3)
    logits = model(queries)
    base = F.cross_entropy(logits[..., mask], labels)
    actual = algorithm.training_loss(
        model, _forward, queries, logits, labels, 1, base, mask
    )
    assert torch.equal(actual, base)


def test_lwf_class_il_matches_independent_old_class_distillation_formula():
    model = _model()
    queries = torch.tensor([[1.0, 0.5], [-0.5, 2.0]])
    labels = torch.tensor([0, 1])
    old_mask = torch.tensor([True, True, False])
    algorithm = LwFClassILAlgorithm(
        regularization=1.7,
        temperature=2.0,
        problem_type="NC",
        incremental_setting="class",
    )
    algorithm.after_task(model, _forward, queries, labels, 0, old_mask)
    teacher_state = copy.deepcopy(algorithm.state["teacher"])
    with torch.no_grad():
        model.weight.add_(0.15)
    logits = model(queries)
    actual = algorithm.additional_loss(model, _forward, queries, logits, labels, 1)

    teacher = copy.deepcopy(model)
    teacher.load_state_dict(teacher_state)
    targets = torch.sigmoid(teacher(queries)[..., old_mask] / 2.0)
    expected = 1.7 * 4.0 * F.binary_cross_entropy_with_logits(
        logits[..., old_mask] / 2.0,
        targets,
    )
    assert torch.allclose(actual, expected)
    assert "buffered_nodes" not in algorithm.state
    assert algorithm.diagnostics()["stores_old_samples"] is False


def test_lwf_class_il_detects_common_old_logit_shift_ignored_by_softmax_lwf():
    teacher_logits = torch.tensor([[5.0, 4.0, 0.0]])
    old_mask = torch.tensor([True, True, False])
    temperature = 2.0
    targets = torch.sigmoid(teacher_logits[..., old_mask] / temperature)
    unshifted = F.binary_cross_entropy_with_logits(
        teacher_logits[..., old_mask] / temperature,
        targets,
    )
    shifted_logits = teacher_logits.clone()
    shifted_logits[..., old_mask] -= 100.0
    shifted = F.binary_cross_entropy_with_logits(
        shifted_logits[..., old_mask] / temperature,
        targets,
    )
    assert shifted > unshifted

    teacher_probabilities = F.softmax(
        teacher_logits[..., old_mask] / temperature,
        dim=-1,
    )
    original_softmax_kd = -(
        teacher_probabilities
        * F.log_softmax(teacher_logits[..., old_mask] / temperature, dim=-1)
    ).sum()
    shifted_softmax_kd = -(
        teacher_probabilities
        * F.log_softmax(shifted_logits[..., old_mask] / temperature, dim=-1)
    ).sum()
    assert torch.equal(original_softmax_kd, shifted_softmax_kd)


def test_lwf_class_il_stage_zero_and_zero_weight_are_exactly_bare():
    model = _model()
    queries = torch.tensor([[1.0, 0.5], [-0.5, 2.0]])
    labels = torch.tensor([0, 1])
    mask = torch.tensor([True, True, False])
    logits = model(queries)
    base = F.cross_entropy(logits[..., mask], labels)
    algorithm = LwFClassILAlgorithm(
        regularization=0.0,
        problem_type="NC",
        incremental_setting="class",
    )
    stage_zero = algorithm.training_loss(
        model, _forward, queries, logits, labels, 0, base, mask
    )
    assert torch.equal(stage_zero, base)
    algorithm.after_task(model, _forward, queries, labels, 0, mask)
    with torch.no_grad():
        model.weight.add_(0.3)
    logits = model(queries)
    base = F.cross_entropy(logits[..., mask], labels)
    later = algorithm.training_loss(
        model, _forward, queries, logits, labels, 1, base, mask
    )
    assert torch.equal(later, base)
    assert "buffered_nodes" not in algorithm.state


def test_ewc_fisher_anchor_and_penalty_match_original_full_batch_formula():
    model = _model()
    reference = copy.deepcopy(model)
    queries = torch.tensor([[1.0, 0.5], [-0.5, 2.0], [0.3, -1.0]])
    labels = torch.tensor([0, 1, 0])
    mask = torch.tensor([True, True, False])
    algorithm = EWCAlgorithm(regularization=3.0)

    reference.zero_grad(set_to_none=True)
    reference_logits = reference(queries).clone()
    reference_logits[..., ~mask] = -1e12
    F.cross_entropy(reference_logits, labels).backward()
    expected_fisher = {
        name: parameter.grad.detach().square().cpu().clone()
        for name, parameter in reference.named_parameters()
    }

    algorithm.after_task(model, _forward, queries, labels, 0, mask)
    for name, parameter in model.named_parameters():
        assert torch.equal(algorithm.state["anchors"][0][name], parameter.detach().cpu())
        assert torch.allclose(
            algorithm.state["fishers"][0][name], expected_fisher[name]
        )

    with torch.no_grad():
        model.weight.add_(0.2)
        model.bias.sub_(0.1)
    logits = model(queries)
    actual = algorithm.additional_loss(model, _forward, queries, logits, labels, 1)
    expected = logits.sum() * 0.0
    for name, parameter in model.named_parameters():
        expected = expected + 3.0 * (
            expected_fisher[name].to(parameter)
            * (parameter - algorithm.state["anchors"][0][name].to(parameter)).square()
        ).sum()
    assert torch.allclose(actual, expected)


def test_ewc_zero_regularization_is_exactly_zero_without_removing_local_state():
    model = _model()
    queries = torch.tensor([[1.0, 0.5], [-0.5, 2.0]])
    labels = torch.tensor([0, 1])
    mask = torch.tensor([True, True, False])
    algorithm = EWCAlgorithm(regularization=0.0)
    algorithm.after_task(model, _forward, queries, labels, 0, mask)
    with torch.no_grad():
        model.weight.add_(1.0)
    logits = model(queries)
    penalty = algorithm.additional_loss(model, _forward, queries, logits, labels, 1)
    assert penalty.item() == 0.0
    assert set(algorithm.state["fishers"]) == {0}


def test_mas_importance_accumulation_anchor_and_penalty_match_original_formula():
    model = _model()
    reference = copy.deepcopy(model)
    queries = torch.tensor([[1.0, 0.5], [-0.5, 2.0], [0.3, -1.0]])
    labels = torch.tensor([0, 1, 0])
    mask = torch.tensor([True, True, True])
    algorithm = MASAlgorithm(regularization=2.5)

    reference.zero_grad(set_to_none=True)
    (torch.linalg.norm(reference(queries), dim=-1).square()).mean().backward()
    expected_importance = {
        name: parameter.grad.detach().abs().cpu().clone()
        for name, parameter in reference.named_parameters()
    }
    algorithm.after_task(model, _forward, queries, labels, 0, mask)
    for name in expected_importance:
        assert torch.allclose(
            algorithm.state["importances"][name], expected_importance[name]
        )

    anchor = copy.deepcopy(algorithm.state["params"])
    with torch.no_grad():
        model.weight.add_(0.1)
        model.bias.sub_(0.05)
    logits = model(queries)
    actual = algorithm.additional_loss(model, _forward, queries, logits, labels, 1)
    expected = logits.sum() * 0.0
    for name, parameter in model.named_parameters():
        expected = expected + 2.5 * (
            expected_importance[name].to(parameter) * (parameter - anchor[name].to(parameter)).square()
        ).sum()
    assert torch.allclose(actual, expected)

    second_reference = copy.deepcopy(model)
    second_reference.zero_grad(set_to_none=True)
    (torch.linalg.norm(second_reference(queries), dim=-1).square()).mean().backward()
    second = {
        name: parameter.grad.detach().abs().cpu().clone()
        for name, parameter in second_reference.named_parameters()
    }
    algorithm.after_task(model, _forward, queries, labels, 1, mask)
    assert algorithm.state["consolidated_task_ids"] == (0, 1)
    for name in second:
        assert torch.allclose(
            algorithm.state["importances"][name],
            expected_importance[name] + second[name],
        )


def test_mas_zero_regularization_reduces_exactly_to_bare_with_importance_state():
    model = _model()
    queries = torch.tensor([[1.0, 0.5], [-0.5, 2.0]])
    labels = torch.tensor([0, 1])
    algorithm = MASAlgorithm(regularization=0.0)
    algorithm.after_task(model, _forward, queries, labels, 0)
    with torch.no_grad():
        model.weight.add_(0.3)
    logits = model(queries)
    base = F.cross_entropy(logits, labels)
    actual = algorithm.training_loss(
        model, _forward, queries, logits, labels, 1, base
    )
    assert torch.equal(actual, base)
    assert "importances" in algorithm.state


def test_ergnn_random_sampler_and_memory_are_client_local_and_deterministic():
    queries = torch.tensor([0, 1, 2, 3, 4, 5])
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    features = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    algorithm = ERGNNAlgorithm(
        problem_type="NC",
        incremental_setting="class",
        client_id=2,
        seed=7,
        sampler_name="random",
        num_experience_nodes=2,
    )
    first = algorithm.sample_nodes(queries, labels, features, 3)
    second = algorithm.sample_nodes(queries, labels, features, 3)
    assert first == second
    assert len(first) == 4
    assert len(set(first) & {0, 1, 2}) == 2
    assert len(set(first) & {3, 4, 5}) == 2

    algorithm.after_task(
        _model(), _forward, queries, labels, 3, None, features
    )
    assert algorithm.state["sampling_history"][3] == tuple(first)
    assert set(algorithm.state["buffered_nodes"].tolist()) <= set(queries.tolist())


def test_ergnn_mf_and_cm_samplers_match_hand_computed_fixtures():
    mf_queries = torch.tensor([0, 1, 2, 3, 4, 5])
    mf_labels = torch.tensor([0, 0, 0, 1, 1, 1])
    mf_features = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [100.0, 0.0],
            [-100.0, 0.0],
            [-2.0, 0.0],
            [-1.0, 0.0],
        ]
    )
    mf = ERGNNAlgorithm(
        problem_type="NC", sampler_name="MF", num_experience_nodes=1
    )
    assert mf.sample_nodes(mf_queries, mf_labels, mf_features, 0) == [1, 4]

    cm_queries = torch.tensor([0, 1, 2, 3])
    cm_labels = torch.tensor([0, 0, 1, 1])
    cm_features = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 3.0]]
    )
    cm = ERGNNAlgorithm(
        problem_type="NC",
        sampler_name="CM",
        num_experience_nodes=1,
        distance_threshold=2.0,
    )
    # The original CM sampler draws comparison nodes with replacement.
    assert cm.sample_nodes(cm_queries, cm_labels, cm_features, 0) == [0, 3]


def test_ergnn_replay_loss_uses_paper_equation_three_beta_weighting():
    model = _model()
    current_queries = torch.tensor([[1.0, 0.5], [-0.5, 2.0]])
    current_labels = torch.tensor([0, 1])
    replay_queries = torch.tensor([[0.3, -1.0]])
    replay_labels = torch.tensor([1])
    algorithm = ERGNNAlgorithm(problem_type="NC")
    algorithm.state["buffered_nodes"] = replay_queries
    algorithm.state["buffered_labels"] = replay_labels
    logits = model(current_queries)
    base_loss = F.cross_entropy(logits, current_labels)
    actual = algorithm.training_loss(
        model,
        _forward,
        current_queries,
        logits,
        current_labels,
        1,
        base_loss,
    )
    beta = 1.0 / 3.0
    replay_loss = F.cross_entropy(model(replay_queries), replay_labels)
    expected = (1.0 - beta) * base_loss + beta * replay_loss
    assert torch.allclose(actual, expected)
    diagnostics = algorithm.diagnostics()
    assert diagnostics["current_weight"] == pytest.approx(1.0 - beta)
    assert diagnostics["replay_weight"] == pytest.approx(beta)
    assert diagnostics["current_loss"] == pytest.approx(float(base_loss.detach()))
    assert diagnostics["replay_loss"] == pytest.approx(float(replay_loss.detach()))
    assert diagnostics["loss_weighting"] == "paper_eq3_beta_replay"
