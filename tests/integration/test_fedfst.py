from __future__ import annotations

from gecko.compat.paths import resolve_source_path

import copy
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import yaml

from gecko.algorithms.federated.fedfst.client import ClientHHKRUpload
from gecko.algorithms.federated.fedfst.client import append_current_train_history
from gecko.algorithms.federated.fedfst.client import history_payload_bytes
from gecko.algorithms.federated.fedfst.core import ConditionalFeatureGenerator
from gecko.algorithms.federated.fedfst.core import FedFSTParameters
from gecko.algorithms.federated.fedfst.core import RAYLEIGH_DENOMINATOR_EPSILON
from gecko.algorithms.federated.fedfst.core import adjust_homophily
from gecko.algorithms.federated.fedfst.core import adjust_spectral_energy
from gecko.algorithms.federated.fedfst.core import balanced_labels
from gecko.algorithms.federated.fedfst.core import canonical_undirected_edges
from gecko.algorithms.federated.fedfst.core import edgewise_low_frequency_kl
from gecko.algorithms.federated.fedfst.core import generate_balanced_features
from gecko.algorithms.federated.fedfst.core import graph_homophily
from gecko.algorithms.federated.fedfst.core import hhkr_edge_type
from gecko.algorithms.federated.fedfst.core import hhkr_loss
from gecko.algorithms.federated.fedfst.core import high_frequency_energy
from gecko.algorithms.federated.fedfst.core import hlst_loss
from gecko.algorithms.federated.fedfst.core import local_generator
from gecko.algorithms.federated.fedfst.core import logical_undirected_edges
from gecko.algorithms.federated.fedfst.core import random_undirected_edges
from gecko.algorithms.federated.fedfst.core import smooth_probabilities
from gecko.algorithms.federated.fedfst.core import spectral_edge_type
from gecko.algorithms.federated.fedfst.core import transition_matrix
from gecko.algorithms.federated.fedfst.core import weighted_generator_average
from gecko.algorithms.federated.fedfst.core import weighted_spectral_target
from gecko.algorithms.federated.fedfst.strategy import FedFSTStrategy
from gecko.engine.coordinator import FederatedCoordinator
from gecko.algorithms.method_config import MethodConfigValidationError
from gecko.algorithms.method_config import resolve_method_config
from gecko.algorithms.method_config import validate_method_config
from gecko.algorithms.federated.catalog import STRATEGIES
from gecko.algorithms.catalog import MethodRegistry
from gecko.models.registry import ModelRegistry

from tests.helpers import make_stream


PATH_EDGES = torch.tensor(
    [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
)
ROOT = Path(__file__).resolve().parents[2]


def _fedfst_config(**overrides: object) -> dict[str, Any]:
    parameters: dict[str, object] = FedFSTParameters().to_dict()
    parameters.update(overrides)
    return {
        "schema": "uefa-method-config",
        "version": 2,
        "name": "fedfst_uefa_v1",
        "strategy": {"name": "fedfst", "parameters": parameters},
        "continual_method": {"name": "Bare", "parameters": {}},
    }


def _tiny_fedfst_config() -> dict[str, Any]:
    return _fedfst_config(
        noise_dim=4,
        generator_dropout=0.0,
        generator_rounds=1,
        generator_epochs=1,
        client_nodes_per_class=2,
        server_nodes_per_class=2,
        generated_edges_per_node=2,
        server_initial_edge_policy="paper_fixed",
        edge_reduction_ratio=0.5,
        topology_tolerance=0.1,
        topology_max_iterations=2,
        sampled_feature_fraction=1.0,
        lambda_kl=1.0,
        smoothing_hops=1,
        lambda_low=1.0,
        distillation_epochs=1,
        distillation_validation_checkpoints=(0, 1),
    )


def _logical_edges(pairs: list[tuple[int, int]], num_nodes: int) -> torch.Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return canonical_undirected_edges(
        torch.tensor(pairs, dtype=torch.long).t().contiguous(), num_nodes
    )


def _history_context(
    *,
    stage_index: int,
    global_task_id: int,
    train_queries: torch.Tensor,
    train_labels: torch.Tensor,
) -> SimpleNamespace:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [9.0, 9.0],
            [0.0, 1.0],
            [-9.0, -9.0],
        ]
    )
    edges = _logical_edges(
        [(0, 1), (0, 2), (1, 2), (2, 3)], num_nodes=4
    )
    return SimpleNamespace(
        problem_type="NC",
        incremental_setting="class",
        stage_index=stage_index,
        global_task_id=global_task_id,
        train_queries=train_queries,
        train_labels=train_labels,
        node_features=features,
        effective_edge_index=edges,
    )


def test_high_frequency_equations_1_2_and_7_match_path_closed_form() -> None:
    features = torch.tensor(
        [[1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]]
    )

    energy = high_frequency_energy(
        features,
        PATH_EDGES,
        feature_indices=torch.tensor([0, 1]),
    )

    assert energy.feature == pytest.approx(0.5)
    assert energy.structure == pytest.approx(5.0 / 3.0)
    assert energy.total == pytest.approx(13.0 / 12.0)
    assert energy.sampled_feature_indices == (0, 1)
    assert energy.valid_feature_signals == 2
    assert energy.valid_structure_signals == 3


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_high_frequency_cuda_matches_cpu_float64_oracle() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.25]],
        dtype=torch.float32,
    )
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
        dtype=torch.long,
    )
    indices = torch.tensor([0, 1], dtype=torch.long)

    oracle = high_frequency_energy(
        features, edges, feature_indices=indices
    )
    cuda = high_frequency_energy(
        features.cuda(),
        edges.cuda(),
        feature_indices=indices.cuda(),
        compute_device="cuda:0",
    )

    assert cuda.feature == pytest.approx(oracle.feature, abs=1e-5, rel=1e-5)
    assert cuda.structure == pytest.approx(oracle.structure, abs=1e-5, rel=1e-5)
    assert cuda.total == pytest.approx(oracle.total, abs=1e-5, rel=1e-5)
    assert cuda.sampled_feature_indices == oracle.sampled_feature_indices
    assert cuda.valid_feature_signals == oracle.valid_feature_signals
    assert cuda.valid_structure_signals == oracle.valid_structure_signals


def test_high_frequency_filters_zero_and_reference_epsilon_denominators() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0, 1e-5],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    energy = high_frequency_energy(
        features,
        PATH_EDGES,
        feature_indices=torch.tensor([0, 1, 2]),
    )

    assert RAYLEIGH_DENOMINATOR_EPSILON == 1e-8
    # The columns have denominators 2, 0, and 1e-10 respectively.  Only the
    # first quotient enters the mean under the frozen reference resolution.
    assert energy.valid_feature_signals == 1
    assert energy.feature == pytest.approx(1.0)
    assert energy.structure == pytest.approx(5.0 / 3.0)
    assert energy.total == pytest.approx(4.0 / 3.0)


def test_high_frequency_structure_excludes_an_isolated_node_column() -> None:
    features = torch.tensor([[1.0], [0.0], [0.0]])
    one_edge_and_isolate = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    energy = high_frequency_energy(
        features,
        one_edge_and_isolate,
        feature_indices=torch.tensor([0]),
    )

    assert energy.valid_feature_signals == 1
    assert energy.valid_structure_signals == 2
    assert energy.feature == pytest.approx(1.0)
    assert energy.structure == pytest.approx(1.0)
    assert energy.total == pytest.approx(1.0)


def test_high_frequency_all_undefined_ratios_resolve_to_zero() -> None:
    features = torch.zeros((3, 2))
    no_edges = torch.empty((2, 0), dtype=torch.long)

    energy = high_frequency_energy(
        features,
        no_edges,
        feature_indices=torch.tensor([0, 1]),
    )

    assert energy.valid_feature_signals == 0
    assert energy.valid_structure_signals == 0
    assert energy.feature == 0.0
    assert energy.structure == 0.0
    assert energy.total == 0.0


def test_undirected_topology_is_loop_free_coalesced_and_homophily_is_logical() -> None:
    noisy = torch.tensor(
        [
            [0, 1, 0, 1, 1, 2, 2, 2],
            [1, 0, 1, 0, 2, 1, 2, 2],
        ],
        dtype=torch.long,
    )
    expected = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
    )

    canonical = canonical_undirected_edges(noisy, num_nodes=3)

    assert torch.equal(canonical, expected)
    assert graph_homophily(canonical, torch.tensor([0, 0, 1])) == pytest.approx(
        0.5
    )
    assert all(
        (target, source) in set(map(tuple, canonical.t().tolist()))
        for source, target in canonical.t().tolist()
    )


def test_client_homophily_matching_removes_complete_heterophilic_pairs() -> None:
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    complete = _logical_edges(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        num_nodes=4,
    )

    adjusted = adjust_homophily(
        complete,
        labels,
        target=0.5,
        reduction_ratio=0.5,
        tolerance=0.0,
        max_iterations=1,
        generator=local_generator(17),
    )

    logical = logical_undirected_edges(adjusted.edge_index, 4)
    same = labels[logical[0]] == labels[logical[1]]
    assert adjusted.initial_value == pytest.approx(1.0 / 3.0)
    assert adjusted.final_value == pytest.approx(0.5)
    assert adjusted.iterations == 1
    assert adjusted.converged
    assert adjusted.removed_edge_types == ("heterophilic",)
    assert int(same.sum()) == 2
    assert int((~same).sum()) == 2


def test_homophily_nonconvergence_returns_best_pruning_candidate() -> None:
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    complete = _logical_edges(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        num_nodes=4,
    )

    adjusted = adjust_homophily(
        complete,
        labels,
        target=0.2,
        reduction_ratio=0.2,
        tolerance=0.0,
        max_iterations=2,
        generator=local_generator(0),
    )

    # The first pruning step reaches float32 homophily 0.20000000298, just
    # outside zero tolerance.  A second step overshoots to zero, so the first
    # candidate must be retained while convergence remains explicitly false.
    assert not adjusted.converged
    assert adjusted.final_value == pytest.approx(0.2)
    assert adjusted.iterations == 1
    assert adjusted.removed_edge_types == ("homophilic",)
    assert logical_undirected_edges(adjusted.edge_index, 4).shape[1] == 5


def test_homophily_best_candidate_keeps_first_exact_error_tie() -> None:
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    complete = _logical_edges(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        num_nodes=4,
    )
    initial = graph_homophily(complete, labels)
    after_one = graph_homophily(
        _logical_edges([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)], 4),
        labels,
    )

    adjusted = adjust_homophily(
        complete,
        labels,
        target=(initial + after_one) / 2.0,
        reduction_ratio=0.2,
        tolerance=0.0,
        max_iterations=1,
        generator=local_generator(0),
    )

    assert not adjusted.converged
    assert adjusted.iterations == 0
    assert adjusted.removed_edge_types == ()
    assert torch.equal(adjusted.edge_index, complete)


def test_client_and_server_topology_select_opposite_paper_targets() -> None:
    assert hhkr_edge_type(0.2, 0.8) == "heterophilic"
    assert hhkr_edge_type(0.8, 0.2) == "homophilic"
    assert spectral_edge_type(0.2, 0.8) == "homophilic"
    assert spectral_edge_type(0.8, 0.2) == "heterophilic"

    features = torch.tensor(
        [[1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]]
    )
    labels = torch.tensor([0, 0, 1], dtype=torch.long)
    high_target, _ = adjust_spectral_energy(
        features,
        labels,
        PATH_EDGES,
        feature_indices=torch.tensor([0, 1]),
        target=100.0,
        reduction_ratio=0.5,
        tolerance=0.0,
        max_iterations=1,
        generator=local_generator(19),
    )
    low_target, _ = adjust_spectral_energy(
        features,
        labels,
        PATH_EDGES,
        feature_indices=torch.tensor([0, 1]),
        target=0.0,
        reduction_ratio=0.5,
        tolerance=0.0,
        max_iterations=1,
        generator=local_generator(19),
    )
    # The direction is homophilic (asserted above), but for this graph that
    # pruning step moves farther from 100; nonconvergence therefore returns
    # the initial best candidate rather than misreporting the attempted one.
    assert not high_target.converged
    assert high_target.removed_edge_types == ()
    assert low_target.removed_edge_types == ("heterophilic",)


def test_spectral_nonconvergence_returns_best_pruning_candidate() -> None:
    features = torch.tensor(
        [[1.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [0.0, -1.0]]
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    edges = _logical_edges([(0, 1), (0, 2)], num_nodes=4)

    adjusted, energy = adjust_spectral_energy(
        features,
        labels,
        edges,
        feature_indices=torch.tensor([0, 1]),
        target=0.7,
        reduction_ratio=0.2,
        tolerance=0.0,
        max_iterations=2,
        generator=local_generator(0),
    )

    # Energies along the pruning path are 1.4583, 0.625, then 0.0.  The
    # middle candidate is closest even though the strict target is not met.
    assert not adjusted.converged
    assert adjusted.initial_value == pytest.approx(35.0 / 24.0)
    assert adjusted.final_value == pytest.approx(0.625)
    assert energy.total == pytest.approx(adjusted.final_value)
    assert adjusted.iterations == 1
    assert adjusted.removed_edge_types == ("heterophilic",)
    assert torch.equal(
        logical_undirected_edges(adjusted.edge_index, 4),
        torch.tensor([[0], [1]], dtype=torch.long),
    )


def test_hhkr_equations_4_through_6_match_closed_form_and_backpropagate() -> None:
    generated = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0]], requires_grad=True
    )
    matched_real = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    labels = torch.tensor([0, 1], dtype=torch.long)
    fixed_teacher = torch.tensor([[0.6, -0.2], [-0.4, 0.7]])
    teacher_logits = generated @ fixed_teacher
    weight = 2.5

    actual = hhkr_loss(
        teacher_logits,
        generated,
        matched_real,
        labels,
        lambda_kl=weight,
    )
    expected_ce = F.cross_entropy(teacher_logits, labels)
    real_probabilities = F.softmax(matched_real, dim=1)
    expected_kl = (
        real_probabilities
        * (real_probabilities.log() - F.log_softmax(generated, dim=1))
    ).sum(dim=1).mean()

    assert torch.allclose(actual.cross_entropy, expected_ce)
    assert torch.allclose(actual.feature_kl, expected_kl)
    assert torch.allclose(actual.total, expected_ce + weight * expected_kl)
    actual.total.backward()
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert torch.count_nonzero(generated.grad) > 0


def test_equations_9_and_10_use_row_normalization_and_one_self_loop() -> None:
    probabilities = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float64
    )
    expected_transition = torch.tensor(
        [
            [1.0 / 2.0, 1.0 / 2.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.0, 1.0 / 2.0, 1.0 / 2.0],
        ],
        dtype=torch.float64,
    )
    expected_two_hops = torch.tensor(
        [
            [7.0 / 12.0, 5.0 / 12.0],
            [5.0 / 9.0, 4.0 / 9.0],
            [7.0 / 12.0, 5.0 / 12.0],
        ],
        dtype=torch.float64,
    )

    transition = transition_matrix(PATH_EDGES, num_nodes=3)

    assert torch.allclose(transition, expected_transition, rtol=0, atol=1e-15)
    assert torch.allclose(transition.sum(dim=1), torch.ones(3, dtype=torch.float64))
    assert torch.equal(smooth_probabilities(probabilities, PATH_EDGES, 0), probabilities)
    assert torch.allclose(
        smooth_probabilities(probabilities, PATH_EDGES, 2),
        expected_two_hops,
        rtol=0,
        atol=1e-15,
    )


def test_equations_12_and_13_use_teacher_destination_to_student_source_kl() -> None:
    teacher = torch.tensor(
        [[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]], dtype=torch.float64
    )
    student = torch.tensor(
        [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]], dtype=torch.float64
    )
    teacher_smoothed = smooth_probabilities(teacher, PATH_EDGES, 1)
    student_smoothed = smooth_probabilities(student, PATH_EDGES, 1)

    low_frequency = edgewise_low_frequency_kl(
        teacher_smoothed, student_smoothed, PATH_EDGES
    )

    assert low_frequency.item() == pytest.approx(
        0.032925568626224, rel=0, abs=1e-14
    )

    labels = torch.tensor([0, 1, 1], dtype=torch.long)
    student_logits = student.log().requires_grad_()
    teacher_logits = teacher.log()
    losses = hlst_loss(
        student_logits,
        teacher_logits,
        labels,
        PATH_EDGES,
        smoothing_hops=1,
        lambda_low=2.0,
    )
    expected_ce = F.cross_entropy(student_logits, labels)
    assert torch.allclose(losses.cross_entropy, expected_ce)
    assert torch.allclose(losses.low_frequency_kl, low_frequency)
    assert torch.allclose(losses.total, expected_ce + 2.0 * low_frequency)
    losses.total.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_equation_8_uses_historical_node_weighting_and_is_order_invariant() -> None:
    states = {
        0: {"weight": torch.tensor([1.0, 1.0])},
        1: {"weight": torch.tensor([5.0, 5.0])},
    }
    counts = {0: 2, 1: 6}

    first = weighted_generator_average(states, counts)
    second = weighted_generator_average(
        {1: states[1], 0: states[0]}, {1: 6, 0: 2}
    )

    assert torch.equal(first["weight"], torch.tensor([4.0, 4.0]))
    assert torch.equal(second["weight"], first["weight"])
    assert weighted_spectral_target({0: 1.0, 1: 3.0}, counts) == pytest.approx(
        2.5
    )


def test_hhkr_upload_schema_cannot_carry_raw_history_or_local_diagnostics() -> None:
    assert {field.name for field in fields(ClientHHKRUpload)} == {
        "client_id",
        "generator_state",
        "spectral_energy",
        "historical_node_count",
    }
    upload = ClientHHKRUpload(
        client_id=0,
        generator_state={"weight": torch.ones(2, 2)},
        spectral_energy=0.25,
        historical_node_count=3,
    )
    assert upload.tensor_payload_bytes == 16
    with pytest.raises(ValueError, match="client ID"):
        ClientHHKRUpload(
            client_id=True,
            generator_state={"weight": torch.ones(2, 2)},
            spectral_energy=0.25,
            historical_node_count=3,
        )


@pytest.mark.parametrize(
    "aggregate",
    [
        pytest.param(
            lambda states, counts: weighted_generator_average(states, counts),
            id="generator",
        ),
        pytest.param(
            lambda _states, counts: weighted_spectral_target(
                {0: 1.0, 1: 2.0}, counts
            ),
            id="spectral",
        ),
    ],
)
@pytest.mark.parametrize("bad_count", [True, 1.5])
def test_equation_8_rejects_noninteger_historical_node_counts(
    aggregate: object, bad_count: object
) -> None:
    states = {
        0: {"weight": torch.tensor([1.0])},
        1: {"weight": torch.tensor([2.0])},
    }
    counts = {0: bad_count, 1: 2}
    with pytest.raises(ValueError, match="weights"):
        aggregate(states, counts)  # type: ignore[operator]


@pytest.mark.gpu
def test_balanced_generation_is_seeded_uses_real_class_ids_and_preserves_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(101)
    rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )

    def reject_global_manual_seed(_seed: int) -> None:
        raise AssertionError("Generator construction must not call torch.manual_seed.")

    monkeypatch.setattr(torch, "manual_seed", reject_global_manual_seed)
    first_model = ConditionalFeatureGenerator(
        feature_dim=4,
        class_ids=(0, 1, 2, 3, 4, 5),
        noise_dim=3,
        dropout=0.25,
        initialization_seed=37,
    )
    first_features, first_labels = generate_balanced_features(
        first_model,
        classes=(5, 2),
        nodes_per_class=3,
        generator=local_generator(43, 1),
        device="cpu",
    )
    first_edges = random_undirected_edges(
        6,
        directed_edges_per_node=2,
        generator=local_generator(43, 2),
    )
    rng_after = torch.get_rng_state().clone()
    cuda_rng_after = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )

    second_model = ConditionalFeatureGenerator(
        feature_dim=4,
        class_ids=(0, 1, 2, 3, 4, 5),
        noise_dim=3,
        dropout=0.25,
        initialization_seed=37,
    )
    second_features, second_labels = generate_balanced_features(
        second_model,
        classes=(2, 5),
        nodes_per_class=3,
        generator=local_generator(43, 1),
        device="cpu",
    )
    second_edges = random_undirected_edges(
        6,
        directed_edges_per_node=2,
        generator=local_generator(43, 2),
    )

    assert torch.equal(rng_before, rng_after)
    assert len(cuda_rng_before) == len(cuda_rng_after)
    assert all(
        torch.equal(before, after)
        for before, after in zip(cuda_rng_before, cuda_rng_after)
    )
    assert torch.equal(first_labels, torch.tensor([2, 2, 2, 5, 5, 5]))
    assert torch.equal(second_labels, first_labels)
    assert torch.equal(second_features, first_features)
    assert torch.equal(second_edges, first_edges)
    assert first_edges.shape == (2, 12)
    assert not bool((first_edges[0] == first_edges[1]).any())
    assert torch.allclose(first_features.norm(dim=1), torch.ones(6))


def test_generator_conditions_only_on_explicit_historical_vocabulary() -> None:
    model = ConditionalFeatureGenerator(
        feature_dim=4,
        class_ids=(2, 5),
        noise_dim=3,
        dropout=0.0,
        initialization_seed=37,
    )

    features, labels = generate_balanced_features(
        model,
        classes=(5, 2),
        nodes_per_class=2,
        generator=local_generator(43, 3),
        device="cpu",
    )

    assert tuple(model.label_embedding.weight.shape) == (2, 2)
    assert tuple(features.shape) == (4, 4)
    assert torch.equal(labels, torch.tensor([2, 2, 5, 5]))
    with pytest.raises(ValueError, match="historical vocabulary"):
        generate_balanced_features(
            model,
            classes=(2, 7),
            nodes_per_class=1,
            generator=local_generator(43, 4),
            device="cpu",
        )


def test_parameter_mapping_is_exact_and_digest_is_order_independent() -> None:
    parameters = FedFSTParameters()
    mapping = parameters.to_dict()
    reversed_mapping = dict(reversed(tuple(mapping.items())))

    assert FedFSTParameters.from_mapping(mapping) == parameters
    assert FedFSTParameters.from_mapping(reversed_mapping).digest == parameters.digest

    missing = dict(mapping)
    missing.pop("lambda_low")
    with pytest.raises(ValueError, match="missing"):
        FedFSTParameters.from_mapping(missing)
    with pytest.raises(ValueError, match="unexpected"):
        FedFSTParameters.from_mapping({**mapping, "ce_weight": 1.0})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("noise_dim", True),
        ("generator_learning_rate", True),
        ("client_learning_rate", 0.0),
        ("client_weight_decay", -1.0),
        ("lambda_kl", True),
        ("edge_reduction_ratio", float("nan")),
        ("topology_tolerance", -1.0),
        ("method_seed", 2**63),
        ("class_il_output_training_policy", "future_labels"),
        ("distillation_early_stop_policy", "loss_plateau"),
        ("server_initial_edge_policy", "adaptive"),
        ("server_max_edges_per_node", 1),
    ],
)
def test_parameter_profile_rejects_boolean_nonfinite_and_out_of_range_values(
    field: str, value: object
) -> None:
    values = FedFSTParameters().to_dict()
    values[field] = value  # type: ignore[assignment]
    with pytest.raises(ValueError):
        FedFSTParameters.from_mapping(values)


def test_paper_full_output_training_applies_only_to_class_il() -> None:
    strategy = FedFSTStrategy(
        **FedFSTParameters(class_il_output_training_policy="paper_full").to_dict()
    )
    mask = torch.tensor([True, True, False])
    class_context = SimpleNamespace(
        incremental_setting="class", valid_class_mask=mask
    )
    task_context = SimpleNamespace(
        incremental_setting="task", valid_class_mask=mask
    )

    assert strategy.training_class_mask(class_context) is None
    assert torch.equal(strategy.training_class_mask(task_context), mask)
    assert strategy.local_optimizer_hyperparameters(
        class_context,
        default_learning_rate=0.01,
        default_weight_decay=0.0,
    ) == (0.01, 0.0)

    seen_strategy = FedFSTStrategy(
        **FedFSTParameters(
            class_il_output_training_policy="benchmark_seen"
        ).to_dict()
    )
    assert torch.equal(seen_strategy.training_class_mask(class_context), mask)


def test_client_history_contains_only_prior_train_nodes_and_is_stage_ordered() -> None:
    container: dict[str, object] = {}
    first = _history_context(
        stage_index=0,
        global_task_id=5,
        train_queries=torch.tensor([2, 0]),
        train_labels=torch.tensor([1, 0]),
    )
    append_current_train_history(container, first)
    history = container["fedfst"]
    assert isinstance(history, dict)

    assert torch.equal(history["node_ids"], torch.tensor([0, 2]))
    assert torch.equal(
        history["features"], torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    )
    assert torch.equal(history["labels"], torch.tensor([0, 1]))
    assert torch.equal(
        history["edge_index"], torch.tensor([[0, 1], [1, 0]])
    )
    assert history_payload_bytes(container) > 0

    second = _history_context(
        stage_index=1,
        global_task_id=3,
        train_queries=torch.tensor([1]),
        train_labels=torch.tensor([2]),
    )
    append_current_train_history(container, second)

    assert torch.equal(history["node_ids"], torch.tensor([0, 1, 2]))
    assert torch.equal(history["node_stage_indices"], torch.tensor([0, 1, 0]))
    assert torch.equal(history["node_task_ids"], torch.tensor([5, 3, 5]))
    assert [record["global_task_id"] for record in history["records"]] == [5, 3]
    assert history["edge_index"].shape == (2, 6)


def test_stream_validation_is_runtime_independent_and_fails_closed() -> None:
    strategy = FedFSTStrategy()

    def stream(
        *,
        orders: dict[int, tuple[int, ...]],
        trace: tuple[object, ...],
        empty_client_task: tuple[int, int] | None = None,
    ) -> SimpleNamespace:
        shards: dict[int, dict[int, SimpleNamespace]] = {}
        for client_id in orders:
            shards[client_id] = {}
            for task_id, labels in {
                0: torch.tensor([0, 1]),
                1: torch.tensor([2, 3]),
            }.items():
                queries = torch.tensor([0, 1])
                if empty_client_task == (client_id, task_id):
                    queries = torch.empty((0,), dtype=torch.long)
                    labels = torch.empty((0,), dtype=torch.long)
                shards[client_id][task_id] = SimpleNamespace(
                    train_queries=queries,
                    train_labels=labels,
                )
        return SimpleNamespace(
            scenario=SimpleNamespace(
                problem_type="NC",
                incremental_type="class",
                num_tasks=2,
                num_classes=4,
                labels=torch.tensor([0, 1, 2, 3]),
                task_masks=torch.tensor(
                    [[True, True, False, False], [False, False, True, True]]
                ),
            ),
            orders=SimpleNamespace(client_orders=orders),
            shards=shards,
            participation=SimpleNamespace(trace=trace),
            config=SimpleNamespace(
                partition=SimpleNamespace(num_clients=2)
            ),
        )

    valid = stream(
        orders={0: (0, 1), 1: (0, 1)},
        trace=(((0, 1),), ((0, 1),)),
    )
    strategy.validate_stream(valid)

    asynchronous = stream(
        orders={0: (0, 1), 1: (1, 0)},
        trace=(((0, 1),), ((0, 1),)),
    )
    strategy.validate_stream(asynchronous)
    assert strategy._prior_tasks(1, 0) == (0,)
    assert strategy._prior_tasks(1, 1) == (1,)
    assert set(strategy._prior_tasks(1)) == {0, 1}

    partial = stream(
        orders={0: (0, 1), 1: (0, 1)},
        trace=(((0,),), ((0, 1),)),
    )
    with pytest.raises(ValueError, match="stage-union"):
        strategy.validate_stream(partial)

    empty = stream(
        orders={0: (0, 1), 1: (0, 1)},
        trace=(((0, 1),), ((0, 1),)),
        empty_client_task=(1, 1),
    )
    with pytest.raises(ValueError, match="non-empty"):
        strategy.validate_stream(empty)


def test_balanced_labels_reject_empty_classes_and_keep_immutable_ids() -> None:
    assert torch.equal(
        balanced_labels((7, 2, 7), 2), torch.tensor([2, 2, 7, 7])
    )
    with pytest.raises(ValueError, match="classes"):
        balanced_labels((), 2)


def test_fedfst_config_registry_and_scope_are_explicit_and_fail_closed() -> None:
    config = _fedfst_config()
    resolved = validate_method_config(
        config,
        expected_strategy="fedfst",
        expected_continual_method="Bare",
        problem_type="NC",
        incremental_setting="class",
    )
    assert resolved.runnable
    assert resolved.benchmark_eligible
    assert resolved.support_status == "supported"
    assert STRATEGIES["fedfst"].requires_method_config
    assert "FedFST" not in MethodRegistry().cli_names()
    assert not any(
        name.startswith("original:begin.algorithms.fedfst.")
        for name in ModelRegistry().all_names()
    )

    task_adaptation = resolve_method_config(
        _fedfst_config(
            distillation_early_stop_policy="author_balance_crossing"
        ),
        problem_type="NC",
        incremental_setting="task",
    )
    assert task_adaptation.runnable
    assert task_adaptation.benchmark_eligible
    assert task_adaptation.support_status == "supported"
    assert task_adaptation.scientific_fidelity == "paper_equation_uefa_adaptation"

    missing = _fedfst_config()
    missing["strategy"]["parameters"].pop("lambda_low")
    with pytest.raises(MethodConfigValidationError, match="missing"):
        resolve_method_config(missing)


def test_fedfst_checked_in_profile_is_complete_and_reference_code_resolved() -> None:
    path = resolve_source_path(ROOT / "configs" / "uefa_v1" / "methods" / "fedfst_nc_class_v1.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = validate_method_config(
        config,
        expected_strategy="fedfst",
        expected_continual_method="Bare",
        problem_type="NC",
        incremental_setting="class",
    )
    assert dict(resolved.strategy_parameters) == FedFSTParameters().to_dict()
    assert resolved.scientific_fidelity == "paper_equation_uefa_adaptation"


def test_fedfst_end_to_end_runs_hhkr_hlst_and_keeps_history_train_only() -> None:
    stream = make_stream("NC", "class", 2, 9187, "mild", "synchronized")
    coordinator = FederatedCoordinator(
        stream,
        strategy_name="fedfst",
        algorithm_name="Bare",
        model_name="uefa_gcn",
        model_seed=31,
        method_config=_tiny_fedfst_config(),
    )

    result = coordinator.run()

    diagnostics = result["strategy_diagnostics"]
    assert result["result_schema"] == "uefa-run-result-v2"
    assert diagnostics["completed_stages"] == 2
    assert diagnostics["hhkr_hlst_stages"] == [1]
    assert diagnostics["teacher_snapshot_stages"] == [0, 1]
    first_task = int(stream.orders.global_task(0, 0))
    historical_classes = torch.where(
        stream.scenario.task_masks[first_task]
    )[0].tolist()
    assert diagnostics["global_generator_classes"] == historical_classes
    assert [
        record["stage_finalization"]["strategy_diagnostics"]["hlst_applied"]
        for record in result["stages"]
    ] == [False, True]
    topology = result["stages"][1]["stage_finalization"][
        "strategy_diagnostics"
    ]["server_topology"]
    assert topology["construction_policy"] == "paper_fixed_global_random"
    assert topology["requested_initial_directed_edges_per_node"] == 2
    assert result["resource_ledger"]["training_auxiliary_uplink_bytes"] > 0
    assert result["resource_ledger"]["training_auxiliary_downlink_bytes"] > 0
    assert result["resource_ledger"]["synthetic_artifact_bytes"] > 0
    assert result["resource_ledger"]["replay_bytes"] > 0

    runtime = coordinator._stateful_runtime
    assert runtime is not None
    generator_state = runtime.strategy.state_dict()["global_generator_state"]
    assert generator_state["label_embedding.weight"].shape[0] == len(
        historical_classes
    )
    shared = runtime.strategy.shared_state
    extracted = coordinator.parameter_policy.extract(coordinator.global_model)
    assert shared.keys() == coordinator.global_state.keys() == extracted.keys()
    for name in shared:
        assert torch.equal(shared[name], coordinator.global_state[name])
        assert torch.equal(shared[name], extracted[name])

    for client_id, client in coordinator.clients.items():
        history = client.state.strategy_state["fedfst"]
        expected_nodes: set[int] = set()
        for stage in range(stream.scenario.num_tasks):
            task_id = int(stream.orders.global_task(client_id, stage))
            expected_nodes.update(
                int(value)
                for value in stream.shards[client_id][task_id].train_queries.tolist()
            )
        assert history["node_ids"].tolist() == sorted(expected_nodes)
        assert len(history["records"]) == stream.scenario.num_tasks
        assert max(history["edge_index"].reshape(-1).tolist(), default=-1) < len(
            expected_nodes
        )
    assert all(
        not record[key]
        for record in diagnostics["stage_records"]
        for key in (
            "contains_raw_feature_upload",
            "contains_raw_label_upload",
            "contains_raw_edge_upload",
        )
    )


def test_fedfst_begin_gcn_freezes_client_local_buffers_during_hlst() -> None:
    stream = make_stream("NC", "class", 2, 9190, "mild", "synchronized")
    try:
        coordinator = FederatedCoordinator(
            stream,
            strategy_name="fedfst",
            algorithm_name="Bare",
            model_name="begin_gcn",
            model_seed=31,
            method_config=_tiny_fedfst_config(),
        )
    except RuntimeError as error:
        pytest.skip(f"begin_gcn dependencies unavailable: {error}")
    runtime = coordinator._stateful_runtime
    assert runtime is not None
    template_buffers = {
        name: value.detach().clone()
        for name, value in runtime.strategy._model_template.named_buffers()
    }
    assert template_buffers

    result = coordinator.run()

    assert result["strategy_diagnostics"]["server_model_state_policy"] == (
        "all_trainable_shared_client_local_buffers_frozen_during_hlst"
    )
    after = dict(runtime.strategy._model_template.named_buffers())
    assert set(after) == set(template_buffers)
    assert all(
        torch.equal(template_buffers[name], after[name])
        for name in template_buffers
    )


def test_fedfst_failed_hlst_does_not_partially_commit_stage_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = make_stream("NC", "class", 2, 9187, "mild", "synchronized")
    coordinator = FederatedCoordinator(
        stream,
        strategy_name="fedfst",
        algorithm_name="Bare",
        model_name="uefa_gcn",
        model_seed=31,
        method_config=_tiny_fedfst_config(),
    )
    runtime = coordinator._stateful_runtime
    assert runtime is not None
    strategy = runtime.strategy
    captured: dict[str, object] = {}

    def fail_after_hhkr(**_kwargs: object) -> object:
        captured["shared"] = strategy.shared_state
        captured["teacher"] = strategy._previous_task_state.materialize()
        captured["generator"] = strategy._global_generator_state.materialize()
        captured["client_records"] = {
            client_id: len(client.state.strategy_state["fedfst"]["records"])
            for client_id, client in coordinator.clients.items()
        }
        captured["client_last_hhkr"] = {
            client_id: dict(client.state.strategy_state["fedfst"]["last_hhkr"])
            for client_id, client in coordinator.clients.items()
        }
        raise RuntimeError("injected HLST failure")

    monkeypatch.setattr(strategy, "_distill", fail_after_hhkr)
    with pytest.raises(RuntimeError, match="injected HLST failure"):
        coordinator.run()

    assert strategy.diagnostics()["completed_stages"] == 1
    assert strategy.diagnostics()["teacher_snapshot_stages"] == [0]
    assert captured["client_records"] == {
        client_id: 1 for client_id in coordinator.clients
    }
    assert captured["client_last_hhkr"] == {
        client_id: {} for client_id in coordinator.clients
    }
    for client in coordinator.clients.values():
        history = client.state.strategy_state["fedfst"]
        assert len(history["records"]) == 1
        assert history["last_hhkr"] == {}
    for name, expected in captured["shared"].items():
        assert torch.equal(strategy.shared_state[name], expected)
    for name, expected in captured["teacher"].items():
        assert torch.equal(strategy._previous_task_state[name], expected)
    assert strategy._global_generator_state.materialize() == captured["generator"]


def test_fedfst_checkpoint_rejects_corruption_without_partial_restore() -> None:
    stream = make_stream("NC", "class", 2, 9187, "mild", "synchronized")
    coordinator = FederatedCoordinator(
        stream,
        strategy_name="fedfst",
        algorithm_name="Bare",
        model_name="uefa_gcn",
        model_seed=31,
        method_config=_tiny_fedfst_config(),
    )
    coordinator.run()
    runtime = coordinator._stateful_runtime
    assert runtime is not None
    strategy = runtime.strategy
    good = strategy.state_dict()
    expected_shared = strategy.shared_state
    expected_diagnostics = strategy.diagnostics()

    corruptions: list[dict[str, object]] = []

    bad_teacher = copy.deepcopy(dict(good))
    bad_teacher["shared_state"] = {
        name: value + 1.0
        for name, value in good["shared_state"].items()
    }
    bad_teacher["previous_task_state"] = dict(good["previous_task_state"])
    first_teacher_key = next(iter(bad_teacher["previous_task_state"]))
    bad_teacher["previous_task_state"][first_teacher_key] = "not-a-tensor"
    corruptions.append(bad_teacher)

    bad_target = copy.deepcopy(dict(good))
    bad_target["global_spectral_target"] = -1.0
    corruptions.append(bad_target)

    bad_vocabulary = copy.deepcopy(dict(good))
    bad_vocabulary["global_generator_classes"] = torch.tensor([999])
    corruptions.append(bad_vocabulary)

    bad_record = copy.deepcopy(dict(good))
    bad_record["stage_records"][1]["stage_index"] = 0
    corruptions.append(bad_record)

    for corrupted in corruptions:
        with pytest.raises((TypeError, ValueError)):
            strategy.load_state_dict(corrupted)
        assert strategy.diagnostics() == expected_diagnostics
        for name, expected in expected_shared.items():
            assert torch.equal(strategy.shared_state[name], expected)


@pytest.mark.parametrize("order_profile", ["synchronized", "binary_mismatch"])
def test_fedfst_stage_checkpoint_resume_matches_uninterrupted_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order_profile: str
) -> None:
    stream = make_stream("NC", "class", 2, 9187, "mild", order_profile)
    baseline = FederatedCoordinator(
        stream,
        strategy_name="fedfst",
        algorithm_name="Bare",
        model_name="uefa_gcn",
        model_seed=31,
        method_config=_tiny_fedfst_config(),
    )
    baseline_result = baseline.run()
    baseline_state = {
        name: value.detach().cpu().clone()
        for name, value in baseline.global_state.items()
    }

    checkpoint_dir = tmp_path / "fedfst-checkpoints"
    interrupted = FederatedCoordinator(
        stream,
        strategy_name="fedfst",
        algorithm_name="Bare",
        model_name="uefa_gcn",
        model_seed=31,
        method_config=_tiny_fedfst_config(),
        checkpoint_dir=checkpoint_dir,
    )
    runtime = interrupted._stateful_runtime
    assert runtime is not None
    original_write = runtime._write_checkpoint

    class StopAfterFirstStage(RuntimeError):
        pass

    def write_then_stop(checkpoint_id: str) -> None:
        original_write(checkpoint_id)
        if checkpoint_id == "stage-0001":
            raise StopAfterFirstStage(checkpoint_id)

    monkeypatch.setattr(runtime, "_write_checkpoint", write_then_stop)
    with pytest.raises(StopAfterFirstStage, match="stage-0001"):
        interrupted.run()

    manifest = checkpoint_dir / "stage-0001.manifest.json"
    assert manifest.is_file()
    resumed = FederatedCoordinator(
        stream,
        strategy_name="fedfst",
        algorithm_name="Bare",
        model_name="uefa_gcn",
        model_seed=31,
        method_config=_tiny_fedfst_config(),
        resume_from=manifest,
    )
    resumed_result = resumed.run()

    assert (
        resumed_result["strategy_diagnostics"]
        == baseline_result["strategy_diagnostics"]
    )
    assert resumed.global_state.keys() == baseline_state.keys()
    for name, expected in baseline_state.items():
        assert torch.equal(resumed.global_state[name].detach().cpu(), expected), name
    assert torch.allclose(
        resumed_result["A[k,s,t]"],
        baseline_result["A[k,s,t]"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert resumed_result["resume_validation"]["identity_matched"] is True
    assert resumed_result["resume_validation"]["diagnostic_override_used"] is False
