"""Behavioral oracles for the opt-in DSLR link-loss normalization."""

from gecko.compat.paths import resolve_source_path

from pathlib import Path

import pytest
import torch
import yaml

from gecko.algorithms.method_config import resolve_method_config
from gecko.algorithms.continual.dslr.algorithm import DSLRAlgorithm
from gecko.algorithms.continual.dslr.structure import link_prediction_loss
from gecko.algorithms.catalog import MethodRegistry


ROOT = Path(__file__).resolve().parents[2]


def _edges(pairs: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def test_normalized_link_bce_is_invariant_to_repeated_edge_cardinality():
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.5, 3.0**0.5 / 2.0], [-0.5, 3.0**0.5 / 2.0]]
    )
    positive = _edges([(0, 1)])
    negative = _edges([(0, 2)])
    repeated_positive = positive.repeat(1, 7)
    repeated_negative = negative.repeat(1, 7)

    summed = link_prediction_loss(
        embeddings, positive, negative, reduction="sum"
    )
    repeated_sum = link_prediction_loss(
        embeddings, repeated_positive, repeated_negative, reduction="sum"
    )
    mean = link_prediction_loss(
        embeddings, positive, negative, reduction="mean"
    )
    repeated_mean = link_prediction_loss(
        embeddings, repeated_positive, repeated_negative, reduction="mean"
    )

    assert repeated_sum == pytest.approx(float(summed) * 7.0)
    assert repeated_mean == pytest.approx(float(mean))
    assert summed == pytest.approx(float(mean) * 2.0)


def test_normalized_identity_is_opt_in_and_does_not_change_literal_dslr():
    path = (
        resolve_source_path(ROOT
        / "configs/gecko_v1/algorithms/dslr_normalized_local_only_nc_class_v1.yaml")
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved = resolve_method_config(
        config, problem_type="NC", incremental_setting="class"
    )
    assert resolved.name == "dslr_normalized_v1"
    assert resolved.continual_method_name == "DSLR-Normalized"
    assert resolved.scientific_fidelity == (
        "official_code_reduction_benchmark_adaptation"
    )
    assert resolved.runnable and not resolved.benchmark_eligible

    common = {
        "client_id": 0,
        "seed": 7,
        "problem_type": "NC",
        "incremental_setting": "class",
        **dict(resolved.continual_method_parameters),
    }
    normalized = MethodRegistry().create(
        "DSLR-Normalized",
        explicit_v2_family="dslr_normalized_v1",
        **common,
    )
    literal = DSLRAlgorithm(**common)
    assert normalized.name == "DSLR-Normalized"
    assert normalized.link_reduction == "mean"
    assert literal.name == "DSLR"
    assert literal.link_reduction == "sum"
    assert normalized.diagnostics()["link_loss_reduction"] == "mean"
    assert literal.diagnostics()["link_loss_reduction"] == "sum"
