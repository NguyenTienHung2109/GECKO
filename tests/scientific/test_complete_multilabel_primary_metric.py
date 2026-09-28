from __future__ import annotations

from gecko.compat.paths import resolve_source_path

from dataclasses import replace
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from gecko.evaluation.legacy import NegativeBinaryCrossEntropyEvaluator
from gecko.data.legacy import evaluator_map
from gecko.config import GECKOConfig
from gecko.evaluation.metrics import compute_metrics
from gecko.evaluation.metrics import negative_binary_cross_entropy
from gecko.engine.coordinator import FederatedCoordinator
from gecko.engine.runtime import StatefulFederatedRuntime

from tests.helpers import make_stream


def _method_config() -> dict[str, Any]:
    return {
        "schema": "uefa-method-config",
        "version": 2,
        "name": "legacy_adapter_v1",
        "strategy": {"name": "fedavg", "parameters": {}},
        "continual_method": {"name": "Bare", "parameters": {}},
    }


def _run(stream) -> dict[str, Any]:
    method_config = _method_config()
    coordinator = FederatedCoordinator(
        stream,
        "fedavg",
        "Bare",
        model_name="uefa_gcn",
        model_seed=31,
        method_config=method_config,
    )
    return StatefulFederatedRuntime(
        coordinator, method_config=method_config
    ).run()


def test_negative_binary_cross_entropy_is_complete_and_missing_target_aware() -> None:
    logits = torch.tensor([[2.0, -2.0, 100.0], [-1.0, 1.0, -100.0]])
    labels = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    valid = labels >= 0
    expected = -F.binary_cross_entropy_with_logits(
        logits[valid], labels[valid]
    ).item()

    value = negative_binary_cross_entropy(logits, labels)
    assert value == pytest.approx(expected)
    assert compute_metrics(
        logits, labels, ("negative_binary_cross_entropy",)
    )["negative_binary_cross_entropy"] == pytest.approx(expected)

    all_zero = torch.zeros((4, 3))
    good = negative_binary_cross_entropy(
        torch.full_like(all_zero, -8.0), all_zero
    )
    bad = negative_binary_cross_entropy(
        torch.full_like(all_zero, 8.0), all_zero
    )
    assert math.isfinite(good)
    assert good > bad
    assert math.isnan(
        negative_binary_cross_entropy(
            torch.zeros((2, 2)), torch.full((2, 2), -1.0)
        )
    )


def test_legacy_scenario_evaluator_supports_complete_multilabel_metric() -> None:
    task_ids = torch.tensor([0, 0, 1, 1])
    evaluator = NegativeBinaryCrossEntropyEvaluator(2, task_ids)
    logits = torch.tensor(
        [[8.0, -8.0], [-8.0, 8.0], [8.0, 8.0], [-8.0, -8.0]]
    )
    labels = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [0.0, -1.0]]
    )

    values = evaluator(logits, labels, torch.arange(4))

    assert evaluator_map["negative_binary_cross_entropy"] is (
        NegativeBinaryCrossEntropyEvaluator
    )
    assert torch.isfinite(values).all()
    assert values.shape == (3,)
    assert values[2].item() == pytest.approx(
        negative_binary_cross_entropy(logits, labels)
    )


def test_nc_domain_complete_metric_config_is_versioned_and_keeps_rocauc() -> None:
    path = (
        resolve_source_path(Path(__file__).resolve().parents[2]
        / "configs"
        / "uefa_v1"
        / "nc_domain_ogbn_proteins_complete_metric_v2.yaml")
    )
    config = GECKOConfig.from_yaml(path)
    assert config.benchmark_schema_version == 2
    assert config.scenario.metrics == (
        "negative_binary_cross_entropy",
        "rocauc",
    )
    assert config.scenario.target_type == "multi_label"


def test_undefined_primary_cells_fail_benchmark_closed_with_coverage() -> None:
    stream = make_stream(
        "NC", "domain", 2, order_profile="synchronized", seed=0
    )
    labels = stream.scenario.labels.clone()
    shard = stream.evaluation_shards[0][0]
    labels[shard.test_query_ids] = 0
    labels[shard.validation_query_ids] = 0
    stream = replace(
        stream, scenario=replace(stream.scenario, labels=labels)
    )

    result = _run(stream)

    support = result["primary_metric_support"]
    assert support == {
        "expected_cell_count": 6,
        "finite_cell_count": 4,
        "undefined_cell_count": 2,
        "coverage": pytest.approx(4.0 / 6.0),
        "final_expected_cell_count": 4,
        "final_finite_cell_count": 3,
        "final_coverage": pytest.approx(3.0 / 4.0),
        "complete": False,
    }
    assert result["primary_metric_benchmark_eligible"] is False
    assert result["validation_primary_metric_support"] == {
        "expected_cell_count": 6,
        "finite_cell_count": 4,
        "undefined_cell_count": 2,
        "coverage": pytest.approx(4.0 / 6.0),
        "complete": False,
    }
    assert result["benchmark_eligible"] is False
    assert (
        "undefined_expected_primary_metric_cells"
        in result["benchmark_ineligibility_reasons"]
    )
    assert (
        "undefined_expected_validation_primary_metric_cells"
        in result["benchmark_ineligibility_reasons"]
    )
    assert result["official_primary_metric_summary"] is None
    assert result["conditional_primary_metric_summary"] == result["summary"]
    assert result["primary_metric_summary_population"] == (
        "conditional_on_finite_primary_metric_cells"
    )
    assert [
        stage["primary_metric_support_coverage"] for stage in result["stages"]
    ] == pytest.approx([0.5, 0.75])


def test_all_undefined_primary_and_validation_are_explicit_not_silent_zero() -> None:
    stream = make_stream(
        "NC", "domain", 2, order_profile="synchronized", seed=3
    )
    labels = stream.scenario.labels.clone()
    for shards in stream.evaluation_shards.values():
        for shard in shards.values():
            labels[shard.validation_query_ids] = 0
            labels[shard.test_query_ids] = 0
    stream = replace(
        stream, scenario=replace(stream.scenario, labels=labels)
    )

    result = _run(stream)

    assert all(stage["stage_test_metric"] is None for stage in result["stages"])
    assert all(stage["validation_metric"] is None for stage in result["stages"])
    assert result["primary_metric_support"]["coverage"] == 0.0
    assert result["primary_metric_support"]["final_coverage"] == 0.0
    assert result["primary_metric_benchmark_eligible"] is False
    assert result["validation_primary_metric_support"]["coverage"] == 0.0
    assert result["validation_primary_metric_support"]["complete"] is False
    assert result["official_primary_metric_summary"] is None
    assert all(
        stage["validation_primary_metric_finite_cell_count"] == 0
        for stage in result["stages"]
    )
    assert all(
        stage["validation_primary_metric_support_coverage"] == 0.0
        for stage in result["stages"]
    )


def test_complete_metric_makes_single_class_runtime_population_finite() -> None:
    stream = make_stream(
        "NC", "domain", 2, order_profile="synchronized", seed=4
    )
    labels = stream.scenario.labels.clone()
    for shards in stream.evaluation_shards.values():
        for shard in shards.values():
            labels[shard.validation_query_ids] = 0
            labels[shard.test_query_ids] = 0
    metrics = ("negative_binary_cross_entropy", "rocauc")
    stream = replace(
        stream,
        config=replace(
            stream.config,
            scenario=replace(stream.config.scenario, metrics=metrics),
        ),
        scenario=replace(stream.scenario, labels=labels, metrics=metrics),
    )

    result = _run(stream)

    assert result["base_metric"] == "negative_binary_cross_entropy"
    assert result["primary_metric_support"]["complete"] is True
    assert result["primary_metric_support"]["coverage"] == 1.0
    assert result["primary_metric_support"]["final_coverage"] == 1.0
    assert result["primary_metric_benchmark_eligible"] is True
    assert result["validation_primary_metric_support"]["coverage"] == 1.0
    assert result["validation_primary_metric_support"]["complete"] is True
    assert result["official_primary_metric_summary"] == result["summary"]
    assert result["primary_metric_summary_population"] == (
        "complete_expected_population"
    )
    assert all(
        isinstance(stage["validation_metric"], float)
        for stage in result["stages"]
    )
