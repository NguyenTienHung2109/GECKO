from __future__ import annotations

import pytest
import torch

from gecko.evaluation import FederatedEvaluator
from gecko.evaluation import analyze_lc_metric_stability
from gecko.engine import FederatedCoordinator

from tests.helpers import make_stream


def _stable_runs(*, unstable: bool = False):
    labels = torch.arange(1000).remainder(2)
    runs = []
    for model_seed in range(5):
        cells = []
        for client in range(2):
            for task in range(2):
                target = labels.clone()
                predictions = target.clone()
                if unstable and model_seed % 2:
                    predictions = 1 - predictions
                else:
                    predictions[:100] = 1 - predictions[:100]
                cells.append(
                    {
                        "client": client,
                        "task": task,
                        "labels": target,
                        "predictions": predictions,
                    }
                )
        runs.append({"model_seed": model_seed, "cells": cells})
    return runs


def test_lc_stability_uses_accuracy_primary_and_fixed_universe_macro_f1():
    report = analyze_lc_metric_stability(
        _stable_runs(), num_classes=2, bootstrap_samples=200, seed=11
    )
    assert report["stability_status"] == "pass"
    assert report["metric_protocol"]["primary"] == "accuracy_micro_f1"
    assert report["initialization_stability"]["accuracy_micro_f1"]["mean"] == pytest.approx(0.9)
    assert report["initialization_stability"]["accuracy_micro_f1"]["std"] == 0.0
    assert report["initialization_stability"]["pooled_fixed_universe_macro_f1"]["mean"] == pytest.approx(0.9)
    assert report["maximum_primary_bootstrap_ci_width"] < 0.05
    assert report["initialization_stability"]["mean_pairwise_client_rank_spearman"] == 1.0


def test_lc_stability_fails_closed_for_initialization_variance():
    report = analyze_lc_metric_stability(
        _stable_runs(unstable=True), num_classes=2, bootstrap_samples=200
    )
    assert report["stability_status"] == "fail"
    assert "primary_initialization_variance_exceeds_policy" in report["failures"]


def test_lc_stability_rejects_too_few_runs_and_query_mismatch():
    with pytest.raises(ValueError, match="at least five"):
        analyze_lc_metric_stability(
            _stable_runs()[:4], num_classes=2, bootstrap_samples=200
        )
    runs = _stable_runs()
    runs[1]["cells"][0]["labels"] = 1 - runs[1]["cells"][0]["labels"]
    with pytest.raises(ValueError, match="identical query labels"):
        analyze_lc_metric_stability(runs, num_classes=2, bootstrap_samples=200)


def test_central_prediction_export_keeps_labels_outside_client_state():
    stream = make_stream("LC", "task", 2, order_profile="synchronized")
    coordinator = FederatedCoordinator(
        stream, "fedavg", "Bare", model_name="uefa_gcn", model_seed=17
    )
    coordinator.run()
    evaluator = FederatedEvaluator(stream)
    client = coordinator.clients[0]
    logits, labels, central_ids = evaluator.predict_central(
        client.model,
        0,
        0,
        set(range(stream.scenario.num_tasks)),
    )
    assert logits.shape[0] == labels.shape[0] == central_ids.shape[0]
    assert torch.equal(labels, stream.scenario.labels[central_ids])
    assert "labels" not in client.state.continual_state


def test_model_seed_changes_initialization_without_changing_stream():
    stream = make_stream("LC", "task", 2, order_profile="synchronized")
    first = FederatedCoordinator(
        stream, "fedavg", "Bare", model_name="uefa_gcn", model_seed=0
    )
    second = FederatedCoordinator(
        stream, "fedavg", "Bare", model_name="uefa_gcn", model_seed=1
    )
    assert first.stream.stream_hash == second.stream.stream_hash
    assert first.initial_model_state_digest != second.initial_model_state_digest
