from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gecko.engine import FederatedCoordinator
from gecko.engine import NCReferenceView
from gecko.engine import REFERENCE_MODES
from gecko.engine import summarize_nc_domain_reference_gaps
from gecko.data.partitioning.materialize import derive_strict_local_node_features
from gecko.data.datasets.synthetic import build_synthetic_spec
from gecko.data.streams import StreamBuilder

from tests.helpers import make_config
from tests.helpers import make_stream


def _strict_local_nc_domain_stream():
    config = make_config(
        "NC",
        "domain",
        2,
        spatial_profile="mild",
        order_profile="synchronized",
    )
    config = replace(
        config,
        scenario=replace(config.scenario, feature_provenance="strict_local_derived"),
    )
    scenario = build_synthetic_spec(
        "NC",
        "domain",
        num_tasks=2,
        num_clients=2,
        seed=0,
    )
    generator = torch.Generator().manual_seed(401)
    edge_features = torch.randn(
        scenario.edge_index.shape[1], scenario.num_features, generator=generator
    )
    metadata = {
        **scenario.metadata,
        "feature_provenance": "strict_local_derived",
        "feature_derivation": "mean_outgoing_visible_edge_features",
        "context_edge_features": edge_features,
        "context_edge_feature_valid_mask": torch.ones(
            edge_features.shape[0], dtype=torch.bool
        ),
    }
    scenario = replace(scenario, metadata=metadata)
    global_features = derive_strict_local_node_features(
        scenario,
        torch.arange(scenario.node_features.shape[0]),
        torch.ones(scenario.edge_index.shape[1], dtype=torch.bool),
    )
    scenario = replace(scenario, node_features=global_features)
    return StreamBuilder(config).build(scenario)


def test_reference_modes_change_only_declared_topology_and_feature_axes():
    stream = _strict_local_nc_domain_stream()
    client_id = 0
    graph = stream.partition.client_graphs[client_id]
    queries = stream.shards[client_id][0].train_queries

    local_local = NCReferenceView(
        stream, "strict_local_topology_local_features"
    )
    full_local = NCReferenceView(stream, "full_topology_local_features")
    local_global = NCReferenceView(
        stream, "strict_local_topology_global_features"
    )
    full_global = NCReferenceView(stream, "full_topology_global_features")

    a_features, a_edges, a_queries = local_local.inputs(client_id, queries)
    b_features, b_edges, b_queries = full_local.inputs(client_id, queries)
    c_features, c_edges, c_queries = local_global.inputs(client_id, queries)
    d_features, d_edges, d_queries = full_global.inputs(client_id, queries)

    assert torch.equal(a_edges, graph.edge_index)
    assert torch.equal(c_edges, graph.edge_index)
    assert torch.equal(b_edges, stream.scenario.edge_index)
    assert torch.equal(d_edges, stream.scenario.edge_index)
    assert torch.equal(a_features, graph.node_features)
    assert torch.equal(c_features, stream.scenario.node_features[graph.local_to_global])
    assert torch.equal(b_features[graph.local_to_global], graph.node_features)
    assert torch.equal(d_features, stream.scenario.node_features)
    assert torch.equal(a_queries, queries)
    assert torch.equal(c_queries, queries)
    assert torch.equal(b_queries, graph.local_to_global[queries])
    assert torch.equal(d_queries, graph.local_to_global[queries])
    assert not torch.equal(a_features, c_features)
    cached_first = full_global.device_inputs(client_id, queries, torch.device("cpu"))
    cached_second = full_global.device_inputs(client_id, queries, torch.device("cpu"))
    assert cached_first[0].data_ptr() == cached_second[0].data_ptr()
    assert cached_first[1].data_ptr() == cached_second[1].data_ptr()


def test_all_reference_modes_share_initialization_schedule_queries_and_budget():
    stream = _strict_local_nc_domain_stream()
    coordinators = {
        mode: FederatedCoordinator(
            stream,
            "centralized_shard_oracle",
            "Bare",
            model_name="uefa_gcn",
            reference_mode=mode,
        )
        for mode in REFERENCE_MODES
    }
    states = [coordinator.global_state for coordinator in coordinators.values()]
    assert len(
        {coordinator.initial_model_state_digest for coordinator in coordinators.values()}
    ) == 1
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        assert all(torch.equal(state[key], states[0][key]) for key in state)

    results = {mode: coordinator.run() for mode, coordinator in coordinators.items()}
    populations = {
        tuple(torch.isfinite(result["client_stage_task_matrix"]).flatten().tolist())
        for result in results.values()
    }
    assert len(populations) == 1
    for mode, result in results.items():
        reference = result["diagnostic_reference"]
        assert reference["name"] == mode
        assert reference["same_supervised_query_set"] is True
        assert reference["same_task_schedule"] is True
        assert reference["leaderboard_method"] is False
        assert result["benchmark_eligible"] is False
        assert result["strategy_benchmark_eligible"] is False
        assert result["strategy_hyperparameters"]["fedprox_mu"] is None
        assert (
            result["initial_model_state_digest"]
            == coordinators[mode].initial_model_state_digest
        )
        assert result["training_budget"]["optimizer"] == "Adam"
        assert len(result["rounds"]) == stream.scenario.num_tasks


def test_reference_mode_rejects_non_nc_domain_and_federated_strategy():
    with pytest.raises(ValueError, match="NC-Domain only"):
        NCReferenceView(make_stream("NC", "class", 2), "full_topology_global_features")
    with pytest.raises(ValueError, match="requires centralized_shard_oracle"):
        FederatedCoordinator(
            _strict_local_nc_domain_stream(),
            "fedavg",
            "Bare",
            model_name="uefa_gcn",
            reference_mode="full_topology_global_features",
        )


def test_reference_gap_formula_is_hand_computed_and_stream_matched():
    values = {
        "strict_local_topology_local_features": 0.50,
        "full_topology_local_features": 0.65,
        "strict_local_topology_global_features": 0.55,
        "full_topology_global_features": 0.75,
    }
    results = {
        name: {
            "stream_id": "fixed",
            "stream_hash": "fixed-hash",
            "resolved_model": "begin_gcn",
            "base_metric": "rocauc",
            "initial_model_state_digest": "initial",
            "training_budget": {"rounds_per_stage": 1},
            "diagnostic_reference": {"name": name},
            "summary": {"final_average_performance": value},
        }
        for name, value in values.items()
    }
    fedavg = {
        "stream_id": "fixed",
        "stream_hash": "fixed-hash",
        "resolved_model": "begin_gcn",
        "base_metric": "rocauc",
        "initial_model_state_digest": "initial",
        "training_budget": {"rounds_per_stage": 1},
        "summary": {"final_average_performance": 0.40},
    }
    report = summarize_nc_domain_reference_gaps(results, fedavg_result=fedavg)
    assert report["topology_gap"] == pytest.approx(0.15)
    assert report["feature_provenance_gap"] == pytest.approx(0.10)
    assert report["federation_gap"] == pytest.approx(0.10)
    assert report["feature_gap_at_strict_local_topology"] == pytest.approx(0.05)
    assert report["topology_gap_with_global_features"] == pytest.approx(0.20)
    assert report["topology_feature_interaction"] == pytest.approx(0.05)
    assert report["matched_protocol_verified"] is True

    mismatched = {**fedavg, "stream_id": "other"}
    with pytest.raises(ValueError, match="one fixed stream"):
        summarize_nc_domain_reference_gaps(results, fedavg_result=mismatched)


def test_nc_domain_diagnostics_include_p90_and_homophily_drift():
    diagnostics = _strict_local_nc_domain_stream().partition.diagnostics
    assert "p90" in diagnostics["retained_degree_ratio_summary"]
    assert diagnostics["global_homophily"] is not None
    assert len(diagnostics["homophily_drift_per_client"]) == 2
