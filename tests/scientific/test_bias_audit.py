from __future__ import annotations

import json
from dataclasses import replace

import pytest

from gecko.data.audits.bias import _nc_topology_recomputation
from gecko.data.audits.bias import audit_bias
from gecko.cli.main import main
from gecko.data.streams import save_stream
from gecko.types import PartitionResult

from tests.helpers import make_stream


def test_nc_bias_audit_reports_topology_support_and_feature_provenance():
    report = audit_bias(make_stream("NC", "domain", 2))
    assert report["release_eligible"] is True
    assert report["topology"]["retained_degree_ratio_summary"]["count"] > 0
    assert "p90" in report["topology"]["retained_degree_ratio_summary"]
    assert report["topology"]["global_homophily"] is not None
    assert len(report["topology"]["homophily_drift_per_client"]) == 2
    assert report["support_rows"]
    assert report["feature_drift"]["available"] is True
    assert report["benchmark_tier"] == "stress"
    assert report["release_status"] == "eligible"
    assert report["benchmark_eligible"] is True
    assert report["leaderboard_ready"] is False


def test_nc_topology_recomputation_matches_partition_diagnostics():
    stream = make_stream("NC", "domain", 2)
    topology = _nc_topology_recomputation(stream)
    diagnostics = stream.partition.diagnostics

    assert topology["global_homophily"] == pytest.approx(
        diagnostics["global_homophily"]
    )
    assert topology["local_homophily"] == pytest.approx(
        diagnostics["local_homophily"]
    )
    assert topology["homophily_drift_per_client"] == pytest.approx(
        diagnostics["homophily_drift_per_client"]
    )


def test_lc_bias_audit_quantifies_label_selection():
    report = audit_bias(make_stream("LC", "domain", 4))
    assert 0.0 <= report["label_selection_bias"]["js_divergence"] <= 1.0
    assert report["query_coverage"]["rows_by_task_split_class"]
    assert report["client_task_split_class_coverage"]
    assert "zero_cells" in report["client_class_support_adequacy"]
    assert report["support_summary_by_split"]["train"]["queries"]["count"] > 0


def test_lp_bias_audit_blocks_evaluation_aware_partition():
    report = audit_bias(make_stream("LP", "domain", 2))
    assert report["status"] == "blocked"
    assert report["invariant_status"] == "pass"
    assert report["release_status"] == "blocked"
    assert "lp_partition_is_conditioned_on_evaluation_positive_endpoints" in report["blockers"]
    provenance = report["link_prediction"]["partition_provenance"]
    assert provenance["support_anchor_query_splits"] == ["train", "validation", "test"]
    by_split = report["link_prediction"]["negative_count_per_group_by_split"]
    assert by_split["val"]["min"] >= 50
    assert by_split["test"]["min"] >= 50


def test_lp_bias_audit_accepts_explicit_train_only_provenance():
    stream = make_stream("LP", "domain", 2)
    diagnostics = {
        **stream.partition.diagnostics,
        "lp_partition_information_scope": "topology_and_train_queries",
        "lp_partition_uses_evaluation_positive_endpoints": False,
        "lp_partition_uses_evaluation_candidates": False,
        "lp_evaluation_support_role": "post_partition_audit_only",
        "lp_support_anchor_query_splits": ["train"],
    }
    partition = PartitionResult(
        stream.partition.node_owner,
        stream.partition.client_graphs,
        stream.partition.micro_community_ids,
        diagnostics,
    )
    report = audit_bias(replace(stream, partition=partition))
    assert report["release_eligible"] is True
    assert report["release_status"] == "eligible"


def test_audit_bias_cli_writes_machine_readable_report(tmp_path):
    stream_path = save_stream(
        make_stream("LP", "domain", 2), tmp_path, repository_root=tmp_path
    )
    output = tmp_path / "bias.json"
    assert main([
        "audit-bias",
        "--stream",
        str(stream_path),
        "--output",
        str(output),
        "--allow-blockers",
    ]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["audit_name"] == "uefa_coverage_bias_audit"
    assert report["status"] == "blocked"
