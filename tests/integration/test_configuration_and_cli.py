from __future__ import annotations

from gecko.compat.paths import resolve_source_path

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gecko.workflows.construct import expected_stream_path
from gecko.cli.main import main
from gecko.config import GECKOConfig
from gecko.engine import FederatedCoordinator
from gecko.data.streams import save_stream
from gecko.data.streams import stream_identity
from gecko.validation import ConfigurationError
from gecko.validation import UnsupportedCombinationError

from tests.helpers import make_config
from tests.helpers import make_stream


def test_seven_default_real_configs_match_authoritative_matrix():
    root = resolve_source_path(Path(__file__).resolve().parents[2] / "configs" / "uefa_v1")
    expected = {
        "nc_task_ogbn_arxiv.yaml": ("ogbn-arxiv", "NC", "task", 8, ("accuracy",)),
        "nc_class_ogbn_arxiv.yaml": ("ogbn-arxiv", "NC", "class", 8, ("accuracy",)),
        "nc_domain_ogbn_proteins.yaml": ("ogbn-proteins", "NC", "domain", 8, ("rocauc",)),
        "lc_task_bitcoin.yaml": ("bitcoin", "LC", "task", 3, ("accuracy", "macro_f1")),
        "lc_class_bitcoin.yaml": ("bitcoin", "LC", "class", 3, ("accuracy", "macro_f1")),
        "lc_domain_bitcoin.yaml": ("bitcoin", "LC", "domain", 4, ("accuracy", "macro_f1")),
        "lp_domain_facebook.yaml": (
            "facebook",
            "LP",
            "domain",
            2,
            ("hits@50", "mrr", "average_precision", "rocauc"),
        ),
    }
    for filename, values in expected.items():
        config = GECKOConfig.from_yaml(root / filename)
        scenario = config.scenario
        assert (
            scenario.dataset,
            scenario.problem,
            scenario.incremental_setting,
            scenario.num_tasks,
            scenario.metrics,
        ) == values
        assert config.partition.num_clients == 10
        assert config.benchmark_seeds == (0, 1, 2, 3, 4)
        if scenario.problem == "LP":
            assert scenario.lp_protocol_name == "uefa_lp_partition_first_fixed_candidates"
            assert scenario.lp_protocol_version == 2
            assert scenario.lp_query_split_protocol == "partition_first_query_split_v1"
            assert scenario.lp_base_edge_ratio == 0.10
            assert scenario.lp_source_num_domains == 8
            assert scenario.lp_domain_mapping == "dominant_source_vs_rest_v1"
            assert scenario.lp_evaluation_positives_per_task == 1000
            assert scenario.lp_evaluation_negatives_per_client_task == 1000
            assert config.partition.lp_partition_information_scope == "topology_only"
            assert config.order.profile == "binary_mismatch"
            assert (
                config.training.aggregation_weight
                == "current_positive_anchor_count"
            )


def test_lp_v2_rejects_generic_mild_order_name():
    path = (
        resolve_source_path(Path(__file__).resolve().parents[2]
        / "configs"
        / "uefa_v1"
        / "lp_domain_facebook.yaml")
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["order"]["profile"] = "mild"
    with pytest.raises(ConfigurationError, match="binary_mismatch"):
        GECKOConfig.from_mapping(raw)


def test_lp_protocol_v1_is_fixed_and_fail_closed(tmp_path):
    stream = make_stream("LP", "domain", 2, order_profile="synchronized")
    path = save_stream(stream, tmp_path, repository_root=tmp_path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["lp_protocol"] == {
        "name": "uefa_lp_fixed_candidates",
        "version": 1,
        "training_negative_policy": "fixed_global_validated",
        "evaluation_candidate_policy": "explicit_fixed_groups",
        "evaluation_grouping": "pooled_per_client_task",
        "tie_policy": "pessimistic",
        "legacy_score_comparable": False,
        "partition_information_scope": "all_positive_splits",
        "evaluation_aware_partition": True,
    }
    assert len(manifest["environment_fingerprint"]) == 64
    assert manifest["invariant_status"] == "pass"
    assert manifest["release_status"] == "blocked"
    assert manifest["release_eligibility"] is False
    assert len(manifest["audit_policy_hash"]) == 64
    invalid = replace(
        stream.config,
        scenario=replace(
            stream.config.scenario, lp_training_negative_policy="runtime_resampled"
        ),
    )
    with pytest.raises(ConfigurationError, match="LP Protocol v1"):
        invalid.validate()


def test_unsupported_scenarios_fail_before_dataset_loading():
    with pytest.raises(UnsupportedCombinationError, match="does not support"):
        GECKOConfig.from_mapping(
            {
                "scenario": {
                    "dataset": "synthetic",
                    "problem": "LP",
                    "incremental_setting": "time",
                    "num_tasks": 2,
                    "metrics": ["hits@50"],
                }
            }
        )


def test_stream_hash_excludes_run_only_hyperparameters_and_paths():
    config = make_config("NC", "task", 2)
    changed = replace(
        config,
        scenario=replace(config.scenario, save_path="another/data/root"),
        training=replace(
            config.training,
            learning_rate=0.123,
            hidden_size=31,
            num_layers=1,
            local_epochs_per_round=3,
        ),
        wandb=replace(config.wandb, mode="offline"),
        output_root="another/output/root",
    )
    assert stream_identity(config) == stream_identity(changed)
    changed_trace = replace(
        config,
        training=replace(config.training, participation_fraction=0.5),
    )
    assert stream_identity(config) != stream_identity(changed_trace)


def test_local_only_and_joint_oracle_report_zero_federated_communication():
    stream = make_stream("NC", "task", 2)
    local = FederatedCoordinator(stream, "local_only", "Bare", model_name="uefa_gcn").run()
    oracle = FederatedCoordinator(stream, "joint_oracle", "Bare", model_name="uefa_gcn").run()
    assert local["communication_payload_bytes"] == 0
    assert oracle["communication_payload_bytes"] == 0
    assert oracle["oracle"] is True


def test_partition_audit_contains_required_realized_diagnostics():
    diagnostics = make_stream("LP", "domain", 2).partition.diagnostics
    required = {
        "node_count_per_client",
        "internal_edge_count_per_client",
        "edge_cut_ratio",
        "boundary_node_ratio",
        "connected_component_count_per_client",
        "largest_component_ratio_per_client",
        "isolated_node_ratio_per_client",
        "degree_distribution_divergence",
        "local_homophily",
        "label_js_divergence",
        "domain_js_divergence",
        "query_js_divergence",
        "client_query_volume_coefficient_of_variation",
        "client_task_support_matrix",
        "client_task_support_summary",
        "retained_degree_ratio_summary",
        "retained_degree_ratio_per_client",
        "lp_internal_positive_coverage",
        "lp_internal_candidate_coverage",
        "lp_cross_client_positive_rate",
        "lp_false_negative_validation_count",
    }
    assert required <= diagnostics.keys()
    assert "lp_candidate_count_summary" in diagnostics
    assert "lp_negative_candidate_count_summary" in diagnostics


def test_blocked_stream_requires_explicit_diagnostic_training_override(tmp_path):
    stream = make_stream("LP", "domain", 2, order_profile="synchronized")
    stream_path = save_stream(stream, tmp_path / "streams", repository_root=tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(stream.config.to_dict(), sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="release_status='blocked'"):
        main([
            "run",
            "--config",
            str(config_path),
            "--stream",
            str(stream_path),
            "--model",
            "uefa_gcn",
            "--wandb-mode",
            "disabled",
        ])


def test_report_excludes_ineligible_results_by_default(tmp_path, capsys):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "eligible.json").write_text(
        json.dumps({"benchmark_eligible": True, "score": 1}), encoding="utf-8"
    )
    (result_dir / "blocked.json").write_text(
        json.dumps({"benchmark_eligible": False, "score": 2}), encoding="utf-8"
    )
    assert main(["report", "--stream", str(tmp_path)]) == 0
    default = json.loads(capsys.readouterr().out)
    assert set(default) == {"eligible.json"}
    assert main([
        "report", "--stream", str(tmp_path), "--include-ineligible"
    ]) == 0
    complete = json.loads(capsys.readouterr().out)
    assert set(complete) == {"eligible.json", "blocked.json"}


def test_cli_generate_audit_run_and_report_use_existing_stream(tmp_path, capsys):
    config = replace(make_config("NC", "task", 2), output_root=str(tmp_path / "streams"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    assert main(["generate", "--config", str(config_path)]) == 0
    stream_path = expected_stream_path(config)
    assert stream_path.exists()
    manifest = json.loads((stream_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["scientific_fingerprint"]) == 64
    assert len(manifest["package_digest"]) == 64
    assert manifest["repository_current_commit_SHA"] != "unknown"
    assert manifest["dataset_design_audit"]["status"] == "passed"
    assert len(manifest["environment_fingerprint"]) == 64
    assert main(["audit", "--stream", str(stream_path)]) == 0
    assert main(
        [
            "run",
            "--config",
            str(config_path),
            "--stream",
            str(stream_path),
            "--strategy",
            "fedavg",
            "--cl-algorithm",
            "Bare",
            "--model",
            "uefa_gcn",
        ]
    ) == 0
    result = stream_path / "results" / "fedavg-Bare-uefa_gcn.json"
    assert result.exists()
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["stream_hash"] == stream_identity(config)[1]
    assert payload["artifact_integrity"]["unchanged"] is True
    assert payload["execution_device"] == "cpu"
    assert "NaN" not in result.read_text(encoding="utf-8")
    assert main(["report", "--stream", str(stream_path)]) == 0


def test_wandb_environment_can_disable_auto_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    from gecko.tracking import WandbLogger

    logger = WandbLogger(mode="auto", project="UEFA", directory=tmp_path)
    assert logger.actual_mode == "disabled"
    assert logger.run is None


def test_cli_generate_output_root_override_supports_duplicate_builds(tmp_path):
    configured_root = tmp_path / "configured"
    override_root = tmp_path / "duplicate-build"
    config = replace(
        make_config("NC", "task", 2), output_root=str(configured_root)
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    assert main(
        [
            "generate",
            "--config",
            str(config_path),
            "--output-root",
            str(override_root),
        ]
    ) == 0
    overridden = replace(config, output_root=str(override_root))
    assert expected_stream_path(overridden).exists()
    assert not expected_stream_path(config).exists()
    assert stream_identity(overridden) == stream_identity(config)


def test_cli_lists_only_truthful_default_methods_and_models(capsys):
    assert main(["list-methods"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Bare", "LwF", "EWC", "MAS", "ERGNN"
    ]

    assert main(["list-methods", "--experimental"]) == 0
    assert set(capsys.readouterr().out.splitlines()) == {
        "generic_replay",
        "generic_importance_regularization",
        "generic_weight_isolation",
    }

    assert main(["list-models"]) == 0
    assert capsys.readouterr().out.splitlines() == ["begin_gcn", "fedfst_gat"]

    assert main(["list-models", "--discovered"]) == 0
    discovered = capsys.readouterr().out.splitlines()
    assert discovered
    assert all(name.startswith("original:") for name in discovered)
