"""Contracts for the reported paper panels, without downloading real datasets."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from gecko.benchmarks import paper
from gecko.config import GECKOConfig


def test_all_seven_default_panels_have_exact_counts_and_lp_fedavg(tmp_path):
    plan = paper.build_plan(paper.select_scenarios(tuple(paper.SCENARIOS)), tmp_path)
    rows = {row["scenario_id"]: row for row in plan["scenarios"]}
    assert plan["total_stream_cells"] == 148
    assert plan["total_training_jobs"] == 676
    assert [rows[key]["stream_cells"] for key in paper.SCENARIOS] == [12, 24, 24, 12, 24, 12, 40]
    assert rows["S3"]["construction_only"] and rows["S3"]["methods"] == []
    assert rows["S7"]["num_clients"] == 3
    assert rows["S7"]["clients_per_round"] == 2
    assert rows["S7"]["configured_participation_fraction"] == 0.5
    assert rows["S7"]["realized_participation_fraction"] == 2 / 3
    assert rows["S7"]["seeds"] == (0, 1, 2, 3, 4)
    assert {method["strategy"] for method in rows["S7"]["methods"]} == {"fedavg"}
    assert {method["method"] for method in rows["S7"]["methods"]} == {"Bare", "EWC", "LwF", "GraphKeeper"}


def test_plan_has_no_dataset_git_or_optional_graph_dependency(tmp_path):
    code = """
import json
import sys
from pathlib import Path
for name in ('dgl', 'ogb', 'dgllife', 'torch_geometric', 'torch_scatter', 'rdkit', 'pymetis'):
    sys.modules[name] = None
from gecko.benchmarks import paper
import subprocess
def forbidden(*args, **kwargs):
    raise AssertionError('No subprocess, dataset, or GPU operation is allowed for plan')
subprocess.run = forbidden
subprocess.check_output = forbidden
import torch
torch.cuda.init = forbidden
from gecko.data import dataloader
dataloader.load_scenario_spec = forbidden
sys.modules['gecko.benchmarks.lpt_dirichlet_temporal'] = None
plan = paper.build_plan(paper.select_scenarios(tuple(paper.SCENARIOS)), Path('unused-output'))
print(json.dumps({'cells': plan['total_stream_cells'], 'jobs': plan['total_training_jobs']}))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, cwd=tmp_path)
    assert json.loads(result.stdout) == {"cells": 148, "jobs": 676}
    assert not (tmp_path / "unused-output").exists()


@pytest.mark.parametrize("key", tuple(paper.SCENARIOS))
def test_scenario_lookup_matches_real_config_and_never_synthetic(key):
    spec = paper.SCENARIOS[key]
    config = GECKOConfig.from_yaml(paper.source_root() / "configs/gecko_v1/scenarios" / spec.config_name)
    assert paper.scenario_id_for_config(config) == key
    assert paper.scenario_id_for_config(replace(config, scenario=replace(config.scenario, synthetic=True))) is None
    assert paper.scenario_id_for_config(replace(config, scenario=replace(config.scenario, num_tasks=99))) is None


def test_fedfst_uses_native_gat_and_only_its_reported_synchronized_cells(tmp_path):
    selected = paper.select_scenarios(["S2"], methods=["FedFST"])
    method = paper.build_plan(selected, tmp_path)["scenarios"][0]["methods"][0]
    assert method["model"] == "fedfst_gat"
    assert method["model_architecture"] == {"family": "GAT", "num_layers": 2, "hidden_size": 64, "heads": 1, "dropout": 0.5}
    assert method["task_orders"] == ["synchronized"]
    assert method["cells"] == 12 and method["supplementary_only"]
    with pytest.raises(ValueError, match="reported only"):
        paper.select_scenarios(["S2"], methods=["FedFST"], task_orders=["unsynchronized"])


@pytest.mark.parametrize("scenario,options,message", [
    ("S3", {"methods": ["Bare"]}, "construction-only"),
    ("S7", {"methods": ["GEM"]}, "outside its reported panel"),
    ("S1", {"allocation_alphas": [1.0]}, "reported allocation-alpha"),
    ("S2", {"seeds": [4]}, "reported seeds"),
    ("S2", {"task_orders": ["hard"]}, "task-order"),
])
def test_outside_paper_panel_is_rejected_before_data_loading(scenario, options, message):
    with pytest.raises(ValueError, match=message):
        paper.select_scenarios([scenario], **options)


@pytest.mark.parametrize("stage", ["construct", "run", "test"])
def test_mutating_or_test_stages_require_explicit_scenarios(stage):
    with pytest.raises(SystemExit):
        paper.main(["--stage", stage])


def test_s3_run_and_lp_single_alpha_construct_fail_before_work(monkeypatch, tmp_path):
    monkeypatch.setattr(paper, "run_commands", lambda *a, **k: pytest.fail("Must reject S3 before reading streams"))
    with pytest.raises(ValueError, match="construction-only"):
        paper.main(["--stage", "run", "--scenarios", "S2", "S3", "--output", str(tmp_path)])
    monkeypatch.setattr(paper, "construct", lambda *a, **k: pytest.fail("Must reject LP subset before downloading anything"))
    with pytest.raises(ValueError, match="at least two"):
        paper.main(["--stage", "construct", "--scenarios", "S5", "S7", "--allocation-alpha", "0.1", "--output", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def _fake_completed_grid(monkeypatch, tmp_path, scenario="S7", methods=None, orders=("synchronized",)):
    import gecko.data.streams

    selection = paper.select_scenarios([scenario], methods=methods, allocation_alphas=[0.1], task_orders=orders, seeds=[0])[0]
    spec = selection.scenario
    base = GECKOConfig.from_yaml(paper.source_root() / "configs/gecko_v1/scenarios" / spec.config_name)
    records = []
    for order in orders:
        profile = "synchronized" if order == "synchronized" else ("binary_mismatch" if spec.problem == "LP" else "hard")
        config = replace(base, partition=replace(base.partition, num_clients=spec.num_clients, dirichlet_alpha=0.1),
                         training=replace(base.training, rounds_per_stage=50),
                         order=replace(base.order, profile=profile, allow_full_permutation_below_four_tasks=profile == "hard" and spec.num_tasks < 4))
        stream = tmp_path / "streams" / scenario / order
        stream.mkdir(parents=True)
        (stream / "config.yaml").write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
        records.append({"seed": 0, "alpha_dirichlet": 0.1, "order_profile": profile,
                        "stream_path": str(stream), "scientific_fingerprint": "scientific-fingerprint"})
    aggregate = {"scenario_id": scenario,
                 "protocol_version": paper.LP_PROTOCOL if scenario == "S7" else paper.DERIVED_PROTOCOL,
                 "records": records, "failed_cells": [], "benchmark_eligible": True}
    destination = tmp_path / scenario / "stream_manifest.json"
    destination.parent.mkdir()
    destination.write_text(json.dumps(aggregate), encoding="utf-8")
    monkeypatch.setattr(gecko.data.streams, "audit_stream", lambda _: {
        "valid": True, "release_status": "eligible", "scientific_fingerprint": "scientific-fingerprint"})
    return selection, destination, aggregate


def test_lp_run_commands_use_stored_exact_config_fedavg_and_no_bypass(monkeypatch, tmp_path):
    selection, _, aggregate = _fake_completed_grid(monkeypatch, tmp_path, orders=paper.TASK_ORDERS)
    commands = paper.run_commands(selection, tmp_path, "cpu")
    assert len(commands) == 8
    for command in commands:
        assert command[command.index("--strategy") + 1] == "fedavg"
        assert command[command.index("--model") + 1] == "begin_gcn"
        assert Path(command[command.index("--config") + 1]).name == "config.yaml"
        assert "--allow-ineligible-stream" not in command
        assert "--allow-legacy-stream-identity" not in command
    assert aggregate["records"][1]["order_profile"] == "binary_mismatch"


def test_fedfst_command_has_native_model_and_skips_unreported_order(monkeypatch, tmp_path):
    selection, _, _ = _fake_completed_grid(monkeypatch, tmp_path, "S2", ["FedFST"], paper.TASK_ORDERS)
    commands = paper.run_commands(selection, tmp_path, "cpu")
    assert len(commands) == 1
    assert commands[0][commands[0].index("--model") + 1] == "fedfst_gat"
    assert commands[0][commands[0].index("--stream") + 1].endswith("/synchronized")


@pytest.mark.parametrize("failure", ["blocked", "fingerprint", "wrong_budget", "wrong_scenario", "failed_cells", "duplicate", "missing", "manipulation"])
def test_run_preflight_rejects_invalid_scientific_inputs(monkeypatch, tmp_path, failure):
    import gecko.data.streams

    selection, destination, aggregate = _fake_completed_grid(monkeypatch, tmp_path)
    if failure in {"blocked", "fingerprint"}:
        monkeypatch.setattr(gecko.data.streams, "audit_stream", lambda _: {
            "valid": True, "release_status": "blocked" if failure == "blocked" else "eligible",
            "scientific_fingerprint": "other" if failure == "fingerprint" else "scientific-fingerprint"})
    elif failure in {"wrong_budget", "wrong_scenario"}:
        config_path = Path(aggregate["records"][0]["stream_path"]) / "config.yaml"
        config = yaml.safe_load(config_path.read_text())
        if failure == "wrong_budget":
            config["training"]["rounds_per_stage"] = 10
        else:
            config["scenario"]["synthetic"] = True
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    elif failure == "failed_cells":
        aggregate["failed_cells"] = [{"message": "quota infeasible"}]
    elif failure == "duplicate":
        aggregate["records"].append(aggregate["records"][0])
    elif failure == "missing":
        aggregate["records"] = []
    else:
        aggregate["benchmark_eligible"] = False
    destination.write_text(json.dumps(aggregate), encoding="utf-8")
    with pytest.raises(ValueError):
        paper.run_commands(selection, tmp_path, "cpu")


def test_every_scenario_is_preflighted_before_training_subprocess(monkeypatch, tmp_path):
    checked = []

    def commands(selection, *_):
        checked.append(selection.scenario.scenario_id)
        if selection.scenario.scenario_id == "S7":
            raise ValueError("S7 corrupt stream")
        return [["never", "start"]]

    monkeypatch.setattr(paper, "run_commands", commands)
    monkeypatch.setattr(paper.subprocess, "run", lambda *a, **k: pytest.fail("Training started before all audits finished"))
    with pytest.raises(ValueError, match="corrupt stream"):
        paper.main(["--stage", "run", "--scenarios", "S2", "S7", "--methods", "Bare", "--output", str(tmp_path)])
    assert checked == ["S2", "S7"]


def test_test_stage_names_retained_behavioral_tests_without_release_requirements():
    selections = paper.select_scenarios(tuple(paper.SCENARIOS))
    command = paper.test_command(selections)
    assert command[:4] == [sys.executable, "-m", "pytest", "-q"]
    assert len(command) == len(set(command))
    assert all("release/" not in item and "research/" not in item for item in command)
    assert all((paper.source_root() / target).is_file() for target in command[4:])
    s3 = paper.test_command(paper.select_scenarios(["S3"]))
    assert s3[4:] == ["tests/unit/test_paper_benchmark.py", "tests/scientific/test_partition_and_streams.py",
                       "tests/scientific/test_semantics_and_leakage.py"]


@pytest.mark.parametrize("scenario", tuple(paper.SCENARIOS))
def test_construct_dispatch_preserves_exact_protocol_and_participation_lineage(monkeypatch, tmp_path, scenario):
    import gecko.benchmarks.lpt_dirichlet_temporal as constructor
    import gecko.data.streams
    from gecko.compat.lpt_protocol import LPT_V1
    from tools import derive_gecko_participation_grid as derivation

    selection = paper.select_scenarios([scenario], allocation_alphas=[0.1, 100.0], seeds=[1])[0]
    spec = selection.scenario
    calls = {"construct": [], "derive": [], "verify": []}

    def prepare(config_path, **kwargs):
        config = GECKOConfig.from_yaml(config_path)
        calls["construct"].append((config, kwargs))
        records = [{"seed": seed, "alpha_dirichlet": alpha,
                    "order_profile": "binary_mismatch" if spec.problem == "LP" and order == "hard" else order,
                    "stream_path": str(tmp_path / f"source-{seed}-{alpha}-{order}"),
                    "stream_id": f"source-{seed}-{alpha}-{order}", "scientific_fingerprint": "old",
                    "participation_hash": "old-participation"}
                   for seed in kwargs["seeds"] for alpha in kwargs["alphas"] for order in kwargs["temporal_profiles"]]
        return {"protocol_version": kwargs["protocol_version"], "config_path": str(config_path),
                "records": records, "failed_cells": [], "benchmark_eligible": True}

    def derive(source, rounds, *, root, repository_root):
        calls["derive"].append((source, rounds, root))
        target = root / source.name
        target.mkdir(parents=True)
        (target / "participation.json").write_text(json.dumps({
            "trace": {"0": {"0": [0, 1]}}, "fraction": 0.5, "rounds_per_stage": 50, "seed": 1,
        }), encoding="utf-8")
        return target

    def verify(source, target):
        calls["verify"].append((source, target))
        return {"stream_id": "derived-" + source.name, "config_hash": "derived-hash",
                "scientific_fingerprint": "derived-fingerprint", "release_status": "eligible"}

    monkeypatch.setattr(constructor, "prepare_stream_grid", prepare)
    monkeypatch.setattr(gecko.data.streams, "derive_participation_variant", derive)
    monkeypatch.setattr(derivation, "verify_derivation", verify)
    destination = paper.construct(selection, tmp_path, tmp_path / "dataset-cache")
    aggregate = json.loads(destination.read_text())
    config, args = calls["construct"][0]
    assert config.partition.num_clients == spec.num_clients
    assert config.benchmark_seeds == spec.seeds
    assert config.scenario.save_path == str(tmp_path / "dataset-cache")
    assert config.training.participation_fraction == 0.5
    assert config.training.local_epochs_per_round == 1
    assert config.wandb.mode == "disabled"
    assert args["seeds"] == (1,)
    assert args["alphas"] == (0.1, 100.0)
    assert args["temporal_profiles"] == ("synchronized", "hard")
    assert aggregate["num_clients"] == spec.num_clients and aggregate["rounds_per_stage"] == 50
    assert len(aggregate["records"]) == 4
    if scenario == "S7":
        assert args["protocol_version"] == paper.LP_PROTOCOL and args["rounds_per_stage"] == 50
        assert not calls["derive"] and not calls["verify"]
    else:
        assert args["protocol_version"] == LPT_V1 and args["rounds_per_stage"] == 10
        assert aggregate["protocol_version"] == paper.DERIVED_PROTOCOL
        assert len(calls["derive"]) == len(calls["verify"]) == 4
        assert all(call[1] == 50 for call in calls["derive"])
        assert aggregate["derived_from"]["old_trace_is_exact_prefix"]
        assert all(record["participation_hash"] != "old-participation" for record in aggregate["records"])
    with pytest.raises(FileExistsError):
        paper.construct(selection, tmp_path, tmp_path / "dataset-cache")


@pytest.mark.parametrize("failure", ["missing", "duplicate", "failed", "lp_audit"])
def test_constructor_grid_validation_does_not_silently_shrink_panel(failure):
    selection = paper.select_scenarios(["S7"], allocation_alphas=[0.1, 100.0], seeds=[0])[0]
    records = [{"seed": seed, "alpha_dirichlet": alpha,
                "order_profile": "synchronized" if order == "synchronized" else "binary_mismatch"}
               for seed, alpha, order in paper._expected_cells(selection)]
    aggregate = {"records": records, "failed_cells": [], "benchmark_eligible": True}
    if failure == "missing":
        aggregate["records"].pop()
    elif failure == "duplicate":
        aggregate["records"].append(records[0])
    elif failure == "failed":
        aggregate["failed_cells"] = [{"message": "quota infeasible"}]
    else:
        aggregate["benchmark_eligible"] = False
    with pytest.raises(ValueError):
        paper.require_complete_grid(aggregate, selection)
