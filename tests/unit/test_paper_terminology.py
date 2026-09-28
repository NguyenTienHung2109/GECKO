"""Paper labels preserve historical scientific identities and construction semantics."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest
import yaml

from gecko.benchmarks.terminology import allocation_metadata
from gecko.benchmarks.terminology import allocation_slug
from gecko.benchmarks.terminology import task_order_label
from gecko.cli.main import build_parser
from gecko.config import GECKOConfig
from gecko.data.streams import audit_stream, load_stream, save_stream
from gecko.data.streams.builder import spatial_partition_slug, stream_identity, stream_relative_path
from gecko.validation import ConfigurationError
from gecko.workflows.construct import _with_profile_overrides, expected_stream_path
from tests.helpers import make_config


@pytest.mark.parametrize("problem,setting,tasks,profile,short", [
    ("NC", "class", 8, "hard", False),
    ("LC", "class", 3, "hard", True),
    ("LC", "domain", 4, "hard", False),
    ("LP", "domain", 2, "binary_mismatch", False),
])
def test_public_config_aliases_preserve_historical_serialization(problem, setting, tasks, profile, short):
    base = make_config(problem, setting, tasks)
    historical = replace(
        base,
        partition=replace(base.partition, dirichlet_alpha=0.1),
        order=replace(base.order, profile=profile, allow_full_permutation_below_four_tasks=short),
    )
    raw = base.to_dict()
    raw["partition"].pop("dirichlet_alpha")
    raw["partition"]["allocation_alpha"] = 0.1
    raw["order"] = {"task_order": "unsynchronized"}
    public = GECKOConfig.from_mapping(raw)
    assert public.to_dict() == historical.to_dict()
    assert stream_identity(public) == stream_identity(historical)
    assert public.partition.allocation_alpha == 0.1
    assert public.order.task_order == "unsynchronized"


@pytest.mark.parametrize("section,changes", [
    ("partition", {"allocation_alpha": 0.1, "dirichlet_alpha": 100.0}),
    ("order", {"task_order": "synchronized", "profile": "hard"}),
    ("order", {"task_order": "unsynchronized", "profile": "mild"}),
])
def test_conflicting_public_and_historical_config_aliases_fail(section, changes):
    raw = make_config("NC", "class", 8).to_dict()
    raw[section].update(changes)
    with pytest.raises(ConfigurationError, match="conflicts"):
        GECKOConfig.from_mapping(raw)


@pytest.mark.parametrize("alpha", [0.0, -0.1, float("nan"), float("inf")])
def test_allocation_alpha_is_finite_and_positive(alpha):
    raw = make_config("NC", "class", 8).to_dict()
    raw["partition"]["dirichlet_alpha"] = alpha
    with pytest.raises(ConfigurationError, match="finite and positive"):
        GECKOConfig.from_mapping(raw)


def test_legacy_profiles_do_not_acquire_fictitious_paper_semantics():
    for spatial in ("easy", "mild", "hard"):
        config = make_config("NC", "task", 2, spatial_profile=spatial, order_profile="mild")
        metadata = allocation_metadata(config)
        assert metadata["allocation_alpha"] is None
        assert metadata["allocation_control"] == "legacy_heuristic"
        assert allocation_slug(config) == f"allocation-legacy-{spatial}"
        assert spatial_partition_slug(config) == f"allocation-legacy-{spatial}"
        assert metadata["task_order"] == "legacy-mild"
    assert task_order_label("unconstrained") == "legacy-unconstrained"
    assert task_order_label("hard") == "unsynchronized"
    assert task_order_label("binary_mismatch") == "unsynchronized"
    assert stream_relative_path(config).parts[-2:] == (
        "allocation-legacy-hard", "task-order-legacy-mild"
    )
    assert stream_relative_path(config, legacy=True).parts[-2:] == (
        "spatial-hard", "order-mild"
    )


def test_public_cli_aliases_resolve_without_changing_order_plan():
    base = make_config("LC", "class", 3)
    arguments = build_parser().parse_args([
        "run", "--config", "unused.yaml", "--allocation-alpha", "0.1",
        "--task-order", "unsynchronized",
    ])
    resolved = _with_profile_overrides(base, arguments)
    assert resolved.partition.dirichlet_alpha == 0.1
    assert resolved.order.profile == "hard"
    assert resolved.order.allow_full_permutation_below_four_tasks
    for options in (
        ["--allocation-alpha", "0.1", "--dirichlet-alpha", "100"],
        ["--task-order", "synchronized", "--order-profile", "hard"],
    ):
        arguments = build_parser().parse_args(["run", "--config", "unused.yaml", *options])
        with pytest.raises(ConfigurationError, match="conflicts"):
            _with_profile_overrides(base, arguments)


def test_generic_construct_rejects_explicit_alpha_before_data_loading(tmp_path, monkeypatch):
    from gecko.workflows import construct

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(make_config("NC", "task", 2).to_dict()), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("Dataset loading must not occur for the wrong constructor.")
    monkeypatch.setattr(construct, "load_scenario_spec", forbidden)
    arguments = build_parser().parse_args([
        "construct", "--config", str(config_path), "--allocation-alpha", "0.1",
    ])
    with pytest.raises(ConfigurationError, match="examples/benchmark.py"):
        arguments.handler(arguments)


def test_new_exact_paths_and_historical_reader_preserve_stream_identity(tmp_path):
    from gecko.benchmarks.lpt_dirichlet_temporal import (
        build_temporal_variants, construct_exact_dirichlet_partition, prepare_lpt_scenario,
    )
    from gecko.workflows.run import command_run

    root = Path(__file__).resolve().parents[2]
    base = GECKOConfig.from_yaml(root / "configs/gecko_v1/scenarios/synthetic_smoke.yaml")
    prepared = prepare_lpt_scenario(base)
    scenario, partition, _ = construct_exact_dirichlet_partition(prepared, alpha=0.1)
    config = replace(
        prepared.config,
        output_root=str(tmp_path / "streams"),
        partition=replace(prepared.config.partition, dirichlet_alpha=0.1),
    )
    streams, _ = build_temporal_variants(config, scenario, partition)
    bundle = streams["hard"]
    identity_before = stream_identity(bundle.config)
    path = save_stream(bundle, repository_root=tmp_path)
    assert path == expected_stream_path(bundle.config)
    assert path.parent.name == "allocation-alpha-0.1"
    assert path.name == "task-order-unsynchronized"
    assert "dirichlet-0p1-hard" in bundle.stream_id
    assert audit_stream(path)["valid"]
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["allocation_alpha"] == 0.1
    assert manifest["task_order"] == "unsynchronized"
    assert manifest["scenario_id"] is None
    assert manifest["order_profile"] == "hard"
    historical = Path(bundle.config.output_root) / stream_relative_path(bundle.config, legacy=True)
    historical.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(historical))
    restored = load_stream(historical)
    assert stream_identity(restored.config) == identity_before
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(bundle.config.to_dict()), encoding="utf-8")
    arguments = build_parser().parse_args([
        "run", "--config", str(config_path), "--model", "uefa_gcn",
        "--allocation-alpha", "0.1", "--task-order", "unsynchronized",
        "--wandb-mode", "disabled", "--allow-ineligible-stream",
    ])
    assert command_run(arguments) == 0
    result = json.loads(next((historical / "results").glob("*.json")).read_text(encoding="utf-8"))
    assert result["allocation_alpha"] == 0.1
    assert result["task_order"] == "unsynchronized"
    assert result["scenario_id"] is None  # Synthetic smoke is not paper S1.
    assert result["artifact_integrity"]["unchanged"]
