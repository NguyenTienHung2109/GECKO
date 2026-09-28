"""Training and checkpoint resume depend on experiment state, not source layout."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from gecko.data.streams.manifest import repository_provenance
from gecko.engine import runtime as runtime_module
from gecko.engine.checkpoint import CheckpointIdentity
from gecko.engine.checkpoint import CheckpointIdentityError
from gecko.engine.checkpoint import load_checkpoint
from gecko.engine.checkpoint import save_checkpoint
from gecko.workflows.run import _execution_repository_provenance
from tests.helpers import make_config


def _identity(**changes) -> CheckpointIdentity:
    identity = CheckpointIdentity(
        stream_id="synthetic", stream_hash="1" * 64,
        scientific_fingerprint="2" * 64, source_sha=None,
        stream_content_digest="3" * 64, source_tree_digest=None,
        run_config_digest="4" * 64, method_config_digest="5" * 64,
        strategy="fedavg", method="Bare", model="uefa_gcn", model_seed=0,
    )
    return replace(identity, **changes)


def test_training_metadata_never_queries_git_or_requires_a_source_manifest(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Training metadata must not invoke Git.")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    (tmp_path / "SOURCE_MANIFEST.json").write_text("invalid obsolete metadata", encoding="utf-8")
    assert runtime_module._resolve_source_sha(None) is None
    assert runtime_module._resolve_source_sha("optional-note") == "optional-note"
    assert runtime_module._resolve_tracked_source_tree() == (None, None, ())
    assert _execution_repository_provenance()["repository_current_commit_SHA"] is None
    assert repository_provenance(tmp_path)["commit_sha"] is None


@pytest.mark.parametrize("source", [None, "unknown", "a" * 40, "edited source"])
def test_resume_ignores_source_metadata_changes(tmp_path, source):
    written = save_checkpoint(
        tmp_path, "round-000", identity=_identity(source_sha="old-source", source_tree_digest="old-tree"),
        state={"weight": torch.ones(1)}, cursor={"round": 1},
    )
    loaded = load_checkpoint(
        written.manifest_path,
        expected_identity=_identity(source_sha=source, source_tree_digest="changed-tree"),
    )
    assert loaded.resume_validation.identity_matched
    assert not loaded.resume_validation.diagnostic_override_used
    assert loaded.resume_validation.benchmark_eligible


@pytest.mark.parametrize("change", [
    {"stream_hash": "a" * 64},
    {"scientific_fingerprint": "b" * 64},
    {"stream_content_digest": "c" * 64},
    {"run_config_digest": "d" * 64},
    {"method_config_digest": "e" * 64},
    {"strategy": "fedprox"},
    {"method": "EWC"},
    {"model": "other-model"},
    {"model_seed": 1},
])
def test_resume_still_rejects_incompatible_experiment_state(tmp_path, change):
    written = save_checkpoint(
        tmp_path, "round-000", identity=_identity(),
        state={"weight": torch.ones(1)}, cursor={"round": 1},
    )
    with pytest.raises(CheckpointIdentityError, match=next(iter(change))):
        load_checkpoint(written.manifest_path, expected_identity=_identity(**change))


def test_copied_source_without_git_constructs_trains_and_resumes_after_code_edit(tmp_path):
    source = Path(__file__).resolve().parents[2]
    copied = tmp_path / "unpacked"
    shutil.copytree(
        source / "src/gecko", copied / "src/gecko",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(source / "pyproject.toml", copied / "pyproject.toml")
    assert not (copied / ".git").exists()
    assert not (copied / "SOURCE_MANIFEST.json").exists()
    config = replace(
        make_config("NC", "class", 2, order_profile="synchronized", rounds=2),
        output_root=str(copied / "streams"),
    )
    (copied / "config.yaml").write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    method = {
        "schema": "uefa-method-config", "version": 2, "name": "legacy_adapter_v1",
        "strategy": {"name": "fedavg", "parameters": {}},
        "continual_method": {"name": "Bare", "parameters": {}},
    }
    (copied / "method.yaml").write_text(yaml.safe_dump(method), encoding="utf-8")
    script = '''
import json
import subprocess
import sys
from pathlib import Path
import gecko
from gecko.cli.main import main
from gecko.config import GECKOConfig
from gecko.workflows.construct import expected_stream_path

root = Path.cwd()
assert Path(gecko.__file__).is_relative_to(root)
def forbid_git(original):
    def guarded(*args, **kwargs):
        command = args[0] if args else kwargs.get("args", [])
        executable = command.split()[0] if isinstance(command, str) else command[0]
        assert Path(executable).name != "git", "Training unexpectedly invoked Git"
        return original(*args, **kwargs)
    return guarded
subprocess.run = forbid_git(subprocess.run)
subprocess.check_output = forbid_git(subprocess.check_output)
phase = sys.argv[1]
if phase == "initial":
    assert main(["construct", "--config", "config.yaml"]) == 0
arguments = ["run", "--config", "config.yaml", "--model", "uefa_gcn",
             "--method-config", "method.yaml", "--model-seed", "17",
             "--run-tag", phase, "--allow-ineligible-stream"]
if phase == "initial":
    arguments.extend(["--checkpoint-dir", "checkpoints", "--checkpoint-every", "1"])
else:
    arguments.extend(["--resume-from", "checkpoints/round-00000001.manifest.json"])
assert main(arguments) == 0
stream = expected_stream_path(GECKOConfig.from_yaml("config.yaml"))
result_file = next((stream / "results").glob("*" + phase + "*.json"))
result = json.loads(result_file.read_text())
assert result["source_sha"] is None
assert result["source_tree_digest"] is None
assert "source_tree_dirty" not in result["benchmark_ineligibility_reasons"]
assert result["run_configuration"]["model_seed"] == 17
if phase == "resumed":
    assert result["resume_validation"]["identity_matched"]
    assert not result["resume_validation"]["diagnostic_override_used"]
    baseline = json.loads(next((stream / "results").glob("*initial*.json")).read_text())
    assert result["summary"] == baseline["summary"]
    assert result["client_stage_task_matrix"] == baseline["client_stage_task_matrix"]
'''
    environment = {
        **os.environ, "PYTHONPATH": str(copied / "src"), "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    for phase in ("initial", "resumed"):
        if phase == "resumed":
            edited = copied / "src/gecko/reproducibility.py"
            edited.write_text(edited.read_text(encoding="utf-8") + "\n# Reviewer edit.\n", encoding="utf-8")
            (copied / "SOURCE_MANIFEST.json").write_text("stale, malformed metadata", encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-c", script, phase], cwd=copied,
            env=environment, capture_output=True, text=True, timeout=90,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
