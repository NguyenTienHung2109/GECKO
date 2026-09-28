from __future__ import annotations

import json

import pytest

from gecko.workflows.run import _atomic_write_text
from gecko.workflows.run import _load_method_config
from gecko.cli.main import build_parser
from gecko.engine.checkpoint import canonical_json_digest


def test_method_config_loader_is_canonical_and_rejects_non_mapping(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(
        "strategy:\n  name: fedavg\nversion: 2\nflags: [true, false]\n",
        encoding="utf-8",
    )
    second.write_text(
        "flags: [true, false]\nversion: 2\nstrategy: {name: fedavg}\n",
        encoding="utf-8",
    )
    first_value, first_digest = _load_method_config(first)
    second_value, second_digest = _load_method_config(second)

    assert first_value == second_value
    assert first_digest == second_digest == canonical_json_digest(first_value)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        _load_method_config(invalid)


def test_run_parser_exposes_requested_v2_checkpoint_options():
    args = build_parser().parse_args(
        [
            "run",
            "--config",
            "config.yaml",
            "--method-config",
            "method.yaml",
            "--checkpoint-dir",
            "checkpoints",
            "--checkpoint-every",
            "2",
            "--resume-from",
            "round.manifest.json",
            "--evaluation-model",
            "personalized",
            "--diagnostic-resume-override",
            "diagnostic only",
        ]
    )
    assert args.method_config == "method.yaml"
    assert args.checkpoint_dir == "checkpoints"
    assert args.checkpoint_every == 2
    assert args.resume_from == "round.manifest.json"
    assert args.evaluation_model == "personalized"
    assert args.diagnostic_resume_override == "diagnostic only"


def test_atomic_v2_result_writer_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "result.json"
    _atomic_write_text(path, json.dumps({"schema": "uefa-run-result-v2"}) + "\n")
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema": "uefa-run-result-v2"
    }
    assert not list(tmp_path.glob(".*.tmp"))
