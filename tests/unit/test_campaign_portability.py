"""Campaign completion bookkeeping validates training options and stream data."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from gecko.cli.main import main
from gecko.workflows import campaign_run


@pytest.fixture()
def completed_campaign(tmp_path: Path) -> tuple[list[str], Path, dict]:
    root = Path(__file__).resolve().parents[2]
    config = root / "configs/gecko_v1/scenarios/synthetic_smoke.yaml"
    output = tmp_path / "streams"
    assert main(["construct", "--config", str(config), "--output-root", str(output)]) == 0
    stream = next(output.rglob("manifest.json")).parent
    method = root / "configs/gecko_v1/algorithms/legacy_bare_fedavg_nc_task_grid_v1.yaml"
    command = [
        sys.executable, "-m", "gecko.cli.main", "run",
        "--config", str(config), "--stream", str(stream),
        "--output-root", str(output), "--method-config", str(method),
        "--strategy", "fedavg", "--cl-algorithm", "Bare",
        "--model", "uefa_gcn", "--device", "cpu", "--wandb-mode", "disabled",
    ]
    assert main(command[3:]) == 0
    result_path = campaign_run._completed_result_path(command)
    assert result_path is not None
    return command, result_path, json.loads(result_path.read_text(encoding="utf-8"))


def test_completion_validates_training_config_and_data(completed_campaign) -> None:
    command, _, result = completed_campaign
    assert campaign_run._matches_run_configuration(command, result)
    assert not campaign_run._matches_run_configuration(
        [*command, "--local-epochs-per-round", "2"], result
    )
    assert not campaign_run._matches_run_configuration(
        [*command, "--model-seed", "19"], result
    )
    altered = copy.deepcopy(result)
    altered["scientific_fingerprint"] = "0" * 64
    assert not campaign_run._matches_run_configuration(command, altered)


def test_missing_training_metadata_is_not_a_completed_run(completed_campaign) -> None:
    command, _, result = completed_campaign
    result.pop("run_configuration")
    assert not campaign_run._matches_run_configuration(command, result)


@pytest.mark.parametrize("invalid", ["[]", "null", "not-json"])
def test_malformed_completed_results_are_rejected(completed_campaign, invalid: str) -> None:
    command, result_path, _ = completed_campaign
    result_path.write_text(invalid, encoding="utf-8")
    assert not campaign_run._is_completed_result(command)


def test_reported_artifact_mutation_is_not_skipped(completed_campaign) -> None:
    command, result_path, result = completed_campaign
    result["artifact_integrity"]["unchanged"] = False
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert not campaign_run._is_completed_result(command)
