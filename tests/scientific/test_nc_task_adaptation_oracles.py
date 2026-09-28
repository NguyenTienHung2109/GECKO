
from gecko.compat.paths import resolve_source_path
from pathlib import Path

import pytest
import torch
import yaml

from gecko.algorithms.method_config import validate_method_config
from gecko.algorithms.continual.cat.node import CondensedGraphRecord
from gecko.algorithms.continual.dslr.algorithm import DSLRAlgorithm
from gecko.algorithms.continual.ssm.node import SSMAlgorithm
from gecko.task_masks import task_aware_cross_entropy


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "config_name",
    [
        "cat_local_only_nc_task_v1.yaml",
        "ssm_local_only_nc_task_v1.yaml",
        "dslr_local_only_nc_task_v1.yaml",
        "power_uefa_full_nc_task_v1.yaml",
        "motion_nc_task_v1.yaml",
    ],
)
def test_nc_task_configs_remain_unverified_and_fail_closed(config_name: str) -> None:
    config = yaml.safe_load(
        (resolve_source_path(ROOT / "configs" / "uefa_v1" / "methods" / config_name)).read_text(encoding="utf-8")
    )
    resolved = validate_method_config(
        config,
        problem_type="NC",
        incremental_setting="task",
    )
    assert resolved.runnable
    assert not resolved.benchmark_eligible
    assert resolved.support_status == "implemented_unverified_task_adaptation"


def test_task_aware_ce_is_invariant_to_other_task_logits() -> None:
    masks = {
        10: torch.tensor([True, True, False, False]),
        20: torch.tensor([False, False, True, True]),
    }
    labels = torch.tensor([0, 1, 2, 3])
    logits = torch.tensor(
        [[3.0, 1.0, -2.0, 8.0], [0.0, 2.0, 7.0, -4.0], [9.0, -3.0, 2.0, 0.0], [-5.0, 6.0, 1.0, 4.0]]
    )
    perturbed = logits.clone()
    perturbed[:2, 2:] += 1.0e4
    perturbed[2:, :2] -= 1.0e4
    assert torch.equal(
        task_aware_cross_entropy(logits, labels, masks),
        task_aware_cross_entropy(perturbed, labels, masks),
    )


def test_task_aware_ce_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="uniquely covered"):
        task_aware_cross_entropy(
            torch.zeros((2, 4)),
            torch.tensor([0, 2]),
            {
                0: torch.tensor([True, False, True, False]),
                1: torch.tensor([True, True, False, False]),
            },
        )


def test_cat_record_checkpoints_its_source_task_head() -> None:
    mask = torch.tensor([False, True, True, False])
    record = CondensedGraphRecord(
        client_id=0,
        global_task_id=7,
        stage_index=0,
        original_node_count=8,
        condensation_steps=1,
        condensation_seed=1,
        condensation_seconds=0.0,
        initial_loss=1.0,
        final_loss=0.5,
        features=torch.ones((2, 3)),
        labels=torch.tensor([1, 2]),
        edge_index=torch.tensor([[0, 1], [0, 1]]),
        class_mask=mask,
    )
    assert torch.equal(CondensedGraphRecord.from_state(record.to_state()).class_mask, mask)


def test_ssm_and_dslr_accept_nc_task_but_not_domain() -> None:
    assert SSMAlgorithm(client_id=0, problem_type="NC", incremental_setting="task")
    assert DSLRAlgorithm(client_id=0, problem_type="NC", incremental_setting="task")
    with pytest.raises(ValueError):
        SSMAlgorithm(client_id=0, problem_type="NC", incremental_setting="domain")
