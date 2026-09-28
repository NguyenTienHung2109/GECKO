from __future__ import annotations

import pytest
import torch

from gecko.algorithms.federated.motion.gepae import motion_gepae_aggregate
from gecko.algorithms.federated.motion.gepae import motion_minmax_normalize
from gecko.algorithms.federated.motion.gepae import motion_pcb_merge
from gecko.algorithms.federated.motion.gepae import motion_percentile_clamp


def test_motion_normalize_and_clamp_match_index_based_oracles() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    normalized = motion_minmax_normalize(values, dim=1)
    assert torch.allclose(normalized[0], torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]))
    assert torch.equal(normalized[1], torch.zeros(4))

    clamped = motion_percentile_clamp(
        values, min_ratio=0.25, max_ratio=0.25
    )
    assert torch.equal(
        clamped,
        torch.tensor([[2.0, 2.0, 3.0, 3.0], [5.0, 5.0, 5.0, 5.0]]),
    )


def test_motion_pcb_merge_preserves_upstream_top_parameter_behavior() -> None:
    merged, clamped, scales = motion_pcb_merge(
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        pcb_ratio=0.5,
        pcb_min_ratio=0.0,
        pcb_max_ratio=0.0,
    )
    assert torch.equal(clamped, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert torch.equal(scales, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    assert torch.equal(merged, torch.tensor([0.0, 0.0, 0.0, 4.0]))


def test_motion_gepae_aggregate_uses_weighted_client_deltas() -> None:
    global_state = {
        "a": torch.zeros(2),
        "b": torch.zeros((1, 2)),
    }
    client_states = {
        0: {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([[3.0, 4.0]])},
        1: {"a": torch.tensor([-2.0, 1.0]), "b": torch.tensor([[1.0, 2.0]])},
    }
    weights = {0: 3, 1: 1}
    expected_delta, _, expected_scales = motion_pcb_merge(
        torch.stack(
            (
                torch.tensor([1.0, 2.0, 3.0, 4.0]) * 0.75,
                torch.tensor([-2.0, 1.0, 1.0, 2.0]) * 0.25,
            )
        ),
        pcb_ratio=0.5,
        pcb_min_ratio=0.0,
        pcb_max_ratio=0.0,
    )
    state, scales = motion_gepae_aggregate(
        global_state,
        client_states,
        weights,
        pcb_ratio=0.5,
        pcb_min_ratio=0.0,
        pcb_max_ratio=0.0,
    )
    actual_delta = torch.cat((state["a"], state["b"].reshape(-1)))
    assert torch.allclose(actual_delta, expected_delta)
    assert torch.allclose(scales, expected_scales)


def test_motion_gepae_rejects_malformed_or_nonfinite_updates() -> None:
    with pytest.raises(ValueError):
        motion_pcb_merge(torch.tensor([[1.0, float("nan")]]))
    with pytest.raises(ValueError):
        motion_gepae_aggregate(
            {"a": torch.zeros(2)},
            {0: {"a": torch.zeros(3)}},
            {0: 1},
        )
