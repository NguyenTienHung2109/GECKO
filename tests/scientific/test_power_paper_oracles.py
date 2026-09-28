from __future__ import annotations

import pytest
import torch

from gecko.algorithms.federated.power.oracles import power_append_replay
from gecko.algorithms.federated.power.oracles import power_class_mean_prototypes
from gecko.algorithms.federated.power.oracles import power_local_global_coverage_selection
from gecko.algorithms.federated.power.oracles import power_replay_payload_bytes


def test_power_eq8_uses_class_mean_not_first_node():
    features = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]]
    )
    prototypes, classes, counts = power_class_mean_prototypes(
        node_features=features,
        train_queries=torch.tensor([0, 1, 2, 3]),
        train_labels=torch.tensor([0, 0, 1, 1]),
    )
    assert torch.equal(classes, torch.tensor([0, 1]))
    assert torch.equal(counts, torch.tensor([2, 2]))
    assert torch.equal(prototypes, torch.tensor([[2.0, 0.0], [0.0, 3.0]]))


def test_power_coverage_selects_highest_local_global_coverage_deterministically():
    local = torch.tensor(
        [[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [4.0, 0.0]]
    )
    global_ = local.clone()
    queries, labels = power_local_global_coverage_selection(
        local_embeddings=local,
        global_embeddings=global_,
        train_queries=torch.tensor([0, 1, 2, 3]),
        train_labels=torch.tensor([2, 2, 2, 2]),
        coverage_threshold=0.1,
    )
    assert torch.equal(queries, torch.tensor([0]))
    assert torch.equal(labels, torch.tensor([2]))


def test_power_replay_is_feature_only_accumulative_and_ceiling_checked():
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    first = power_append_replay(
        replay_state=None,
        node_features=features,
        selected_queries=torch.tensor([0]),
        selected_labels=torch.tensor([1]),
        ceiling_bytes=1024,
    )
    second = power_append_replay(
        replay_state=first,
        node_features=features,
        selected_queries=torch.tensor([2]),
        selected_labels=torch.tensor([3]),
        ceiling_bytes=1024,
    )
    assert torch.equal(second["features"], torch.tensor([[1.0, 2.0], [5.0, 6.0]]))
    assert torch.equal(second["labels"], torch.tensor([1, 3]))
    assert power_replay_payload_bytes(second) == 2 * 2 * 4 + 2 * 8
    with pytest.raises(ValueError, match="byte ceiling"):
        power_append_replay(
            replay_state=first,
            node_features=features,
            selected_queries=torch.tensor([2]),
            selected_labels=torch.tensor([3]),
            ceiling_bytes=1,
        )
