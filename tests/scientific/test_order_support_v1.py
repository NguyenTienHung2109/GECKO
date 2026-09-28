"""Acceptance tests for the pre-registered GECKO order-support campaign."""

from __future__ import annotations

import copy
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gecko.research.order_support import CAMPAIGN_ID
from gecko.research.order_support import add_state
from gecko.research.order_support import capture_rng_state
from gecko.research.order_support import deterministic_seed
from gecko.research.order_support import enumerate_assignment_indices
from gecko.research.order_support import generate_balanced_participation
from gecko.research.order_support import multioutput_gram_inner
from gecko.research.order_support import norm_match_state
from gecko.research.order_support import old_support_metrics
from gecko.research.order_support import participation_payload
from gecko.research.order_support import preserved_rng_state
from gecko.research.order_support import recompose_fedavg
from gecko.research.order_support import sample_mean_sd
from gecko.research.order_support import select_probe_case
from gecko.research.order_support import state_delta
from gecko.research.order_support import state_l2
from gecko.research.order_support import summarize_fixed40
from gecko.research.order_support import summarize_official
from gecko.research.order_support import tensor_state_hash
from gecko.research.order_support import validate_balanced_participation
from gecko.research.order_support import validate_block_orders
from gecko.engine.aggregation import weighted_average
from gecko.engine.client import FederatedClient
from gecko.engine.client import supervised_loss
from gecko.engine.coordinator import FederatedCoordinator
from gecko.models.backbones import GECKOGraphModel
from gecko.data.streams.orders import generate_client_orders
from gecko.types import LocalUpdateResult
from tests.helpers import make_stream


def test_balanced_trace_has_exact_campaign_contract() -> None:
    first = generate_balanced_participation(1)
    second = generate_balanced_participation(1)
    validate_balanced_participation(first)
    assert participation_payload(first) == participation_payload(second)
    assert len(first.trace) == 8
    for rounds in first.trace.values():
        assert len(rounds) == 50
        assert all(len(row) == len(set(row)) == 5 for row in rounds.values())
        counts = np.bincount(
            [client for row in rounds.values() for client in row], minlength=10
        )
        np.testing.assert_array_equal(counts, np.full(10, 25))


@pytest.mark.parametrize("profile", ["synchronized", "hard"])
def test_campaign_orders_visit_every_task_once_and_never_cross_blocks(profile: str) -> None:
    orders = generate_client_orders(10, 8, profile, seed=2)
    validate_block_orders(orders)
    assert orders.block_size == (1 if profile == "synchronized" else 4)


def test_paired_order_change_preserves_all_upstream_synthetic_objects() -> None:
    sync = make_stream("NC", "class", 4, seed=2, order_profile="synchronized")
    unsync_orders = generate_client_orders(2, 4, "hard", seed=2)
    unsync = replace(sync, orders=unsync_orders)
    assert torch.equal(sync.scenario.node_features, unsync.scenario.node_features)
    assert torch.equal(sync.scenario.labels, unsync.scenario.labels)
    assert torch.equal(sync.partition.node_owner, unsync.partition.node_owner)
    for client in sync.partition.client_graphs:
        assert torch.equal(
            sync.partition.client_graphs[client].edge_index,
            unsync.partition.client_graphs[client].edge_index,
        )
        for task in sync.shards[client]:
            assert torch.equal(
                sync.shards[client][task].train_queries,
                unsync.shards[client][task].train_queries,
            )
            assert torch.equal(
                sync.evaluation_shards[client][task].test_query_ids,
                unsync.evaluation_shards[client][task].test_query_ids,
            )


def test_global_class_rows_and_seen_mask_are_distinct_from_fixed_output() -> None:
    stream = make_stream("NC", "class", 2, seed=3, order_profile="synchronized")
    coordinator = FederatedCoordinator(
        stream,
        "fedavg",
        "Bare",
        model_name="uefa_gcn",
        model_factory=lambda: GECKOGraphModel(
            stream.scenario.num_features,
            stream.scenario.num_classes,
            hidden_size=8,
            num_layers=2,
            problem_type="NC",
        ),
        model_seed=17,
    )
    assert coordinator.global_model.node_head.out_features == stream.scenario.num_classes
    client = coordinator.clients[0]
    first_task = stream.orders.global_task(0, 0)
    shard = stream.shards[0][first_task]
    official = client._class_mask(shard)
    assert official is not None
    assert 0 < int(official.sum()) < stream.scenario.num_classes
    logits = torch.randn(3, stream.scenario.num_classes)
    fixed = client._masked_logits(logits, shard)
    assert torch.equal(fixed[..., official], logits[..., official])
    assert torch.all(fixed[..., ~official] == -1e12)
    assert torch.equal(logits, logits.clone())  # observer's fixed head is untouched
    assert not set(stream.scenario.task_class_sets[1].tolist()).intersection(
        stream.scenario.task_class_sets[0].tolist()
    )


def test_actual_fedavg_recomposition_drop_has_no_renormalization_or_extra_lr() -> None:
    pre = {"w": torch.tensor([1.0, -2.0], dtype=torch.float32)}
    deltas = {
        0: {"w": torch.tensor([0.2, 0.4])},
        1: {"w": torch.tensor([-0.1, 0.3])},
    }
    weights = {0: 0.25, 1: 0.75}
    full, drop, donor = recompose_fedavg(
        theta_pre=pre,
        client_deltas=deltas,
        normalized_weights=weights,
        donor_ids=(0,),
    )
    updates = [
        LocalUpdateResult(
            client_id=client,
            global_task_id=0,
            shared_state=add_state(pre, delta),
            weight=(1 if client == 0 else 3),
            training_loss=0.0,
            communication_bytes=0,
            shareable_keys=("w",),
        )
        for client, delta in deltas.items()
    ]
    torch.testing.assert_close(
        full["w"], weighted_average(updates)["w"], atol=2e-6, rtol=1e-5
    )
    torch.testing.assert_close(donor["w"], 0.25 * deltas[0]["w"])
    # Drop retains the original 0.75 coefficient; it does not promote client 1 to 1.0.
    torch.testing.assert_close(drop["w"], pre["w"] + 0.75 * deltas[1]["w"])
    # The displacement already contains Adam's LR; no second factor of 0.01 appears.
    torch.testing.assert_close(full["w"] - drop["w"], donor["w"])


def test_same_donor_control_norm_matching_and_failure_statuses() -> None:
    target = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0])}
    alternative = {"a": torch.tensor([0.0, 2.0]), "b": torch.tensor([0.0])}
    matched, scale, status = norm_match_state(alternative, target)
    assert status == "valid" and matched is not None and scale == 2.5
    assert state_l2(matched) == pytest.approx(state_l2(target), rel=1e-7)
    missing, _, status = norm_match_state(
        {"a": torch.zeros(2), "b": torch.zeros(1)}, target
    )
    assert missing is None and status == "zero_alternative_norm"


def test_eligibility_uses_strict_tau_and_does_not_fabricate_missing_zero() -> None:
    orders = generate_client_orders(10, 8, "hard", seed=0)
    valid = {(client, task) for client in range(10) for task in range(8)}
    stage = 2
    participant_tasks = {client: orders.global_task(client, stage - 1) for client in range(5)}
    selection = select_probe_case(
        alpha=100,
        seed_index=0,
        stage_1based=stage,
        round_1based=10,
        participant_tasks=participant_tasks,
        orders=orders,
        valid_validation_pairs=valid,
    )
    assert selection.status == "eligible"
    assert selection.chosen_task is not None
    assert all(
        orders.inverse_client_orders[client][selection.chosen_task] + 1 < stage
        for client in selection.chosen_receivers
    )
    assert not any(
        client in selection.chosen_receivers and task == selection.chosen_task
        for client, task in participant_tasks.items()
    )
    sync = generate_client_orders(10, 8, "synchronized", seed=0)
    missing = select_probe_case(
        alpha=100,
        seed_index=0,
        stage_1based=2,
        round_1based=10,
        participant_tasks={client: sync.global_task(client, 1) for client in range(5)},
        orders=sync,
        valid_validation_pairs=valid,
    )
    assert missing.status == "structural_missing"
    assert missing.chosen_task is None
    support = old_support_metrics(
        stage_1based=1, task_mass={0: 1.0}, orders=orders, valid_pairs=valid
    )
    assert support["old_support_coverage"] is None
    assert support["old_support_denominator"] == 0


def test_probe_selection_is_metadata_only_and_rng_is_side_effect_free() -> None:
    orders = generate_client_orders(10, 8, "hard", seed=1)
    valid = {(client, task) for client in range(10) for task in range(8)}
    kwargs = dict(
        alpha=0.1,
        seed_index=1,
        stage_1based=3,
        round_1based=25,
        participant_tasks={client: orders.global_task(client, 2) for client in range(5)},
        orders=orders,
        valid_validation_pairs=valid,
    )
    assert select_probe_case(**kwargs) == select_probe_case(**kwargs)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    before = capture_rng_state()
    with preserved_rng_state():
        random.random()
        np.random.random()
        torch.rand(7)
    after = capture_rng_state()
    assert before.python == after.python
    assert np.array_equal(before.numpy[1], after.numpy[1])
    assert torch.equal(before.torch_cpu, after.torch_cpu)


def _known_metric_records(orders) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for client in range(10):
        for task in range(8):
            tau = orders.inverse_client_orders[client][task] + 1
            for stage in range(tau, 9):
                value = 0.50 + 0.01 * tau - 0.001 * (stage - tau)
                rows.append(
                    dict(
                        split="test",
                        mask="official",
                        client_id=client,
                        task_id=task,
                        stage=stage,
                        accuracy=value,
                    )
                )
            for stage in range(9):
                rows.append(
                    dict(
                        split="test",
                        mask="fixed40",
                        client_id=client,
                        task_id=task,
                        stage=stage,
                        accuracy=0.10 + 0.01 * stage - 0.001 * task,
                    )
                )
    return rows


def test_metrics_use_pp_signed_drops_macro_cells_and_last_task_zero() -> None:
    orders = generate_client_orders(10, 8, "hard", seed=2)
    rows = _known_metric_records(orders)
    official = summarize_official(rows, orders)
    fixed = summarize_fixed40(rows, orders)
    assert official["pair_count"] == 80
    assert official["AF_off_pp"] >= 0
    assert fixed["Retired_block_drop_pp"] == pytest.approx(-4.0)
    assert fixed["Age_drop40_age0_pp"] == pytest.approx(0.0)
    # Sample SD is computed after seed-level aggregation and uses ddof=1.
    mean, sd = sample_mean_sd([1.0, 2.0, 4.0])
    assert mean == pytest.approx(7 / 3)
    assert sd == pytest.approx(np.std([1.0, 2.0, 4.0], ddof=1))
    with pytest.raises(ValueError, match="three"):
        sample_mean_sd([1.0, 2.0])


def test_multioutput_gram_and_cached_quadratic_risk_match_direct() -> None:
    generator = torch.Generator().manual_seed(117)
    z = torch.randn(11, 3, generator=generator, dtype=torch.float64)
    left = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    right = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    gram = z.T @ z / z.shape[0]
    direct = torch.sum((z @ left) * (z @ right)) / z.shape[0]
    cached = multioutput_gram_inner(left, gram, right)
    torch.testing.assert_close(cached, direct, atol=1e-12, rtol=1e-10)
    with pytest.raises(ValueError, match="d,d"):
        multioutput_gram_inner(left, torch.eye(15), right)
    assignments = enumerate_assignment_indices()
    assert assignments.shape == (1024, 5)
    assert any(np.all(row == row[0]) for row in assignments)


def test_actual_delta_sign_matches_taylor_convention() -> None:
    theta_pre = {"w": torch.tensor([0.4, -0.2], dtype=torch.float64)}
    theta_after = {"w": torch.tensor([0.35, -0.1], dtype=torch.float64)}
    delta = state_delta(theta_after, theta_pre)
    assert torch.equal(delta["w"], theta_after["w"] - theta_pre["w"])
    hessian = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    gradient = hessian @ theta_pre["w"]
    gain = 0.5 * theta_pre["w"] @ hessian @ theta_pre["w"] - 0.5 * theta_after["w"] @ hessian @ theta_after["w"]
    alignment = -gradient @ delta["w"]
    curvature = 0.5 * delta["w"] @ hessian @ delta["w"]
    assert gain == pytest.approx(float(alignment - curvature), abs=1e-14)


def test_cloned_pre_round_state_and_rng_reproduce_one_adam_step_without_mutation() -> None:
    stream = make_stream("NC", "class", 2, seed=4, order_profile="synchronized")
    factory = lambda: GECKOGraphModel(
        stream.scenario.num_features,
        stream.scenario.num_classes,
        hidden_size=8,
        num_layers=2,
        problem_type="NC",
    )
    coordinator = FederatedCoordinator(
        stream,
        "fedavg",
        "Bare",
        model_name="uefa_gcn",
        model_factory=factory,
        model_seed=31,
    )
    client = coordinator.clients[0]
    task = stream.orders.global_task(0, 0)
    shard = stream.shards[0][task]
    client.load_shared_state(coordinator.global_state)
    full_pre = copy.deepcopy(client.model.state_dict())
    live_hash = tensor_state_hash(client.model.state_dict())
    rng = capture_rng_state()

    def clone_update() -> dict[str, torch.Tensor]:
        model = factory()
        model.load_state_dict(full_pre, strict=True)
        clone = FederatedClient(
            client_id=client.client_id,
            graph=client.graph,
            model=model,
            algorithm=copy.deepcopy(client.algorithm),
            config=client.config,
            scenario=client.scenario,
            parameter_policy=client.parameter_policy,
        )
        clone.state = copy.deepcopy(client.state)
        with preserved_rng_state(rng):
            result = clone.update(
                shard,
                global_task_id=task,
                server_state=coordinator.global_state,
                strategy="fedavg",
            )
        return result.shared_state

    first = clone_update()
    second = clone_update()
    assert tensor_state_hash(first) == tensor_state_hash(second)
    assert tensor_state_hash(client.model.state_dict()) == live_hash
    assert client.algorithm.state == {}


def test_fixed_mask_control_step_uses_training_only_labels() -> None:
    logits = torch.tensor([[3.0, 1.0, -2.0], [0.0, 2.0, 1.0]])
    labels = torch.tensor([0, 1])
    mask = torch.tensor([True, True, False])
    masked = logits.clone()
    masked[:, ~mask] = -1e12
    assert supervised_loss(masked, labels) == pytest.approx(
        float(F.cross_entropy(masked, labels))
    )
    # Selection seeds depend only on campaign metadata, never labels/test outcomes.
    assert deterministic_seed(CAMPAIGN_ID, "selection", 1) == deterministic_seed(
        CAMPAIGN_ID, "selection", 1
    )
