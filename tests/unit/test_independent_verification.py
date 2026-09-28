from __future__ import annotations

from dataclasses import replace
import json
import time

import pytest
import torch

from gecko.evaluation.metrics import summarize_continual_matrix
from gecko.evaluation.evaluator import FederatedEvaluator
from gecko.evaluation.metrics import hits_at_k
from gecko.evaluation.metrics import macro_f1
from gecko.evaluation.metrics import mean_reciprocal_rank
from gecko.evaluation.metrics import rocauc
from gecko.engine import FederatedCoordinator
from gecko.data.datasets.synthetic import build_synthetic_spec
from gecko.data.streams import StreamBuilder
from gecko.data.streams import save_stream
from gecko.data.streams.orders import generate_client_orders
from gecko.types import OrderPlan

from tests.helpers import CASES
from tests.helpers import assert_tensor_map_equal
from tests.helpers import make_config
from tests.helpers import make_stream


def _single_client_stream(*, fedprox_mu: float = 0.0):
    config = make_config("NC", "task", 2, num_clients=1)
    config = replace(config, training=replace(config.training, fedprox_mu=fedprox_mu))
    spec = build_synthetic_spec("NC", "task", num_tasks=2, num_clients=1, seed=0)
    return StreamBuilder(config).build(spec)


def _run_one_client(strategy: str):
    torch.manual_seed(982451653)
    coordinator = FederatedCoordinator(
        _single_client_stream(fedprox_mu=0.0),
        strategy,
        "Bare",
        model_name="uefa_gcn",
    )
    result = coordinator.run()
    model = coordinator.clients[0].model if strategy == "local_only" else coordinator.global_model
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    return result, state


def test_independent_minimum_support_holds_for_every_client_task_split():
    for problem, incremental, num_tasks in CASES:
        stream = make_stream(problem, incremental, num_tasks)
        config = stream.config.partition
        minimums = {
            "train": config.minimum_train_queries_per_client_task,
            "val": config.minimum_validation_queries_per_client_task,
            "test": config.minimum_test_queries_per_client_task,
        }
        for client_id, tasks in stream.evaluation_shards.items():
            for task_id, central in tasks.items():
                counts = {
                    "train": central.train_query_ids.numel(),
                    "val": central.validation_query_ids.numel(),
                    "test": central.test_query_ids.numel(),
                }
                if problem == "LP":
                    counts = {
                        "train": int(
                            (stream.scenario.labels[central.train_query_ids] == 1).sum()
                        ),
                        "val": int(
                            (stream.scenario.labels[central.validation_query_ids] == 1).sum()
                        ),
                        "test": int(
                            (stream.scenario.labels[central.test_query_ids] == 1).sum()
                        ),
                    }
                assert all(counts[split] >= minimums[split] for split in minimums), (
                    client_id,
                    task_id,
                    counts,
                    minimums,
                )


def test_independent_order_only_change_preserves_tasks_ownership_queries_and_lp_negatives():
    synchronized = make_stream("LP", "domain", 2, order_profile="synchronized")
    hard = make_stream("LP", "domain", 2, order_profile="hard")
    assert torch.equal(synchronized.partition.node_owner, hard.partition.node_owner)
    assert torch.equal(synchronized.scenario.query_task_ids, hard.scenario.query_task_ids)
    assert torch.equal(synchronized.scenario.query_endpoints, hard.scenario.query_endpoints)
    assert torch.equal(synchronized.scenario.labels, hard.scenario.labels)
    assert_tensor_map_equal(
        synchronized.scenario.query_ids_by_task_split,
        hard.scenario.query_ids_by_task_split,
    )
    for client_id in synchronized.shards:
        for task_id in synchronized.shards[client_id]:
            first = synchronized.shards[client_id][task_id]
            second = hard.shards[client_id][task_id]
            assert torch.equal(first.train_queries, second.train_queries)
            assert torch.equal(first.train_labels, second.train_labels)
            first_eval = synchronized.evaluation_shards[client_id][task_id]
            second_eval = hard.evaluation_shards[client_id][task_id]
            assert torch.equal(first_eval.validation_queries, second_eval.validation_queries)
            assert torch.equal(first_eval.test_queries, second_eval.test_queries)


def test_independent_same_seed_core_artifact_checksums_are_identical(tmp_path):
    config = make_config("LP", "domain", 2, seed=0)

    def generate(root):
        spec = build_synthetic_spec("LP", "domain", num_tasks=2, num_clients=2, seed=0)
        bundle = StreamBuilder(config).build(spec)
        path = save_stream(bundle, root, repository_root=tmp_path)
        return json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    first = generate(tmp_path / "first")
    time.sleep(0.001)
    second = generate(tmp_path / "second")
    assert first["artifact_checksums"] == second["artifact_checksums"]
    assert first["scientific_fingerprint"] == second["scientific_fingerprint"]
    assert first["package_digest"] != second["package_digest"]


@pytest.mark.parametrize("incremental,num_tasks", [("task", 2), ("class", 2), ("domain", 4)])
def test_independent_reverse_lc_arcs_share_logical_owner(incremental, num_tasks):
    stream = make_stream("LC", incremental, num_tasks)
    owner = stream.partition.node_owner
    spec = stream.scenario
    for logical_id, raw_ids in spec.logical_edge_to_raw_edges.items():
        raw_edges = spec.edge_index[:, raw_ids].t()
        source_owner = owner[raw_edges[:, 0]]
        target_owner = owner[raw_edges[:, 1]]
        raw_owner = torch.where(
            source_owner == target_owner,
            source_owner,
            torch.full_like(source_owner, -1),
        )
        assert bool((spec.logical_edge_ids[raw_ids] == logical_id).all())
        assert bool((raw_owner == raw_owner[0]).all())


def test_independent_synchronized_order_has_zero_kendall_mismatch():
    plan = generate_client_orders(10, 8, "synchronized", 0)
    assert plan.diagnostics["normalized_pairwise_kendall_distance"] == 0.0
    assert plan.diagnostics["normalized_distance_from_reference"] == 0.0


def test_independent_hard_order_has_more_realized_mismatch_than_mild():
    mild = generate_client_orders(10, 8, "mild", 0)
    hard = generate_client_orders(10, 8, "hard", 0)
    assert (
        hard.diagnostics["normalized_distance_from_reference"]
        > mild.diagnostics["normalized_distance_from_reference"]
    )
    assert (
        hard.diagnostics["normalized_pairwise_kendall_distance"]
        > mild.diagnostics["normalized_pairwise_kendall_distance"]
    )


def test_independent_fedprox_mu_zero_matches_fedavg():
    fedavg_result, fedavg_state = _run_one_client("fedavg")
    fedprox_result, fedprox_state = _run_one_client("fedprox")
    assert torch.allclose(
        fedavg_result["client_stage_task_matrix"],
        fedprox_result["client_stage_task_matrix"],
        equal_nan=True,
    )
    assert fedavg_state.keys() == fedprox_state.keys()
    for key in fedavg_state:
        assert torch.equal(fedavg_state[key], fedprox_state[key]), key


def test_independent_k1_localonly_fedavg_fedprox_mu_zero_equivalence():
    local_result, local_state = _run_one_client("local_only")
    average_result, average_state = _run_one_client("fedavg")
    prox_result, prox_state = _run_one_client("fedprox")
    for candidate in (average_result, prox_result):
        assert torch.allclose(
            local_result["client_stage_task_matrix"],
            candidate["client_stage_task_matrix"],
            equal_nan=True,
        )
    for key in local_state:
        assert torch.equal(local_state[key], average_state[key]), key
        assert torch.equal(local_state[key], prox_state[key]), key


def test_hand_computed_continual_summary_uses_tau_and_client_query_weights():
    orders = OrderPlan(
        canonical_order=(0, 1, 2),
        client_orders={0: (2, 0, 1), 1: (0, 2, 1)},
        inverse_client_orders={0: {2: 0, 0: 1, 1: 2}, 1: {0: 0, 2: 1, 1: 2}},
        cohort_assignments={0: 0, 1: 1},
        block_size=3,
        seed=0,
        diagnostics={},
    )
    nan = float("nan")
    matrix = torch.tensor(
        [
            [[nan, nan, 0.8], [0.6, nan, 0.7], [0.4, 0.9, 0.5]],
            [[0.9, nan, nan], [0.8, nan, 0.7], [0.6, 0.4, 0.5]],
        ]
    )
    query_counts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    summary = summarize_continual_matrix(
        matrix,
        orders,
        query_counts,
        base_metric="accuracy",
    )
    assert summary["final_average_performance"] == pytest.approx(0.55)
    assert summary["average_forgetting"] == pytest.approx(1.0 / 6.0)
    assert summary["bottom_10_percent_client_final_performance"] == pytest.approx(0.5)
    assert summary["macro_client_final_performance"] == pytest.approx(0.55)
    assert summary["micro_query_final_performance"] == pytest.approx(11.1 / 21.0)

    hits_summary = summarize_continual_matrix(
        matrix,
        orders,
        query_counts,
        base_metric="hits@50",
    )
    assert hits_summary["final_average_performance"] == pytest.approx(11.1 / 21.0)
    assert hits_summary["micro_query_final_performance"] == pytest.approx(11.1 / 21.0)
    assert hits_summary["macro_cell_final_performance"] == pytest.approx(0.55)
    assert hits_summary["macro_client_final_performance"] == pytest.approx(0.55)


def test_hand_computed_task_il_and_client_local_class_il_masks():
    task_stream = make_stream("NC", "task", 2)
    task_evaluator = FederatedEvaluator(task_stream)
    assert torch.equal(
        task_evaluator._class_mask({0, 1}, task_id=1, final_stage=False),
        task_stream.scenario.task_masks[1],
    )

    class_stream = make_stream("LC", "class", 2)
    class_evaluator = FederatedEvaluator(class_stream)
    full_evaluator = FederatedEvaluator(class_stream, class_mask_policy="full")
    assert full_evaluator._class_mask({0}, task_id=0, final_stage=False) is None
    client_id = next(iter(class_stream.orders.client_orders))
    first_task = class_stream.orders.client_orders[client_id][0]
    expected = class_stream.scenario.task_masks[first_task]
    assert torch.equal(
        class_evaluator._class_mask({first_task}, task_id=first_task, final_stage=False),
        expected,
    )
    assert bool(
        class_evaluator._class_mask({first_task}, task_id=first_task, final_stage=True).all()
    )


def test_full_class_output_policy_reaches_client_training_context() -> None:
    stream = make_stream("NC", "class", 2)
    coordinator = FederatedCoordinator(
        stream,
        "fedavg",
        "Bare",
        model_name="uefa_gcn",
        class_mask_policy="full",
    )
    client_id = 0
    task_id = stream.orders.global_task(client_id, 0)
    context = coordinator.clients[client_id].build_method_context(
        stream.shards[client_id][task_id],
        global_task_id=task_id,
        stage_index=0,
        round_index=0,
    )
    assert context.valid_class_mask is None
    with pytest.raises(ValueError, match="class_mask_policy"):
        FederatedCoordinator(stream, class_mask_policy="invalid")


def test_hand_computed_multilabel_rocauc_masks_missing_targets():
    logits = torch.tensor(
        [[0.1, 0.0, 0.1], [0.9, 0.2, 0.2], [0.8, 0.8, 0.3], [0.2, 1.0, 0.4]]
    )
    labels = torch.tensor([[0, -1, 1], [1, 1, 1], [1, 0, 1], [0, -1, 1]])
    assert rocauc(logits, labels) == pytest.approx(0.5)


def test_hand_computed_macro_f1():
    predictions = torch.tensor([0, 0, 1, 2, 2, 2])
    logits = torch.nn.functional.one_hot(predictions, num_classes=3).float()
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    assert macro_f1(logits, labels) == pytest.approx((1.0 + 2.0 / 3.0 + 0.8) / 3.0)


def test_hand_computed_hits50_and_mrr():
    hit_scores = torch.cat([torch.arange(60, dtype=torch.float32), torch.tensor([100.0, 10.0])])
    hit_labels = torch.cat([torch.zeros(60, dtype=torch.long), torch.ones(2, dtype=torch.long)])
    assert hits_at_k(hit_scores, hit_labels, 50) == pytest.approx(0.5)

    rank_scores = torch.tensor([0.2, 0.4, 0.6, 0.7, 0.4])
    rank_labels = torch.tensor([0, 0, 0, 1, 1])
    assert mean_reciprocal_rank(rank_scores, rank_labels) == pytest.approx(2.0 / 3.0)


def test_hits50_uses_explicit_candidate_groups_not_a_global_pool():
    first_negatives = torch.arange(50, dtype=torch.float32)
    second_negatives = torch.arange(100, 150, dtype=torch.float32)
    scores = torch.cat(
        [first_negatives, torch.tensor([1.0]), second_negatives, torch.tensor([50.0])]
    )
    labels = torch.cat(
        [torch.zeros(50), torch.ones(1), torch.zeros(50), torch.ones(1)]
    ).long()
    groups = torch.cat(
        [torch.zeros(51), torch.ones(51)]
    ).long()
    assert hits_at_k(scores, labels, 50, group_ids=groups) == pytest.approx(0.5)


def test_hits50_tie_policies_are_explicit():
    scores = torch.cat([torch.arange(50, dtype=torch.float32), torch.tensor([0.0])])
    labels = torch.cat([torch.zeros(50), torch.ones(1)]).long()
    assert hits_at_k(scores, labels, 50, tie_policy="pessimistic") == 0.0
    assert hits_at_k(scores, labels, 50, tie_policy="optimistic") == 1.0
    assert hits_at_k(scores, labels, 50, tie_policy="average") == 0.5


def test_hits50_rejects_candidate_groups_with_too_few_negatives():
    scores = torch.cat([torch.arange(49, dtype=torch.float32), torch.tensor([100.0])])
    labels = torch.cat([torch.zeros(49), torch.ones(1)]).long()
    with pytest.raises(ValueError, match="only 49 negatives"):
        hits_at_k(scores, labels, 50)
