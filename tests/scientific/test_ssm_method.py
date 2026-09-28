from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from gecko.engine.accounting import tensor_payload_bytes
from gecko.algorithms.context import ClientMethodContext
from gecko.algorithms.continual.ssm.node import SSMAlgorithm
from gecko.algorithms.continual.ssm.records import SSMRecord
from gecko.algorithms.continual.ssm.replay import SSMReplayStore
from gecko.algorithms.continual.ssm.records import SSM_MAIN_HOP_BUDGETS
from gecko.algorithms.continual.ssm.records import SSM_NODE_ONLY_HOP_BUDGETS
from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT
from gecko.algorithms.continual.ssm.records import deserialize_ssm_record
from gecko.algorithms.continual.ssm.sampling import sample_sparse_computation_record
from gecko.algorithms.continual.ssm.records import serialize_ssm_record


def _sampling_graph() -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    # Incoming first frontier: 1,2,3 -> 0.  Higher frontiers connect only
    # through an already sampled node.  The final arc is deliberately
    # irrelevant to the root computation graph.
    edges = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 7],
            [0, 0, 0, 1, 1, 2, 3, 6],
        ],
        dtype=torch.long,
    )
    return features, edges


def _sample_record(
    *,
    mode: str = "uniform",
    budgets: tuple[int, ...] = (2, 2),
    seed: int = 17,
    stage: int = 0,
    task: int = 4,
    label: int = 1,
) -> SSMRecord:
    features, edges = _sampling_graph()
    return sample_sparse_computation_record(
        node_features=features,
        edge_index=edges,
        root_index=0,
        root_label=label,
        client_id=3,
        global_task_id=task,
        stage_index=stage,
        sampler_mode=mode,
        hop_budgets=budgets,
        rng_seed=seed,
        rng_base_seed=5,
    )


def _large_node_only_record(
    *,
    width: int,
    source_root: int,
    class_id: int,
    stage: int = 0,
    task: int = 11,
) -> SSMRecord:
    return SSMRecord(
        client_id=7,
        global_task_id=task,
        stage_index=stage,
        class_id=class_id,
        source_root_index=source_root,
        root_index=0,
        root_label=class_id,
        sampler_mode="uniform",
        hop_budgets=SSM_NODE_ONLY_HOP_BUDGETS,
        rng_algorithm="torch.Generator.cpu.manual_seed",
        rng_base_seed=1,
        rng_record_seed=source_root + 1,
        sampled_node_count=0,
        features=torch.arange(width, dtype=torch.float32).reshape(1, width),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        source_local_nodes=torch.tensor([source_root], dtype=torch.long),
    )


def _context(
    *,
    stage: int,
    task: int,
    capture: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    train_queries: torch.Tensor | None = None,
    train_labels: torch.Tensor | None = None,
    valid_class_mask: torch.Tensor | None = None,
    graph_sensitive: bool = False,
) -> ClientMethodContext:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ]
    )
    edges = torch.tensor([[2, 3, 4, 5, 3, 4], [0, 0, 1, 1, 2, 2]], dtype=torch.long)

    def forward(
        model: nn.Module,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        if capture is not None:
            capture.append(
                (
                    queries.detach().cpu().clone(),
                    node_features.detach().cpu().clone(),
                    edge_index.detach().cpu().clone(),
                )
            )
        values = node_features
        if graph_sensitive:
            messages = torch.zeros_like(node_features)
            if edge_index.numel():
                messages.index_add_(0, edge_index[1], node_features[edge_index[0]])
            values = node_features + messages
        return model(values)[queries]

    def encode(
        model: nn.Module,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        layer_index: int | None,
    ) -> torch.Tensor:
        del edge_index, layer_index
        return model(node_features)

    return ClientMethodContext(
        client_id=7,
        global_task_id=task,
        stage_index=stage,
        round_index=0,
        problem_type="NC",
        incremental_setting="class",
        train_queries=(
            torch.tensor([0, 1], dtype=torch.long)
            if train_queries is None
            else train_queries
        ),
        train_labels=(
            torch.tensor([0, 1], dtype=torch.long)
            if train_labels is None
            else train_labels
        ),
        valid_class_mask=(
            torch.tensor([True, True, True])
            if valid_class_mask is None
            else valid_class_mask
        ),
        node_features=features,
        base_edge_index=edges,
        context_edge_index=None,
        forward_queries=forward,
        encode_nodes=encode,
    )


def test_uniform_frontier_sampling_is_without_replacement_and_arc_exact() -> None:
    features, edges = _sampling_graph()
    original_features = features.clone()
    record = _sample_record()

    source_nodes = record.source_local_nodes.tolist()
    assert source_nodes[0] == 0
    assert len(source_nodes) == len(set(source_nodes))
    assert record.sampled_node_count == len(source_nodes) - 1
    # Compact indices are root, then hop 1, then hop 2.  Every retained arc
    # must enter the preceding frontier; no induced or unrelated arc is added.
    assert all(
        int(source) > int(target) for source, target in record.edge_index.t().tolist()
    )
    original_arcs = {
        (int(source), int(target)) for source, target in edges.t().tolist()
    }
    mapped_arcs = {
        (source_nodes[int(source)], source_nodes[int(target)])
        for source, target in record.edge_index.t().tolist()
    }
    first_frontier = set(source_nodes[1:3])
    second_frontier = set(source_nodes[3:])
    expected_arcs = {
        (source, target)
        for source, target in original_arcs
        if (source in first_frontier and target == 0)
        or (source in second_frontier and target in first_frontier)
    }
    assert isinstance(record.edge_count, int)
    assert record.edge_count == len(expected_arcs)
    assert mapped_arcs == expected_arcs
    assert torch.equal(record.features, original_features[record.source_local_nodes])

    # Defensive copies make the record independent of its arrived client view.
    features.fill_(-999)
    edges.fill_(0)
    assert torch.equal(record.features, original_features[record.source_local_nodes])


def test_sampling_is_invariant_to_input_arc_order_and_uses_local_rng_only() -> None:
    features, edges = _sampling_graph()
    state_before = torch.random.get_rng_state().clone()
    first = sample_sparse_computation_record(
        node_features=features,
        edge_index=edges,
        root_index=0,
        root_label=1,
        client_id=3,
        global_task_id=4,
        stage_index=0,
        sampler_mode="uniform",
        hop_budgets=(2, 2),
        rng_seed=17,
        rng_base_seed=5,
    )
    second = sample_sparse_computation_record(
        node_features=features,
        edge_index=edges[:, torch.arange(edges.shape[1] - 1, -1, -1)],
        root_index=0,
        root_label=1,
        client_id=3,
        global_task_id=4,
        stage_index=0,
        sampler_mode="uniform",
        hop_budgets=(2, 2),
        rng_seed=17,
        rng_base_seed=5,
    )
    assert first.record_id == second.record_id
    assert serialize_ssm_record(first) == serialize_ssm_record(second)
    assert torch.equal(torch.random.get_rng_state(), state_before)


def test_degree_sampling_uses_neighbor_in_degree_without_replacement() -> None:
    features = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    # 1 and 2 enter root 0.  Candidate 1 has positive in-degree; candidate 2
    # has none, so a one-node degree sample selects 1 deterministically.
    edges = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 1, 1, 1]], dtype=torch.long)
    record = sample_sparse_computation_record(
        node_features=features,
        edge_index=edges,
        root_index=0,
        root_label=0,
        client_id=0,
        global_task_id=0,
        stage_index=0,
        sampler_mode="degree",
        hop_budgets=(1,),
        rng_seed=9,
    )
    assert record.source_local_nodes.tolist() == [0, 1]
    assert record.edge_index.tolist() == [[1], [0]]
    # When the budget exceeds the positive-weight population, zero-degree
    # candidates fill the remaining slots without replacement rather than
    # causing torch.multinomial to fail.
    all_candidates = sample_sparse_computation_record(
        node_features=features,
        edge_index=edges,
        root_index=0,
        root_label=0,
        client_id=0,
        global_task_id=0,
        stage_index=0,
        sampler_mode="degree",
        hop_budgets=(2,),
        rng_seed=9,
    )
    assert all_candidates.source_local_nodes.tolist() == [0, 1, 2]


def test_node_only_zero_budget_is_a_matched_diagnostic() -> None:
    features, edges = _sampling_graph()
    record = sample_sparse_computation_record(
        node_features=features,
        edge_index=edges,
        root_index=3,
        root_label=2,
        client_id=1,
        global_task_id=2,
        stage_index=0,
        sampler_mode="uniform",
        hop_budgets=SSM_NODE_ONLY_HOP_BUDGETS,
        rng_seed=0,
    )
    assert record.source_local_nodes.tolist() == [3]
    assert torch.equal(record.features, features[3:4])
    assert record.edge_index.shape == (2, 0)
    assert record.context_node_count == 0
    assert record.sampled_node_count == 0


def test_strict_local_endpoint_validation_fails_closed() -> None:
    features = torch.zeros((3, 2))
    with pytest.raises(ValueError, match="strict-local"):
        sample_sparse_computation_record(
            node_features=features,
            edge_index=torch.tensor([[1, 3], [0, 0]], dtype=torch.long),
            root_index=0,
            root_label=0,
            client_id=0,
            global_task_id=0,
            stage_index=0,
            sampler_mode="uniform",
            hop_budgets=(1,),
            rng_seed=0,
        )


def test_safe_record_codec_round_trips_and_detects_tampering() -> None:
    record = _sample_record()
    payload = serialize_ssm_record(record)
    restored = deserialize_ssm_record(payload)
    assert restored.record_id == record.record_id
    assert serialize_ssm_record(restored) == payload
    assert record.serialized_bytes == len(payload)
    assert not hasattr(restored, "labels")
    assert isinstance(restored.root_label, int)
    with pytest.raises(ValueError, match="sampled_node_count"):
        SSMRecord(
            client_id=record.client_id,
            global_task_id=record.global_task_id,
            stage_index=record.stage_index,
            class_id=record.class_id,
            source_root_index=record.source_root_index,
            root_index=record.root_index,
            root_label=record.root_label,
            sampler_mode=record.sampler_mode,
            hop_budgets=record.hop_budgets,
            rng_algorithm=record.rng_algorithm,
            rng_base_seed=record.rng_base_seed,
            rng_record_seed=record.rng_record_seed,
            sampled_node_count=record.sampled_node_count + 1,
            features=record.features,
            edge_index=record.edge_index,
            source_local_nodes=record.source_local_nodes,
        )

    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        deserialize_ssm_record(corrupted)
    with pytest.raises(ValueError, match="trailing"):
        deserialize_ssm_record(payload + b"x")


def test_fixed_16_mib_store_partitions_stage_and_class_without_reuse() -> None:
    store = SSMReplayStore(client_id=7)
    assert store.total_ceiling_bytes == SSM_REPLAY_CEILING_BYTES
    assert store.stage_count == SSM_STAGE_COUNT
    assert store.stage_slice_bytes == 2 * 1024 * 1024

    large_a = _large_node_only_record(width=170_000, source_root=0, class_id=0)
    large_b = _large_node_only_record(width=170_000, source_root=1, class_id=0)
    tiny_other_class = _large_node_only_record(width=2, source_root=2, class_id=1)
    allocation = store.reserve_stage(
        stage_index=0,
        global_task_id=11,
        observed_classes=[0, 1],
        candidate_records=[large_b, tiny_other_class, large_a],
    )
    class_zero = allocation["class_allocations"][0]
    class_one = allocation["class_allocations"][1]
    assert class_zero["quota_bytes"] == 1024 * 1024
    assert class_one["quota_bytes"] == 1024 * 1024
    assert class_zero["inserted_count"] == 1
    assert class_zero["record_ids"] == [large_a.record_id]
    assert class_zero["rejected_count"] == 1
    assert class_one["inserted_count"] == 1
    # The large class cannot consume the tiny class's unused quota.
    assert class_one["unused_bytes"] > class_zero["unused_bytes"]
    assert allocation["used_bytes"] < allocation["stage_slice_bytes"]
    assert allocation["unused_bytes"] > 0
    assert store.used_bytes == sum(
        record.serialized_bytes for record in store.records()
    )
    assert store.safe_checkpoint_bytes == store.used_bytes
    assert tensor_payload_bytes(store.to_state()["payloads"]) == store.used_bytes


def test_stage_slices_are_immutable_and_never_evict_or_reallocate() -> None:
    store = SSMReplayStore(client_id=7)
    stage_zero = _large_node_only_record(
        width=8, source_root=0, class_id=0, stage=0, task=11
    )
    store.reserve_stage(
        stage_index=0,
        global_task_id=11,
        observed_classes=[0],
        candidate_records=[stage_zero],
    )
    first_payload = serialize_ssm_record(store.records()[0])
    with pytest.raises(ValueError, match="already been finalized"):
        store.reserve_stage(
            stage_index=0,
            global_task_id=12,
            observed_classes=[1],
            candidate_records=[],
        )

    stage_one = _large_node_only_record(
        width=8, source_root=1, class_id=1, stage=1, task=12
    )
    store.reserve_stage(
        stage_index=1,
        global_task_id=12,
        observed_classes=[1],
        candidate_records=[stage_one],
    )
    assert serialize_ssm_record(store.records()[0]) == first_payload
    assert store.allocations()[0]["record_ids"] == [stage_zero.record_id]
    with pytest.raises(ValueError, match="exceeds"):
        store.reserve_stage(
            stage_index=8,
            global_task_id=13,
            observed_classes=[2],
            candidate_records=[],
        )


def test_store_preserves_partial_participation_gaps_and_rejects_backfill() -> None:
    sparse_store = SSMReplayStore(client_id=7)
    stage_one = _large_node_only_record(
        width=8, source_root=1, class_id=1, stage=1, task=12
    )
    sparse_store.reserve_stage(
        stage_index=1,
        global_task_id=12,
        observed_classes=[1],
        candidate_records=[stage_one],
    )
    before = sparse_store.to_state()
    skipped_stage_zero = _large_node_only_record(
        width=8, source_root=0, class_id=0, stage=0, task=11
    )
    with pytest.raises(ValueError, match="older skipped stage"):
        sparse_store.reserve_stage(
            stage_index=0,
            global_task_id=11,
            observed_classes=[0],
            candidate_records=[skipped_stage_zero],
        )
    assert sparse_store.allocations().keys() == {1}
    assert sparse_store.used_bytes == tensor_payload_bytes(before["payloads"])
    assert [record.record_id for record in sparse_store.records()] == [
        stage_one.record_id
    ]
    algorithm = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        hop_budgets=SSM_NODE_ONLY_HOP_BUDGETS,
    )
    algorithm.consolidate(nn.Linear(2, 3), _context(stage=1, task=12))
    diagnostics = algorithm.diagnostics()
    assert diagnostics["missed_stage_reserved_bytes"] == 2 * 1024 * 1024
    assert diagnostics["finalized_stage_reserved_bytes"] == 2 * 1024 * 1024
    assert diagnostics["future_stage_reserved_bytes"] == 6 * 2 * 1024 * 1024


def test_sparse_gap_checkpoint_round_trip_preserves_missed_slices() -> None:
    original = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        hop_budgets=SSM_NODE_ONLY_HOP_BUDGETS,
    )
    model = nn.Linear(2, 3)
    stage_one_context = _context(stage=1, task=12)
    original.consolidate(model, stage_one_context)
    expected_record_ids = [
        record.record_id for record in original.replay_samples(stage_one_context)
    ]

    restored = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        hop_budgets=SSM_NODE_ONLY_HOP_BUDGETS,
    )
    restored.load_method_state(original.save_method_state())
    assert restored.diagnostics()["stage_allocations"].keys() == {"1"}
    assert restored.diagnostics()["missed_stage_reserved_bytes"] == 2 * 1024 * 1024
    assert [
        record.record_id for record in restored.replay_samples(stage_one_context)
    ] == expected_record_ids

    with pytest.raises(ValueError, match="older skipped stage"):
        restored.consolidate(model, _context(stage=0, task=11))
    assert [
        record.record_id for record in restored.replay_samples(stage_one_context)
    ] == expected_record_ids

    stage_three_context = _context(stage=3, task=14)
    restored.consolidate(model, stage_three_context)
    diagnostics = restored.diagnostics()
    assert diagnostics["stage_allocations"].keys() == {"1", "3"}
    assert diagnostics["missed_stage_reserved_bytes"] == 2 * 2 * 1024 * 1024
    assert diagnostics["finalized_stage_reserved_bytes"] == 2 * 2 * 1024 * 1024
    assert diagnostics["future_stage_reserved_bytes"] == 4 * 2 * 1024 * 1024

    reloaded = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        hop_budgets=SSM_NODE_ONLY_HOP_BUDGETS,
    )
    reloaded.load_method_state(restored.save_method_state())
    assert reloaded.diagnostics() == diagnostics


def test_store_rejects_duplicate_root_candidates_atomically() -> None:
    duplicate_store = SSMReplayStore(client_id=7)
    duplicate = _large_node_only_record(
        width=8, source_root=0, class_id=0, stage=0, task=11
    )
    with pytest.raises(ValueError, match="must be unique"):
        duplicate_store.reserve_stage(
            stage_index=0,
            global_task_id=11,
            observed_classes=[0],
            candidate_records=[duplicate, duplicate],
        )
    assert duplicate_store.used_bytes == 0
    assert duplicate_store.allocations() == {}


def test_store_checkpoint_round_trip_validates_exact_accounting() -> None:
    store = SSMReplayStore(client_id=7)
    records = [
        _large_node_only_record(width=8, source_root=index, class_id=index % 2)
        for index in range(4)
    ]
    store.reserve_stage(
        stage_index=0,
        global_task_id=11,
        observed_classes=[0, 1],
        candidate_records=records,
    )
    state = store.to_state()
    restored = SSMReplayStore.from_state(state)
    assert [record.record_id for record in restored.records()] == [
        record.record_id for record in store.records()
    ]
    assert restored.used_bytes == store.used_bytes

    corrupted = copy.deepcopy(state)
    corrupted["allocations"][0]["used_bytes"] += 1
    with pytest.raises(ValueError, match="stage serialized-byte accounting"):
        SSMReplayStore.from_state(corrupted)


def test_algorithm_fails_closed_outside_nc_class_and_fixed_configs() -> None:
    with pytest.raises(ValueError, match="NC-Class"):
        SSMAlgorithm(
            client_id=0,
            problem_type="NC",
            incremental_setting="domain",
        )
    with pytest.raises(ValueError, match="permits only"):
        SSMAlgorithm(
            client_id=0,
            problem_type="NC",
            incremental_setting="class",
            hop_budgets=(1, 2),
        )
    with pytest.raises(ValueError, match="fixes replay_weight=1.0"):
        SSMAlgorithm(
            client_id=0,
            problem_type="NC",
            incremental_setting="class",
            replay_weight=0.5,
        )
    for mode in ("uniform", "degree"):
        algorithm = SSMAlgorithm(
            client_id=0,
            problem_type="NC",
            incremental_setting="class",
            sampler_mode=mode,
            hop_budgets=SSM_MAIN_HOP_BUDGETS,
        )
        assert algorithm.hyperparameters()["replay_ceiling_bytes"] == 16 * 1024 * 1024
    diagnostic = SSMAlgorithm(
        client_id=0,
        problem_type="NC",
        incremental_setting="class",
        hop_budgets=SSM_NODE_ONLY_HOP_BUDGETS,
    )
    assert diagnostic.hyperparameters()["hop_budgets"] == [0, 0]


def test_consolidation_rejects_inactive_or_future_class_before_persistence() -> None:
    algorithm = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="uniform",
    )
    context = _context(
        stage=0,
        task=20,
        valid_class_mask=torch.tensor([True, False, True]),
    )
    with pytest.raises(ValueError, match="future or inactive"):
        algorithm.consolidate(nn.Linear(2, 3), context)
    assert algorithm.diagnostics()["replay_targets"] == 0


def test_equation7_inverse_class_frequency_matches_closed_form() -> None:
    algorithm = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="uniform",
    )
    context = _context(
        stage=0,
        task=20,
        train_queries=torch.tensor([0, 1, 2], dtype=torch.long),
        train_labels=torch.tensor([0, 0, 1], dtype=torch.long),
    )
    model = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.1], [-0.3, 0.4], [0.1, 0.2]]))
    logits = context.forward_queries(model)
    base_loss = F.cross_entropy(logits, context.train_labels)
    objective = algorithm.augment_loss(model, context, logits, base_loss)
    per_root = F.cross_entropy(logits, context.train_labels, reduction="none")
    expected = (0.5 * per_root[0] + 0.5 * per_root[1] + per_root[2]) / 2.0
    assert float(objective.detach()) == pytest.approx(float(expected.detach()))
    diagnostics = algorithm.diagnostics()
    assert diagnostics["class_balance_counts"] == {"0": 2, "1": 1}
    assert diagnostics["class_balance_weights"] == {"0": 0.5, "1": 1.0}
    assert diagnostics["replay_loss"] == 0.0


def test_equation7_asymmetric_current_and_replay_matches_closed_form() -> None:
    algorithm = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="uniform",
        hop_budgets=SSM_NODE_ONLY_HOP_BUDGETS,
        seed=31,
    )
    memory_context = _context(
        stage=0,
        task=20,
        train_queries=torch.tensor([0], dtype=torch.long),
        train_labels=torch.tensor([0], dtype=torch.long),
    )
    algorithm.consolidate(nn.Linear(2, 3), memory_context)

    current_context = _context(
        stage=1,
        task=21,
        train_queries=torch.tensor([1, 2, 3], dtype=torch.long),
        train_labels=torch.tensor([0, 0, 1], dtype=torch.long),
    )
    model = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.1], [-0.3, 0.4], [0.1, 0.2]]))
    current_logits = current_context.forward_queries(model)
    record = algorithm.replay_samples(current_context)[0]
    replay_logits = current_context.forward_queries(
        model,
        torch.tensor([record.root_index]),
        node_features=record.features,
        edge_index=record.edge_index,
    )
    objective = algorithm.augment_loss(
        model,
        current_context,
        current_logits,
        F.cross_entropy(current_logits, current_context.train_labels),
    )
    current_per_root = F.cross_entropy(
        current_logits,
        current_context.train_labels,
        reduction="none",
    )
    replay_per_root = F.cross_entropy(
        replay_logits,
        torch.tensor([record.root_label]),
        reduction="none",
    )
    expected = (
        (current_per_root[0] + current_per_root[1] + replay_per_root[0]) / 3.0
        + current_per_root[2]
    ) / 2.0
    assert float(objective.detach()) == pytest.approx(float(expected.detach()))
    diagnostics = algorithm.diagnostics()
    assert diagnostics["class_balance_counts"] == {"0": 3, "1": 1}
    assert diagnostics["class_balance_weights"] == {"0": 1.0 / 3.0, "1": 1.0}


def test_replay_forward_uses_eval_batchnorm_without_disabling_gradients() -> None:
    algorithm = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="uniform",
        hop_budgets=SSM_NODE_ONLY_HOP_BUDGETS,
        seed=37,
    )
    memory_context = _context(
        stage=0,
        task=20,
        train_queries=torch.tensor([0], dtype=torch.long),
        train_labels=torch.tensor([0], dtype=torch.long),
    )
    algorithm.consolidate(nn.Linear(2, 3), memory_context)

    current_context = _context(
        stage=1,
        task=21,
        train_queries=torch.tensor([1, 2, 3], dtype=torch.long),
        train_labels=torch.tensor([0, 0, 1], dtype=torch.long),
    )
    model = nn.Sequential(
        nn.Linear(2, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 3),
    )
    model.train()
    current_logits = current_context.forward_queries(model)
    objective = algorithm.augment_loss(
        model,
        current_context,
        current_logits,
        F.cross_entropy(current_logits, current_context.train_labels),
    )
    assert model.training
    objective.backward()
    assert model[0].weight.grad is not None
    assert bool(torch.isfinite(model[0].weight.grad).all())
    assert algorithm.diagnostics()["replay_loss"] > 0.0


def test_topology_memory_changes_logits_and_gradients_vs_node_only_ablation() -> None:
    topology = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="uniform",
        hop_budgets=SSM_MAIN_HOP_BUDGETS,
        seed=41,
    )
    node_only = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="uniform",
        hop_budgets=SSM_NODE_ONLY_HOP_BUDGETS,
        seed=41,
    )
    arrived = _context(stage=0, task=20, graph_sensitive=True)
    topology.consolidate(nn.Linear(2, 3), arrived)
    node_only.consolidate(nn.Linear(2, 3), arrived)
    assert sum(record.edge_count for record in topology.replay_samples(arrived)) > 0
    assert sum(record.edge_count for record in node_only.replay_samples(arrived)) == 0

    topology_model = nn.Linear(2, 3, bias=False)
    node_only_model = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        topology_model.weight.copy_(
            torch.tensor([[0.2, -0.1], [-0.3, 0.4], [0.1, 0.2]])
        )
        node_only_model.load_state_dict(topology_model.state_dict())
    current = _context(stage=1, task=21, graph_sensitive=True)
    topology_record = topology.replay_samples(current)[0]
    node_only_record = node_only.replay_samples(current)[0]
    topology_replay_logits = current.forward_queries(
        topology_model,
        torch.tensor([topology_record.root_index]),
        node_features=topology_record.features,
        edge_index=topology_record.edge_index,
    )
    node_only_replay_logits = current.forward_queries(
        node_only_model,
        torch.tensor([node_only_record.root_index]),
        node_features=node_only_record.features,
        edge_index=node_only_record.edge_index,
    )
    assert not torch.allclose(topology_replay_logits, node_only_replay_logits)
    topology_logits = current.forward_queries(topology_model)
    node_only_logits = current.forward_queries(node_only_model)
    topology_objective = topology.augment_loss(
        topology_model,
        current,
        topology_logits,
        F.cross_entropy(topology_logits, current.train_labels),
    )
    node_only_objective = node_only.augment_loss(
        node_only_model,
        current,
        node_only_logits,
        F.cross_entropy(node_only_logits, current.train_labels),
    )
    topology_objective.backward()
    node_only_objective.backward()
    assert topology_model.weight.grad is not None
    assert node_only_model.weight.grad is not None
    assert not torch.allclose(topology_objective, node_only_objective)
    assert not torch.allclose(
        topology_model.weight.grad,
        node_only_model.weight.grad,
    )


def test_algorithm_consolidates_owned_records_and_replays_root_labels_only() -> None:
    capture: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    first_context = _context(stage=0, task=20)
    algorithm = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="uniform",
        seed=123,
    )
    model = nn.Linear(2, 3, bias=False)
    algorithm.consolidate(model, first_context)
    records = algorithm.replay_samples(first_context)
    assert [record.source_root_index for record in records] == [0, 1]
    assert [record.root_label for record in records] == [0, 1]
    assert all(record.client_id == 7 for record in records)
    assert all(record.global_task_id == 20 for record in records)

    second_context = _context(stage=1, task=21, capture=capture)
    current_logits = second_context.forward_queries(model)
    base_loss = F.cross_entropy(current_logits, second_context.train_labels)
    capture.clear()
    objective = algorithm.augment_loss(model, second_context, current_logits, base_loss)
    objective.backward()
    assert bool(torch.isfinite(objective))
    assert model.weight.grad is not None
    # One capability call per stored graph, each supervising its single root.
    assert len(capture) == len(records)
    for record, (queries, features, edges) in zip(records, capture):
        assert queries.tolist() == [record.root_index]
        assert torch.equal(features, record.features)
        assert torch.equal(edges, record.edge_index)

    diagnostics = algorithm.diagnostics()
    assert diagnostics["replay_targets"] == 2
    assert diagnostics["class_representation"] == {"0": 1, "1": 1}
    assert diagnostics["class_balance_counts"] == {"0": 2, "1": 2}
    assert diagnostics["class_balance_weights"] == {"0": 0.5, "1": 0.5}
    assert diagnostics["task_representation"] == {"20": 2}
    assert diagnostics["safe_checkpoint_bytes"] == sum(
        record.serialized_bytes for record in records
    )
    assert diagnostics["current_loss"] == pytest.approx(float(base_loss.detach()))
    assert diagnostics["replay_loss"] > 0


def test_algorithm_state_round_trip_is_private_deterministic_and_strict() -> None:
    context = _context(stage=0, task=20)
    original = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="degree",
        seed=8,
    )
    model = nn.Linear(2, 3)
    original.consolidate(model, context)
    next_context = _context(stage=1, task=21)
    logits = next_context.forward_queries(model)
    original.augment_loss(
        model,
        next_context,
        logits,
        F.cross_entropy(logits, next_context.train_labels),
    )
    state = original.save_method_state()
    restored = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="degree",
        seed=8,
    )
    restored.load_method_state(state)
    assert [record.record_id for record in restored.replay_samples(context)] == [
        record.record_id for record in original.replay_samples(context)
    ]
    assert (
        restored.diagnostics()["safe_checkpoint_bytes"]
        == original.diagnostics()["safe_checkpoint_bytes"]
    )
    for key in (
        "current_loss",
        "replay_loss",
        "last_inserted_records",
        "last_rejected_records",
        "class_balance_counts",
        "class_balance_weights",
    ):
        assert restored.diagnostics()[key] == original.diagnostics()[key]

    provenance_tamper = copy.deepcopy(state)
    provenance_tamper["method_hyperparameters"]["sampler_mode"] = "uniform"
    wrong_sampler = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="uniform",
        seed=8,
    )
    with pytest.raises(ValueError, match="provenance"):
        wrong_sampler.load_method_state(provenance_tamper)
    assert wrong_sampler.diagnostics()["replay_targets"] == 0

    corrupted = copy.deepcopy(state)
    corrupted["replay_store"]["payloads"][0][-1] ^= 1
    empty = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="degree",
        seed=8,
    )
    with pytest.raises(ValueError, match="checksum"):
        empty.load_method_state(corrupted)
    assert empty.diagnostics()["replay_targets"] == 0


def test_consolidation_is_independent_of_unavailable_heldout_labels() -> None:
    forbidden = {"heldout_labels", "future_labels", "global_dataset"}

    class LeakageTripwire:
        def __init__(self, base: ClientMethodContext, heldout: torch.Tensor) -> None:
            object.__setattr__(self, "_base", base)
            object.__setattr__(self, "_heldout", heldout.clone())
            object.__setattr__(self, "accessed", set())

        def __getattribute__(self, name: str) -> object:
            if name in forbidden:
                raise AssertionError(f"SSM attempted forbidden access: {name}")
            if name.startswith("_") or name == "accessed":
                return object.__getattribute__(self, name)
            accessed = object.__getattribute__(self, "accessed")
            accessed.add(name)
            return getattr(object.__getattribute__(self, "_base"), name)

    context_a = LeakageTripwire(_context(stage=0, task=20), torch.tensor([0, 1, 2, 2]))
    context_b = LeakageTripwire(_context(stage=0, task=20), torch.tensor([2, 2, 0, 1]))
    first = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="uniform",
        seed=99,
    )
    second = SSMAlgorithm(
        client_id=7,
        problem_type="NC",
        incremental_setting="class",
        sampler_mode="uniform",
        seed=99,
    )
    first.consolidate(nn.Linear(2, 3), context_a)
    second.consolidate(nn.Linear(2, 3), context_b)
    assert [record.record_id for record in first.replay_samples(context_a)] == [
        record.record_id for record in second.replay_samples(context_b)
    ]
    allowed = {
        "base_edge_index",
        "client_id",
        "global_task_id",
        "incremental_setting",
        "node_features",
        "problem_type",
        "stage_index",
        "train_labels",
        "train_queries",
        "valid_class_mask",
    }
    assert context_a.accessed <= allowed
    assert context_b.accessed <= allowed
