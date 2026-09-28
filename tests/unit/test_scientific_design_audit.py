from __future__ import annotations

from dataclasses import replace
import statistics

import pytest
import torch

from gecko.config import GECKOConfig
from gecko.data.partitioning import materialize as ownership_module
from gecko.data.partitioning.assignment import assign_micro_communities
from gecko.data.partitioning.materialize import derive_strict_local_node_features
from gecko.data.splits.lp import BASE_EDGE_SELECTOR
from gecko.data.scenario import build_lp_spec
from gecko.data.splits.lp import finalize_partition_first_lp
from gecko.data.splits.lp import prepare_partition_first_lp_pool
from gecko.data.splits.lp import stable_base_edge_mask
from gecko.data.scenario import build_nc_spec
from gecko.data.datasets.synthetic import build_synthetic_spec
from gecko.data.streams import StreamBuilder
from gecko.data.streams import load_stream
from gecko.data.streams import save_stream
from gecko.data.streams.orders import generate_client_orders
from gecko.validation import ConfigurationError
from gecko.validation import DatasetDesignError
from gecko.validation import PartitionInfeasibleError

from tests.helpers import make_config
from tests.helpers import make_stream


@pytest.mark.parametrize("num_tasks", [2, 3, 4, 8])
@pytest.mark.parametrize("seed", range(5))
def test_order_properties_across_supported_task_counts_and_seeds(num_tasks, seed):
    plans = {
        profile: generate_client_orders(10, num_tasks, profile, seed)
        for profile in ("synchronized", "mild", "hard")
    }
    canonical = tuple(range(num_tasks))
    for plan in plans.values():
        for client, order in plan.client_orders.items():
            assert tuple(sorted(order)) == canonical
            assert all(
                plan.inverse_client_orders[client][task] == stage
                for stage, task in enumerate(order)
            )
    synchronized = plans["synchronized"].diagnostics
    assert synchronized["normalized_pairwise_kendall_distance"] == 0.0
    assert synchronized["normalized_distance_from_reference"] == 0.0
    assert synchronized["global_exposure_curve"] == {
        stage: stage + 1 for stage in range(num_tasks)
    }

    mild_distance = plans["mild"].diagnostics["normalized_distance_from_reference"]
    hard_distance = plans["hard"].diagnostics["normalized_distance_from_reference"]
    if num_tasks == 2:
        assert hard_distance == mild_distance
    else:
        assert hard_distance > mild_distance


def test_spatial_semantic_divergence_increases_on_average_across_five_seeds():
    realized = {profile: [] for profile in ("easy", "mild", "hard")}
    spec = build_synthetic_spec("NC", "task", num_tasks=4, num_clients=4)
    for seed in range(5):
        for profile in realized:
            base = make_config("NC", "task", 4, spatial_profile=profile)
            config = replace(base, seed=seed)
            stream = StreamBuilder(config).build(spec)
            values = [
                float(value)
                for value in stream.partition.diagnostics["label_js_divergence"]
                if value is not None
            ]
            realized[profile].append(statistics.fmean(values))
            assert not stream.partition.diagnostics["constraint_failures"]
    means = {
        profile: statistics.fmean(values)
        for profile, values in realized.items()
    }
    assert means["easy"] < means["mild"] < means["hard"], means


def test_paired_query_partition_records_and_controls_task_volume_metric():
    realized = {profile: [] for profile in ("easy", "mild", "hard")}
    spec = build_synthetic_spec("LC", "domain", num_tasks=4, num_clients=4)
    for seed in range(5):
        for profile in realized:
            config = replace(
                make_config("LC", "domain", 4, spatial_profile=profile),
                seed=seed,
            )
            stream = StreamBuilder(config).build(spec)
            diagnostics = stream.partition.diagnostics
            assert diagnostics["semantic_control_status"] == "controlled"
            assert diagnostics["semantic_control_metric"] == "client_task_volume_l2"
            value = diagnostics["assignment_task_volume_l2_divergence"]
            assert value is not None
            realized[profile].append(float(value))
            assert not diagnostics["constraint_failures"]
    means = {
        profile: statistics.fmean(values)
        for profile, values in realized.items()
    }
    assert means["easy"] < means["mild"] < means["hard"], means


@pytest.mark.parametrize("num_tasks", [2, 3])
def test_real_small_t_hard_order_fails_before_dataset_loading(num_tasks):
    with pytest.raises(ConfigurationError, match="unsupported when num_tasks < 4"):
        GECKOConfig.from_mapping(
            {
                "scenario": {
                    "dataset": "real-placeholder",
                    "problem": "NC",
                    "incremental_setting": "task",
                    "num_tasks": num_tasks,
                    "metrics": ["accuracy"],
                },
                "order": {"profile": "hard"},
            }
        )


def test_proteins_species_with_official_split_fails_before_loading():
    with pytest.raises(ConfigurationError, match="species domains conflict"):
        GECKOConfig.from_mapping(
            {
                "scenario": {
                    "dataset": "ogbn-proteins",
                    "problem": "NC",
                    "incremental_setting": "domain",
                    "num_tasks": 8,
                    "metrics": ["rocauc"],
                    "domain_constructor": "species",
                    "split_protocol": "official_dataset",
                    "feature_provenance": "strict_local_derived",
                }
            }
        )


def test_strict_local_derived_features_exclude_cross_client_edges():
    spec = build_nc_spec(
        dataset_name="feature-provenance",
        incremental_type="domain",
        metrics=("accuracy",),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        node_features=torch.zeros(4, 1),
        labels=torch.tensor([0, 1, 0, 1]),
        task_ids=torch.zeros(4, dtype=torch.long),
        train_mask=torch.tensor([True, False, False, False]),
        validation_mask=torch.tensor([False, True, False, False]),
        test_mask=torch.tensor([False, False, True, True]),
        num_tasks=1,
        num_classes=2,
        metadata={
            "feature_provenance": "strict_local_derived",
            "context_edge_features": torch.tensor([[1.0], [100.0]]),
            "context_edge_feature_valid_mask": torch.tensor([True, True]),
        },
    )
    derived = derive_strict_local_node_features(
        spec,
        global_nodes=torch.tensor([0, 1]),
        internal_edge_mask=torch.tensor([True, False]),
    )
    assert torch.equal(derived, torch.tensor([[1.0], [0.0]]))


def test_strict_local_derived_features_survive_artifact_round_trip(tmp_path):
    config = make_config("NC", "domain", 1, num_clients=1)
    config = replace(
        config,
        scenario=replace(
            config.scenario, feature_provenance="strict_local_derived"
        ),
    )
    spec = build_nc_spec(
        dataset_name="synthetic",
        incremental_type="domain",
        metrics=("rocauc",),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        node_features=torch.zeros(4, 1),
        labels=torch.tensor([0, 1, 0, 1]),
        task_ids=torch.zeros(4, dtype=torch.long),
        train_mask=torch.tensor([True, False, False, False]),
        validation_mask=torch.tensor([False, True, False, False]),
        test_mask=torch.tensor([False, False, True, True]),
        num_tasks=1,
        num_classes=2,
        metadata={
            "feature_provenance": "strict_local_derived",
            "context_edge_features": torch.tensor([[1.0], [100.0]]),
            "context_edge_feature_valid_mask": torch.tensor([True, True]),
        },
    )
    stream = StreamBuilder(config).build(spec)
    restored = load_stream(save_stream(stream, tmp_path, repository_root=tmp_path))
    assert torch.equal(
        restored.partition.client_graphs[0].node_features,
        stream.partition.client_graphs[0].node_features,
    )


def test_lp_partitioner_receives_only_heldout_free_context(monkeypatch):
    config = make_config("LP", "domain", 2)
    spec = build_synthetic_spec("LP", "domain", num_tasks=2, num_clients=2)
    captured = {}
    original = ownership_module.generate_micro_communities

    def spy(edge_index, *args, **kwargs):
        captured["edge_index"] = edge_index.clone()
        return original(edge_index, *args, **kwargs)

    monkeypatch.setattr(ownership_module, "generate_micro_communities", spy)
    StreamBuilder(config).build(spec)
    assert torch.equal(captured["edge_index"], spec.context_edge_index)
    heldout = spec.metadata["positive_pairs"][spec.metadata["positive_splits"] != 0]
    context = {
        (min(source, target), max(source, target))
        for source, target in captured["edge_index"].t().tolist()
    }
    assert not context.intersection(
        {(min(source, target), max(source, target)) for source, target in heldout.tolist()}
    )


def test_lp_context_preserves_unsupervised_training_positives():
    supervised = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    known = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
    context = torch.tensor([[0, 1], [4, 5]], dtype=torch.long)
    safe_negative = torch.tensor([[0, 4]], dtype=torch.long)
    negatives = {
        0: {
            "train": safe_negative,
            "val": safe_negative,
            "test": safe_negative,
        }
    }
    spec = build_lp_spec(
        dataset_name="background-positive",
        metrics=("hits@1",),
        node_features=torch.randn(6, 4),
        positive_pairs=supervised,
        positive_task_ids=torch.zeros(2, dtype=torch.long),
        positive_splits=torch.tensor([0, 1]),
        negative_pairs_by_task_split=negatives,
        num_tasks=1,
        undirected=True,
        known_positive_pairs=known,
        context_positive_pairs=context,
    )
    actual = {
        (min(source, target), max(source, target))
        for source, target in spec.context_edge_index.t().tolist()
    }
    assert actual == {(0, 1), (4, 5)}
    assert torch.equal(spec.metadata["known_positive_pairs"], known)


def test_lp_context_preserves_raw_edge_order_and_multiplicity():
    supervised = torch.tensor([[0, 1]], dtype=torch.long)
    raw_context = torch.tensor(
        [[1, 0, 1, 2, 2], [0, 1, 0, 2, 2]], dtype=torch.long
    )
    safe_negative = torch.tensor([[0, 2]], dtype=torch.long)
    spec = build_lp_spec(
        dataset_name="raw-context",
        metrics=("hits@1",),
        node_features=torch.randn(3, 2),
        positive_pairs=supervised,
        positive_task_ids=torch.zeros(1, dtype=torch.long),
        positive_splits=torch.zeros(1, dtype=torch.long),
        negative_pairs_by_task_split={
            0: {
                "train": safe_negative,
                "val": safe_negative,
                "test": safe_negative,
            }
        },
        num_tasks=1,
        undirected=True,
        known_positive_pairs=torch.tensor([[0, 1], [2, 2]]),
        context_edge_index=raw_context,
    )
    assert torch.equal(spec.context_edge_index, raw_context)
    assert torch.equal(spec.edge_index, raw_context)


def test_lp_negative_cannot_collide_with_unsupervised_known_positive():
    supervised = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    known = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
    collision = torch.tensor([[4, 5]], dtype=torch.long)
    negatives = {
        0: {"train": collision, "val": collision, "test": collision}
    }
    with pytest.raises(ValueError, match="known positive"):
        build_lp_spec(
            dataset_name="background-positive",
            metrics=("hits@1",),
            node_features=torch.randn(6, 4),
            positive_pairs=supervised,
            positive_task_ids=torch.zeros(2, dtype=torch.long),
            positive_splits=torch.tensor([0, 1]),
            negative_pairs_by_task_split=negatives,
            num_tasks=1,
            undirected=True,
            known_positive_pairs=known,
            context_positive_pairs=torch.tensor([[0, 1], [4, 5]]),
        )


def _small_lp_spec_with_eight_evaluation_negatives():
    positives = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], dtype=torch.long
    )
    tasks = torch.tensor([0, 0, 0, 1, 1, 1])
    splits = torch.tensor([0, 1, 2, 0, 1, 2])
    positive_keys = {tuple(pair) for pair in positives.tolist()}
    candidates = []
    for source in range(12):
        for target in range(source + 1, 12):
            if (source, target) not in positive_keys:
                candidates.append((source, target))
    pool = torch.tensor(candidates[:8], dtype=torch.long)
    negatives = {
        task: {"train": pool[:1], "val": pool, "test": pool}
        for task in range(2)
    }
    return build_lp_spec(
        dataset_name="small-candidates",
        metrics=("hits@50",),
        node_features=torch.randn(12, 4),
        positive_pairs=positives,
        positive_task_ids=tasks,
        positive_splits=splits,
        negative_pairs_by_task_split=negatives,
        num_tasks=2,
        undirected=True,
    )


def test_hits50_with_fewer_than_50_negatives_fails_stream_design_audit():
    config = make_config("LP", "domain", 2, num_clients=1)
    with pytest.raises(DatasetDesignError, match="negatives=8 required=50"):
        StreamBuilder(config).build(_small_lp_spec_with_eight_evaluation_negatives())


def test_unacceptable_lc_internal_query_coverage_fails():
    config = make_config("LC", "task", 2)
    config = replace(
        config,
        partition=replace(
            config.partition, minimum_lc_internal_query_coverage=0.99
        ),
    )
    spec = build_synthetic_spec("LC", "task", num_tasks=2, num_clients=2)
    with pytest.raises(
        PartitionInfeasibleError, match="internal_query_coverage"
    ):
        StreamBuilder(config).build(spec)


def test_unacceptable_lp_internal_positive_coverage_fails():
    config = make_config("LP", "domain", 2)
    config = replace(
        config,
        partition=replace(
            config.partition,
            minimum_lp_internal_candidate_coverage=0.0,
            minimum_lp_internal_positive_coverage=0.99,
        ),
    )
    spec = build_synthetic_spec("LP", "domain", num_tasks=2, num_clients=2)
    with pytest.raises(DatasetDesignError, match="LP internal-positive coverage"):
        StreamBuilder(config).build(spec)


def test_train_only_lp_partition_is_invariant_to_evaluation_endpoints():
    stream = make_stream("LP", "domain", 2)
    spec = stream.scenario
    changed_endpoints = spec.query_endpoints.clone()
    evaluation_ids = torch.cat(
        [
            ids
            for splits in spec.query_ids_by_task_split.values()
            for split, ids in splits.items()
            if split in {"val", "test"}
        ]
    )
    changed_endpoints[evaluation_ids, 0] = (
        changed_endpoints[evaluation_ids, 0] + 1
    ) % spec.node_features.shape[0]
    changed_endpoints[evaluation_ids, 1] = (
        changed_endpoints[evaluation_ids, 1] + 3
    ) % spec.node_features.shape[0]
    changed = replace(spec, query_endpoints=changed_endpoints)
    arguments = {
        "micro_ids": stream.partition.micro_community_ids,
        "num_clients": 2,
        "seed": 0,
        "spatial_profile": "mild",
        "client_size_tolerance": 0.60,
        "minimum_support": (1, 1, 1),
        "maximum_iterations": 1,
        "allow_infeasible": False,
        "minimum_internal_candidate_coverage": None,
        "lp_partition_information_scope": "topology_and_train_queries",
    }
    original = assign_micro_communities(spec, **arguments)
    modified = assign_micro_communities(changed, **arguments)
    assert torch.equal(original.node_owner, modified.node_owner)


def _partition_first_provisional_spec():
    num_nodes = 80
    pairs = []
    tasks = []
    for client_start in (0, 40):
        for task in range(2):
            for offset in range(8):
                source = client_start + task * 10 + offset
                target = client_start + task * 10 + offset + 1
                pairs.append((source, target))
                tasks.append(task)
    positives = torch.tensor(pairs, dtype=torch.long)
    base = torch.tensor([[0, 40]], dtype=torch.long)
    known = torch.cat((base, positives), dim=0)
    empty_negatives = {
        task: {
            split: torch.empty((0, 2), dtype=torch.long)
            for split in ("train", "val", "test")
        }
        for task in range(2)
    }
    return build_lp_spec(
        dataset_name="partition-first-synthetic",
        metrics=("hits@50",),
        node_features=torch.randn(num_nodes, 4),
        positive_pairs=positives,
        positive_task_ids=torch.tensor(tasks, dtype=torch.long),
        positive_splits=torch.zeros(len(pairs), dtype=torch.long),
        negative_pairs_by_task_split=empty_negatives,
        num_tasks=2,
        undirected=True,
        known_positive_pairs=known,
        context_edge_index=torch.tensor([[0, 40], [40, 0]], dtype=torch.long),
        metadata={
            "partition_first_pending": True,
            "partition_first_protocol": "partition_first_query_split_v1",
            "partition_base_policy": "endpoint_hash_holdout",
            "partition_base_edge_ratio": 0.05,
            "partition_base_positive_pairs": base,
            "partition_query_positive_pool_pairs": positives,
            "partition_query_positive_pool_count": len(pairs),
        },
    )


def test_partition_first_finalizer_splits_after_ownership_without_leakage():
    provisional = _partition_first_provisional_spec()
    owner = torch.cat((torch.zeros(40), torch.ones(40))).long()
    final = finalize_partition_first_lp(
        provisional,
        owner,
        num_clients=2,
        seed=0,
        minimum_support=(5, 1, 1),
        training_negative_ratio=1.0,
        evaluation_negatives_per_client_task=50,
    )
    assert final.metadata["partition_first_release_protocol"] is True
    query_owner = owner[final.query_endpoints[:, 0]]
    assert torch.equal(query_owner, owner[final.query_endpoints[:, 1]])
    context = {
        (min(source, target), max(source, target))
        for source, target in final.context_edge_index.t().tolist()
    }
    for task, splits in final.query_ids_by_task_split.items():
        for split, ids in splits.items():
            labels = final.labels[ids]
            for client in range(2):
                local = ids[query_owner[ids] == client]
                positive_count = int((final.labels[local] == 1).sum())
                assert positive_count >= {"train": 5, "val": 1, "test": 1}[split]
            if split in {"val", "test"}:
                for source, target in final.query_endpoints[ids][labels == 1].tolist():
                    assert (min(source, target), max(source, target)) not in context
    base_context = set(map(tuple, final.context_edge_index.t().tolist()))
    task_contexts = final.metadata["task_context_edge_index"]
    assert set(task_contexts) == {0, 1}
    for task, task_context in task_contexts.items():
        task_arcs = set(map(tuple, task_context.t().tolist()))
        assert base_context <= task_arcs
        train_ids = final.query_ids_by_task_split[task]["train"]
        train_pairs = final.query_endpoints[train_ids][final.labels[train_ids] == 1]
        expected_train_arcs = {
            arc
            for source, target in train_pairs.tolist()
            for arc in ((source, target), (target, source))
        }
        assert task_arcs == base_context | expected_train_arcs
        other_ids = final.query_ids_by_task_split[1 - task]["train"]
        other_pairs = final.query_endpoints[other_ids][final.labels[other_ids] == 1]
        other_arcs = {
            arc
            for source, target in other_pairs.tolist()
            for arc in ((source, target), (target, source))
        }
        assert (other_arcs - base_context).isdisjoint(task_arcs)
    groups = final.metadata["candidate_group_ids"]
    assert torch.unique(groups).numel() == 2 * 2 * 3
    known = {
        tuple(pair) for pair in final.metadata["known_positive_pairs"].tolist()
    }
    negatives = final.query_endpoints[final.labels == 0]
    assert all(tuple(pair) not in known for pair in negatives.tolist())


def test_partition_first_builder_retries_topology_proposals_before_split(
    monkeypatch,
):
    import gecko.data.streams.builder as builder_module

    provisional = _partition_first_provisional_spec()
    base = make_config("LP", "domain", 2, order_profile="synchronized")
    config = replace(
        base,
        scenario=replace(
            base.scenario,
            metrics=("average_precision",),
            minimum_evaluation_negatives=1,
            lp_evaluation_negatives_per_client_task=5,
        ),
        partition=replace(
            base.partition,
            lp_partition_information_scope="topology_only",
            maximum_lp_partition_proposals=2,
            minimum_train_queries_per_client_task=1,
            minimum_validation_queries_per_client_task=0,
            minimum_test_queries_per_client_task=0,
        ),
    )
    original = builder_module.finalize_partition_first_lp
    calls = 0

    def reject_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError(
                "Partition-first LP support is infeasible:\n- fixture proposal"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        builder_module, "finalize_partition_first_lp", reject_first
    )
    monkeypatch.setattr(
        builder_module,
        "audit_partition_design",
        lambda config, spec, partition, evaluation_shards, scenario_audit: {
            **scenario_audit,
            "status": "passed",
        },
    )
    stream = StreamBuilder(config).build(provisional)
    assert calls == 2
    assert stream.scenario.metadata["partition_proposal_index"] == 1
    assert stream.scenario.metadata["partition_rejected_proposal_count"] == 1
    assert stream.partition.diagnostics["lp_partition_proposal_index"] == 1
    assert stream.partition.diagnostics["lp_partition_proposal_selection"] == (
        "topology_only_then_pre_split_positive_support_audit"
    )


def test_partition_first_pool_is_disjoint_and_domain_mapping_is_explicit():
    source = _partition_first_provisional_spec()
    source = replace(
        source,
        num_tasks=8,
        query_ids_by_task_split={
            task: {
                "train": torch.nonzero(
                    source.metadata["positive_task_ids"] == (task % 2), as_tuple=True
                )[0],
                "val": torch.empty(0, dtype=torch.long),
                "test": torch.empty(0, dtype=torch.long),
            }
            for task in range(8)
        },
        metadata={
            **source.metadata,
            "positive_task_ids": torch.arange(
                source.metadata["positive_pairs"].shape[0]
            ).remainder(8),
        },
    )
    prepared = prepare_partition_first_lp_pool(
        source,
        target_num_tasks=2,
        source_num_tasks=8,
        base_edge_ratio=0.20,
        seed=0,
        domain_mapping="dominant_source_vs_rest_v1",
    )
    base = {
        tuple(pair) for pair in prepared.metadata["partition_base_positive_pairs"].tolist()
    }
    query = {
        tuple(pair)
        for pair in prepared.metadata["partition_query_positive_pool_pairs"].tolist()
    }
    assert base.isdisjoint(query)
    assert int(prepared.metadata["positive_task_ids"].max()) < 2
    assert (
        prepared.metadata["partition_domain_mapping"]
        == "dominant_source_vs_rest_v1"
    )
    assert prepared.metadata["partition_source_to_target_mapping"] == {
        0: 0,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
    }
    assert prepared.metadata["partition_base_selector"] == BASE_EDGE_SELECTOR


def test_stable_base_selector_is_width_orientation_and_save_load_stable(tmp_path):
    pairs32 = torch.tensor([[1, 9], [9, 1], [2, 7], [3, 4]], dtype=torch.int32)
    pairs64 = pairs32.long()
    undirected32 = stable_base_edge_mask(
        pairs32, seed=17, ratio=0.5, undirected=True
    )
    undirected64 = stable_base_edge_mask(
        pairs64, seed=17, ratio=0.5, undirected=True
    )
    assert torch.equal(undirected32, undirected64)
    assert undirected64[0] == undirected64[1]
    path = tmp_path / "pairs.pt"
    torch.save(pairs64, path)
    loaded = torch.load(path)
    assert torch.equal(
        undirected64,
        stable_base_edge_mask(loaded, seed=17, ratio=0.5, undirected=True),
    )
    directed = stable_base_edge_mask(
        pairs64, seed=17, ratio=0.1, undirected=False
    )
    undirected_low_ratio = stable_base_edge_mask(
        pairs64, seed=17, ratio=0.1, undirected=True
    )
    assert directed.tolist() == [False, True, False, False]
    assert undirected_low_ratio.tolist() == [False, False, False, False]


def test_binary_mismatch_is_balanced_and_requires_two_tasks():
    plan = generate_client_orders(10, 2, "binary_mismatch", 0)
    counts = {
        order: list(plan.client_orders.values()).count(order)
        for order in set(plan.client_orders.values())
    }
    assert counts == {(0, 1): 5, (1, 0): 5}
    assert plan.diagnostics["normalized_pairwise_kendall_distance"] > 0
    with pytest.raises(ValueError, match="exactly two tasks"):
        generate_client_orders(10, 3, "binary_mismatch", 0)
