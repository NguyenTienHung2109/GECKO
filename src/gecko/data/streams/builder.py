"""Build fixed client graphs, query shards, orders, and participation."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Dict

import torch

from gecko.config import GECKOConfig
from gecko.benchmarks.terminology import allocation_slug
from gecko.benchmarks.terminology import task_order_label
from gecko.data.audits.design import audit_partition_design
from gecko.data.audits.design import audit_scenario_design
from gecko.data.partitioning import PersistentSubgraphPartitioner
from gecko.data.partitioning.materialize import rematerialize_partition_context
from gecko.data.splits.lp import finalize_partition_first_lp
from gecko.types import CentralEvaluationShard
from gecko.types import ClientTaskShard
from gecko.types import OrderPlan
from gecko.types import ParticipationPlan
from gecko.types import PartitionResult
from gecko.types import ScenarioSpec
from gecko.validation import PartitionInfeasibleError
from gecko.validation import ScenarioValidationError
from gecko.data.streams.orders import generate_client_orders
from gecko.data.streams.participation import generate_participation


@dataclass(frozen=True)
class StreamBundle:
    config: GECKOConfig
    scenario: ScenarioSpec
    partition: PartitionResult
    shards: Dict[int, Dict[int, ClientTaskShard]]
    evaluation_shards: Dict[int, Dict[int, CentralEvaluationShard]]
    orders: OrderPlan
    participation: ParticipationPlan
    stream_id: str
    stream_hash: str
    timings: Dict[str, float]


def spatial_partition_slug(config: GECKOConfig) -> str:
    """Compatibility wrapper for current allocation-directory names."""

    return allocation_slug(config)


def stream_relative_path(config: GECKOConfig, *, legacy: bool = False) -> Path:
    """Locate new conditions with paper labels, retaining historical path lookup."""

    scenario = config.scenario
    allocation = (
        f"spatial-{spatial_profile_label(config)}"
        if legacy else spatial_partition_slug(config)
    )
    order = (
        f"order-{config.order.profile}"
        if legacy else f"task-order-{task_order_label(config.order.profile)}"
    )
    return (
        Path("uefa-v1") / scenario.dataset / scenario.problem.lower()
        / scenario.incremental_setting / f"seed-{config.seed}" / allocation / order
    )


def spatial_profile_label(config: GECKOConfig) -> str:
    """Describe the active spatial control without legacy easy/mild/hard leakage."""

    alpha = config.partition.dirichlet_alpha
    if alpha is None:
        return config.partition.spatial_profile
    alpha_token = format(float(alpha), ".12g").replace("-", "m").replace(".", "p")
    return f"dirichlet-{alpha_token}"


def stream_identity(config: GECKOConfig) -> tuple[str, str]:
    config_payload = config.to_dict()
    scenario_payload = config_payload["scenario"]
    scenario_payload.pop("save_path", None)
    if scenario_payload.get("class_task_policy") == "benchmark_defined":
        scenario_payload.pop("class_task_policy", None)
    partition_payload = config_payload["partition"]
    if partition_payload.get("dirichlet_alpha") is None:
        partition_payload.pop("dirichlet_alpha", None)
    order_payload = config_payload["order"]
    if not order_payload.get("allow_full_permutation_below_four_tasks", False):
        order_payload.pop("allow_full_permutation_below_four_tasks", None)
    payload = json.dumps(
        {
            "benchmark_name": config.benchmark_name,
            "benchmark_schema_version": config.benchmark_schema_version,
            "seed": config.seed,
            "scenario": scenario_payload,
            "partition": partition_payload,
            "order": order_payload,
            "participation_fraction": config.training.participation_fraction,
            "rounds_per_stage": config.training.rounds_per_stage,
            "lp_train_negative_ratio": config.training.lp_train_negative_ratio,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    scenario = config.scenario
    stream_id = (
        f"uefa-v1-{scenario.dataset}-{scenario.problem.lower()}-"
        f"{scenario.incremental_setting}-seed{config.seed}-"
        f"{spatial_profile_label(config)}-{config.order.profile}-{digest[:12]}"
    )
    return stream_id, digest


def _local_queries(
    spec: ScenarioSpec,
    query_ids: torch.Tensor,
    client_id: int,
    partition: PartitionResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    graph = partition.client_graphs[client_id]
    if spec.problem_type == "NC":
        owners = partition.node_owner[query_ids]
        internal_ids = query_ids[owners == client_id]
        local = torch.tensor(
            [graph.global_to_local[int(query)] for query in internal_ids.tolist()],
            dtype=torch.long,
        )
        return local, internal_ids
    assert spec.query_endpoints is not None
    endpoints = spec.query_endpoints[query_ids]
    internal = (
        (partition.node_owner[endpoints[:, 0]] == client_id)
        & (partition.node_owner[endpoints[:, 1]] == client_id)
    )
    internal_ids = query_ids[internal]
    local_pairs = torch.tensor(
        [
            [graph.global_to_local[int(source)], graph.global_to_local[int(target)]]
            for source, target in endpoints[internal].tolist()
        ],
        dtype=torch.long,
    )
    if local_pairs.numel() == 0:
        local_pairs = torch.empty((0, 2), dtype=torch.long)
    return local_pairs, internal_ids


def _build_client_task_views(
    spec: ScenarioSpec,
    partition: PartitionResult,
    config: GECKOConfig,
) -> tuple[
    Dict[int, Dict[int, ClientTaskShard]],
    Dict[int, Dict[int, CentralEvaluationShard]],
]:
    shards: Dict[int, Dict[int, ClientTaskShard]] = {}
    evaluation_shards: Dict[int, Dict[int, CentralEvaluationShard]] = {}
    minimums = {
        "train": config.partition.minimum_train_queries_per_client_task,
        "val": config.partition.minimum_validation_queries_per_client_task,
        "test": config.partition.minimum_test_queries_per_client_task,
    }
    failures: list[str] = []
    task_contexts = spec.metadata.get("task_context_edge_index", {})
    for client_id in range(config.partition.num_clients):
        shards[client_id] = {}
        evaluation_shards[client_id] = {}
        for task_id in range(spec.num_tasks):
            local_context = None
            if task_id in task_contexts:
                global_context = task_contexts[task_id]
                graph = partition.client_graphs[client_id]
                source_local = torch.tensor(
                    [graph.global_to_local.get(int(value), -1) for value in global_context[0]],
                    dtype=torch.long,
                )
                target_local = torch.tensor(
                    [graph.global_to_local.get(int(value), -1) for value in global_context[1]],
                    dtype=torch.long,
                )
                internal_context = (source_local >= 0) & (target_local >= 0)
                local_context = torch.stack(
                    (source_local[internal_context], target_local[internal_context])
                )
            split_queries = {}
            split_central_ids = {}
            for split, query_ids in spec.query_ids_by_task_split[task_id].items():
                local, central_ids = _local_queries(
                    spec, query_ids, client_id, partition
                )
                split_queries[split] = local
                split_central_ids[split] = central_ids
                support = int(central_ids.shape[0])
                support_name = "query_support"
                if spec.problem_type == "LP":
                    support = int((spec.labels[central_ids] == 1).sum())
                    support_name = "positive_support"
                if support < minimums[split]:
                    failures.append(
                        f"client={client_id} task={task_id} split={split} "
                        f"{support_name}={support} required={minimums[split]}"
                    )
            shards[client_id][task_id] = ClientTaskShard(
                client_id=client_id,
                global_task_id=task_id,
                train_queries=split_queries["train"],
                train_labels=spec.labels[split_central_ids["train"]].detach().cpu().clone(),
                task_class_mask=(
                    None if spec.task_masks is None else spec.task_masks[task_id].clone()
                ),
                context_edge_index=(
                    None if local_context is None else local_context.clone()
                ),
            )
            evaluation_shards[client_id][task_id] = CentralEvaluationShard(
                client_id=client_id,
                global_task_id=task_id,
                train_query_ids=split_central_ids["train"],
                validation_queries=split_queries["val"],
                validation_query_ids=split_central_ids["val"],
                test_queries=split_queries["test"],
                test_query_ids=split_central_ids["test"],
                context_edge_index=(
                    None if local_context is None else local_context.clone()
                ),
            )
    if failures and not config.partition.allow_infeasible:
        raise PartitionInfeasibleError(
            "Client query shards violate dense-support constraints.", failures
        )
    return shards, evaluation_shards


def build_client_task_shards(
    spec: ScenarioSpec,
    partition: PartitionResult,
    config: GECKOConfig,
) -> Dict[int, Dict[int, ClientTaskShard]]:
    """Build label-safe train shards without exposing central evaluation data."""

    shards, _ = _build_client_task_views(spec, partition, config)
    return shards


@dataclass(frozen=True)
class StreamBuilder:
    config: GECKOConfig

    def build(self, scenario: ScenarioSpec) -> StreamBundle:
        build_start = time.perf_counter()
        self.config.validate()
        scenario.validate()
        expected = self.config.scenario
        if (
            scenario.problem_type != expected.problem.upper()
            or scenario.incremental_type != expected.incremental_setting.lower()
            or scenario.num_tasks != expected.num_tasks
        ):
            raise ScenarioValidationError(
                "ScenarioSpec does not match the validated UEFA configuration."
            )
        scenario_design_audit = audit_scenario_design(self.config, scenario)
        partition_start = time.perf_counter()
        partition_first_pending = scenario.metadata.get(
            "partition_first_pending", False
        )
        maximum_proposals = (
            self.config.partition.maximum_lp_partition_proposals
            if partition_first_pending
            else 1
        )
        rejected_proposals: list[str] = []
        base_scenario = scenario
        partition: PartitionResult | None = None
        for proposal_index in range(maximum_proposals):
            proposal_seed = self.config.seed + proposal_index * 1_000_003
            candidate_partition = PersistentSubgraphPartitioner(
                self.config.partition, proposal_seed
            ).partition(base_scenario)
            if not partition_first_pending:
                partition = candidate_partition
                break
            try:
                candidate_scenario = finalize_partition_first_lp(
                    base_scenario,
                    candidate_partition.node_owner,
                    num_clients=self.config.partition.num_clients,
                    seed=self.config.seed,
                    minimum_support=(
                        self.config.partition.minimum_train_queries_per_client_task,
                        self.config.partition.minimum_validation_queries_per_client_task,
                        self.config.partition.minimum_test_queries_per_client_task,
                    ),
                    training_negative_ratio=self.config.training.lp_train_negative_ratio,
                    evaluation_negatives_per_client_task=(
                        self.config.scenario.lp_evaluation_negatives_per_client_task
                    ),
                )
            except ValueError as error:
                if not str(error).startswith(
                    "Partition-first LP support is infeasible:"
                ):
                    raise
                rejected_proposals.append(str(error))
                continue
            scenario = replace(
                candidate_scenario,
                metadata={
                    **candidate_scenario.metadata,
                    "partition_proposal_index": proposal_index,
                    "partition_proposal_seed": proposal_seed,
                    "partition_rejected_proposal_count": len(rejected_proposals),
                    "partition_proposal_selection": (
                        "topology_only_then_pre_split_positive_support_audit"
                    ),
                },
            )
            partition = rematerialize_partition_context(
                scenario, candidate_partition, self.config.partition
            )
            partition = replace(
                partition,
                diagnostics={
                    **partition.diagnostics,
                    "lp_partition_proposal_index": proposal_index,
                    "lp_partition_proposal_seed": proposal_seed,
                    "lp_partition_rejected_proposal_count": len(
                        rejected_proposals
                    ),
                    "lp_partition_proposal_selection": (
                        "topology_only_then_pre_split_positive_support_audit"
                    ),
                },
            )
            scenario_design_audit = audit_scenario_design(self.config, scenario)
            break
        if partition is None:
            raise PartitionInfeasibleError(
                "No topology-only LP partition proposal satisfies pre-split "
                "positive support.",
                tuple(rejected_proposals),
            )
        partition_seconds = time.perf_counter() - partition_start
        shard_start = time.perf_counter()
        shards, evaluation_shards = _build_client_task_views(
            scenario, partition, self.config
        )
        design_audit = audit_partition_design(
            self.config,
            scenario,
            partition,
            evaluation_shards,
            scenario_design_audit,
        )
        partition = replace(
            partition,
            diagnostics={
                **partition.diagnostics,
                "dataset_design_audit": design_audit,
            },
        )
        shard_seconds = time.perf_counter() - shard_start
        order_start = time.perf_counter()
        orders = generate_client_orders(
            self.config.partition.num_clients,
            scenario.num_tasks,
            self.config.order.profile,
            self.config.seed,
        )
        order_seconds = time.perf_counter() - order_start
        participation_start = time.perf_counter()
        participation = generate_participation(
            num_clients=self.config.partition.num_clients,
            num_stages=scenario.num_tasks,
            rounds_per_stage=self.config.training.rounds_per_stage,
            fraction=self.config.training.participation_fraction,
            seed=self.config.seed,
        )
        participation_seconds = time.perf_counter() - participation_start
        participating_tasks = {
            stage: {
                round_id: dict(
                    sorted(
                        Counter(
                            orders.global_task(client_id, stage)
                            for client_id in clients
                        ).items()
                    )
                )
                for round_id, clients in rounds.items()
            }
            for stage, rounds in participation.trace.items()
        }
        orders = replace(
            orders,
            diagnostics={
                **orders.diagnostics,
                "participating_client_task_distribution_by_round": participating_tasks,
            },
        )
        stream_id, stream_hash = stream_identity(self.config)
        return StreamBundle(
            config=self.config,
            scenario=scenario.detached_clone(),
            partition=partition,
            shards=shards,
            evaluation_shards=evaluation_shards,
            orders=orders,
            participation=participation,
            stream_id=stream_id,
            stream_hash=stream_hash,
            timings={
                "partition_generation_seconds": partition_seconds,
                "query_shard_construction_seconds": shard_seconds,
                "order_generation_seconds": order_seconds,
                "participation_generation_seconds": participation_seconds,
                "stream_build_total_seconds": time.perf_counter() - build_start,
            },
        )
