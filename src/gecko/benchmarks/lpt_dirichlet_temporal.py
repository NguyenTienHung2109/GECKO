from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Sequence
import torch
from gecko.compat.lpt_protocol import v1_aggregate
from gecko.compat.lpt_protocol import LPT_V1, validate_lpt_protocol, v1_diagnostics
from gecko.config import GECKOConfig
from gecko.data.partitioning.base import EmptyCSRRows
from gecko.data.partitioning.base import WeightedLogicalTopology
from gecko.data.partitioning.base import build_large_bidirected_simple_topology
from gecko.data.partitioning.base import build_weighted_logical_topology
from gecko.data.partitioning.materialize import materialize_direct_partition
from gecko.data.partitioning.base import order_hash
from gecko.data.partitioning.base import participation_hash
from gecko.data.partitioning.base import query_shard_hash
from gecko.data.transforms import scenario_with_selected_training_queries
from gecko.data.partitioning.base import strict_local_graph_hash
from gecko.data.partitioning.dirichlet.quota import ExactDirichletQuotaGenerator
from gecko.data.partitioning.dirichlet.quota import ExactDirichletQuotaInfeasibleError
from gecko.data.transforms import apply_balanced_class_task_plan
from gecko.data.transforms import BALANCED_CLASS_TASK_POLICY_VERSION
from gecko.data.transforms import BalancedClassTaskPlan
from gecko.data.transforms import build_balanced_class_task_plan
from gecko.data.dataloader import load_scenario_spec
from gecko.data.streams import audit_stream
from gecko.data.streams import save_stream
from gecko.data.streams.builder import StreamBundle
from gecko.data.streams.builder import _build_client_task_views
from gecko.data.streams.builder import stream_identity
from gecko.data.streams.orders import generate_client_orders
from gecko.data.streams.participation import generate_participation
from gecko.types import PartitionResult
from gecko.types import ScenarioSpec
from gecko.data.partitioning.community.link import LCSoftCommunityConfig
from gecko.data.partitioning.community.link import LCSoftCommunityError
from gecko.data.partitioning.community.link import LCSoftCommunityResult
from gecko.data.partitioning.community.link import build_lc_soft_community_partition
from gecko.data.partitioning.community.node import NCSoftCommunityConfig
from gecko.data.partitioning.community.node import NCSoftCommunityResult
from gecko.data.partitioning.community.node import build_nc_soft_community_owner
from gecko.data.partitioning.community.node import build_nc_soft_community_from_spec
from gecko.data.partitioning.community.common import MicroCommunityResult
from gecko.data.partitioning.community.common import SoftMicroCommunityConfig
from gecko.data.partitioning.community.common import build_large_metis_microcommunities

LPT_DIRICHLET_TEMPORAL_VERSION = "lpt_exact_dirichlet_temporal_v4"


LP_QUOTA_DIAGNOSTICS_VERSION = "lp_selected_positive_quota_diagnostics_v1"


DIRICHLET_MANIPULATION_AUDIT_VERSION = "dirichlet_realized_skew_gate_v1"


DEFAULT_ALPHAS = (0.1, 1.0, 10.0, 100.0)


DEFAULT_TEMPORAL_PROFILES = ("synchronized", "hard")


LARGE_GRAPH_EDGE_THRESHOLD = 5_000_000


LP_SELECTED_TRAIN_POSITIVE_FRACTION = 0.001


@dataclass(frozen=True)
class PreparedScenario:
    """A scenario plus the optional train-only LPT construction record."""

    config: GECKOConfig
    scenario: ScenarioSpec
    lpt_plan: BalancedClassTaskPlan | None


def _plan_payload(plan: BalancedClassTaskPlan | None) -> dict[str, Any]:
    if plan is None:
        return {
            "applicable": False,
            "policy_version": None,
            "reason": "Domain-IL preserves the benchmark-defined immutable tasks.",
        }
    return {
        "applicable": True,
        "policy_version": plan.policy_version,
        "class_groups": [list(group) for group in plan.class_groups],
        "excluded_classes": list(plan.excluded_classes),
        "train_class_counts": list(plan.train_class_counts),
        "task_train_counts": list(plan.task_train_counts),
        "classes_per_task": plan.classes_per_task,
    }


def _js_divergence_from_counts(
    first: torch.Tensor, second: torch.Tensor
) -> float:
    """Return Jensen--Shannon divergence for two non-negative count rows."""

    left = first.detach().cpu().double().reshape(-1)
    right = second.detach().cpu().double().reshape(-1)
    if left.numel() != right.numel() or left.numel() == 0:
        raise ValueError("Dirichlet diagnostic count rows must have equal width.")
    if bool((left < 0).any()) or bool((right < 0).any()):
        raise ValueError("Dirichlet diagnostic counts must be non-negative.")
    left = left / left.sum().clamp_min(1)
    right = right / right.sum().clamp_min(1)
    midpoint = 0.5 * (left + right)

    def _kl(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        valid = values > 0
        return (
            values[valid] * (values[valid] / reference[valid]).log()
        ).sum()

    return float(0.5 * (_kl(left, midpoint) + _kl(right, midpoint)))


def _lp_quota_skew_diagnostics(
    quota_rows: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Summarize realized LP positive allocation without evaluation candidates."""

    quota = torch.as_tensor(quota_rows, dtype=torch.double)
    if quota.ndim != 2 or not quota.shape[0] or not quota.shape[1]:
        raise ValueError("LP quota diagnostics require a non-empty client-by-task matrix.")
    if bool((quota < 0).any()) or bool((quota.sum(dim=0) <= 0).any()):
        raise ValueError("Every LP quota task must have positive non-negative support.")

    global_semantics = quota.sum(dim=0)
    semantic_js = [
        _js_divergence_from_counts(row, global_semantics) for row in quota
    ]
    return {
        "lp_quota_diagnostics_version": LP_QUOTA_DIAGNOSTICS_VERSION,
        "dirichlet_realized_semantic_js_divergence_per_client": semantic_js,
        "dirichlet_realized_semantic_js_divergence_mean": (
            sum(semantic_js) / len(semantic_js)
        ),
        **_quota_group_allocation_diagnostics(quota),
    }


def _quota_group_allocation_diagnostics(
    quota_rows: Sequence[Sequence[int]] | torch.Tensor,
) -> dict[str, Any]:
    """Measure realized client concentration for each allocated semantic group."""

    quota = torch.as_tensor(quota_rows, dtype=torch.double)
    if quota.ndim != 2 or not quota.shape[0] or not quota.shape[1]:
        raise ValueError("Dirichlet allocation diagnostics require a non-empty matrix.")
    if bool((quota < 0).any()) or bool((quota.sum(dim=0) <= 0).any()):
        raise ValueError("Every Dirichlet semantic group must have positive support.")
    uniform_clients = torch.ones(quota.shape[0], dtype=torch.double)
    allocation_js = [
        _js_divergence_from_counts(quota[:, group], uniform_clients)
        for group in range(quota.shape[1])
    ]
    probabilities = quota / quota.sum(dim=0, keepdim=True)
    entropy_denominator = math.log(int(quota.shape[0]))
    normalized_entropy = [
        float(
            -(
                probabilities[:, group][probabilities[:, group] > 0]
                * probabilities[:, group][probabilities[:, group] > 0].log()
            ).sum()
            / entropy_denominator
        )
        if quota.shape[0] > 1
        else 1.0
        for group in range(quota.shape[1])
    ]
    maximum_share = probabilities.max(dim=0).values.tolist()
    return {
        "dirichlet_group_allocation_js_to_uniform_per_group": allocation_js,
        "dirichlet_group_allocation_js_to_uniform_mean": (
            sum(allocation_js) / len(allocation_js)
        ),
        "dirichlet_group_allocation_normalized_entropy_per_group": normalized_entropy,
        "dirichlet_group_allocation_normalized_entropy_mean": (
            sum(normalized_entropy) / len(normalized_entropy)
        ),
        "dirichlet_group_allocation_max_client_share_per_group": maximum_share,
        "dirichlet_group_allocation_max_client_share_mean": (
            sum(maximum_share) / len(maximum_share)
        ),
    }


def _merge_lp_final_diagnostics(
    construction: dict[str, Any], rematerialized: dict[str, Any]
) -> dict[str, Any]:
    """Keep construction metadata while final-query diagnostics win conflicts."""

    return {**construction, **rematerialized}


def _capacity_aware_positive_quota(
    capacities: Sequence[int] | torch.Tensor,
    *,
    total: int,
    minimum: int,
) -> torch.Tensor:
    """Allocate an exact held-out positive budget under finite capacities.

    Every client first receives ``minimum`` positives.  The remaining budget
    is assigned to the currently smallest quota, subject to finite capacity;
    deterministic client IDs break ties.  This is balanced-first water filling:
    capacity constrains evaluation support but does not weight it.  No quota
    can exceed the capacity left after reserving validation and train support.
    """

    available = torch.as_tensor(capacities, dtype=torch.long).reshape(-1)
    if available.numel() == 0 or bool((available < 0).any()):
        raise ValueError("LP held-out capacities must be non-negative and non-empty.")
    if total < 0 or minimum < 0:
        raise ValueError("LP held-out total and minimum must be non-negative.")
    quota = torch.full_like(available, int(minimum))
    if bool((quota > available).any()):
        raise ValueError("LP held-out minimum exceeds a client capacity.")
    if total < int(quota.sum()) or total > int(available.sum()):
        raise ValueError(
            "LP held-out positive budget is infeasible for the client capacities."
        )
    if total == int(quota.sum()):
        return quota
    while int(quota.sum()) < total:
        eligible = torch.nonzero(quota < available, as_tuple=True)[0]
        client = min(
            eligible.tolist(),
            key=lambda value: (
                int(quota[value]),
                value,
            ),
        )
        quota[client] += 1
    return quota


def prepare_lpt_scenario(config: GECKOConfig) -> PreparedScenario:
    """Load one scenario and apply train-only LPT grouping when meaningful."""

    problem = config.scenario.problem.upper()
    incremental = config.scenario.incremental_setting.lower()
    if problem not in {"NC", "LC", "LP"}:
        raise ValueError(
            "Exact semantic Dirichlet ownership is defined only for NC/LC/LP."
        )
    scenario = load_scenario_spec(config)
    if problem == "LP" and incremental != "domain":
        raise ValueError("Exact LP Dirichlet construction currently supports Domain-IL only.")
    if incremental == "domain":
        return PreparedScenario(config=config, scenario=scenario, lpt_plan=None)
    if incremental not in {"task", "class"}:
        raise ValueError(f"Unsupported incremental setting: {incremental}")
    plan = build_balanced_class_task_plan(
        scenario, target_num_tasks=config.scenario.num_tasks
    )
    transformed = apply_balanced_class_task_plan(scenario, plan)
    transformed_config = replace(
        config,
        scenario=replace(
            config.scenario,
            num_tasks=transformed.num_tasks,
            class_task_policy=BALANCED_CLASS_TASK_POLICY_VERSION,
        ),
    )
    transformed_config.validate()
    return PreparedScenario(
        config=transformed_config,
        scenario=transformed,
        lpt_plan=plan,
    )


def _partition_from_result(
    prepared: PreparedScenario,
    result: NCSoftCommunityResult | LCSoftCommunityResult,
    *, protocol_version: str = LPT_DIRICHLET_TEMPORAL_VERSION,
) -> tuple[ScenarioSpec, PartitionResult]:
    scenario = prepared.scenario
    if isinstance(result, LCSoftCommunityResult):
        scenario = scenario_with_selected_training_queries(scenario, result)
    diagnostics = {
        **result.diagnostics,
        "protocol_version": protocol_version,
        "class_task_policy": prepared.config.scenario.class_task_policy,
        "lpt_plan": _plan_payload(prepared.lpt_plan),
        "hconn_target": None,
        "cload_target": None,
        "hconn_cload_role": "reported_diagnostics_only_not_a_constraint",
    }
    partition = materialize_direct_partition(
        scenario,
        result.owner,
        prepared.config,
        scientific_diagnostics=diagnostics,
    )
    partition = replace(
        partition,
        diagnostics={
            **partition.diagnostics,
            "partition_mode": protocol_version,
            "direct_stream_version": protocol_version,
            "grid_version": protocol_version,
            "spatial_control": "exact_dirichlet_alpha",
            "legacy_easy_mild_hard_spatial_profile_ignored": True,
            "community_generation_used": True,
            "community_generation_role": (
                "topology_only_scaffold_for_exact_dirichlet_ownership"
            ),
        },
    )
    if protocol_version == LPT_V1:
        partition = replace(partition, diagnostics=v1_diagnostics(
            partition.diagnostics, problem=scenario.problem_type, setting=scenario.incremental_type))
    return scenario, partition



def construct_exact_dirichlet_partition(
    prepared: PreparedScenario,
    *,
    alpha: float,
    protocol_version: str = LPT_DIRICHLET_TEMPORAL_VERSION,
    beam_width: int = 16,
    branch_factor: int = 4,
    lc_maximum_search_expansions: int = 200_000,
    precomputed_topology: WeightedLogicalTopology | None = None,
    precomputed_micro_result: MicroCommunityResult | None = None,
) -> tuple[ScenarioSpec, PartitionResult, dict[str, Any]]:
    """Construct one ownership cell without Hconn/Cload treatment matching."""
    from gecko.data.partitioning.dirichlet.lp import _construct_lp_exact_dirichlet_partition

    validate_lpt_protocol(protocol_version, prepared.scenario.problem_type)
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    config = replace(
        prepared.config,
        partition=replace(prepared.config.partition, dirichlet_alpha=float(alpha)),
    )
    prepared = replace(prepared, config=config)
    if prepared.scenario.problem_type == "LP":
        scenario, partition, result = _construct_lp_exact_dirichlet_partition(
            prepared,
            alpha=float(alpha),
            beam_width=beam_width,
            branch_factor=branch_factor,
            maximum_search_expansions=lc_maximum_search_expansions,
        )
        return scenario, partition, {
            "alpha_dirichlet": float(alpha),
            "owner_hash": result.diagnostics.get("ownership_hash"),
            "quota_hash": result.diagnostics.get("quota_hash"),
            "selected_query_count": result.selected_positive_count,
            "construction_status": result.status,
            "dirichlet_group_allocation_js_to_uniform_mean": (
                result.diagnostics[
                    "dirichlet_group_allocation_js_to_uniform_mean"
                ]
            ),
            "dirichlet_group_allocation_normalized_entropy_mean": (
                result.diagnostics[
                    "dirichlet_group_allocation_normalized_entropy_mean"
                ]
            ),
            "dirichlet_group_allocation_max_client_share_mean": (
                result.diagnostics[
                    "dirichlet_group_allocation_max_client_share_mean"
                ]
            ),
        }
    common = {
        "num_clients": config.partition.num_clients,
        "alpha_dirichlet": float(alpha),
        "seed": config.seed,
        "client_size_tolerance": config.partition.client_size_tolerance,
        "minimum_train_queries_per_client_task": (
            config.partition.minimum_train_queries_per_client_task
        ),
        "minimum_validation_queries_per_client_task": (
            config.partition.minimum_validation_queries_per_client_task
        ),
        "minimum_test_queries_per_client_task": (
            config.partition.minimum_test_queries_per_client_task
        ),
    }
    if prepared.scenario.problem_type == "NC":
        compact_large_graph = (
            precomputed_topology is not None
            and isinstance(precomputed_topology.incident_edge_ids, EmptyCSRRows)
        )
        nc_config = NCSoftCommunityConfig(
            **common,
            quota_maximum_retries=4096,
            reserve_minimum_train_support=True,
            maximum_refinement_swaps=0 if compact_large_graph else 16,
        )
        if prepared.scenario.incremental_type == "domain":
            train_ids = torch.cat(
                [
                    prepared.scenario.query_ids_by_task_split[task]["train"]
                    for task in range(prepared.scenario.num_tasks)
                ]
            ).detach().cpu().long()
            task_labels = prepared.scenario.query_task_ids[train_ids].long()
            if bool((task_labels < 0).any()):
                raise ValueError("A domain training query has no immutable task ID.")
            counts = torch.bincount(
                task_labels, minlength=prepared.scenario.num_tasks
            )
            quota = ExactDirichletQuotaGenerator(maximum_retries=4096).generate(
                counts.tolist(),
                config.partition.num_clients,
                float(alpha),
                config.seed,
                None,
                tuple((task,) for task in range(prepared.scenario.num_tasks)),
                config.partition.minimum_train_queries_per_client_task,
                True,
            )
            topology = precomputed_topology or build_weighted_logical_topology(
                prepared.scenario.edge_index,
                num_nodes=int(prepared.scenario.node_features.shape[0]),
                directed=False,
                representation_id="nc_domain_task_dirichlet_topology_v1",
            )
            result = build_nc_soft_community_owner(
                topology=topology,
                train_ids=train_ids,
                train_labels=task_labels,
                quota=quota,
                config=nc_config,
                micro_result=precomputed_micro_result,
                heldout_query_ids_by_task_split={
                    task: {
                        "val": prepared.scenario.query_ids_by_task_split[task]["val"],
                        "test": prepared.scenario.query_ids_by_task_split[task]["test"],
                    }
                    for task in range(prepared.scenario.num_tasks)
                },
            )
            result = replace(
                result,
                diagnostics={
                    **result.diagnostics,
                    **_quota_group_allocation_diagnostics(
                        result.quota.integer_quota
                    ),
                    "dirichlet_semantic_unit": "immutable_domain_task_id",
                    "target_labels_accessed_for_ownership": False,
                    "large_graph_compact_topology": compact_large_graph,
                    "edge_id_refinement_disabled": compact_large_graph,
                },
            )
        else:
            result = build_nc_soft_community_from_spec(
                prepared.scenario,
                nc_config,
            )
            result = replace(
                result,
                diagnostics={
                    **result.diagnostics,
                    **_quota_group_allocation_diagnostics(
                        result.quota.integer_quota
                    ),
                    "dirichlet_semantic_unit": "train_class_label",
                },
            )
    else:
        result = build_lc_soft_community_partition(
            prepared.scenario,
            LCSoftCommunityConfig(
                **common,
                supervised_budget_fraction=0.05,
                beam_width=beam_width,
                branch_factor=branch_factor,
                maximum_search_expansions=lc_maximum_search_expansions,
            ),
        )
    scenario, partition = _partition_from_result(prepared, result, protocol_version=protocol_version)
    return scenario, partition, {
        "alpha_dirichlet": float(alpha),
        "owner_hash": result.diagnostics.get(
            "owner_hash", result.diagnostics.get("ownership_hash")
        ),
        "quota_hash": result.diagnostics.get("quota_hash"),
        "selected_query_count": (
            int(result.selected_query_ids.numel())
            if isinstance(result, LCSoftCommunityResult)
            else None
        ),
        "construction_status": getattr(result, "status", "success"),
        "dirichlet_group_allocation_js_to_uniform_mean": (
            result.diagnostics.get(
                "dirichlet_group_allocation_js_to_uniform_mean"
            )
        ),
        "dirichlet_group_allocation_normalized_entropy_mean": (
            result.diagnostics.get(
                "dirichlet_group_allocation_normalized_entropy_mean"
            )
        ),
        "dirichlet_group_allocation_max_client_share_mean": (
            result.diagnostics.get(
                "dirichlet_group_allocation_max_client_share_mean"
            )
        ),
    }



def build_temporal_variants(
    config: GECKOConfig,
    scenario: ScenarioSpec,
    partition: PartitionResult,
    *,
    profiles: Sequence[str] = DEFAULT_TEMPORAL_PROFILES,
) -> tuple[dict[str, StreamBundle], list[dict[str, Any]]]:
    """Attach paired order plans to an otherwise identical frozen stream."""

    requested = tuple(dict.fromkeys(str(value) for value in profiles))
    if not requested or any(value not in {"synchronized", "hard"} for value in requested):
        raise ValueError("Temporal profiles must be synchronized and/or hard.")
    synchronized = replace(
        config,
        order=replace(
            config.order,
            profile="synchronized",
            allow_full_permutation_below_four_tasks=False,
        ),
    )
    synchronized.validate()
    shards, evaluations = _build_client_task_views(
        scenario, partition, synchronized
    )
    participation = generate_participation(
        num_clients=config.partition.num_clients,
        num_stages=scenario.num_tasks,
        rounds_per_stage=config.training.rounds_per_stage,
        fraction=config.training.participation_fraction,
        seed=config.seed,
    )
    graph_digest = strict_local_graph_hash(partition)
    shard_digest = query_shard_hash(shards, evaluations)
    participation_digest = participation_hash(participation)
    streams: dict[str, StreamBundle] = {}
    records: list[dict[str, Any]] = []
    for profile in requested:
        # LP-Domain has exactly two tasks.  Its established fail-closed order
        # name is binary_mismatch; semantically it is the requested hard
        # temporal contrast (the only non-synchronized two-task permutation).
        effective_profile = (
            "binary_mismatch"
            if scenario.problem_type == "LP" and profile == "hard" and scenario.num_tasks == 2
            else profile
        )
        allow_short = effective_profile == "hard" and scenario.num_tasks < 4
        cell_config = replace(
            config,
            order=replace(
                config.order,
                profile=effective_profile,
                allow_full_permutation_below_four_tasks=allow_short,
            ),
        )
        cell_config.validate()
        orders = generate_client_orders(
            cell_config.partition.num_clients,
            scenario.num_tasks,
            effective_profile,
            cell_config.seed,
        )
        stream_id, stream_hash = stream_identity(cell_config)
        streams[profile] = StreamBundle(
            config=cell_config,
            # ScenarioSpec is immutable. Sharing it between paired order cells
            # avoids two multi-GB clones on OGBN-Proteins while preserving the
            # exact same frozen scientific object for both variants.
            scenario=scenario,
            partition=partition,
            shards=shards,
            evaluation_shards=evaluations,
            orders=orders,
            participation=participation,
            stream_id=stream_id,
            stream_hash=stream_hash,
            timings={},
        )
        records.append(
            {
                "order_profile": effective_profile,
                "stream_key": profile,
                "requested_temporal_profile": profile,
                "effective_temporal_profile": effective_profile,
                "order_hash": order_hash(orders),
                "short_hard_order_override": allow_short,
                "strict_local_graph_hash": graph_digest,
                "query_shard_hash": shard_digest,
                "participation_hash": participation_digest,
                "stream_id": stream_id,
                "stream_hash": stream_hash,
            }
        )
    return streams, records


def build_dirichlet_manipulation_audit(
    records: Sequence[dict[str, Any]],
    failed_cells: Sequence[dict[str, Any]],
    *,
    alphas: Sequence[float],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Fail closed unless the realized aggregate skew follows the alpha grid."""

    alpha_values = tuple(sorted({float(value) for value in alphas}))
    seed_values = tuple(sorted({int(value) for value in seeds}))
    reasons: list[str] = []
    if len(alpha_values) < 2:
        reasons.append("at_least_two_alpha_levels_are_required")
    if failed_cells:
        reasons.append("one_or_more_construction_cells_failed")

    values_by_cell: dict[tuple[int, float], list[float]] = {}
    nonfinite_cells: list[dict[str, Any]] = []
    for record in records:
        seed = int(record["seed"])
        alpha = float(record["alpha_dirichlet"])
        raw = record.get("dirichlet_group_allocation_js_to_uniform_mean")
        if raw is None or not math.isfinite(float(raw)):
            nonfinite_cells.append({"seed": seed, "alpha_dirichlet": alpha})
            continue
        values_by_cell.setdefault((seed, alpha), []).append(float(raw))
    if nonfinite_cells:
        reasons.append("missing_or_nonfinite_realized_skew_diagnostic")

    missing_cells = [
        {"seed": seed, "alpha_dirichlet": alpha}
        for seed in seed_values
        for alpha in alpha_values
        if (seed, alpha) not in values_by_cell
    ]
    if missing_cells:
        reasons.append("requested_seed_alpha_cell_is_missing")

    temporal_mismatch_cells: list[dict[str, Any]] = []
    realized_by_seed: dict[str, dict[str, float]] = {}
    for (seed, alpha), values in sorted(values_by_cell.items()):
        if max(values) - min(values) > 1e-12:
            temporal_mismatch_cells.append(
                {
                    "seed": seed,
                    "alpha_dirichlet": alpha,
                    "values": values,
                }
            )
        realized_by_seed.setdefault(str(seed), {})[f"{alpha:.17g}"] = (
            sum(values) / len(values)
        )
    if temporal_mismatch_cells:
        reasons.append("paired_temporal_streams_disagree_on_spatial_skew")

    aggregate_by_alpha: dict[str, float] = {}
    for alpha in alpha_values:
        values = [
            realized_by_seed[str(seed)][f"{alpha:.17g}"]
            for seed in seed_values
            if f"{alpha:.17g}" in realized_by_seed.get(str(seed), {})
        ]
        if values:
            aggregate_by_alpha[f"{alpha:.17g}"] = sum(values) / len(values)

    aggregate_strictly_decreasing = len(aggregate_by_alpha) == len(alpha_values) and all(
        aggregate_by_alpha[f"{left:.17g}"]
        > aggregate_by_alpha[f"{right:.17g}"] + 1e-12
        for left, right in zip(alpha_values, alpha_values[1:])
    )
    if len(alpha_values) >= 2 and not aggregate_strictly_decreasing:
        reasons.append("aggregate_realized_skew_is_not_strictly_decreasing")

    per_seed_strictly_decreasing = {
        str(seed): all(
            realized_by_seed.get(str(seed), {}).get(f"{left:.17g}", float("-inf"))
            > realized_by_seed.get(str(seed), {}).get(f"{right:.17g}", float("inf"))
            + 1e-12
            for left, right in zip(alpha_values, alpha_values[1:])
        )
        for seed in seed_values
    }
    eligible = not reasons
    return {
        "version": DIRICHLET_MANIPULATION_AUDIT_VERSION,
        "status": "pass" if eligible else "fail",
        "benchmark_eligible": eligible,
        "metric": "dirichlet_group_allocation_js_to_uniform_mean",
        "expected_direction": "strictly_decreasing_as_alpha_increases",
        "gate_scope": "mean_across_requested_seeds",
        "alphas_ascending": list(alpha_values),
        "seeds": list(seed_values),
        "realized_by_seed": realized_by_seed,
        "aggregate_by_alpha": aggregate_by_alpha,
        "aggregate_strictly_decreasing": aggregate_strictly_decreasing,
        "per_seed_strictly_decreasing_diagnostic": per_seed_strictly_decreasing,
        "nonfinite_cells": nonfinite_cells,
        "missing_cells": missing_cells,
        "temporal_mismatch_cells": temporal_mismatch_cells,
        "failure_reasons": reasons,
    }


def prepare_stream_grid(
    config_path: str | Path,
    *,
    store_root: str | Path,
    protocol_version: str = LPT_DIRICHLET_TEMPORAL_VERSION,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
    temporal_profiles: Sequence[str] = DEFAULT_TEMPORAL_PROFILES,
    seeds: Iterable[int] = (0,),
    num_clients: int | None = None,
    rounds_per_stage: int | None = None,
    local_epochs_per_round: int | None = None,
    beam_width: int = 16,
    branch_factor: int = 4,
    lc_maximum_search_expansions: int = 200_000,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Materialize and audit every requested immutable stream cell."""

    alpha_values = tuple(float(value) for value in alphas)
    profile_values = tuple(str(value) for value in temporal_profiles)
    seed_values = tuple(int(value) for value in seeds)
    if not alpha_values or not seed_values:
        raise ValueError("At least one alpha and one seed are required.")
    base = GECKOConfig.from_yaml(config_path)
    validate_lpt_protocol(protocol_version, base.scenario.problem)
    if num_clients is not None:
        base = replace(
            base, partition=replace(base.partition, num_clients=num_clients)
        )
    training_updates: dict[str, int] = {}
    if rounds_per_stage is not None:
        training_updates["rounds_per_stage"] = rounds_per_stage
    if local_epochs_per_round is not None:
        training_updates["local_epochs_per_round"] = local_epochs_per_round
    if training_updates:
        base = replace(base, training=replace(base.training, **training_updates))
    base.validate()
    records: list[dict[str, Any]] = []
    failed_cells: list[dict[str, Any]] = []
    shared_domain_prepared: PreparedScenario | None = None
    shared_domain_topology: WeightedLogicalTopology | None = None
    large_domain = (
        base.scenario.problem == "NC"
        and base.scenario.incremental_setting == "domain"
    )
    if large_domain:
        first_seed = seed_values[0]
        print(
            "[construct] preparing shared NC-domain scenario/topology "
            f"from seed={first_seed}",
            flush=True,
        )
        shared_domain_prepared = prepare_lpt_scenario(replace(base, seed=first_seed))
        scenario = shared_domain_prepared.scenario
        if scenario.edge_index.shape[1] >= LARGE_GRAPH_EDGE_THRESHOLD:
            print(
                "[construct] building compact large-graph topology "
                f"edges={int(scenario.edge_index.shape[1])}",
                flush=True,
            )
            shared_domain_topology = build_large_bidirected_simple_topology(
                scenario.edge_index,
                num_nodes=int(scenario.node_features.shape[0]),
                representation_id="nc_domain_task_dirichlet_compact_bidirected_v1",
            )
        else:
            print(
                "[construct] building weighted logical topology "
                f"edges={int(scenario.edge_index.shape[1])}",
                flush=True,
            )
            shared_domain_topology = build_weighted_logical_topology(
                scenario.edge_index,
                num_nodes=int(scenario.node_features.shape[0]),
                directed=False,
                representation_id="nc_domain_task_dirichlet_topology_v1",
            )
    total_cells = len(seed_values) * len(alpha_values)
    completed_cells = 0
    for seed in seed_values:
        seeded = replace(base, seed=seed)
        print(f"[construct] seed={seed}: preparing scenario", flush=True)
        prepared = (
            replace(shared_domain_prepared, config=seeded)
            if shared_domain_prepared is not None
            else prepare_lpt_scenario(seeded)
        )
        precomputed_micro_result: MicroCommunityResult | None = None
        if shared_domain_topology is not None and isinstance(
            shared_domain_topology.incident_edge_ids, EmptyCSRRows
        ):
            print(f"[construct] seed={seed}: building METIS microcommunities", flush=True)
            precomputed_micro_result = build_large_metis_microcommunities(
                shared_domain_topology,
                SoftMicroCommunityConfig(
                    seed=seed,
                    maximum_microcommunity_nodes=4096,
                    louvain_resolution=1.0,
                ),
            )
        for alpha in alpha_values:
            alpha_config = replace(
                prepared.config,
                partition=replace(
                    prepared.config.partition, dirichlet_alpha=alpha
                ),
            )
            alpha_prepared = replace(prepared, config=alpha_config)
            try:
                print(
                    "[construct] "
                    f"cell {completed_cells + 1}/{total_cells}: "
                    f"seed={seed} alpha={alpha:g} partition/search start",
                    flush=True,
                )
                scenario, partition, construction = construct_exact_dirichlet_partition(
                    alpha_prepared,
                    alpha=alpha,
                    protocol_version=protocol_version,
                    beam_width=beam_width,
                    branch_factor=branch_factor,
                    lc_maximum_search_expansions=lc_maximum_search_expansions,
                    precomputed_topology=shared_domain_topology,
                    precomputed_micro_result=precomputed_micro_result,
                )
            except (LCSoftCommunityError, ExactDirichletQuotaInfeasibleError) as error:
                completed_cells += 1
                print(
                    "[construct] "
                    f"cell {completed_cells}/{total_cells}: "
                    f"seed={seed} alpha={alpha:g} failed "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                failed_cells.append(
                    {
                        "seed": seed,
                        "alpha_dirichlet": alpha,
                        "status": getattr(error, "status", "quota_infeasible"),
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "certificate": getattr(error, "certificate", {}),
                        "lpt_plan": _plan_payload(prepared.lpt_plan),
                    }
                )
                continue
            streams, cells = build_temporal_variants(
                alpha_config,
                scenario,
                partition,
                profiles=profile_values,
            )
            for cell in cells:
                profile = str(cell["order_profile"])
                stream_key = str(cell.get("stream_key", profile))
                print(
                    "[construct] "
                    f"seed={seed} alpha={alpha:g} order={profile}: saving stream",
                    flush=True,
                )
                path = save_stream(
                    streams[stream_key], store_root, repository_root=repository_root
                )
                audit = audit_stream(path)
                print(
                    "[construct] "
                    f"seed={seed} alpha={alpha:g} order={profile}: "
                    f"audit={audit.get('release_status', 'unassessed')}",
                    flush=True,
                )
                records.append(
                    {
                        **cell,
                        **construction,
                        "seed": seed,
                        "stream_path": str(path.resolve()),
                        "scientific_fingerprint": audit["scientific_fingerprint"],
                        "release_status": audit.get("release_status", "unassessed"),
                        "lpt_plan": _plan_payload(prepared.lpt_plan),
                    }
                )
            completed_cells += 1
            print(
                "[construct] "
                f"cell {completed_cells}/{total_cells}: "
                f"seed={seed} alpha={alpha:g} done; "
                f"records={len(records)} failed={len(failed_cells)}",
                flush=True,
            )
            del streams, scenario, partition
            gc.collect()
    manipulation_audit = build_dirichlet_manipulation_audit(
        records,
        failed_cells,
        alphas=alpha_values,
        seeds=seed_values,
    )
    aggregate = {
        "schema": "lpt-dirichlet-temporal-stream-manifest",
        "version": 1 if protocol_version == LPT_V1 else 4,
        "protocol_version": protocol_version,
        "config_path": str(Path(config_path)),
        "store_root": str(Path(store_root).resolve()),
        "alphas": list(alpha_values),
        "temporal_profiles": list(profile_values),
        "seeds": list(seed_values),
        "hconn_cload_role": "not_constrained_not_matched_report_only",
        "dirichlet_manipulation_audit": manipulation_audit,
        "benchmark_eligible": manipulation_audit["benchmark_eligible"],
        "records": records,
        "failed_cells": failed_cells,
    }
    return v1_aggregate(aggregate) if protocol_version == LPT_V1 else aggregate





_RELOCATED_EXPORTS = {'LPDirichletConstructionResult': ('gecko.data.partitioning.dirichlet.lp', 'LPDirichletConstructionResult'), '_construct_lp_exact_dirichlet_partition': ('gecko.data.partitioning.dirichlet.lp', '_construct_lp_exact_dirichlet_partition')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)
