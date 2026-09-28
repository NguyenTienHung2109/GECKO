from __future__ import annotations

from gecko.benchmarks.lpt_dirichlet_temporal import LPT_DIRICHLET_TEMPORAL_VERSION
from gecko.benchmarks.lpt_dirichlet_temporal import LP_SELECTED_TRAIN_POSITIVE_FRACTION

from dataclasses import dataclass
from dataclasses import replace
from typing import Any
import torch
from gecko.data.splits.lp import finalize_partition_first_lp_selected_positives
from gecko.reproducibility import torch_generator
from gecko.data.partitioning.materialize import PersistentSubgraphPartitioner
from gecko.data.partitioning.materialize import rematerialize_partition_context
from gecko.types import PartitionResult
from gecko.types import ScenarioSpec
from gecko.data.partitioning.community.link import LCSoftCommunityError

@dataclass(frozen=True)
class LPDirichletConstructionResult:
    """Minimal immutable evidence emitted by LP capacity-projected Dirichlet."""

    status: str
    owner: torch.Tensor
    selected_positive_count: int
    diagnostics: dict[str, Any]


def _construct_lp_exact_dirichlet_partition(
    prepared: PreparedScenario,
    *,
    alpha: float,
    beam_width: int,
    branch_factor: int,
    maximum_search_expansions: int,
) -> tuple[ScenarioSpec, PartitionResult, LPDirichletConstructionResult]:
    """Construct LP-Domain with a topology-only METIS owner and exact quota.

    Ownership is made before the query split from the fixed base graph.  This
    prevents LP positives/candidates from changing the topology partition.  We
    then reserve local validation support and an exact capacity-aware global
    test-positive budget before projecting one seeded Dirichlet draw onto the
    remaining training-positive capacities.  Both integer allocations are
    exact and evaluation identities never influence the Dirichlet draw.
    """
    from gecko.benchmarks.lpt_dirichlet_temporal import LPT_DIRICHLET_TEMPORAL_VERSION
    from gecko.benchmarks.lpt_dirichlet_temporal import LP_SELECTED_TRAIN_POSITIVE_FRACTION
    from gecko.benchmarks.lpt_dirichlet_temporal import PreparedScenario
    from gecko.benchmarks.lpt_dirichlet_temporal import _capacity_aware_positive_quota
    from gecko.benchmarks.lpt_dirichlet_temporal import _lp_quota_skew_diagnostics
    from gecko.benchmarks.lpt_dirichlet_temporal import _merge_lp_final_diagnostics
    from gecko.benchmarks.lpt_dirichlet_temporal import _plan_payload

    del beam_width, branch_factor, maximum_search_expansions
    pending = prepared.scenario
    if pending.query_endpoints is None or not pending.metadata.get(
        "partition_first_pending", False
    ):
        raise ValueError("LP exact Dirichlet construction requires a pending LP pool.")
    num_clients = prepared.config.partition.num_clients
    min_train = prepared.config.partition.minimum_train_queries_per_client_task
    min_val = prepared.config.partition.minimum_validation_queries_per_client_task
    min_test = prepared.config.partition.minimum_test_queries_per_client_task
    configured_test_budget = (
        prepared.config.scenario.lp_evaluation_positives_per_task
    )
    test_budget_per_task = (
        num_clients * min_test
        if configured_test_budget is None
        else int(configured_test_budget)
    )
    required_support = min_train + min_val + min_test
    rejected_proposals: list[dict[str, Any]] = []
    base_partition: PartitionResult | None = None
    accepted_proposal_index: int | None = None
    accepted_proposal_seed: int | None = None
    for proposal_index in range(
        prepared.config.partition.maximum_lp_partition_proposals
    ):
        proposal_seed = prepared.config.seed + proposal_index * 1_000_003
        candidate = PersistentSubgraphPartitioner(
            prepared.config.partition, proposal_seed
        ).partition(pending)
        candidate_owner = candidate.node_owner.detach().cpu().long()
        failures: list[dict[str, int]] = []
        for task in range(pending.num_tasks):
            query_ids = pending.query_ids_by_task_split[task]["train"].detach().cpu().long()
            endpoints = pending.query_endpoints[query_ids].detach().cpu().long()
            endpoint_owner = candidate_owner[endpoints[:, 0]]
            internal = endpoint_owner == candidate_owner[endpoints[:, 1]]
            counts = torch.bincount(
                endpoint_owner[internal], minlength=num_clients
            )
            for client in range(num_clients):
                available = int(counts[client])
                if available < required_support:
                    failures.append(
                        {
                            "task": task,
                            "client": client,
                            "available": available,
                            "required": required_support,
                        }
                    )
            test_capacities = counts - min_val - min_train
            if int(test_capacities.clamp_min(0).sum()) < test_budget_per_task:
                failures.append(
                    {
                        "task": task,
                        "client": -1,
                        "available": int(test_capacities.clamp_min(0).sum()),
                        "required": test_budget_per_task,
                    }
                )
        if not failures:
            base_partition = candidate
            accepted_proposal_index = proposal_index
            accepted_proposal_seed = proposal_seed
            break
        rejected_proposals.append(
            {
                "proposal_index": proposal_index,
                "proposal_seed": proposal_seed,
                "failure_count": len(failures),
                "minimum_available": min(
                    (failure["available"] for failure in failures), default=0
                ),
                "failures": failures,
            }
        )
    if base_partition is None:
        raise LCSoftCommunityError(
            "structurally_infeasible",
            "No deterministic topology-only LP partition proposal satisfies "
            "minimum positive support for every client-task.",
            certificate={
                "stage": "lp_partition_proposal_support_audit",
                "maximum_proposals": (
                    prepared.config.partition.maximum_lp_partition_proposals
                ),
                "required_support": required_support,
                "rejected_proposals": rejected_proposals,
            },
        )
    owner = base_partition.node_owner.detach().cpu().long()
    selected_pairs: dict[int, torch.Tensor] = {}
    heldout_pairs: dict[int, dict[str, torch.Tensor]] = {}
    quota_rows: list[list[int]] = [[0] * pending.num_tasks for _ in range(num_clients)]
    capacity_rows: list[list[int]] = [[0] * pending.num_tasks for _ in range(num_clients)]
    weights_rows: list[list[float]] = [[0.0] * pending.num_tasks for _ in range(num_clients)]
    test_quota_rows: list[list[int]] = [
        [0] * pending.num_tasks for _ in range(num_clients)
    ]
    for task in range(pending.num_tasks):
        query_ids = pending.query_ids_by_task_split[task]["train"].detach().cpu().long()
        endpoints = pending.query_endpoints[query_ids].detach().cpu().long()
        internal = owner[endpoints[:, 0]] == owner[endpoints[:, 1]]
        if not bool(internal.any()):
            raise LCSoftCommunityError(
                "structurally_infeasible",
                f"LP task={task} has no internal positives after topology-only ownership.",
                certificate={"stage": "lp_internal_positive_support", "task": task},
            )
        ordered_by_client: dict[int, torch.Tensor] = {}
        heldout_pairs[task] = {
            "val": torch.empty((0, 2), dtype=torch.long),
            "test": torch.empty((0, 2), dtype=torch.long),
        }
        for client in range(num_clients):
            ids = query_ids[internal & (owner[endpoints[:, 0]] == client)]
            required = required_support
            if ids.numel() < required:
                raise LCSoftCommunityError(
                    "structurally_infeasible",
                    f"LP task={task} client={client} has internal_positives={ids.numel()} "
                    f"but requires={required}.",
                    certificate={
                        "stage": "lp_internal_positive_support",
                        "task": task,
                        "client": client,
                        "available": int(ids.numel()),
                        "required": required,
                    },
                )
            ordered = ids[
                torch.randperm(
                    ids.numel(),
                    generator=torch_generator(
                        prepared.config.seed, "lp-dirichlet-local-positive-order", task, client
                    ),
                )
            ]
            ordered_by_client[client] = ordered
        test_capacities = torch.tensor(
            [
                int(ordered_by_client[client].numel()) - min_val - min_train
                for client in range(num_clients)
            ],
            dtype=torch.long,
        )
        try:
            test_quota = _capacity_aware_positive_quota(
                test_capacities,
                total=test_budget_per_task,
                minimum=min_test,
            )
        except ValueError as error:
            raise LCSoftCommunityError(
                "structurally_infeasible",
                f"LP task={task} cannot allocate the fixed test-positive budget: {error}",
                certificate={
                    "stage": "lp_capacity_aware_test_positive_allocation",
                    "task": task,
                    "test_budget": test_budget_per_task,
                    "minimum_per_client": min_test,
                    "capacities": test_capacities.tolist(),
                },
            ) from error
        by_client: dict[int, torch.Tensor] = {}
        for client in range(num_clients):
            ordered = ordered_by_client[client]
            client_test = int(test_quota[client])
            heldout_pairs[task]["val"] = torch.cat(
                (heldout_pairs[task]["val"], pending.query_endpoints[ordered[:min_val]])
            )
            heldout_pairs[task]["test"] = torch.cat(
                (
                    heldout_pairs[task]["test"],
                    pending.query_endpoints[
                        ordered[min_val : min_val + client_test]
                    ],
                )
            )
            by_client[client] = ordered[min_val + client_test :]
            capacity_rows[client][task] = int(by_client[client].numel())
            test_quota_rows[client][task] = client_test
        capacities = torch.tensor(
            [capacity_rows[client][task] for client in range(num_clients)], dtype=torch.long
        )
        target = min(
            int(capacities.sum()),
            max(num_clients * min_train, int(capacities.sum() * LP_SELECTED_TRAIN_POSITIVE_FRACTION)),
        )
        generator = torch_generator(prepared.config.seed, "lp-dirichlet-capacity-projection", task, alpha)
        raw = torch._standard_gamma(torch.full((num_clients,), float(alpha)), generator=generator)
        weights = raw / raw.sum()
        quota = torch.full((num_clients,), min_train, dtype=torch.long)
        if bool((capacities < quota).any()):
            raise AssertionError("LP support check did not preserve training capacity.")
        for _ in range(target - int(quota.sum())):
            available = torch.nonzero(quota < capacities, as_tuple=True)[0]
            if available.numel() == 0:
                break
            client = min(
                available.tolist(),
                key=lambda value: (
                    -float(weights[value] * target - quota[value]),
                    value,
                ),
            )
            quota[client] += 1
        if int(quota.sum()) != target:
            raise AssertionError("LP Dirichlet capacity projection did not reach its fixed budget.")
        selected_parts = []
        for client in range(num_clients):
            selected_parts.append(by_client[client][: int(quota[client])])
            quota_rows[client][task] = int(quota[client])
            weights_rows[client][task] = float(weights[client])
        selected_ids = torch.cat(selected_parts)
        selected_pairs[task] = pending.query_endpoints[selected_ids].detach().cpu().long()
    scenario = finalize_partition_first_lp_selected_positives(
        pending,
        owner,
        selected_train_pairs_by_task=selected_pairs,
        heldout_positive_pairs_by_task_split=heldout_pairs,
        num_clients=num_clients,
        seed=prepared.config.seed,
        training_negative_ratio=prepared.config.training.lp_train_negative_ratio,
        evaluation_negatives_per_client_task=(
            prepared.config.scenario.lp_evaluation_negatives_per_client_task
        ),
    )
    scenario = replace(
        scenario,
        metadata={
            **scenario.metadata,
            "lp_evaluation_positive_allocation_policy": (
                "fixed_per_task_balanced_capacity_capped_with_client_minimum_v1"
            ),
            "lp_evaluation_positives_per_task": test_budget_per_task,
            "lp_test_positive_quota_client_by_task": test_quota_rows,
            "lp_primary_metric_aggregation": "positive_query_micro",
            "lp_macro_client_metric_role": "diagnostic_only",
        },
    )
    diagnostics = {
        **base_partition.diagnostics,
        "protocol_version": LPT_DIRICHLET_TEMPORAL_VERSION,
        "class_task_policy": prepared.config.scenario.class_task_policy,
        "lpt_plan": _plan_payload(prepared.lpt_plan),
        "dirichlet_semantic_unit": "selected_train_positive_edge_domain_task_id",
        "target_labels_accessed_for_ownership": False,
        "lp_evaluation_candidates_accessed_for_ownership": False,
        "lp_reserved_positive_anchor_policy": "seeded_identity_only_pre_split",
        "lp_partition_proposal_index": accepted_proposal_index,
        "lp_partition_proposal_seed": accepted_proposal_seed,
        "lp_partition_rejected_proposal_count": len(rejected_proposals),
        "lp_partition_rejected_proposals": rejected_proposals,
        "lp_partition_proposal_selection": (
            "topology_only_then_pre_split_positive_support_audit_first_feasible"
        ),
        "lp_selected_train_positive_fraction": LP_SELECTED_TRAIN_POSITIVE_FRACTION,
        "lp_evaluation_positive_allocation_policy": (
            "fixed_per_task_balanced_capacity_capped_with_client_minimum_v1"
        ),
        "lp_evaluation_positives_per_task": test_budget_per_task,
        "lp_test_positive_quota_client_by_task": test_quota_rows,
        "lp_primary_metric_aggregation": "positive_query_micro",
        "lp_macro_client_metric_role": "diagnostic_only",
        "lp_dirichlet_capacity_projection": {
            "integer_quota_client_by_task": quota_rows,
            "internal_positive_capacity_client_by_task": capacity_rows,
            "sampled_dirichlet_weights_client_by_task": weights_rows,
            "projection": "finite_capacity_greedy_exact_budget_v1",
        },
        **_lp_quota_skew_diagnostics(quota_rows),
        "hconn_target": None,
        "cload_target": None,
        "hconn_cload_role": "reported_diagnostics_only_not_a_constraint",
    }
    partition = rematerialize_partition_context(
        scenario, base_partition, prepared.config.partition
    )
    partition = replace(
        partition,
        diagnostics={
            **_merge_lp_final_diagnostics(diagnostics, partition.diagnostics),
            "partition_mode": LPT_DIRICHLET_TEMPORAL_VERSION,
            "direct_stream_version": LPT_DIRICHLET_TEMPORAL_VERSION,
            "grid_version": LPT_DIRICHLET_TEMPORAL_VERSION,
            "spatial_control": "exact_dirichlet_alpha",
            "legacy_easy_mild_hard_spatial_profile_ignored": True,
            "community_generation_used": True,
            "community_generation_role": "metis_louvain_scaffold_for_exact_lp_positive_dirichlet_ownership",
        },
    )
    result = LPDirichletConstructionResult(
        status="success",
        owner=owner,
        selected_positive_count=sum(
            int(selected_pairs[task].shape[0])
            for task in range(pending.num_tasks)
        ),
        diagnostics=partition.diagnostics,
    )
    return scenario, partition, result


