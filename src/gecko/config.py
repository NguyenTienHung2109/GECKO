"""Typed and validated UEFA configuration."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import math
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Tuple

import yaml

from gecko.benchmarks.terminology import normalize_task_order
from gecko.benchmarks.terminology import task_order_label

from gecko.data.datasets.registry import SUPPORTED_SCENARIOS
from gecko.validation import ConfigurationError
from gecko.validation import UnsupportedCombinationError


@dataclass(frozen=True)
class ScenarioConfig:
    dataset: str
    problem: str
    incremental_setting: str
    num_tasks: int
    metrics: Tuple[str, ...]
    synthetic: bool = False
    domain_constructor: str | None = None
    target_type: str = "single_label"
    save_path: str = "data"
    split_protocol: str = "benchmark_defined"
    feature_provenance: str = "provided_node_attributes"
    lp_candidate_grouping: str = "pooled_per_client_task"
    hits_tie_policy: str = "pessimistic"
    minimum_evaluation_negatives: int = 50
    lp_protocol_name: str = "uefa_lp_fixed_candidates"
    lp_protocol_version: int = 1
    lp_training_negative_policy: str = "fixed_global_validated"
    lp_evaluation_candidate_policy: str = "explicit_fixed_groups"
    lp_legacy_score_comparable: bool = False
    lp_query_split_protocol: str = "prepartition_fixed_split_v1"
    lp_base_topology_policy: str = "legacy_training_context"
    lp_base_edge_ratio: float = 0.0
    lp_source_num_domains: int | None = None
    lp_domain_mapping: str = "identity"
    lp_evaluation_positives_per_task: int | None = None
    lp_evaluation_negatives_per_client_task: int = 1000
    class_task_policy: str = "benchmark_defined"


@dataclass(frozen=True)
class PartitionConfig:
    num_clients: int = 10
    spatial_profile: str = "mild"
    micro_partition_multiplier: int = 10
    client_size_tolerance: float = 0.20
    minimum_train_queries_per_client_task: int = 5
    minimum_validation_queries_per_client_task: int = 1
    minimum_test_queries_per_client_task: int = 1
    maximum_assignment_iterations: int = 500
    maximum_lp_partition_proposals: int = 5
    allow_infeasible: bool = False
    minimum_lc_internal_query_coverage: float | None = None
    minimum_lp_internal_candidate_coverage: float | None = None
    minimum_lp_internal_positive_coverage: float | None = None
    lp_partition_information_scope: str = "all_positive_splits"
    dirichlet_alpha: float | None = None

    @property
    def allocation_alpha(self) -> float | None:
        """Paper-facing name; serialized configurations retain the historical field."""

        return self.dirichlet_alpha


@dataclass(frozen=True)
class OrderConfig:
    profile: str = "mild"
    allow_full_permutation_below_four_tasks: bool = False

    @property
    def task_order(self) -> str:
        return task_order_label(self.profile)


@dataclass(frozen=True)
class TrainingConfig:
    participation_fraction: float = 0.5
    rounds_per_stage: int = 10
    local_epochs_per_round: int = 1
    local_training_mode: str = "full_local_graph"
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    optimizer_state_across_rounds: str = "reset"
    continual_state_across_rounds: str = "persist"
    continual_state_across_stages: str = "persist"
    aggregation_weight: str = "current_supervised_query_count"
    fedprox_mu: float = 0.01
    lp_train_negative_ratio: float = 1.0
    hidden_size: int = 256
    num_layers: int = 3


@dataclass(frozen=True)
class WandbConfig:
    mode: str = "auto"
    project: str = "UEFA"


@dataclass(frozen=True)
class GECKOConfig:
    scenario: ScenarioConfig
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    order: OrderConfig = field(default_factory=OrderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    seed: int = 0
    benchmark_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    output_root: str = "generated_streams"
    benchmark_name: str = "UEFA"
    benchmark_schema_version: int = 1

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GECKOConfig":
        scenario_raw = dict(raw.get("scenario", {}))
        if "problem" in scenario_raw:
            scenario_raw["problem"] = str(scenario_raw["problem"]).upper()
        if "incremental_setting" in scenario_raw:
            scenario_raw["incremental_setting"] = str(
                scenario_raw["incremental_setting"]
            ).lower()
        metrics = scenario_raw.get("metrics", ())
        if isinstance(metrics, str):
            metrics = (metrics,)
        scenario_raw["metrics"] = tuple(metrics)
        partition_raw = dict(raw.get("partition", {}))
        if "allocation_alpha" in partition_raw:
            alpha = partition_raw.pop("allocation_alpha")
            if "dirichlet_alpha" in partition_raw and partition_raw["dirichlet_alpha"] != alpha:
                raise ConfigurationError("allocation_alpha conflicts with dirichlet_alpha.")
            partition_raw["dirichlet_alpha"] = alpha
        order_raw = dict(raw.get("order", {}))
        public_order = order_raw.pop("task_order", None)
        profile = order_raw.get("profile", public_order)
        problem = scenario_raw.get("problem", "")
        num_tasks = scenario_raw.get("num_tasks", 0)
        if public_order is not None:
            normalized_public = normalize_task_order(public_order, problem=problem, num_tasks=num_tasks)
            if profile is not None and normalize_task_order(profile, problem=problem, num_tasks=num_tasks) != normalized_public:
                raise ConfigurationError("task_order conflicts with order.profile.")
            profile = public_order
        if profile is not None:
            order_raw["profile"] = normalize_task_order(profile, problem=problem, num_tasks=num_tasks)
            if profile == "unsynchronized" and order_raw["profile"] == "hard" and num_tasks < 4:
                if order_raw.get("allow_full_permutation_below_four_tasks") is False:
                    raise ConfigurationError("Unsynchronized task order requires full permutation for fewer than four tasks.")
                order_raw["allow_full_permutation_below_four_tasks"] = True
        config = cls(
            scenario=ScenarioConfig(**scenario_raw),
            partition=PartitionConfig(**partition_raw),
            order=OrderConfig(**order_raw),
            training=TrainingConfig(**dict(raw.get("training", {}))),
            wandb=WandbConfig(**dict(raw.get("wandb", {}))),
            seed=int(raw.get("seed", 0)),
            benchmark_seeds=tuple(int(value) for value in raw.get("benchmark_seeds", (0, 1, 2, 3, 4))),
            output_root=str(raw.get("output_root", "generated_streams")),
            benchmark_name=str(raw.get("benchmark_name", "UEFA")),
            benchmark_schema_version=int(raw.get("benchmark_schema_version", 1)),
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GECKOConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise ConfigurationError("The root YAML value must be a mapping.")
        return cls.from_mapping(raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.benchmark_name != "UEFA":
            raise ConfigurationError("benchmark_name must be exactly 'UEFA'.")
        if self.benchmark_schema_version not in {1, 2}:
            raise ConfigurationError("benchmark_schema_version must be 1 or 2.")
        key = (self.scenario.problem.upper(), self.scenario.incremental_setting.lower())
        if key not in SUPPORTED_SCENARIOS:
            raise UnsupportedCombinationError(
                f"UEFA v1 does not support problem={key[0]} with "
                f"incremental_setting={key[1]}."
            )
        if self.scenario.num_tasks < 1:
            raise ConfigurationError("num_tasks must be positive.")
        if self.scenario.class_task_policy not in {
            "benchmark_defined",
            "drop_rarest_train_lpt_balanced_v1",
        }:
            raise ConfigurationError("Unknown class-task construction policy.")
        if (
            self.scenario.class_task_policy == "drop_rarest_train_lpt_balanced_v1"
            and (key[0] not in {"NC", "LC"} or key[1] not in {"task", "class"})
        ):
            raise ConfigurationError(
                "The train-only LPT class policy is valid only for NC/LC Task-IL "
                "or Class-IL."
            )
        if self.scenario.split_protocol not in {
            "benchmark_defined",
            "official_dataset",
            "uefa_within_domain",
            "custom",
            "synthetic",
            "uefa_partition_first",
        }:
            raise ConfigurationError("Unknown split_protocol.")
        if self.scenario.feature_provenance not in {
            "provided_node_attributes",
            "strict_local_derived",
            "public_global_precomputed",
        }:
            raise ConfigurationError("Unknown feature_provenance policy.")
        if self.scenario.lp_candidate_grouping not in {
            "pooled_per_client_task",
            "per_positive",
        }:
            raise ConfigurationError("Unknown LP candidate grouping policy.")
        if self.scenario.hits_tie_policy not in {
            "optimistic",
            "pessimistic",
            "average",
        }:
            raise ConfigurationError("Unknown Hits@K tie policy.")
        if self.scenario.minimum_evaluation_negatives < 1:
            raise ConfigurationError("minimum_evaluation_negatives must be positive.")
        if key[0] == "LP" and self.scenario.lp_protocol_name == "uefa_lp_fixed_candidates":
            expected_lp_protocol = {
                "lp_protocol_name": "uefa_lp_fixed_candidates",
                "lp_protocol_version": 1,
                "lp_training_negative_policy": "fixed_global_validated",
                "lp_evaluation_candidate_policy": "explicit_fixed_groups",
                "lp_legacy_score_comparable": False,
            }
            for field_name, expected_value in expected_lp_protocol.items():
                if getattr(self.scenario, field_name) != expected_value:
                    raise ConfigurationError(
                        f"UEFA LP Protocol v1 requires {field_name}={expected_value!r}."
                    )
        elif key[0] == "LP" and self.scenario.lp_protocol_name == "uefa_lp_partition_first_fixed_candidates":
            expected_lp_protocol = {
                "lp_protocol_version": 2,
                "lp_training_negative_policy": "fixed_global_validated",
                "lp_evaluation_candidate_policy": "explicit_fixed_groups",
                "lp_legacy_score_comparable": False,
                "lp_query_split_protocol": "partition_first_query_split_v1",
                "lp_base_topology_policy": "endpoint_hash_holdout",
                "lp_domain_mapping": "dominant_source_vs_rest_v1",
            }
            for field_name, expected_value in expected_lp_protocol.items():
                if getattr(self.scenario, field_name) != expected_value:
                    raise ConfigurationError(
                        f"UEFA LP Protocol v2 requires {field_name}={expected_value!r}."
                    )
            if self.scenario.split_protocol != "uefa_partition_first":
                raise ConfigurationError(
                    "UEFA LP Protocol v2 requires split_protocol='uefa_partition_first'."
                )
            if self.partition.lp_partition_information_scope != "topology_only":
                raise ConfigurationError(
                    "UEFA LP Protocol v2 requires topology-only partition assignment."
                )
            if self.scenario.lp_source_num_domains is None or (
                self.scenario.lp_source_num_domains < self.scenario.num_tasks
            ):
                raise ConfigurationError(
                    "UEFA LP Protocol v2 requires lp_source_num_domains >= num_tasks."
                )
            if not 0 < self.scenario.lp_base_edge_ratio < 1:
                raise ConfigurationError("lp_base_edge_ratio must be in (0, 1).")
            if (
                self.scenario.lp_evaluation_negatives_per_client_task
                < self.scenario.minimum_evaluation_negatives
            ):
                raise ConfigurationError(
                    "LP evaluation negative pool must satisfy minimum_evaluation_negatives."
                )
            if (
                self.scenario.lp_evaluation_positives_per_task is not None
                and self.scenario.lp_evaluation_positives_per_task
                < self.partition.num_clients
                * self.partition.minimum_test_queries_per_client_task
            ):
                raise ConfigurationError(
                    "LP evaluation positive budget must cover the minimum test "
                    "support of every client."
                )
            if self.order.profile not in {"synchronized", "binary_mismatch"}:
                raise ConfigurationError(
                    "LP-Domain-v2-T2 supports synchronized or binary_mismatch order only."
                )
            if self.training.aggregation_weight != "current_positive_anchor_count":
                raise ConfigurationError(
                    "LP-Domain-v2-T2 aggregates by current_positive_anchor_count."
                )
        elif key[0] == "LP":
            raise ConfigurationError(
                f"Unknown UEFA LP protocol: {self.scenario.lp_protocol_name!r}."
            )
        if (
            not self.scenario.synthetic
            and self.scenario.dataset == "ogbn-proteins"
            and key == ("NC", "domain")
            and self.scenario.domain_constructor == "species"
            and self.scenario.split_protocol == "official_dataset"
        ):
            raise ConfigurationError(
                "OGBN-Proteins species domains conflict with the official "
                "species-based split. Use split_protocol='uefa_within_domain', "
                "a different domain constructor, or a different dataset."
            )
        if (
            not self.scenario.synthetic
            and self.order.profile == "hard"
            and self.scenario.num_tasks < 4
            and not self.order.allow_full_permutation_below_four_tasks
        ):
            raise ConfigurationError(
                "The core hard-order profile is unsupported when num_tasks < 4."
            )
        if (
            self.order.allow_full_permutation_below_four_tasks
            and self.order.profile != "hard"
        ):
            raise ConfigurationError(
                "allow_full_permutation_below_four_tasks is valid only for hard order."
            )
        for metric in self.scenario.metrics:
            normalized = metric.lower()
            if normalized.startswith("hits@"):
                k = int(normalized.split("@", 1)[1])
                if self.scenario.minimum_evaluation_negatives < k:
                    raise ConfigurationError(
                        f"{metric} requires at least {k} evaluation negatives "
                        "per candidate group."
                    )
        if not self.benchmark_seeds:
            raise ConfigurationError("benchmark_seeds must not be empty.")
        if self.partition.num_clients < 1:
            raise ConfigurationError("num_clients must be positive.")
        if not 1 <= self.partition.maximum_lp_partition_proposals <= 20:
            raise ConfigurationError(
                "maximum_lp_partition_proposals must be between 1 and 20."
            )
        coverage_fields = (
            self.partition.minimum_lc_internal_query_coverage,
            self.partition.minimum_lp_internal_candidate_coverage,
            self.partition.minimum_lp_internal_positive_coverage,
        )
        if any(value is not None and not 0 <= value <= 1 for value in coverage_fields):
            raise ConfigurationError("Internal-query coverage thresholds must be in [0, 1].")
        if not self.scenario.synthetic and key[0] == "LC":
            if self.partition.minimum_lc_internal_query_coverage is None:
                raise ConfigurationError(
                    "Real LC configs must predeclare minimum_lc_internal_query_coverage."
                )
        if not self.scenario.synthetic and key[0] == "LP":
            if (
                self.partition.minimum_lp_internal_candidate_coverage is None
                or self.partition.minimum_lp_internal_positive_coverage is None
            ):
                raise ConfigurationError(
                    "Real LP configs must predeclare internal candidate and positive coverage."
                )
        if self.partition.spatial_profile not in {"easy", "mild", "hard"}:
            raise ConfigurationError("spatial_profile must be easy, mild, or hard.")
        if (
            self.partition.dirichlet_alpha is not None
            and (
                not math.isfinite(self.partition.dirichlet_alpha)
                or self.partition.dirichlet_alpha <= 0
            )
        ):
            raise ConfigurationError("allocation_alpha must be finite and positive when provided.")
        if self.partition.lp_partition_information_scope not in {
            "all_positive_splits",
            "topology_and_train_queries",
            "topology_only",
        }:
            raise ConfigurationError(
                "lp_partition_information_scope must be all_positive_splits, "
                "topology_and_train_queries, or topology_only."
            )
        if self.order.profile not in {
            "synchronized", "mild", "hard", "unconstrained", "binary_mismatch"
        }:
            raise ConfigurationError("Unknown task-order profile.")
        if not 0 < self.training.participation_fraction <= 1:
            raise ConfigurationError("participation_fraction must be in (0, 1].")
        if self.training.rounds_per_stage < 1 or self.training.local_epochs_per_round < 1:
            raise ConfigurationError("Training round and epoch counts must be positive.")
        if self.training.hidden_size < 1 or self.training.num_layers < 1:
            raise ConfigurationError("hidden_size and num_layers must be positive.")
        if self.training.learning_rate <= 0 or self.training.weight_decay < 0:
            raise ConfigurationError("learning_rate must be positive and weight_decay non-negative.")
        if self.training.lp_train_negative_ratio <= 0:
            raise ConfigurationError("lp_train_negative_ratio must be positive.")
        if self.training.optimizer_state_across_rounds != "reset":
            raise ConfigurationError("UEFA v1 supports optimizer_state_across_rounds='reset' only.")
        if self.training.local_training_mode != "full_local_graph":
            raise ConfigurationError("UEFA v1 supports local_training_mode='full_local_graph' only.")
        if (
            self.training.continual_state_across_rounds != "persist"
            or self.training.continual_state_across_stages != "persist"
        ):
            raise ConfigurationError("UEFA v1 requires continual algorithm state to persist.")
        if self.training.aggregation_weight not in {
            "current_supervised_query_count", "current_positive_anchor_count"
        }:
            raise ConfigurationError(
                "Unknown aggregation_weight policy."
            )
        if (
            self.training.aggregation_weight == "current_positive_anchor_count"
            and key[0] != "LP"
        ):
            raise ConfigurationError(
                "current_positive_anchor_count is defined for LP only."
            )
        if self.wandb.mode not in {"auto", "online", "offline", "disabled"}:
            raise ConfigurationError("wandb.mode must be auto, online, offline, or disabled.")
        if self.wandb.project != "UEFA":
            raise ConfigurationError("The UEFA benchmark W&B project must be 'UEFA'.")
