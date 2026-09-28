"""Fail-closed validation and support resolution for UEFA v2 method configs.

The v2 config name identifies an implementation family; nested strategy and
continual-method names are not trusted to confer support.  In particular, a
known but unimplemented published strategy composed with ``Bare`` remains
unsupported and benchmark-ineligible.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Tuple

from gecko.algorithms.catalog import MethodRegistry


METHOD_CONFIG_SCHEMA = "uefa-method-config"
METHOD_CONFIG_VERSION = 2
MethodParameterValue = bool | int | float | str | None | Tuple[int | float, ...]

_TOP_LEVEL_FIELDS = frozenset(
    {"schema", "version", "name", "strategy", "continual_method"}
)
_COMPONENT_FIELDS = frozenset({"name", "parameters"})
_LEGACY_STRATEGIES = ("local_only", "fedavg", "fedprox")
_LEGACY_CONTINUAL_METHODS = (
    "Bare",
    "LwF",
    "EWC",
    "MAS",
    "ERGNN",
    "generic_replay",
    "generic_importance_regularization",
    "generic_weight_isolation",
)
_SCENARIOS = {
    ("NC", "task"),
    ("NC", "class"),
    ("NC", "domain"),
    ("LC", "task"),
    ("LC", "class"),
    ("LC", "domain"),
    ("LP", "domain"),
}


class MethodConfigValidationError(ValueError):
    """A v2 method config is malformed or has inconsistent identities."""


class UnsupportedMethodConfigError(MethodConfigValidationError):
    """A well-formed known config is not runnable in the resolved setting."""

    def __init__(self, resolution: "ResolvedMethodConfig") -> None:
        self.resolution = resolution
        super().__init__(
            f"Unsupported v2 method config {resolution.name!r}: {resolution.reason}"
        )


@dataclass(frozen=True)
class ResolvedMethodConfig:
    """Immutable support and eligibility decision for one canonical config."""

    schema: str
    version: int
    name: str
    strategy_name: str
    continual_method_name: str
    strategy_parameters: Tuple[Tuple[str, MethodParameterValue], ...]
    continual_method_parameters: Tuple[Tuple[str, MethodParameterValue], ...]
    support_status: str
    scientific_fidelity: str
    runnable: bool
    benchmark_eligible: bool
    reason: str
    test_coverage: str
    problem_type: str | None
    incremental_setting: str | None

    def to_dict(self) -> Dict[str, object]:
        """Render a JSON-compatible record for results and audit evidence."""

        return {
            "schema": self.schema,
            "version": self.version,
            "name": self.name,
            "strategy": {
                "name": self.strategy_name,
                "parameters": dict(self.strategy_parameters),
            },
            "continual_method": {
                "name": self.continual_method_name,
                "parameters": dict(self.continual_method_parameters),
            },
            "support_status": self.support_status,
            "scientific_fidelity": self.scientific_fidelity,
            "runnable": self.runnable,
            "benchmark_eligible": self.benchmark_eligible,
            "reason": self.reason,
            "test_coverage": self.test_coverage,
            "problem_type": self.problem_type,
            "incremental_setting": self.incremental_setting,
        }


@dataclass(frozen=True)
class _FamilySpec:
    name: str
    strategies: Tuple[str, ...]
    continual_methods: Tuple[str, ...]
    strategy_parameter_fields: Tuple[str, ...]
    continual_parameter_fields: Tuple[str, ...]
    implemented: bool
    support_status: str
    scientific_fidelity: str
    reason: str
    test_coverage: str


_FAMILIES = {
    "legacy_adapter_v1": _FamilySpec(
        name="legacy_adapter_v1",
        strategies=_LEGACY_STRATEGIES,
        continual_methods=_LEGACY_CONTINUAL_METHODS,
        strategy_parameter_fields=(),
        continual_parameter_fields=(),
        implemented=True,
        support_status="legacy_compatibility_resolution_required",
        scientific_fidelity="registry_resolved",
        reason="Resolved from the existing UEFA method compatibility registry.",
        test_coverage="legacy_v1_evidence",
    ),
    "scaffold_uefa_adam_v1": _FamilySpec(
        name="scaffold_uefa_adam_v1",
        strategies=("scaffold",),
        continual_methods=("Bare",),
        strategy_parameter_fields=(
            "control_updates_enabled",
            "correction_enabled",
        ),
        continual_parameter_fields=(),
        implemented=True,
        support_status="implemented_unverified",
        scientific_fidelity="mechanism_adaptation",
        reason=(
            "The UEFA Adam SCAFFOLD adaptation is integrated, but real seed-0 "
            "and five-seed promotion evidence is pending."
        ),
        test_coverage="closed_form_state_accounting_and_resume_unit_tests",
    ),
    "twp_uefa_v1": _FamilySpec(
        "twp_uefa_v1",
        ("local_only", "fedavg", "fedprox", "scaffold"),
        ("TWP",),
        (),
        (
            "beta",
            "lambda_l",
            "lambda_t",
            "middle_layer_index",
            "significant_threshold",
        ),
        True,
        "implemented_unverified",
        "mechanism_adaptation",
        (
            "The TWP equations and client-private state are implemented; real "
            "seed-0 and five-seed promotion evidence is pending."
        ),
        "paper_equation_state_checkpoint_and_synthetic_tests",
    ),
    "gem_uefa_v1": _FamilySpec(
        "gem_uefa_v1",
        ("local_only", "fedavg", "fedprox", "scaffold"),
        ("GEM",),
        (),
        (
            "margin",
            "memory_size",
            "projection_epsilon",
            "violation_tolerance",
        ),
        True,
        "implemented_unverified",
        "mechanism_adaptation",
        (
            "Client-private per-task replay gradients and the GEM dual-QP "
            "projection are integrated for NC/LC task, class, and domain streams; "
            "real-data promotion evidence is pending."
        ),
        "qp_oracle_memory_state_checkpoint_and_six_scenario_synthetic_tests",
    ),
    "ssm_uefa_v1": _FamilySpec(
        "ssm_uefa_v1",
        ("local_only", "fedavg", "fedprox", "scaffold"),
        ("SSM",),
        (),
        (
            "hop_budgets",
            "replay_ceiling_bytes",
            "replay_weight",
            "sampler_mode",
            "stage_count",
        ),
        True,
        "implemented_unverified",
        "mechanism_adaptation",
        (
            "The strict-local SSM replay mechanism and private serialized-byte "
            "store are implemented; real seed-0 and five-seed promotion evidence "
            "is pending."
        ),
        "paper_sampling_state_accounting_checkpoint_and_synthetic_tests",
    ),
    "cat_uefa_v1": _FamilySpec(
        "cat_uefa_v1",
        ("local_only", "fedavg", "fedprox", "scaffold"),
        ("CaT",),
        (),
        (
            "condensation_lr",
            "condensation_steps",
            "feature_initialization",
            "memory_ceiling_bytes",
            "stage_count",
            "synthetic_nodes_per_class",
        ),
        True,
        "implemented_unverified",
        "mechanism_adaptation",
        (
            "CaT class-wise random-encoder condensation and balanced "
            "Training-in-Memory are integrated for strict-local NC-Class; "
            "real-data promotion evidence is pending."
        ),
        "condensation_memory_balance_checkpoint_and_synthetic_tests",
    ),
    "graphkeeper_uefa_v1": _FamilySpec(
        "graphkeeper_uefa_v1",
        ("local_only", "fedavg", "fedprox"),
        ("GraphKeeper", "GraphKeeper-LP"),
        (),
        (
            "adapter_learning_rate",
            "dbscan_eps",
            "dbscan_min_samples",
            "edge_drop",
            "feature_drop",
            "inter_weight",
            "intra_weight",
            "max_cluster_nodes",
            "max_prototypes_per_domain",
            "pretrain_edges",
            "pretrain_weight",
            "rank",
            "ridge_lambda",
            "router_projection_dim",
            "stage_count",
            "temperature",
        ),
        True,
        "implemented_seed0_sanity",
        "paper_faithful_benchmark_adaptation",
        (
            "Layer-wise graph-LoRA experts, DBSCAN intra/inter disentanglement, "
            "recursive analytic preservation, and frozen-random-GNN hidden-domain "
            "routing are integrated for strict-local Domain-IL. Under FedAvg or "
            "FedProx, only the graph backbone model is shared; experts, routers, "
            "prototypes, and recursive ridge state remain client-private."
        ),
        "paper_component_oracles_checkpoint_state_boundary_and_real_gpu_seed0_sanity",
    ),
    "dslr_uefa_v1": _FamilySpec(
        "dslr_uefa_v1",
        ("local_only", "fedavg", "fedprox", "scaffold"),
        ("DSLR",),
        (),
        (
            "beta",
            "candidate_k",
            "radius",
            "replay_ceiling_bytes",
            "replay_fraction",
            "selection_mode",
            "structure_epochs",
            "structure_heads",
            "structure_hidden_dim",
            "structure_lambda",
            "structure_learning_rate",
            "structure_mode",
            "tau",
            "top_n",
            "undirected",
        ),
        True,
        "implemented_unverified",
        "mechanism_adaptation",
        (
            "The strict-local DSLR coverage replay, private GAT structure learner, "
            "and per-task topology overlays are implemented; real seed-0 and "
            "five-seed promotion evidence is pending."
        ),
        "paper_equation_leakage_state_checkpoint_and_synthetic_tests",
    ),
    "dslr_normalized_v1": _FamilySpec(
        "dslr_normalized_v1",
        ("local_only",),
        ("DSLR-Normalized",),
        (),
        (
            "beta",
            "candidate_k",
            "radius",
            "replay_ceiling_bytes",
            "replay_fraction",
            "selection_mode",
            "structure_epochs",
            "structure_heads",
            "structure_hidden_dim",
            "structure_lambda",
            "structure_learning_rate",
            "structure_mode",
            "tau",
            "top_n",
            "undirected",
        ),
        True,
        "implemented_unverified",
        "official_code_reduction_benchmark_adaptation",
        (
            "DSLR-Normalized preserves DSLR's mechanisms but uses mean link BCE, "
            "matching the pinned authors' executable reduction and keeping the "
            "fixed lambda meaningful across client graph sizes."
        ),
        "reduction_scale_behavioral_oracle_real_data_smoke_pending",
    ),
    "dslr_diagnostic_v1": _FamilySpec(
        "dslr_diagnostic_v1",
        ("local_only",),
        ("DSLRDiagnostic",),
        (),
        (
            "beta",
            "candidate_k",
            "radius",
            "replay_ceiling_bytes",
            "replay_fraction",
            "selection_mode",
            "structure_epochs",
            "structure_heads",
            "structure_hidden_dim",
            "structure_lambda",
            "structure_learning_rate",
            "structure_mode",
            "tau",
            "top_n",
            "undirected",
        ),
        True,
        "diagnostic_only_component_ablation",
        "diagnostic_only",
        (
            "Named paper Table-4 component ablation; it must not be reported "
            "as the full DSLR method."
        ),
        "paper_component_ablation_behavior_tests",
    ),
    "fedgta_uefa_v1": _FamilySpec(
        "fedgta_uefa_v1",
        ("fedgta",),
        ("Bare", "TWP", "SSM", "DSLR"),
        (
            "moment_order",
            "moment_type",
            "propagation_alpha",
            "propagation_steps",
            "similarity_threshold",
            "temperature",
        ),
        (),
        True,
        "implemented_unverified",
        "mechanism_adaptation",
        (
            "FedGTA strict-local personalized aggregation is implemented, but "
            "coordinator/runtime integration and real-data evidence are incomplete."
        ),
        "paper_oracle_state_accounting_and_leakage_unit_tests",
    ),
    "fed_pub_uefa_v1": _FamilySpec(
        "fed_pub_uefa_v1",
        ("fed_pub",),
        ("Bare", "TWP", "DSLR"),
        ("lambda1", "lambda2", "proxy_seed", "tau"),
        (),
        True,
        "implemented_unverified",
        "paper_faithful_benchmark_adaptation",
        (
            "FED-PUB differentiable private masks, random-graph functional "
            "embeddings, and personalized similarity aggregation are integrated; "
            "five-seed real-data promotion evidence remains pending."
        ),
        "paper_mask_proxy_similarity_state_accounting_unit_tests",
    ),
    "feddc_uefa_v1": _FamilySpec(
        "feddc_uefa_v1",
        ("feddc",),
        ("Bare",),
        ("alpha", "drift_enabled"),
        (),
        True,
        "implemented_unverified",
        "paper_faithful_benchmark_adaptation",
        (
            "FedDC Eq. (4) penalty/correction, Eq. (6) persistent drift, and "
            "post-update corrected-model aggregation are integrated; five-seed "
            "real-data promotion evidence remains pending."
        ),
        "paper_equation_drift_state_accounting_checkpoint_unit_tests",
    ),
    "power_uefa_v1": _FamilySpec(
        "power_uefa_v1",
        ("power_uefa",),
        ("Bare",),
        (
            "ablation_mode",
            "alpha",
            "beta",
            "coverage_threshold",
            "reconstruction_steps",
            "replay_ceiling_bytes",
            "samples_per_class",
            "server_epochs",
            "server_learning_rate",
            "trajectory_decay",
        ),
        (),
        True,
        "implemented_unverified",
        "paper_faithful_benchmark_adaptation",
        (
            "POWER equations 2--14, gradient inversion, cumulative trajectory, "
            "pseudo-graph reconstruction, and expertise transfer are integrated; "
            "five-seed real-data promotion evidence remains pending."
        ),
        "paper_equation_gradient_reconstruction_transfer_accounting_checkpoint_tests",
    ),
    "motion_uefa_v1": _FamilySpec(
        "motion_uefa_v1",
        ("motion",),
        ("Bare",),
        (
            "buffer_size",
            "expert_select",
            "k_list",
            "node_reduction_rate",
            "pcb_max_ratio",
            "pcb_min_ratio",
            "pcb_ratio",
            "replay_ceiling_bytes",
            "replay_weight",
            "similarity_threshold",
            "use_node_mahalanobis",
            "use_node_mmd",
            "use_node_positional",
        ),
        (),
        True,
        "implemented_unverified",
        "clean_room_leakage_safe_mechanism_adaptation",
        (
            "MOTION G-TMSC and G-EPAE are integrated for NC-Class with "
            "strict-local client memory and no graph/feature uplink; a bounded "
            "paired Cora seed-0 smoke passed, while canonical multi-seed "
            "promotion evidence remains pending."
        ),
        "module_oracle_leakage_state_checkpoint_synthetic_and_cora_seed0_smoke",
    ),
    "fedfst_uefa_v1": _FamilySpec(
        "fedfst_uefa_v1",
        ("fedfst",),
        ("Bare",),
        (
            "client_nodes_per_class",
            "client_learning_rate",
            "client_weight_decay",
            "class_il_output_training_policy",
            "distillation_epochs",
            "distillation_early_stop_policy",
            "distillation_learning_rate",
            "distillation_validation_checkpoints",
            "edge_reduction_ratio",
            "generated_edges_per_node",
            "generator_dropout",
            "generator_epochs",
            "generator_learning_rate",
            "generator_rounds",
            "lambda_kl",
            "lambda_low",
            "method_seed",
            "noise_dim",
            "sampled_feature_fraction",
            "server_initial_edge_policy",
            "server_max_edges_per_node",
            "server_nodes_per_class",
            "smoothing_hops",
            "topology_max_iterations",
            "topology_tolerance",
        ),
        (),
        True,
        "supported",
        "paper_equation_uefa_adaptation",
        (
            "FedFST HHKR, equation-8 generator aggregation, and edge-wise HLST "
            "are integrated for client-ordered NC-Class and task-head-aware NC-Task. "
            "Auxiliary federation uses the all-client union of each immutable "
            "ordinary participation stage; full NC-Class/Task CUDA smokes pass."
        ),
        "paper_equation_task_head_topology_leakage_state_checkpoint_synthetic_and_real_cuda_tests",
    ),
}

_KNOWN_STRATEGIES = frozenset(
    strategy for family in _FAMILIES.values() for strategy in family.strategies
)
_KNOWN_CONTINUAL_METHODS = frozenset(
    method for family in _FAMILIES.values() for method in family.continual_methods
)


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if unknown:
            parts.append(f"unknown {sorted(unknown)}")
        raise MethodConfigValidationError(
            f"{label} fields do not match: {', '.join(parts)}."
        )


def _component(value: object, *, label: str) -> Tuple[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise MethodConfigValidationError(f"{label} must be a mapping.")
    _exact_fields(value, _COMPONENT_FIELDS, label=label)
    name = value["name"]
    parameters = value["parameters"]
    if not isinstance(name, str) or not name:
        raise MethodConfigValidationError(f"{label}.name must be a non-empty string.")
    if not isinstance(parameters, Mapping):
        raise MethodConfigValidationError(f"{label}.parameters must be a mapping.")
    return name, parameters


def _boolean_parameters(
    values: Mapping[str, Any], expected_fields: Tuple[str, ...], *, label: str
) -> Tuple[Tuple[str, bool], ...]:
    expected = frozenset(expected_fields)
    _exact_fields(values, expected, label=label)
    output = []
    for name in sorted(expected):
        value = values[name]
        if not isinstance(value, bool):
            raise MethodConfigValidationError(f"{label}.{name} must be boolean.")
        output.append((name, value))
    return tuple(output)


def _legacy_continual_parameters(
    values: Mapping[str, Any], *, continual_method: str, label: str
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    """Validate optional, fully frozen parameters for legacy CL adapters.

    Empty mappings remain valid for backward compatibility with the released
    legacy configs.  New benchmark configs may make the published defaults
    explicit so their values participate in the method-config digest.
    """

    expected_by_method = {
        "Bare": (),
        "EWC": ("regularization",),
        "LwF": ("regularization", "temperature"),
    }
    expected = expected_by_method.get(continual_method)
    if expected is None or not values:
        _exact_fields(values, frozenset(), label=label)
        return ()
    _exact_fields(values, frozenset(expected), label=label)
    output: list[tuple[str, MethodParameterValue]] = []
    for name in sorted(expected):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must be finite and positive."
            )
        output.append((name, float(value)))
    return tuple(output)


def _twp_parameters(
    values: Mapping[str, Any], expected_fields: Tuple[str, ...], *, label: str
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    """Validate fully explicit numeric TWP parameters for config hashing."""

    expected = frozenset(expected_fields)
    optional = frozenset({"topology_max_edges"})
    actual = frozenset(values)
    missing = expected - actual
    unexpected = actual - expected - optional
    if missing or unexpected:
        raise MethodConfigValidationError(
            f"{label} fields differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}."
        )
    output: list[tuple[str, MethodParameterValue]] = []
    for name in sorted(expected):
        value = values[name]
        if name == "middle_layer_index":
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise MethodConfigValidationError(
                    f"{label}.{name} must be null or a non-negative integer."
                )
            output.append((name, value))
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must be finite and non-negative."
            )
        output.append((name, float(value)))
    if "topology_max_edges" in values:
        value = values["topology_max_edges"]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise MethodConfigValidationError(
                f"{label}.topology_max_edges must be null or a positive integer."
            )
        output.append(("topology_max_edges", value))
    return tuple(output)


def _gem_parameters(
    values: Mapping[str, Any], expected_fields: Tuple[str, ...], *, label: str
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    """Validate the exact bounded-memory and QP controls for GEM."""

    expected = frozenset(expected_fields)
    _exact_fields(values, expected, label=label)
    memory_size = values["memory_size"]
    if (
        isinstance(memory_size, bool)
        or not isinstance(memory_size, int)
        or memory_size < 1
    ):
        raise MethodConfigValidationError(
            f"{label}.memory_size must be a positive integer."
        )
    normalized: Dict[str, MethodParameterValue] = {"memory_size": memory_size}
    for name in ("margin", "projection_epsilon", "violation_tolerance"):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or (name == "projection_epsilon" and float(value) == 0.0)
        ):
            qualifier = "positive" if name == "projection_epsilon" else "non-negative"
            raise MethodConfigValidationError(
                f"{label}.{name} must be finite and {qualifier}."
            )
        normalized[name] = float(value)
    return tuple((name, normalized[name]) for name in sorted(expected))


def _ssm_parameters(
    values: Mapping[str, Any], expected_fields: Tuple[str, ...], *, label: str
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    """Validate the fully explicit fixed SSM memory and sampling contract."""

    expected = frozenset(expected_fields)
    _exact_fields(values, expected, label=label)
    sampler_mode = values["sampler_mode"]
    if not isinstance(sampler_mode, str) or sampler_mode not in {"degree", "uniform"}:
        raise MethodConfigValidationError(
            f"{label}.sampler_mode must be 'degree' or 'uniform'."
        )
    raw_budgets = values["hop_budgets"]
    if not isinstance(raw_budgets, (list, tuple)) or len(raw_budgets) != 2:
        raise MethodConfigValidationError(
            f"{label}.hop_budgets must contain exactly two integers."
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_budgets
    ):
        raise MethodConfigValidationError(
            f"{label}.hop_budgets must contain exactly two integers."
        )
    hop_budgets = tuple(int(value) for value in raw_budgets)
    if hop_budgets not in {(10, 25), (0, 0)}:
        raise MethodConfigValidationError(
            f"{label}.hop_budgets must be [10, 25] or the [0, 0] diagnostic."
        )
    replay_weight = values["replay_weight"]
    if (
        isinstance(replay_weight, bool)
        or not isinstance(replay_weight, (int, float))
        or not math.isfinite(float(replay_weight))
        or float(replay_weight) != 1.0
    ):
        raise MethodConfigValidationError(f"{label}.replay_weight must equal 1.0.")
    replay_ceiling = values["replay_ceiling_bytes"]
    if (
        isinstance(replay_ceiling, bool)
        or not isinstance(replay_ceiling, int)
        or replay_ceiling != 16 * 1024 * 1024
    ):
        raise MethodConfigValidationError(
            f"{label}.replay_ceiling_bytes must equal 16777216."
        )
    stage_count = values["stage_count"]
    if (
        isinstance(stage_count, bool)
        or not isinstance(stage_count, int)
        or stage_count != 8
    ):
        raise MethodConfigValidationError(f"{label}.stage_count must equal 8.")
    normalized: Dict[str, MethodParameterValue] = {
        "hop_budgets": hop_budgets,
        "replay_ceiling_bytes": replay_ceiling,
        "replay_weight": float(replay_weight),
        "sampler_mode": sampler_mode,
        "stage_count": stage_count,
    }
    return tuple((name, normalized[name]) for name in sorted(expected))


def _cat_parameters(
    values: Mapping[str, Any], expected_fields: Tuple[str, ...], *, label: str
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    """Validate explicit CaT condensation, memory, and stage controls."""

    expected = frozenset(expected_fields)
    _exact_fields(values, expected, label=label)
    normalized: Dict[str, MethodParameterValue] = {}
    for name in (
        "condensation_steps",
        "memory_ceiling_bytes",
        "stage_count",
        "synthetic_nodes_per_class",
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MethodConfigValidationError(
                f"{label}.{name} must be a positive integer."
            )
        normalized[name] = int(value)
    learning_rate = values["condensation_lr"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0.0
    ):
        raise MethodConfigValidationError(
            f"{label}.condensation_lr must be finite and positive."
        )
    normalized["condensation_lr"] = float(learning_rate)
    initialization = values["feature_initialization"]
    if initialization != "random_choice":
        raise MethodConfigValidationError(
            f"{label}.feature_initialization must equal 'random_choice'."
        )
    normalized["feature_initialization"] = initialization
    return tuple((name, normalized[name]) for name in sorted(expected))


def _graphkeeper_parameters(
    values: Mapping[str, Any], expected_fields: Tuple[str, ...], *, label: str
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    expected = frozenset(expected_fields)
    _exact_fields(values, expected, label=label)
    normalized: Dict[str, MethodParameterValue] = {}
    for name in (
        "dbscan_min_samples",
        "max_cluster_nodes",
        "max_prototypes_per_domain",
        "pretrain_edges",
        "rank",
        "router_projection_dim",
        "stage_count",
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MethodConfigValidationError(f"{label}.{name} must be a positive integer.")
        normalized[name] = value
    for name in (
        "adapter_learning_rate",
        "dbscan_eps",
        "inter_weight",
        "intra_weight",
        "pretrain_weight",
        "ridge_lambda",
        "temperature",
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            raise MethodConfigValidationError(f"{label}.{name} must be finite and positive.")
        normalized[name] = float(value)
    for name in ("edge_drop", "feature_drop"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) < 1:
            raise MethodConfigValidationError(f"{label}.{name} must lie in [0,1).")
        normalized[name] = float(value)
    return tuple((name, normalized[name]) for name in sorted(expected))


def _power_parameters(
    values: Mapping[str, Any], expected_fields: Tuple[str, ...], *, label: str
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    """Validate POWER-UEFA fixed controls and beta/trajectory grid."""

    expected = frozenset(expected_fields)
    _exact_fields(values, expected, label=label)
    fixed_floats = {"alpha": 0.5, "coverage_threshold": 0.1}
    grid_floats = {
        "beta": {0.01, 0.05},
        "server_learning_rate": {1e-4, 1e-3, 1e-2},
        "trajectory_decay": {0.1, 0.3},
    }
    fixed_ints = {
        "replay_ceiling_bytes": 16 * 1024 * 1024,
        "reconstruction_steps": 300,
        "server_epochs": 3,
    }
    grid_ints = {"samples_per_class": {1, 5, 10, 20}}
    normalized: Dict[str, MethodParameterValue] = {}
    ablation_mode = values["ablation_mode"]
    if ablation_mode not in {"local_only", "server_only", "full"}:
        raise MethodConfigValidationError(
            f"{label}.ablation_mode must be local_only, server_only, or full."
        )
    normalized["ablation_mode"] = ablation_mode
    for name, expected_value in fixed_floats.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != expected_value
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must equal {expected_value}."
            )
        normalized[name] = float(value)
    for name, allowed in grid_floats.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) not in allowed
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must be one of {sorted(allowed)}."
            )
        normalized[name] = float(value)
    for name, expected_value in fixed_ints.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != expected_value
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must equal {expected_value}."
            )
        normalized[name] = value
    for name, allowed in grid_ints.items():
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
            raise MethodConfigValidationError(
                f"{label}.{name} must be one of {sorted(allowed)}."
            )
        normalized[name] = value
    return tuple((name, normalized[name]) for name in sorted(expected))


def _motion_parameters(
    values: Mapping[str, Any], expected_fields: Tuple[str, ...], *, label: str
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    """Validate the bounded clean-room MOTION NC-Class tuning contract."""

    expected = frozenset(expected_fields)
    _exact_fields(values, expected, label=label)
    fixed_ints = {
        "buffer_size": 200,
        "replay_ceiling_bytes": 16 * 1024 * 1024,
    }
    grid_ints = {"expert_select": {2, 3}}
    fixed_floats = {
        "pcb_max_ratio": 0.0001,
        "pcb_min_ratio": 0.0001,
        "pcb_ratio": 0.1,
    }
    grid_floats = {
        "node_reduction_rate": {0.25, 0.5},
        "replay_weight": {0.5, 1.0, 2.0},
        "similarity_threshold": {0.5, 0.7},
    }
    fixed_booleans = {
        "use_node_mahalanobis": True,
        "use_node_mmd": True,
        "use_node_positional": True,
    }
    normalized: Dict[str, MethodParameterValue] = {}
    for name, expected_value in fixed_ints.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != expected_value
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must equal {expected_value}."
            )
        normalized[name] = value
    for name, allowed in grid_ints.items():
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
            raise MethodConfigValidationError(
                f"{label}.{name} must be one of {sorted(allowed)}."
            )
        normalized[name] = value
    for name, expected_value in fixed_floats.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != expected_value
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must equal {expected_value}."
            )
        normalized[name] = float(value)
    for name, allowed in grid_floats.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) not in allowed
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must be one of {sorted(allowed)}."
            )
        normalized[name] = float(value)
    for name, expected_value in fixed_booleans.items():
        value = values[name]
        if not isinstance(value, bool) or value is not expected_value:
            raise MethodConfigValidationError(
                f"{label}.{name} must be {expected_value}."
            )
        normalized[name] = value
    raw_k_list = values["k_list"]
    if not isinstance(raw_k_list, (list, tuple)) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in raw_k_list
    ):
        raise MethodConfigValidationError(
            f"{label}.k_list must contain numeric values."
        )
    if (
        not all(math.isfinite(float(value)) for value in raw_k_list)
        or tuple(float(value) for value in raw_k_list) != (0.2, 0.4, 0.6, 0.8)
    ):
        raise MethodConfigValidationError(
            f"{label}.k_list must equal [0.2, 0.4, 0.6, 0.8]."
        )
    normalized["k_list"] = tuple(float(value) for value in raw_k_list)
    return tuple((name, normalized[name]) for name in sorted(expected))


def _fedfst_parameters(
    values: Mapping[str, Any], expected_fields: Tuple[str, ...], *, label: str
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    """Validate the explicit clean-room FedFST resolution profile."""

    expected = frozenset(expected_fields)
    _exact_fields(values, expected, label=label)
    positive_integers = {
        "client_nodes_per_class",
        "distillation_epochs",
        "generated_edges_per_node",
        "generator_epochs",
        "generator_rounds",
        "noise_dim",
        "server_max_edges_per_node",
        "server_nodes_per_class",
        "smoothing_hops",
        "topology_max_iterations",
    }
    nonnegative_integers = {"method_seed"}
    positive_floats = {
        "client_learning_rate",
        "distillation_learning_rate",
        "generator_learning_rate",
    }
    closed_open_unit = {"generator_dropout"}
    open_closed_unit = {"edge_reduction_ratio", "sampled_feature_fraction"}
    nonnegative_floats = {
        "client_weight_decay",
        "lambda_kl",
        "lambda_low",
        "topology_tolerance",
    }
    normalized: Dict[str, MethodParameterValue] = {}
    class_output_policy = values["class_il_output_training_policy"]
    if class_output_policy not in {"paper_full", "benchmark_seen"}:
        raise MethodConfigValidationError(
            f"{label}.class_il_output_training_policy must be 'paper_full' "
            "or 'benchmark_seen'."
        )
    normalized["class_il_output_training_policy"] = class_output_policy
    early_stop_policy = values["distillation_early_stop_policy"]
    if early_stop_policy not in {"none", "author_balance_crossing"}:
        raise MethodConfigValidationError(
            f"{label}.distillation_early_stop_policy must be 'none' or "
            "'author_balance_crossing'."
        )
    normalized["distillation_early_stop_policy"] = early_stop_policy
    edge_policy = values["server_initial_edge_policy"]
    if edge_policy not in {
        "paper_fixed",
        "target_scaled",
        "target_scaled_capped",
    }:
        raise MethodConfigValidationError(
            f"{label}.server_initial_edge_policy must be 'paper_fixed', "
            "'target_scaled', or 'target_scaled_capped'."
        )
    normalized["server_initial_edge_policy"] = edge_policy
    for name in sorted(positive_integers):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MethodConfigValidationError(
                f"{label}.{name} must be a positive integer."
            )
        normalized[name] = value
    for name in sorted(nonnegative_integers):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (name == "method_seed" and value >= 2**63)
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must be an integer in [0, 2**63)."
            )
        normalized[name] = value
    for name in sorted(positive_floats):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must be finite and positive."
            )
        normalized[name] = float(value)
    for name in sorted(closed_open_unit):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) < 1.0
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must lie in [0, 1)."
            )
        normalized[name] = float(value)
    for name in sorted(open_closed_unit):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must lie in (0, 1]."
            )
        normalized[name] = float(value)
    for name in sorted(nonnegative_floats):
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must be finite and non-negative."
            )
        normalized[name] = float(value)
    raw_checkpoints = values["distillation_validation_checkpoints"]
    if (
        not isinstance(raw_checkpoints, (list, tuple))
        or not raw_checkpoints
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_checkpoints
        )
    ):
        raise MethodConfigValidationError(
            f"{label}.distillation_validation_checkpoints must contain integers."
        )
    checkpoints = tuple(raw_checkpoints)
    if (
        checkpoints[0] != 0
        or checkpoints[-1] != normalized["distillation_epochs"]
        or any(left >= right for left, right in zip(checkpoints, checkpoints[1:]))
    ):
        raise MethodConfigValidationError(
            f"{label}.distillation_validation_checkpoints must be strictly "
            "increasing, begin at 0, and end at distillation_epochs."
        )
    normalized["distillation_validation_checkpoints"] = checkpoints
    if (
        normalized["server_max_edges_per_node"]
        < normalized["generated_edges_per_node"]
    ):
        raise MethodConfigValidationError(
            f"{label}.server_max_edges_per_node must be at least "
            "generated_edges_per_node."
        )
    if set(normalized) != expected:
        raise RuntimeError("FedFST parameter validator does not cover its schema.")
    return tuple((name, normalized[name]) for name in sorted(expected))


def _feddc_parameters(
    values: Mapping[str, Any], expected_fields: Tuple[str, ...], *, label: str
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    """Validate FedDC alpha grid and exact-degeneration switch."""

    expected = frozenset(expected_fields)
    _exact_fields(values, expected, label=label)
    alpha = values["alpha"]
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) not in {0.01, 0.1, 0.2, 0.0}
    ):
        raise MethodConfigValidationError(
            f"{label}.alpha must be one of [0.0, 0.01, 0.1, 0.2]."
        )
    drift_enabled = values["drift_enabled"]
    if not isinstance(drift_enabled, bool):
        raise MethodConfigValidationError(f"{label}.drift_enabled must be boolean.")
    return (("alpha", float(alpha)), ("drift_enabled", drift_enabled))


def _fedpub_parameters(
    values: Mapping[str, Any], expected_fields: Tuple[str, ...], *, label: str
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    """Validate FED-PUB's fixed proxy/mask controls and lambda2 grid."""

    expected = frozenset(expected_fields)
    _exact_fields(values, expected, label=label)
    normalized: Dict[str, MethodParameterValue] = {}
    fixed_floats = {"tau": 3.0, "lambda1": 1e-3}
    grid_floats = {"lambda2": {1e-3, 1e-1}}
    for name, expected_value in fixed_floats.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != expected_value
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must equal {expected_value}."
            )
        normalized[name] = float(value)
    for name, allowed in grid_floats.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) not in allowed
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must be one of {sorted(allowed)}."
            )
        normalized[name] = float(value)
    proxy_seed = values["proxy_seed"]
    if (
        isinstance(proxy_seed, bool)
        or not isinstance(proxy_seed, int)
        or proxy_seed < 0
    ):
        raise MethodConfigValidationError(
            f"{label}.proxy_seed must be a non-negative integer."
        )
    normalized["proxy_seed"] = proxy_seed
    return tuple((name, normalized[name]) for name in sorted(expected))


def _fedgta_parameters(
    values: Mapping[str, Any], expected_fields: Tuple[str, ...], *, label: str
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    """Validate FedGTA's declared NC validation grid and fixed controls."""

    expected = frozenset(expected_fields)
    _exact_fields(values, expected, label=label)
    fixed_integers = {"propagation_steps": 5}
    grid_integers = {"moment_order": {6, 13, 20}}
    fixed_floats = {"propagation_alpha": 0.5, "temperature": 20.0}
    grid_floats = {"similarity_threshold": {0.5, 0.65}}
    normalized: Dict[str, MethodParameterValue] = {}
    for name, expected_value in fixed_integers.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != expected_value
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must equal {expected_value}."
            )
        normalized[name] = value
    for name, allowed in grid_integers.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in allowed
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must be one of {sorted(allowed)}."
            )
        normalized[name] = value
    for name, expected_value in fixed_floats.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != expected_value
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must equal {expected_value}."
            )
        normalized[name] = float(value)
    for name, allowed in grid_floats.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) not in allowed
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must be one of {sorted(allowed)}."
            )
        normalized[name] = float(value)
    moment_type = values["moment_type"]
    if moment_type not in {"origin", "mean", "hybrid"}:
        raise MethodConfigValidationError(
            f"{label}.moment_type must be one of: origin, mean, hybrid."
        )
    normalized["moment_type"] = moment_type
    return tuple((name, normalized[name]) for name in sorted(expected))


def _dslr_parameters(
    values: Mapping[str, Any],
    expected_fields: Tuple[str, ...],
    *,
    label: str,
    diagnostic: bool,
) -> Tuple[Tuple[str, MethodParameterValue], ...]:
    """Validate the DSLR OGB-Arxiv full method or named component controls."""

    expected = frozenset(expected_fields)
    _exact_fields(values, expected, label=label)
    normalized: Dict[str, MethodParameterValue] = {}
    pair = (values["selection_mode"], values["structure_mode"])
    full_pair = ("coverage_diversity", "full")
    diagnostic_pairs = {
        ("mean_feature", "none"),
        ("coverage_diversity", "none"),
        ("mean_feature", "full"),
        ("coverage_diversity", "link_only"),
        ("coverage_diversity", "node_only"),
    }
    if diagnostic and pair not in diagnostic_pairs:
        raise MethodConfigValidationError(
            f"{label} does not name a declared DSLR component ablation."
        )
    if not diagnostic and pair != full_pair:
        raise MethodConfigValidationError(
            f"{label} may use the DSLR identity only for CD plus full structure."
        )
    normalized["selection_mode"] = pair[0]
    normalized["structure_mode"] = pair[1]
    allowed_floats = {
        "beta": {0.05, 0.1},
        "radius": {0.15, 0.2},
        "replay_fraction": {0.05},
        "structure_lambda": {0.5},
        "structure_learning_rate": {0.01},
        "tau": {0.8},
    }
    fixed_integers = {
        "candidate_k": 50,
        "replay_ceiling_bytes": 16 * 1024 * 1024,
        "structure_epochs": 99,
        "structure_heads": 4,
        "structure_hidden_dim": 64,
        "top_n": 5,
    }
    for name, allowed in allowed_floats.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) not in allowed
        ):
            rendered = sorted(allowed)
            raise MethodConfigValidationError(
                f"{label}.{name} must be one of {rendered}."
            )
        normalized[name] = float(value)
    for name, expected_value in fixed_integers.items():
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != expected_value
        ):
            raise MethodConfigValidationError(
                f"{label}.{name} must equal {expected_value}."
            )
        normalized[name] = value
    if values["undirected"] is not True:
        raise MethodConfigValidationError(f"{label}.undirected must be true.")
    normalized["undirected"] = True
    return tuple((name, normalized[name]) for name in sorted(expected))


def _normalize_scenario(
    problem_type: str | None, incremental_setting: str | None
) -> Tuple[str | None, str | None]:
    if (problem_type is None) != (incremental_setting is None):
        raise MethodConfigValidationError(
            "problem_type and incremental_setting must be supplied together."
        )
    if problem_type is None:
        return None, None
    problem = str(problem_type).upper()
    incremental = str(incremental_setting).lower()
    if (problem, incremental) not in _SCENARIOS:
        raise MethodConfigValidationError(
            f"Unknown UEFA scenario identity: {problem}-{incremental}."
        )
    return problem, incremental


def _legacy_resolution(
    *,
    name: str,
    strategy_name: str,
    continual_method_name: str,
    strategy_parameters: Tuple[Tuple[str, MethodParameterValue], ...],
    continual_parameters: Tuple[Tuple[str, MethodParameterValue], ...],
    problem_type: str | None,
    incremental_setting: str | None,
) -> ResolvedMethodConfig:
    if problem_type is None:
        return ResolvedMethodConfig(
            METHOD_CONFIG_SCHEMA,
            METHOD_CONFIG_VERSION,
            name,
            strategy_name,
            continual_method_name,
            strategy_parameters,
            continual_parameters,
            "runnable_scenario_unresolved",
            "registry_resolved",
            True,
            False,
            "Scenario identity is required before benchmark eligibility can be resolved.",
            "legacy_v1_evidence",
            None,
            None,
        )
    registry = MethodRegistry()
    entry = next(
        (
            candidate
            for candidate in registry.compatibility()
            if candidate.algorithm == continual_method_name
            and candidate.server_strategy == strategy_name
            and candidate.problem_type == problem_type
            and candidate.incremental_setting == incremental_setting
        ),
        None,
    )
    if entry is None:
        raise MethodConfigValidationError(
            "The legacy compatibility registry has no matching identity row."
        )
    return ResolvedMethodConfig(
        METHOD_CONFIG_SCHEMA,
        METHOD_CONFIG_VERSION,
        name,
        strategy_name,
        continual_method_name,
        strategy_parameters,
        continual_parameters,
        entry.support_status,
        entry.scientific_fidelity,
        entry.runnable,
        entry.benchmark_eligible,
        entry.reason,
        entry.test_coverage,
        problem_type,
        incremental_setting,
    )


def resolve_method_config(
    config: Mapping[str, Any],
    *,
    expected_strategy: str | None = None,
    expected_continual_method: str | None = None,
    problem_type: str | None = None,
    incremental_setting: str | None = None,
) -> ResolvedMethodConfig:
    """Validate canonical identities and resolve support without running code."""

    if not isinstance(config, Mapping):
        raise MethodConfigValidationError("v2 method config must be a mapping.")
    _exact_fields(config, _TOP_LEVEL_FIELDS, label="v2 method config")
    if config["schema"] != METHOD_CONFIG_SCHEMA:
        raise MethodConfigValidationError(
            f"method config schema must be {METHOD_CONFIG_SCHEMA!r}."
        )
    version = config["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise MethodConfigValidationError("method config version must be an integer.")
    if version != METHOD_CONFIG_VERSION:
        raise MethodConfigValidationError(
            f"method config version must be {METHOD_CONFIG_VERSION}."
        )
    name = config["name"]
    if not isinstance(name, str) or not name:
        raise MethodConfigValidationError(
            "method config name must be a non-empty string."
        )
    if name not in _FAMILIES:
        raise MethodConfigValidationError(f"Unknown v2 method config name: {name!r}.")
    family = _FAMILIES[name]
    strategy_name, raw_strategy_parameters = _component(
        config["strategy"], label="strategy"
    )
    continual_name, raw_continual_parameters = _component(
        config["continual_method"], label="continual_method"
    )
    if strategy_name not in _KNOWN_STRATEGIES:
        raise MethodConfigValidationError(
            f"Unknown v2 strategy name: {strategy_name!r}."
        )
    if continual_name not in _KNOWN_CONTINUAL_METHODS:
        raise MethodConfigValidationError(
            f"Unknown v2 continual-method name: {continual_name!r}."
        )
    if strategy_name not in family.strategies:
        raise MethodConfigValidationError(
            f"Config family {name!r} does not implement strategy {strategy_name!r}."
        )
    if continual_name not in family.continual_methods:
        raise MethodConfigValidationError(
            f"Config family {name!r} does not implement continual method "
            f"{continual_name!r}."
        )
    if expected_strategy is not None and strategy_name != expected_strategy:
        raise MethodConfigValidationError(
            "Configured strategy identity does not match the coordinator: "
            f"{strategy_name!r} != {expected_strategy!r}."
        )
    if (
        expected_continual_method is not None
        and continual_name != expected_continual_method
    ):
        raise MethodConfigValidationError(
            "Configured continual-method identity does not match the coordinator: "
            f"{continual_name!r} != {expected_continual_method!r}."
        )
    if name == "fedfst_uefa_v1":
        strategy_parameters = _fedfst_parameters(
            raw_strategy_parameters,
            family.strategy_parameter_fields,
            label="strategy.parameters",
        )
    elif name == "fedgta_uefa_v1":
        strategy_parameters = _fedgta_parameters(
            raw_strategy_parameters,
            family.strategy_parameter_fields,
            label="strategy.parameters",
        )
    elif name == "fed_pub_uefa_v1":
        strategy_parameters = _fedpub_parameters(
            raw_strategy_parameters,
            family.strategy_parameter_fields,
            label="strategy.parameters",
        )
    elif name == "feddc_uefa_v1":
        strategy_parameters = _feddc_parameters(
            raw_strategy_parameters,
            family.strategy_parameter_fields,
            label="strategy.parameters",
        )
    elif name == "power_uefa_v1":
        strategy_parameters = _power_parameters(
            raw_strategy_parameters,
            family.strategy_parameter_fields,
            label="strategy.parameters",
        )
    elif name == "motion_uefa_v1":
        strategy_parameters = _motion_parameters(
            raw_strategy_parameters,
            family.strategy_parameter_fields,
            label="strategy.parameters",
        )
    else:
        strategy_parameters = _boolean_parameters(
            raw_strategy_parameters,
            family.strategy_parameter_fields,
            label="strategy.parameters",
        )
    if name == "legacy_adapter_v1":
        continual_parameters = _legacy_continual_parameters(
            raw_continual_parameters,
            continual_method=continual_name,
            label="continual_method.parameters",
        )
    elif name == "gem_uefa_v1":
        continual_parameters = _gem_parameters(
            raw_continual_parameters,
            family.continual_parameter_fields,
            label="continual_method.parameters",
        )
    elif name == "twp_uefa_v1":
        continual_parameters = _twp_parameters(
            raw_continual_parameters,
            family.continual_parameter_fields,
            label="continual_method.parameters",
        )
    elif name == "ssm_uefa_v1":
        continual_parameters = _ssm_parameters(
            raw_continual_parameters,
            family.continual_parameter_fields,
            label="continual_method.parameters",
        )
    elif name == "cat_uefa_v1":
        continual_parameters = _cat_parameters(
            raw_continual_parameters,
            family.continual_parameter_fields,
            label="continual_method.parameters",
        )
    elif name == "graphkeeper_uefa_v1":
        continual_parameters = _graphkeeper_parameters(
            raw_continual_parameters,
            family.continual_parameter_fields,
            label="continual_method.parameters",
        )
    elif name in {
        "dslr_uefa_v1",
        "dslr_normalized_v1",
        "dslr_diagnostic_v1",
    }:
        continual_parameters = _dslr_parameters(
            raw_continual_parameters,
            family.continual_parameter_fields,
            label="continual_method.parameters",
            diagnostic=name == "dslr_diagnostic_v1",
        )
    elif name == "fedgta_uefa_v1" and continual_name == "TWP":
        continual_parameters = _twp_parameters(
            raw_continual_parameters,
            _FAMILIES["twp_uefa_v1"].continual_parameter_fields,
            label="continual_method.parameters",
        )
    elif name == "fedgta_uefa_v1" and continual_name == "SSM":
        continual_parameters = _ssm_parameters(
            raw_continual_parameters,
            _FAMILIES["ssm_uefa_v1"].continual_parameter_fields,
            label="continual_method.parameters",
        )
    elif name == "fedgta_uefa_v1" and continual_name == "DSLR":
        continual_parameters = _dslr_parameters(
            raw_continual_parameters,
            _FAMILIES["dslr_uefa_v1"].continual_parameter_fields,
            label="continual_method.parameters",
            diagnostic=False,
        )
    elif name == "fed_pub_uefa_v1" and continual_name == "TWP":
        continual_parameters = _twp_parameters(
            raw_continual_parameters,
            _FAMILIES["twp_uefa_v1"].continual_parameter_fields,
            label="continual_method.parameters",
        )
    elif name == "fed_pub_uefa_v1" and continual_name == "DSLR":
        continual_parameters = _dslr_parameters(
            raw_continual_parameters,
            _FAMILIES["dslr_uefa_v1"].continual_parameter_fields,
            label="continual_method.parameters",
            diagnostic=False,
        )
    else:
        continual_parameters = _boolean_parameters(
            raw_continual_parameters,
            family.continual_parameter_fields,
            label="continual_method.parameters",
        )
    problem, incremental = _normalize_scenario(problem_type, incremental_setting)
    if name == "legacy_adapter_v1":
        return _legacy_resolution(
            name=name,
            strategy_name=strategy_name,
            continual_method_name=continual_name,
            strategy_parameters=strategy_parameters,
            continual_parameters=continual_parameters,
            problem_type=problem,
            incremental_setting=incremental,
        )

    runnable = family.implemented
    support_status = family.support_status
    benchmark_eligible = False
    reason = family.reason
    coverage = family.test_coverage
    scientific_fidelity = family.scientific_fidelity
    if name == "scaffold_uefa_adam_v1":
        values = dict(strategy_parameters)
        if not values["correction_enabled"] or not values["control_updates_enabled"]:
            support_status = "diagnostic_only_disabled_mechanism"
            reason = (
                "Disabling either SCAFFOLD correction or control updates is a "
                "diagnostic ablation, not the benchmark method."
            )
            coverage = "degeneration_and_state_unit_tests"
        if problem is not None and problem != "NC":
            runnable = False
            support_status = "unsupported_scientific"
            reason = "Initial SCAFFOLD UEFA support is restricted to NC scenarios."
            coverage = "fail_closed_compatibility_test"
    elif name == "gem_uefa_v1":
        if problem is not None and problem not in {"NC", "LC"}:
            runnable = False
            support_status = "unsupported_scientific"
            reason = "GEM UEFA support is restricted to NC and LC scenarios."
            coverage = "fail_closed_problem_compatibility_test"
    elif name == "twp_uefa_v1":
        values = dict(continual_parameters)
        if problem is not None and problem not in {"NC", "LC"}:
            runnable = False
            support_status = "unsupported_scientific"
            reason = "TWP UEFA support is restricted to NC and LC scenarios."
            coverage = "fail_closed_problem_compatibility_test"
        elif (
            problem == "NC" and incremental == "task" and strategy_name == "local_only"
        ):
            support_status = "implemented_unverified_fidelity_candidate"
            scientific_fidelity = "faithful_supported_candidate"
            reason = (
                "LocalOnly NC-Task is the paper-fidelity anchor, pending real "
                "seed-0 and five-seed promotion evidence."
            )
        if runnable and values["lambda_l"] == 0.0 and values["lambda_t"] == 0.0:
            support_status = "diagnostic_only_disabled_mechanism"
            reason = (
                "All-zero TWP importance coefficients are a Bare degeneration oracle."
            )
            coverage = "exact_zero_coefficient_degeneration_test"
        elif runnable and values["lambda_t"] == 0.0:
            support_status = "diagnostic_only_loss_importance"
            reason = "lambda_t=0 is the declared loss-importance-only TWP oracle."
            coverage = "topology_capability_not_called_test"
        elif runnable and values.get("topology_max_edges") is not None:
            support_status = "implemented_unverified_memory_bounded_topology"
            scientific_fidelity = "memory_bounded_topology_approximation"
            reason = (
                "TWP retains its loss and topology mechanisms while applying a "
                "deterministic local topology-edge budget for large-graph memory safety."
            )
            coverage = "twp_topology_edge_budget_unit_and_gpu_smoke"
    elif name == "ssm_uefa_v1":
        values = dict(continual_parameters)
        if problem is not None and not (
            (problem == "NC" and incremental in {"class", "task"})
            or (problem == "LC" and incremental in {"class", "task"})
        ):
            runnable = False
            support_status = "unsupported_scientific"
            reason = (
                "SSM support is restricted to NC/LC Class/Task-IL; "
                "multi-label domain replay is not implemented."
            )
            coverage = "fail_closed_problem_and_incremental_compatibility_tests"
        elif tuple(values["hop_budgets"]) == (0, 0):
            support_status = "diagnostic_only_node_only_ablation"
            reason = "[0, 0] is the declared node-only matched SSM diagnostic."
            coverage = "node_only_topology_ablation_test"
        elif (
            problem == "LC"
            and incremental == "task"
            and values == {
                "hop_budgets": (10, 25),
                "replay_ceiling_bytes": 16 * 1024 * 1024,
                "replay_weight": 1.0,
                "sampler_mode": "degree",
                "stage_count": 8,
            }
        ):
            support_status = "supported"
            scientific_fidelity = "task_aware_mechanism_adaptation"
            benchmark_eligible = True
            reason = (
                "SSM-LC checkpoints an immutable source-task output mask in every "
                "directed-edge replay record and applies grouped source-head losses."
            )
            coverage = (
                "task_head_oracle_and_bitcoin_lc_task_seed2_full_50x1_"
                "localonly_fedavg_fedprox_cuda_matrix"
            )
        elif problem in {"NC", "LC"} and incremental == "task":
            support_status = "implemented_unverified_task_adaptation"
            scientific_fidelity = "task_aware_mechanism_adaptation"
            reason = "SSM stores and applies the source global-task class mask per replay record."
            coverage = "task_mask_behavioral_oracle_and_real_bitcoin_lc_task_cuda_smoke"
    elif name == "cat_uefa_v1":
        values = dict(continual_parameters)
        if problem is not None and not (
            (problem == "NC" and incremental in {"class", "task"})
            or (problem == "LC" and incremental in {"class", "task"})
        ):
            runnable = False
            support_status = "unsupported_scientific"
            reason = (
                "CaT support is restricted to NC/LC Class/Task-IL; "
                "multi-label domain condensation is not implemented."
            )
            coverage = "fail_closed_problem_and_incremental_compatibility_tests"
        elif (
            problem == "LC"
            and incremental == "task"
            and values == {
                "condensation_lr": 0.001,
                "condensation_steps": 32,
                "feature_initialization": "random_choice",
                "memory_ceiling_bytes": 16 * 1024 * 1024,
                "stage_count": 8,
                "synthetic_nodes_per_class": 2,
            }
        ):
            support_status = "supported"
            scientific_fidelity = "task_aware_mechanism_adaptation"
            benchmark_eligible = True
            reason = (
                "CaT-LC checkpoints an immutable source-task output mask in every "
                "condensed edge graph and trains grouped memories through source heads."
            )
            coverage = (
                "task_head_oracle_and_bitcoin_lc_task_seed2_full_50x1_"
                "localonly_fedavg_fedprox_cuda_matrix"
            )
        elif problem in {"NC", "LC"} and incremental == "task":
            support_status = "implemented_unverified_task_adaptation"
            scientific_fidelity = "task_aware_mechanism_adaptation"
            reason = "CaT stores the source task head inside every condensed graph record."
            coverage = "task_mask_behavioral_oracle_and_real_bitcoin_lc_task_cuda_smoke"
    elif name == "graphkeeper_uefa_v1":
        expected_problem = "LP" if continual_name == "GraphKeeper-LP" else "NC"
        if problem is not None and (
            problem != expected_problem or incremental != "domain"
        ):
            runnable = False
            support_status = "unsupported_scientific"
            reason = (
                f"{continual_name} support is restricted to "
                f"{expected_problem} Domain-IL."
            )
            coverage = "fail_closed_problem_and_incremental_compatibility_tests"
        elif continual_name == "GraphKeeper-LP":
            support_status = "implemented_unverified"
            scientific_fidelity = "mechanism_adaptation"
            reason = (
                "GraphKeeper-LP adapts layer-wise graph-LoRA experts, hidden-domain "
                "routing, DBSCAN edge prototypes, and recursive ridge preservation "
                "to strict-local LP endpoint-product representations. Federated "
                "variants aggregate only the shared graph backbone while all "
                "GraphKeeper continual state remains client-private; real-data "
                "promotion evidence is pending."
            )
            coverage = (
                "lp_edge_oracle_checkpoint_federated_state_boundary_fail_closed_"
                "and_synthetic_end_to_end_tests"
            )
    elif name in {
        "dslr_uefa_v1",
        "dslr_normalized_v1",
        "dslr_diagnostic_v1",
    }:
        if problem is not None and (problem != "NC" or incremental not in {"class", "task"}):
            runnable = False
            support_status = "unsupported_scientific"
            reason = (
                "DSLR support is restricted to NC Class/Task-IL; NC-Domain "
                "multi-label structure learning is not implemented."
            )
            coverage = "fail_closed_problem_and_incremental_compatibility_tests"
        elif problem == "NC" and incremental == "task":
            support_status = "implemented_unverified_task_adaptation"
            scientific_fidelity = "task_aware_mechanism_adaptation"
            reason = "DSLR groups structure and downstream replay by source global-task head."
            coverage = "task_mask_behavioral_oracle_and_real_data_smoke_pending"
        elif name == "dslr_normalized_v1":
            reason = (
                "DSLR-Normalized is an opt-in benchmark adaptation matching the "
                "authors' mean-reduced executable link loss; it is not the literal "
                "summed Equation-6 DSLR identity."
            )
            coverage = "reduction_scale_behavioral_oracle_real_data_smoke_pending"
    elif name == "fedgta_uefa_v1":
        if problem is not None and (problem != "NC" or incremental == "domain"):
            runnable = False
            support_status = "unsupported_scientific"
            reason = (
                "Initial FedGTA support is restricted to NC-Task and NC-Class; "
                "the paper's single-label probability moments do not support "
                "the NC-Domain multi-label setting."
            )
            coverage = "fail_closed_problem_and_domain_compatibility_tests"
        elif continual_name == "TWP":
            if problem == "NC" and incremental == "class":
                support_status = "implemented_unverified_interaction_pilot"
                scientific_fidelity = "mechanism_adaptation_interaction_pilot"
                reason = (
                    "FedGTA+TWP is enabled only as the declared NC-Class "
                    "one-seed interaction pilot; it is not benchmark evidence."
                )
                coverage = "fedgta_twp_pilot_policy_and_seed0_gpu_smoke"
            else:
                runnable = False
                support_status = "unsupported_composition"
                reason = (
                    "FedGTA+TWP pilot support is restricted to NC-Class; "
                    "NC-Task fidelity and NC-Domain multi-label interactions remain closed."
                )
                coverage = "fail_closed_composition_compatibility_test"
        elif continual_name == "SSM":
            if problem == "NC" and incremental == "class":
                support_status = "implemented_unverified_interaction_pilot"
                scientific_fidelity = "mechanism_adaptation_interaction_pilot"
                reason = (
                    "FedGTA+SSM is enabled only as the declared NC-Class "
                    "one-seed interaction pilot; it is not benchmark evidence."
                )
                coverage = "fedgta_ssm_pilot_policy_and_seed0_gpu_smoke"
            else:
                runnable = False
                support_status = "unsupported_composition"
                reason = (
                    "FedGTA+SSM pilot support is restricted to NC-Class; "
                    "task-tagged and multi-label replay compositions remain closed."
                )
                coverage = "fail_closed_composition_compatibility_test"
        elif continual_name == "DSLR":
            if problem == "NC" and incremental == "class":
                support_status = "implemented_unverified_interaction_pilot"
                scientific_fidelity = "mechanism_adaptation_interaction_pilot"
                reason = (
                    "FedGTA+DSLR is enabled only as the declared NC-Class "
                    "one-seed interaction pilot; it is not benchmark evidence."
                )
                coverage = "fedgta_dslr_pilot_policy_and_seed0_gpu_smoke"
            else:
                runnable = False
                support_status = "unsupported_composition"
                reason = (
                    "FedGTA+DSLR pilot support is restricted to NC-Class; "
                    "task-tagged and multi-label structure interactions remain closed."
                )
                coverage = "fail_closed_composition_compatibility_test"
        elif continual_name != "Bare":
            runnable = False
            support_status = "unsupported_composition"
            reason = (
                "FedGTA continual-method compositions remain unavailable until "
                "their joint state, leakage, and real-data gates are passed."
            )
            coverage = "fail_closed_composition_compatibility_test"
    elif name == "fed_pub_uefa_v1":
        if problem is not None and (problem != "NC" or incremental == "domain"):
            runnable = False
            support_status = "unsupported_scientific"
            reason = (
                "Initial FED-PUB support is restricted to NC-Task and NC-Class; "
                "NC-Domain multi-label personalized masks are not implemented."
            )
            coverage = "fail_closed_problem_and_domain_compatibility_tests"
        elif continual_name == "TWP":
            if problem == "NC" and incremental == "class":
                support_status = "implemented_unverified_interaction_pilot"
                scientific_fidelity = "mechanism_adaptation_interaction_pilot"
                reason = (
                    "FED-PUB+TWP is enabled only as the declared NC-Class "
                    "one-seed interaction pilot; it is not benchmark evidence."
                )
                coverage = "fedpub_twp_pilot_policy_and_seed0_gpu_smoke"
            else:
                runnable = False
                support_status = "unsupported_composition"
                reason = (
                    "FED-PUB+TWP pilot support is restricted to NC-Class; "
                    "NC-Task fidelity and NC-Domain multi-label mask interactions remain closed."
                )
                coverage = "fail_closed_composition_compatibility_test"
        elif continual_name == "DSLR":
            if problem == "NC" and incremental == "class":
                support_status = "implemented_unverified_interaction_pilot"
                scientific_fidelity = "mechanism_adaptation_interaction_pilot"
                reason = (
                    "FED-PUB+DSLR is enabled only as the declared NC-Class "
                    "one-seed interaction pilot; it is not benchmark evidence."
                )
                coverage = "fedpub_dslr_pilot_policy_and_seed0_gpu_smoke"
            else:
                runnable = False
                support_status = "unsupported_composition"
                reason = (
                    "FED-PUB+DSLR pilot support is restricted to NC-Class; "
                    "task-tagged and multi-label structure/mask interactions remain closed."
                )
                coverage = "fail_closed_composition_compatibility_test"
        elif continual_name != "Bare":
            runnable = False
            support_status = "unsupported_composition"
            reason = (
                "FED-PUB continual-method compositions remain unavailable until "
                "their joint mask, leakage, and real-data gates are passed."
            )
            coverage = "fail_closed_composition_compatibility_test"
    elif name == "feddc_uefa_v1":
        if problem is not None and problem != "NC":
            runnable = False
            support_status = "unsupported_scientific"
            reason = "Initial FedDC support is restricted to NC scenarios."
            coverage = "fail_closed_problem_compatibility_tests"
        values = dict(strategy_parameters)
        if values.get("alpha") == 0.0 and values.get("drift_enabled") is False:
            support_status = "diagnostic_only_disabled_mechanism"
            reason = "FedDC additions are disabled; this is the exact FedAvg degeneration oracle."
            coverage = "disabled_drift_degeneration_test"
    elif name == "power_uefa_v1":
        if problem is not None and (problem != "NC" or incremental not in {"class", "task"}):
            runnable = False
            support_status = "unsupported_scientific"
            reason = "POWER-UEFA support is restricted to NC Class/Task-IL."
            coverage = "fail_closed_problem_and_incremental_compatibility_tests"
        elif problem == "NC" and incremental == "task":
            support_status = "implemented_unverified_task_adaptation"
            scientific_fidelity = "task_aware_mechanism_adaptation"
            reason = "POWER masks local replay and server trajectory transfer by source task head."
            coverage = "task_mask_behavioral_oracle_and_real_data_smoke_pending"
    elif name == "motion_uefa_v1":
        if problem is not None and (problem != "NC" or incremental not in {"class", "task"}):
            runnable = False
            support_status = "unsupported_scientific"
            reason = "MOTION support is restricted to NC Class/Task-IL."
            coverage = "fail_closed_problem_and_incremental_compatibility_tests"
        elif problem == "NC" and incremental == "task":
            support_status = "implemented_unverified_task_adaptation"
            scientific_fidelity = "task_aware_mechanism_adaptation"
            reason = "MOTION groups coarsened replay nodes by their unique source task head."
            coverage = "task_mask_behavioral_oracle_and_real_data_smoke_pending"
    elif name == "fedfst_uefa_v1":
        values = dict(strategy_parameters)
        if problem is not None and (
            problem != "NC" or incremental not in {"class", "task"}
        ):
            runnable = False
            support_status = "unsupported_scientific"
            reason = (
                "FedFST support is restricted to single-label "
                "NC-Class/Task streams; LP, LC, and domain variants are unsupported."
            )
            coverage = "fail_closed_problem_and_incremental_compatibility_tests"
        elif values["lambda_kl"] == 0.0 or values["lambda_low"] == 0.0:
            support_status = "diagnostic_only_disabled_component"
            reason = (
                "lambda_kl=0 or lambda_low=0 disables a declared FedFST "
                "component and is a diagnostic ablation, not the full method."
            )
            coverage = "component_ablation_identity_and_equation_tests"
        elif (
            problem == "NC"
            and incremental in {"class", "task"}
            and values["class_il_output_training_policy"] == "paper_full"
            and values["server_initial_edge_policy"] == "target_scaled_capped"
            and values["server_max_edges_per_node"] == 14
            and values["client_learning_rate"] == 0.01
            and values["client_weight_decay"] == 0.0
            and values["distillation_early_stop_policy"]
            == (
                "none"
                if incremental == "class"
                else "author_balance_crossing"
            )
        ):
            support_status = "supported"
            scientific_fidelity = "paper_equation_uefa_adaptation"
            benchmark_eligible = True
            if incremental == "task":
                reason = (
                    "FedFST uses only prior immutable task heads and block-diagonal "
                    "synthetic topology for HHKR/HLST under client-ordered NC-Task."
                )
                coverage = (
                    "task_mask_block_diagonal_behavioral_oracle_and_real_cuda_smoke"
                )
        else:
            support_status = "diagnostic_only_noncanonical_profile"
            reason = (
                "FedFST benchmark eligibility is restricted to the frozen "
                "scope-specific output, optimizer, topology, and selector policies."
            )
            coverage = "canonical_profile_identity_tests"
    return ResolvedMethodConfig(
        METHOD_CONFIG_SCHEMA,
        METHOD_CONFIG_VERSION,
        name,
        strategy_name,
        continual_name,
        strategy_parameters,
        continual_parameters,
        support_status,
        scientific_fidelity,
        runnable,
        benchmark_eligible,
        reason,
        coverage,
        problem,
        incremental,
    )


def validate_method_config(
    config: Mapping[str, Any],
    *,
    expected_strategy: str | None = None,
    expected_continual_method: str | None = None,
    problem_type: str | None = None,
    incremental_setting: str | None = None,
) -> ResolvedMethodConfig:
    """Resolve a v2 config and reject known-but-unsupported combinations."""

    resolved = resolve_method_config(
        config,
        expected_strategy=expected_strategy,
        expected_continual_method=expected_continual_method,
        problem_type=problem_type,
        incremental_setting=incremental_setting,
    )
    if not resolved.runnable:
        raise UnsupportedMethodConfigError(resolved)
    return resolved


def known_method_config_names() -> Tuple[str, ...]:
    """Return the complete versioned catalog, including unsupported rows."""

    return tuple(sorted(_FAMILIES))
