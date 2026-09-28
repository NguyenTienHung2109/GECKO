"""Truthful method registry for runnable and audited BeGin algorithms."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Dict
from typing import Tuple
from typing import Type

from gecko.algorithms.base import ClientContinualAlgorithm
from gecko.algorithms.continual.cat.node import CaTAlgorithm
from gecko.algorithms.continual.cat.link import CaTLCAlgorithm
from gecko.algorithms.continual.dslr.algorithm import DSLRAlgorithm
from gecko.algorithms.continual.ergnn import ERGNNAlgorithm
from gecko.algorithms.continual.gem import GEMAlgorithm
from gecko.algorithms.continual.bare import BareAlgorithm
from gecko.algorithms.continual.ewc import EWCAlgorithm
from gecko.algorithms.continual.generic import GenericImportanceRegularizationAlgorithm
from gecko.algorithms.continual.generic import GenericReplayAlgorithm
from gecko.algorithms.continual.generic import GenericWeightIsolationAlgorithm
from gecko.algorithms.continual.lwf import LwFAlgorithm
from gecko.algorithms.continual.lwf import LwFClassILAlgorithm
from gecko.algorithms.continual.mas import MASAlgorithm
from gecko.algorithms.continual.ssm.node import SSMAlgorithm
from gecko.algorithms.continual.ssm.link import SSMLCAlgorithm
from gecko.algorithms.continual.graphkeeper.algorithm import GraphKeeperAlgorithm
from gecko.algorithms.continual.graphkeeper.lp import GraphKeeperLPAlgorithm
from gecko.algorithms.continual.twp import TWPAlgorithm


ORIGINAL_ALGORITHMS = (
    "Bare",
    "LwF",
    "EWC",
    "MAS",
    "GEM",
    "TWP",
    "ERGNN",
    "CGNN",
    "PackNet",
    "Piggyback",
    "HAT",
    "PIGNN",
    "CaT",
)
GENERIC_ALGORITHMS = (
    "generic_replay",
    "generic_importance_regularization",
    "generic_weight_isolation",
)
SERVER_STRATEGIES = (
    "local_only",
    "fedavg",
    "fedprox",
    "centralized_shard_oracle",
)
SCENARIOS = (
    ("NC", "task"),
    ("NC", "class"),
    ("NC", "domain"),
    ("LC", "task"),
    ("LC", "class"),
    ("LC", "domain"),
    ("LP", "domain"),
)


@dataclass(frozen=True)
class CompatibilityEntry:
    algorithm: str
    server_strategy: str
    problem_type: str
    incremental_setting: str
    model_family: str
    support_status: str
    reason: str
    test_coverage: str
    runnable: bool
    benchmark_eligible: bool
    scientific_fidelity: str


class MethodRegistry:
    """Separate runnable mechanisms from original-name audit records."""

    def __init__(self) -> None:
        def create_ssm(**kwargs):
            algorithm = SSMLCAlgorithm if kwargs.get("problem_type") == "LC" else SSMAlgorithm
            return algorithm(**kwargs)

        def create_cat(**kwargs):
            algorithm = CaTLCAlgorithm if kwargs.get("problem_type") == "LC" else CaTAlgorithm
            return algorithm(**kwargs)

        def create_dslr_normalized(**kwargs):
            return DSLRAlgorithm(link_reduction="mean", **kwargs)

        self._factories: Dict[str, Type[ClientContinualAlgorithm]] = {
            "Bare": BareAlgorithm,
            "LwF": LwFAlgorithm,
            "EWC": EWCAlgorithm,
            "MAS": MASAlgorithm,
            "ERGNN": ERGNNAlgorithm,
            "generic_replay": GenericReplayAlgorithm,
            "generic_importance_regularization": GenericImportanceRegularizationAlgorithm,
            "generic_weight_isolation": GenericWeightIsolationAlgorithm,
        }
        self._explicit_v2_factories = {
            "gem_uefa_v1": {"GEM": GEMAlgorithm},
            "twp_uefa_v1": {"TWP": TWPAlgorithm},
            "ssm_uefa_v1": {"SSM": create_ssm},
            "cat_uefa_v1": {"CaT": create_cat},
            "graphkeeper_uefa_v1": {
                "GraphKeeper": GraphKeeperAlgorithm,
                "GraphKeeper-LP": GraphKeeperLPAlgorithm,
            },
            "fedgta_uefa_v1": {
                "DSLR": DSLRAlgorithm,
                "SSM": SSMAlgorithm,
                "TWP": TWPAlgorithm,
            },
            "fed_pub_uefa_v1": {"DSLR": DSLRAlgorithm, "TWP": TWPAlgorithm},
            "dslr_uefa_v1": {"DSLR": DSLRAlgorithm},
            "dslr_normalized_v1": {
                "DSLR-Normalized": create_dslr_normalized,
            },
            "dslr_diagnostic_v1": {"DSLRDiagnostic": DSLRAlgorithm},
        }
        self._compatibility = self._build_compatibility()

    def _build_compatibility(self) -> Tuple[CompatibilityEntry, ...]:
        entries = []
        unfaithful = set(ORIGINAL_ALGORITHMS) - {"Bare", "LwF", "EWC", "MAS", "ERGNN"}
        for algorithm in ORIGINAL_ALGORITHMS:
            for strategy in SERVER_STRATEGIES:
                for problem, incremental in SCENARIOS:
                    if algorithm in unfaithful:
                        status = "unsupported_original_implementation"
                        reason = (
                            "No faithful UEFA adapter is implemented. The former generic "
                            "placeholder was removed from this scientific name."
                        )
                        coverage = "registry_audit_only"
                        runnable = False
                        eligible = False
                        fidelity = "false"
                    elif algorithm == "ERGNN" and (
                        problem != "NC" or incremental == "domain"
                    ):
                        status = "unsupported_scientific"
                        reason = (
                            "Original BeGin ERGNN requires scalar node classes. UEFA LC/LP "
                            "have no ERGNN mechanism, and NC-Domain uses multi-label "
                            "OGBN-Proteins targets."
                        )
                        coverage = "registry_and_fail_closed_test"
                        runnable = False
                        eligible = False
                        fidelity = "not_applicable"
                    elif algorithm == "Bare":
                        status = "native"
                        reason = (
                            "Native UEFA supervised local update, verified by "
                            "synthetic invariants and the G11 real K=10 matrix "
                            "across all seven released families."
                        )
                        coverage = "synthetic_invariants_and_g11_real_k10_all_families"
                        runnable = True
                        eligible = strategy != "centralized_shard_oracle"
                        fidelity = "verified"
                    elif strategy == "centralized_shard_oracle":
                        status = "unsupported_scientific"
                        reason = (
                            "CentralizedShardReference is a diagnostic query-weighted "
                            "objective over strict-local shards, not a client continual-"
                            "algorithm composition or an empirical upper bound."
                        )
                        coverage = "validation_only"
                        runnable = False
                        eligible = False
                        fidelity = "not_applicable"
                    elif problem == "NC" and incremental == "class":
                        status = "faithful_supported"
                        reason = (
                            "Mechanism oracle, client-local state isolation, and real K=10 "
                            "LocalOnly/FedAvg/FedProx execution are verified by G12 v1."
                        )
                        coverage = "g12_v1_mechanism_and_real_nc_class_matrix"
                        runnable = True
                        eligible = True
                        fidelity = "verified"
                    else:
                        status = "implemented_unverified"
                        reason = (
                            "A dedicated UEFA implementation runs on synthetic streams, but "
                            "original BeGin parity and real-data behavior are unverified."
                        )
                        coverage = "synthetic_smoke_no_original_parity"
                        runnable = True
                        eligible = False
                        fidelity = "unverified"
                    entries.append(
                        CompatibilityEntry(
                            algorithm,
                            strategy,
                            problem,
                            incremental,
                            "uefa_graph_model",
                            status,
                            reason,
                            coverage,
                            runnable,
                            eligible,
                            fidelity,
                        )
                    )

        for algorithm in GENERIC_ALGORITHMS:
            for strategy in SERVER_STRATEGIES:
                for problem, incremental in SCENARIOS:
                    allowed = strategy != "centralized_shard_oracle"
                    if (
                        algorithm == "generic_weight_isolation"
                        and strategy != "local_only"
                    ):
                        allowed = False
                    entries.append(
                        CompatibilityEntry(
                            algorithm,
                            strategy,
                            problem,
                            incremental,
                            "uefa_graph_model",
                            "experimental_placeholder"
                            if allowed
                            else "unsupported_scientific",
                            (
                                "Explicitly named generic mechanism; not attributable to an "
                                "original published method."
                                if allowed
                                else "This generic mechanism is not valid with the selected strategy."
                            ),
                            "opt_in_synthetic_smoke" if allowed else "validation_only",
                            allowed,
                            False,
                            "not_an_original_method",
                        )
                    )
        return tuple(entries)

    def names(self) -> Tuple[str, ...]:
        """Return default runnable names; experimental mechanisms are excluded."""

        return ("Bare", "LwF", "EWC", "MAS", "ERGNN")

    def experimental_names(self) -> Tuple[str, ...]:
        return GENERIC_ALGORITHMS

    def all_runnable_names(self) -> Tuple[str, ...]:
        return self.names() + self.experimental_names()

    def explicit_v2_names(self) -> Tuple[str, ...]:
        """Return names constructible only through a validated v2 family."""

        return tuple(
            sorted(
                {
                    name
                    for factories in self._explicit_v2_factories.values()
                    for name in factories
                }
            )
        )

    def cli_names(self) -> Tuple[str, ...]:
        """Include explicit-v2 names so the CLI can validate their config."""

        return self.all_runnable_names() + self.explicit_v2_names()

    def original_names(self) -> Tuple[str, ...]:
        return ORIGINAL_ALGORITHMS

    def create(
        self,
        name: str,
        *,
        allow_experimental_placeholder: bool = False,
        explicit_v2_family: str | None = None,
        **kwargs,
    ) -> ClientContinualAlgorithm:
        if explicit_v2_family is not None:
            factories = self._explicit_v2_factories.get(explicit_v2_family)
            if factories is None or name not in factories:
                raise ValueError(
                    "The explicit v2 method family does not authorize algorithm "
                    f"{name!r}: {explicit_v2_family!r}."
                )
            return factories[name](**kwargs)
        if name in self.explicit_v2_names():
            raise ValueError(
                f"Original method {name!r} is disabled: no faithful UEFA adapter "
                "is exposed without its validated explicit v2 method config."
            )
        if name in set(ORIGINAL_ALGORITHMS) - set(self.names()):
            raise ValueError(
                f"Original method {name!r} is disabled: no faithful UEFA adapter exists."
            )
        if name in GENERIC_ALGORITHMS and not allow_experimental_placeholder:
            raise ValueError(
                f"{name!r} is an experimental generic mechanism; explicit opt-in is required."
            )
        if name not in self._factories:
            raise ValueError(f"Unknown client continual algorithm: {name}")
        class_il_single_head = bool(kwargs.pop("class_il_single_head", False))
        if class_il_single_head:
            if name != "LwF" or kwargs.get("incremental_setting") != "class":
                raise ValueError(
                    "class_il_single_head is valid only for LwF on NC-Class."
                )
            return LwFClassILAlgorithm(**kwargs)
        return self._factories[name](**kwargs)

    def compatibility(self) -> Tuple[CompatibilityEntry, ...]:
        return self._compatibility

    def validate(
        self,
        algorithm: str,
        strategy: str,
        problem: str,
        incremental: str,
        *,
        allow_experimental_placeholder: bool = False,
    ) -> CompatibilityEntry:
        for entry in self._compatibility:
            if (
                entry.algorithm == algorithm
                and entry.server_strategy == strategy
                and entry.problem_type == problem
                and entry.incremental_setting == incremental
            ):
                if not entry.runnable:
                    raise ValueError(f"Unsupported combination: {entry.reason}")
                if (
                    entry.support_status == "experimental_placeholder"
                    and not allow_experimental_placeholder
                ):
                    raise ValueError(
                        "Experimental generic mechanisms require "
                        "allow_experimental_placeholder=True."
                    )
                return entry
        raise ValueError("Compatibility entry is missing.")

    def as_records(self):
        return [asdict(entry) for entry in self._compatibility]
