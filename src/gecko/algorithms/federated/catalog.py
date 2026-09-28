"""Server strategy descriptors kept separate from client CL algorithms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FederatedStrategy:
    name: str
    aggregates: bool
    uses_proximal_objective: bool = False
    oracle: bool = False
    requires_method_config: bool = False


STRATEGIES = {
    "local_only": FederatedStrategy("local_only", aggregates=False),
    "fedavg": FederatedStrategy("fedavg", aggregates=True),
    "fedprox": FederatedStrategy("fedprox", aggregates=True, uses_proximal_objective=True),
    "scaffold": FederatedStrategy(
        "scaffold", aggregates=True, requires_method_config=True
    ),
    "fedgta": FederatedStrategy(
        "fedgta", aggregates=True, requires_method_config=True
    ),
    "fed_pub": FederatedStrategy(
        "fed_pub", aggregates=True, requires_method_config=True
    ),
    "feddc": FederatedStrategy(
        "feddc", aggregates=True, requires_method_config=True
    ),
    "power_uefa": FederatedStrategy(
        "power_uefa", aggregates=True, requires_method_config=True
    ),
    "motion": FederatedStrategy(
        "motion", aggregates=True, requires_method_config=True
    ),
    "fedfst": FederatedStrategy(
        "fedfst", aggregates=True, requires_method_config=True
    ),
    "centralized_shard_oracle": FederatedStrategy(
        "centralized_shard_oracle", aggregates=False, oracle=True
    ),
}
