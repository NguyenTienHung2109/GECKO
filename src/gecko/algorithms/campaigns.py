from __future__ import annotations

from gecko.workflows._campaign_paths import LC_ALL
from gecko.workflows._campaign_paths import LP_DOMAIN
from gecko.workflows._campaign_paths import NC_ALL
from gecko.workflows._campaign_paths import NC_TASK
from gecko.workflows._campaign_paths import NC_TASK_CLASS
from gecko.workflows._campaign_paths import ROOT

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence
import yaml
from gecko.workflows._campaign_paths import LC_ALL
from gecko.workflows._campaign_paths import LP_DOMAIN
from gecko.workflows._campaign_paths import NC_ALL
from gecko.workflows._campaign_paths import NC_TASK
from gecko.workflows._campaign_paths import NC_TASK_CLASS

@dataclass(frozen=True)
class MethodCell:
    """One canonical runnable method configuration and its valid scenarios."""

    label: str
    strategy: str
    continual_method: str
    config_by_problem: Mapping[str, str]
    scenarios: frozenset[tuple[str, str]]
    config_by_scenario: Mapping[tuple[str, str], str] | None = None
    strategy_by_scenario: Mapping[tuple[str, str], str] | None = None
    order_profiles: frozenset[str] | None = None
    model: str | None = None

    def method_config(self, problem: str, incremental: str) -> Path:
        from gecko.workflows._campaign_paths import ROOT
        if self.config_by_scenario is not None:
            override = self.config_by_scenario.get((problem, incremental))
            if override is not None:
                return ROOT / override
        try:
            relative = self.config_by_problem[problem]
        except KeyError as error:
            raise ValueError(f"{self.label} has no {problem} method config.") from error
        return ROOT / relative

    def strategy_for(self, problem: str, incremental: str) -> str:
        if self.strategy_by_scenario is not None:
            override = self.strategy_by_scenario.get((problem, incremental))
            if override is not None:
                return override
        return self.strategy


METHODS: dict[str, MethodCell] = {
    "Bare": MethodCell(
        "Bare", "fedavg", "Bare",
        {
            "NC": "configs/gecko_v1/algorithms/legacy_bare_fedavg_nc_task_grid_v1.yaml",
            "LC": "configs/gecko_v1/algorithms/legacy_bare_fedavg_nc_task_grid_v1.yaml",
            "LP": "configs/gecko_v1/algorithms/legacy_bare_local_only_lp_domain_v1.yaml",
        },
        NC_ALL | LC_ALL | LP_DOMAIN,
        {("LP", "domain"): "configs/gecko_v1/algorithms/legacy_bare_local_only_lp_domain_v1.yaml"},
        {("LP", "domain"): "local_only"},
    ),
    "EWC": MethodCell(
        "EWC", "fedavg", "EWC",
        {
            "NC": "configs/gecko_v1/algorithms/legacy_ewc_fedavg_promotion_v3.yaml",
            "LP": "configs/gecko_v1/algorithms/legacy_ewc_local_only_lp_domain_v1.yaml",
        },
        NC_TASK | LP_DOMAIN,
        {("LP", "domain"): "configs/gecko_v1/algorithms/legacy_ewc_local_only_lp_domain_v1.yaml"},
        {("LP", "domain"): "local_only"},
    ),
    "LwF": MethodCell(
        "LwF", "fedavg", "LwF",
        {
            "NC": "configs/gecko_v1/algorithms/legacy_lwf_fedavg_promotion_v3.yaml",
            "LP": "configs/gecko_v1/algorithms/legacy_lwf_local_only_lp_domain_v1.yaml",
        },
        NC_TASK | LP_DOMAIN,
        {("LP", "domain"): "configs/gecko_v1/algorithms/legacy_lwf_local_only_lp_domain_v1.yaml"},
        {("LP", "domain"): "local_only"},
    ),
    "MAS": MethodCell(
        "MAS", "fedavg", "MAS",
        {"NC": "configs/gecko_v1/algorithms/legacy_mas_fedavg_promotion_v3.yaml"},
        NC_TASK,
    ),
    "ERGNN": MethodCell(
        "ERGNN", "fedavg", "ERGNN",
        {"NC": "configs/gecko_v1/algorithms/legacy_ergnn_fedavg_promotion_v3.yaml"},
        NC_TASK,
    ),
    "GEM": MethodCell(
        "GEM", "fedavg", "GEM",
        {"NC": "configs/gecko_v1/algorithms/gem_fedavg_nc_lc_v1.yaml", "LC": "configs/gecko_v1/algorithms/gem_fedavg_nc_lc_v1.yaml"},
        NC_ALL | LC_ALL,
    ),
    "TWP": MethodCell(
        "TWP", "fedavg", "TWP",
        {"NC": "configs/gecko_v1/algorithms/twp_fedavg_nc_lc_v1.yaml", "LC": "configs/gecko_v1/algorithms/twp_fedavg_nc_lc_v1.yaml"},
        NC_ALL | LC_ALL,
        {("NC", "domain"): "configs/gecko_v1/algorithms/twp_fedavg_nc_domain_memory_bounded_v1.yaml"},
    ),
    "CaT": MethodCell(
        "CaT", "local_only", "CaT",
        {"NC": "configs/gecko_v1/algorithms/cat_local_only_nc_class_v1.yaml", "LC": "configs/gecko_v1/algorithms/cat_local_only_lc_class_v1.yaml"},
        frozenset({("NC", "task"), ("NC", "class"), ("LC", "class")}),
        {("NC", "task"): "configs/gecko_v1/algorithms/cat_local_only_nc_task_v1.yaml"},
    ),
    "CaT-FedAvg": MethodCell(
        "CaT-FedAvg", "fedavg", "CaT",
        {
            "NC": "configs/gecko_v1/algorithms/cat_fedavg_nc_class_v1.yaml",
            "LC": "configs/gecko_v1/algorithms/cat_fedavg_lc_class_v1.yaml",
        },
        frozenset({("NC", "task"), ("NC", "class"), ("LC", "class")}),
        {("NC", "task"): "configs/gecko_v1/algorithms/cat_fedavg_nc_task_v1.yaml"},
    ),
    "CaT-FedProx": MethodCell(
        "CaT-FedProx", "fedprox", "CaT",
        {
            "NC": "configs/gecko_v1/algorithms/cat_fedprox_nc_class_v1.yaml",
            "LC": "configs/gecko_v1/algorithms/cat_fedprox_lc_class_v1.yaml",
        },
        frozenset({("NC", "task"), ("NC", "class"), ("LC", "class")}),
        {("NC", "task"): "configs/gecko_v1/algorithms/cat_fedprox_nc_task_v1.yaml"},
    ),
    "DSLR": MethodCell(
        "DSLR", "local_only", "DSLR",
        {"NC": "configs/gecko_v1/algorithms/dslr_local_only_nc_class_grid_beta_0p1_r_0p2_v1.yaml"},
        NC_TASK_CLASS,
        {("NC", "task"): "configs/gecko_v1/algorithms/dslr_local_only_nc_task_v1.yaml"},
    ),
    "DSLR-FedAvg": MethodCell(
        "DSLR-FedAvg", "fedavg", "DSLR",
        {"NC": "configs/gecko_v1/algorithms/dslr_fedavg_nc_class_frozen_v1.yaml"},
        NC_TASK_CLASS,
        {("NC", "task"): "configs/gecko_v1/algorithms/dslr_fedavg_nc_task_v1.yaml"},
    ),
    "DSLR-FedProx": MethodCell(
        "DSLR-FedProx", "fedprox", "DSLR",
        {"NC": "configs/gecko_v1/algorithms/dslr_fedprox_nc_class_frozen_v1.yaml"},
        NC_TASK_CLASS,
        {("NC", "task"): "configs/gecko_v1/algorithms/dslr_fedprox_nc_task_v1.yaml"},
    ),
    "DSLR-Normalized": MethodCell(
        "DSLR-Normalized", "local_only", "DSLR-Normalized",
        {"NC": "configs/gecko_v1/algorithms/dslr_normalized_local_only_nc_class_v1.yaml"},
        NC_TASK_CLASS,
    ),
    "SSM": MethodCell(
        "SSM", "local_only", "SSM",
        {"NC": "configs/gecko_v1/algorithms/ssm_local_only_nc_class_uniform_v1.yaml", "LC": "configs/gecko_v1/algorithms/ssm_local_only_lc_class_degree_v1.yaml"},
        frozenset({("NC", "task"), ("NC", "class"), ("LC", "class")}),
        {("NC", "task"): "configs/gecko_v1/algorithms/ssm_local_only_nc_task_v1.yaml"},
    ),
    "SSM-FedAvg": MethodCell(
        "SSM-FedAvg", "fedavg", "SSM",
        {
            "NC": "configs/gecko_v1/algorithms/ssm_fedavg_nc_class_frozen_v1.yaml",
            "LC": "configs/gecko_v1/algorithms/ssm_fedavg_lc_class_v1.yaml",
        },
        frozenset({("NC", "task"), ("NC", "class"), ("LC", "class")}),
        {("NC", "task"): "configs/gecko_v1/algorithms/ssm_fedavg_nc_task_v1.yaml"},
    ),
    "SSM-FedProx": MethodCell(
        "SSM-FedProx", "fedprox", "SSM",
        {
            "NC": "configs/gecko_v1/algorithms/ssm_fedprox_nc_class_frozen_v1.yaml",
            "LC": "configs/gecko_v1/algorithms/ssm_fedprox_lc_class_v1.yaml",
        },
        frozenset({("NC", "task"), ("NC", "class"), ("LC", "class")}),
        {("NC", "task"): "configs/gecko_v1/algorithms/ssm_fedprox_nc_task_v1.yaml"},
    ),
    "FedGTA": MethodCell(
        "FedGTA", "fedgta", "Bare",
        {"NC": "configs/gecko_v1/algorithms/fedgta_bare_nc_promotion_v3.yaml"},
        NC_TASK_CLASS,
    ),
    "FedPUB": MethodCell(
        "FedPUB", "fed_pub", "Bare",
        {"NC": "configs/gecko_v1/algorithms/fedpub_bare_nc_promotion_v3.yaml"},
        NC_TASK_CLASS,
    ),
    "FedDC": MethodCell(
        "FedDC", "feddc", "Bare",
        {"NC": "configs/gecko_v1/algorithms/feddc_bare_nc_promotion_v3.yaml"},
        NC_ALL,
    ),
    "POWER": MethodCell(
        "POWER", "power_uefa", "Bare",
        {"NC": "configs/gecko_v1/algorithms/power_uefa_full_nc_class_promotion_v3.yaml"},
        NC_TASK_CLASS,
        {("NC", "task"): "configs/gecko_v1/algorithms/power_uefa_full_nc_task_v1.yaml"},
    ),
    "MOTION": MethodCell(
        "MOTION", "motion", "Bare",
        {"NC": "configs/gecko_v1/algorithms/motion_nc_class_v1.yaml"},
        NC_TASK_CLASS,
        {("NC", "task"): "configs/gecko_v1/algorithms/motion_nc_task_v1.yaml"},
    ),
    "FedFST": MethodCell(
        "FedFST",
        "fedfst",
        "Bare",
        {"NC": "configs/gecko_v1/algorithms/fedfst_nc_class_v1.yaml"},
        NC_TASK_CLASS,
        config_by_scenario={
            ("NC", "task"): "configs/gecko_v1/algorithms/fedfst_nc_task_v1.yaml"
        },
        model="uefa_gcn",
    ),
    "GraphKeeper": MethodCell(
        "GraphKeeper", "local_only", "GraphKeeper",
        {"NC": "configs/gecko_v1/algorithms/graphkeeper_local_only_nc_domain_v1.yaml"},
        frozenset({("NC", "domain")}),
    ),
    "GraphKeeper-LP": MethodCell(
        "GraphKeeper-LP", "local_only", "GraphKeeper-LP",
        {"LP": "configs/gecko_v1/algorithms/graphkeeper_lp_local_only_lp_domain_v1.yaml"},
        LP_DOMAIN,
    ),
    "EWC-LP-FedAvg": MethodCell(
        "EWC-LP-FedAvg", "fedavg", "EWC",
        {"LP": "configs/gecko_v1/algorithms/legacy_ewc_fedavg_lp_domain_v1.yaml"},
        LP_DOMAIN,
    ),
    "EWC-LP-FedProx": MethodCell(
        "EWC-LP-FedProx", "fedprox", "EWC",
        {"LP": "configs/gecko_v1/algorithms/legacy_ewc_fedprox_lp_domain_v1.yaml"},
        LP_DOMAIN,
    ),
    "LwF-LP-FedAvg": MethodCell(
        "LwF-LP-FedAvg", "fedavg", "LwF",
        {"LP": "configs/gecko_v1/algorithms/legacy_lwf_fedavg_lp_domain_v1.yaml"},
        LP_DOMAIN,
    ),
    "LwF-LP-FedProx": MethodCell(
        "LwF-LP-FedProx", "fedprox", "LwF",
        {"LP": "configs/gecko_v1/algorithms/legacy_lwf_fedprox_lp_domain_v1.yaml"},
        LP_DOMAIN,
    ),
    "GraphKeeper-LP-FedAvg": MethodCell(
        "GraphKeeper-LP-FedAvg", "fedavg", "GraphKeeper-LP",
        {"LP": "configs/gecko_v1/algorithms/graphkeeper_lp_fedavg_lp_domain_v1.yaml"},
        LP_DOMAIN,
    ),
    "GraphKeeper-LP-FedProx": MethodCell(
        "GraphKeeper-LP-FedProx", "fedprox", "GraphKeeper-LP",
        {"LP": "configs/gecko_v1/algorithms/graphkeeper_lp_fedprox_lp_domain_v1.yaml"},
        LP_DOMAIN,
    ),
}


def _method_payload(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Method config is not a mapping: {path}")
    return dict(raw)


def _resolve_methods(names: Sequence[str]) -> tuple[MethodCell, ...]:
    if len(names) == 1 and names[0].lower() == "all":
        return tuple(METHODS.values())
    aliases = {key.lower(): value for key, value in METHODS.items()}
    resolved: list[MethodCell] = []
    for name in names:
        try:
            cell = aliases[name.lower()]
        except KeyError as error:
            raise ValueError(
                f"Unknown method {name!r}; choose from {', '.join(METHODS)}."
            ) from error
        if cell not in resolved:
            resolved.append(cell)
    return tuple(resolved)


