"""Paper scenario catalog and explicit, fail-closed reproduction commands.

The catalog describes the reported experiment panels, not a claim that a fresh
construction is byte-identical to historical data or reproduces reported scores.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence

import yaml

from gecko.config import GECKOConfig
from gecko.benchmarks.terminology import task_order_label
from gecko.reproducibility import repository_root


ALLOCATION_ALPHAS = (0.1, 1.0, 10.0, 100.0)
TASK_ORDERS = ("synchronized", "unsynchronized")
DERIVED_PROTOCOL = "lpt_exact_dirichlet_temporal_participation50_derived_v1"
LP_PROTOCOL = "lpt_exact_dirichlet_temporal_v4"


@dataclass(frozen=True)
class PaperScenario:
    """One reported scenario and its default experiment panel."""

    scenario_id: str
    scope: str
    config_name: str
    dataset: str
    problem: str
    incremental_setting: str
    num_tasks: int
    num_clients: int
    seeds: tuple[int, ...]
    allocation_alphas: tuple[float, ...]
    main_methods: tuple[str, ...]
    coverage: str
    supplementary_methods: tuple[str, ...] = ()

    @property
    def construction_only(self) -> bool:
        return self.scenario_id == "S3"

    @property
    def methods(self) -> tuple[str, ...]:
        return self.main_methods + self.supplementary_methods


@dataclass(frozen=True)
class PaperMethod:
    """Exact method recipe; public labels never imply a local-only baseline."""

    label: str
    strategy: str
    continual_method: str
    configs: dict[str, str]
    model: str = "begin_gcn"
    task_orders: tuple[str, ...] = TASK_ORDERS


SCENARIOS: dict[str, PaperScenario] = {
    "S1": PaperScenario("S1", "nc_task", "nc_task_ogbn_arxiv.yaml", "ogbn-arxiv", "NC", "task", 8, 10,
                        (0, 1, 2), (0.1, 100.0), ("GEM", "TWP", "FedGTA", "FedDC", "POWER", "MOTION"), "matched anchors"),
    "S2": PaperScenario("S2", "nc_class", "nc_class_ogbn_arxiv.yaml", "ogbn-arxiv", "NC", "class", 8, 10,
                        (0, 1, 2), ALLOCATION_ALPHAS,
                        ("Bare", "CaT", "DSLR", "SSM", "GEM", "TWP", "FedGTA", "FedPUB", "FedDC", "POWER", "MOTION"),
                        "full grid", ("FedFST",)),
    "S3": PaperScenario("S3", "nc_domain", "nc_domain_ogbn_proteins.yaml", "ogbn-proteins", "NC", "domain", 8, 10,
                        (0, 1, 2), ALLOCATION_ALPHAS, (), "construction only; no reported learner panel"),
    "S4": PaperScenario("S4", "lc_task", "lc_task_bitcoin.yaml", "bitcoin", "LC", "task", 3, 10,
                        (0, 1, 2), (0.1, 100.0), ("GEM", "TWP"), "matched anchors"),
    "S5": PaperScenario("S5", "lc_class", "lc_class_bitcoin.yaml", "bitcoin", "LC", "class", 3, 10,
                        (0, 1, 2), ALLOCATION_ALPHAS, ("Bare", "CaT", "SSM", "TWP", "GEM"), "full grid"),
    "S6": PaperScenario("S6", "lc_domain", "lc_domain_bitcoin.yaml", "bitcoin", "LC", "domain", 4, 10,
                        (0, 1, 2), (0.1, 100.0), ("GEM", "TWP"), "four-anchor extension; ineligible streams remain blocked"),
    "S7": PaperScenario("S7", "lp_domain", "lp_domain_facebook.yaml", "facebook", "LP", "domain", 2, 3,
                        (0, 1, 2, 3, 4), ALLOCATION_ALPHAS, ("Bare", "EWC", "LwF", "GraphKeeper"), "full grid"),
}

METHODS: dict[str, PaperMethod] = {
    "Bare": PaperMethod("Bare", "fedavg", "Bare", dict.fromkeys(("S2", "S5", "S7"), "legacy_bare_fedavg_nc_task_grid_v1.yaml")),
    "CaT": PaperMethod("CaT", "fedavg", "CaT", {"S2": "cat_fedavg_nc_class_v1.yaml", "S5": "cat_fedavg_lc_class_v1.yaml"}),
    "DSLR": PaperMethod("DSLR", "fedavg", "DSLR", {"S2": "dslr_fedavg_nc_class_frozen_v1.yaml"}),
    "SSM": PaperMethod("SSM", "fedavg", "SSM", {"S2": "ssm_fedavg_nc_class_frozen_v1.yaml", "S5": "ssm_fedavg_lc_class_v1.yaml"}),
    "GEM": PaperMethod("GEM", "fedavg", "GEM", dict.fromkeys(("S1", "S2", "S4", "S5", "S6"), "gem_fedavg_nc_lc_v1.yaml")),
    "TWP": PaperMethod("TWP", "fedavg", "TWP", dict.fromkeys(("S1", "S2", "S4", "S5", "S6"), "twp_fedavg_nc_lc_v1.yaml")),
    "FedGTA": PaperMethod("FedGTA", "fedgta", "Bare", dict.fromkeys(("S1", "S2"), "fedgta_bare_nc_promotion_v3.yaml")),
    "FedPUB": PaperMethod("FedPUB", "fed_pub", "Bare", {"S2": "fedpub_bare_nc_promotion_v3.yaml"}),
    "FedDC": PaperMethod("FedDC", "feddc", "Bare", dict.fromkeys(("S1", "S2"), "feddc_bare_nc_promotion_v3.yaml")),
    "POWER": PaperMethod("POWER", "power_uefa", "Bare", {"S1": "power_uefa_full_nc_task_v1.yaml", "S2": "power_uefa_full_nc_class_promotion_v3.yaml"}),
    "MOTION": PaperMethod("MOTION", "motion", "Bare", {"S1": "motion_nc_task_v1.yaml", "S2": "motion_nc_class_v1.yaml"}),
    "FedFST": PaperMethod("FedFST", "fedfst", "Bare", {"S2": "fedfst_nc_class_v1.yaml"}, "fedfst_gat", ("synchronized",)),
    "EWC": PaperMethod("EWC", "fedavg", "EWC", {"S7": "legacy_ewc_fedavg_lp_domain_v1.yaml"}),
    "LwF": PaperMethod("LwF", "fedavg", "LwF", {"S7": "legacy_lwf_fedavg_lp_domain_v1.yaml"}),
    "GraphKeeper": PaperMethod("GraphKeeper", "fedavg", "GraphKeeper-LP", {"S7": "graphkeeper_lp_fedavg_lp_domain_v1.yaml"}),
}


@dataclass(frozen=True)
class PaperSelection:
    """A checked subset of a scenario's reported cells."""

    scenario: PaperScenario
    methods: tuple[str, ...]
    allocation_alphas: tuple[float, ...]
    task_orders: tuple[str, ...]
    seeds: tuple[int, ...]


def scenario_id_for_config(config: GECKOConfig) -> str | None:
    """Recognize real paper families without assigning IDs to synthetic data."""
    current = config.scenario
    if current.synthetic:
        return None
    for key, spec in SCENARIOS.items():
        if (current.dataset, current.problem, current.incremental_setting, current.num_tasks) == (
            spec.dataset, spec.problem, spec.incremental_setting, spec.num_tasks
        ):
            return key
    return None


def source_root() -> Path:
    """Locate the unpacked source distribution without source-control metadata."""
    root = repository_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def select_scenarios(
    scenario_ids: Sequence[str], *, methods: Sequence[str] | None = None,
    allocation_alphas: Sequence[float] | None = None,
    task_orders: Sequence[str] | None = None, seeds: Sequence[int] | None = None,
) -> tuple[PaperSelection, ...]:
    """Reject unsupported selections before any dataset is loaded."""
    selected = []
    for key in dict.fromkeys(scenario_ids):
        if key not in SCENARIOS:
            raise ValueError(f"Unknown paper scenario: {key}")
        spec = SCENARIOS[key]
        names = tuple(dict.fromkeys(spec.methods if methods is None else methods))
        alphas = tuple(dict.fromkeys(spec.allocation_alphas if allocation_alphas is None else allocation_alphas))
        orders = tuple(dict.fromkeys(TASK_ORDERS if task_orders is None else task_orders))
        seed_values = tuple(dict.fromkeys(spec.seeds if seeds is None else seeds))
        if not alphas or not set(alphas).issubset(spec.allocation_alphas):
            raise ValueError(f"{key}: reported allocation-alpha values are {spec.allocation_alphas}")
        if not orders or not set(orders).issubset(TASK_ORDERS):
            raise ValueError(f"{key}: task-order must be synchronized or unsynchronized")
        if not seed_values or not set(seed_values).issubset(spec.seeds):
            raise ValueError(f"{key}: reported seeds are {spec.seeds}")
        if spec.construction_only and names:
            raise ValueError("S3 is construction-only; the paper reports no learner methods")
        if not spec.construction_only and not names:
            raise ValueError(f"{key}: at least one method is required")
        for name in names:
            if name not in spec.methods:
                raise ValueError(f"{key}: method {name!r} is outside its reported panel {spec.methods}")
            if not set(orders).intersection(METHODS[name].task_orders):
                raise ValueError(f"{key}/{name}: reported only for task-order {METHODS[name].task_orders}")
        selected.append(PaperSelection(spec, names, alphas, orders, seed_values))
    if not selected:
        raise ValueError("At least one scenario must be selected")
    return tuple(selected)


def resolve_method(selection: PaperSelection, name: str) -> dict[str, Any]:
    """Validate the exact method recipe without loading datasets."""
    from gecko.algorithms.method_config import validate_method_config

    spec = selection.scenario
    method = METHODS[name]
    relative = Path("configs/gecko_v1/algorithms") / method.configs[spec.scenario_id]
    payload = yaml.safe_load((source_root() / relative).read_text(encoding="utf-8"))
    resolved = validate_method_config(
        payload, expected_strategy=method.strategy,
        expected_continual_method=method.continual_method,
        problem_type=spec.problem, incremental_setting=spec.incremental_setting,
    )
    orders = [order for order in selection.task_orders if order in method.task_orders]
    return {"method": name, "strategy": method.strategy,
            "continual_method": method.continual_method, "model": method.model,
            "model_architecture": ({"family": "GAT", "num_layers": 2, "hidden_size": 64,
                                    "heads": 1, "dropout": 0.5} if method.model == "fedfst_gat"
                                   else {"family": "GCN", "num_layers": 3, "hidden_size": 256}),
            "method_config": relative.as_posix(), "task_orders": orders,
            "cells": len(selection.seeds) * len(selection.allocation_alphas) * len(orders),
            "support_status": resolved.support_status,
            "benchmark_eligible": resolved.benchmark_eligible,
            "supplementary_only": name in spec.supplementary_methods}


def build_plan(selections: Sequence[PaperSelection], output: Path) -> dict[str, Any]:
    """Describe exact requested cells, support status and resource-sensitive steps."""
    records = []
    for selection in selections:
        spec = selection.scenario
        methods = [resolve_method(selection, name) for name in selection.methods]
        records.append({
            "scenario_id": spec.scenario_id, "scope": spec.scope, "dataset": spec.dataset,
            "problem": spec.problem, "incremental_setting": spec.incremental_setting,
            "num_tasks": spec.num_tasks, "num_clients": spec.num_clients,
            "coverage": spec.coverage, "construction_only": spec.construction_only,
            "config": f"configs/gecko_v1/scenarios/{spec.config_name}",
            "allocation_alphas": selection.allocation_alphas, "task_orders": selection.task_orders,
            "seeds": selection.seeds, "rounds_per_stage": 50, "local_epochs_per_round": 1,
            "clients_per_round": 2 if spec.problem == "LP" else 5,
            "configured_participation_fraction": 0.5,
            "realized_participation_fraction": 2 / 3 if spec.problem == "LP" else 0.5,
            "participation_note": "Client count is round(num_clients * 0.5), bounded to [1, num_clients].",
            "construction": "exact v4 partition-first LP, direct 50x1" if spec.problem == "LP"
                            else "exact v1 10x1, then participation-only extension to 50x1",
            "construction_requires_two_alphas": spec.problem == "LP",
            "stream_cells": len(selection.seeds) * len(selection.allocation_alphas) * len(selection.task_orders),
            "methods": methods, "training_jobs": sum(method["cells"] for method in methods),
            "stream_manifest": str(output / spec.scenario_id / "stream_manifest.json"),
        })
    return {"schema": "gecko-paper-plan-v1", "scenarios": records,
            "total_stream_cells": sum(row["stream_cells"] for row in records),
            "total_training_jobs": sum(row["training_jobs"] for row in records),
            "claim": "Recipe reconstruction, not frozen-payload or reported-score equality; method support statuses are not upgraded.",
            "test_scope": "No-download support contracts and synthetic behavioral tests; not the full real-data paper experiments."}


def _public_order(profile: str) -> str:
    label = task_order_label(profile)
    if label not in TASK_ORDERS:
        raise ValueError(f"Not a paper task order: {profile}")
    return label


def _cell(record: dict[str, Any]) -> tuple[int, float, str]:
    return int(record["seed"]), float(record["alpha_dirichlet"]), _public_order(record["order_profile"])


def _expected_cells(selection: PaperSelection) -> set[tuple[int, float, str]]:
    return {(seed, alpha, order) for seed in selection.seeds
            for alpha in selection.allocation_alphas for order in selection.task_orders}


def require_complete_grid(aggregate: dict[str, Any], selection: PaperSelection) -> None:
    """Keep every requested cell and reject failed or duplicate constructions."""
    records = aggregate.get("records", [])
    actual = [_cell(row) for row in records]
    if aggregate.get("failed_cells") or len(actual) != len(set(actual)) or set(actual) != _expected_cells(selection):
        raise ValueError(f"{selection.scenario.scenario_id}: incomplete construction; inspect failed_cells in its manifest")
    if selection.scenario.problem == "LP" and not aggregate.get("benchmark_eligible", False):
        raise ValueError("S7: allocation manipulation audit did not pass; inspect dirichlet_manipulation_audit")


def construct(selection: PaperSelection, output: Path, data_root: Path | None = None) -> Path:
    """Construct real immutable streams, preserving NC/LC 10-to-50 lineage."""
    root = source_root()
    from gecko.benchmarks.lpt_dirichlet_temporal import prepare_stream_grid
    from gecko.compat.lpt_protocol import LPT_V1
    from gecko.data.partitioning.base import participation_hash
    from gecko.data.streams import derive_participation_variant
    from gecko.types import ParticipationPlan
    from tools.derive_gecko_participation_grid import atomic_json, atomic_yaml, verify_derivation

    spec = selection.scenario
    if spec.problem == "LP" and len(selection.allocation_alphas) < 2:
        raise ValueError("S7 construct requires at least two allocation-alpha levels for its scientific manipulation audit; run may select one from an audited grid")
    destination = output / spec.scenario_id / "stream_manifest.json"
    if destination.exists():
        raise FileExistsError(f"Already constructed: {destination}; select run or a fresh --output")
    rounds = 50 if spec.problem == "LP" else 10
    base = GECKOConfig.from_yaml(root / "configs/gecko_v1/scenarios" / spec.config_name)
    base = replace(base, benchmark_seeds=spec.seeds,
        scenario=replace(base.scenario, save_path=str((data_root or root / "data").resolve())),
        partition=replace(base.partition, num_clients=spec.num_clients),
        training=replace(base.training, participation_fraction=0.5, rounds_per_stage=rounds, local_epochs_per_round=1),
        wandb=replace(base.wandb, mode="disabled"))
    config_path = destination.parent / f"base_config_{rounds}x1.yaml"
    atomic_yaml(config_path, base.to_dict())
    aggregate = prepare_stream_grid(
        config_path, store_root=destination.parent / f"streams_{rounds}x1",
        protocol_version=LP_PROTOCOL if spec.problem == "LP" else LPT_V1,
        alphas=selection.allocation_alphas,
        temporal_profiles=tuple("hard" if order == "unsynchronized" else order for order in selection.task_orders),
        seeds=selection.seeds, num_clients=spec.num_clients,
        rounds_per_stage=rounds, local_epochs_per_round=1, repository_root=root,
    )
    aggregate.update(scenario_id=spec.scenario_id, num_clients=spec.num_clients,
                     rounds_per_stage=rounds, local_epochs_per_round=1)
    source_manifest = destination if spec.problem == "LP" else destination.parent / "source_manifest_10x1.json"
    atomic_json(source_manifest, aggregate)
    require_complete_grid(aggregate, selection)
    if spec.problem != "LP":
        records = []
        for record in aggregate["records"]:
            source = Path(record["stream_path"])
            target = derive_participation_variant(source, 50, root=destination.parent / "streams_50x1", repository_root=root)
            manifest = verify_derivation(source, target)
            participation = json.loads((target / "participation.json").read_text(encoding="utf-8"))
            plan = ParticipationPlan(
                trace={int(stage): {int(round_id): tuple(clients) for round_id, clients in rounds.items()}
                       for stage, rounds in participation["trace"].items()},
                fraction=participation["fraction"], rounds_per_stage=participation["rounds_per_stage"],
                seed=participation["seed"],
            )
            records.append({**record, "stream_path": str(target.resolve()),
                            "stream_id": manifest["stream_id"], "stream_hash": manifest["config_hash"],
                            "scientific_fingerprint": manifest["scientific_fingerprint"],
                            "participation_hash": participation_hash(plan),
                            "release_status": manifest.get("release_status", "unassessed"),
                            "derived_from_stream_id": record["stream_id"]})
        target_config = replace(base, training=replace(base.training, rounds_per_stage=50), output_root=".")
        config_path = destination.parent / "base_config_50x1.yaml"
        atomic_yaml(config_path, target_config.to_dict())
        aggregate = {**aggregate, "config_path": str(config_path), "protocol_version": DERIVED_PROTOCOL,
                     "store_root": str(destination.parent / "streams_50x1"), "rounds_per_stage": 50,
                     "records": records, "derived_from": {
                         "operation": "participation_round_extension_v1", "source_manifest": str(source_manifest),
                         "source_protocol_version": LPT_V1, "source_rounds_per_stage": 10,
                         "old_trace_is_exact_prefix": True,
                         "preserved_components": ["scenario", "partition", "queries", "evaluation"]}}
        atomic_json(destination, aggregate)
    return destination


def run_commands(selection: PaperSelection, output: Path, device: str) -> list[list[str]]:
    """Audit every selected real cell before returning strict training commands."""
    from gecko.data.streams import audit_stream

    spec = selection.scenario
    if spec.construction_only:
        raise ValueError("S3 is construction-only; use --stage construct or --stage test")
    methods = [resolve_method(selection, name) for name in selection.methods]
    manifest_path = output / spec.scenario_id / "stream_manifest.json"
    aggregate = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_protocol = LP_PROTOCOL if spec.problem == "LP" else DERIVED_PROTOCOL
    if aggregate.get("protocol_version") != expected_protocol or aggregate.get("scenario_id") != spec.scenario_id:
        raise ValueError(f"{spec.scenario_id}: unexpected construction protocol/scenario in {manifest_path}")
    if aggregate.get("failed_cells"):
        raise ValueError(f"{spec.scenario_id}: construction recorded failed cells")
    if spec.problem == "LP" and not aggregate.get("benchmark_eligible", False):
        raise ValueError("S7: aggregate allocation manipulation audit did not pass")
    cells = {}
    for record in aggregate["records"]:
        key = _cell(record)
        if key in cells:
            raise ValueError(f"Duplicate stream cell: {key}")
        cells[key] = record
    requested = {key for key in _expected_cells(selection)
                 if any(key[2] in method["task_orders"] for method in methods)}
    missing = requested.difference(cells)
    if missing:
        raise ValueError(f"{spec.scenario_id}: requested cells are missing: {sorted(missing)}")
    root = source_root()
    commands = []
    for key in sorted(requested):
        row = cells[key]
        stream = Path(row["stream_path"])
        audit = audit_stream(stream)
        if not audit.get("valid") or audit.get("release_status") != "eligible":
            raise ValueError(f"Ineligible stream: {stream}; status={audit.get('release_status')}")
        if audit.get("scientific_fingerprint") != row.get("scientific_fingerprint"):
            raise ValueError(f"Stream fingerprint disagrees with aggregate: {stream}")
        config_path = stream / "config.yaml"
        config = GECKOConfig.from_yaml(config_path)
        if scenario_id_for_config(config) != spec.scenario_id:
            raise ValueError(f"Wrong scenario in stream: {stream}")
        if (config.partition.num_clients, config.training.rounds_per_stage,
                config.training.local_epochs_per_round, config.training.participation_fraction) != (spec.num_clients, 50, 1, 0.5):
            raise ValueError(f"Stream does not use the reported client/participation/50x1 budget: {stream}")
        if (config.seed, config.partition.dirichlet_alpha, _public_order(config.order.profile)) != key:
            raise ValueError(f"Stream config disagrees with its grid cell: {stream}")
        for method in methods:
            if key[2] not in method["task_orders"]:
                continue
            commands.append([sys.executable, "-m", "gecko", "run", "--config", str(config_path),
                "--stream", str(stream), "--strategy", method["strategy"], "--cl-algorithm", method["continual_method"],
                "--method-config", str(root / method["method_config"]), "--model", method["model"],
                "--seed", str(key[0]), "--model-seed", str(key[0]), "--device", device,
                "--wandb-mode", "disabled", "--run-tag", f"paper-{spec.scenario_id}-50x1"])
    return commands


METHOD_TESTS = {
    "Bare": ("integration/test_e2e.py",),
    "GEM": ("scientific/test_gem_method.py",),
    "EWC": ("scientific/test_method_fidelity_mechanisms.py",),
    "LwF": ("scientific/test_method_fidelity_mechanisms.py",),
    "CaT": ("scientific/test_cat_method.py",), "DSLR": ("scientific/test_dslr_method.py",),
    "SSM": ("scientific/test_ssm_method.py",), "TWP": ("scientific/test_twp_method.py",),
    "FedGTA": ("scientific/test_fedgta_oracle.py",), "FedPUB": ("integration/test_fedpub_runtime.py",),
    "FedDC": ("integration/test_feddc_runtime.py",), "POWER": ("scientific/test_power_paper_oracles.py",),
    "MOTION": ("unit/test_motion_gepae.py", "unit/test_motion_gtmsc.py"),
    "FedFST": ("integration/test_fedfst.py", "integration/test_fedfst_gat.py"),
    "GraphKeeper": ("scientific/test_graphkeeper_lp.py",),
}


def test_command(selections: Sequence[PaperSelection]) -> list[str]:
    """Select retained no-download contracts and mechanism tests, not paper runs."""
    targets = ["unit/test_paper_benchmark.py", "scientific/test_partition_and_streams.py",
               "scientific/test_semantics_and_leakage.py"]
    for selection in selections:
        if selection.scenario.problem == "LP":
            targets.append("scientific/test_lp_v2_protocol.py")
        for method in selection.methods:
            targets.extend(METHOD_TESTS.get(method, ()))
    return [sys.executable, "-m", "pytest", "-q", *(f"tests/{name}" for name in dict.fromkeys(targets))]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public source-distribution example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "construct", "run", "test"), default="plan")
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS))
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), help="Default: each scenario's reported panel, including synchronized-only S2 FedFST supplement.")
    parser.add_argument("--allocation-alpha", nargs="+", type=float, help="Select reported allocation-alpha levels; defaults depend on scenario.")
    parser.add_argument("--task-order", nargs="+", choices=TASK_ORDERS, help="Default: both orders; FedFST is scheduled only on synchronized cells.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Select reported seeds: 0..2 for NC/LC, 0..4 for LP.")
    parser.add_argument("--output", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--data-root", type=Path, help="Dataset download/cache directory, used only by construct.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--print-commands", action="store_true", help="Audit then print run commands, or print test command, without executing.")
    args = parser.parse_args(argv)
    if args.stage != "plan" and args.scenarios is None:
        parser.error("construct, run and test require explicit --scenarios")
    selections = select_scenarios(args.scenarios or tuple(SCENARIOS), methods=args.methods,
        allocation_alphas=args.allocation_alpha, task_orders=args.task_order, seeds=args.seeds)
    root = source_root()
    output = args.output.resolve()
    plan = build_plan(selections, output)
    if args.stage == "plan":
        print(json.dumps(plan, indent=2))
    elif args.stage == "construct":
        # Check cross-scenario restrictions before the first download starts.
        for selection in selections:
            if selection.scenario.problem == "LP" and len(selection.allocation_alphas) < 2:
                raise ValueError("S7 construct requires at least two allocation-alpha levels for its manipulation audit")
        for selection in selections:
            print(construct(selection, output, args.data_root), flush=True)
    elif args.stage == "run":
        if any(selection.scenario.construction_only for selection in selections):
            raise ValueError("S3 is construction-only; it has no reported learner run")
        # All requested cells are audited before any subprocess starts training.
        commands = [command for selection in selections for command in run_commands(selection, output, args.device)]
        for command in commands:
            print(shlex.join(command), flush=True)
            if not args.print_commands:
                subprocess.run(command, cwd=root, check=True)
    else:
        command = test_command(selections)
        print(plan["test_scope"], flush=True)
        print(shlex.join(command), flush=True)
        if not args.print_commands:
            subprocess.run(command, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
