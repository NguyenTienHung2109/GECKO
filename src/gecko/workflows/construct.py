from __future__ import annotations

from gecko.workflows._logging import LOGGER

from gecko.reproducibility import repository_root as _repository_root
import argparse
import time
from dataclasses import replace
from pathlib import Path
from gecko.config import GECKOConfig
from gecko.benchmarks.terminology import normalize_task_order
from gecko.validation import ConfigurationError
from gecko.data.dataloader import load_scenario_spec
from gecko.data.streams import StreamBuilder
from gecko.data.streams import save_stream
from gecko.data.streams.builder import stream_relative_path

def _with_profile_overrides(config: GECKOConfig, args: argparse.Namespace) -> GECKOConfig:
    partition = config.partition
    order = config.order
    training = config.training
    wandb = config.wandb
    if getattr(args, "spatial_profile", None):
        partition = replace(partition, spatial_profile=args.spatial_profile)
    allocation_alpha = getattr(args, "allocation_alpha", None)
    historical_alpha = getattr(args, "dirichlet_alpha", None)
    if allocation_alpha is not None and historical_alpha is not None and allocation_alpha != historical_alpha:
        raise ConfigurationError("--allocation-alpha conflicts with --dirichlet-alpha.")
    alpha = allocation_alpha if allocation_alpha is not None else historical_alpha
    if alpha is not None:
        partition = replace(partition, dirichlet_alpha=alpha)
    if getattr(args, "num_clients", None) is not None:
        partition = replace(partition, num_clients=args.num_clients)
    public_order = getattr(args, "task_order", None)
    historical_order = getattr(args, "order_profile", None)
    if public_order is not None and historical_order is not None:
        normalized = normalize_task_order(
            public_order, problem=config.scenario.problem, num_tasks=config.scenario.num_tasks
        )
        if normalized != historical_order:
            raise ConfigurationError("--task-order conflicts with --order-profile.")
    requested_order = public_order if public_order is not None else historical_order
    if requested_order:
        profile = normalize_task_order(
            requested_order, problem=config.scenario.problem, num_tasks=config.scenario.num_tasks
        )
        order = replace(order, profile=profile)
        if public_order is not None:
            order = replace(
                order,
                allow_full_permutation_below_four_tasks=(
                    profile == "hard" and config.scenario.num_tasks < 4
                ),
            )
    if getattr(args, "allow_short_hard_order", False):
        order = replace(order, allow_full_permutation_below_four_tasks=True)
    if getattr(args, "class_task_policy", None):
        scenario = replace(
            config.scenario, class_task_policy=args.class_task_policy
        )
    else:
        scenario = config.scenario
    training_overrides = {
        field: getattr(args, field, None)
        for field in (
            "participation_fraction",
            "rounds_per_stage",
            "local_epochs_per_round",
            "fedprox_mu",
            "hidden_size",
            "num_layers",
        )
        if getattr(args, field, None) is not None
    }
    if training_overrides:
        training = replace(training, **training_overrides)
    if getattr(args, "wandb_mode", None):
        wandb = replace(wandb, mode=args.wandb_mode)
    updated = replace(
        config,
        scenario=scenario,
        partition=partition,
        order=order,
        training=training,
        wandb=wandb,
        seed=config.seed if getattr(args, "seed", None) is None else args.seed,
        output_root=(
            config.output_root
            if getattr(args, "output_root", None) is None
            else args.output_root
        ),
    )
    updated.validate()
    return updated


def expected_stream_path(config: GECKOConfig) -> Path:
    return Path(config.output_root) / stream_relative_path(config)


def command_generate(args: argparse.Namespace) -> int:
    from gecko.workflows._logging import LOGGER
    config = _with_profile_overrides(GECKOConfig.from_yaml(args.config), args)
    if config.partition.dirichlet_alpha is not None:
        raise ConfigurationError(
            "Explicit allocation_alpha requires exact Dirichlet construction. "
            "Use examples/benchmark.py; generic construct only builds legacy heuristic streams."
        )
    load_start = time.perf_counter()
    scenario = load_scenario_spec(config)
    load_seconds = time.perf_counter() - load_start
    build_start = time.perf_counter()
    bundle = StreamBuilder(config).build(scenario)
    build_seconds = time.perf_counter() - build_start
    serialization_start = time.perf_counter()
    path = save_stream(bundle, repository_root=_repository_root())
    serialization_seconds = time.perf_counter() - serialization_start
    LOGGER.info(
        "Generated stream %s (load/task %.3fs, build %.3fs, serialize %.3fs)",
        path,
        load_seconds,
        build_seconds,
        serialization_seconds,
    )
    print(path)
    return 0

