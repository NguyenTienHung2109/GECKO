from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import torch

from gecko.config import GECKOConfig
from gecko.data.datasets.synthetic import build_synthetic_spec
from gecko.data.streams import StreamBuilder
from gecko.data.streams import StreamBundle


CASES = (
    ("NC", "task", 2),
    ("NC", "class", 2),
    ("NC", "domain", 2),
    ("LC", "task", 2),
    ("LC", "class", 2),
    ("LC", "domain", 4),
    ("LP", "domain", 2),
)


def metrics_for(problem: str, incremental: str) -> list[str]:
    if problem == "LP":
        return ["hits@50", "mrr", "average_precision", "rocauc"]
    if problem == "LC":
        return ["accuracy", "macro_f1"]
    if incremental == "domain":
        return ["rocauc"]
    return ["accuracy"]


def make_config(
    problem: str,
    incremental: str,
    num_tasks: int,
    *,
    seed: int = 0,
    spatial_profile: str = "mild",
    order_profile: str = "hard",
    num_clients: int = 2,
    rounds: int = 1,
    minimum_support: int = 1,
    allow_infeasible: bool = False,
) -> GECKOConfig:
    return GECKOConfig.from_mapping(
        {
            "scenario": {
                "dataset": "synthetic",
                "problem": problem,
                "incremental_setting": incremental,
                "num_tasks": num_tasks,
                "metrics": metrics_for(problem, incremental),
                "synthetic": True,
            },
            "partition": {
                "num_clients": num_clients,
                "spatial_profile": spatial_profile,
                "client_size_tolerance": 0.60,
                "minimum_train_queries_per_client_task": minimum_support,
                "minimum_validation_queries_per_client_task": minimum_support,
                "minimum_test_queries_per_client_task": minimum_support,
                "maximum_assignment_iterations": 500,
                "allow_infeasible": allow_infeasible,
            },
            "order": {"profile": order_profile},
            "training": {
                "participation_fraction": 1.0,
                "rounds_per_stage": rounds,
                "local_epochs_per_round": 1,
                "learning_rate": 0.01,
                "hidden_size": 8,
                "num_layers": 2,
            },
            "wandb": {"mode": "disabled", "project": "UEFA"},
            "seed": seed,
        }
    )


@lru_cache(maxsize=None)
def make_stream(
    problem: str,
    incremental: str,
    num_tasks: int,
    seed: int = 0,
    spatial_profile: str = "mild",
    order_profile: str = "hard",
    rounds: int = 1,
) -> StreamBundle:
    config = make_config(
        problem,
        incremental,
        num_tasks,
        seed=seed,
        spatial_profile=spatial_profile,
        order_profile=order_profile,
        rounds=rounds,
    )
    scenario = build_synthetic_spec(
        problem,
        incremental,
        num_tasks=num_tasks,
        num_clients=config.partition.num_clients,
        seed=seed,
    )
    return StreamBuilder(config).build(scenario)


def assert_tensor_map_equal(first: dict, second: dict) -> None:
    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], dict):
            assert_tensor_map_equal(first[key], second[key])
        elif torch.is_tensor(first[key]):
            assert torch.equal(first[key], second[key])
        else:
            assert first[key] == second[key]


def positive_key_set(stream: StreamBundle) -> set[tuple[int, int]]:
    pairs = stream.scenario.metadata["positive_pairs"]
    undirected = bool(stream.scenario.metadata["undirected"])
    return {
        (min(source, target), max(source, target))
        if undirected
        else (source, target)
        for source, target in pairs.tolist()
    }
