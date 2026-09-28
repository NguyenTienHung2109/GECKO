"""Deterministic synthetic scenarios used by the CPU test suite."""

from __future__ import annotations

from typing import Dict
from typing import Tuple

import torch

from gecko.reproducibility import torch_generator
from gecko.types import ScenarioSpec
from gecko.data.splits.common import bitcoin_structural_degree_q4
from gecko.data.splits.common import canonicalize_logical_edges
from gecko.data.scenario import build_lc_spec
from gecko.data.scenario import build_lp_spec
from gecko.data.scenario import build_nc_spec


def _static_graph(num_nodes: int) -> torch.Tensor:
    arcs: list[tuple[int, int]] = []
    for node in range(num_nodes):
        for offset in (1, 3):
            target = (node + offset) % num_nodes
            arcs.extend([(node, target), (target, node)])
    return torch.tensor(sorted(set(arcs)), dtype=torch.long).t().contiguous()


def _split_codes(count: int, num_tasks: int) -> torch.Tensor:
    per_task_position = torch.arange(count) // num_tasks
    return per_task_position.remainder(5).where(
        per_task_position.remainder(5) < 3,
        torch.full((count,), 2, dtype=torch.long),
    )


def _negative_pairs(
    num_nodes: int,
    positive_pairs: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    positives = {
        (min(source, target), max(source, target))
        for source, target in positive_pairs.tolist()
    }
    chosen: list[tuple[int, int]] = []
    used = set(positives)
    attempts = 0
    while len(chosen) < count and attempts < num_nodes * num_nodes * 10:
        source = int(torch.randint(num_nodes, (1,), generator=generator))
        target = int(torch.randint(num_nodes, (1,), generator=generator))
        attempts += 1
        key = (min(source, target), max(source, target))
        if source == target or key in used:
            continue
        used.add(key)
        chosen.append(key)
    if len(chosen) != count:
        raise RuntimeError("Synthetic graph has too few non-edges.")
    return torch.tensor(chosen, dtype=torch.long)


def build_synthetic_spec(
    problem: str,
    incremental_setting: str,
    *,
    num_tasks: int = 2,
    num_clients: int = 2,
    seed: int = 0,
) -> ScenarioSpec:
    """Build a small static scenario with dense per-task support."""

    problem = problem.upper()
    incremental_setting = incremental_setting.lower()
    num_nodes = max(48, num_clients * num_tasks * 12)
    generator = torch_generator(seed, "synthetic", problem, incremental_setting)
    features = torch.randn(num_nodes, 8, generator=generator)
    graph_edges = _static_graph(num_nodes)

    if problem == "NC":
        if incremental_setting in {"task", "class"}:
            num_classes = num_tasks * 2
            labels = torch.arange(num_nodes).remainder(num_classes)
            task_ids = labels.div(2, rounding_mode="floor")
            task_classes = {
                task: torch.tensor([2 * task, 2 * task + 1], dtype=torch.long)
                for task in range(num_tasks)
            }
        else:
            num_classes = 3
            first = torch.arange(num_nodes).remainder(2)
            second = torch.arange(num_nodes).div(2, rounding_mode="floor").remainder(2)
            third = (first ^ second).long()
            labels = torch.stack([first, second, third], dim=1).float()
            task_ids = torch.arange(num_nodes).remainder(num_tasks)
            task_classes = {}
        position = torch.arange(num_nodes).div(num_tasks, rounding_mode="floor").remainder(5)
        train_mask = position < 3
        validation_mask = position == 3
        test_mask = position == 4
        return build_nc_spec(
            dataset_name="synthetic",
            incremental_type=incremental_setting,
            metrics=("rocauc",) if incremental_setting == "domain" else ("accuracy",),
            edge_index=graph_edges,
            node_features=features,
            labels=labels,
            task_ids=task_ids,
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            num_tasks=num_tasks,
            num_classes=num_classes,
            task_class_sets=task_classes,
            domains=task_ids if incremental_setting == "domain" else None,
            metadata={"synthetic": True, "target_type": "multi_label" if labels.ndim == 2 else "single_label"},
        )

    logical_pairs, logical_ids, _ = canonicalize_logical_edges(
        graph_edges, undirected=True
    )
    logical_count = logical_pairs.shape[0]
    split_codes = _split_codes(logical_count, num_tasks)
    logical_train = split_codes == 0
    logical_val = split_codes == 1
    logical_test = split_codes == 2

    if problem == "LC":
        num_classes = num_tasks * 2 if incremental_setting in {"task", "class"} else 4
        logical_labels = torch.arange(logical_count).remainder(num_classes)
        if incremental_setting in {"task", "class"}:
            logical_tasks = logical_labels.div(2, rounding_mode="floor")
            task_classes = {
                task: torch.tensor([2 * task, 2 * task + 1], dtype=torch.long)
                for task in range(num_tasks)
            }
            domain_metadata: Dict[str, object] = {}
        else:
            logical_tasks, domain_metadata = bitcoin_structural_degree_q4(
                logical_pairs, logical_train, num_nodes=num_nodes
            )
            num_tasks = 4
            task_classes = {}
        return build_lc_spec(
            dataset_name="synthetic",
            incremental_type=incremental_setting,
            metrics=("accuracy", "macro_f1"),
            edge_index=graph_edges,
            node_features=features,
            raw_labels=logical_labels[logical_ids],
            raw_task_ids=logical_tasks[logical_ids],
            raw_train_mask=logical_train[logical_ids],
            raw_validation_mask=logical_val[logical_ids],
            raw_test_mask=logical_test[logical_ids],
            num_tasks=num_tasks,
            num_classes=num_classes,
            undirected=True,
            task_class_sets=task_classes,
            domains=logical_tasks[logical_ids] if incremental_setting == "domain" else None,
            metadata={"synthetic": True, **domain_metadata},
        )

    if problem != "LP" or incremental_setting != "domain":
        raise ValueError(f"Unsupported synthetic combination: {problem}-{incremental_setting}")
    positive_pairs = logical_pairs
    positive_tasks = torch.arange(logical_count).remainder(num_tasks)
    positive_splits = split_codes
    negatives: Dict[int, Dict[str, torch.Tensor]] = {}
    for task in range(num_tasks):
        negatives[task] = {}
        for split, code in (("train", 0), ("val", 1), ("test", 2)):
            positive_count = int(((positive_tasks == task) & (positive_splits == code)).sum())
            negative_count = (
                positive_count
                if split == "train"
                else max(positive_count, 200 * num_clients * num_clients)
            )
            negatives[task][split] = _negative_pairs(
                num_nodes,
                positive_pairs,
                negative_count,
                torch_generator(seed, "negative", task, split),
            )
    return build_lp_spec(
        dataset_name="synthetic",
        metrics=("hits@50", "mrr", "average_precision", "rocauc"),
        node_features=features,
        positive_pairs=positive_pairs,
        positive_task_ids=positive_tasks,
        positive_splits=positive_splits,
        negative_pairs_by_task_split=negatives,
        num_tasks=num_tasks,
        undirected=True,
        metadata={"synthetic": True},
    )
