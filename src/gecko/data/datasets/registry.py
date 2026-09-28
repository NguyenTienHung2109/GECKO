"""Canonical UEFA v1 scenario registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from typing import Tuple


@dataclass(frozen=True)
class ScenarioRegistryEntry:
    dataset: str
    problem: str
    incremental_setting: str
    num_tasks: int
    metrics: Tuple[str, ...]


SUPPORTED_SCENARIOS: Dict[Tuple[str, str], ScenarioRegistryEntry] = {
    ("NC", "task"): ScenarioRegistryEntry(
        "ogbn-arxiv", "NC", "task", 8, ("accuracy",)
    ),
    ("NC", "class"): ScenarioRegistryEntry(
        "ogbn-arxiv", "NC", "class", 8, ("accuracy",)
    ),
    ("NC", "domain"): ScenarioRegistryEntry(
        "ogbn-proteins", "NC", "domain", 8, ("rocauc",)
    ),
    ("LC", "task"): ScenarioRegistryEntry(
        "bitcoin", "LC", "task", 3, ("accuracy", "macro_f1")
    ),
    ("LC", "class"): ScenarioRegistryEntry(
        "bitcoin", "LC", "class", 3, ("accuracy", "macro_f1")
    ),
    ("LC", "domain"): ScenarioRegistryEntry(
        "bitcoin", "LC", "domain", 4, ("accuracy", "macro_f1")
    ),
    ("LP", "domain"): ScenarioRegistryEntry(
        "facebook", "LP", "domain", 2, ("hits@50", "mrr", "average_precision", "rocauc")
    ),
}

UNSUPPORTED_V1 = {
    ("LP", "task"),
    ("LP", "class"),
    ("NC", "time"),
    ("LC", "time"),
    ("LP", "time"),
    ("GC", "task"),
    ("GC", "class"),
    ("GC", "domain"),
    ("GC", "time"),
}



DATASET_VERSIONS = {
    "bitcoin": "snap-bitcoinotc-static+begin-metadata-v1",
    "citeseer": "dgl-citeseer-static-v1",
    "cora": "dgl-cora-v2-static-v1",
    "facebook": "snap-gemsec-facebook-static+begin-metadata-v1",
    "ogbn-arxiv": "ogb-release-v1",
    "ogbn-proteins": "ogb-release-v1+uefa-within-domain-v1",
}


