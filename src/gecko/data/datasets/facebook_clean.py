"""Injective Facebook identities for the versioned partition-first LP track."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from gecko.data.scenario import build_lp_spec
from gecko.types import ScenarioSpec

CONSTRUCTOR = "facebook_partition_first_local_base_v2"
DATA_VERSION = "snap-gemsec-facebook-injective-local-base-v2"
DOMAINS = ("artist", "athletes", "company", "government", "new_sites",
           "politician", "public_figure", "tvshow")


def remap_category(pairs: torch.Tensor, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign a contiguous, reversible range even when local IDs have gaps."""
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long:
        raise ValueError("Expected int64 endpoint pairs with shape [N, 2].")
    if pairs.numel() == 0 or bool((pairs < 0).any()) or offset < 0:
        raise ValueError("Category edges must be nonempty with nonnegative identities.")
    local_ids, inverse = torch.unique(pairs.flatten(), sorted=True, return_inverse=True)
    return inverse.reshape_as(pairs) + offset, local_ids


def load_clean_facebook(root: Path) -> ScenarioSpec:
    """Read original CSVs, never the legacy graph or ambiguous endpoint metadata."""
    candidates = sorted(root.glob("facebook_*/facebook_clean_data"))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one original Facebook CSV directory in data root.")
    source = candidates[0]
    offset = 0
    positives, tasks, local_ids, category_ids = [], [], [], []
    checksums: dict[str, str] = {}
    for category, domain in enumerate(DOMAINS):
        path = source / f"{domain}_edges.csv"
        raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
        mapped, original_ids = remap_category(torch.from_numpy(raw), offset)
        # Mutual likes are logical undirected edges; discard duplicate/reverse rows.
        canonical = torch.sort(mapped, dim=1).values
        canonical = torch.unique(canonical, dim=0, sorted=True)
        positives.append(canonical)
        tasks.append(torch.full((len(canonical),), category, dtype=torch.long))
        local_ids.append(original_ids)
        category_ids.append(torch.full((len(original_ids),), category, dtype=torch.long))
        offset += len(original_ids)
        checksums[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    pairs = torch.cat(positives)
    empty = {task: {split: torch.empty((0, 2), dtype=torch.long)
                    for split in ("train", "val", "test")} for task in range(len(DOMAINS))}
    return build_lp_spec(
        dataset_name="facebook", metrics=("hits@50", "mrr", "average_precision", "rocauc"),
        node_features=torch.zeros((offset, 1)), positive_pairs=pairs,
        positive_task_ids=torch.cat(tasks), positive_splits=torch.zeros(len(pairs), dtype=torch.long),
        negative_pairs_by_task_split=empty, num_tasks=len(DOMAINS), undirected=True,
        context_edge_index=torch.empty((2, 0), dtype=torch.long),
        metadata={"dataset_version": DATA_VERSION, "constructor_version": 2,
                  "node_identity_policy": "category_sorted_unique_local_id_v2",
                  "node_source_categories": list(DOMAINS),
                  "node_source_category_ids": torch.cat(category_ids),
                  "node_source_local_ids": torch.cat(local_ids),
                  "raw_dataset_checksums": checksums, "processed_cache_checksums": {},
                  "legacy_endpoint_metadata_reused": False,
                  "negative_policy": "regenerate_after_partition_against_complete_corrected_positive_universe"},
    )
