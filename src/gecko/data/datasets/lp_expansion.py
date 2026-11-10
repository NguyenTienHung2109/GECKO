"""Raw, duplicate-free LP inputs for the 10-client rebuttal study."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from gecko.data.scenario import build_lp_spec
from gecko.types import ScenarioSpec


def _pairs_from_keys(keys: np.ndarray, num_nodes: int) -> torch.Tensor:
    return torch.from_numpy(np.column_stack((keys // num_nodes, keys % num_nodes)))


def _collab(path: Path) -> tuple[int, torch.Tensor, np.ndarray, dict]:
    with ZipFile(path) as archive:
        def read_raw(name: str) -> np.ndarray:
            with archive.open(f"collab/raw/{name}.csv.gz") as member:
                with gzip.GzipFile(fileobj=member) as raw:
                    return np.loadtxt(raw, delimiter=",", dtype=np.int64)

        num_nodes = int(read_raw("num-node-list").item())
        endpoints = read_raw("edge")
        years = read_raw("edge_year")
        numpy_types = [
            (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
            np.ndarray, np.dtype, type(np.dtype(np.int64)),
        ]
        # OGB v1 stores the later-year positives separately from the raw graph.
        with torch.serialization.safe_globals(numpy_types):
            later_years = [
                torch.load(io.BytesIO(archive.read(f"collab/split/time/{split}.pt")),
                           map_location="cpu", weights_only=True)
                for split in ("valid", "test")
            ]
        endpoints = np.concatenate([endpoints, *(split["edge"] for split in later_years)])
        years = np.concatenate([years, *(split["year"] for split in later_years)])

    source, target = np.sort(endpoints, axis=1).T
    nonloop = source != target
    unique, inverse = np.unique(
        source[nonloop] * num_nodes + target[nonloop], return_inverse=True
    )
    first_year = np.full(len(unique), years.max(), dtype=years.dtype)
    np.minimum.at(first_year, inverse, years[nonloop])
    # A deterministic, label-free cutoff on the older (official training-year)
    # portion of the source. The GECKO evaluation then splits pairs locally.
    cutoff = int(np.median(first_year[first_year <= 2017]))
    return num_nodes, _pairs_from_keys(unique, num_nodes), first_year > cutoff, {
        "domains": [f"first collaboration through {cutoff}", f"first collaboration after {cutoff}"],
        "first_year_cutoff": cutoff,
        "source_year_range": [int(years.min()), int(years.max())],
        "official_ogb_time_split_used": False,
    }


def _wikics(path: Path) -> tuple[int, torch.Tensor, np.ndarray, dict]:
    with path.open() as source:
        raw = json.load(source)
    labels = np.asarray(raw["labels"], dtype=np.int64)
    links = raw["links"]
    num_nodes = len(labels)
    if len(links) != num_nodes:
        raise ValueError("Wiki-CS links and topic labels have different node counts.")
    left = np.repeat(np.arange(num_nodes, dtype=np.int64), [len(row) for row in links])
    right = np.fromiter((target for row in links for target in row), dtype=np.int64)
    nonloop = left != right
    unique = np.unique(
        np.minimum(left[nonloop], right[nonloop]) * num_nodes
        + np.maximum(left[nonloop], right[nonloop])
    )
    pairs = _pairs_from_keys(unique, num_nodes)
    return num_nodes, pairs, labels[pairs[:, 0]] != labels[pairs[:, 1]], {
        "domains": ["within topic", "across topics"],
        "topic_labels_used_as_features": False,
    }


def load_lp_expansion(dataset: str, root: Path) -> ScenarioSpec:
    if dataset == "ogbl-collab":
        path = root / "lp_expansion_raw/collab.zip"
        num_nodes, pairs, late_domain, domain_details = _collab(path)
        data_version = "ogb-collab-first-collaboration-two-domain-v1"
    elif dataset == "wikics":
        path = root / "lp_expansion_raw/wiki_cs.json"
        num_nodes, pairs, late_domain, domain_details = _wikics(path)
        data_version = "wikics-topic-agreement-two-domain-v1"
    else:
        raise ValueError(f"Unsupported 10-client LP dataset {dataset!r}.")

    tasks = torch.from_numpy(late_domain.astype(np.int64))
    if tasks.unique().numel() != 2:
        raise ValueError(f"Both LP domains must have positive support in {dataset}.")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    empty = {
        task: {split: torch.empty((0, 2), dtype=torch.long) for split in ("train", "val", "test")}
        for task in range(2)
    }
    return build_lp_spec(
        dataset_name=dataset,
        metrics=("hits@50", "mrr", "average_precision", "rocauc"),
        node_features=torch.zeros((num_nodes, 1)),
        positive_pairs=pairs,
        positive_task_ids=tasks,
        positive_splits=torch.zeros(len(pairs), dtype=torch.long),
        negative_pairs_by_task_split=empty,
        num_tasks=2,
        undirected=True,
        context_edge_index=torch.empty((2, 0), dtype=torch.long),
        metadata={
            "dataset_version": data_version,
            "constructor_version": 1,
            "raw_dataset_checksums": {path.relative_to(root).as_posix(): digest.hexdigest()},
            "processed_cache_checksums": {},
            "lp_evaluation_negatives_first": True,
            "lp_expansion_domain_details": domain_details,
        },
    )
