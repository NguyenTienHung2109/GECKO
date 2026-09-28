"""Manifest and environment helpers."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any
from typing import Dict

import torch

def repository_sha(repository_root: str | Path) -> str:
    """Collect optional Git metadata only from an explicit checkout root."""

    if not (Path(repository_root) / ".git").exists():
        return "unknown"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def repository_provenance(
    repository_root: str | Path, *, collect_git: bool = False
) -> Dict[str, Any]:
    """Return informational metadata; source inspection is explicitly opt-in."""

    unavailable = {
        "provenance_type": "unavailable",
        "source_identifier": None,
        "commit_sha": None,
        "worktree_clean": None,
        "dirty_status_sha256": None,
    }
    if not collect_git:
        return unavailable
    root = Path(repository_root)
    sha = repository_sha(root)
    if sha == "unknown":
        return unavailable
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "provenance_type": "git",
            "source_identifier": sha,
            "commit_sha": sha,
            "worktree_clean": None,
            "dirty_status_sha256": None,
        }
    return {
        "provenance_type": "git",
        "source_identifier": sha,
        "commit_sha": sha,
        "worktree_clean": status == "",
        "dirty_status_sha256": (
            None
            if status == ""
            else hashlib.sha256(status.encode("utf-8")).hexdigest()
        ),
    }


def environment_snapshot() -> Dict[str, Any]:
    packages = {}
    for distribution in (
        "torch",
        "dgl",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "ogb",
        "pymetis",
        "wandb",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    snapshot = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    stable_identity = {
        key: snapshot[key]
        for key in ("python", "platform", "packages", "torch_cuda")
    }
    snapshot["environment_fingerprint"] = hashlib.sha256(
        json.dumps(
            stable_identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return snapshot


import hashlib
import json
from typing import Any
from typing import Dict

MANIFEST_FORMAT = "uefa-stream-manifest-v1"


STORE_FORMAT = "uefa-content-addressed-store-v1"


STORE_MARKER = ".uefa-store.json"


STORE_LOCK = ".uefa-store.lock"


MAX_MANIFEST_BYTES = 16 * 1024 * 1024


MAX_JSON_BYTES = 128 * 1024 * 1024


MAX_ARTIFACT_BYTES = 32 * 1024 * 1024 * 1024


MAX_STREAM_BYTES = 128 * 1024 * 1024 * 1024


SIGNATURE_POLICIES = frozenset({"allow_unsigned", "require_signed", "ignore"})


SCIENTIFIC_ARTIFACT_NAMES = frozenset(
    {
        "scenario.pt",
        "ownership.pt",
        "client_subgraph_index.pt",
        "client_query_shards.pt",
        "evaluation.pt",
        "tasks.json",
        "orders.json",
        "participation.json",
        "audit.json",
        "logical_edges.pt",
        "logical_edge_to_raw_edges.pt",
        "logical_edge_owner.pt",
        "edge_domain_scores.pt",
        "edge_domain_metadata.json",
        "context_edges.pt",
        "positive_train.pt",
        "positive_validation.pt",
        "positive_test.pt",
        "known_positive_pairs.pt",
        "negative_train.pt",
        "negative_validation.pt",
        "negative_test.pt",
        "domain_bundles.pt",
        "pair_owner.pt",
    }
)


COMPONENT_FILES = {
    "scenario": {
        "scenario.pt",
        "tasks.json",
        "logical_edges.pt",
        "logical_edge_to_raw_edges.pt",
        "edge_domain_scores.pt",
        "edge_domain_metadata.json",
        "context_edges.pt",
        "positive_train.pt",
        "positive_validation.pt",
        "positive_test.pt",
        "known_positive_pairs.pt",
    },
    "partition": {
        "ownership.pt",
        "client_subgraph_index.pt",
        "audit.json",
        "logical_edge_owner.pt",
        "pair_owner.pt",
    },
    "queries": {"client_query_shards.pt", "domain_bundles.pt"},
    "evaluation": {
        "evaluation.pt",
        "negative_train.pt",
        "negative_validation.pt",
        "negative_test.pt",
    },
    "order": {"orders.json"},
    "participation": {"participation.json"},
    "provenance": {"config.yaml", "environment.json"},
}


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _component_name(filename: str) -> str:
    for component, names in COMPONENT_FILES.items():
        if filename in names:
            return component
    return "provenance"


def _scientific_fingerprint(manifest: Dict[str, Any]) -> str:
    checksums = manifest["artifact_checksums"]
    payload = {
        "benchmark_name": manifest["benchmark_name"],
        "benchmark_schema_version": manifest["benchmark_schema_version"],
        "dataset_identifier": manifest["dataset_identifier"],
        "dataset_version": manifest.get("dataset_version"),
        "raw_dataset_checksums": manifest.get("raw_dataset_checksums", {}),
        "config_hash": manifest["config_hash"],
        "domain_constructor_version": manifest.get("domain_constructor_version"),
        "partitioner_version": manifest["partitioner_version"],
        "negative_sampler_version": manifest["negative_sampler_version"],
        "environment_fingerprint": manifest["environment_fingerprint"],
        "lp_protocol": manifest.get("lp_protocol"),
        "scientific_artifacts": {
            name: checksums[name]
            for name in sorted(checksums)
            if name in SCIENTIFIC_ARTIFACT_NAMES
        },
    }
    return _canonical_digest(payload)


def _package_digest(manifest: Dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"package_digest", "signature"}
    }
    return _canonical_digest(payload)
