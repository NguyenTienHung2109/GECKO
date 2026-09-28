from __future__ import annotations

from gecko.data.streams.manifest import MANIFEST_FORMAT
from gecko.data.streams.manifest import MAX_MANIFEST_BYTES
from gecko.data.streams.manifest import STORE_FORMAT
from gecko.data.streams.manifest import STORE_LOCK
from gecko.data.streams.manifest import STORE_MARKER

import json
import os
import shutil
import stat
import time
import uuid
import weakref
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
import torch
import yaml
from gecko.config import GECKOConfig
from gecko.benchmarks.terminology import allocation_metadata
from gecko.data.audits.policy import AUDIT_POLICY_VERSION
from gecko.data.audits.policy import audit_policy_hash
from gecko.types import CentralEvaluationShard
from gecko.types import ClientGraph
from gecko.types import ClientTaskShard
from gecko.types import OrderPlan
from gecko.types import ParticipationPlan
from gecko.types import PartitionResult
from gecko.types import ScenarioSpec
from gecko.validation import ArtifactIntegrityError
from gecko.data.streams.builder import StreamBundle
from gecko.data.streams.builder import stream_relative_path
from gecko.data.streams.builder import spatial_profile_label
from gecko.data.streams.builder import stream_identity
from gecko.data.streams.manifest import environment_snapshot
from gecko.data.streams.manifest import repository_provenance
from gecko.data.streams.orders import generate_client_orders
from gecko.data.streams.participation import generate_participation

_SCENARIO_OBJECT_CACHE: dict[
    tuple[int, str], tuple[weakref.ReferenceType[ScenarioSpec], dict[str, Any]]
] = {}


SignatureVerifier = Callable[[str, str, str, str], bool]


@dataclass(frozen=True)
class ArtifactStore:
    """Filesystem facade for immutable, checksummed UEFA streams."""

    root: str | Path = "generated_streams"
    repository_root: str | Path = "."

    def save(self, bundle: StreamBundle) -> Path:
        return save_stream(
            bundle,
            self.root,
            repository_root=self.repository_root,
        )

    def load(
        self,
        path: str | Path,
        *,
        verify: bool = True,
        signature_policy: str = "allow_unsigned",
        signature_verifier: SignatureVerifier | None = None,
    ) -> StreamBundle:
        return load_stream(
            path,
            verify=verify,
            signature_policy=signature_policy,
            signature_verifier=signature_verifier,
        )

    def audit(self, path: str | Path) -> Dict[str, Any]:
        from gecko.data.streams.integrity import audit_stream
        return audit_stream(path)


def _atomic_publish(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _json_dump(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
    _atomic_publish(path, lambda temporary: temporary.write_bytes(payload))


def _yaml_dump(path: Path, value: Any) -> None:
    payload = yaml.safe_dump(value, sort_keys=True).encode("utf-8")
    _atomic_publish(path, lambda temporary: temporary.write_bytes(payload))


def _torch_dump(path: Path, value: Any) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            torch.save(value, handle)

    _atomic_publish(path, write)


def _initialize_store(root: Path) -> None:
    from gecko.data.streams.manifest import STORE_FORMAT
    from gecko.data.streams.manifest import STORE_LOCK
    from gecko.data.streams.manifest import STORE_MARKER
    from gecko.data.streams.integrity import _read_json
    root.mkdir(parents=True, exist_ok=True)
    object_root = root / "objects"
    if object_root.is_symlink():
        raise ArtifactIntegrityError("Refusing a symlink as the object-store root.")
    lock_path = root / STORE_LOCK
    if lock_path.is_symlink():
        raise ArtifactIntegrityError("Refusing a symlink as the object-store lock.")
    lock_path.touch(exist_ok=True)
    marker = root / STORE_MARKER
    if marker.is_symlink():
        raise ArtifactIntegrityError("Refusing a symlink as the object-store marker.")
    if not marker.exists():
        _json_dump(marker, {"format": STORE_FORMAT, "object_root": "objects"})
    elif _read_json(marker, maximum_bytes=64 * 1024) != {
        "format": STORE_FORMAT,
        "object_root": "objects",
    }:
        raise ArtifactIntegrityError(f"Malformed UEFA store marker: {marker}")


def _discover_store_root(stream_path: Path) -> Path:
    from gecko.data.streams.manifest import STORE_FORMAT
    from gecko.data.streams.manifest import STORE_MARKER
    from gecko.data.streams.integrity import _read_json
    for candidate in (stream_path, *stream_path.parents):
        marker = candidate / STORE_MARKER
        if marker.exists() and not marker.is_symlink():
            content = _read_json(marker, maximum_bytes=64 * 1024)
            if content != {"format": STORE_FORMAT, "object_root": "objects"}:
                raise ArtifactIntegrityError(f"Malformed UEFA store marker: {marker}")
            object_root = candidate / "objects"
            if object_root.is_symlink():
                raise ArtifactIntegrityError("Refusing a symlink as the object-store root.")
            return candidate
    raise ArtifactIntegrityError(
        "No UEFA object-store marker was found above the stream path. "
        "Use the explicitly trusted legacy loader only for historical artifacts."
    )


@contextmanager
def _store_lock(root: Path, *, exclusive: bool) -> Iterator[None]:
    from gecko.data.streams.manifest import STORE_LOCK
    lock_path = root / STORE_LOCK
    if lock_path.is_symlink():
        raise ArtifactIntegrityError("Refusing a symlink as the object-store lock.")
    if not lock_path.exists():
        if not exclusive:
            raise ArtifactIntegrityError("UEFA store lock is missing.")
        lock_path.touch(exist_ok=True)
    flags = os.O_RDWR if exclusive else os.O_RDONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags)
    try:
        try:
            import fcntl

            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
        except ImportError:
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        yield
    finally:
        try:
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except ImportError:
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _clear_previous_payload_links(stream_path: Path) -> None:
    manifest_path = stream_path / "manifest.json"
    names = set()
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            names.update(manifest.get("artifact_checksums", {}))
        except (OSError, ValueError):
            pass
    names.update({"manifest.json", "checksums.json"})
    for name in names:
        target = stream_path / name
        if target.is_file() or target.is_symlink():
            target.unlink()


def _cleanup_stale_object_temporaries(directory: Path, *, age_seconds: int = 3600) -> int:
    cutoff = time.time() - age_seconds
    removed = 0
    for temporary in directory.glob(".*.tmp"):
        try:
            if temporary.stat().st_mtime < cutoff:
                temporary.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def _cleanup_store_temporaries(
    root: Path,
    *,
    age_seconds: int = 3600,
    dry_run: bool = False,
) -> int:
    cutoff = time.time() - age_seconds
    removed = 0
    for temporary in root.rglob(".*.tmp"):
        try:
            if temporary.is_file() and temporary.stat().st_mtime < cutoff:
                if not dry_run:
                    temporary.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def _materialize_content_addressed_objects(
    stream_path: Path, root: Path
) -> tuple[Dict[str, Any], Dict[str, str]]:
    from gecko.data.streams.manifest import _canonical_digest
    records: Dict[str, Any] = {}
    grouped: Dict[str, Dict[str, str]] = {}
    for payload in sorted(stream_path.iterdir()):
        if not payload.is_file() or payload.name in {"manifest.json", "checksums.json"}:
            continue
        record = _materialize_payload_object(payload, root)
        records[payload.name] = record
        grouped.setdefault(record["component"], {})[payload.name] = record["sha256"]
    component_digests = {
        component: _canonical_digest(files)
        for component, files in sorted(grouped.items())
    }
    return records, component_digests


def _materialize_payload_object(payload: Path, root: Path) -> Dict[str, Any]:
    from gecko.data.streams.integrity import _checksum
    from gecko.data.streams.manifest import _component_name
    digest = _checksum(payload)
    component = _component_name(payload.name)
    object_path = root / "objects" / component / digest / payload.name
    object_path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_object_temporaries(object_path.parent)
    if object_path.exists():
        if _checksum(object_path) != digest:
            raise ArtifactIntegrityError(
                f"Content-addressed object is corrupt: {object_path}"
            )
    else:
        temporary = object_path.with_name(
            f".{object_path.name}.{uuid.uuid4().hex}.tmp"
        )
        shutil.copy2(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if _checksum(temporary) != digest:
            temporary.unlink(missing_ok=True)
            raise ArtifactIntegrityError(
                f"Content-addressed object copy failed: {payload.name}"
            )
        os.replace(temporary, object_path)
        _fsync_directory(object_path.parent)
    object_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        same_inode = os.path.samefile(payload, object_path)
    except OSError:
        same_inode = False
    if not same_inode:
        payload.unlink()
        try:
            os.link(object_path, payload)
        except OSError:
            shutil.copy2(object_path, payload)
    return {
        "component": component,
        "sha256": digest,
        "size_bytes": payload.stat().st_size,
        "object_key": PurePosixPath(component, digest, payload.name).as_posix(),
    }


def _stream_path(bundle: StreamBundle, root: str | Path | None) -> Path:
    base = Path(root or bundle.config.output_root)
    return base / stream_relative_path(bundle.config)


def _scenario_payload(spec: ScenarioSpec) -> Dict[str, Any]:
    return dict(spec.__dict__)


def _partition_payload(partition: PartitionResult) -> Dict[str, Any]:
    return {
        "node_owner": partition.node_owner,
        "micro_community_ids": partition.micro_community_ids,
        "diagnostics": partition.diagnostics,
        "client_graphs": {
            client: {
                "client_id": graph.client_id,
                "edge_index": graph.edge_index,
                "node_features": graph.node_features,
                "local_to_global": graph.local_to_global,
                "boundary_mask": graph.boundary_mask,
            }
            for client, graph in partition.client_graphs.items()
        },
    }


def _shard_payload(shards: Dict[int, Dict[int, ClientTaskShard]]) -> Dict[int, Any]:
    return {
        client: {task: dict(shard.__dict__) for task, shard in tasks.items()}
        for client, tasks in shards.items()
    }


def _evaluation_payload(bundle: StreamBundle) -> Dict[str, Any]:
    return {
        "labels": bundle.scenario.labels,
        "query_ids_by_task_split": bundle.scenario.query_ids_by_task_split,
        "query_task_ids": bundle.scenario.query_task_ids,
        "client_shards": {
            client: {
                task: dict(shard.__dict__) for task, shard in tasks.items()
            }
            for client, tasks in bundle.evaluation_shards.items()
        },
    }


def _internal_query_owner(bundle: StreamBundle) -> torch.Tensor:
    spec = bundle.scenario
    owner = bundle.partition.node_owner
    if spec.problem_type == "NC":
        return owner.clone()
    assert spec.query_endpoints is not None
    source_owner = owner[spec.query_endpoints[:, 0]]
    target_owner = owner[spec.query_endpoints[:, 1]]
    return torch.where(
        source_owner == target_owner,
        source_owner,
        torch.full_like(source_owner, -1),
    )


def save_stream(
    bundle: StreamBundle,
    root: str | Path | None = None,
    *,
    repository_root: str | Path = ".",
) -> Path:
    """Persist a complete stream before any method execution."""

    storage_root = Path(root or bundle.config.output_root).resolve()
    _initialize_store(storage_root)
    with _store_lock(storage_root, exclusive=True):
        return _save_stream_unlocked(
            bundle,
            storage_root,
            repository_root=repository_root,
        )


def _save_stream_unlocked(
    bundle: StreamBundle,
    root: str | Path,
    *,
    repository_root: str | Path,
) -> Path:
    """Serialize while the caller holds the exclusive store lock."""
    from gecko.benchmarks.paper import scenario_id_for_config
    from gecko.data.streams.manifest import MANIFEST_FORMAT
    from gecko.data.streams.integrity import _checksum
    from gecko.data.streams.manifest import _package_digest
    from gecko.data.streams.manifest import _scientific_fingerprint

    storage_root = Path(root)
    _cleanup_store_temporaries(storage_root)
    path = _stream_path(bundle, storage_root)
    path.mkdir(parents=True, exist_ok=True)
    _clear_previous_payload_links(path)
    config_payload = bundle.config.to_dict()
    if bundle.config.benchmark_schema_version >= 2:
        config_payload["output_root"] = "."
    _yaml_dump(path / "config.yaml", config_payload)
    scenario_cache_key = (id(bundle.scenario), str(storage_root.resolve()))
    cached_scenario = _SCENARIO_OBJECT_CACHE.get(scenario_cache_key)
    scenario_linked = False
    if cached_scenario is not None and cached_scenario[0]() is bundle.scenario:
        record = cached_scenario[1]
        object_path = storage_root / "objects" / record["object_key"]
        if object_path.is_file():
            os.link(object_path, path / "scenario.pt")
            scenario_linked = True
    if not scenario_linked:
        _torch_dump(path / "scenario.pt", _scenario_payload(bundle.scenario))
    _torch_dump(path / "ownership.pt", bundle.partition.node_owner)
    _torch_dump(
        path / "client_subgraph_index.pt", _partition_payload(bundle.partition)
    )
    _torch_dump(path / "client_query_shards.pt", _shard_payload(bundle.shards))
    _torch_dump(path / "evaluation.pt", _evaluation_payload(bundle))
    task_json = {
        "num_tasks": bundle.scenario.num_tasks,
        "task_class_sets": {
            str(task): values.tolist()
            for task, values in bundle.scenario.task_class_sets.items()
        },
    }
    _json_dump(path / "tasks.json", task_json)
    _json_dump(
        path / "orders.json",
        {
            "canonical_order": bundle.orders.canonical_order,
            "client_orders": bundle.orders.client_orders,
            "inverse_client_orders": bundle.orders.inverse_client_orders,
            "cohort_assignments": bundle.orders.cohort_assignments,
            "block_size": bundle.orders.block_size,
            "seed": bundle.orders.seed,
            "diagnostics": bundle.orders.diagnostics,
        },
    )
    _json_dump(
        path / "participation.json",
        {
            "trace": bundle.participation.trace,
            "fraction": bundle.participation.fraction,
            "rounds_per_stage": bundle.participation.rounds_per_stage,
            "seed": bundle.participation.seed,
        },
    )
    _json_dump(path / "audit.json", bundle.partition.diagnostics)
    environment = environment_snapshot()
    repository = repository_provenance(repository_root)
    _json_dump(path / "environment.json", environment)
    if bundle.scenario.problem_type == "LC":
        assert bundle.scenario.query_endpoints is not None
        _torch_dump(path / "logical_edges.pt", bundle.scenario.query_endpoints)
        _torch_dump(
            path / "logical_edge_to_raw_edges.pt",
            bundle.scenario.logical_edge_to_raw_edges,
        )
        _torch_dump(path / "logical_edge_owner.pt", _internal_query_owner(bundle))
        if torch.is_tensor(bundle.scenario.metadata.get("scores")):
            _torch_dump(
                path / "edge_domain_scores.pt",
                bundle.scenario.metadata["scores"],
            )
        _json_dump(
            path / "edge_domain_metadata.json",
            {
                key: value
                for key, value in bundle.scenario.metadata.items()
                if not torch.is_tensor(value)
            },
        )
    if bundle.scenario.problem_type == "LP":
        _torch_dump(path / "context_edges.pt", bundle.scenario.context_edge_index)
        positive_pairs = bundle.scenario.metadata["positive_pairs"]
        _torch_dump(
            path / "known_positive_pairs.pt",
            bundle.scenario.metadata.get("known_positive_pairs", positive_pairs),
        )
        positive_splits = bundle.scenario.metadata["positive_splits"]
        positive_tasks = bundle.scenario.metadata["positive_task_ids"]
        for split, code in (("train", 0), ("validation", 1), ("test", 2)):
            _torch_dump(
                path / f"positive_{split}.pt",
                positive_pairs[positive_splits == code],
            )
        for split in ("train", "validation", "test"):
            code = {"train": "train", "validation": "val", "test": "test"}[split]
            negatives = []
            for task in range(bundle.scenario.num_tasks):
                ids = bundle.scenario.query_ids_by_task_split[task][code]
                labels = bundle.scenario.labels[ids]
                negatives.append(bundle.scenario.query_endpoints[ids][labels == 0])
            combined = (
                torch.cat(negatives, dim=0)
                if negatives
                else torch.empty((0, 2), dtype=torch.long)
            )
            _torch_dump(path / f"negative_{split}.pt", combined)
        domain_bundles = {}
        for task in range(bundle.scenario.num_tasks):
            task_bundle = {
                "training_context_edges": positive_pairs[
                    (positive_tasks == task) & (positive_splits == 0)
                ]
            }
            for split in ("train", "val", "test"):
                ids = bundle.scenario.query_ids_by_task_split[task][split]
                task_bundle[f"{split}_queries"] = bundle.scenario.query_endpoints[ids]
                task_bundle[f"{split}_labels"] = bundle.scenario.labels[ids]
            domain_bundles[task] = task_bundle
        _torch_dump(path / "domain_bundles.pt", domain_bundles)
        _torch_dump(
            path / "pair_owner.pt",
            {
                "query_endpoints": bundle.scenario.query_endpoints,
                "query_pair_owner": _internal_query_owner(bundle),
            },
        )
    base_sha = "ff3cbdc4160342488db981a762e1b2c6c81efbc9"
    component_objects: Dict[str, Any] = {}
    component_digests: Dict[str, str] = {}
    if bundle.config.benchmark_schema_version >= 2:
        (
            component_objects,
            component_digests,
        ) = _materialize_content_addressed_objects(path, storage_root)
        scenario_record = component_objects.get("scenario.pt")
        if scenario_record is not None:
            _SCENARIO_OBJECT_CACHE[scenario_cache_key] = (
                weakref.ref(bundle.scenario),
                dict(scenario_record),
            )
    from gecko.data.audits.bias import audit_bias

    bias_status = audit_bias(bundle)
    manifest = {
        "manifest_format": MANIFEST_FORMAT,
        "benchmark_name": "UEFA",
        "benchmark_schema_version": bundle.config.benchmark_schema_version,
        "BeGin_base_commit_SHA": base_sha,
        "repository_current_commit_SHA": repository["commit_sha"],
        "source_provenance_type": repository["provenance_type"],
        "source_identifier": repository["source_identifier"],
        "repository_worktree_clean": repository["worktree_clean"],
        "repository_dirty_status_sha256": repository["dirty_status_sha256"],
        "dataset_identifier": bundle.scenario.dataset_name,
        "dataset_version": bundle.scenario.metadata.get("dataset_version"),
        "raw_dataset_checksums": bundle.scenario.metadata.get(
            "raw_dataset_checksums", {}
        ),
        "processed_cache_checksums": bundle.scenario.metadata.get(
            "processed_cache_checksums", {}
        ),
        "problem": bundle.scenario.problem_type,
        "incremental_setting": bundle.scenario.incremental_type,
        "number_of_clients": bundle.config.partition.num_clients,
        "number_of_tasks": bundle.scenario.num_tasks,
        "scenario_id": scenario_id_for_config(bundle.config),
        "benchmark_seed": bundle.config.seed,
        "spatial_profile": spatial_profile_label(bundle.config),
        "legacy_spatial_profile_field_ignored": (
            bundle.config.partition.dirichlet_alpha is not None
        ),
        "order_profile": bundle.config.order.profile,
        **allocation_metadata(bundle.config),
        "config_hash": bundle.stream_hash,
        "stream_id": bundle.stream_id,
        "creation_timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "supported_scenario_matrix": [
            "NC-Task", "NC-Class", "NC-Domain", "LC-Task", "LC-Class", "LC-Domain", "LP-Domain"
        ],
        "domain_constructor_version": bundle.scenario.metadata.get("constructor_version"),
            "partitioner_version": 7,
        "negative_sampler_version": 1,
        "environment_fingerprint": environment["environment_fingerprint"],
        "lp_protocol": (
            {
                "name": bundle.config.scenario.lp_protocol_name,
                "version": bundle.config.scenario.lp_protocol_version,
                "training_negative_policy": (
                    bundle.config.scenario.lp_training_negative_policy
                ),
                "evaluation_candidate_policy": (
                    bundle.config.scenario.lp_evaluation_candidate_policy
                ),
                "evaluation_grouping": bundle.config.scenario.lp_candidate_grouping,
                "tie_policy": bundle.config.scenario.hits_tie_policy,
                "legacy_score_comparable": (
                    bundle.config.scenario.lp_legacy_score_comparable
                ),
                "partition_information_scope": (
                    bundle.config.partition.lp_partition_information_scope
                ),
                "evaluation_aware_partition": (
                    bundle.config.partition.lp_partition_information_scope
                    == "all_positive_splits"
                ),
                **(
                    {
                        "query_split_protocol": (
                            bundle.config.scenario.lp_query_split_protocol
                        ),
                        "base_topology_policy": (
                            bundle.config.scenario.lp_base_topology_policy
                        ),
                        "base_edge_ratio": bundle.config.scenario.lp_base_edge_ratio,
                        "source_num_domains": (
                            bundle.config.scenario.lp_source_num_domains
                        ),
                        "domain_mapping": bundle.config.scenario.lp_domain_mapping,
                        "evaluation_negatives_per_client_task": (
                            bundle.config.scenario.lp_evaluation_negatives_per_client_task
                        ),
                    }
                    if bundle.config.scenario.lp_protocol_version >= 2
                    else {}
                ),
            }
            if bundle.scenario.problem_type == "LP"
            else None
        ),
        "generation_timings": bundle.timings,
        "dataset_design_audit": bundle.partition.diagnostics.get(
            "dataset_design_audit", {}
        ),
        "audit_policy_version": AUDIT_POLICY_VERSION,
        "audit_policy_hash": audit_policy_hash(),
        "invariant_status": bias_status["invariant_status"],
        "quality_status": bias_status["quality_status"],
        "release_status": bias_status["release_status"],
        "release_eligibility": bias_status["release_eligible"],
        "benchmark_tier": bias_status["benchmark_tier"],
        "benchmark_eligible": bias_status["benchmark_eligible"],
        "leaderboard_ready": bias_status["leaderboard_ready"],
        "release_blockers": bias_status["blockers"],
        "quality_warnings": bias_status["warnings"],
        "artifact_layout": (
            "content_addressed_hardlink_v2"
            if bundle.config.benchmark_schema_version >= 2
            else "self_contained_v1"
        ),
        "object_store": {
            "scope": "storage_root",
            "root_name": "objects",
            "key_format": "component/sha256/filename",
        },
        "component_objects": component_objects,
        "component_digests": component_digests,
    }
    manifest["artifact_checksums"] = {
        file.name: _checksum(file)
        for file in sorted(path.iterdir())
        if file.is_file() and file.name not in {"manifest.json", "checksums.json"}
    }
    manifest["artifact_sizes"] = {
        file.name: file.stat().st_size
        for file in sorted(path.iterdir())
        if file.is_file() and file.name not in {"manifest.json", "checksums.json"}
    }
    manifest["scientific_fingerprint"] = _scientific_fingerprint(manifest)
    manifest["package_digest_scope"] = (
        "canonical provenance manifest and all payload-file SHA-256 digests; "
        "manifest.json and checksums.json excluded to avoid self-reference"
    )
    manifest["package_digest"] = _package_digest(manifest)
    manifest["signature"] = {
        "mode": "unsigned_development",
        "scheme": None,
        "identity": None,
        "signature": None,
        "signed_package_digest": None,
    }
    _json_dump(path / "manifest.json", manifest)
    checksums = {
        file.name: _checksum(file)
        for file in sorted(path.iterdir())
        if file.is_file() and file.name != "checksums.json"
    }
    _json_dump(path / "checksums.json", checksums)
    return path


def derive_order_variant(
    source: str | Path,
    order_profile: str,
    *,
    root: str | Path | None = None,
    repository_root: str | Path = ".",
    _source_verified: bool = False,
) -> Path:
    """Publish an order-only schema-v2 variant without reserializing graph tensors."""
    from gecko.data.streams.manifest import MAX_MANIFEST_BYTES
    from gecko.data.streams.integrity import _audit_stream_unlocked
    from gecko.data.streams.manifest import _canonical_digest
    from gecko.data.streams.integrity import _checksum
    from gecko.data.streams.manifest import _package_digest
    from gecko.data.streams.integrity import _read_json
    from gecko.data.streams.integrity import _resolve_public_stream_path
    from gecko.data.streams.manifest import _scientific_fingerprint
    from gecko.data.streams.integrity import _validate_object_key

    started = time.perf_counter()
    source_path = _resolve_public_stream_path(source)
    source_root = _discover_store_root(source_path)
    target_root = Path(root).resolve() if root is not None else source_root
    if target_root != source_root:
        raise ValueError(
            "Order-only derivation currently requires source and target in one "
            "content-addressed store."
        )
    source_manifest = _read_json(
        source_path / "manifest.json", maximum_bytes=MAX_MANIFEST_BYTES
    )
    if source_manifest["benchmark_schema_version"] != 2:
        raise ValueError("Order-only derivation requires a schema-v2 source stream.")

    source_config = GECKOConfig.from_yaml(source_path / "config.yaml")
    config = replace(
        source_config,
        order=replace(source_config.order, profile=order_profile),
        output_root=".",
    )
    config.validate()
    if source_config.order.profile == order_profile:
        raise ValueError("Order-only derivation requires a different order profile.")
    tasks = _read_json(source_path / "tasks.json")
    participation = _read_json(source_path / "participation.json")
    num_tasks = int(tasks["num_tasks"])
    orders = generate_client_orders(
        config.partition.num_clients,
        num_tasks,
        order_profile,
        config.seed,
    )
    trace = participation["trace"]
    participating_tasks = {
        int(stage): {
            int(round_id): dict(
                sorted(
                    Counter(
                        orders.global_task(int(client_id), int(stage))
                        for client_id in clients
                    ).items()
                )
            )
            for round_id, clients in rounds.items()
        }
        for stage, rounds in trace.items()
    }
    orders = replace(
        orders,
        diagnostics={
            **orders.diagnostics,
            "participating_client_task_distribution_by_round": participating_tasks,
        },
    )
    stream_id, stream_hash = stream_identity(config)
    target_path = target_root / stream_relative_path(config)
    if target_path == source_path:
        raise ValueError("Order-only derivation cannot overwrite its source stream.")

    _initialize_store(target_root)
    with _store_lock(target_root, exclusive=True):
        if not _source_verified:
            _audit_stream_unlocked(
                source_path,
                public_safe=True,
                signature_policy="allow_unsigned",
                signature_verifier=None,
                store_root=source_root,
            )
        target_path.mkdir(parents=True, exist_ok=True)
        source_names = set(source_manifest["artifact_checksums"])
        for name in source_names | {"manifest.json", "checksums.json"}:
            destination = target_path / name
            if destination.is_file() or destination.is_symlink():
                destination.unlink()

        changed = {"config.yaml", "orders.json"}
        records: Dict[str, Dict[str, Any]] = {}
        for name, source_record in source_manifest["component_objects"].items():
            if name in changed:
                continue
            component = source_record["component"]
            key = _validate_object_key(
                source_record["object_key"], component=component, name=name
            )
            object_path = target_root / "objects" / key
            destination = target_path / name
            try:
                os.link(object_path, destination)
            except OSError:
                shutil.copy2(object_path, destination)
            records[name] = dict(source_record)

        config_payload = config.to_dict()
        config_payload["output_root"] = "."
        _yaml_dump(target_path / "config.yaml", config_payload)
        _json_dump(
            target_path / "orders.json",
            {
                "canonical_order": orders.canonical_order,
                "client_orders": orders.client_orders,
                "inverse_client_orders": orders.inverse_client_orders,
                "cohort_assignments": orders.cohort_assignments,
                "block_size": orders.block_size,
                "seed": orders.seed,
                "diagnostics": orders.diagnostics,
            },
        )
        for name in sorted(changed):
            records[name] = _materialize_payload_object(target_path / name, target_root)

        grouped: Dict[str, Dict[str, str]] = {}
        for name, record in records.items():
            grouped.setdefault(record["component"], {})[name] = record["sha256"]
        component_digests = {
            component: _canonical_digest(files)
            for component, files in sorted(grouped.items())
        }
        repository = repository_provenance(repository_root)
        manifest = dict(source_manifest)
        manifest.update(
            {
                "repository_current_commit_SHA": repository["commit_sha"],
                "source_provenance_type": repository["provenance_type"],
                "source_identifier": repository["source_identifier"],
                "repository_worktree_clean": repository["worktree_clean"],
                "repository_dirty_status_sha256": repository[
                    "dirty_status_sha256"
                ],
                "order_profile": order_profile,
                **allocation_metadata(config),
                "config_hash": stream_hash,
                "stream_id": stream_id,
                "creation_timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "component_objects": records,
                "component_digests": component_digests,
                "artifact_checksums": {
                    name: record["sha256"] for name, record in sorted(records.items())
                },
                "artifact_sizes": {
                    name: record["size_bytes"]
                    for name, record in sorted(records.items())
                },
                "generation_timings": {
                    **source_manifest.get("generation_timings", {}),
                    "order_variant_derivation_seconds": time.perf_counter() - started,
                },
                "signature": {
                    "mode": "unsigned_development",
                    "scheme": None,
                    "identity": None,
                    "signature": None,
                    "signed_package_digest": None,
                },
            }
        )
        manifest["scientific_fingerprint"] = _scientific_fingerprint(manifest)
        manifest["package_digest"] = _package_digest(manifest)
        _json_dump(target_path / "manifest.json", manifest)
        checksums = {
            file.name: _checksum(file)
            for file in sorted(target_path.iterdir())
            if file.is_file() and file.name != "checksums.json"
        }
        _json_dump(target_path / "checksums.json", checksums)
    return target_path


def derive_order_variants(
    source: str | Path,
    order_profiles: list[str] | tuple[str, ...],
    *,
    root: str | Path | None = None,
    repository_root: str | Path = ".",
) -> list[Path]:
    """Safely derive multiple order profiles after one full source audit."""

    profiles = tuple(order_profiles)
    if not profiles:
        raise ValueError("At least one order profile is required.")
    if len(set(profiles)) != len(profiles):
        raise ValueError("Order profiles must be unique within one derivation batch.")
    return [
        derive_order_variant(
            source,
            profile,
            root=root,
            repository_root=repository_root,
            _source_verified=index > 0,
        )
        for index, profile in enumerate(profiles)
    ]


def derive_participation_variant(
    source: str | Path,
    rounds_per_stage: int,
    *,
    root: str | Path,
    repository_root: str | Path = ".",
) -> Path:
    """Publish a participation-only schema-v2 variant in a separate store.

    Data-bearing payloads are copied byte-for-byte from the audited source.
    Only ``config.yaml``, ``participation.json``, and the participation-derived
    diagnostics inside ``orders.json`` are rewritten.  A separate target store
    is mandatory because stream paths intentionally do not encode the training
    budget and an in-place derivation would overwrite the immutable source.
    """
    from gecko.data.streams.manifest import MAX_MANIFEST_BYTES
    from gecko.data.streams.integrity import _audit_stream_unlocked
    from gecko.data.streams.integrity import _checksum
    from gecko.data.streams.manifest import _package_digest
    from gecko.data.streams.integrity import _read_json
    from gecko.data.streams.integrity import _resolve_public_stream_path
    from gecko.data.streams.integrity import _safe_payload_path
    from gecko.data.streams.manifest import _scientific_fingerprint
    from gecko.data.streams.integrity import audit_stream

    if isinstance(rounds_per_stage, bool) or rounds_per_stage < 1:
        raise ValueError("rounds_per_stage must be a positive integer.")
    started = time.perf_counter()
    source_path = _resolve_public_stream_path(source)
    source_root = _discover_store_root(source_path)
    target_root = Path(root).resolve()
    if target_root == source_root:
        raise ValueError(
            "Participation-only derivation requires a separate target store; "
            "the source stream is immutable."
        )
    source_manifest = _read_json(
        source_path / "manifest.json", maximum_bytes=MAX_MANIFEST_BYTES
    )
    if source_manifest["benchmark_schema_version"] != 2:
        raise ValueError(
            "Participation-only derivation requires a schema-v2 source stream."
        )
    audit_stream(source_path)

    source_config = GECKOConfig.from_yaml(source_path / "config.yaml")
    old_rounds = source_config.training.rounds_per_stage
    if old_rounds == rounds_per_stage:
        raise ValueError(
            "Participation-only derivation requires a different rounds_per_stage."
        )
    config = replace(
        source_config,
        training=replace(
            source_config.training,
            rounds_per_stage=int(rounds_per_stage),
        ),
        output_root=".",
    )
    config.validate()
    tasks = _read_json(source_path / "tasks.json")
    num_tasks = int(tasks["num_tasks"])
    participation = generate_participation(
        num_clients=config.partition.num_clients,
        num_stages=num_tasks,
        rounds_per_stage=rounds_per_stage,
        fraction=config.training.participation_fraction,
        seed=config.seed,
    )
    old_participation = _read_json(source_path / "participation.json")
    for stage, old_trace in old_participation["trace"].items():
        for round_id, clients in old_trace.items():
            if list(participation.trace[int(stage)][int(round_id)]) != list(clients):
                raise ArtifactIntegrityError(
                    "Canonical participation extension changed an existing trace prefix."
                )

    orders = _read_json(source_path / "orders.json")
    participating_tasks = {
        int(stage): {
            int(round_id): dict(
                sorted(
                    Counter(
                        int(orders["client_orders"][str(client_id)][int(stage)])
                        for client_id in clients
                    ).items()
                )
            )
            for round_id, clients in rounds.items()
        }
        for stage, rounds in participation.trace.items()
    }
    order_payload = {
        **orders,
        "diagnostics": {
            **orders["diagnostics"],
            "participating_client_task_distribution_by_round": participating_tasks,
        },
    }
    stream_id, stream_hash = stream_identity(config)
    target_path = target_root / stream_relative_path(config)

    _initialize_store(target_root)
    with _store_lock(target_root, exclusive=True):
        if target_path.exists():
            if not (target_path / "manifest.json").is_file():
                raise ArtifactIntegrityError(
                    f"Refusing to overwrite an incomplete target stream: {target_path}"
                )
            existing = _audit_stream_unlocked(
                target_path,
                public_safe=True,
                signature_policy="allow_unsigned",
                signature_verifier=None,
                store_root=target_root,
            )
            manifest = _read_json(
                target_path / "manifest.json", maximum_bytes=MAX_MANIFEST_BYTES
            )
            lineage = manifest.get("derived_from", {})
            if (
                manifest.get("config_hash") == stream_hash
                and lineage.get("package_digest") == source_manifest["package_digest"]
                and existing.get("valid") is True
            ):
                return target_path
            raise ArtifactIntegrityError(
                f"Refusing to overwrite a different target stream: {target_path}"
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.mkdir()
        try:
            changed = {"config.yaml", "orders.json", "participation.json"}
            for name in source_manifest["artifact_checksums"]:
                if name in changed:
                    continue
                source_payload = _safe_payload_path(source_path, name)
                destination = temporary / name
                try:
                    os.link(source_payload, destination)
                except OSError:
                    shutil.copy2(source_payload, destination)

            config_payload = config.to_dict()
            config_payload["output_root"] = "."
            _yaml_dump(temporary / "config.yaml", config_payload)
            _json_dump(temporary / "orders.json", order_payload)
            _json_dump(
                temporary / "participation.json",
                {
                    "trace": participation.trace,
                    "fraction": participation.fraction,
                    "rounds_per_stage": participation.rounds_per_stage,
                    "seed": participation.seed,
                },
            )
            records, component_digests = _materialize_content_addressed_objects(
                temporary, target_root
            )
            repository = repository_provenance(repository_root)
            manifest = dict(source_manifest)
            manifest.update(
                {
                    "repository_current_commit_SHA": repository["commit_sha"],
                    "source_provenance_type": repository["provenance_type"],
                    "source_identifier": repository["source_identifier"],
                    "repository_worktree_clean": repository["worktree_clean"],
                    "repository_dirty_status_sha256": repository[
                        "dirty_status_sha256"
                    ],
                    "config_hash": stream_hash,
                    **allocation_metadata(config),
                    "stream_id": stream_id,
                    "creation_timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "component_objects": records,
                    "component_digests": component_digests,
                    "artifact_checksums": {
                        name: record["sha256"]
                        for name, record in sorted(records.items())
                    },
                    "artifact_sizes": {
                        name: record["size_bytes"]
                        for name, record in sorted(records.items())
                    },
                    "derived_from": {
                        "operation": "participation_round_extension_v1",
                        "stream_id": source_manifest["stream_id"],
                        "config_hash": source_manifest["config_hash"],
                        "scientific_fingerprint": source_manifest[
                            "scientific_fingerprint"
                        ],
                        "package_digest": source_manifest["package_digest"],
                        "rounds_per_stage": old_rounds,
                    },
                    "generation_timings": {
                        **source_manifest.get("generation_timings", {}),
                        "participation_variant_derivation_seconds": (
                            time.perf_counter() - started
                        ),
                    },
                    "signature": {
                        "mode": "unsigned_development",
                        "scheme": None,
                        "identity": None,
                        "signature": None,
                        "signed_package_digest": None,
                    },
                }
            )
            manifest["scientific_fingerprint"] = _scientific_fingerprint(manifest)
            manifest["package_digest"] = _package_digest(manifest)
            _json_dump(temporary / "manifest.json", manifest)
            checksums = {
                file.name: _checksum(file)
                for file in sorted(temporary.iterdir())
                if file.is_file() and file.name != "checksums.json"
            }
            _json_dump(temporary / "checksums.json", checksums)
            os.replace(temporary, target_path)
            _fsync_directory(target_path.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return target_path


def _torch_load(path: Path, *, public_safe: bool) -> Any:
    """Load tensors safely by default; pickle-capable loading is trusted-only."""

    return torch.load(
        path,
        map_location="cpu",
        weights_only=public_safe,
    )


def load_stream(
    path: str | Path,
    *,
    verify: bool = True,
    signature_policy: str = "allow_unsigned",
    signature_verifier: SignatureVerifier | None = None,
) -> StreamBundle:
    """Load a public artifact after full validation with weights-only PyTorch IO."""
    from gecko.data.streams.integrity import _audit_stream_unlocked
    from gecko.data.streams.integrity import _resolve_public_stream_path

    if not verify:
        raise ValueError(
            "Public-safe loading cannot disable verification. "
            "Use load_trusted_legacy_stream only for explicitly trusted history."
        )
    stream_path = _resolve_public_stream_path(path)
    store_root = _discover_store_root(stream_path)
    with _store_lock(store_root, exclusive=False):
        _audit_stream_unlocked(
            stream_path,
            public_safe=True,
            signature_policy=signature_policy,
            signature_verifier=signature_verifier,
            store_root=store_root,
        )
        return _load_stream_unlocked(stream_path, public_safe=True)


def load_trusted_legacy_stream(
    path: str | Path,
    *,
    verify: bool = True,
) -> StreamBundle:
    """Load a trusted historical artifact with pickle-capable PyTorch IO.

    Never use this API for downloaded or otherwise untrusted artifacts.
    """
    from gecko.data.streams.integrity import _audit_stream_unlocked

    stream_path = Path(path).resolve()
    if verify:
        _audit_stream_unlocked(
            stream_path,
            public_safe=False,
            signature_policy="ignore",
            signature_verifier=None,
            store_root=None,
        )
    return _load_stream_unlocked(stream_path, public_safe=False)


def _load_stream_unlocked(
    stream_path: Path,
    *,
    public_safe: bool,
) -> StreamBundle:
    config = GECKOConfig.from_yaml(stream_path / "config.yaml")
    scenario = ScenarioSpec(
        **_torch_load(stream_path / "scenario.pt", public_safe=public_safe)
    )
    partition_raw = _torch_load(
        stream_path / "client_subgraph_index.pt", public_safe=public_safe
    )
    client_graphs = {
        int(client): ClientGraph(
            **{
                **raw,
                "node_features": raw.get(
                    "node_features",
                    scenario.node_features[raw["local_to_global"]].clone(),
                ),
                "global_to_local": {
                    int(global_id): local_id
                    for local_id, global_id in enumerate(raw["local_to_global"].tolist())
                },
            }
        )
        for client, raw in partition_raw["client_graphs"].items()
    }
    partition = PartitionResult(
        node_owner=partition_raw["node_owner"],
        client_graphs=client_graphs,
        micro_community_ids=partition_raw["micro_community_ids"],
        diagnostics=partition_raw["diagnostics"],
    )
    shards_raw = _torch_load(
        stream_path / "client_query_shards.pt", public_safe=public_safe
    )
    shards = {
        int(client): {
            int(task): ClientTaskShard(**raw) for task, raw in tasks.items()
        }
        for client, tasks in shards_raw.items()
    }
    evaluation_raw = _torch_load(
        stream_path / "evaluation.pt", public_safe=public_safe
    )
    evaluation_shards = {
        int(client): {
            int(task): CentralEvaluationShard(**raw) for task, raw in tasks.items()
        }
        for client, tasks in evaluation_raw["client_shards"].items()
    }
    orders_raw = json.loads((stream_path / "orders.json").read_text(encoding="utf-8"))
    orders = OrderPlan(
        canonical_order=tuple(orders_raw["canonical_order"]),
        client_orders={int(k): tuple(v) for k, v in orders_raw["client_orders"].items()},
        inverse_client_orders={
            int(k): {int(task): int(stage) for task, stage in v.items()}
            for k, v in orders_raw["inverse_client_orders"].items()
        },
        cohort_assignments={int(k): int(v) for k, v in orders_raw["cohort_assignments"].items()},
        block_size=int(orders_raw["block_size"]),
        seed=int(orders_raw["seed"]),
        diagnostics=orders_raw["diagnostics"],
    )
    participation_raw = json.loads(
        (stream_path / "participation.json").read_text(encoding="utf-8")
    )
    participation = ParticipationPlan(
        trace={
            int(stage): {
                int(round_id): tuple(clients)
                for round_id, clients in rounds.items()
            }
            for stage, rounds in participation_raw["trace"].items()
        },
        fraction=float(participation_raw["fraction"]),
        rounds_per_stage=int(participation_raw["rounds_per_stage"]),
        seed=int(participation_raw["seed"]),
    )
    manifest = json.loads((stream_path / "manifest.json").read_text(encoding="utf-8"))
    return StreamBundle(
        config=config,
        scenario=scenario,
        partition=partition,
        shards=shards,
        evaluation_shards=evaluation_shards,
        orders=orders,
        participation=participation,
        stream_id=manifest["stream_id"],
        stream_hash=manifest["config_hash"],
        timings={
            key: float(value)
            for key, value in manifest.get("generation_timings", {}).items()
        },
    )


def gc_objects(root: str | Path, *, dry_run: bool = True) -> Dict[str, Any]:
    """Find or delete only content-addressed objects unreferenced by manifests."""
    from gecko.data.streams.manifest import STORE_MARKER

    storage_root = Path(root).resolve()
    marker = storage_root / STORE_MARKER
    if not marker.exists():
        object_root = storage_root / "objects"
        if not object_root.exists():
            return {
                "dry_run": dry_run,
                "referenced_objects": 0,
                "unreferenced_objects": 0,
                "deleted_objects": 0,
                "objects": [],
            }
        raise ArtifactIntegrityError(
            "Refusing GC on an object store without a validated UEFA store marker."
        )
    _discover_store_root(storage_root)
    with _store_lock(storage_root, exclusive=True):
        return _gc_objects_unlocked(storage_root, dry_run=dry_run)


def _gc_objects_unlocked(
    storage_root: Path,
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    from gecko.data.streams.manifest import MAX_MANIFEST_BYTES
    from gecko.data.streams.integrity import _read_json
    from gecko.data.streams.integrity import _validate_manifest_schema
    from gecko.data.streams.integrity import _validate_object_key
    stale_temporaries = _cleanup_store_temporaries(
        storage_root,
        dry_run=dry_run,
    )
    object_root = (storage_root / "objects").resolve()
    if not object_root.exists():
        return {
            "dry_run": dry_run,
            "referenced_objects": 0,
            "unreferenced_objects": 0,
            "deleted_objects": 0,
            "stale_temporaries_deleted": stale_temporaries,
            "objects": [],
        }
    referenced: set[Path] = set()
    manifest_count = 0
    for manifest_path in storage_root.rglob("manifest.json"):
        if manifest_path.is_relative_to(object_root):
            continue
        if manifest_path.is_symlink():
            raise ArtifactIntegrityError(f"Refusing symlink manifest during GC: {manifest_path}")
        manifest = _validate_manifest_schema(
            _read_json(manifest_path, maximum_bytes=MAX_MANIFEST_BYTES),
            public_safe=True,
        )
        for name, record in manifest.get("component_objects", {}).items():
            key = _validate_object_key(
                record.get("object_key"),
                component=record["component"],
                name=name,
            )
            candidate = (object_root / key).resolve()
            if not candidate.is_relative_to(object_root):
                raise ArtifactIntegrityError(
                    f"Manifest references an object outside the store: {manifest_path}"
                )
            referenced.add(candidate)
        manifest_count += 1
    objects = {
        path.resolve()
        for path in object_root.rglob("*")
        if path.is_file() and not path.is_symlink() and not path.name.endswith(".tmp")
    }
    unreferenced = sorted(objects - referenced)
    deleted = 0
    if not dry_run:
        for path in unreferenced:
            path.unlink()
            deleted += 1
        for directory in sorted(
            (path for path in object_root.rglob("*") if path.is_dir()),
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
    return {
        "dry_run": dry_run,
        "manifests_scanned": manifest_count,
        "referenced_objects": len(referenced),
        "unreferenced_objects": len(unreferenced),
        "deleted_objects": deleted,
        "stale_temporaries_deleted": stale_temporaries,
        "objects": [path.relative_to(storage_root).as_posix() for path in unreferenced],
    }




_RELOCATED_EXPORTS = {'COMPONENT_FILES': ('gecko.data.streams.manifest', 'COMPONENT_FILES'), 'MANIFEST_FORMAT': ('gecko.data.streams.manifest', 'MANIFEST_FORMAT'), 'MAX_ARTIFACT_BYTES': ('gecko.data.streams.manifest', 'MAX_ARTIFACT_BYTES'), 'MAX_JSON_BYTES': ('gecko.data.streams.manifest', 'MAX_JSON_BYTES'), 'MAX_MANIFEST_BYTES': ('gecko.data.streams.manifest', 'MAX_MANIFEST_BYTES'), 'MAX_STREAM_BYTES': ('gecko.data.streams.manifest', 'MAX_STREAM_BYTES'), 'SCIENTIFIC_ARTIFACT_NAMES': ('gecko.data.streams.manifest', 'SCIENTIFIC_ARTIFACT_NAMES'), 'SIGNATURE_POLICIES': ('gecko.data.streams.manifest', 'SIGNATURE_POLICIES'), 'STORE_FORMAT': ('gecko.data.streams.manifest', 'STORE_FORMAT'), 'STORE_LOCK': ('gecko.data.streams.manifest', 'STORE_LOCK'), 'STORE_MARKER': ('gecko.data.streams.manifest', 'STORE_MARKER'), '_CHECKSUM_CACHE': ('gecko.data.streams.integrity', '_CHECKSUM_CACHE'), '_audit_stream_unlocked': ('gecko.data.streams.integrity', '_audit_stream_unlocked'), '_canonical_digest': ('gecko.data.streams.manifest', '_canonical_digest'), '_checksum': ('gecko.data.streams.integrity', '_checksum'), '_component_name': ('gecko.data.streams.manifest', '_component_name'), '_package_digest': ('gecko.data.streams.manifest', '_package_digest'), '_read_json': ('gecko.data.streams.integrity', '_read_json'), '_resolve_public_stream_path': ('gecko.data.streams.integrity', '_resolve_public_stream_path'), '_safe_payload_path': ('gecko.data.streams.integrity', '_safe_payload_path'), '_scientific_fingerprint': ('gecko.data.streams.manifest', '_scientific_fingerprint'), '_validate_filename': ('gecko.data.streams.integrity', '_validate_filename'), '_validate_manifest_schema': ('gecko.data.streams.integrity', '_validate_manifest_schema'), '_validate_object_key': ('gecko.data.streams.integrity', '_validate_object_key'), '_validate_signature_schema': ('gecko.data.streams.integrity', '_validate_signature_schema'), '_verify_manifest_signature': ('gecko.data.streams.integrity', '_verify_manifest_signature'), 'audit_stream': ('gecko.data.streams.integrity', 'audit_stream'), 'migrate_stream_to_v2': ('gecko.data.streams.migration', 'migrate_stream_to_v2')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)
