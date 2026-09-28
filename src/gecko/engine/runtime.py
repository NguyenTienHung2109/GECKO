"""Opt-in stateful UEFA v2 execution around the unchanged coordinator.

The legacy :class:`FederatedCoordinator` loop remains the default.  This
runtime is constructed only when an explicit method configuration is supplied;
it drives the typed strategy/client lifecycle, split resource accounting, and
safe round/stage checkpointing without changing stream construction.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import random
import time
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping

import numpy as np
import torch

from gecko.evaluation.metrics import summarize_continual_matrix
from gecko.evaluation.evaluator import FederatedEvaluator
from gecko.algorithms.topology import TopologyOverlay
from gecko.models.capabilities import attach_v2_model_capabilities
from gecko.engine.accounting import ResourceLedger
from gecko.engine.accounting import tensor_payload_bytes
from gecko.engine.checkpoint import CheckpointIdentity
from gecko.engine.checkpoint import CheckpointIntegrityError
from gecko.engine.checkpoint import ResumeValidation
from gecko.engine.checkpoint import canonical_json_digest
from gecko.engine.checkpoint import load_checkpoint
from gecko.engine.checkpoint import save_checkpoint
from gecko.algorithms.method_config import validate_method_config
from gecko.engine.parameter_manifest import build_parameter_manifest
from gecko.engine.protocol import AggregationResult
from gecko.engine.protocol import EvaluationSelection
from gecko.engine.protocol import RoundContext
from gecko.engine.protocol import StatefulStrategyProtocol
from gecko.algorithms.federated.legacy import LegacyStrategyAdapter
from gecko.algorithms.federated.scaffold import ScaffoldStrategy


RESULT_SCHEMA = "uefa-run-result-v2"
RUNTIME_VERSION = "uefa-stateful-runtime-v1"
_EVALUATION_MODELS = {
    "strategy",
    "shared",
    "post_broadcast",
    "post_local",
    "personalized",
}


def _json_safe(value: Any, *, path: str = "value") -> Any:
    """Return an owned finite JSON value, rejecting execution objects."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} keys must be non-empty strings.")
            output[key] = _json_safe(item, path=f"{path}.{key}")
        return {key: output[key] for key in sorted(output)}
    if isinstance(value, (tuple, list)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} must contain only finite JSON values; got {type(value).__name__}."
    )


def _thaw_protocol_metadata(value: object) -> object:
    """Recover JSON containers from protocol-frozen metadata tuples."""

    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            for item in value
        ):
            return {
                item[0]: _thaw_protocol_metadata(item[1])
                for item in value
            }
        return [_thaw_protocol_metadata(item) for item in value]
    return value


def _clone_safe(value: Any, *, path: str = "state") -> Any:
    """Clone values accepted by the weights-only checkpoint tree."""

    if torch.is_tensor(value):
        return value.detach().cpu().contiguous().clone()
    if value is None or type(value) in {bool, int, float, str}:
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return value
    if type(value) is dict:
        output: dict[str | int, Any] = {}
        for key, item in value.items():
            if type(key) not in {str, int}:
                raise TypeError(f"{path} keys must be strings or integers.")
            output[key] = _clone_safe(item, path=f"{path}.{key}")
        return output
    if type(value) is list:
        return [
            _clone_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is tuple:
        return tuple(
            _clone_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} contains unsupported {type(value).__name__}; checkpoint state "
        "must use tensors and primitive containers only."
    )


def _safe_equal(first: Any, second: Any) -> bool:
    if torch.is_tensor(first) or torch.is_tensor(second):
        return (
            torch.is_tensor(first)
            and torch.is_tensor(second)
            and first.dtype == second.dtype
            and first.shape == second.shape
            and torch.equal(first.detach().cpu(), second.detach().cpu())
        )
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _safe_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(
            _safe_equal(left, right) for left, right in zip(first, second)
        )
    return bool(first == second)


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _digest_update(digest: "hashlib._Hash", value: Any) -> None:
    """Deterministically hash a nested scientific identity payload."""

    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        return
    if value is None:
        digest.update(b"none\0")
        return
    if type(value) in {bool, int, float, str}:
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(value).encode("utf-8"))
        digest.update(b"\0")
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _digest_update(digest, key)
            _digest_update(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"sequence\0")
        digest.update(str(len(value)).encode("ascii"))
        digest.update(b"\0")
        for item in value:
            _digest_update(digest, item)
        return
    raise TypeError(f"Unsupported scientific identity value: {type(value).__name__}.")


def _derive_scientific_fingerprint(coordinator: Any) -> str:
    """Fingerprint fixed scientific objects for in-memory diagnostic streams."""

    stream = coordinator.stream
    scenario = stream.scenario
    payload = {
        "stream_id": stream.stream_id,
        "stream_hash": stream.stream_hash,
        "scenario": {
            "problem_type": scenario.problem_type,
            "incremental_type": scenario.incremental_type,
            "num_tasks": scenario.num_tasks,
            "num_classes": scenario.num_classes,
            "edge_index": scenario.edge_index,
            "node_features": scenario.node_features,
            "labels": scenario.labels,
            "query_task_ids": scenario.query_task_ids,
            "query_endpoints": scenario.query_endpoints,
            "task_masks": scenario.task_masks,
            "logical_edge_ids": scenario.logical_edge_ids,
            "context_edge_index": scenario.context_edge_index,
            "query_ids_by_task_split": scenario.query_ids_by_task_split,
        },
        "partition": {
            "node_owner": stream.partition.node_owner,
            "client_graphs": {
                client_id: {
                    "edge_index": graph.edge_index,
                    "node_features": graph.node_features,
                    "local_to_global": graph.local_to_global,
                    "boundary_mask": graph.boundary_mask,
                }
                for client_id, graph in stream.partition.client_graphs.items()
            },
        },
        "shards": {
            client_id: {
                task_id: {
                    "queries": shard.train_queries,
                    "labels": shard.train_labels,
                    "mask": shard.task_class_mask,
                    "context": shard.context_edge_index,
                }
                for task_id, shard in tasks.items()
            }
            for client_id, tasks in stream.shards.items()
        },
        "evaluation": {
            client_id: {
                task_id: {
                    "train_ids": shard.train_query_ids,
                    "validation_queries": shard.validation_queries,
                    "validation_ids": shard.validation_query_ids,
                    "test_queries": shard.test_queries,
                    "test_ids": shard.test_query_ids,
                    "context": shard.context_edge_index,
                }
                for task_id, shard in tasks.items()
            }
            for client_id, tasks in stream.evaluation_shards.items()
        },
        "orders": stream.orders.client_orders,
        "participation": stream.participation.trace,
    }
    digest = hashlib.sha256()
    _digest_update(digest, payload)
    return digest.hexdigest()


def _resolve_source_sha(value: str | None) -> str | None:
    """Keep optional caller metadata without inspecting the source checkout."""

    return None if value is None else str(value)


def _update_untracked_runtime_source_digest(
    digest: "hashlib._Hash",
    repository: Path,
    raw_paths: Iterable[bytes],
) -> tuple[str, ...]:
    """Bind nonignored untracked runtime source without following symlinks."""

    rendered_paths: list[str] = []
    for raw_path in sorted(set(raw_paths)):
        if not raw_path or b"\0" in raw_path:
            raise ValueError("Git returned an invalid untracked source path.")
        rendered = raw_path.decode("utf-8", errors="surrogateescape")
        relative = Path(rendered)
        normalized = relative.as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not (
                normalized == "tools/run_gecko.py"
                or normalized.startswith("src/gecko/")
                or normalized.startswith("tools/")
                or normalized.startswith("research/")
            )
        ):
            raise ValueError(f"Unsafe untracked runtime source path: {rendered!r}.")
        source_path = repository / relative
        digest.update(b"untracked-runtime-source\0")
        digest.update(raw_path)
        digest.update(b"\0")
        if source_path.is_symlink():
            target = str(source_path.readlink()).encode(
                "utf-8", errors="surrogateescape"
            )
            digest.update(b"symlink\0")
            digest.update(hashlib.sha256(target).digest())
        elif source_path.is_file():
            digest.update(
                b"executable\0" if source_path.stat().st_mode & 0o111 else b"regular\0"
            )
            content = hashlib.sha256()
            with source_path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    content.update(block)
            digest.update(content.digest())
        else:
            raise ValueError(f"Untracked runtime source is not a file: {rendered!r}.")
        rendered_paths.append(normalized)
    return tuple(rendered_paths)


def _resolve_tracked_source_tree() -> tuple[None, None, tuple[str, ...]]:
    """Return unavailable source metadata without hashing code or querying Git."""

    return None, None, ()


def _model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if not torch.is_tensor(value):
            raise TypeError(
                f"Model state {key!r} is not a tensor and cannot enter a weights-only checkpoint."
            )
        output[key] = value.detach().cpu().contiguous().clone()
    return output


def _module_attribute(model: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    module_name, separator, attribute = name.rpartition(".")
    module = model.get_submodule(module_name) if separator else model
    if not hasattr(module, attribute):
        raise AttributeError(f"Model dynamic state {name!r} is missing.")
    return module, attribute


def _dynamic_model_state(model: torch.nn.Module, manifest: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for entry in manifest.entries:
        if entry.kind != "dynamic_metadata" or not entry.checkpoint_persistent:
            continue
        module, attribute = _module_attribute(model, entry.name)
        output[entry.name] = _clone_safe(
            getattr(module, attribute), path=f"model.{entry.name}"
        )
    return output


def _move_dynamic(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone().to(device)
    if isinstance(value, dict):
        return {key: _move_dynamic(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_dynamic(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_dynamic(item, device) for item in value)
    return value


def _restore_dynamic_model_state(
    model: torch.nn.Module, manifest: Any, state: Mapping[str, Any]
) -> None:
    expected = {
        entry.name
        for entry in manifest.entries
        if entry.kind == "dynamic_metadata" and entry.checkpoint_persistent
    }
    if set(state) != expected:
        raise CheckpointIntegrityError(
            "Checkpoint dynamic-model fields do not match the parameter manifest."
        )
    device = next(model.parameters()).device
    for name, value in state.items():
        module, attribute = _module_attribute(model, name)
        setattr(module, attribute, _move_dynamic(value, device))


def _overlay_state(value: Any, *, path: str) -> Any:
    if isinstance(value, TopologyOverlay):
        return {
            "__uefa_type__": "topology_overlay_v1",
            "client_id": value.client_id,
            "global_task_id": value.global_task_id,
            "method_name": value.method_name,
            "num_nodes": value.num_nodes,
            "base_edge_sha256": value.base_edge_sha256,
            "overlay_id": value.overlay_id,
            "undirected": value.undirected,
            "added_edge_index": value.added_edge_index,
            "deleted_edge_index": value.deleted_edge_index,
        }
    return _clone_safe(value, path=path)


def _restore_overlay(value: Any, client: Any) -> Any:
    if (
        not isinstance(value, Mapping)
        or value.get("__uefa_type__") != "topology_overlay_v1"
    ):
        return _clone_safe(value, path="topology_overlay")
    expected = {
        "__uefa_type__",
        "client_id",
        "global_task_id",
        "method_name",
        "num_nodes",
        "base_edge_sha256",
        "overlay_id",
        "undirected",
        "added_edge_index",
        "deleted_edge_index",
    }
    if set(value) != expected or int(value["client_id"]) != client.client_id:
        raise CheckpointIntegrityError(
            "Malformed or cross-client topology overlay state."
        )
    restored = TopologyOverlay(
        client_id=int(value["client_id"]),
        global_task_id=int(value["global_task_id"]),
        method_name=str(value["method_name"]),
        num_nodes=int(value["num_nodes"]),
        base_edge_index=client.graph.edge_index,
        added_edge_index=value["added_edge_index"],
        deleted_edge_index=value["deleted_edge_index"],
        undirected=bool(value["undirected"]),
    )
    if (
        restored.base_edge_sha256 != value["base_edge_sha256"]
        or restored.overlay_id != value["overlay_id"]
    ):
        raise CheckpointIntegrityError("Topology overlay checkpoint digest mismatch.")
    return restored


class StatefulFederatedRuntime:
    """Execute an explicit stateful v2 run using an existing coordinator."""

    def __init__(
        self,
        coordinator: Any,
        *,
        method_config: Mapping[str, Any],
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int | None = None,
        resume_from: str | Path | None = None,
        evaluation_model: str = "strategy",
        stream_scientific_fingerprint: str | None = None,
        source_sha: str | None = None,
        diagnostic_resume_override: str | None = None,
    ) -> None:
        if not isinstance(method_config, Mapping):
            raise TypeError("method_config must be a mapping for the explicit v2 path.")
        if coordinator.strategy.oracle:
            raise ValueError(
                "The stateful v2 runtime does not adapt oracle strategies."
            )
        self.coordinator = coordinator
        attach_v2_model_capabilities(self.coordinator.global_model)
        for client in self.coordinator.clients.values():
            attach_v2_model_capabilities(client.model)
        self.method_config = _json_safe(method_config, path="method_config")
        self.method_resolution = validate_method_config(
            self.method_config,
            expected_strategy=coordinator.strategy_name,
            expected_continual_method=coordinator.algorithm_name,
            problem_type=coordinator.stream.scenario.problem_type,
            incremental_setting=coordinator.stream.scenario.incremental_type,
        )
        self.method_config_digest = canonical_json_digest(self.method_config)
        self.checkpoint_dir = (
            None if checkpoint_dir is None else Path(checkpoint_dir).absolute()
        )
        if checkpoint_every is not None and (
            isinstance(checkpoint_every, bool)
            or not isinstance(checkpoint_every, int)
            or checkpoint_every <= 0
        ):
            raise ValueError("checkpoint_every must be a positive round interval.")
        if checkpoint_every is not None and self.checkpoint_dir is None:
            raise ValueError("checkpoint_every requires checkpoint_dir.")
        self.checkpoint_every = checkpoint_every
        self.resume_from = None if resume_from is None else Path(resume_from).absolute()
        normalized_evaluation = str(evaluation_model).lower().replace("-", "_")
        if normalized_evaluation == "global":
            normalized_evaluation = "shared"
        if normalized_evaluation not in _EVALUATION_MODELS:
            raise ValueError(
                "evaluation_model must be strategy/shared/post-broadcast/"
                "post-local/personalized."
            )
        self.evaluation_model = normalized_evaluation
        self.source_sha = _resolve_source_sha(source_sha)
        self.source_provenance_type = (
            "provided" if self.source_sha is not None else "unavailable"
        )
        (
            self.source_tree_digest,
            self.tracked_source_tree_clean,
            self.untracked_runtime_source_paths,
        ) = _resolve_tracked_source_tree()
        self.source_tree_clean = (
            self.tracked_source_tree_clean and not self.untracked_runtime_source_paths
        )
        self.stream_content_digest = _derive_scientific_fingerprint(coordinator)
        if stream_scientific_fingerprint is None:
            self.stream_scientific_fingerprint = self.stream_content_digest
            self.stream_scientific_fingerprint_source = "derived_in_memory"
        else:
            self.stream_scientific_fingerprint = str(
                stream_scientific_fingerprint
            ).lower()
            self.stream_scientific_fingerprint_source = "stream_manifest"
        run_config = {
            "runtime": RUNTIME_VERSION,
            "uefa_config": coordinator.stream.config.to_dict(),
            "strategy": coordinator.strategy_name,
            "algorithm": coordinator.algorithm_name,
            "algorithm_parameters": dict(coordinator.algorithm_parameters),
            "requested_model": coordinator.model_name,
            "resolved_model": coordinator.resolved_model_name,
            "model_seed": coordinator.resolved_model_seed,
            "device": str(coordinator.device),
            "reference_mode": coordinator.reference_mode,
            "evaluation_model": self.evaluation_model,
            "class_mask_policy": coordinator.class_mask_policy,
            "deterministic_execution": dict(
                coordinator.deterministic_execution
            ),
        }
        self.run_config_digest = canonical_json_digest(_json_safe(run_config))
        self.identity = CheckpointIdentity(
            stream_id=coordinator.stream.stream_id,
            stream_hash=coordinator.stream.stream_hash,
            scientific_fingerprint=self.stream_scientific_fingerprint,
            source_sha=self.source_sha,
            stream_content_digest=self.stream_content_digest,
            source_tree_digest=self.source_tree_digest,
            run_config_digest=self.run_config_digest,
            method_config_digest=self.method_config_digest,
            strategy=coordinator.strategy_name,
            method=coordinator.algorithm_name,
            model=coordinator.resolved_model_name,
            model_seed=coordinator.resolved_model_seed,
        )
        self.strategy = self._build_strategy()
        bind_training_protocol = getattr(self.strategy, "bind_training_protocol", None)
        if callable(bind_training_protocol):
            training = coordinator.stream.config.training
            bind_training_protocol(
                learning_rate=training.learning_rate,
                local_steps=training.local_epochs_per_round,
            )
        bind_runtime = getattr(self.strategy, "bind_runtime", None)
        if callable(bind_runtime):
            bind_runtime(
                model_template=coordinator.global_model,
                feature_dim=coordinator.stream.scenario.num_features,
                num_classes=coordinator.stream.scenario.num_classes,
                model_seed=coordinator.resolved_model_seed,
                device=coordinator.device,
            )
        validate_stream = getattr(self.strategy, "validate_stream", None)
        if callable(validate_stream):
            validate_stream(coordinator.stream)
        if not isinstance(self.strategy, StatefulStrategyProtocol):
            raise TypeError(
                "Stateful strategy does not implement the required protocol."
            )
        self.parameter_manifest = build_parameter_manifest(
            coordinator.global_model,
            coordinator.parameter_policy,
        )
        self.strategy.initialize(
            coordinator.global_state,
            self.parameter_manifest,
            coordinator.clients,
        )

        num_clients = coordinator.stream.config.partition.num_clients
        num_tasks = coordinator.stream.scenario.num_tasks
        self.matrix = torch.full((num_clients, num_tasks, num_tasks), float("nan"))
        self.query_counts = torch.zeros(num_clients, num_tasks)
        self.round_records: list[dict[str, Any]] = []
        self.stage_records: list[dict[str, Any]] = []
        self.resource_ledger = ResourceLedger()
        self.personalized_states: dict[int, dict[str, torch.Tensor]] = {}
        self.checkpoint_records: list[dict[str, Any]] = []
        self.checkpoint_io_records: list[dict[str, Any]] = []
        self.begun_tasks: set[tuple[int, int]] = set()
        self.cursor: dict[str, Any] = {
            "boundary": "initial",
            "next_stage": 0,
            "next_round": 0,
            "stage_participants": [],
            "global_step": 0,
            "stage_elapsed_seconds": 0.0,
        }
        self.resume_validation = ResumeValidation(
            identity_matched=True,
            diagnostic_override_used=False,
            benchmark_eligible=True,
            mismatch_fields={},
            diagnostic_override_reason=None,
        )
        self._prior_peak_memory_bytes = 0
        self._runtime_accumulated_seconds = 0.0
        self._session_start: float | None = None
        self._has_run = False
        self._fresh_initialized = False
        self._stream_versions_start = coordinator._stream_tensor_versions()
        self._stream_guard_start = self._stream_content_guard()
        if self.resume_from is not None:
            loaded = load_checkpoint(
                self.resume_from,
                expected_identity=self.identity,
                diagnostic_override_reason=diagnostic_resume_override,
            )
            self.resume_validation = loaded.resume_validation
            self._restore_checkpoint(loaded.state, loaded.cursor)
            checkpoint_id = loaded.manifest_path.name.removesuffix(".manifest.json")
            if any(
                record.get("event") == "save"
                and record.get("checkpoint_id") == checkpoint_id
                for record in self.checkpoint_records
            ):
                raise CheckpointIntegrityError(
                    "Checkpoint artifact history already contains its own manifest."
                )
            artifact_record = {
                "event": "save",
                "checkpoint_id": checkpoint_id,
                "manifest_file": loaded.manifest_path.name,
                "weights_file": loaded.weights_path.name,
                "manifest_bytes": loaded.manifest_bytes,
                "weights_bytes": loaded.weights_bytes,
                "total_bytes": loaded.total_bytes,
                "manifest_sha256": loaded.manifest_sha256,
                "weights_sha256": loaded.weights_sha256,
            }
            self.resource_ledger.add(server_checkpoint_bytes=loaded.total_bytes)
            self.checkpoint_records.append(artifact_record)
            self.checkpoint_io_records.append({**artifact_record, "event": "load"})

    def _build_strategy(self) -> StatefulStrategyProtocol:
        """Use an injected v2 strategy when present, otherwise adapt v1."""

        factory = getattr(self.coordinator, "build_stateful_strategy", None)
        if callable(factory):
            return factory(self.method_config)
        injected = getattr(self.coordinator, "stateful_strategy", None)
        if injected is not None:
            return injected
        if self.method_resolution.name in {
            "gem_uefa_v1",
            "twp_uefa_v1",
            "ssm_uefa_v1",
            "cat_uefa_v1",
            "graphkeeper_uefa_v1",
            "dslr_uefa_v1",
            "dslr_normalized_v1",
            "dslr_diagnostic_v1",
        }:
            strategy_name = self.method_resolution.strategy_name
            if strategy_name == "scaffold":
                return ScaffoldStrategy(
                    correction_enabled=True,
                    control_updates_enabled=True,
                )
            return LegacyStrategyAdapter(strategy_name)
        if self.method_resolution.name == "fedgta_uefa_v1":
            from gecko.algorithms.federated.fedgta import FedGTAStrategy

            return FedGTAStrategy(
                **dict(self.method_resolution.strategy_parameters)
            )
        if self.method_resolution.name == "fed_pub_uefa_v1":
            from gecko.algorithms.federated.fedpub import FedPUBStrategy

            return FedPUBStrategy(
                **dict(self.method_resolution.strategy_parameters)
            )
        if self.method_resolution.name == "feddc_uefa_v1":
            from gecko.algorithms.federated.feddc import FedDCStrategy

            return FedDCStrategy(
                **dict(self.method_resolution.strategy_parameters)
            )
        if self.method_resolution.name == "power_uefa_v1":
            from gecko.algorithms.federated.power.gecko import PowerUEFAStrategy

            return PowerUEFAStrategy(
                **dict(self.method_resolution.strategy_parameters)
            )
        if self.method_resolution.name == "motion_uefa_v1":
            from gecko.algorithms.federated.motion.strategy import MotionStrategy

            return MotionStrategy(
                **dict(self.method_resolution.strategy_parameters)
            )
        if self.method_resolution.name == "fedfst_uefa_v1":
            from gecko.algorithms.federated.fedfst.strategy import FedFSTStrategy

            return FedFSTStrategy(
                **dict(self.method_resolution.strategy_parameters)
            )
        if self.method_resolution.name == "scaffold_uefa_adam_v1":
            parameters = dict(self.method_resolution.strategy_parameters)
            correction_enabled = parameters["correction_enabled"]
            control_updates_enabled = parameters["control_updates_enabled"]
            if not correction_enabled and not control_updates_enabled:
                return LegacyStrategyAdapter("fedavg")
            return ScaffoldStrategy(
                correction_enabled=correction_enabled,
                control_updates_enabled=control_updates_enabled,
            )
        if self.method_resolution.name != "legacy_adapter_v1":
            raise RuntimeError(
                "The validated v2 method has no integrated strategy factory: "
                f"{self.method_resolution.name!r}."
            )
        return LegacyStrategyAdapter(self.method_resolution.strategy_name)

    def _stream_content_guard(self) -> dict[str, str]:
        return {"scientific_content": _derive_scientific_fingerprint(self.coordinator)}

    @staticmethod
    def _capture_rng() -> dict[str, Any]:
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": numpy_state[0],
                "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": (
                tuple(
                    value.detach().cpu().clone()
                    for value in torch.cuda.get_rng_state_all()
                )
                if torch.cuda.is_available()
                else ()
            ),
        }

    @staticmethod
    def _restore_rng(state: Mapping[str, Any]) -> None:
        if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
            raise CheckpointIntegrityError("Checkpoint RNG fields do not match.")
        random.setstate(tuple(state["python"]))
        numpy_state = state["numpy"]
        if set(numpy_state) != {
            "bit_generator",
            "keys",
            "position",
            "has_gauss",
            "cached_gaussian",
        }:
            raise CheckpointIntegrityError("Malformed NumPy RNG state.")
        keys = numpy_state["keys"]
        if not torch.is_tensor(keys):
            raise CheckpointIntegrityError("Malformed NumPy RNG key tensor.")
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                keys.detach().cpu().numpy().astype(np.uint32, copy=True),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(state["torch_cpu"].detach().cpu())
        cuda_states = tuple(state["torch_cuda"])
        if cuda_states:
            if (
                not torch.cuda.is_available()
                or len(cuda_states) != torch.cuda.device_count()
            ):
                raise CheckpointIntegrityError(
                    "Checkpoint CUDA RNG state does not match the current runtime."
                )
            torch.cuda.set_rng_state_all(
                [value.detach().cpu() for value in cuda_states]
            )

    def _capture_clients(self) -> dict[int, Any]:
        clients: dict[int, Any] = {}
        for client_id, client in self.coordinator.clients.items():
            method_state = client.algorithm.save_method_state()
            if (
                client.state.continual_state is not client.algorithm.state
                and not _safe_equal(client.state.continual_state, method_state)
            ):
                raise CheckpointIntegrityError(
                    f"Client {client_id} continual/method state identity diverged."
                )
            overlays = {
                str(key): _overlay_state(
                    value, path=f"client.{client_id}.topology_overlays.{key}"
                )
                for key, value in client.state.topology_overlays.items()
            }
            clients[int(client_id)] = {
                "full_model_state": _model_state(client.model),
                "dynamic_model_state": _dynamic_model_state(
                    client.model,
                    build_parameter_manifest(
                        client.model,
                        self.coordinator.parameter_policy,
                    ),
                ),
                "seen_tasks": tuple(
                    sorted(int(value) for value in client.state.seen_tasks)
                ),
                "seen_class_mask": (
                    None
                    if client.state.seen_class_mask is None
                    else client.state.seen_class_mask.detach().cpu().clone()
                ),
                "continual_state": method_state,
                "method_state": method_state,
                "local_model_state": _clone_safe(
                    client.state.local_model_state,
                    path=f"client.{client_id}.local_model_state",
                ),
                "strategy_state": _clone_safe(
                    client.state.strategy_state,
                    path=f"client.{client_id}.strategy_state",
                ),
                "personalized_model_state": _clone_safe(
                    client.state.personalized_model_state,
                    path=f"client.{client_id}.personalized_model_state",
                ),
                "topology_overlays": overlays,
            }
        return clients

    def _capture_state(self) -> dict[str, Any]:
        elapsed = self._runtime_accumulated_seconds
        if self._session_start is not None:
            elapsed += time.perf_counter() - self._session_start
        current_peak = (
            int(torch.cuda.max_memory_allocated(self.coordinator.device))
            if self.coordinator.device.type == "cuda"
            else 0
        )
        return {
            "runtime_version": RUNTIME_VERSION,
            "global_model_state": _model_state(self.coordinator.global_model),
            "global_shared_state": {
                key: value.detach().cpu().clone()
                for key, value in self.coordinator.global_state.items()
            },
            "global_dynamic_model_state": _dynamic_model_state(
                self.coordinator.global_model, self.parameter_manifest
            ),
            "clients": self._capture_clients(),
            "strategy_state": _clone_safe(
                dict(self.strategy.state_dict()), path="strategy_state"
            ),
            "personalized_states": {
                client_id: {
                    key: value.detach().cpu().clone() for key, value in state.items()
                }
                for client_id, state in self.personalized_states.items()
            },
            "begun_tasks": tuple(sorted(self.begun_tasks)),
            "matrix": self.matrix.detach().cpu().clone(),
            "query_counts": self.query_counts.detach().cpu().clone(),
            "round_records": _clone_safe(self.round_records, path="round_records"),
            "stage_records": _clone_safe(self.stage_records, path="stage_records"),
            "resource_ledger": self.resource_ledger.to_dict(),
            "checkpoint_records": _clone_safe(
                self.checkpoint_records, path="checkpoint_records"
            ),
            "rng": self._capture_rng(),
            "stream_content_guard": dict(self._stream_guard_start),
            "runtime_elapsed_seconds": float(elapsed),
            "prior_peak_memory_bytes": max(self._prior_peak_memory_bytes, current_peak),
            "optimizer_state_saved": False,
        }

    def _validate_cursor(self, cursor: Mapping[str, Any]) -> dict[str, Any]:
        expected = {
            "boundary",
            "next_stage",
            "next_round",
            "stage_participants",
            "global_step",
            "stage_elapsed_seconds",
        }
        if not isinstance(cursor, Mapping) or set(cursor) != expected:
            raise CheckpointIntegrityError("Checkpoint cursor fields do not match.")
        num_tasks = self.coordinator.stream.scenario.num_tasks
        rounds = self.coordinator.stream.config.training.rounds_per_stage
        stage = int(cursor["next_stage"])
        round_index = int(cursor["next_round"])
        global_step = int(cursor["global_step"])
        participants = [int(value) for value in cursor["stage_participants"]]
        elapsed = float(cursor["stage_elapsed_seconds"])
        boundary = cursor["boundary"]
        if (
            boundary not in {"initial", "round", "stage"}
            or not (0 <= stage <= num_tasks)
            or not (0 <= round_index <= rounds)
            or global_step < 0
            or elapsed < 0
            or not math.isfinite(elapsed)
            or len(set(participants)) != len(participants)
            or not set(participants).issubset(self.coordinator.clients)
        ):
            raise CheckpointIntegrityError("Checkpoint cursor values are invalid.")
        if boundary == "initial":
            consistent = (
                stage == 0
                and round_index == 0
                and global_step == 0
                and not participants
                and elapsed == 0.0
            )
        elif boundary == "round":
            consistent = (
                stage < num_tasks
                and 1 <= round_index <= rounds
                and global_step == stage * rounds + round_index
            )
        else:
            consistent = (
                1 <= stage <= num_tasks
                and round_index == 0
                and global_step == stage * rounds
                and not participants
                and elapsed == 0.0
            )
        if not consistent:
            raise CheckpointIntegrityError(
                "Checkpoint cursor boundary is internally inconsistent."
            )
        return {
            "boundary": str(cursor["boundary"]),
            "next_stage": stage,
            "next_round": round_index,
            "stage_participants": sorted(participants),
            "global_step": global_step,
            "stage_elapsed_seconds": elapsed,
        }

    def _validate_restored_progress(self) -> None:
        """Cross-check resumable progress against the immutable stream."""

        stream = self.coordinator.stream
        num_clients = stream.config.partition.num_clients
        num_tasks = stream.scenario.num_tasks
        rounds = stream.config.training.rounds_per_stage
        expected_matrix_shape = (num_clients, num_tasks, num_tasks)
        expected_query_shape = (num_clients, num_tasks)
        if (
            tuple(self.matrix.shape) != expected_matrix_shape
            or not torch.is_floating_point(self.matrix)
            or bool(torch.isinf(self.matrix).any())
        ):
            raise CheckpointIntegrityError(
                "Checkpoint continual matrix values are invalid."
            )
        if (
            tuple(self.query_counts.shape) != expected_query_shape
            or not torch.is_floating_point(self.query_counts)
            or not bool(torch.isfinite(self.query_counts).all())
            or bool((self.query_counts < 0).any())
            or not torch.equal(self.query_counts, self.query_counts.round())
        ):
            raise CheckpointIntegrityError("Checkpoint query-count tensor is invalid.")
        if not isinstance(self.round_records, list) or not isinstance(
            self.stage_records, list
        ):
            raise CheckpointIntegrityError("Checkpoint progress records must be lists.")
        if len(self.round_records) != self.cursor["global_step"]:
            raise CheckpointIntegrityError("Checkpoint round-record cursor mismatch.")
        if len(self.stage_records) != self.cursor["next_stage"]:
            raise CheckpointIntegrityError("Checkpoint stage-record cursor mismatch.")

        expected_begun: set[tuple[int, int]] = set()
        for index, record in enumerate(self.round_records):
            if not isinstance(record, Mapping):
                raise CheckpointIntegrityError(
                    "Checkpoint round record must be a mapping."
                )
            expected_stage, expected_round = divmod(index, rounds)
            expected_participants = [
                int(value)
                for value in stream.participation.trace[expected_stage][expected_round]
            ]
            actual_participants = [
                int(value) for value in record.get("participant_ids", ())
            ]
            expected_tasks = {
                client_id: int(stream.orders.global_task(client_id, expected_stage))
                for client_id in expected_participants
            }
            raw_tasks = record.get("participant_global_task_ids", {})
            if not isinstance(raw_tasks, Mapping):
                raise CheckpointIntegrityError(
                    "Checkpoint participant-task map is invalid."
                )
            actual_tasks = {
                int(client_id): int(task_id) for client_id, task_id in raw_tasks.items()
            }
            if (
                record.get("stage") != expected_stage
                or record.get("round") != expected_round
                or actual_participants != expected_participants
                or actual_tasks != expected_tasks
            ):
                raise CheckpointIntegrityError(
                    "Checkpoint round record disagrees with the fixed trace."
                )
            expected_begun.update(expected_tasks.items())

        if self.cursor["boundary"] == "round":
            stage_start = self.cursor["next_stage"] * rounds
            current_records = self.round_records[stage_start:]
            participant_union = sorted(
                {
                    int(client_id)
                    for record in current_records
                    for client_id in record["participant_ids"]
                }
            )
        else:
            participant_union = []
        if participant_union != self.cursor["stage_participants"]:
            raise CheckpointIntegrityError(
                "Checkpoint stage-participant union is inconsistent."
            )
        if self.begun_tasks != expected_begun:
            raise CheckpointIntegrityError(
                "Checkpoint begun-task set disagrees with round participation."
            )
        undefined_cells = {
            (
                int(cell["client"]),
                int(cell["stage"]),
                int(cell["task"]),
            )
            for record in self.stage_records
            for cell in record.get("undefined_primary_metric_cells", ())
        }

        completed_stages = self.cursor["next_stage"]
        strategy_diagnostics = self.strategy.diagnostics()
        if "completed_stages" in strategy_diagnostics and (
            type(strategy_diagnostics["completed_stages"]) is not int
            or strategy_diagnostics["completed_stages"] != completed_stages
        ):
            raise CheckpointIntegrityError(
                "Checkpoint strategy stage state disagrees with the runtime cursor."
            )
        for client_id, client in self.coordinator.clients.items():
            order = stream.orders.client_orders[client_id]
            expected_seen = set(order[:completed_stages])
            if client.state.seen_tasks != expected_seen:
                raise CheckpointIntegrityError(
                    f"Checkpoint client {client_id} seen-task set is inconsistent."
                )
            expected_mask = None
            for task_id in expected_seen:
                task_mask = stream.shards[client_id][task_id].task_class_mask
                if task_mask is not None:
                    expected_mask = (
                        task_mask.detach().cpu().clone()
                        if expected_mask is None
                        else expected_mask | task_mask.detach().cpu()
                    )
            actual_mask = client.state.seen_class_mask
            if expected_mask is None:
                mask_matches = actual_mask is None
            else:
                mask_matches = (
                    torch.is_tensor(actual_mask)
                    and actual_mask.dtype == torch.bool
                    and torch.equal(actual_mask.detach().cpu(), expected_mask)
                )
            if not mask_matches:
                raise CheckpointIntegrityError(
                    f"Checkpoint client {client_id} seen-class mask is inconsistent."
                )
            for task_id in range(num_tasks):
                expected_count = (
                    stream.evaluation_shards[client_id][task_id].test_query_ids.shape[0]
                    if task_id in expected_seen
                    else 0
                )
                if int(self.query_counts[client_id, task_id]) != expected_count:
                    raise CheckpointIntegrityError(
                        "Checkpoint query counts disagree with completed evaluation."
                    )
            for stage_index in range(num_tasks):
                populated_tasks = (
                    set(order[: stage_index + 1])
                    if stage_index < completed_stages
                    else set()
                )
                for task_id in range(num_tasks):
                    populated = bool(
                        torch.isfinite(self.matrix[client_id, stage_index, task_id])
                        or (client_id, stage_index, task_id) in undefined_cells
                    )
                    if populated != (task_id in populated_tasks):
                        raise CheckpointIntegrityError(
                            "Checkpoint continual-matrix population is inconsistent."
                        )

    def _restore_checkpoint(self, state: Any, cursor: Any) -> None:
        if not isinstance(state, Mapping):
            raise CheckpointIntegrityError(
                "Checkpoint runtime state must be a mapping."
            )
        expected = {
            "runtime_version",
            "global_model_state",
            "global_shared_state",
            "global_dynamic_model_state",
            "clients",
            "strategy_state",
            "personalized_states",
            "begun_tasks",
            "matrix",
            "query_counts",
            "round_records",
            "stage_records",
            "resource_ledger",
            "checkpoint_records",
            "rng",
            "stream_content_guard",
            "runtime_elapsed_seconds",
            "prior_peak_memory_bytes",
            "optimizer_state_saved",
        }
        if set(state) != expected or state["runtime_version"] != RUNTIME_VERSION:
            raise CheckpointIntegrityError("Checkpoint runtime-state schema mismatch.")
        if state["optimizer_state_saved"] is not False:
            raise CheckpointIntegrityError(
                "Round checkpoints must not contain optimizer state."
            )
        self.cursor = self._validate_cursor(cursor)
        stored_guard = dict(state["stream_content_guard"])
        if (
            stored_guard != self._stream_guard_start
            and not self.resume_validation.diagnostic_override_used
        ):
            raise CheckpointIntegrityError(
                "Checkpoint fixed-stream content guard mismatch."
            )

        global_model_state = state["global_model_state"]
        self.coordinator.global_model.load_state_dict(global_model_state, strict=True)
        _restore_dynamic_model_state(
            self.coordinator.global_model,
            self.parameter_manifest,
            state["global_dynamic_model_state"],
        )
        global_shared = state["global_shared_state"]
        if set(global_shared) != set(self.parameter_manifest.shared_trainable):
            raise CheckpointIntegrityError("Checkpoint global shared keys mismatch.")
        self.coordinator.global_state = {
            key: value.detach().cpu().clone() for key, value in global_shared.items()
        }

        client_states = state["clients"]
        if set(client_states) != set(self.coordinator.clients):
            raise CheckpointIntegrityError("Checkpoint client IDs mismatch.")
        for client_id, client in self.coordinator.clients.items():
            client_state = client_states[client_id]
            expected_client = {
                "full_model_state",
                "dynamic_model_state",
                "seen_tasks",
                "seen_class_mask",
                "continual_state",
                "method_state",
                "local_model_state",
                "strategy_state",
                "personalized_model_state",
                "topology_overlays",
            }
            if (
                not isinstance(client_state, Mapping)
                or set(client_state) != expected_client
            ):
                raise CheckpointIntegrityError(
                    f"Checkpoint client {client_id} state schema mismatch."
                )
            if not _safe_equal(
                client_state["continual_state"], client_state["method_state"]
            ):
                raise CheckpointIntegrityError(
                    f"Checkpoint client {client_id} continual/method state mismatch."
                )
            client.model.load_state_dict(client_state["full_model_state"], strict=True)
            _restore_dynamic_model_state(
                client.model,
                build_parameter_manifest(
                    client.model,
                    self.coordinator.parameter_policy,
                ),
                client_state["dynamic_model_state"],
            )
            client.algorithm.load_method_state(client_state["method_state"])
            client.state.continual_state = client.algorithm.state
            client.state.seen_tasks = {
                int(value) for value in client_state["seen_tasks"]
            }
            seen_mask = client_state["seen_class_mask"]
            client.state.seen_class_mask = (
                None if seen_mask is None else seen_mask.detach().cpu().clone()
            )
            client.state.local_model_state = _clone_safe(
                client_state["local_model_state"]
            )
            client.state.strategy_state = _clone_safe(client_state["strategy_state"])
            client.state.personalized_model_state = _clone_safe(
                client_state["personalized_model_state"]
            )
            client.state.topology_overlays = {
                key: _restore_overlay(value, client)
                for key, value in client_state["topology_overlays"].items()
            }

        self.strategy.load_state_dict(state["strategy_state"])
        validate_restored_clients = getattr(
            self.strategy, "validate_restored_clients", None
        )
        if callable(validate_restored_clients):
            validate_restored_clients(
                self.coordinator.clients, int(self.cursor["next_stage"])
            )
        self.personalized_states = {
            int(client_id): {
                key: value.detach().cpu().clone() for key, value in values.items()
            }
            for client_id, values in state["personalized_states"].items()
        }
        self.begun_tasks = {
            (int(pair[0]), int(pair[1])) for pair in state["begun_tasks"]
        }
        if not torch.is_tensor(state["matrix"]) or not torch.is_tensor(
            state["query_counts"]
        ):
            raise CheckpointIntegrityError(
                "Checkpoint matrix and query counts must be tensors."
            )
        expected_matrix = (
            self.coordinator.stream.config.partition.num_clients,
            self.coordinator.stream.scenario.num_tasks,
            self.coordinator.stream.scenario.num_tasks,
        )
        if tuple(state["matrix"].shape) != expected_matrix:
            raise CheckpointIntegrityError(
                "Checkpoint continual matrix shape mismatch."
            )
        self.matrix = state["matrix"].detach().cpu().clone()
        self.query_counts = state["query_counts"].detach().cpu().clone()
        self.round_records = _clone_safe(state["round_records"])
        self.stage_records = _clone_safe(state["stage_records"])
        if len(self.round_records) != self.cursor["global_step"]:
            raise CheckpointIntegrityError("Checkpoint round-record cursor mismatch.")
        if len(self.stage_records) != self.cursor["next_stage"]:
            raise CheckpointIntegrityError("Checkpoint stage-record cursor mismatch.")
        self.resource_ledger = ResourceLedger.from_dict(state["resource_ledger"])
        self.checkpoint_records = _clone_safe(state["checkpoint_records"])
        self._validate_restored_progress()
        self._runtime_accumulated_seconds = float(state["runtime_elapsed_seconds"])
        self._prior_peak_memory_bytes = int(state["prior_peak_memory_bytes"])
        self._restore_rng(state["rng"])
        self._fresh_initialized = True

    def _write_checkpoint(self, checkpoint_id: str) -> None:
        if self.checkpoint_dir is None:
            return
        written = save_checkpoint(
            self.checkpoint_dir,
            checkpoint_id,
            identity=self.identity,
            cursor=dict(self.cursor),
            state=self._capture_state(),
        )
        self.resource_ledger.add(server_checkpoint_bytes=written.total_bytes)
        self.checkpoint_records.append(
            {
                "event": "save",
                "checkpoint_id": checkpoint_id,
                "manifest_file": written.manifest_path.name,
                "weights_file": written.weights_path.name,
                "manifest_bytes": written.manifest_bytes,
                "weights_bytes": written.weights_bytes,
                "total_bytes": written.total_bytes,
                "manifest_sha256": written.manifest_sha256,
                "weights_sha256": written.weights_sha256,
            }
        )

    def _initial_broadcast(self) -> None:
        if self._fresh_initialized:
            return
        clients = tuple(sorted(self.coordinator.clients))
        tasks = {
            client_id: int(self.coordinator.stream.orders.global_task(client_id, 0))
            for client_id in clients
        }
        context = RoundContext(0, 0, clients, tasks)
        for client_id in clients:
            client = self.coordinator.clients[client_id]
            task_id = tasks[client_id]
            shard = self.coordinator.stream.shards[client_id][task_id]
            method_context = client.build_method_context(
                shard,
                global_task_id=task_id,
                stage_index=0,
                round_index=0,
            )
            payload = self.strategy.prepare_payload(
                context, client_id, "initialization"
            )
            if payload is not None:
                self.strategy.client_receive(client, payload, method_context)
                self.resource_ledger.merge(payload.resources)
        self._fresh_initialized = True

    def _capture_evaluation_client_state(self, client: Any) -> dict[str, Any]:
        """Snapshot every client-owned value that evaluation may touch."""

        manifest = build_parameter_manifest(
            client.model, self.coordinator.parameter_policy
        )
        return {
            "model_state": _model_state(client.model),
            "dynamic_model_state": _dynamic_model_state(client.model, manifest),
            "training_modes": {
                name: module.training for name, module in client.model.named_modules()
            },
            "method_state": client.algorithm.save_method_state(),
            "seen_tasks": set(client.state.seen_tasks),
            "seen_class_mask": (
                None
                if client.state.seen_class_mask is None
                else client.state.seen_class_mask.detach().cpu().clone()
            ),
            "local_model_state": _clone_safe(client.state.local_model_state),
            "strategy_state": _clone_safe(client.state.strategy_state),
            "personalized_model_state": _clone_safe(
                client.state.personalized_model_state
            ),
            "topology_overlays": {
                key: _overlay_state(value, path=f"evaluation.topology_overlays.{key}")
                for key, value in client.state.topology_overlays.items()
            },
        }

    def _restore_evaluation_client_state(
        self, client: Any, snapshot: Mapping[str, Any]
    ) -> None:
        """Make evaluation observational with respect to future training."""

        manifest = build_parameter_manifest(
            client.model, self.coordinator.parameter_policy
        )
        client.model.load_state_dict(snapshot["model_state"], strict=True)
        _restore_dynamic_model_state(
            client.model, manifest, snapshot["dynamic_model_state"]
        )
        client.algorithm.load_method_state(snapshot["method_state"])
        client.state.continual_state = client.algorithm.state
        client.state.seen_tasks = set(snapshot["seen_tasks"])
        seen_mask = snapshot["seen_class_mask"]
        client.state.seen_class_mask = (
            None if seen_mask is None else seen_mask.detach().cpu().clone()
        )
        client.state.local_model_state = _clone_safe(snapshot["local_model_state"])
        client.state.strategy_state = _clone_safe(snapshot["strategy_state"])
        client.state.personalized_model_state = _clone_safe(
            snapshot["personalized_model_state"]
        )
        client.state.topology_overlays = {
            key: _restore_overlay(value, client)
            for key, value in snapshot["topology_overlays"].items()
        }
        modes = snapshot["training_modes"]
        if set(modes) != {name for name, _ in client.model.named_modules()}:
            raise RuntimeError("Evaluation changed the model module structure.")
        for name, module in client.model.named_modules():
            module.training = bool(modes[name])

    def _selection_for(self, client_id: int, stage: int) -> EvaluationSelection:
        if self.evaluation_model == "strategy":
            return self.strategy.select_evaluation(client_id, stage)
        if self.evaluation_model == "post_local":
            return EvaluationSelection(client_id=client_id, source="post_local")
        if self.evaluation_model == "personalized":
            state = self.personalized_states.get(client_id)
            if state is None:
                state = self.coordinator.clients[
                    client_id
                ].state.personalized_model_state
            if not state:
                raise RuntimeError(
                    f"Client {client_id} has no personalized evaluation state."
                )
            return EvaluationSelection(
                client_id=client_id,
                source="personalized",
                model_state=state,
                count_evaluation_sync=False,
            )
        return EvaluationSelection(
            client_id=client_id,
            source=self.evaluation_model,
            model_state=self.coordinator.global_state,
            count_evaluation_sync=True,
        )

    @staticmethod
    def _overlay_for(
        client: Any, context: Any, selection: EvaluationSelection
    ) -> TopologyOverlay | None:
        selected = None
        if selection.topology_overlay_id is not None:
            if selection.topology_overlay_id not in client.state.topology_overlays:
                raise RuntimeError(
                    f"Unknown topology overlay {selection.topology_overlay_id!r}."
                )
            selected = client.state.topology_overlays[selection.topology_overlay_id]
        method_overlay = client.algorithm.evaluation_topology(context)
        if method_overlay is not None:
            if (
                selected is not None
                and method_overlay.overlay_id != selected.overlay_id
            ):
                raise RuntimeError(
                    "Strategy and method selected different topology overlays."
                )
            selected = method_overlay
        if selected is not None and not isinstance(selected, TopologyOverlay):
            raise TypeError("Evaluation topology must be a TopologyOverlay.")
        if selected is not None and selected.client_id != client.client_id:
            raise RuntimeError("Evaluation topology belongs to another client.")
        return selected

    @staticmethod
    def _evaluate_call(
        evaluator: FederatedEvaluator,
        model: torch.nn.Module,
        client_id: int,
        task_id: int,
        seen_tasks: set[int],
        *,
        split: str,
        final_stage: bool,
        edge_index_override: torch.Tensor | None,
    ) -> dict[str, float]:
        kwargs: dict[str, Any] = {"split": split, "final_stage": final_stage}
        if edge_index_override is not None:
            if (
                "edge_index_override"
                not in inspect.signature(evaluator.evaluate).parameters
            ):
                raise RuntimeError(
                    "The evaluator does not support method-owned topology overlays."
                )
            kwargs["edge_index_override"] = edge_index_override
        return evaluator.evaluate(
            model,
            client_id,
            task_id,
            seen_tasks,
            **kwargs,
        )

    def _primary_evaluation_support(self, query_ids: torch.Tensor) -> int:
        """Return positive-query support for Hits and row support otherwise."""

        primary = self.coordinator.stream.scenario.metrics[0].lower()
        if primary.startswith("hits@"):
            return int(
                (self.coordinator.stream.scenario.labels[query_ids] == 1).sum()
            )
        return int(query_ids.numel())

    def _evaluate_validation_checkpoint(
        self,
        evaluator: FederatedEvaluator,
        stage: int,
        shared_state: Mapping[str, torch.Tensor],
    ) -> dict[str, float | int]:
        """Score a candidate centrally; return aggregates, never held-out data."""

        evaluator.clear_reference_cache()
        primary = self.coordinator.stream.scenario.metrics[0]
        final_stage = stage == self.coordinator.stream.scenario.num_tasks - 1
        all_values: list[float] = []
        current_values: list[float] = []
        previous_values: list[float] = []
        expected_cells = 0
        for client_id, client in self.coordinator.clients.items():
            snapshot = self._capture_evaluation_client_state(client)
            try:
                current_task = int(
                    self.coordinator.stream.orders.global_task(client_id, stage)
                )
                seen_tasks = set(client.state.seen_tasks) | {current_task}
                client.load_shared_state(dict(shared_state))
                for task_id in sorted(seen_tasks):
                    expected_cells += 1
                    metrics = self._evaluate_call(
                        evaluator,
                        client.model,
                        client_id,
                        task_id,
                        seen_tasks,
                        split="validation",
                        final_stage=final_stage,
                        edge_index_override=None,
                    )
                    value = float(metrics[primary])
                    if not math.isfinite(value):
                        continue
                    all_values.append(value)
                    if task_id == current_task:
                        current_values.append(value)
                    else:
                        previous_values.append(value)
            finally:
                self._restore_evaluation_client_state(client, snapshot)
        if not all_values or not current_values:
            raise RuntimeError("FedFST validation selector has no finite cells.")
        mean_previous = (
            float(sum(previous_values) / len(previous_values))
            if previous_values
            else float(sum(current_values) / len(current_values))
        )
        return {
            "mean_seen_validation": float(sum(all_values) / len(all_values)),
            "mean_current_validation": float(
                sum(current_values) / len(current_values)
            ),
            "mean_previous_validation": mean_previous,
            "finite_cells": len(all_values),
            "expected_cells": expected_cells,
        }

    def _evaluate_stage(
        self, evaluator: FederatedEvaluator, stage: int
    ) -> dict[str, Any]:
        evaluation_start = time.perf_counter()
        evaluator.clear_reference_cache()
        primary = self.coordinator.stream.scenario.metrics[0]
        stage_values: list[float] = []
        stage_weights: list[int] = []
        validation_values: list[float] = []
        validation_weights: list[int] = []
        graph_values: dict[str, list[float]] = {}
        primary_by_client: dict[int, list[float]] = {}
        undefined_cells: list[dict[str, int | str]] = []
        undefined_validation_cells: list[dict[str, int | str]] = []
        expected_primary_cells = 0
        sources: dict[int, str] = {}
        final_stage = stage == self.coordinator.stream.scenario.num_tasks - 1
        all_clients = tuple(sorted(self.coordinator.clients))
        task_map = {
            client_id: int(self.coordinator.stream.orders.global_task(client_id, stage))
            for client_id in all_clients
        }
        round_context = RoundContext(
            stage,
            self.coordinator.stream.config.training.rounds_per_stage,
            all_clients,
            task_map,
        )
        preserve_legacy_broadcast = self.evaluation_model == "strategy" and isinstance(
            self.strategy, LegacyStrategyAdapter
        )

        def record_validation(
            value: float, *, client_id: int, task_id: int
        ) -> None:
            if math.isfinite(value):
                validation_values.append(value)
                validation_weights.append(
                    self._primary_evaluation_support(
                        self.coordinator.stream.evaluation_shards[client_id][
                            task_id
                        ].validation_query_ids
                    )
                )
                return
            undefined_validation_cells.append(
                {
                    "client": int(client_id),
                    "stage": int(stage),
                    "task": int(task_id),
                    "metric": str(primary),
                    "reason": (
                        "metric_undefined_for_local_eval_shard"
                        if math.isnan(value)
                        else "non_finite_metric_for_local_eval_shard"
                    ),
                }
            )

        for client_id, client in self.coordinator.clients.items():
            evaluation_snapshot = (
                None
                if preserve_legacy_broadcast
                else self._capture_evaluation_client_state(client)
            )
            try:
                current_task = task_map[client_id]
                current_shard = self.coordinator.stream.shards[client_id][current_task]
                current_context = client.build_method_context(
                    current_shard,
                    global_task_id=current_task,
                    stage_index=stage,
                    round_index=self.coordinator.stream.config.training.rounds_per_stage,
                )
                selection = self._selection_for(client_id, stage)
                sources[int(client_id)] = selection.source
                delivered = False
                if (
                    selection.count_evaluation_sync
                    and self.evaluation_model == "strategy"
                ):
                    payload = self.strategy.prepare_payload(
                        round_context, client_id, "evaluation"
                    )
                    if payload is not None:
                        if (
                            payload.model_state.payload_bytes
                            != selection.model_state.payload_bytes
                        ):
                            raise RuntimeError(
                                "Evaluation payload/selection byte mismatch."
                            )
                        self.strategy.client_receive(client, payload, current_context)
                        self.resource_ledger.merge(payload.resources)
                        delivered = True
                if selection.model_state is not None and not delivered:
                    client.load_shared_state(selection.model_state.materialize())
                    if selection.count_evaluation_sync:
                        self.resource_ledger.add(
                            evaluation_sync_bytes=selection.model_state.payload_bytes
                        )
                model = client.model
                for task_id in sorted(client.state.seen_tasks):
                    expected_primary_cells += 1
                    task_shard = self.coordinator.stream.shards[client_id][task_id]
                    method_context = client.build_method_context(
                        task_shard,
                        global_task_id=task_id,
                        stage_index=stage,
                        round_index=self.coordinator.stream.config.training.rounds_per_stage,
                    )
                    overlay = self._overlay_for(client, method_context, selection)
                    edge_override = (
                        None
                        if overlay is None
                        else overlay.apply(method_context.effective_edge_index)
                    )
                    metrics = self._evaluate_call(
                        evaluator,
                        model,
                        client_id,
                        task_id,
                        client.state.seen_tasks,
                        split="test",
                        final_stage=final_stage,
                        edge_index_override=edge_override,
                    )
                    value = metrics[primary]
                    central = self.coordinator.stream.evaluation_shards[client_id][
                        task_id
                    ]
                    test_support = self._primary_evaluation_support(
                        central.test_query_ids
                    )
                    self.query_counts[client_id, task_id] = test_support
                    if not math.isfinite(value):
                        if math.isnan(value):
                            undefined_cells.append(
                                {
                                    "client": int(client_id),
                                    "stage": int(stage),
                                    "task": int(task_id),
                                    "metric": str(primary),
                                    "reason": (
                                        "metric_undefined_for_local_eval_shard"
                                    ),
                                }
                            )
                            validation_metrics = self._evaluate_call(
                                evaluator,
                                model,
                                client_id,
                                task_id,
                                client.state.seen_tasks,
                                split="validation",
                                final_stage=final_stage,
                                edge_index_override=edge_override,
                            )
                            validation_value = validation_metrics[primary]
                            record_validation(
                                validation_value,
                                client_id=client_id,
                                task_id=task_id,
                            )
                            for name, metric_value in metrics.items():
                                if name != primary and metric_value == metric_value:
                                    graph_values.setdefault(name, []).append(metric_value)
                            continue
                        raise RuntimeError(
                            f"Non-finite {primary} for client={client_id}, "
                            f"stage={stage}, task={task_id}."
                        )
                    self.matrix[client_id, stage, task_id] = value
                    stage_values.append(value)
                    stage_weights.append(test_support)
                    primary_by_client.setdefault(int(client_id), []).append(value)
                    validation_metrics = self._evaluate_call(
                        evaluator,
                        model,
                        client_id,
                        task_id,
                        client.state.seen_tasks,
                        split="validation",
                        final_stage=final_stage,
                        edge_index_override=edge_override,
                    )
                    validation_value = validation_metrics[primary]
                    record_validation(
                        validation_value,
                        client_id=client_id,
                        task_id=task_id,
                    )
                    for name, metric_value in metrics.items():
                        if name != primary and metric_value == metric_value:
                            graph_values.setdefault(name, []).append(metric_value)
            finally:
                if evaluation_snapshot is not None:
                    self._restore_evaluation_client_state(client, evaluation_snapshot)
        positive_query_micro = primary.lower().startswith("hits@")

        def aggregate(values: list[float], weights: list[int]) -> float | None:
            if not values:
                return None
            if not positive_query_micro:
                return sum(values) / len(values)
            denominator = sum(weights)
            if denominator <= 0:
                return None
            return sum(
                value * weight for value, weight in zip(values, weights)
            ) / denominator

        output: dict[str, Any] = {
            "validation_metric": (
                aggregate(validation_values, validation_weights)
            ),
            "stage_test_metric": (
                aggregate(stage_values, stage_weights)
            ),
            "diagnostic_macro_cell_stage_test_metric": (
                sum(stage_values) / len(stage_values) if stage_values else None
            ),
            "stage": float(stage),
            "evaluation_model_sources": sources,
            "undefined_primary_metric_count": len(undefined_cells),
            "undefined_primary_metric_cells": undefined_cells,
            "primary_metric_expected_cell_count": expected_primary_cells,
            "primary_metric_finite_cell_count": len(stage_values),
            "primary_metric_support_coverage": (
                len(stage_values) / expected_primary_cells
                if expected_primary_cells
                else 0.0
            ),
            "undefined_validation_primary_metric_count": len(
                undefined_validation_cells
            ),
            "undefined_validation_primary_metric_cells": (
                undefined_validation_cells
            ),
            "validation_primary_metric_expected_cell_count": (
                expected_primary_cells
            ),
            "validation_primary_metric_finite_cell_count": len(
                validation_values
            ),
            "validation_primary_metric_support_coverage": (
                len(validation_values) / expected_primary_cells
                if expected_primary_cells
                else 0.0
            ),
            "per_client_test_metric": {
                client_id: sum(values) / len(values)
                for client_id, values in primary_by_client.items()
                if values
            },
        }
        if self.personalized_states:
            shared_by_client: dict[int, list[float]] = {}
            for client_id, client in self.coordinator.clients.items():
                evaluation_snapshot = self._capture_evaluation_client_state(client)
                try:
                    client.load_shared_state(self.coordinator.global_state)
                    shared_selection = EvaluationSelection(
                        client_id=int(client_id),
                        source="shared",
                        model_state=self.coordinator.global_state,
                        count_evaluation_sync=False,
                    )
                    for task_id in sorted(client.state.seen_tasks):
                        task_shard = self.coordinator.stream.shards[client_id][
                            task_id
                        ]
                        method_context = client.build_method_context(
                            task_shard,
                            global_task_id=task_id,
                            stage_index=stage,
                            round_index=(
                                self.coordinator.stream.config.training.rounds_per_stage
                            ),
                        )
                        overlay = self._overlay_for(
                            client, method_context, shared_selection
                        )
                        edge_override = (
                            None
                            if overlay is None
                            else overlay.apply(method_context.effective_edge_index)
                        )
                        metrics = self._evaluate_call(
                            evaluator,
                            client.model,
                            client_id,
                            task_id,
                            client.state.seen_tasks,
                            split="test",
                            final_stage=final_stage,
                            edge_index_override=edge_override,
                        )
                        value = metrics[primary]
                        if not math.isfinite(value):
                            if math.isnan(value):
                                continue
                            raise RuntimeError(
                                "Non-finite shared diagnostic metric for "
                                f"client={client_id}, stage={stage}, task={task_id}."
                            )
                        shared_by_client.setdefault(int(client_id), []).append(value)
                    self.resource_ledger.add(
                        evaluation_sync_bytes=sum(
                            tensor.numel() * tensor.element_size()
                            for tensor in self.coordinator.global_state.values()
                        )
                    )
                finally:
                    self._restore_evaluation_client_state(client, evaluation_snapshot)
            output["diagnostic_shared_stage_test_metric"] = sum(
                value for values in shared_by_client.values() for value in values
            ) / max(1, sum(len(values) for values in shared_by_client.values()))
            output["diagnostic_shared_per_client_test_metric"] = {
                client_id: sum(values) / len(values)
                for client_id, values in shared_by_client.items()
                if values
            }
        output.update(
            {
                name: sum(values) / len(values)
                for name, values in graph_values.items()
                if values
            }
        )
        output["evaluation_runtime_seconds"] = time.perf_counter() - evaluation_start
        return output

    def _apply_aggregation_result(self, aggregation: AggregationResult) -> None:
        """Synchronize one typed strategy result with runtime/client models."""

        if not isinstance(aggregation, AggregationResult):
            raise TypeError("Strategy aggregation must return AggregationResult.")
        self.coordinator.global_state = aggregation.shared_state.materialize()
        if self.strategy.aggregates:
            self.coordinator.parameter_policy.load(
                self.coordinator.global_model, self.coordinator.global_state
            )
        for client_id, state in aggregation.personalized_states:
            materialized = state.materialize()
            self.personalized_states[client_id] = materialized
            self.coordinator.clients[client_id].state.personalized_model_state = {
                key: value.clone() for key, value in materialized.items()
            }

    def _run_round(self, stage: int, round_index: int) -> dict[str, Any]:
        stream = self.coordinator.stream
        participants = tuple(
            int(value) for value in stream.participation.trace[stage][round_index]
        )
        participant_tasks = {
            client_id: int(stream.orders.global_task(client_id, stage))
            for client_id in participants
        }
        if (
            stream.config.order.profile == "synchronized"
            and len(set(participant_tasks.values())) != 1
        ):
            raise RuntimeError(
                "Synchronized order assigned different global tasks in one round."
            )
        context = RoundContext(stage, round_index, participants, participant_tasks)
        prepare_round = getattr(self.strategy, "prepare_round", None)
        if callable(prepare_round):
            all_client_weights = {}
            for client_id in sorted(self.coordinator.clients):
                global_task_id = int(stream.orders.global_task(client_id, stage))
                shard = stream.shards[client_id][global_task_id]
                all_client_weights[client_id] = int(
                    self.coordinator._shard_weight(shard)
                )
            prepare_round(context, all_client_weights)
        uploads = []
        client_deltas: dict[int, float] = {}
        client_method_diagnostics: dict[int, Any] = {}
        round_resources = ResourceLedger()
        round_start = time.perf_counter()
        for client_id in participants:
            client = self.coordinator.clients[client_id]
            global_task_id = participant_tasks[client_id]
            shard = stream.shards[client_id][global_task_id]
            method_context = client.build_method_context(
                shard,
                global_task_id=global_task_id,
                stage_index=stage,
                round_index=round_index,
            )
            task_key = (client_id, global_task_id)
            if task_key not in self.begun_tasks:
                client.begin_task_stateful(method_context)
                self.begun_tasks.add(task_key)
            before_client_state = self.coordinator.parameter_policy.extract(
                client.model
            )
            payload = self.strategy.prepare_payload(context, client_id, "training")
            server_state = self.coordinator.global_state
            if payload is not None:
                self.strategy.client_receive(client, payload, method_context)
                round_resources.merge(payload.resources)
                if payload.model_state:
                    server_state = payload.model_state.materialize()
            upload = client.update_stateful(
                method_context,
                server_state=server_state,
                strategy=self.strategy,
            )
            if upload.global_task_id != global_task_id:
                raise RuntimeError("Client upload reported the wrong global task ID.")
            if not math.isfinite(upload.training_loss):
                raise RuntimeError(
                    f"Non-finite training loss for client={client_id}, "
                    f"stage={stage}, task={global_task_id}."
                )
            uploads.append(upload)
            client_method_diagnostics[client_id] = _json_safe(
                client.algorithm.diagnostics()
            )
            post_local_state = self.coordinator.parameter_policy.extract(client.model)
            client_deltas[client_id] = self.coordinator._state_delta_l2(
                before_client_state, post_local_state
            )
        empty_clients = [upload.client_id for upload in uploads if upload.weight <= 0]
        if empty_clients:
            raise RuntimeError(
                f"Active clients produced empty uploads: {empty_clients}."
            )
        raw_weights = {upload.client_id: int(upload.weight) for upload in uploads}
        total_weight = sum(raw_weights.values())
        normalized_weights = {
            client_id: weight / total_weight
            for client_id, weight in raw_weights.items()
        }
        normalized_weight_sum = sum(normalized_weights.values())
        if not math.isclose(normalized_weight_sum, 1.0, abs_tol=1e-12):
            raise RuntimeError("Normalized aggregation weights do not sum to one.")
        server_before = {
            key: value.clone() for key, value in self.coordinator.global_state.items()
        }
        aggregation = self.strategy.aggregate(context, tuple(uploads))
        aggregation = self.strategy.personalize(context, aggregation)
        round_resources.merge(aggregation.resources)
        self._apply_aggregation_result(aggregation)
        self.resource_ledger.merge(round_resources)
        return {
            "stage": stage,
            "round": round_index,
            "participants": len(participants),
            "participant_ids": list(participants),
            "participant_global_task_ids": participant_tasks,
            "raw_aggregation_weights": raw_weights,
            "normalized_aggregation_weights": normalized_weights,
            "normalized_aggregation_weight_sum": normalized_weight_sum,
            "empty_updates": 0,
            "mean_training_loss": sum(upload.training_loss for upload in uploads)
            / len(uploads),
            "round_runtime_seconds": time.perf_counter() - round_start,
            "communication_payload_bytes": round_resources.communication_payload_bytes,
            "resource_ledger": round_resources.to_dict(),
            "client_parameter_delta_l2": client_deltas,
            "server_parameter_delta_l2": (
                self.coordinator._state_delta_l2(
                    server_before, self.coordinator.global_state
                )
                if self.strategy.aggregates
                else None
            ),
            "strategy_diagnostics": _json_safe(dict(self.strategy.diagnostics())),
            "client_method_diagnostics": client_method_diagnostics,
        }

    def _measure_persistent_state(self) -> None:
        client_bytes = 0
        overlay_bytes = 0
        replay_bytes = 0
        for client in self.coordinator.clients.values():
            client_bytes += tensor_payload_bytes(client.model.state_dict())
            method_state_bytes = tensor_payload_bytes(client.algorithm.state)
            declared_replay_bytes = client.algorithm.replay_payload_bytes()
            declared_overlay_bytes = client.algorithm.topology_overlay_payload_bytes()
            if (
                isinstance(declared_replay_bytes, bool)
                or not isinstance(declared_replay_bytes, int)
                or declared_replay_bytes < 0
            ):
                raise RuntimeError(
                    "Client continual methods must declare non-negative replay bytes."
                )
            if (
                isinstance(declared_overlay_bytes, bool)
                or not isinstance(declared_overlay_bytes, int)
                or declared_overlay_bytes < 0
            ):
                raise RuntimeError(
                    "Client continual methods must declare non-negative topology bytes."
                )
            categorized_method_bytes = declared_replay_bytes + declared_overlay_bytes
            if categorized_method_bytes > method_state_bytes:
                raise RuntimeError(
                    "Declared replay/overlay bytes exceed method tensor state."
                )
            client_bytes += method_state_bytes - categorized_method_bytes
            replay_bytes += declared_replay_bytes
            overlay_bytes += declared_overlay_bytes
            strategy_state_bytes = tensor_payload_bytes(client.state.strategy_state)
            strategy_replay_bytes = 0
            replay_counter = getattr(self.strategy, "client_replay_payload_bytes", None)
            if callable(replay_counter):
                strategy_replay_bytes = replay_counter(client)
                if (
                    isinstance(strategy_replay_bytes, bool)
                    or not isinstance(strategy_replay_bytes, int)
                    or strategy_replay_bytes < 0
                    or strategy_replay_bytes > strategy_state_bytes
                ):
                    raise RuntimeError(
                        "Strategy replay bytes must be a valid subset of client state."
                    )
            client_bytes += strategy_state_bytes - strategy_replay_bytes
            replay_bytes += strategy_replay_bytes
            client_bytes += tensor_payload_bytes(client.state.personalized_model_state)
            for overlay in client.state.topology_overlays.values():
                if isinstance(overlay, TopologyOverlay):
                    overlay_bytes += overlay.payload_bytes
                else:
                    overlay_bytes += tensor_payload_bytes(overlay)
        server_bytes = tensor_payload_bytes(self.coordinator.global_model.state_dict())
        strategy_state = dict(self.strategy.state_dict())
        for key in ("shared_state", "global_state", "model_state"):
            strategy_state.pop(key, None)
        server_bytes += tensor_payload_bytes(strategy_state)
        server_bytes += tensor_payload_bytes(self.personalized_states)
        self.resource_ledger.client_persistent_bytes = max(
            self.resource_ledger.client_persistent_bytes, client_bytes
        )
        self.resource_ledger.replay_bytes = max(
            self.resource_ledger.replay_bytes, replay_bytes
        )
        self.resource_ledger.server_persistent_bytes = max(
            self.resource_ledger.server_persistent_bytes, server_bytes
        )
        self.resource_ledger.topology_overlay_bytes = max(
            self.resource_ledger.topology_overlay_bytes, overlay_bytes
        )

    def _validate_population(self) -> None:
        stream = self.coordinator.stream
        undefined_cells = {
            (
                int(cell["client"]),
                int(cell["stage"]),
                int(cell["task"]),
            )
            for record in self.stage_records
            for cell in record.get("undefined_primary_metric_cells", ())
        }
        for client_id in range(stream.config.partition.num_clients):
            order = stream.orders.client_orders[client_id]
            for stage in range(stream.scenario.num_tasks):
                expected = set(order[: stage + 1])
                for task_id in range(stream.scenario.num_tasks):
                    populated = bool(
                        torch.isfinite(self.matrix[client_id, stage, task_id])
                        or (client_id, stage, task_id) in undefined_cells
                    )
                    if populated != (task_id in expected):
                        raise RuntimeError(
                            "A[k,s,t] population mismatch for "
                            f"client={client_id}, stage={stage}, task={task_id}."
                        )

    def _trajectories(self) -> dict[int, dict[int, list[float | None]]]:
        output: dict[int, dict[int, list[float | None]]] = {}
        for client_id in range(self.matrix.shape[0]):
            output[client_id] = {}
            for task_id in range(self.matrix.shape[2]):
                output[client_id][task_id] = [
                    None if not torch.isfinite(value) else float(value)
                    for value in self.matrix[client_id, :, task_id]
                ]
        return output

    def _personalized_global_gap(self) -> dict[str, Any]:
        """Return final-stage personalized-minus-shared diagnostic metrics."""

        if not self.personalized_states:
            return {
                "applicable": False,
                "status": "not_applicable_shared_only_strategy",
                "macro_personalized_minus_global": None,
                "per_client": {},
            }
        if not self.stage_records:
            raise RuntimeError(
                "Personalized diagnostic evaluation has no stage records."
            )
        final_stage = self.stage_records[-1]
        personalized = final_stage.get("per_client_test_metric")
        shared = final_stage.get("diagnostic_shared_per_client_test_metric")
        if not isinstance(personalized, Mapping) or not isinstance(shared, Mapping):
            raise RuntimeError(
                "Personalized/shared diagnostic evaluation is missing."
            )
        if set(personalized) != set(shared) or not personalized:
            raise RuntimeError(
                "Personalized/shared diagnostic client identities differ."
            )
        gaps = {
            int(client_id): float(personalized[client_id]) - float(shared[client_id])
            for client_id in sorted(personalized)
        }
        if not all(math.isfinite(value) for value in gaps.values()):
            raise RuntimeError("Personalized/shared diagnostic gap is non-finite.")
        return {
            "applicable": True,
            "status": "diagnostic_shared_mean_evaluated",
            "macro_personalized_minus_global": sum(gaps.values()) / len(gaps),
            "per_client": gaps,
        }

    def run(self) -> dict[str, Any]:
        if self._has_run:
            raise RuntimeError("A StatefulFederatedRuntime instance can run only once.")
        self._has_run = True
        self._session_start = time.perf_counter()
        if self.coordinator.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.coordinator.device)
        self._initial_broadcast()
        evaluator = FederatedEvaluator(
            self.coordinator.stream,
            reference_view=self.coordinator.reference_view,
            class_mask_policy=self.coordinator.class_mask_policy,
        )
        num_tasks = self.coordinator.stream.scenario.num_tasks
        rounds_per_stage = self.coordinator.stream.config.training.rounds_per_stage

        start_stage = int(self.cursor["next_stage"])
        for stage in range(start_stage, num_tasks):
            stage_base_elapsed = (
                float(self.cursor["stage_elapsed_seconds"])
                if stage == start_stage
                else 0.0
            )
            stage_segment_start = time.perf_counter()
            stage_participants = (
                set(int(value) for value in self.cursor["stage_participants"])
                if stage == start_stage
                else set()
            )
            first_round = int(self.cursor["next_round"]) if stage == start_stage else 0
            for round_index in range(first_round, rounds_per_stage):
                record = self._run_round(stage, round_index)
                stage_participants.update(record["participant_ids"])
                self.round_records.append(record)
                self.cursor = {
                    "boundary": "round",
                    "next_stage": stage,
                    "next_round": round_index + 1,
                    "stage_participants": sorted(stage_participants),
                    "global_step": len(self.round_records),
                    "stage_elapsed_seconds": stage_base_elapsed
                    + (time.perf_counter() - stage_segment_start),
                }
                self.coordinator.run_logger.log(
                    record, step=self.cursor["global_step"] - 1
                )
                if (
                    self.checkpoint_dir is not None
                    and self.checkpoint_every is not None
                    and self.cursor["global_step"] % self.checkpoint_every == 0
                ):
                    self._write_checkpoint(f"round-{self.cursor['global_step']:08d}")

            method_contexts: dict[int, Any] = {}
            for client_id in sorted(stage_participants):
                client = self.coordinator.clients[client_id]
                task_id = int(
                    self.coordinator.stream.orders.global_task(client_id, stage)
                )
                shard = self.coordinator.stream.shards[client_id][task_id]
                method_contexts[client_id] = client.build_method_context(
                    shard,
                    global_task_id=task_id,
                    stage_index=stage,
                    round_index=rounds_per_stage,
                )
            stage_finalization: dict[str, Any] | None = None
            finalize_stage = getattr(self.strategy, "finalize_stage", None)
            if callable(finalize_stage):
                finalization_context = RoundContext(
                    stage,
                    rounds_per_stage,
                    tuple(sorted(stage_participants)),
                    {
                        client_id: int(
                            self.coordinator.stream.orders.global_task(client_id, stage)
                        )
                        for client_id in sorted(stage_participants)
                    },
                )
                server_before = {
                    key: value.clone()
                    for key, value in self.coordinator.global_state.items()
                }
                finalization_kwargs: dict[str, Any] = {}
                if (
                    "validation_evaluator"
                    in inspect.signature(finalize_stage).parameters
                ):
                    finalization_kwargs["validation_evaluator"] = (
                        lambda state, stage_index=stage: (
                            self._evaluate_validation_checkpoint(
                                evaluator, stage_index, state
                            )
                        )
                    )
                aggregation = finalize_stage(
                    finalization_context,
                    {
                        client_id: self.coordinator.clients[client_id]
                        for client_id in sorted(stage_participants)
                    },
                    method_contexts,
                    **finalization_kwargs,
                )
                self._apply_aggregation_result(aggregation)
                self.resource_ledger.merge(aggregation.resources)
                stage_finalization = {
                    "server_parameter_delta_l2": self.coordinator._state_delta_l2(
                        server_before, self.coordinator.global_state
                    ),
                    "resource_ledger": aggregation.resources.to_dict(),
                    "strategy_diagnostics": _json_safe(
                        _thaw_protocol_metadata(aggregation.diagnostics),
                        path="stage_finalization.strategy_diagnostics",
                    ),
                }
            for client_id in sorted(stage_participants):
                client = self.coordinator.clients[client_id]
                context = method_contexts[client_id]
                client.consolidate_task_stateful(context)
                consolidate_client = getattr(self.strategy, "consolidate_client", None)
                if callable(consolidate_client):
                    consolidate_client(client, context)
            for client_id, client in self.coordinator.clients.items():
                task_id = int(
                    self.coordinator.stream.orders.global_task(client_id, stage)
                )
                client.mark_seen(
                    task_id,
                    self.coordinator.stream.shards[client_id][task_id].task_class_mask,
                )
            stage_metrics = self._evaluate_stage(evaluator, stage)
            if stage_finalization is not None:
                stage_metrics["stage_finalization"] = stage_finalization
            stage_metrics["stage_runtime_seconds"] = stage_base_elapsed + (
                time.perf_counter() - stage_segment_start
            )
            self.stage_records.append(stage_metrics)
            self.coordinator.run_logger.log(stage_metrics, step=len(self.round_records))
            self.cursor = {
                "boundary": "stage",
                "next_stage": stage + 1,
                "next_round": 0,
                "stage_participants": [],
                "global_step": len(self.round_records),
                "stage_elapsed_seconds": 0.0,
            }
            self._measure_persistent_state()
            if self.checkpoint_dir is not None:
                self._write_checkpoint(f"stage-{stage + 1:04d}")

        self._validate_population()
        final_versions = self.coordinator._stream_tensor_versions()
        final_guard = self._stream_content_guard()
        if (
            self._stream_versions_start != final_versions
            or self._stream_guard_start != final_guard
        ):
            changed = sorted(
                set(self._stream_guard_start)
                | set(final_guard)
                | {
                    key
                    for key in self._stream_versions_start
                    if self._stream_versions_start[key] != final_versions.get(key)
                }
            )
            raise RuntimeError(f"In-memory stream tensors were mutated: {changed}.")
        self._measure_persistent_state()
        summary = summarize_continual_matrix(
            self.matrix,
            self.coordinator.stream.orders,
            self.query_counts,
            base_metric=self.coordinator.stream.scenario.metrics[0],
        )
        num_clients = self.coordinator.stream.config.partition.num_clients
        expected_primary_cell_count = (
            num_clients * num_tasks * (num_tasks + 1) // 2
        )
        finite_primary_cell_count = int(torch.isfinite(self.matrix).sum())
        undefined_primary_metric_count = sum(
            int(record.get("undefined_primary_metric_count", 0))
            for record in self.stage_records
        )
        final_expected_primary_cell_count = num_clients * num_tasks
        final_finite_primary_cell_count = int(
            torch.isfinite(self.matrix[:, -1, :]).sum()
        )
        primary_metric_complete = (
            finite_primary_cell_count == expected_primary_cell_count
            and undefined_primary_metric_count == 0
            and final_finite_primary_cell_count
            == final_expected_primary_cell_count
        )
        primary_metric_support = {
            "expected_cell_count": expected_primary_cell_count,
            "finite_cell_count": finite_primary_cell_count,
            "undefined_cell_count": undefined_primary_metric_count,
            "coverage": (
                finite_primary_cell_count / expected_primary_cell_count
                if expected_primary_cell_count
                else 0.0
            ),
            "final_expected_cell_count": final_expected_primary_cell_count,
            "final_finite_cell_count": final_finite_primary_cell_count,
            "final_coverage": (
                final_finite_primary_cell_count
                / final_expected_primary_cell_count
                if final_expected_primary_cell_count
                else 0.0
            ),
            "complete": primary_metric_complete,
        }
        validation_expected_primary_cell_count = sum(
            int(record.get("validation_primary_metric_expected_cell_count", 0))
            for record in self.stage_records
        )
        validation_finite_primary_cell_count = sum(
            int(record.get("validation_primary_metric_finite_cell_count", 0))
            for record in self.stage_records
        )
        validation_undefined_primary_metric_count = sum(
            int(record.get("undefined_validation_primary_metric_count", 0))
            for record in self.stage_records
        )
        validation_primary_metric_complete = (
            validation_finite_primary_cell_count
            == validation_expected_primary_cell_count
            and validation_undefined_primary_metric_count == 0
        )
        validation_primary_metric_support = {
            "expected_cell_count": validation_expected_primary_cell_count,
            "finite_cell_count": validation_finite_primary_cell_count,
            "undefined_cell_count": validation_undefined_primary_metric_count,
            "coverage": (
                validation_finite_primary_cell_count
                / validation_expected_primary_cell_count
                if validation_expected_primary_cell_count
                else 0.0
            ),
            "complete": validation_primary_metric_complete,
        }
        primary_metric_protocol_complete = (
            primary_metric_complete and validation_primary_metric_complete
        )
        current_peak = (
            int(torch.cuda.max_memory_allocated(self.coordinator.device))
            if self.coordinator.device.type == "cuda"
            else 0
        )
        peak_memory = max(self._prior_peak_memory_bytes, current_peak)
        runtime_seconds = self._runtime_accumulated_seconds + (
            time.perf_counter() - self._session_start
        )
        model_parameters = sum(
            parameter.numel()
            for parameter in self.coordinator.global_model.parameters()
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.coordinator.global_model.parameters()
            if parameter.requires_grad
        )
        personalized_parameters = sum(
            sum(tensor.numel() for tensor in state.values())
            for state in self.personalized_states.values()
        )
        benchmark_ineligibility_reasons: list[str] = []
        if not self.method_resolution.benchmark_eligible:
            benchmark_ineligibility_reasons.append("method_config_ineligible")
        if not self.coordinator.method_compatibility.benchmark_eligible:
            benchmark_ineligibility_reasons.append("method_scope_ineligible")
        if not self.coordinator.model_benchmark_eligible:
            benchmark_ineligibility_reasons.append("model_ineligible")
        if not self.resume_validation.benchmark_eligible:
            benchmark_ineligibility_reasons.append("resume_validation_ineligible")
        if self.coordinator.strategy.oracle:
            benchmark_ineligibility_reasons.append("oracle_strategy")
        if not primary_metric_complete:
            benchmark_ineligibility_reasons.append(
                "undefined_expected_primary_metric_cells"
            )
        if not validation_primary_metric_complete:
            benchmark_ineligibility_reasons.append(
                "undefined_expected_validation_primary_metric_cells"
            )
        benchmark_eligible = not benchmark_ineligibility_reasons
        ledger = self.resource_ledger.to_dict()
        result = {
            "result_schema": RESULT_SCHEMA,
            "result_schema_version": 2,
            "runtime_version": RUNTIME_VERSION,
            "benchmark_name": "UEFA",
            "stream_id": self.coordinator.stream.stream_id,
            "stream_hash": self.coordinator.stream.stream_hash,
            "scientific_fingerprint": self.stream_scientific_fingerprint,
            "scientific_fingerprint_source": self.stream_scientific_fingerprint_source,
            "stream_content_digest": self.stream_content_digest,
            "source_sha": self.source_sha,
            "source_provenance_type": self.source_provenance_type,
            "source_tree_digest": self.source_tree_digest,
            "source_tree_clean": self.source_tree_clean,
            "tracked_source_tree_clean": self.tracked_source_tree_clean,
            "untracked_runtime_source_paths": list(self.untracked_runtime_source_paths),
            "run_config_digest": self.run_config_digest,
            "method_config_digest": self.method_config_digest,
            "method_config": self.method_config,
            "method_config_resolution": self.method_resolution.to_dict(),
            "strategy": self.coordinator.strategy_name,
            "client_continual_algorithm": self.coordinator.algorithm_name,
            "algorithm_parameters": dict(self.coordinator.algorithm_parameters),
            "method_support_status": self.method_resolution.support_status,
            "model_support_status": (
                self.coordinator.model_compatibility.release_status
                if self.coordinator.model_compatibility is not None
                else "unregistered_custom_model"
            ),
            "model_benchmark_tier": (
                self.coordinator.model_compatibility.benchmark_tier
                if self.coordinator.model_compatibility is not None
                else "unsupported_custom"
            ),
            "model_scientific_fidelity": (
                self.coordinator.model_compatibility.scientific_fidelity
                if self.coordinator.model_compatibility is not None
                else "not_verified"
            ),
            "model_benchmark_eligible": self.coordinator.model_benchmark_eligible,
            "strategy_benchmark_eligible": not self.coordinator.strategy.oracle,
            "method_config_benchmark_eligible": self.method_resolution.benchmark_eligible,
            "primary_metric_benchmark_eligible": primary_metric_protocol_complete,
            "benchmark_eligible": benchmark_eligible,
            "benchmark_ineligibility_reasons": benchmark_ineligibility_reasons,
            "scientific_fidelity": self.method_resolution.scientific_fidelity,
            "model": self.coordinator.model_name,
            "resolved_model": self.coordinator.resolved_model_name,
            "execution_device": str(self.coordinator.device),
            "deterministic_execution": dict(
                self.coordinator.deterministic_execution
            ),
            "initial_model_state_digest": self.coordinator.initial_model_state_digest,
            "model_seed": self.coordinator.resolved_model_seed,
            "training_budget": {
                "optimizer": "Adam",
                "optimizer_state_checkpointed": False,
                "learning_rate": self.coordinator.stream.config.training.learning_rate,
                "weight_decay": self.coordinator.stream.config.training.weight_decay,
                "rounds_per_stage": rounds_per_stage,
                "local_epochs_per_round": self.coordinator.stream.config.training.local_epochs_per_round,
                "strategy_optimizer_steps_per_local_epoch": int(
                    getattr(self.strategy, "local_optimizer_steps_per_epoch", 1)
                ),
                "effective_optimizer_steps_per_round": (
                    self.coordinator.stream.config.training.local_epochs_per_round
                    * int(
                        getattr(
                            self.strategy, "local_optimizer_steps_per_epoch", 1
                        )
                    )
                ),
            },
            "strategy_hyperparameters": {
                "fedprox_mu": (
                    self.coordinator.stream.config.training.fedprox_mu
                    if self.coordinator.strategy_name == "fedprox"
                    else None
                ),
            },
            "strategy_diagnostics": _json_safe(dict(self.strategy.diagnostics())),
            "client_algorithm_hyperparameters": {
                client_id: client.algorithm.hyperparameters()
                for client_id, client in self.coordinator.clients.items()
            },
            "client_algorithm_local_state_keys": {
                client_id: client.algorithm.local_state_keys()
                for client_id, client in self.coordinator.clients.items()
            },
            "client_method_diagnostics": {
                client_id: _json_safe(client.algorithm.diagnostics())
                for client_id, client in self.coordinator.clients.items()
            },
            "evaluation_model": self.evaluation_model,
            "class_mask_policy": self.coordinator.class_mask_policy,
            "base_metric": self.coordinator.stream.scenario.metrics[0],
            "client_stage_task_matrix": self.matrix,
            "A[k,s,t]": self.matrix,
            "query_counts": self.query_counts,
            "client_stage_task_trajectories": self._trajectories(),
            "client_stage_task_matrix_population_verified": True,
            "primary_metric_support": primary_metric_support,
            "validation_primary_metric_support": validation_primary_metric_support,
            "in_memory_stream_tensors_unchanged": True,
            "summary": summary,
            "primary_metric_aggregation": (
                "positive_query_micro"
                if self.coordinator.stream.scenario.metrics[0]
                .lower()
                .startswith("hits@")
                else "macro_cell"
            ),
            "macro_client_metric_role": (
                "diagnostic_only"
                if self.coordinator.stream.scenario.metrics[0]
                .lower()
                .startswith("hits@")
                else "co_primary_summary"
            ),
            "conditional_primary_metric_summary": summary,
            "official_primary_metric_summary": (
                summary if primary_metric_complete else None
            ),
            "primary_metric_summary_population": (
                "complete_expected_population"
                if primary_metric_complete
                else "conditional_on_finite_primary_metric_cells"
            ),
            "personalized_global_performance_gap": self._personalized_global_gap(),
            "rounds": self.round_records,
            "stages": self.stage_records,
            "graph_specific_report": {
                **self.coordinator.stream.partition.diagnostics,
                **(
                    {
                        key: value
                        for key, value in self.stage_records[-1].items()
                        if key.startswith(("boundary_", "interior_"))
                    }
                    if self.stage_records
                    else {}
                ),
            },
            "communication_payload_bytes": ledger["communication_payload_bytes"],
            "training_wire_bytes": ledger["training_wire_bytes"],
            "evaluation_sync_bytes": ledger["evaluation_sync_bytes"],
            "resource_ledger": ledger,
            "persistent_state_measurement_semantics": (
                "logical live tensor payload bytes; model/shared payloads counted once per owner"
            ),
            "checkpoint_records": self.checkpoint_records,
            "resume_validation": self.resume_validation.to_dict(),
            "runtime_seconds": runtime_seconds,
            "checkpoint_io_records": self.checkpoint_io_records,
            "peak_memory_bytes": (
                peak_memory if self.coordinator.device.type == "cuda" else None
            ),
            "model_parameter_count": model_parameters,
            "trainable_parameter_count": trainable_parameters,
            "personalized_parameter_count": personalized_parameters,
        }
        self.coordinator.run_logger.log(summary, step=len(self.round_records) + 1)
        return result
