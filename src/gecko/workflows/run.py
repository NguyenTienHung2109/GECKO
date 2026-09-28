from __future__ import annotations

import argparse
import os
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Mapping
import uuid
from gecko.config import GECKOConfig
from gecko.benchmarks.terminology import allocation_metadata
from gecko.data.streams.builder import stream_relative_path
import yaml
from gecko.evaluation.reports import json_safe
from gecko.engine import FederatedCoordinator
from gecko.tracking import WandbLogger
from gecko.algorithms.catalog import MethodRegistry
from gecko.algorithms.method_config import validate_method_config
from gecko.engine.checkpoint import canonical_json_digest
from gecko.data.streams import audit_stream
from gecko.data.streams import load_stream
from gecko.data.streams import stream_identity

def _execution_repository_provenance() -> dict[str, object]:
    """Report optional metadata without requiring a particular source checkout."""

    return {
        "provenance_type": "unavailable",
        "source_identifier": None,
        "repository_current_commit_SHA": None,
        "tracked_worktree_clean": None,
        "tracked_status": [],
    }


def _load_method_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load a JSON-safe explicit v2 method configuration and its digest."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Method configuration must be a YAML mapping.")
    if any(not isinstance(key, str) or not key for key in raw):
        raise ValueError("Method configuration keys must be non-empty strings.")
    config = dict(raw)
    digest = canonical_json_digest(config)
    return config, digest


def _atomic_write_text(path: Path, payload: str) -> None:
    """Publish a v2 result atomically, with the final rename as commit point."""

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def command_run(args: argparse.Namespace) -> int:
    from gecko.benchmarks.paper import scenario_id_for_config
    from gecko.workflows.construct import _with_profile_overrides
    from gecko.workflows.construct import expected_stream_path
    execution_provenance = _execution_repository_provenance()
    config = _with_profile_overrides(GECKOConfig.from_yaml(args.config), args)

    method_config = None
    method_config_digest = None
    method_config_path = getattr(args, "method_config", None)
    if method_config_path:
        method_config, method_config_digest = _load_method_config(method_config_path)
        validate_method_config(
            method_config,
            expected_strategy=args.strategy,
            expected_continual_method=args.cl_algorithm,
            problem_type=config.scenario.problem,
            incremental_setting=config.scenario.incremental_setting,
        )
        if method_config.get("name") == "fedfst_uefa_v1":
            if args.model not in {"uefa_gcn", "begin_gcn", "fedfst_gat"}:
                raise ValueError(
                    "FedFST requires an explicit --model fedfst_gat, uefa_gcn, "
                    "or begin_gcn; "
                    "auto fallback is not an auditable model identity. Client-local "
                    "buffers are frozen during server HLST and never aggregated."
                )
    elif args.cl_algorithm in MethodRegistry().explicit_v2_names():
        raise ValueError(
            f"The {args.cl_algorithm} continual method requires an explicit "
            "validated --method-config before any stream is loaded."
        )
    elif args.strategy == "scaffold":
        raise ValueError(
            "The scaffold strategy requires an explicit --method-config; "
            "the legacy coordinator path must not approximate SCAFFOLD as FedAvg."
        )
    elif (
        any(
            getattr(args, name, None) is not None
            for name in (
                "checkpoint_dir",
                "checkpoint_every",
                "resume_from",
                "diagnostic_resume_override",
            )
        )
        or getattr(args, "evaluation_model", "strategy") != "strategy"
    ):
        raise ValueError(
            "Checkpoint and evaluation-model options require --method-config."
        )
    stream_path = Path(args.stream) if args.stream else expected_stream_path(config)
    if not args.stream and not stream_path.exists():
        historical_path = Path(config.output_root) / stream_relative_path(config, legacy=True)
        if historical_path.exists():
            stream_path = historical_path
    if not stream_path.exists():
        raise FileNotFoundError(
            f"Fixed stream does not exist at {stream_path}. Run `generate` first; `run` never regenerates it."
        )
    artifact_before = audit_stream(stream_path)
    stream = load_stream(stream_path)
    _, requested_stream_hash = stream_identity(config)
    if requested_stream_hash != stream.stream_hash:
        _, stored_config_current_hash = stream_identity(stream.config)
        legacy_identity_matches = bool(
            getattr(args, "allow_legacy_stream_identity", False)
            and requested_stream_hash == stored_config_current_hash
        )
        if not legacy_identity_matches:
            raise ValueError(
                "The run config changes a stream-defining field. Generate or select "
                "the matching fixed stream instead."
            )
    stream = replace(
        stream,
        config=replace(
            stream.config,
            training=config.training,
            wandb=config.wandb,
            output_root=config.output_root,
            benchmark_seeds=config.benchmark_seeds,
        ),
    )
    run_suffix = f"-{args.run_tag}" if args.run_tag else ""
    method_suffix = f"-mc{method_config_digest[:12]}" if method_config_digest else ""
    run_name = (
        f"{args.strategy}-{args.cl_algorithm}-{args.model}-{stream.scenario.dataset_name}-"
        f"{stream.scenario.problem_type}-{stream.scenario.incremental_type}-seed{stream.config.seed}"
        f"{('-' + args.reference_mode) if args.reference_mode else ''}"
        f"{run_suffix}{method_suffix}"
    )
    manifest = json.loads((stream_path / "manifest.json").read_text(encoding="utf-8"))
    release_status = manifest.get("release_status", "unassessed")
    stream_benchmark_eligible = release_status == "eligible"
    if not stream_benchmark_eligible and not args.allow_ineligible_stream:
        raise ValueError(
            f"Stream release_status={release_status!r}; pass "
            "--allow-ineligible-stream only for diagnostic training."
        )
    spatial = stream.partition.diagnostics
    spatial_summary = {
        key: spatial.get(key)
        for key in (
            "edge_cut_ratio",
            "boundary_node_ratio",
            "client_query_volume_coefficient_of_variation",
            "internal_query_coverage",
            "lc_internal_query_coverage",
            "lp_internal_positive_coverage",
            "lp_internal_candidate_coverage",
        )
        if key in spatial
    }
    logger = WandbLogger(
        mode=config.wandb.mode,
        project=config.wandb.project,
        config={
            **config.to_dict(),
            "stream_id": stream.stream_id,
            "stream_hash": stream.stream_hash,
            "scientific_fingerprint": manifest.get("scientific_fingerprint"),
            "package_digest": manifest.get("package_digest"),
            "git_sha": manifest.get("repository_current_commit_SHA"),
            "strategy": args.strategy,
            "diagnostic_reference_mode": args.reference_mode,
            "client_continual_algorithm": args.cl_algorithm,
            "method_version": 2 if method_config is not None else 1,
            "method_config": method_config,
            "method_config_digest": method_config_digest,
            "result_schema": "uefa-run-result-v2"
            if method_config is not None
            else "uefa-v1",
            "model": args.model,
            "model_seed": args.model_seed,
            "num_clients": stream.config.partition.num_clients,
            "num_tasks": stream.scenario.num_tasks,
            "scenario_id": scenario_id_for_config(stream.config),
            "target_alpha": stream.config.partition.dirichlet_alpha,
            **allocation_metadata(stream.config),
            "realized_spatial_diagnostics": spatial_summary,
            "order_diagnostics": stream.orders.diagnostics,
            "stream_invariant_status": manifest.get("invariant_status"),
            "stream_quality_status": manifest.get("quality_status"),
            "stream_release_status": release_status,
            "stream_benchmark_eligible": stream_benchmark_eligible,
            "replay_memory_assumption": (
                "client-local; unbounded prior-task train-only graph history"
                if args.strategy == "fedfst"
                else (
                    "client-local; fixed 16777216-byte serialized SSM ceiling"
                    if args.cl_algorithm == "SSM"
                    else (
                        "client-local; balanced condensed graph memory with "
                        f"{dict(method_config['continual_method']['parameters'])['memory_ceiling_bytes']} byte ceiling"
                        if args.cl_algorithm == "CaT" and method_config is not None
                        else "client-local; default adapter capacity 128"
                    )
                )
            ),
        },
        run_name=run_name,
        directory=stream_path,
    )
    coordinator = FederatedCoordinator(
        stream,
        strategy_name=args.strategy,
        algorithm_name=args.cl_algorithm,
        model_name=args.model,
        run_logger=logger,
        allow_experimental_placeholder=args.allow_experimental_placeholder,
        device=args.device,
        reference_mode=args.reference_mode,
        model_seed=args.model_seed,
        method_config=method_config,
        checkpoint_dir=getattr(args, "checkpoint_dir", None),
        checkpoint_every=getattr(args, "checkpoint_every", None),
        resume_from=getattr(args, "resume_from", None),
        evaluation_model=getattr(args, "evaluation_model", "strategy"),
        stream_scientific_fingerprint=manifest.get("scientific_fingerprint"),
        source_sha=execution_provenance["source_identifier"],
        diagnostic_resume_override=getattr(args, "diagnostic_resume_override", None),
    )
    logger.update_config(
        {
            "resolved_model": coordinator.resolved_model_name,
            "model_parameter_count": sum(
                parameter.numel() for parameter in coordinator.global_model.parameters()
            ),
            "method_support_status": coordinator.method_compatibility.support_status,
            "model_support_status": (
                coordinator.model_compatibility.release_status
                if coordinator.model_compatibility is not None
                else "unregistered_custom_model"
            ),
            "model_benchmark_eligible": coordinator.model_benchmark_eligible,
            "benchmark_eligible": (
                coordinator.method_compatibility.benchmark_eligible
                and coordinator.model_benchmark_eligible
                and stream_benchmark_eligible
                and not coordinator.strategy.oracle
            ),
            "scientific_fidelity": coordinator.method_compatibility.scientific_fidelity,
        }
    )
    result = coordinator.run()
    artifact_after = audit_stream(stream_path)
    artifact_unchanged = artifact_before == artifact_after
    if not artifact_unchanged:
        raise RuntimeError("Stream artifacts changed during benchmark execution.")
    result["artifact_integrity"] = {
        "unchanged": True,
        "scientific_fingerprint_before": artifact_before["scientific_fingerprint"],
        "scientific_fingerprint_after": artifact_after["scientific_fingerprint"],
        "package_digest_before": artifact_before["package_digest"],
        "package_digest_after": artifact_after["package_digest"],
    }
    result["scientific_fingerprint"] = manifest["scientific_fingerprint"]
    result["package_digest"] = manifest["package_digest"]
    result["stream_release_status"] = release_status
    result["execution_repository_provenance"] = execution_provenance
    result.update(allocation_metadata(stream.config))
    result["scenario_id"] = scenario_id_for_config(stream.config)
    result["run_configuration"] = {
        "config": stream.config.to_dict(),
        "strategy": args.strategy,
        "cl_algorithm": args.cl_algorithm,
        "model": args.model,
        "model_seed": coordinator.resolved_model_seed,
        "device": str(coordinator.device),
        "reference_mode": args.reference_mode,
        "evaluation_model": getattr(args, "evaluation_model", "strategy"),
    }
    result["benchmark_eligible"] = bool(
        result.get("benchmark_eligible", True)
        and coordinator.method_compatibility.benchmark_eligible
        and coordinator.model_benchmark_eligible
        and stream_benchmark_eligible
        and not coordinator.strategy.oracle
    )
    result_dir = stream_path / "results"
    result_dir.mkdir(exist_ok=True)
    result_stem = f"{args.strategy}-{args.cl_algorithm}-{args.model}"
    if args.reference_mode:
        result_stem += f"-{args.reference_mode}"
    if args.run_tag:
        result_stem += f"-{args.run_tag}"
    if method_suffix:
        result_stem += method_suffix
    result_path = result_dir / f"{result_stem}.json"
    result_payload = json.dumps(json_safe(result), indent=2, sort_keys=True)
    if method_config is None:
        result_path.write_text(result_payload, encoding="utf-8")
    else:
        _atomic_write_text(result_path, result_payload + "\n")
    if logger.run is not None:
        logger.log_artifact(
            stream_path / "manifest.json",
            name=f"{stream.stream_id}-manifest",
            artifact_type="uefa-stream-manifest",
        )
        logger.log_artifact(
            result_path,
            name=f"{stream.stream_id}-{args.strategy}-{args.cl_algorithm}-{args.model}-results",
            artifact_type="uefa-aggregate-results",
        )
    logger.finish()
    print(result_path)
    return 0
