#!/usr/bin/env python3
"""Derive immutable 50x1 grids from existing 10x1 aggregate manifests."""

from __future__ import annotations

from gecko.compat.paths import resolve_source_path

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
from typing import Mapping
from typing import Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gecko.config import GECKOConfig  # noqa: E402
from gecko.data.partitioning.base import participation_hash  # noqa: E402
from gecko.data.streams import audit_stream
from gecko.data.streams import derive_participation_variant
from gecko.types import ParticipationPlan  # noqa: E402


SCOPES = ("nc_task", "nc_class", "nc_domain", "lc_task", "lc_class", "lc_domain")
SOURCE_ROUNDS = 10
TARGET_ROUNDS = 50
LOCAL_EPOCHS = 1
PROTOCOL = "lpt_exact_dirichlet_temporal_participation50_derived_v1"
PRESERVED_COMPONENTS = ("scenario", "partition", "queries", "evaluation")
PRESERVED_ORDER_FIELDS = (
    "canonical_order",
    "client_orders",
    "inverse_client_orders",
    "cohort_assignments",
    "block_size",
    "seed",
)


def read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(payload), handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_derivation(source: Path, target: Path) -> Mapping[str, Any]:
    source_manifest = read_json(source / "manifest.json")
    target_manifest = read_json(target / "manifest.json")
    for component in PRESERVED_COMPONENTS:
        if source_manifest["component_digests"][component] != target_manifest[
            "component_digests"
        ][component]:
            raise RuntimeError(
                f"{target}: data component {component!r} changed during derivation"
            )
    changed_files = {"config.yaml", "orders.json", "participation.json"}
    for name, digest in source_manifest["artifact_checksums"].items():
        if name not in changed_files and target_manifest["artifact_checksums"].get(name) != digest:
            raise RuntimeError(f"{target}: immutable payload changed: {name}")
    source_orders = read_json(source / "orders.json")
    target_orders = read_json(target / "orders.json")
    for field in PRESERVED_ORDER_FIELDS:
        if source_orders[field] != target_orders[field]:
            raise RuntimeError(f"{target}: task-order field changed: {field}")
    source_participation = read_json(source / "participation.json")
    target_participation = read_json(target / "participation.json")
    if source_participation["rounds_per_stage"] != SOURCE_ROUNDS:
        raise RuntimeError(f"{source}: expected source participation {SOURCE_ROUNDS} rounds")
    if target_participation["rounds_per_stage"] != TARGET_ROUNDS:
        raise RuntimeError(f"{target}: expected target participation {TARGET_ROUNDS} rounds")
    for stage, rounds in source_participation["trace"].items():
        for round_id, clients in rounds.items():
            if target_participation["trace"][stage][round_id] != clients:
                raise RuntimeError(
                    f"{target}: old participation prefix changed at stage={stage}, "
                    f"round={round_id}"
                )
    target_audit = audit_stream(target)
    if not target_audit.get("valid"):
        raise RuntimeError(f"Derived target failed audit: {target}")
    return target_manifest


def derive_scope(
    scope: str,
    *,
    source_manifest_path: Path,
    target_store_root: Path,
    target_artifact_root: Path,
) -> Mapping[str, Any]:
    aggregate = read_json(source_manifest_path)
    if aggregate.get("rounds_per_stage") != SOURCE_ROUNDS:
        raise ValueError(
            f"{scope}: source aggregate must be {SOURCE_ROUNDS}x{LOCAL_EPOCHS}"
        )
    if aggregate.get("local_epochs_per_round") != LOCAL_EPOCHS:
        raise ValueError(f"{scope}: source aggregate is not one local epoch")
    records = aggregate.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{scope}: source aggregate contains no records")
    seeds = {int(record["seed"]) for record in records}
    if seeds != {0, 1, 2}:
        raise ValueError(f"{scope}: expected exactly seeds 0,1,2; found {sorted(seeds)}")

    derived_records = []
    for index, record in enumerate(records, start=1):
        source = Path(str(record["stream_path"])).resolve()
        print(
            f"[{scope} {index}/{len(records)}] derive "
            f"seed={record['seed']} alpha={record['alpha_dirichlet']} "
            f"order={record['order_profile']}",
            flush=True,
        )
        target = derive_participation_variant(
            source,
            TARGET_ROUNDS,
            root=target_store_root,
            repository_root=ROOT,
        )
        target_manifest = verify_derivation(source, target)
        participation_payload = read_json(target / "participation.json")
        target_participation = ParticipationPlan(
            trace={
                int(stage): {
                    int(round_id): tuple(int(client) for client in clients)
                    for round_id, clients in rounds.items()
                }
                for stage, rounds in participation_payload["trace"].items()
            },
            fraction=float(participation_payload["fraction"]),
            rounds_per_stage=int(participation_payload["rounds_per_stage"]),
            seed=int(participation_payload["seed"]),
        )
        derived_records.append(
            {
                **record,
                "stream_path": str(target.resolve()),
                "stream_id": target_manifest["stream_id"],
                "stream_hash": target_manifest["config_hash"],
                "scientific_fingerprint": target_manifest[
                    "scientific_fingerprint"
                ],
                "participation_hash": participation_hash(
                    target_participation
                ),
                "release_status": target_manifest.get(
                    "release_status", "unassessed"
                ),
                "derived_from_stream_id": read_json(source / "manifest.json")[
                    "stream_id"
                ],
            }
        )

    source_config_path = Path(str(aggregate["config_path"])).resolve()
    source_config = GECKOConfig.from_yaml(source_config_path)
    target_config = replace(
        source_config,
        training=replace(
            source_config.training,
            rounds_per_stage=TARGET_ROUNDS,
            local_epochs_per_round=LOCAL_EPOCHS,
        ),
        output_root=".",
    )
    target_config_path = target_artifact_root / scope / "base_config_50x1.yaml"
    atomic_yaml(target_config_path, target_config.to_dict())
    derived = {
        **aggregate,
        "protocol_version": PROTOCOL,
        "config_path": str(target_config_path.resolve()),
        "rounds_per_stage": TARGET_ROUNDS,
        "local_epochs_per_round": LOCAL_EPOCHS,
        "records": derived_records,
        "derived_from": {
            "operation": "participation_round_extension_v1",
            "source_manifest": str(source_manifest_path.resolve()),
            "source_protocol_version": aggregate.get("protocol_version"),
            "source_rounds_per_stage": SOURCE_ROUNDS,
            "preserved_components": list(PRESERVED_COMPONENTS),
            "preserved_order_fields": list(PRESERVED_ORDER_FIELDS),
            "old_trace_is_exact_prefix": True,
        },
    }
    destination = target_artifact_root / scope / "stream_manifest.json"
    atomic_json(destination, derived)
    return derived


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--source-manifest-root",
        type=Path,
        default=resolve_source_path(ROOT / "artifacts" / "all_7_problem_il_v1"),
    )
    result.add_argument(
        "--target-store-root",
        type=Path,
        default=resolve_source_path(ROOT / "generated_streams" / "all_7_problem_il_50x1_v1"),
    )
    result.add_argument(
        "--target-artifact-root",
        type=Path,
        default=resolve_source_path(ROOT / "artifacts" / "all_7_problem_il_50x1_v1"),
    )
    result.add_argument("--scopes", nargs="+", choices=SCOPES, default=SCOPES)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for scope in args.scopes:
        aggregate = derive_scope(
            scope,
            source_manifest_path=(
                args.source_manifest_root.resolve() / scope / "stream_manifest.json"
            ),
            target_store_root=args.target_store_root.resolve() / scope,
            target_artifact_root=args.target_artifact_root.resolve(),
        )
        print(
            f"[{scope}] published {len(aggregate['records'])} audited 50x1 streams",
            flush=True,
        )
    return 0




if __name__ == "__main__":
    raise SystemExit(main())

