from __future__ import annotations

from gecko.reproducibility import repository_root as _repository_root
import argparse
import json
from pathlib import Path
from gecko.data.audits.bias import audit_bias
from gecko.evaluation.reports import json_safe
from gecko.algorithms.catalog import MethodRegistry
from gecko.models.registry import ModelRegistry
from gecko.data.streams import audit_stream
from gecko.data.streams import gc_objects
from gecko.data.streams import load_stream
from gecko.data.streams import migrate_stream_to_v2

def command_audit(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            audit_stream(args.stream, signature_policy=args.signature_policy),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_audit_bias(args: argparse.Namespace) -> int:
    report = audit_bias(load_stream(args.stream))
    payload = json.dumps(json_safe(report), indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["release_eligible"] or args.allow_blockers else 2


def command_report(args: argparse.Namespace) -> int:
    result_dir = Path(args.stream) / "results"
    reports = {}
    for path in sorted(result_dir.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if args.include_ineligible or report.get("benchmark_eligible", False):
            reports[path.name] = report
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


def command_gc_objects(args: argparse.Namespace) -> int:
    report = gc_objects(args.root, dry_run=not args.execute)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_migrate_stream(args: argparse.Namespace) -> int:
    path = migrate_stream_to_v2(
        args.stream,
        args.output_root,
        repository_root=_repository_root(),
        trusted_legacy=args.trusted_legacy,
    )
    print(path)
    return 0


def command_list_methods(args: argparse.Namespace) -> int:
    registry = MethodRegistry()
    if args.original:
        names = registry.original_names()
    elif args.experimental:
        names = registry.experimental_names()
    elif args.all:
        names = registry.original_names() + registry.experimental_names()
    else:
        names = registry.names()
    print("\n".join(names))
    return 0


def command_list_models(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    if args.discovered:
        names = registry.discovered_names()
    elif args.factories:
        names = registry.factory_names()
    elif args.all:
        names = registry.all_names()
    else:
        names = registry.names()
    print("\n".join(names))
    return 0


