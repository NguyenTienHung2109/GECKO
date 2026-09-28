from __future__ import annotations

import argparse
import json
import logging
from typing import Sequence
from gecko.engine import REFERENCE_MODES
from gecko.algorithms.catalog import MethodRegistry
from gecko.models.registry import ModelRegistry
from gecko.benchmarks.terminology import PAPER_TASK_ORDERS

from gecko.workflows._logging import LOGGER


def build_parser() -> argparse.ArgumentParser:
    from gecko.workflows.inspect import command_audit
    from gecko.workflows.inspect import command_audit_bias
    from gecko.workflows.inspect import command_gc_objects
    from gecko.workflows.construct import command_generate
    from gecko.workflows.inspect import command_list_methods
    from gecko.workflows.inspect import command_list_models
    from gecko.workflows.inspect import command_migrate_stream
    from gecko.workflows.inspect import command_report
    from gecko.workflows.run import command_run
    parser = argparse.ArgumentParser(prog="gecko", description="GECKO graph-learning benchmark")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="plan an LPT campaign; use gecko plan --help for its options")
    generate = subparsers.add_parser("generate", aliases=["construct"], help="construct a fixed stream")
    generate.add_argument("--config", required=True)
    generate.add_argument("--spatial-profile", choices=("easy", "mild", "hard"), help="Legacy heuristic partition profile.")
    generate.add_argument("--allocation-alpha", type=float, help="Explicit allocation alpha; exact construction uses examples/benchmark.py.")
    generate.add_argument("--dirichlet-alpha", type=float, help=argparse.SUPPRESS)
    generate.add_argument("--task-order", choices=PAPER_TASK_ORDERS)
    generate.add_argument(
        "--order-profile",
        choices=("synchronized", "mild", "hard", "unconstrained", "binary_mismatch"),
        help="Legacy task-order profile (compatibility).",
    )
    generate.add_argument("--seed", type=int)
    generate.add_argument("--output-root")
    generate.add_argument("--participation-fraction", type=float)
    generate.add_argument("--rounds-per-stage", type=int)
    generate.set_defaults(handler=command_generate)
    run = subparsers.add_parser("run", help="run a method on an existing stream")
    run.add_argument("--config", required=True)
    run.add_argument("--stream")
    run.add_argument("--output-root")
    run.add_argument(
        "--strategy",
        default="fedavg",
        choices=(
            "local_only",
            "fedavg",
            "fedprox",
            "scaffold",
            "fedgta",
            "fed_pub",
            "feddc",
            "power_uefa",
            "motion",
            "fedfst",
            "centralized_shard_oracle",
            "joint_oracle",
        ),
    )
    run.add_argument(
        "--cl-algorithm",
        default="Bare",
        choices=MethodRegistry().cli_names(),
    )
    run.add_argument(
        "--allow-experimental-placeholder",
        action="store_true",
        help="allow a generic mechanism that is not benchmark eligible",
    )
    run.add_argument(
        "--model", default="auto", choices=("auto",) + ModelRegistry().factory_names()
    )
    run.add_argument("--spatial-profile", choices=("easy", "mild", "hard"), help="Legacy heuristic partition profile.")
    run.add_argument("--allocation-alpha", type=float, help="Client-allocation concentration alpha.")
    run.add_argument("--dirichlet-alpha", type=float, help=argparse.SUPPRESS)
    run.add_argument("--task-order", choices=PAPER_TASK_ORDERS)
    run.add_argument("--num-clients", type=int)
    run.add_argument(
        "--class-task-policy",
        choices=("benchmark_defined", "drop_rarest_train_lpt_balanced_v1"),
    )
    run.add_argument(
        "--order-profile",
        choices=("synchronized", "mild", "hard", "unconstrained", "binary_mismatch"),
        help="Legacy task-order profile (compatibility).",
    )
    run.add_argument("--seed", type=int)
    run.add_argument("--allow-short-hard-order", action="store_true")
    run.add_argument("--participation-fraction", type=float)
    run.add_argument("--rounds-per-stage", type=int)
    run.add_argument("--local-epochs-per-round", type=int)
    run.add_argument("--fedprox-mu", type=float)
    run.add_argument("--hidden-size", type=int)
    run.add_argument("--num-layers", type=int)
    run.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"))
    run.add_argument("--device", default="cpu")
    run.add_argument(
        "--model-seed",
        type=int,
        help="model initialization seed; does not alter the fixed stream artifact",
    )
    run.add_argument(
        "--reference-mode",
        choices=tuple(REFERENCE_MODES),
        help=(
            "NC-Domain centralized diagnostic visibility; requires "
            "centralized_shard_oracle and never enters the leaderboard"
        ),
    )
    run.add_argument(
        "--method-config",
        help="explicit UEFA v2 method/strategy YAML; excluded from stream identity",
    )
    run.add_argument("--checkpoint-dir")
    run.add_argument("--checkpoint-every", type=int)
    run.add_argument("--resume-from")
    run.add_argument(
        "--evaluation-model",
        choices=(
            "strategy",
            "shared",
            "post-broadcast",
            "post-local",
            "personalized",
        ),
        default="strategy",
    )
    run.add_argument(
        "--diagnostic-resume-override",
        help=(
            "nonempty reason for an identity-mismatched diagnostic resume; "
            "forces benchmark_eligible=false"
        ),
    )
    run.add_argument("--run-tag")
    run.add_argument(
        "--allow-ineligible-stream",
        action="store_true",
        help="allow diagnostic training while forcing benchmark_eligible=false",
    )
    run.add_argument(
        "--allow-legacy-stream-identity",
        action="store_true",
        help=(
            "accept an older stored stream-hash formula only when the requested "
            "stream-defining config matches the config embedded in that stream"
        ),
    )
    run.set_defaults(handler=command_run)
    audit = subparsers.add_parser("audit", help="verify checksums and manifest")
    audit.add_argument("--stream", required=True)
    audit.add_argument(
        "--signature-policy",
        choices=("allow_unsigned", "require_signed", "ignore"),
        default="allow_unsigned",
    )
    audit.set_defaults(handler=command_audit)
    bias = subparsers.add_parser(
        "audit-bias", help="audit retained-query bias and LP partition provenance"
    )
    bias.add_argument("--stream", required=True)
    bias.add_argument("--output")
    bias.add_argument(
        "--allow-blockers",
        action="store_true",
        help="report release blockers without returning a failing exit status",
    )
    bias.set_defaults(handler=command_audit_bias)
    report = subparsers.add_parser("report", help="print aggregate result files")
    report.add_argument("--stream", required=True)
    report.add_argument("--include-ineligible", action="store_true")
    report.set_defaults(handler=command_report)
    gc = subparsers.add_parser(
        "gc-objects", help="find unreferenced content-addressed objects"
    )
    gc.add_argument("--root", required=True)
    gc_mode = gc.add_mutually_exclusive_group()
    gc_mode.add_argument("--dry-run", action="store_true")
    gc_mode.add_argument("--execute", action="store_true")
    gc.set_defaults(handler=command_gc_objects)
    migrate = subparsers.add_parser(
        "migrate-stream", help="repackage a verified schema-v1 stream as schema v2"
    )
    migrate.add_argument("--stream", required=True)
    migrate.add_argument("--output-root", required=True)
    migrate.add_argument(
        "--trusted-legacy",
        action="store_true",
        help="allow pickle-capable loading only for a locally trusted old stream",
    )
    migrate.set_defaults(handler=command_migrate_stream)
    methods = subparsers.add_parser("list-methods")
    method_scope = methods.add_mutually_exclusive_group()
    method_scope.add_argument("--all", action="store_true")
    method_scope.add_argument("--original", action="store_true")
    method_scope.add_argument("--experimental", action="store_true")
    methods.set_defaults(handler=command_list_methods)
    models = subparsers.add_parser("list-models")
    model_scope = models.add_mutually_exclusive_group()
    model_scope.add_argument("--all", action="store_true")
    model_scope.add_argument("--factories", action="store_true")
    model_scope.add_argument("--discovered", action="store_true")
    models.set_defaults(handler=command_list_models)
    compatibility = subparsers.add_parser("compatibility")
    compatibility.set_defaults(
        handler=lambda args: (
            print(json.dumps(MethodRegistry().as_records(), indent=2, sort_keys=True))
            or 0
        )
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "plan":
        from gecko.cli.plan import main as campaign_main
        return campaign_main([*arguments[1:], "--stage", "plan"])
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return int(args.handler(args))






_RELOCATED_EXPORTS = {'_atomic_write_text': ('gecko.workflows.run', '_atomic_write_text'), '_execution_repository_provenance': ('gecko.workflows.run', '_execution_repository_provenance'), '_load_method_config': ('gecko.workflows.run', '_load_method_config'), '_with_profile_overrides': ('gecko.workflows.construct', '_with_profile_overrides'), 'command_audit': ('gecko.workflows.inspect', 'command_audit'), 'command_audit_bias': ('gecko.workflows.inspect', 'command_audit_bias'), 'command_gc_objects': ('gecko.workflows.inspect', 'command_gc_objects'), 'command_generate': ('gecko.workflows.construct', 'command_generate'), 'command_list_methods': ('gecko.workflows.inspect', 'command_list_methods'), 'command_list_models': ('gecko.workflows.inspect', 'command_list_models'), 'command_migrate_stream': ('gecko.workflows.inspect', 'command_migrate_stream'), 'command_report': ('gecko.workflows.inspect', 'command_report'), 'command_run': ('gecko.workflows.run', 'command_run'), 'expected_stream_path': ('gecko.workflows.construct', 'expected_stream_path')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)


if __name__ == "__main__":
    raise SystemExit(main())
