from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence
from gecko.config import GECKOConfig  # noqa: E402
from gecko.benchmarks.lpt_dirichlet_temporal import DEFAULT_ALPHAS
from gecko.benchmarks.lpt_dirichlet_temporal import DEFAULT_TEMPORAL_PROFILES
from gecko.benchmarks.lpt_dirichlet_temporal import LPT_DIRICHLET_TEMPORAL_VERSION
from gecko.benchmarks.lpt_dirichlet_temporal import prepare_stream_grid



from gecko.workflows._campaign_paths import ROOT


















def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "construct", "run", "all"), default="plan")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--temporal", nargs="+", choices=DEFAULT_TEMPORAL_PROFILES, default=list(DEFAULT_TEMPORAL_PROFILES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-clients", type=int)
    parser.add_argument("--rounds-per-stage", type=int)
    parser.add_argument(
        "--local-epochs",
        type=int,
        help=(
            "training-only local epochs per round; may be overridden at run "
            "time without reconstructing immutable streams"
        ),
    )
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--branch-factor", type=int, default=4)
    parser.add_argument("--lc-maximum-search-expansions", type=int, default=200_000)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="disabled")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument(
        "--run-tag-prefix",
        default="lpt",
        help="result-tag namespace for a provenance-isolated campaign",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="skip only intact v4 result files matching the generated command",
    )
    parser.add_argument(
        "--allow-ineligible-manipulation-audit",
        action="store_true",
        help=(
            "permit an explicitly diagnostic one-seed pilot when the aggregate "
            "alpha manipulation gate fails; the manifest remains ineligible"
        ),
    )
    parser.add_argument(
        "--allow-legacy-protocol",
        action="store_true",
        help=(
            "run against an explicitly supplied older immutable manifest "
            "without relabeling it as the current protocol"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from gecko.workflows.plan import _atomic_json
    from gecko.workflows.campaign_run import _current_source_tree_digest
    from gecko.workflows.campaign_run import _is_completed_result
    from gecko.workflows.campaign_run import _run_commands
    from gecko.workflows.campaign_run import _with_run_training_overrides
    from gecko.workflows.plan import build_run_plan
    args = _parser().parse_args(argv)
    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must lie in [0, worker-count).")
    manifest_path = args.output_root / "stream_manifest.json"
    plan_path = args.output_root / "run_plan.json"
    plan = build_run_plan(args.config, args.methods, manifest_path=manifest_path)
    _atomic_json(plan_path, plan)
    if args.stage == "plan":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.stage in {"construct", "all"}:
        base = GECKOConfig.from_yaml(args.config)
        manifest = prepare_stream_grid(
            args.config,
            store_root=args.store_root,
            alphas=args.alphas,
            temporal_profiles=args.temporal,
            seeds=args.seeds,
            num_clients=args.num_clients,
            rounds_per_stage=args.rounds_per_stage,
            local_epochs_per_round=args.local_epochs,
            beam_width=args.beam_width,
            branch_factor=args.branch_factor,
            lc_maximum_search_expansions=args.lc_maximum_search_expansions,
            repository_root=ROOT,
        )
        manifest.update(
            {
                "num_clients": args.num_clients or base.partition.num_clients,
                "rounds_per_stage": args.rounds_per_stage or base.training.rounds_per_stage,
                "local_epochs_per_round": args.local_epochs or base.training.local_epochs_per_round,
            }
        )
        _atomic_json(manifest_path, manifest)
        print(f"Prepared {len(manifest['records'])} immutable stream cells.", flush=True)
        if args.stage == "construct":
            return 0
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = _with_run_training_overrides(
            manifest, local_epochs=args.local_epochs
        )
    commands = _run_commands(
        plan,
        manifest,
        device=args.device,
        model=args.model,
        wandb_mode=args.wandb_mode,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        run_tag_prefix=args.run_tag_prefix,
        allow_ineligible_manipulation_audit=(
            args.allow_ineligible_manipulation_audit
        ),
        allow_legacy_protocol=args.allow_legacy_protocol,
    )
    source_tree_digest = _current_source_tree_digest()
    command_path = args.output_root / f"commands-worker-{args.worker_index}.json"
    _atomic_json(
        command_path,
        {
            "protocol_version": LPT_DIRICHLET_TEMPORAL_VERSION,
            "stream_manifest_protocol_version": manifest.get("protocol_version"),
            "source_tree_digest": source_tree_digest,
            "commands": commands,
        },
    )
    if args.print_commands:
        for command in commands:
            print(" ".join(command))
    runnable_commands = (
        [command for command in commands if not _is_completed_result(command)]
        if args.skip_completed
        else commands
    )
    if args.skip_completed:
        print(
            f"Resuming: skipped {len(commands) - len(runnable_commands)} intact results; "
            f"running {len(runnable_commands)} cells.",
            flush=True,
        )
    for index, command in enumerate(runnable_commands, start=1):
        print(f"[{index}/{len(runnable_commands)}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    return 0






_RELOCATED_EXPORTS = {'METHODS': ('gecko.algorithms.campaigns', 'METHODS'), 'MethodCell': ('gecko.algorithms.campaigns', 'MethodCell'), '_assert_source_tree_unchanged': ('gecko.workflows.campaign_run', '_assert_source_tree_unchanged'), '_atomic_json': ('gecko.workflows.plan', '_atomic_json'), '_command_option': ('gecko.workflows.campaign_run', '_command_option'), '_completed_result_path': ('gecko.workflows.campaign_run', '_completed_result_path'), '_current_source_tree_digest': ('gecko.workflows.campaign_run', '_current_source_tree_digest'), '_is_completed_result': ('gecko.workflows.campaign_run', '_is_completed_result'), '_method_payload': ('gecko.algorithms.campaigns', '_method_payload'), '_resolve_methods': ('gecko.algorithms.campaigns', '_resolve_methods'), '_run_commands': ('gecko.workflows.campaign_run', '_run_commands'), '_with_run_training_overrides': ('gecko.workflows.campaign_run', '_with_run_training_overrides'), 'build_run_plan': ('gecko.workflows.plan', 'build_run_plan')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)

from gecko.workflows._campaign_paths import DEFAULT_OUTPUT, DEFAULT_STORE, LC_ALL, LP_DOMAIN, NC_ALL, NC_CLASS, NC_TASK, NC_TASK_CLASS, ROOT


if __name__ == "__main__":
    raise SystemExit(main())
