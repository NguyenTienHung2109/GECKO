from __future__ import annotations

from gecko.benchmarks.lpt_dirichlet_temporal import LPT_DIRICHLET_TEMPORAL_VERSION
from gecko.workflows._campaign_paths import ROOT

from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any
from typing import Mapping
from typing import Sequence
from gecko.engine.checkpoint import canonical_json_digest  # noqa: E402
from gecko.benchmarks.lpt_dirichlet_temporal import LPT_DIRICHLET_TEMPORAL_VERSION
from gecko.engine.runtime import _resolve_tracked_source_tree  # noqa: E402

def _run_commands(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    device: str,
    model: str,
    wandb_mode: str,
    worker_index: int,
    worker_count: int,
    run_tag_prefix: str = "lpt",
    allow_ineligible_manipulation_audit: bool = False,
    allow_legacy_protocol: bool = False,
) -> list[list[str]]:
    from gecko.workflows._campaign_paths import ROOT
    if (
        manifest.get("protocol_version") != LPT_DIRICHLET_TEMPORAL_VERSION
        and not allow_legacy_protocol
    ):
        raise RuntimeError(
            "The stream manifest was constructed under a different LPT protocol; "
            "construct a versioned v4 stream grid before training."
        )
    manipulation = manifest.get("dirichlet_manipulation_audit")
    if (
        not isinstance(manipulation, Mapping)
        or not bool(manipulation.get("benchmark_eligible"))
    ) and not allow_ineligible_manipulation_audit:
        reasons = (
            manipulation.get("failure_reasons", [])
            if isinstance(manipulation, Mapping)
            else ["missing_dirichlet_manipulation_audit"]
        )
        raise RuntimeError(
            "Dirichlet manipulation gate failed closed; training commands were "
            f"not generated. Reasons: {reasons}"
        )
    commands: list[list[str]] = []
    config_path = str(Path(str(plan["config_path"])))
    records = list(manifest.get("records", []))
    methods = list(plan.get("runnable_methods", []))
    cell_index = 0
    for record in records:
        for method in methods:
            order_profiles = method.get("order_profiles")
            if (
                order_profiles is not None
                and str(record["order_profile"]) not in order_profiles
            ):
                continue
            if cell_index % worker_count != worker_index:
                cell_index += 1
                continue
            command = [
                sys.executable,
                "-m",
                "gecko.cli.main",
                "run",
                "--config",
                config_path,
                "--stream",
                str(record["stream_path"]),
                "--strategy",
                str(method["strategy"]),
                "--cl-algorithm",
                str(method["continual_method"]),
                "--method-config",
                str(ROOT / str(method["method_config"])),
                "--device",
                device,
                "--model",
                str(method.get("model", model)),
                "--seed",
                str(record["seed"]),
                "--model-seed",
                str(record["seed"]),
                "--dirichlet-alpha",
                str(record["alpha_dirichlet"]),
                "--num-clients",
                str(manifest["num_clients"]),
                "--order-profile",
                str(record["order_profile"]),
                "--rounds-per-stage",
                str(manifest["rounds_per_stage"]),
                "--local-epochs-per-round",
                str(manifest["local_epochs_per_round"]),
                "--wandb-mode",
                wandb_mode,
                "--allow-ineligible-stream",
                "--run-tag",
                f"{run_tag_prefix}-a{record['alpha_dirichlet']:g}-{record['order_profile']}",
            ]
            if bool(record.get("lpt_plan", {}).get("applicable")):
                command.extend(
                    [
                        "--class-task-policy",
                        "drop_rarest_train_lpt_balanced_v1",
                    ]
                )
            if bool(record.get("short_hard_order_override")):
                command.append("--allow-short-hard-order")
            if allow_legacy_protocol:
                command.append("--allow-legacy-stream-identity")
            if str(method["strategy"]) == "fedprox":
                fedprox_mu = float(method["fedprox_mu"])
                command.extend(["--fedprox-mu", f"{fedprox_mu:g}"])
            commands.append(command)
            cell_index += 1
    return commands


def _with_run_training_overrides(
    manifest: Mapping[str, Any], *, local_epochs: int | None
) -> dict[str, Any]:
    """Apply training-only overrides without changing an immutable stream."""

    effective = dict(manifest)
    if local_epochs is not None:
        if isinstance(local_epochs, bool) or local_epochs < 1:
            raise ValueError("local-epochs must be a positive integer.")
        effective["local_epochs_per_round"] = int(local_epochs)
    return effective


def _command_option(command: Sequence[str], option: str) -> str | None:
    """Return one explicit CLI option value from a generated command."""

    try:
        index = command.index(option)
    except ValueError:
        return None
    if index + 1 >= len(command):
        raise ValueError(f"Generated command has no value for {option}.")
    return command[index + 1]


def _completed_result_path(command: Sequence[str]) -> Path | None:
    """Resolve the exact v4 result path produced by a generated run command."""
    from gecko.algorithms.campaigns import _method_payload

    stream = _command_option(command, "--stream")
    strategy = _command_option(command, "--strategy")
    algorithm = _command_option(command, "--cl-algorithm")
    model = _command_option(command, "--model")
    method_config = _command_option(command, "--method-config")
    if None in {stream, strategy, algorithm, model, method_config}:
        return None
    payload = _method_payload(Path(str(method_config)))
    digest = canonical_json_digest(payload)[:12]
    run_tag = _command_option(command, "--run-tag")
    stem = f"{strategy}-{algorithm}-{model}"
    if run_tag:
        stem += f"-{run_tag}"
    return Path(str(stream)) / "results" / f"{stem}-mc{digest}.json"


@lru_cache(maxsize=1)
def _current_source_tree_digest() -> str:
    """Return the exact runtime-source identity used by result provenance."""

    digest, _, _ = _resolve_tracked_source_tree()
    return digest


def _assert_source_tree_unchanged(expected_digest: str) -> None:
    """Stop a panel before source edits can create a mixed-code result table."""

    current_digest, _, _ = _resolve_tracked_source_tree()
    if current_digest != expected_digest:
        raise RuntimeError(
            "Runtime source changed after this panel was planned. The runner "
            "stopped fail-closed before mixing source-tree digests. Restart the "
            "panel with --skip-completed after source changes are finished."
        )


def _normalized_run_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Compare scientific options independently of cache and output locations."""

    normalized = dict(value)
    config = dict(normalized["config"])
    scenario = dict(config["scenario"])
    scenario.pop("save_path", None)
    config["scenario"] = scenario
    config.pop("output_root", None)
    config.pop("wandb", None)
    normalized["config"] = config
    return normalized


def _matches_run_configuration(
    command: Sequence[str], payload: Mapping[str, Any]
) -> bool:
    """Validate saved training options against the current generated command."""

    from dataclasses import replace

    import torch

    from gecko.cli.main import build_parser
    from gecko.config import GECKOConfig
    from gecko.data.streams import audit_stream, stream_identity
    from gecko.workflows.construct import _with_profile_overrides

    saved = payload.get("run_configuration")
    if not isinstance(saved, Mapping) or not isinstance(saved.get("config"), Mapping):
        return False
    if not isinstance(saved["config"].get("scenario"), Mapping):
        return False
    if "run" not in command or _command_option(command, "--config") is None:
        return False
    args = build_parser().parse_args(command[command.index("run"):])
    requested = _with_profile_overrides(GECKOConfig.from_yaml(args.config), args)
    stream_path = Path(str(args.stream))
    stored = GECKOConfig.from_yaml(stream_path / "config.yaml")
    manifest = json.loads((stream_path / "manifest.json").read_text(encoding="utf-8"))
    _, requested_hash = stream_identity(requested)
    if requested_hash != manifest["config_hash"]:
        _, stored_hash = stream_identity(stored)
        if not (args.allow_legacy_stream_identity and requested_hash == stored_hash):
            return False
    effective = replace(
        stored,
        training=requested.training,
        wandb=requested.wandb,
        output_root=requested.output_root,
        benchmark_seeds=requested.benchmark_seeds,
    )
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    expected = {
        "config": effective.to_dict(),
        "strategy": args.strategy,
        "cl_algorithm": args.cl_algorithm,
        "model": args.model,
        "model_seed": effective.seed if args.model_seed is None else args.model_seed,
        "device": str(torch.device(device)),
        "reference_mode": args.reference_mode,
        "evaluation_model": args.evaluation_model,
    }
    if canonical_json_digest(_normalized_run_configuration(saved)) != canonical_json_digest(
        _normalized_run_configuration(expected)
    ):
        return False
    audit = audit_stream(stream_path)
    integrity = payload["artifact_integrity"]
    return all(
        payload.get(field) == audit[field]
        and integrity.get(f"{field}_before") == audit[field]
        and integrity.get(f"{field}_after") == audit[field]
        for field in ("scientific_fingerprint", "package_digest")
    )


def _is_completed_result(command: Sequence[str]) -> bool:
    """Accept only an intact result matching this exact generated command."""
    from gecko.algorithms.campaigns import _method_payload

    path = _completed_result_path(command)
    if path is None or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    integrity = payload.get("artifact_integrity")
    return bool(
        payload.get("method_config_digest")
        == canonical_json_digest(_method_payload(Path(str(_command_option(command, "--method-config")))))
        and payload.get("source_tree_digest") == _current_source_tree_digest()
        and isinstance(integrity, Mapping)
        and integrity.get("unchanged") is True
        and payload.get("in_memory_stream_tensors_unchanged") is True
        and _matches_run_configuration(command, payload)
    )
