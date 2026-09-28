from __future__ import annotations

from gecko.benchmarks.lpt_dirichlet_temporal import LPT_DIRICHLET_TEMPORAL_VERSION
from gecko.workflows._campaign_paths import ROOT

import json
import os
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence
from gecko.config import GECKOConfig  # noqa: E402
from gecko.benchmarks.lpt_dirichlet_temporal import LPT_DIRICHLET_TEMPORAL_VERSION
from gecko.algorithms.method_config import validate_method_config  # noqa: E402

def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_run_plan(
    config_path: Path,
    methods: Sequence[str],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    """Resolve compatibility without loading a dataset or starting training."""
    from gecko.workflows._campaign_paths import ROOT
    from gecko.algorithms.campaigns import _method_payload
    from gecko.algorithms.campaigns import _resolve_methods

    config = GECKOConfig.from_yaml(config_path)
    problem = config.scenario.problem.upper()
    incremental = config.scenario.incremental_setting.lower()
    key = (problem, incremental)
    runnable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cell in _resolve_methods(methods):
        if key not in cell.scenarios:
            skipped.append(
                {
                    "method": cell.label,
                    "scenario": f"{problem}-{incremental}",
                    "reason": "method_scenario_not_implemented",
                }
            )
            continue
        method_config = cell.method_config(problem, incremental)
        strategy = cell.strategy_for(problem, incremental)
        resolution = validate_method_config(
            _method_payload(method_config),
            expected_strategy=strategy,
            expected_continual_method=cell.continual_method,
            problem_type=problem,
            incremental_setting=incremental,
        )
        row: dict[str, Any] = {
            "method": cell.label,
            "strategy": strategy,
            "continual_method": cell.continual_method,
            "method_config": str(method_config.relative_to(ROOT)),
            "support_status": resolution.support_status,
            "benchmark_eligible_before_promotion": resolution.benchmark_eligible,
        }
        if strategy == "fedprox":
            row["fedprox_mu"] = float(config.training.fedprox_mu)
        if cell.order_profiles is not None:
            row["order_profiles"] = sorted(cell.order_profiles)
        if cell.model is not None:
            row["model"] = cell.model
        runnable.append(row)
    return {
        "schema": "lpt-dirichlet-temporal-run-plan",
        "version": 4,
        "protocol_version": LPT_DIRICHLET_TEMPORAL_VERSION,
        "scenario": f"{problem}-{incremental}",
        "config_path": str(config_path),
        "stream_manifest": str(manifest_path),
        "runnable_methods": runnable,
        "skipped_methods": skipped,
    }


