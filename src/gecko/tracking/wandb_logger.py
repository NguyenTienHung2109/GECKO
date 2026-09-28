"""Credential-safe optional Weights & Biases integration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from typing import Dict

from gecko.evaluation.reports import json_safe


class WandbLogger:
    """Lazy W&B logger supporting online, offline, auto, and disabled modes."""

    def __init__(
        self,
        *,
        mode: str,
        project: str = "UEFA",
        config: Dict[str, Any] | None = None,
        run_name: str | None = None,
        directory: str | Path | None = None,
    ) -> None:
        self.run = None
        self.mode = mode
        self.actual_mode = "disabled"
        environment_mode = os.getenv("WANDB_MODE", "").lower()
        if mode == "auto" and environment_mode in {"online", "offline", "disabled"}:
            mode = environment_mode
        if mode == "disabled":
            return
        try:
            import wandb
        except ImportError:
            if mode == "online":
                raise RuntimeError("W&B online mode requested but wandb is not installed.")
            return
        authenticated = bool(os.getenv("WANDB_API_KEY"))
        if not authenticated:
            try:
                authenticated = bool(getattr(wandb.api, "api_key", None))
            except Exception:
                authenticated = False
        requested = mode
        if mode == "auto":
            requested = "online" if authenticated else "offline"
        if requested == "online" and not authenticated:
            requested = "offline"
        self.actual_mode = requested
        init_kwargs: Dict[str, Any] = {
            "project": os.getenv("WANDB_PROJECT", project),
            "entity": os.getenv("WANDB_ENTITY") or None,
            "mode": requested,
            "config": json_safe(config or {}),
            "name": run_name,
            "reinit": True,
        }
        if directory is not None:
            init_kwargs["dir"] = str(directory)
        self.run = wandb.init(**init_kwargs)

    def log(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(json_safe(metrics), step=step)

    def update_config(self, values: Dict[str, Any]) -> None:
        if self.run is not None:
            self.run.config.update(json_safe(values), allow_val_change=True)

    def log_artifact(self, path: str | Path, *, name: str, artifact_type: str) -> None:
        if self.run is None:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
