"""Small JSONL logger for offline reproducible runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Dict

from gecko.evaluation.reports import json_safe


class LocalLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        payload = {"step": step, "metrics": json_safe(metrics)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def finish(self) -> None:
        return None
