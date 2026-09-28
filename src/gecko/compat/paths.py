"""Resolve source paths recorded by historical policies without editing them."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def _source_paths() -> dict[str, str]:
    return json.loads(Path(__file__).with_name("source_paths.json").read_text(encoding="utf-8"))


def resolve_source_path(path: str | Path, *, root: str | Path | None = None) -> Path:
    """Map only repository source/config references; leave artifact paths intact."""
    from gecko.reproducibility import repository_root

    value = Path(path)
    base = Path(root) if root is not None else repository_root()
    if value.is_absolute():
        try:
            relative = value.relative_to(base).as_posix()
        except ValueError:
            return value
    else:
        relative = value.as_posix()
    mapped = _source_paths().get(relative)
    if mapped is None:
        for old, new in (
            ("configs/uefa_v1/methods", "configs/gecko_v1/algorithms"),
            ("configs/uefa_v1/experimental", "configs/gecko_v1/campaigns"),
            ("configs/uefa_v1", "configs/gecko_v1/scenarios"),
        ):
            if relative == old or relative.startswith(old + "/"):
                mapped = new + relative[len(old):]
                break
    if mapped is None:
        return value
    return base / mapped if value.is_absolute() else Path(mapped)
