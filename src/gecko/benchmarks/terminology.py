"""Paper-facing allocation and task-order names with historical wire aliases."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from gecko.validation import ConfigurationError

if TYPE_CHECKING:
    from gecko.config import GECKOConfig


PAPER_TASK_ORDERS = ("synchronized", "unsynchronized")
LEGACY_TASK_ORDER_PROFILES = ("mild", "hard", "unconstrained", "binary_mismatch")


def normalize_task_order(value: str, *, problem: str, num_tasks: int) -> str:
    """Translate the paper's unsynchronized setting without changing its order plan."""

    if value != "unsynchronized":
        return value
    if problem.upper() == "LP" and num_tasks == 2:
        return "binary_mismatch"
    if problem.upper() in {"NC", "LC"}:
        return "hard"
    raise ConfigurationError("Unsynchronized task order is defined for NC/LC and two-task LP.")


def task_order_label(profile: str) -> str:
    """Describe task order without treating legacy two-task blocks as paper conditions."""

    if profile in {"hard", "binary_mismatch", "unsynchronized"}:
        return "unsynchronized"
    if profile == "synchronized":
        return profile
    return f"legacy-{profile}"


def allocation_slug(config: GECKOConfig) -> str:
    """Describe actual allocation control; heuristic profiles are never numeric alpha."""

    alpha = config.partition.dirichlet_alpha
    if alpha is None:
        return f"allocation-legacy-{config.partition.spatial_profile}"
    if not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ConfigurationError("allocation_alpha must be finite and positive.")
    return f"allocation-alpha-{format(float(alpha), '.12g')}"


def allocation_metadata(config: GECKOConfig) -> dict[str, object]:
    """Return truthful public allocation fields without guessing a Dirichlet target."""

    alpha = config.partition.dirichlet_alpha
    return {
        "allocation_alpha": None if alpha is None else float(alpha),
        "allocation_control": "legacy_heuristic" if alpha is None else "dirichlet",
        "allocation_label": allocation_slug(config),
        "task_order": task_order_label(config.order.profile),
    }
