"""Validation-only selection for predeclared UEFA training protocols."""

from __future__ import annotations

import math
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class TrainingBudget:
    """The two training-budget controls that are shared across scenarios."""

    rounds_per_stage: int
    local_epochs_per_round: int

    def __post_init__(self) -> None:
        if self.rounds_per_stage <= 0 or self.local_epochs_per_round <= 0:
            raise ValueError("Training budgets must be positive.")

    @property
    def key(self) -> str:
        return f"r{self.rounds_per_stage}-e{self.local_epochs_per_round}"

    @property
    def local_update_cost(self) -> int:
        return self.rounds_per_stage * self.local_epochs_per_round

    def to_dict(self) -> dict[str, int]:
        return {
            "rounds_per_stage": self.rounds_per_stage,
            "local_epochs_per_round": self.local_epochs_per_round,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingBudget":
        return cls(
            rounds_per_stage=int(value["rounds_per_stage"]),
            local_epochs_per_round=int(value["local_epochs_per_round"]),
        )


def validation_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only validation and optimization observables from a run result.

    Test metrics deliberately never enter this function.  This gives the pilot
    a narrow, inspectable interface for choosing a training budget.
    """

    stages = result.get("stages")
    if not isinstance(stages, Sequence) or not stages:
        raise ValueError("Run result has no stage records.")
    trajectory: list[float] = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ValueError("Stage record is not a mapping.")
        value = stage.get("validation_metric")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Run result has a non-finite validation metric.")
        trajectory.append(float(value))

    losses: list[float] = []
    rounds = result.get("rounds", [])
    if not isinstance(rounds, Sequence):
        raise ValueError("Run result has invalid round records.")
    for record in rounds:
        if not isinstance(record, Mapping):
            raise ValueError("Round record is not a mapping.")
        value = record.get("mean_training_loss")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Run result has a non-finite training loss.")
        losses.append(float(value))
    if not losses:
        raise ValueError("Run result has no round records.")

    return {
        "terminal_validation_metric": trajectory[-1],
        "validation_trajectory": trajectory,
        "round_count": len(losses),
        "first_training_loss": losses[0],
        "last_training_loss": losses[-1],
        "all_training_losses_finite": True,
    }


def _candidate_scores(
    report: Mapping[str, Any], expected_budgets: set[str]
) -> dict[str, float]:
    if report.get("status") != "completed":
        raise ValueError(f"Pilot report {report.get('family')!r} is incomplete.")
    candidates = report.get("candidates")
    if not isinstance(candidates, Sequence):
        raise ValueError("Pilot report has invalid candidate records.")
    scores: dict[str, float] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or candidate.get("status") != "completed":
            raise ValueError("Pilot candidate is incomplete.")
        budget = TrainingBudget.from_mapping(candidate["budget"])
        summary = candidate.get("validation")
        if not isinstance(summary, Mapping):
            raise ValueError("Pilot candidate is missing validation summary.")
        metric = summary.get("terminal_validation_metric")
        if not isinstance(metric, (int, float)) or not math.isfinite(metric):
            raise ValueError("Pilot candidate has invalid validation metric.")
        if budget.key in scores:
            raise ValueError(f"Duplicate pilot budget {budget.key!r}.")
        scores[budget.key] = float(metric)
    if set(scores) != expected_budgets:
        raise ValueError("Pilot reports do not contain the predeclared budget grid.")
    return scores


def select_shared_budget(
    family_reports: Mapping[str, Mapping[str, Any]],
    budgets: Sequence[TrainingBudget],
    *,
    absolute_tolerance: float,
) -> dict[str, Any]:
    """Choose the cheapest common budget within each family's validation band."""

    if not family_reports:
        raise ValueError("At least one family report is required.")
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative.")
    if not budgets:
        raise ValueError("At least one predeclared budget is required.")
    ordered = sorted(set(budgets), key=lambda item: (
        item.local_update_cost,
        item.rounds_per_stage,
        item.local_epochs_per_round,
    ))
    expected = {budget.key for budget in ordered}
    scores = {
        family: _candidate_scores(report, expected)
        for family, report in sorted(family_reports.items())
    }
    family_best = {family: max(values.values()) for family, values in scores.items()}
    ranked: list[dict[str, Any]] = []
    for budget in ordered:
        per_family = {
            family: {
                "terminal_validation_metric": values[budget.key],
                "best_terminal_validation_metric": family_best[family],
                "validation_gap_to_best": family_best[family] - values[budget.key],
            }
            for family, values in scores.items()
        }
        eligible = all(
            values["validation_gap_to_best"] <= absolute_tolerance
            for values in per_family.values()
        )
        ranked.append(
            {
                "budget": budget.to_dict(),
                "local_update_cost": budget.local_update_cost,
                "within_tolerance_for_every_family": eligible,
                "per_family": per_family,
            }
        )
    selected = next(
        (candidate for candidate in ranked if candidate["within_tolerance_for_every_family"]),
        None,
    )
    return {
        "selection_target": "shared",
        "selection_scope": "validation_only",
        "test_metrics_used_for_selection": False,
        "absolute_tolerance": absolute_tolerance,
        "family_best_terminal_validation_metrics": family_best,
        "ranked_candidates": ranked,
        "selected_budget": None if selected is None else selected["budget"],
        "selection_status": "selected" if selected is not None else "no_shared_budget",
    }


def select_family_budgets(
    family_reports: Mapping[str, Mapping[str, Any]],
    budgets: Sequence[TrainingBudget],
    *,
    absolute_tolerance: float,
) -> dict[str, Any]:
    """Choose one predeclared validation-only budget for every family.

    Families may have legitimately different optimization scales (for example,
    node classification, edge classification, and link prediction).  This
    selector therefore never forces a universal budget.  It instead applies
    the same predeclared candidate grid, tolerance, and deterministic cost
    tie-break independently to each family.  Test metrics never enter the
    report interface used here.
    """

    if not family_reports:
        raise ValueError("At least one family report is required.")
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative.")
    if not budgets:
        raise ValueError("At least one predeclared budget is required.")
    ordered = sorted(
        set(budgets),
        key=lambda item: (
            item.local_update_cost,
            item.rounds_per_stage,
            item.local_epochs_per_round,
        ),
    )
    expected = {budget.key for budget in ordered}
    scores = {
        family: _candidate_scores(report, expected)
        for family, report in sorted(family_reports.items())
    }
    family_best = {family: max(values.values()) for family, values in scores.items()}
    ranked_by_family: dict[str, list[dict[str, Any]]] = {}
    selected_budgets: dict[str, dict[str, int]] = {}
    for family, values in scores.items():
        best = family_best[family]
        ranked = [
            {
                "budget": budget.to_dict(),
                "local_update_cost": budget.local_update_cost,
                "terminal_validation_metric": values[budget.key],
                "best_terminal_validation_metric": best,
                "validation_gap_to_best": best - values[budget.key],
                "within_tolerance": best - values[budget.key] <= absolute_tolerance,
            }
            for budget in ordered
        ]
        selected = next(candidate for candidate in ranked if candidate["within_tolerance"])
        ranked_by_family[family] = ranked
        selected_budgets[family] = selected["budget"]
    return {
        "selection_target": "family_specific",
        "selection_scope": "validation_only",
        "test_metrics_used_for_selection": False,
        "absolute_tolerance": absolute_tolerance,
        "family_best_terminal_validation_metrics": family_best,
        "ranked_candidates_by_family": ranked_by_family,
        "selected_budgets": selected_budgets,
        "selection_status": "selected",
    }
