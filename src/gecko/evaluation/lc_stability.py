"""Support-aware metric stability analysis for edge classification."""

from __future__ import annotations

import math
from typing import Any
from typing import Iterable

import torch


LC_STABILITY_POLICY = {
    "minimum_initialization_runs": 5,
    "maximum_primary_initialization_std": 0.03,
    "maximum_b10_initialization_std": 0.05,
    "maximum_primary_bootstrap_ci_width": 0.05,
    "minimum_mean_client_rank_correlation": 0.50,
    "minimum_bootstrap_samples": 200,
}


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    tensor = torch.tensor(list(values), dtype=torch.double)
    if tensor.numel() == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan")}
    standard_deviation = float(tensor.std(unbiased=False))
    half_width = 1.96 * standard_deviation / math.sqrt(tensor.numel())
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "std": standard_deviation,
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean_ci95_low": float(tensor.mean() - half_width),
        "mean_ci95_high": float(tensor.mean() + half_width),
    }


def _fixed_macro_f1(
    predictions: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> float:
    scores = []
    for class_id in range(num_classes):
        predicted = predictions == class_id
        actual = labels == class_id
        true_positive = (predicted & actual).sum().double()
        precision = true_positive / predicted.sum().clamp_min(1)
        recall = true_positive / actual.sum().clamp_min(1)
        scores.append(2 * precision * recall / (precision + recall).clamp_min(1e-12))
    return float(torch.stack(scores).mean())


def _accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    return float((predictions == labels).double().mean())


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty(values.shape[0], dtype=torch.double)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    first_rank = _average_ranks(first.double())
    second_rank = _average_ranks(second.double())
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if denominator == 0:
        return 1.0 if torch.equal(first_rank, second_rank) else 0.0
    return float((first_centered * second_centered).sum() / denominator)


def _interval(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.double)
    low = torch.quantile(tensor, 0.025)
    high = torch.quantile(tensor, 0.975)
    return {
        "mean": float(tensor.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci95_width": float(high - low),
    }


def _bootstrap_metrics(
    cells: list[dict[str, Any]],
    *,
    num_classes: int,
    samples: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    labels = torch.cat([cell["labels"].long() for cell in cells])
    predictions = torch.cat([cell["predictions"].long() for cell in cells])
    clients = sorted({int(cell["client"]) for cell in cells})
    per_client = {
        client: (
            torch.cat([cell["labels"].long() for cell in cells if cell["client"] == client]),
            torch.cat(
                [cell["predictions"].long() for cell in cells if cell["client"] == client]
            ),
        )
        for client in clients
    }
    accuracy_values, macro_values, b10_values = [], [], []
    bottom_count = max(1, math.ceil(len(clients) * 0.1))
    for _ in range(samples):
        selected = torch.randint(labels.shape[0], (labels.shape[0],), generator=generator)
        accuracy_values.append(_accuracy(predictions[selected], labels[selected]))
        macro_values.append(
            _fixed_macro_f1(predictions[selected], labels[selected], num_classes)
        )
        client_scores = []
        for client in clients:
            client_labels, client_predictions = per_client[client]
            client_selected = torch.randint(
                client_labels.shape[0], (client_labels.shape[0],), generator=generator
            )
            client_scores.append(
                _accuracy(client_predictions[client_selected], client_labels[client_selected])
            )
        b10_values.append(
            float(torch.sort(torch.tensor(client_scores)).values[:bottom_count].mean())
        )
    return {
        "accuracy_micro_f1": _interval(accuracy_values),
        "pooled_fixed_universe_macro_f1": _interval(macro_values),
        "bottom_10_percent_client_accuracy": _interval(b10_values),
    }


def analyze_lc_metric_stability(
    runs: list[dict[str, Any]],
    *,
    num_classes: int,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Analyze fixed-query predictions from repeated model initializations."""

    if len(runs) < LC_STABILITY_POLICY["minimum_initialization_runs"]:
        raise ValueError("LC stability requires at least five initialization runs.")
    if bootstrap_samples < LC_STABILITY_POLICY["minimum_bootstrap_samples"]:
        raise ValueError("LC stability requires at least 200 bootstrap samples.")
    expected_keys = {
        (int(cell["client"]), int(cell["task"])) for cell in runs[0]["cells"]
    }
    reference_labels = {
        (int(cell["client"]), int(cell["task"])): cell["labels"].long()
        for cell in runs[0]["cells"]
    }
    reference_query_ids = {
        (int(cell["client"]), int(cell["task"])): cell.get("query_ids")
        for cell in runs[0]["cells"]
    }
    for run in runs:
        keys = {(int(cell["client"]), int(cell["task"])) for cell in run["cells"]}
        if keys != expected_keys:
            raise ValueError("Initialization runs do not use identical client-task cells.")
        for cell in run["cells"]:
            key = (int(cell["client"]), int(cell["task"]))
            if not torch.equal(cell["labels"].long(), reference_labels[key]):
                raise ValueError("Initialization runs do not use identical query labels.")
            expected_query_ids = reference_query_ids[key]
            if expected_query_ids is not None and not torch.equal(
                cell.get("query_ids"), expected_query_ids
            ):
                raise ValueError("Initialization runs do not use identical query IDs.")

    relevant_by_task: dict[int, set[int]] = {}
    for cell in runs[0]["cells"]:
        relevant_by_task.setdefault(int(cell["task"]), set()).update(
            cell["labels"].long().tolist()
        )
    clients = sorted({key[0] for key in expected_keys})
    bottom_count = max(1, math.ceil(len(clients) * 0.1))
    run_rows, client_vectors, bootstrap_rows = [], [], []
    cell_accuracy: dict[tuple[int, int], list[float]] = {
        key: [] for key in expected_keys
    }
    for run_index, run in enumerate(runs):
        labels = torch.cat([cell["labels"].long() for cell in run["cells"]])
        predictions = torch.cat([cell["predictions"].long() for cell in run["cells"]])
        client_scores, eligible_macro = [], []
        ineligible_cells = 0
        for client in clients:
            selected_cells = [cell for cell in run["cells"] if cell["client"] == client]
            client_labels = torch.cat([cell["labels"].long() for cell in selected_cells])
            client_predictions = torch.cat(
                [cell["predictions"].long() for cell in selected_cells]
            )
            client_scores.append(_accuracy(client_predictions, client_labels))
        for cell in run["cells"]:
            key = (int(cell["client"]), int(cell["task"]))
            cell_labels = cell["labels"].long()
            cell_predictions = cell["predictions"].long()
            cell_accuracy[key].append(_accuracy(cell_predictions, cell_labels))
            eligible = (
                cell_labels.numel() > 1
                and set(cell_labels.tolist()) == relevant_by_task[key[1]]
            )
            if eligible:
                eligible_macro.append(
                    _fixed_macro_f1(cell_predictions, cell_labels, num_classes)
                )
            else:
                ineligible_cells += 1
        client_tensor = torch.tensor(client_scores)
        client_vectors.append(client_tensor)
        run_rows.append(
            {
                "model_seed": int(run["model_seed"]),
                "accuracy": _accuracy(predictions, labels),
                "micro_f1": _accuracy(predictions, labels),
                "pooled_fixed_universe_macro_f1": _fixed_macro_f1(
                    predictions, labels, num_classes
                ),
                "eligible_client_task_macro_f1": (
                    sum(eligible_macro) / len(eligible_macro)
                    if eligible_macro
                    else float("nan")
                ),
                "macro_f1_ineligible_cell_count": ineligible_cells,
                "bottom_10_percent_client_accuracy": float(
                    torch.sort(client_tensor).values[:bottom_count].mean()
                ),
                "client_accuracy": client_scores,
            }
        )
        bootstrap_rows.append(
            {
                "model_seed": int(run["model_seed"]),
                **_bootstrap_metrics(
                    run["cells"],
                    num_classes=num_classes,
                    samples=bootstrap_samples,
                    generator=torch.Generator().manual_seed(seed + run_index),
                ),
            }
        )

    correlations = [
        _spearman(client_vectors[first], client_vectors[second])
        for first in range(len(client_vectors))
        for second in range(first + 1, len(client_vectors))
    ]
    one_query_variance, larger_cell_variance = [], []
    for key, values in cell_accuracy.items():
        target = one_query_variance if reference_labels[key].numel() == 1 else larger_cell_variance
        target.append(float(torch.tensor(values).std(unbiased=False)))
    initialization = {
        "accuracy_micro_f1": _summary(row["accuracy"] for row in run_rows),
        "pooled_fixed_universe_macro_f1": _summary(
            row["pooled_fixed_universe_macro_f1"] for row in run_rows
        ),
        "bottom_10_percent_client_accuracy": _summary(
            row["bottom_10_percent_client_accuracy"] for row in run_rows
        ),
        "mean_pairwise_client_rank_spearman": (
            sum(correlations) / len(correlations) if correlations else 1.0
        ),
        "pairwise_client_rank_spearman": correlations,
    }
    maximum_bootstrap_width = max(
        row["accuracy_micro_f1"]["ci95_width"] for row in bootstrap_rows
    )
    failures, warnings = [], []
    if initialization["accuracy_micro_f1"]["std"] > LC_STABILITY_POLICY[
        "maximum_primary_initialization_std"
    ]:
        failures.append("primary_initialization_variance_exceeds_policy")
    if initialization["bottom_10_percent_client_accuracy"]["std"] > LC_STABILITY_POLICY[
        "maximum_b10_initialization_std"
    ]:
        failures.append("b10_initialization_variance_exceeds_policy")
    if maximum_bootstrap_width > LC_STABILITY_POLICY[
        "maximum_primary_bootstrap_ci_width"
    ]:
        failures.append("primary_query_bootstrap_interval_exceeds_policy")
    if initialization["mean_pairwise_client_rank_spearman"] < LC_STABILITY_POLICY[
        "minimum_mean_client_rank_correlation"
    ]:
        warnings.append("client_ranking_is_initialization_sensitive")
    return {
        "metric_protocol": {
            "primary": "accuracy_micro_f1",
            "secondary": "pooled_fixed_universe_macro_f1",
            "bottom_client": "bottom_10_percent_client_accuracy",
            "client_task_macro_f1_policy": (
                "diagnostic_only_when_more_than_one_query_and_all_task-test "
                "classes_are_present"
            ),
        },
        "policy": LC_STABILITY_POLICY,
        "run_rows": run_rows,
        "initialization_stability": initialization,
        "query_bootstrap": bootstrap_rows,
        "maximum_primary_bootstrap_ci_width": maximum_bootstrap_width,
        "support_effect": {
            "one_query_cell_accuracy_std": _summary(one_query_variance),
            "larger_cell_accuracy_std": _summary(larger_cell_variance),
        },
        "failures": failures,
        "warnings": warnings,
        "stability_status": "pass" if not failures else "fail",
    }
