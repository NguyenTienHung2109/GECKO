from __future__ import annotations

"""Metric implementations that do not require scikit-learn."""

from typing import Dict

import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1) if logits.ndim > 1 else (logits > 0).long()
    return float((predictions == labels.long()).float().mean())


def macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1) if logits.ndim > 1 else (logits > 0).long()
    classes = torch.unique(torch.cat([predictions.long(), labels.long()]))
    scores = []
    for class_id in classes.tolist():
        predicted = predictions == class_id
        actual = labels == class_id
        true_positive = (predicted & actual).sum().float()
        precision = true_positive / predicted.sum().clamp_min(1)
        recall = true_positive / actual.sum().clamp_min(1)
        scores.append(2 * precision * recall / (precision + recall).clamp_min(1e-12))
    return float(torch.stack(scores).mean()) if scores else float("nan")


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(0).double()
    starts = ends - counts.double() + 1
    average_ranks = 0.5 * (starts + ends)
    sorted_ranks = torch.repeat_interleave(average_ranks, counts)
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    positive_rank_sum = ranks[positives].sum()
    count_positive = positives.sum().double()
    count_negative = negatives.sum().double()
    auc = (positive_rank_sum - count_positive * (count_positive + 1) / 2) / (
        count_positive * count_negative
    )
    return float(auc)


def rocauc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.ndim == 1:
        return _binary_auc(logits.reshape(-1), labels.long())
    scores = []
    for column in range(labels.shape[1]):
        valid = labels[:, column] >= 0
        value = _binary_auc(logits[valid, column], labels[valid, column].long())
        if value == value:
            scores.append(value)
    return sum(scores) / len(scores) if scores else float("nan")


def negative_binary_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> float:
    """Return missing-target-aware binary log score, with higher being better.

    UEFA represents unavailable multilabel targets with negative values.  They
    are excluded from the score exactly as they are from ``rocauc``.  Unlike
    ROC-AUC, binary cross entropy remains defined when a local evaluation shard
    contains only one target class, which makes the negated loss suitable as a
    complete client-local primary metric.  A shard with no observed target at
    all remains undefined instead of being assigned an artificial score.
    """

    if logits.shape != labels.shape:
        raise ValueError(
            "negative_binary_cross_entropy requires aligned logits and labels."
        )
    valid = labels >= 0
    if not bool(valid.any()):
        return float("nan")
    value = F.binary_cross_entropy_with_logits(
        logits[valid], labels[valid].to(dtype=logits.dtype)
    )
    return -float(value)


def hits_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    *,
    group_ids: torch.Tensor | None = None,
    tie_policy: str = "pessimistic",
) -> float:
    """Compute Hits@K within explicit candidate groups."""

    if tie_policy not in {"optimistic", "pessimistic", "average"}:
        raise ValueError(f"Unknown Hits@K tie policy: {tie_policy}")
    groups = (
        torch.zeros_like(labels, dtype=torch.long)
        if group_ids is None
        else group_ids.long()
    )
    if groups.shape != labels.shape:
        raise ValueError("Hits@K group_ids must align with labels.")
    credits = []
    for group in torch.unique(groups[labels == 1]).tolist():
        selected = groups == group
        negatives = scores[selected & (labels == 0)]
        positives = scores[selected & (labels == 1)]
        if negatives.numel() < k:
            raise ValueError(
                f"Hits@{k} candidate group {group} has only "
                f"{negatives.numel()} negatives."
            )
        threshold = torch.topk(negatives, k).values[-1]
        if tie_policy == "optimistic":
            group_credit = (positives >= threshold).float()
        elif tie_policy == "average":
            group_credit = (positives > threshold).float()
            group_credit += 0.5 * (positives == threshold).float()
        else:
            group_credit = (positives > threshold).float()
        credits.append(group_credit)
    return float(torch.cat(credits).mean()) if credits else float("nan")


def mean_reciprocal_rank(scores: torch.Tensor, labels: torch.Tensor) -> float:
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    if positives.numel() == 0:
        return float("nan")
    ranks = 1 + (negatives.unsqueeze(0) >= positives.unsqueeze(1)).sum(dim=1)
    return float((1.0 / ranks.float()).mean())


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    if (labels == 1).sum() == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    ordered = labels[order].float()
    precision = ordered.cumsum(0) / torch.arange(1, ordered.numel() + 1)
    return float((precision * ordered).sum() / ordered.sum())


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metrics: tuple[str, ...],
    *,
    candidate_group_ids: torch.Tensor | None = None,
    hits_tie_policy: str = "pessimistic",
) -> Dict[str, float]:
    output = {}
    for metric in metrics:
        normalized = metric.lower()
        if normalized == "accuracy":
            output[metric] = accuracy(logits, labels)
        elif normalized == "macro_f1":
            output[metric] = macro_f1(logits, labels)
        elif normalized == "rocauc":
            output[metric] = rocauc(logits, labels)
        elif normalized == "negative_binary_cross_entropy":
            output[metric] = negative_binary_cross_entropy(logits, labels)
        elif normalized.startswith("hits@"):
            output[metric] = hits_at_k(
                logits.reshape(-1),
                labels.reshape(-1),
                int(normalized.split("@")[1]),
                group_ids=(
                    None
                    if candidate_group_ids is None
                    else candidate_group_ids.reshape(-1)
                ),
                tie_policy=hits_tie_policy,
            )
        elif normalized == "mrr":
            output[metric] = mean_reciprocal_rank(logits.reshape(-1), labels.reshape(-1))
        elif normalized == "average_precision":
            output[metric] = average_precision(logits.reshape(-1), labels.reshape(-1))
        else:
            raise ValueError(f"Unsupported UEFA metric: {metric}")
    return output


"""Order-aware continual metrics over A[k, s, t]."""

from typing import Dict

import torch

from gecko.types import OrderPlan


def summarize_continual_matrix(
    matrix: torch.Tensor,
    orders: OrderPlan,
    query_counts: torch.Tensor,
    *,
    base_metric: str,
) -> Dict[str, float]:
    num_clients, num_stages, num_tasks = matrix.shape
    final = matrix[:, -1, :]
    valid_final = ~torch.isnan(final)
    client_final = torch.nanmean(final, dim=1)
    macro_client = float(torch.nanmean(client_final))
    macro_cell = float(torch.nanmean(final))
    flattened = client_final[~torch.isnan(client_final)]
    bottom_count = max(1, int(torch.ceil(torch.tensor(flattened.numel() * 0.1))))
    bottom = float(torch.sort(flattened).values[:bottom_count].mean()) if flattened.numel() else float("nan")
    weighted_numerator = torch.nansum(final * query_counts)
    weighted_denominator = query_counts[valid_final].sum().clamp_min(1)
    micro_query = float(weighted_numerator / weighted_denominator)
    forgetting_values = []
    for client in range(num_clients):
        for task in range(num_tasks):
            first_stage = orders.inverse_client_orders[client][task]
            history = matrix[client, first_stage:, task]
            history = history[~torch.isnan(history)]
            if history.numel():
                forgetting_values.append(history.max() - history[-1])
    forgetting = float(torch.stack(forgetting_values).mean()) if forgetting_values else float("nan")
    output = {
        # Hits@K is an average over positive ranking queries.  Once
        # ``query_counts`` stores positive support, this weighted value is the
        # exact pooled/micro result.  Equal-weight client/task averages remain
        # explicit diagnostics instead of silently serving as the headline.
        "final_average_performance": (
            micro_query
            if base_metric.lower().startswith("hits@")
            else macro_cell
        ),
        "average_forgetting": forgetting,
        "bottom_10_percent_client_final_performance": bottom,
        "macro_client_final_performance": macro_client,
        "macro_cell_final_performance": macro_cell,
        "micro_query_final_performance": micro_query,
    }
    if base_metric == "accuracy":
        output["AFA"] = output["final_average_performance"]
        output["AF"] = forgetting
    return output
