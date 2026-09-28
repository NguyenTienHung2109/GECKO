"""Paper-faithful, leakage-safe POWER mechanism oracles."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def power_class_mean_prototypes(
    *,
    node_features: torch.Tensor,
    train_queries: torch.Tensor,
    train_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Eq. (8) class-mean prototypes, class IDs, and counts."""

    if node_features.ndim != 2 or not node_features.is_floating_point():
        raise ValueError("POWER node_features must be floating [nodes, features].")
    if train_queries.ndim != 1 or train_queries.dtype != torch.long:
        raise ValueError("POWER train_queries must be one-dimensional int64.")
    if (
        train_labels.ndim != 1
        or train_labels.dtype != torch.long
        or train_labels.shape[0] != train_queries.shape[0]
    ):
        raise ValueError("POWER train_labels must align with train_queries.")
    if train_queries.numel() == 0:
        raise ValueError("POWER requires at least one current train query.")
    if torch.any(train_queries < 0) or torch.any(
        train_queries >= node_features.shape[0]
    ):
        raise ValueError("POWER train_queries contain a non-local node.")

    device = node_features.device
    queries = train_queries.to(device)
    labels = train_labels.to(device)
    classes = torch.unique(labels).sort().values
    prototypes = []
    counts = []
    for class_id in classes.tolist():
        selected = node_features[queries[labels == int(class_id)]]
        if selected.shape[0] == 0:
            raise RuntimeError("POWER class selection unexpectedly became empty.")
        prototypes.append(selected.mean(dim=0))
        counts.append(selected.shape[0])
    return (
        torch.stack(prototypes).detach().clone().contiguous(),
        classes.detach().cpu().to(dtype=torch.long),
        torch.tensor(counts, dtype=torch.long),
    )


def power_local_global_coverage_selection(
    *,
    local_embeddings: torch.Tensor,
    global_embeddings: torch.Tensor,
    train_queries: torch.Tensor,
    train_labels: torch.Tensor,
    alpha: float = 0.5,
    coverage_threshold: float = 0.1,
    samples_per_class: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select Eq. (2)--(4) experience nodes by greedy maximum coverage."""

    if (
        local_embeddings.ndim != 2
        or global_embeddings.ndim != 2
        or local_embeddings.shape != global_embeddings.shape
        or not local_embeddings.is_floating_point()
        or not global_embeddings.is_floating_point()
    ):
        raise ValueError("POWER local/global embeddings must be aligned floating matrices.")
    if not torch.isfinite(local_embeddings).all() or not torch.isfinite(
        global_embeddings
    ).all():
        raise ValueError("POWER embeddings must be finite.")
    if train_queries.ndim != 1 or train_queries.dtype != torch.long:
        raise ValueError("POWER train_queries must be one-dimensional int64.")
    if (
        train_labels.ndim != 1
        or train_labels.dtype != torch.long
        or train_labels.shape[0] != train_queries.shape[0]
    ):
        raise ValueError("POWER train_labels must align with train_queries.")
    if train_queries.numel() == 0:
        raise ValueError("POWER coverage selection requires current train nodes.")
    if torch.any(train_queries < 0) or torch.any(
        train_queries >= local_embeddings.shape[0]
    ):
        raise ValueError("POWER train_queries contain a non-local node.")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("POWER alpha must lie in [0, 1].")
    if not 0.0 <= float(coverage_threshold) <= 1.0:
        raise ValueError("POWER coverage_threshold must lie in [0, 1].")
    if (
        isinstance(samples_per_class, bool)
        or not isinstance(samples_per_class, int)
        or samples_per_class <= 0
    ):
        raise ValueError("POWER samples_per_class must be positive.")

    device = local_embeddings.device
    queries = train_queries.to(device)
    labels = train_labels.to(device)
    mixed = float(alpha) * local_embeddings + (
        1.0 - float(alpha)
    ) * global_embeddings
    selected_queries: list[torch.Tensor] = []
    selected_labels: list[int] = []
    for class_id in torch.unique(labels).sort().values.tolist():
        class_queries = queries[labels == int(class_id)]
        class_embeddings = mixed[class_queries]
        distances = torch.cdist(class_embeddings, class_embeddings, p=2)
        radius = distances.mean() * float(coverage_threshold)
        adjacency = distances < radius
        available = torch.ones(
            class_queries.shape[0], dtype=torch.bool, device=device
        )
        for _ in range(min(samples_per_class, class_queries.shape[0])):
            coverage = adjacency[:, available].sum(dim=1)
            coverage[~available] = -1
            index = int(torch.argmax(coverage))
            selected_queries.append(class_queries[index])
            selected_labels.append(int(class_id))
            newly_covered = adjacency[index] & available
            available[newly_covered] = False
            available[index] = False
            if not bool(available.any()):
                break
    return (
        torch.stack(selected_queries).detach().cpu().to(dtype=torch.long),
        torch.tensor(selected_labels, dtype=torch.long),
    )


def power_append_replay(
    *,
    replay_state: Mapping[str, torch.Tensor] | None,
    node_features: torch.Tensor,
    selected_queries: torch.Tensor,
    selected_labels: torch.Tensor,
    ceiling_bytes: int,
) -> dict[str, torch.Tensor]:
    """Append selected feature-only experiences to a client-private buffer."""

    if selected_queries.ndim != 1 or selected_queries.dtype != torch.long:
        raise ValueError("POWER selected_queries must be one-dimensional int64.")
    if (
        selected_labels.ndim != 1
        or selected_labels.dtype != torch.long
        or selected_labels.shape[0] != selected_queries.shape[0]
    ):
        raise ValueError("POWER selected_labels must align with selected_queries.")
    if selected_queries.numel() == 0:
        raise ValueError("POWER replay append requires at least one selected node.")
    if torch.any(selected_queries < 0) or torch.any(
        selected_queries >= node_features.shape[0]
    ):
        raise ValueError("POWER selected replay query is non-local.")
    if (
        isinstance(ceiling_bytes, bool)
        or not isinstance(ceiling_bytes, int)
        or ceiling_bytes <= 0
    ):
        raise ValueError("POWER replay ceiling must be positive.")

    added_features = (
        node_features[selected_queries.to(node_features.device)]
        .detach()
        .cpu()
        .clone()
        .contiguous()
    )
    added_labels = selected_labels.detach().cpu().clone().contiguous()
    if replay_state:
        if set(replay_state) != {"features", "labels"}:
            raise ValueError("POWER replay state fields do not match.")
        old_features = replay_state["features"]
        old_labels = replay_state["labels"]
        if (
            old_features.ndim != 2
            or old_features.shape[1] != added_features.shape[1]
            or old_labels.ndim != 1
            or old_labels.shape[0] != old_features.shape[0]
        ):
            raise ValueError("POWER replay state is malformed.")
        features = torch.cat((old_features.detach().cpu(), added_features), dim=0)
        labels = torch.cat((old_labels.detach().cpu(), added_labels), dim=0)
    else:
        features = added_features
        labels = added_labels
    payload_bytes = (
        features.numel() * features.element_size()
        + labels.numel() * labels.element_size()
    )
    if payload_bytes > ceiling_bytes:
        raise ValueError("POWER replay buffer exceeds its byte ceiling.")
    return {
        "features": features.contiguous(),
        "labels": labels.contiguous(),
    }


def power_replay_payload_bytes(replay_state: Mapping[str, torch.Tensor]) -> int:
    """Return exact raw tensor bytes for a validated replay state."""

    if not replay_state:
        return 0
    if set(replay_state) != {"features", "labels"}:
        raise ValueError("POWER replay state fields do not match.")
    return sum(
        tensor.numel() * tensor.element_size() for tensor in replay_state.values()
    )
