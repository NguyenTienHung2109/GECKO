from __future__ import annotations

from gecko.algorithms.continual.dslr.records import _FLOAT_DTYPES

from fractions import Fraction
from typing import Dict
from typing import Mapping
from typing import Tuple
import torch

def coverage_sets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    radius: float,
) -> Dict[int, Tuple[int, ...]]:
    """Compute paper equation 3 independently inside every class.

    The class radius is ``r`` times the mean of its complete pairwise-distance
    matrix, matching the paper's official implementation.  The strict ``<``
    relation in equation 3 is preserved.
    """
    from gecko.algorithms.continual.dslr.records import _FLOAT_DTYPES
    from gecko.algorithms.continual.dslr.records import _owned_tensor
    from gecko.algorithms.continual.dslr.records import _validate_local_indices
    from gecko.algorithms.continual.dslr.records import _validate_positive_float

    checked_radius = _validate_positive_float(radius, name="radius")
    values = _owned_tensor(embeddings, name="embeddings")
    if values.dtype not in _FLOAT_DTYPES or values.ndim != 2 or not values.shape[0]:
        raise ValueError("embeddings must be a non-empty floating [nodes, dim] tensor.")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("embeddings must be finite.")
    classes = _owned_tensor(labels, name="labels")
    if (
        classes.dtype != torch.long
        or classes.ndim != 1
        or classes.shape[0] != values.shape[0]
    ):
        raise ValueError("labels must be int64 and align with embeddings.")
    indices = _validate_local_indices(
        candidate_indices,
        num_nodes=values.shape[0],
        name="candidate_indices",
        unique=True,
    )
    output: Dict[int, Tuple[int, ...]] = {}
    candidate_values = indices.tolist()
    candidate_labels = classes[indices].tolist()
    for class_id in sorted(set(int(value) for value in candidate_labels)):
        members = [
            int(index)
            for index, label in zip(candidate_values, candidate_labels)
            if int(label) == class_id
        ]
        member_tensor = torch.tensor(members, dtype=torch.long)
        distances = torch.cdist(
            values[member_tensor].double(), values[member_tensor].double()
        )
        covered = distances < checked_radius * distances.mean()
        for row, source in enumerate(members):
            output[source] = tuple(
                member_tensor[covered[row]].tolist()
            )
    return output


def allocate_class_quotas(
    class_counts: Mapping[int, int], *, requested_count: int
) -> Dict[int, int]:
    """Allocate a proportional total with one node per seen class.

    Remaining slots are assigned by largest proportional deficit, with class
    ID as the deterministic tie breaker.  A target unable to retain one node
    per seen class fails rather than silently dropping a class.
    """
    from gecko.algorithms.continual.dslr.records import _validate_nonnegative_int
    from gecko.algorithms.continual.dslr.records import _validate_positive_int

    requested = _validate_positive_int(requested_count, name="requested_count")
    if not isinstance(class_counts, Mapping) or not class_counts:
        raise ValueError("class_counts must contain at least one seen class.")
    counts: Dict[int, int] = {}
    for raw_class, raw_count in class_counts.items():
        class_id = _validate_nonnegative_int(raw_class, name="class_id")
        count = _validate_positive_int(raw_count, name=f"class_counts[{class_id}]")
        counts[class_id] = count
    if requested < len(counts):
        raise ValueError(
            "DSLR replay target cannot retain one snapshot per seen class; "
            "increase cumulative training nodes or replay_fraction."
        )
    if requested > sum(counts.values()):
        raise ValueError("DSLR replay target exceeds the available labeled nodes.")
    total = sum(counts.values())
    quotas = {class_id: 1 for class_id in counts}
    for _ in range(requested - len(counts)):
        eligible = [
            class_id for class_id in counts if quotas[class_id] < counts[class_id]
        ]
        if not eligible:
            raise RuntimeError("DSLR proportional allocation exhausted candidates.")
        selected = min(
            eligible,
            key=lambda class_id: (
                Fraction(quotas[class_id], 1)
                - Fraction(requested * counts[class_id], total),
                class_id,
            ),
        )
        quotas[selected] += 1
    return dict(sorted(quotas.items()))


def greedy_coverage_selection(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    class_quotas: Mapping[int, int],
    radius: float,
) -> Tuple[int, ...]:
    """Run paper Algorithm 2 with covered candidates removed per class."""
    from gecko.algorithms.continual.dslr.records import _owned_tensor
    from gecko.algorithms.continual.dslr.records import _validate_local_indices
    from gecko.algorithms.continual.dslr.records import _validate_nonnegative_int
    from gecko.algorithms.continual.dslr.records import _validate_positive_float
    from gecko.algorithms.continual.dslr.records import _validate_positive_int

    values = _owned_tensor(embeddings, name="embeddings")
    classes = _owned_tensor(labels, name="labels")
    indices = _validate_local_indices(
        candidate_indices,
        num_nodes=values.shape[0],
        name="candidate_indices",
        unique=True,
    )
    selected: list[int] = []
    checked_radius = _validate_positive_float(radius, name="radius")
    candidate_values = indices.tolist()
    candidate_labels = classes[indices].tolist()
    for raw_class, raw_quota in sorted(class_quotas.items()):
        class_id = _validate_nonnegative_int(raw_class, name="class_id")
        quota = _validate_positive_int(raw_quota, name=f"class_quotas[{class_id}]")
        members = [
            int(index)
            for index, label in zip(candidate_values, candidate_labels)
            if int(label) == class_id
        ]
        if quota > len(members):
            raise ValueError(f"DSLR class {class_id} quota exceeds its candidates.")
        member_tensor = torch.tensor(members, dtype=torch.long)
        distances = torch.cdist(
            values[member_tensor].double(), values[member_tensor].double()
        )
        covers = distances < checked_radius * distances.mean()
        remaining = torch.ones(len(members), dtype=torch.bool)
        for _ in range(quota):
            if not bool(remaining.any()):
                raise ValueError(
                    f"DSLR class {class_id} coverage exhausted before its quota; "
                    "the configuration cannot achieve the declared replay target."
                )
            counts = (covers & remaining.unsqueeze(0)).sum(dim=1)
            counts = counts + ((~covers.diagonal()) & remaining).long()
            counts[~remaining] = -1
            best_position = int(torch.argmax(counts))
            selected.append(members[best_position])
            remaining &= ~covers[best_position]
            remaining[best_position] = False
    return tuple(selected)


def mean_feature_selection(
    features: torch.Tensor,
    labels: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    class_quotas: Mapping[int, int],
) -> Tuple[int, ...]:
    """Select the paper MF controls closest to each class feature mean."""
    from gecko.algorithms.continual.dslr.records import _FLOAT_DTYPES
    from gecko.algorithms.continual.dslr.records import _owned_tensor
    from gecko.algorithms.continual.dslr.records import _validate_local_indices
    from gecko.algorithms.continual.dslr.records import _validate_nonnegative_int
    from gecko.algorithms.continual.dslr.records import _validate_positive_int

    values = _owned_tensor(features, name="features")
    if values.dtype not in _FLOAT_DTYPES or values.ndim != 2:
        raise ValueError("features must be a floating [nodes, dim] tensor.")
    classes = _owned_tensor(labels, name="labels")
    if (
        classes.dtype != torch.long
        or classes.ndim != 1
        or classes.shape[0] != values.shape[0]
    ):
        raise ValueError("labels must be int64 and align with features.")
    indices = _validate_local_indices(
        candidate_indices,
        num_nodes=values.shape[0],
        name="candidate_indices",
        unique=True,
    )
    selected: list[int] = []
    for raw_class, raw_quota in sorted(class_quotas.items()):
        class_id = _validate_nonnegative_int(raw_class, name="class_id")
        quota = _validate_positive_int(raw_quota, name=f"class_quotas[{class_id}]")
        members = sorted(
            int(index) for index in indices.tolist() if int(classes[index]) == class_id
        )
        if quota > len(members):
            raise ValueError(f"DSLR class {class_id} quota exceeds its candidates.")
        member_tensor = torch.tensor(members, dtype=torch.long)
        center = values[member_tensor].double().mean(dim=0)
        distances = torch.linalg.vector_norm(
            values[member_tensor].double() - center, dim=1
        )
        ranked = sorted(
            zip(members, distances.tolist()),
            key=lambda item: (float(item[1]), int(item[0])),
        )
        selected.extend(node for node, _ in ranked[:quota])
    return tuple(selected)


def select_prior_candidates(
    prior_embeddings: torch.Tensor,
    *,
    replay_index: int,
    available_indices: torch.Tensor,
    candidate_k: int,
) -> torch.Tensor:
    """Select equation-11 top-K nearest prior-task endpoints."""
    from gecko.algorithms.continual.dslr.records import _FLOAT_DTYPES
    from gecko.algorithms.continual.dslr.records import _owned_tensor
    from gecko.algorithms.continual.dslr.records import _validate_local_indices
    from gecko.algorithms.continual.dslr.records import _validate_nonnegative_int
    from gecko.algorithms.continual.dslr.records import _validate_positive_int

    values = _owned_tensor(prior_embeddings, name="prior_embeddings")
    if values.dtype not in _FLOAT_DTYPES or values.ndim != 2 or not values.shape[0]:
        raise ValueError("prior_embeddings must be non-empty [nodes, dim].")
    root = _validate_nonnegative_int(replay_index, name="replay_index")
    if root >= values.shape[0]:
        raise ValueError("replay_index is outside the strict-local graph.")
    indices = _validate_local_indices(
        available_indices,
        num_nodes=values.shape[0],
        name="available_indices",
        unique=True,
    )
    k = _validate_positive_int(candidate_k, name="candidate_k")
    candidates = torch.tensor(
        sorted(int(value) for value in indices.tolist() if int(value) != root),
        dtype=torch.long,
    )
    if not candidates.numel():
        return candidates
    distances = torch.linalg.vector_norm(
        values[candidates].double() - values[root].double(), dim=1
    )
    order = torch.argsort(distances, stable=True)
    return candidates[order[:k]].clone()


