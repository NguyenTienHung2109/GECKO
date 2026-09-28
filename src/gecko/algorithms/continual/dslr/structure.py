from __future__ import annotations

from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_EPOCHS
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_HEADS
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_LAMBDA
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_LR
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_TAU
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_TOP_N
from gecko.algorithms.continual.dslr.records import DSLR_LINK_REDUCTIONS
from gecko.algorithms.continual.dslr.records import _FLOAT_DTYPES

from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_EPOCHS
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_HEADS
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_LAMBDA
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_LR
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_TAU
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_TOP_N

from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_HEADS

import hashlib
import math
import sys
from typing import Dict
from typing import Mapping
from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from gecko.algorithms.topology import TopologyOverlay
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_EPOCHS
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_LAMBDA
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_LR
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_TAU
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_TOP_N

def _logical_positive_universe(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {
        (min(int(source), int(target)), max(int(source), int(target)))
        for source, target in edge_index.t().tolist()
    }


def _sample_unique_ordinals(
    population_size: int,
    count: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample a uniform subset without materializing the population."""

    if count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if count == population_size:
        return torch.arange(population_size, dtype=torch.long, device=device)

    sample_complement = count > population_size // 2
    target = population_size - count if sample_complement else count
    selected = torch.empty(0, dtype=torch.long, device=device)
    while selected.numel() < target:
        remaining = target - int(selected.numel())
        draws = torch.randint(
            population_size,
            (remaining,),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        selected = torch.unique(torch.cat((selected, draws)), sorted=True)

    if not sample_complement:
        return selected
    keep = torch.ones(population_size, dtype=torch.bool, device=device)
    keep[selected] = False
    return torch.arange(population_size, dtype=torch.long, device=device)[keep]


def _complement_ordinals_to_pair_ranks(
    ordinals: torch.Tensor,
    *,
    total_pair_count: int,
    positive_ranks: torch.Tensor,
) -> torch.Tensor:
    """Map complement ordinals to ranks in the full unordered-pair universe."""

    if ordinals.numel() == 0:
        return ordinals.clone()
    device = ordinals.device
    positives = positive_ranks.to(dtype=torch.long, device=device)
    values = ordinals.to(dtype=torch.long, device=device)
    lower = values.clone()
    upper = torch.clamp(values + positives.numel(), max=total_pair_count - 1)
    while bool(torch.any(lower < upper)):
        middle = torch.div(lower + upper, 2, rounding_mode="floor")
        excluded = torch.searchsorted(positives, middle, right=True)
        available_through_middle = middle + 1 - excluded
        advance = available_through_middle <= values
        lower = torch.where(advance, middle + 1, lower)
        upper = torch.where(advance, upper, middle)
    if positives.numel():
        insertion = torch.searchsorted(positives, lower)
        comparable = insertion.clamp_max(positives.numel() - 1)
        if bool(
            torch.any(
                (insertion < positives.numel()) & (positives[comparable] == lower)
            )
        ):
            raise RuntimeError("DSLR complement mapping selected a positive edge.")
    return lower


def sample_strict_local_negative_edges(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    count: int,
    generator: torch.Generator,
    allowed_nodes: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Uniformly sample logical negatives without materializing the complement."""
    from gecko.algorithms.continual.dslr.records import _validate_local_edges
    from gecko.algorithms.continual.dslr.records import _validate_local_indices
    from gecko.algorithms.continual.dslr.records import _validate_nonnegative_int

    edges = _validate_local_edges(edge_index, num_nodes=num_nodes)
    target_device = edges.device if device is None else torch.device(device)
    requested = _validate_nonnegative_int(count, name="count")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a local torch.Generator.")
    if allowed_nodes is None:
        nodes = tuple(range(num_nodes))
    else:
        checked = _validate_local_indices(
            allowed_nodes,
            num_nodes=num_nodes,
            name="allowed_nodes",
            unique=True,
        )
        nodes = tuple(sorted(int(value) for value in checked.tolist()))

    node_count = len(nodes)
    total_pair_count = node_count * (node_count - 1) // 2
    positions = torch.full((num_nodes,), -1, dtype=torch.long, device=target_device)
    node_tensor = torch.tensor(nodes, dtype=torch.long, device=target_device)
    positions[node_tensor] = torch.arange(node_count, dtype=torch.long, device=target_device)
    edge_positions = positions[edges.to(target_device)]
    valid_positive = (
        (edge_positions[0] >= 0)
        & (edge_positions[1] >= 0)
        & (edge_positions[0] != edge_positions[1])
    )
    lower = torch.minimum(
        edge_positions[0, valid_positive], edge_positions[1, valid_positive]
    )
    upper = torch.maximum(
        edge_positions[0, valid_positive], edge_positions[1, valid_positive]
    )
    positive_ranks = torch.unique(
        lower * (2 * node_count - lower - 1) // 2 + upper - lower - 1,
        sorted=True,
    )
    available_count = total_pair_count - int(positive_ranks.numel())
    if requested > available_count:
        raise ValueError(
            f"Requested {requested} DSLR negatives, but only {available_count} "
            "strict-local logical non-edges exist."
        )
    if requested == 0:
        return torch.empty((2, 0), dtype=torch.long, device=target_device)

    complement_ordinals = _sample_unique_ordinals(
        available_count,
        requested,
        generator=generator,
        device=target_device,
    )
    pair_ranks = _complement_ordinals_to_pair_ranks(
        complement_ordinals,
        total_pair_count=total_pair_count,
        positive_ranks=positive_ranks,
    )
    row_starts = torch.tensor(
        [index * (2 * node_count - index - 1) // 2 for index in range(node_count)],
        dtype=torch.long,
        device=target_device,
    )
    source_positions = torch.searchsorted(row_starts, pair_ranks, right=True) - 1
    target_positions = source_positions + 1 + pair_ranks - row_starts[source_positions]
    return torch.stack(
        (node_tensor[source_positions], node_tensor[target_positions]), dim=0
    ).contiguous()


def cosine_link_scores(
    embeddings: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """Return paper link probabilities ``(cos(z_i,z_j)+1)/2``."""
    from gecko.algorithms.continual.dslr.records import _validate_local_edges

    if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_nodes, hidden_size].")
    edges = _validate_local_edges(
        edge_index, num_nodes=embeddings.shape[0], name="score edge_index"
    ).to(embeddings.device)
    if edges.shape[1] == 0:
        return embeddings.new_empty((0,))
    cosine = F.cosine_similarity(
        embeddings[edges[0]], embeddings[edges[1]], dim=-1, eps=1e-12
    )
    return (cosine + 1.0) * 0.5


def link_prediction_loss(
    embeddings: torch.Tensor,
    positive_edge_index: torch.Tensor,
    negative_edge_index: torch.Tensor,
    *,
    reduction: str = "sum",
) -> torch.Tensor:
    """Evaluate equation 6 with an explicit, provenance-visible reduction.

    ``sum`` is the literal paper equation.  ``mean`` is reserved for the
    separately named DSLR-Normalized benchmark adaptation and matches the
    reduction used by the authors' pinned executable snapshot.
    """
    from gecko.algorithms.continual.dslr.records import DSLR_LINK_REDUCTIONS

    if reduction not in DSLR_LINK_REDUCTIONS:
        raise ValueError(f"reduction must be one of {DSLR_LINK_REDUCTIONS}.")

    positive = cosine_link_scores(embeddings, positive_edge_index)
    negative = cosine_link_scores(embeddings, negative_edge_index)
    probabilities = torch.cat((positive, negative))
    if probabilities.numel() == 0:
        return embeddings.sum() * 0.0
    targets = torch.cat((torch.ones_like(positive), torch.zeros_like(negative)))
    epsilon = torch.finfo(probabilities.dtype).eps
    return F.binary_cross_entropy(
        probabilities.clamp(min=epsilon, max=1.0 - epsilon),
        targets,
        reduction=reduction,
    )


def node_supervision_loss(
    logits: torch.Tensor,
    *,
    current_indices: torch.Tensor,
    current_labels: torch.Tensor,
    replay_indices: torch.Tensor | None,
    replay_labels: torch.Tensor | None,
    beta: float,
    current_class_mask: torch.Tensor | None = None,
    replay_task_ids: torch.Tensor | None = None,
    task_class_masks: Mapping[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Evaluate paper equation 7 using current and replay labels only."""
    from gecko.algorithms.continual.dslr.algorithm import _classification_loss
    from gecko.algorithms.continual.dslr.records import _owned_tensor
    from gecko.algorithms.continual.dslr.records import _validate_local_indices
    from gecko.algorithms.continual.dslr.records import _validate_probability

    checked_beta = _validate_probability(beta, name="beta")
    if not torch.is_tensor(logits) or logits.ndim != 2 or not logits.shape[0]:
        raise ValueError("logits must have shape [num_nodes, num_classes].")
    current = _validate_local_indices(
        current_indices,
        num_nodes=logits.shape[0],
        name="current_indices",
    ).to(logits.device)
    labels = _owned_tensor(current_labels, name="current_labels").to(logits.device)
    if (
        labels.dtype != torch.long
        or labels.ndim != 1
        or labels.shape[0] != current.shape[0]
    ):
        raise ValueError("current_labels must be int64 and align with current_indices.")
    if current.numel() == 0:
        raise ValueError("DSLR node supervision requires current training labels.")
    current_loss = _classification_loss(logits[current], labels, current_class_mask)
    if replay_indices is None and replay_labels is None:
        return current_loss
    if replay_indices is None or replay_labels is None:
        raise ValueError("replay_indices and replay_labels must be supplied together.")
    replay = _validate_local_indices(
        replay_indices,
        num_nodes=logits.shape[0],
        name="replay_indices",
    ).to(logits.device)
    old_labels = _owned_tensor(replay_labels, name="replay_labels").to(logits.device)
    if (
        old_labels.dtype != torch.long
        or old_labels.ndim != 1
        or old_labels.shape[0] != replay.shape[0]
        or replay.numel() == 0
    ):
        raise ValueError("replay_labels must align with non-empty replay_indices.")
    if replay_task_ids is None and task_class_masks is None:
        replay_loss = F.cross_entropy(logits[replay], old_labels)
    else:
        if replay_task_ids is None or task_class_masks is None:
            raise ValueError("DSLR task-aware replay IDs and masks must coexist.")
        replay_tasks = _owned_tensor(
            replay_task_ids, name="replay_task_ids"
        ).to(logits.device)
        if replay_tasks.dtype != torch.long or replay_tasks.shape != old_labels.shape:
            raise ValueError("DSLR replay task IDs must align with replay labels.")
        weighted_terms = []
        for task_id in sorted(set(int(value) for value in replay_tasks.tolist())):
            positions = (replay_tasks == task_id).nonzero(as_tuple=False).reshape(-1)
            if task_id not in task_class_masks:
                raise ValueError("DSLR replay task has no class mask.")
            weighted_terms.append(
                (
                    _classification_loss(
                        logits[replay[positions]],
                        old_labels[positions],
                        task_class_masks[task_id],
                    ),
                    int(positions.numel()),
                )
            )
        replay_loss = sum(loss * count for loss, count in weighted_terms) / sum(
            count for _, count in weighted_terms
        )
    return checked_beta * current_loss + (1.0 - checked_beta) * replay_loss


def structure_learning_loss(
    *,
    link_loss: torch.Tensor,
    node_loss: torch.Tensor,
    structure_lambda: float,
) -> torch.Tensor:
    """Combine paper equations 6 and 7 exactly as equation 8."""
    from gecko.algorithms.continual.dslr.records import _validate_probability

    weight = _validate_probability(structure_lambda, name="structure_lambda")
    if not torch.is_tensor(link_loss) or link_loss.ndim != 0:
        raise ValueError("link_loss must be a scalar tensor.")
    if not torch.is_tensor(node_loss) or node_loss.ndim != 0:
        raise ValueError("node_loss must be a scalar tensor.")
    return weight * link_loss + (1.0 - weight) * node_loss


def downstream_classification_loss(
    current_loss: torch.Tensor,
    replay_loss: torch.Tensor | None,
    *,
    beta: float,
) -> torch.Tensor:
    """Combine current and replay objectives according to paper equation 12."""
    from gecko.algorithms.continual.dslr.records import _validate_probability

    weight = _validate_probability(beta, name="beta")
    if not torch.is_tensor(current_loss) or current_loss.ndim != 0:
        raise ValueError("current_loss must be a scalar tensor.")
    if replay_loss is None:
        return current_loss
    if not torch.is_tensor(replay_loss) or replay_loss.ndim != 0:
        raise ValueError("replay_loss must be a scalar tensor.")
    return weight * current_loss + (1.0 - weight) * replay_loss


class DSLRStructureLearner(nn.Module):
    """Private two-layer GAT link predictor used by paper equations 6--8."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        initialization_seed: int,
        heads: int = DSLR_DEFAULT_STRUCTURE_HEADS,
    ) -> None:
        from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_HEADS
        from gecko.algorithms.continual.dslr.records import _validate_nonnegative_int
        from gecko.algorithms.continual.dslr.records import _validate_positive_int
        super().__init__()
        self.input_dim = _validate_positive_int(input_dim, name="input_dim")
        self.hidden_dim = _validate_positive_int(hidden_dim, name="hidden_dim")
        self.num_classes = _validate_positive_int(num_classes, name="num_classes")
        self.heads = _validate_positive_int(heads, name="heads")
        self.initialization_seed = _validate_nonnegative_int(
            initialization_seed, name="initialization_seed"
        )
        if self.initialization_seed >= 2**63:
            raise ValueError("DSLR initialization_seed must fit signed 63 bits.")
        # Match the official two-layer multi-head GAT while isolating its
        # initialization from process-global RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.encoder_one = GATConv(
                self.input_dim,
                self.hidden_dim,
                heads=self.heads,
                concat=True,
            )
            self.encoder_two = GATConv(
                self.hidden_dim * self.heads,
                self.hidden_dim,
                heads=self.heads,
                concat=True,
            )
            self.classifier = nn.Linear(
                self.hidden_dim * self.heads,
                self.num_classes,
                bias=True,
            )

    def encode(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        from gecko.algorithms.continual.dslr.records import _validate_local_edges
        if not torch.is_tensor(features) or features.ndim != 2:
            raise ValueError("features must have shape [num_nodes, input_dim].")
        if features.shape[1] != self.input_dim:
            raise ValueError("DSLR structure-learner feature dimension mismatch.")
        edges = _validate_local_edges(
            edge_index, num_nodes=features.shape[0], name="structure edge_index"
        ).to(features.device)
        hidden = F.relu(self.encoder_one(features, edges))
        return self.encoder_two(hidden, edges)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(F.relu(self.encode(features, edge_index)))

    def state_sha256(self) -> str:
        from gecko.algorithms.continual.dslr.records import _tensor_bytes
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(repr(tuple(value.shape)).encode("ascii"))
            digest.update(_tensor_bytes(value))
        return digest.hexdigest()


def fit_structure_learner(
    learner: DSLRStructureLearner,
    *,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    allowed_nodes: torch.Tensor,
    current_indices: torch.Tensor,
    current_labels: torch.Tensor,
    replay_indices: torch.Tensor,
    replay_labels: torch.Tensor,
    current_class_mask: torch.Tensor | None = None,
    replay_task_ids: torch.Tensor | None = None,
    task_class_masks: Mapping[int, torch.Tensor] | None = None,
    beta: float,
    structure_lambda: float = DSLR_DEFAULT_STRUCTURE_LAMBDA,
    epochs: int = DSLR_DEFAULT_STRUCTURE_EPOCHS,
    learning_rate: float = DSLR_DEFAULT_STRUCTURE_LR,
    negative_ratio: float = 0.5,
    link_reduction: str = "sum",
    rng_seed: int,
) -> Dict[str, float | int]:
    """Optimize a fresh private GAT on only the task-visible strict-local graph."""
    from gecko.algorithms.continual.dslr.records import DSLR_LINK_REDUCTIONS
    from gecko.algorithms.continual.dslr.records import _validate_local_edges
    from gecko.algorithms.continual.dslr.records import _validate_local_indices
    from gecko.algorithms.continual.dslr.records import _validate_nonnegative_int
    from gecko.algorithms.continual.dslr.records import _validate_positive_float
    from gecko.algorithms.continual.dslr.records import _validate_positive_int
    from gecko.algorithms.continual.dslr.records import _validate_probability

    if not isinstance(learner, DSLRStructureLearner):
        raise TypeError("learner must be a DSLRStructureLearner.")
    features = node_features
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError("node_features must have shape [num_nodes, feature_dim].")
    edges = _validate_local_edges(edge_index, num_nodes=features.shape[0])
    allowed = _validate_local_indices(
        allowed_nodes,
        num_nodes=features.shape[0],
        name="allowed_nodes",
        unique=True,
    )
    if allowed.numel() == 0:
        raise ValueError("DSLR structure learning requires visible local nodes.")
    allowed_set = set(int(value) for value in allowed.tolist())
    current = _validate_local_indices(
        current_indices,
        num_nodes=features.shape[0],
        name="current_indices",
        unique=True,
    )
    replay = _validate_local_indices(
        replay_indices,
        num_nodes=features.shape[0],
        name="replay_indices",
        unique=True,
    )
    supervised = set(int(value) for value in current.tolist()) | set(
        int(value) for value in replay.tolist()
    )
    if not supervised <= allowed_set:
        raise ValueError(
            "DSLR supervision contains a node outside the visible context."
        )
    edge_pairs = [
        (int(source), int(target))
        for source, target in edges.detach().cpu().t().tolist()
        if int(source) in allowed_set and int(target) in allowed_set
    ]
    safe_edges = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    if safe_edges.shape[1] != edges.shape[1]:
        raise ValueError(
            "DSLR positive topology contains a node outside the visible context."
        )
    rounds = _validate_positive_int(epochs, name="epochs")
    lr = _validate_positive_float(learning_rate, name="learning_rate")
    ratio = _validate_probability(negative_ratio, name="negative_ratio")
    if link_reduction not in DSLR_LINK_REDUCTIONS:
        raise ValueError(
            f"link_reduction must be one of {DSLR_LINK_REDUCTIONS}."
        )
    seed = _validate_nonnegative_int(rng_seed, name="rng_seed")
    if seed >= 2**63:
        raise ValueError("DSLR RNG seed must fit signed 63 bits.")
    device = next(learner.parameters()).device
    features = features.to(device)
    safe_edges = safe_edges.to(device)
    logical_positive = sorted(_logical_positive_universe(safe_edges.detach().cpu()))
    positive_edges = (
        torch.tensor(logical_positive, dtype=torch.long, device=device).t().contiguous()
        if logical_positive
        else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    nonself_positive = {pair for pair in logical_positive if pair[0] != pair[1]}
    possible_negative_count = allowed.numel() * (allowed.numel() - 1) // 2 - len(
        nonself_positive
    )
    desired_negative_count = int(math.ceil(len(logical_positive) * ratio))
    negative_count = min(desired_negative_count, max(0, possible_negative_count))
    optimizer = torch.optim.Adam(learner.parameters(), lr=lr)
    final_link = 0.0
    final_node = 0.0
    final_total = 0.0
    for epoch in range(rounds):
        generator = torch.Generator(device=device)
        generator.manual_seed((seed + epoch) % (2**63))
        negative_edges = sample_strict_local_negative_edges(
            safe_edges,
            num_nodes=features.shape[0],
            count=negative_count,
            generator=generator,
            allowed_nodes=allowed,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        embeddings = learner.encode(features, safe_edges)
        link = link_prediction_loss(
            embeddings,
            positive_edges,
            negative_edges,
            reduction=link_reduction,
        )
        node = node_supervision_loss(
            learner.classifier(F.relu(embeddings)),
            current_indices=current,
            current_labels=current_labels,
            replay_indices=replay,
            replay_labels=replay_labels,
            beta=beta,
            current_class_mask=current_class_mask,
            replay_task_ids=replay_task_ids,
            task_class_masks=task_class_masks,
        )
        total = structure_learning_loss(
            link_loss=link,
            node_loss=node,
            structure_lambda=structure_lambda,
        )
        if not bool(torch.isfinite(total)):
            raise RuntimeError("DSLR structure-learning objective became non-finite.")
        total.backward()
        optimizer.step()
        final_link = float(link.detach().cpu())
        final_node = float(node.detach().cpu())
        final_total = float(total.detach().cpu())
    pair_count = len(logical_positive) + negative_count
    link_per_pair = final_link / pair_count if pair_count else 0.0
    link_contribution = float(structure_lambda) * final_link
    node_contribution = (1.0 - float(structure_lambda)) * final_node
    dominance_ratio = (
        link_contribution / node_contribution
        if node_contribution > 0.0
        else sys.float_info.max if link_contribution > 0.0 else 0.0
    )
    return {
        "structure_epochs": rounds,
        "structure_visible_nodes": int(allowed.numel()),
        "positive_logical_edges": len(logical_positive),
        "negative_logical_edges_per_epoch": negative_count,
        "link_loss_pair_count": pair_count,
        "link_loss_reduction": link_reduction,
        "final_link_loss": final_link,
        "final_link_loss_per_pair": link_per_pair,
        "final_node_loss": final_node,
        "final_structure_loss": final_total,
        "weighted_link_contribution": link_contribution,
        "weighted_node_contribution": node_contribution,
        "weighted_link_to_node_ratio": dominance_ratio,
    }


def build_dslr_overlay(
    *,
    client_id: int,
    global_task_id: int,
    base_edge_index: torch.Tensor,
    allowed_nodes: torch.Tensor,
    structure_embeddings: torch.Tensor,
    replay_snapshots: Sequence[DSLRReplaySnapshot],
    top_n: int = DSLR_DEFAULT_TOP_N,
    tau: float = DSLR_DEFAULT_TAU,
    undirected: bool = True,
) -> TopologyOverlay:
    """Apply paper equations 9--10 over visible stored equation-11 candidates."""
    from gecko.algorithms.continual.dslr.records import DSLRReplaySnapshot
    from gecko.algorithms.continual.dslr.records import _FLOAT_DTYPES
    from gecko.algorithms.continual.dslr.records import _owned_tensor
    from gecko.algorithms.continual.dslr.records import _validate_local_edges
    from gecko.algorithms.continual.dslr.records import _validate_local_indices
    from gecko.algorithms.continual.dslr.records import _validate_nonnegative_int
    from gecko.algorithms.continual.dslr.records import _validate_positive_int
    from gecko.algorithms.continual.dslr.records import _validate_probability

    client = _validate_nonnegative_int(client_id, name="client_id")
    task = _validate_nonnegative_int(global_task_id, name="global_task_id")
    count = _validate_positive_int(top_n, name="top_n")
    threshold = _validate_probability(tau, name="tau")
    values = _owned_tensor(structure_embeddings, name="structure_embeddings")
    if values.dtype not in _FLOAT_DTYPES or values.ndim != 2 or not values.shape[0]:
        raise ValueError("structure_embeddings must be non-empty [nodes, dim].")
    allowed = _validate_local_indices(
        allowed_nodes,
        num_nodes=values.shape[0],
        name="allowed_nodes",
        unique=True,
    )
    allowed_set = set(int(value) for value in allowed.tolist())
    edges = _validate_local_edges(
        base_edge_index, num_nodes=values.shape[0], name="base_edge_index"
    )
    base_arcs = {(int(source), int(target)) for source, target in edges.t().tolist()}
    if any(
        source not in allowed_set or target not in allowed_set
        for source, target in base_arcs
    ):
        raise ValueError(
            "DSLR base topology contains a node outside the visible context."
        )
    if undirected and any(
        (target, source) not in base_arcs for source, target in base_arcs
    ):
        raise ValueError("DSLR undirected inference requires reverse base arcs.")
    added: set[tuple[int, int]] = set()
    deleted: set[tuple[int, int]] = set()
    roots: set[int] = set()
    for snapshot in replay_snapshots:
        if not isinstance(snapshot, DSLRReplaySnapshot):
            raise TypeError("replay_snapshots must contain DSLRReplaySnapshot values.")
        if snapshot.client_id != client:
            raise ValueError("DSLR replay snapshot belongs to another client.")
        root = snapshot.source_local_index
        if root not in allowed_set:
            raise ValueError("DSLR replay root is outside the visible context.")
        if root in roots:
            raise ValueError("DSLR replay snapshots contain a duplicate root.")
        roots.add(root)
        candidates = _validate_local_indices(
            snapshot.candidate_local_indices,
            num_nodes=values.shape[0],
            name="stored candidate_local_indices",
            unique=True,
        )
        if any(int(value) not in allowed_set for value in candidates.tolist()):
            raise ValueError("DSLR stored candidate is outside the visible context.")
        nonneighbors = torch.tensor(
            [
                int(endpoint)
                for endpoint in candidates.tolist()
                if int(endpoint) != root
                and (root, int(endpoint)) not in base_arcs
                and (int(endpoint), root) not in base_arcs
            ],
            dtype=torch.long,
        )
        if nonneighbors.numel():
            candidate_edges = torch.stack(
                (
                    torch.full_like(nonneighbors, root),
                    nonneighbors,
                )
            )
            candidate_scores = cosine_link_scores(values, candidate_edges)
            ranking = sorted(
                zip(nonneighbors.tolist(), candidate_scores.tolist()),
                key=lambda item: (-float(item[1]), int(item[0])),
            )[:count]
            for endpoint, _ in ranking:
                arc = (root, int(endpoint))
                added.add(arc)
                if undirected:
                    added.add((int(endpoint), root))
        logical_neighbors = sorted(
            {
                target
                for source, target in base_arcs
                if source == root and target != root
            }
            | {
                source
                for source, target in base_arcs
                if undirected and target == root and source != root
            }
        )
        if logical_neighbors:
            neighbor_tensor = torch.tensor(logical_neighbors, dtype=torch.long)
            neighbor_edges = torch.stack(
                (torch.full_like(neighbor_tensor, root), neighbor_tensor)
            )
            neighbor_scores = cosine_link_scores(values, neighbor_edges)
            for endpoint, score in zip(logical_neighbors, neighbor_scores.tolist()):
                if float(score) <= threshold:
                    deleted.add((root, endpoint))
                    if undirected:
                        deleted.add((endpoint, root))
    added_edges = (
        torch.tensor(sorted(added), dtype=torch.long).t().contiguous()
        if added
        else torch.empty((2, 0), dtype=torch.long)
    )
    deleted_edges = (
        torch.tensor(sorted(deleted), dtype=torch.long).t().contiguous()
        if deleted
        else torch.empty((2, 0), dtype=torch.long)
    )
    return TopologyOverlay(
        client_id=client,
        global_task_id=task,
        method_name="DSLR",
        num_nodes=values.shape[0],
        base_edge_index=edges,
        added_edge_index=added_edges,
        deleted_edge_index=deleted_edges,
        undirected=undirected,
    )


