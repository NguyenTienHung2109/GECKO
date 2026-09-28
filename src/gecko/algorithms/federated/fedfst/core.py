"""Paper-equation primitives for FedFST.

This module is a clean-room implementation of equations 1--13 in the FedFST
paper.  It deliberately does not import the authors' executable repository:
that snapshot has no license and its HLST loss does not implement equation 12.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


PAPER_DOI = "10.1145/3770855.3817730"
REFERENCE_CODE_COMMIT = "d83fac0eff2ebf764d5574997a5fec0abf9cc2c7"
FEDFST_IMPLEMENTATION_VERSION = "uefa-fedfst-client-order-adaptation-v10"
# Equation 7 does not specify how to average undefined Rayleigh quotients.
# The frozen reference snapshot treats denominators at or below 1e-8 as zero.
RAYLEIGH_DENOMINATOR_EPSILON = 1e-8


@dataclass(frozen=True)
class FedFSTParameters:
    """Frozen resolution profile for paper details omitted from the text."""

    noise_dim: int = 128
    generator_dropout: float = 0.5
    generator_rounds: int = 2
    generator_epochs: int = 200
    generator_learning_rate: float = 0.005
    client_learning_rate: float = 0.01
    client_weight_decay: float = 0.0
    client_nodes_per_class: int = 200
    server_nodes_per_class: int = 300
    generated_edges_per_node: int = 2
    class_il_output_training_policy: str = "paper_full"
    server_initial_edge_policy: str = "target_scaled_capped"
    server_max_edges_per_node: int = 14
    edge_reduction_ratio: float = 0.05
    topology_tolerance: float = 0.05
    topology_max_iterations: int = 60
    sampled_feature_fraction: float = 0.2
    lambda_kl: float = 100.0
    smoothing_hops: int = 2
    lambda_low: float = 1.0
    distillation_epochs: int = 200
    distillation_early_stop_policy: str = "none"
    distillation_validation_checkpoints: tuple[int, ...] = (
        0,
        1,
        2,
        5,
        10,
        20,
        50,
        75,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
    )
    distillation_learning_rate: float = 0.002
    method_seed: int = 24

    def __post_init__(self) -> None:
        positive_integers = {
            "noise_dim": self.noise_dim,
            "generator_rounds": self.generator_rounds,
            "generator_epochs": self.generator_epochs,
            "client_nodes_per_class": self.client_nodes_per_class,
            "server_nodes_per_class": self.server_nodes_per_class,
            "generated_edges_per_node": self.generated_edges_per_node,
            "server_max_edges_per_node": self.server_max_edges_per_node,
            "topology_max_iterations": self.topology_max_iterations,
            "smoothing_hops": self.smoothing_hops,
            "distillation_epochs": self.distillation_epochs,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        checkpoints = self.distillation_validation_checkpoints
        if (
            not isinstance(checkpoints, tuple)
            or not checkpoints
            or any(isinstance(value, bool) or not isinstance(value, int) for value in checkpoints)
            or checkpoints[0] != 0
            or checkpoints[-1] != self.distillation_epochs
            or any(left >= right for left, right in zip(checkpoints, checkpoints[1:]))
        ):
            raise ValueError(
                "distillation_validation_checkpoints must be a strictly increasing "
                "tuple beginning at 0 and ending at distillation_epochs."
            )
        if (
            isinstance(self.method_seed, bool)
            or not isinstance(self.method_seed, int)
            or self.method_seed < 0
            or self.method_seed >= 2**63
        ):
            raise ValueError("method_seed must be an integer in [0, 2**63).")
        if self.server_initial_edge_policy not in {
            "paper_fixed",
            "target_scaled",
            "target_scaled_capped",
        }:
            raise ValueError(
                "server_initial_edge_policy must be 'paper_fixed', "
                "'target_scaled', or 'target_scaled_capped'."
            )
        if self.class_il_output_training_policy not in {
            "paper_full",
            "benchmark_seen",
        }:
            raise ValueError(
                "class_il_output_training_policy must be 'paper_full' or "
                "'benchmark_seen'."
            )
        if self.distillation_early_stop_policy not in {
            "none",
            "author_balance_crossing",
        }:
            raise ValueError(
                "distillation_early_stop_policy must be 'none' or "
                "'author_balance_crossing'."
            )
        if self.server_max_edges_per_node < self.generated_edges_per_node:
            raise ValueError(
                "server_max_edges_per_node must be at least "
                "generated_edges_per_node."
            )
        for name, value in {
            "generator_learning_rate": self.generator_learning_rate,
            "client_learning_rate": self.client_learning_rate,
            "distillation_learning_rate": self.distillation_learning_rate,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        if (
            isinstance(self.client_weight_decay, bool)
            or not isinstance(self.client_weight_decay, (int, float))
            or not math.isfinite(float(self.client_weight_decay))
            or float(self.client_weight_decay) < 0.0
        ):
            raise ValueError("client_weight_decay must be finite and non-negative.")
        for name, value in {
            "edge_reduction_ratio": self.edge_reduction_ratio,
            "sampled_feature_fraction": self.sampled_feature_fraction,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 1.0
            ):
                raise ValueError(f"{name} must lie in (0, 1].")
        if (
            isinstance(self.generator_dropout, bool)
            or not isinstance(self.generator_dropout, (int, float))
            or not math.isfinite(float(self.generator_dropout))
            or not 0.0 <= float(self.generator_dropout) < 1.0
        ):
            raise ValueError("generator_dropout must lie in [0, 1).")
        for name, value in {
            "topology_tolerance": self.topology_tolerance,
            "lambda_kl": self.lambda_kl,
            "lambda_low": self.lambda_low,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "FedFSTParameters":
        """Construct from an exact, already schema-validated mapping."""

        expected = set(cls.__dataclass_fields__)
        if set(values) != expected:
            raise ValueError(
                "FedFST parameter fields differ: "
                f"missing={sorted(expected - set(values))}, "
                f"unexpected={sorted(set(values) - expected)}."
            )
        return cls(**dict(values))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, int | float | str | tuple[int, ...]]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.to_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(repr(value).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()


def derive_seed(base_seed: int, *coordinates: int) -> int:
    """Derive an independent deterministic PyTorch seed without global RNGs."""

    digest = hashlib.blake2b(digest_size=8, person=b"uefa-fst")
    for value in (base_seed, *coordinates):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < -(2**63)
            or value >= 2**63
        ):
            raise ValueError("FedFST seed coordinates must be signed int64 values.")
        digest.update(int(value).to_bytes(8, byteorder="little", signed=True))
    return int.from_bytes(digest.digest(), byteorder="little") % (2**63 - 1)


def local_generator(
    base_seed: int,
    *coordinates: int,
    device: torch.device | str = "cpu",
) -> torch.Generator:
    """Return an owned deterministic RNG on the computation device."""

    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(derive_seed(base_seed, *coordinates))
    return generator


class ConditionalFeatureGenerator(nn.Module):
    """Class-conditional MLP used by HHKR in the reference-code profile."""

    def __init__(
        self,
        *,
        feature_dim: int,
        class_ids: Sequence[int],
        noise_dim: int,
        dropout: float,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        normalized_classes = tuple(sorted({int(value) for value in class_ids}))
        if (
            not normalized_classes
            or len(normalized_classes) != len(tuple(class_ids))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in class_ids
            )
        ):
            raise ValueError(
                "Generator class IDs must be unique non-negative integers."
            )
        if feature_dim <= 0 or noise_dim <= 0:
            raise ValueError("Generator dimensions must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Generator dropout must lie in [0, 1).")
        self.feature_dim = int(feature_dim)
        self.class_ids = normalized_classes
        self.num_classes = len(normalized_classes)
        self._condition_by_class = {
            class_id: index for index, class_id in enumerate(normalized_classes)
        }
        lookup = torch.full(
            (max(normalized_classes) + 1,), -1, dtype=torch.long
        )
        lookup[torch.tensor(normalized_classes, dtype=torch.long)] = torch.arange(
            len(normalized_classes), dtype=torch.long
        )
        # This is deterministic derived metadata, not generator state to upload.
        self.register_buffer("_class_lookup", lookup, persistent=False)
        self.noise_dim = int(noise_dim)
        self.dropout = float(dropout)
        # Module constructors consume the global CPU RNG.  Seed only a local
        # CPU generator, copy its state inside a CPU-only fork, and let the fork
        # restore the caller's CPU state.  ``torch.manual_seed`` is deliberately
        # avoided because it also mutates every CUDA generator even when CUDA
        # tensors are not involved.
        initialization_generator = torch.Generator(device="cpu")
        initialization_generator.manual_seed(int(initialization_seed))
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initialization_generator.get_state())
            self.label_embedding = nn.Embedding(self.num_classes, self.num_classes)
            dimensions = (noise_dim + self.num_classes, 64, 128, 256)
            self.hidden_layers = nn.ModuleList(
                nn.Linear(dimensions[index], dimensions[index + 1])
                for index in range(len(dimensions) - 1)
            )
            self.output_layer = nn.Linear(256, feature_dim)

    @staticmethod
    def _dropout_with_generator(
        value: torch.Tensor, probability: float, generator: torch.Generator
    ) -> torch.Tensor:
        if probability == 0.0:
            return value
        keep = torch.rand(
            tuple(value.shape),
            generator=generator,
            device=value.device,
            dtype=value.dtype,
        ) >= probability
        return value * keep.to(dtype=value.dtype) / (1.0 - probability)

    def forward(
        self,
        noise: torch.Tensor,
        labels: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.noise_dim:
            raise ValueError("Generator noise has the wrong shape.")
        if labels.dtype != torch.long or labels.ndim != 1:
            raise ValueError("Generator labels must be one-dimensional int64 values.")
        if labels.shape[0] != noise.shape[0]:
            raise ValueError("Generator labels and noise must align.")
        if labels.numel() and (
            int(labels.min()) < 0 or int(labels.max()) >= self._class_lookup.numel()
        ):
            raise ValueError(
                "Generator label is outside the historical class vocabulary."
            )
        conditions = self._class_lookup[labels]
        if bool((conditions < 0).any()):
            raise ValueError(
                "Generator label is outside the historical class vocabulary."
            )
        hidden = torch.cat((self.label_embedding(conditions), noise), dim=1)
        for layer in self.hidden_layers:
            hidden = torch.tanh(layer(hidden))
            if self.training and self.dropout:
                if generator is None:
                    raise ValueError("Train-mode generator dropout requires a local RNG.")
                hidden = self._dropout_with_generator(
                    hidden, self.dropout, generator
                )
        return self.output_layer(hidden)


def balanced_labels(
    classes: Sequence[int], nodes_per_class: int, *, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Return balanced labels using actual immutable class IDs."""

    normalized = tuple(sorted({int(value) for value in classes}))
    if not normalized or nodes_per_class <= 0:
        raise ValueError("Balanced generation requires classes and positive capacity.")
    return torch.tensor(normalized, dtype=torch.long, device=device).repeat_interleave(
        int(nodes_per_class)
    )


def generate_balanced_features(
    model: ConditionalFeatureGenerator,
    classes: Sequence[int],
    nodes_per_class: int,
    *,
    generator: torch.Generator,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = balanced_labels(classes, nodes_per_class, device=device)
    if not set(int(value) for value in classes).issubset(model.class_ids):
        raise ValueError("Generated classes are outside the historical vocabulary.")
    noise = torch.randn(
        (int(labels.shape[0]), model.noise_dim),
        generator=generator,
        device=device,
    )
    features = model(noise, labels, generator=generator)
    return F.normalize(features, p=2, dim=1), labels


def _validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> None:
    if (
        not torch.is_tensor(edge_index)
        or edge_index.dtype != torch.long
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError("edge_index must have int64 shape [2, num_edges].")
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError("edge_index contains an out-of-range endpoint.")


def logical_undirected_edges(
    edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Coalesce reverse arcs into sorted, loop-free logical edges."""

    _validate_edge_index(edge_index, num_nodes)
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    source, target = edge_index
    lower = torch.minimum(source, target)
    upper = torch.maximum(source, target)
    keep = lower != upper
    if not bool(keep.any()):
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    keys = lower[keep] * max(1, num_nodes) + upper[keep]
    keys = torch.unique(keys, sorted=True)
    return torch.stack((keys // max(1, num_nodes), keys % max(1, num_nodes)))


def canonical_undirected_edges(
    edge_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Return exactly two arcs for every loop-free logical edge."""

    logical = logical_undirected_edges(edge_index, num_nodes)
    if logical.numel() == 0:
        return logical
    arcs = torch.cat((logical, logical.flip(0)), dim=1)
    keys = arcs[0] * max(1, num_nodes) + arcs[1]
    return arcs[:, torch.argsort(keys, stable=True)].contiguous()


def add_self_loops_once(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Coalesce arcs and add exactly one self-loop per node for equations 9--12."""

    _validate_edge_index(edge_index, num_nodes)
    nodes = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
    loops = torch.stack((nodes, nodes))
    combined = torch.cat((edge_index, loops), dim=1)
    if combined.numel() == 0:
        return combined
    keys = torch.unique(
        combined[0] * max(1, num_nodes) + combined[1], sorted=True
    )
    return torch.stack((keys // max(1, num_nodes), keys % max(1, num_nodes)))


def random_undirected_edges(
    num_nodes: int,
    *,
    directed_edges_per_node: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a simple undirected graph with the reference edge density."""

    if num_nodes < 0 or directed_edges_per_node <= 0:
        raise ValueError("Random graph dimensions must be non-negative/positive.")
    resolved_device = torch.device(device)
    if torch.device(generator.device) != resolved_device:
        raise ValueError("Random-graph RNG and output device must match.")
    maximum = num_nodes * (num_nodes - 1) // 2
    target_count = min(
        maximum,
        int(math.ceil(num_nodes * directed_edges_per_node / 2.0)),
    )
    if target_count == 0:
        return torch.empty((2, 0), dtype=torch.long, device=resolved_device)
    keys = torch.empty((0,), dtype=torch.long, device=resolved_device)
    while int(keys.numel()) < target_count:
        remaining = target_count - int(keys.numel())
        candidates = torch.randint(
            0,
            num_nodes,
            (2, max(8, remaining * 3)),
            generator=generator,
            device=resolved_device,
        )
        lower = torch.minimum(candidates[0], candidates[1])
        upper = torch.maximum(candidates[0], candidates[1])
        keep = lower != upper
        candidate_keys = lower[keep] * max(1, num_nodes) + upper[keep]
        keys = torch.unique(torch.cat((keys, candidate_keys)), sorted=False)
    if int(keys.numel()) > target_count:
        selection = torch.randperm(
            int(keys.numel()), generator=generator, device=resolved_device
        )[:target_count]
        keys = keys.index_select(0, selection)
    keys = keys.sort().values
    logical = torch.stack(
        (keys // max(1, num_nodes), keys % max(1, num_nodes))
    )
    return canonical_undirected_edges(logical, num_nodes)


def random_block_diagonal_edges(
    labels: torch.Tensor,
    task_class_masks: Mapping[int, torch.Tensor],
    *,
    directed_edges_per_node: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create one density-preserving random graph inside each task head.

    Generating a global random graph and then deleting cross-task arcs makes
    Task-IL replay progressively sparser as the number of historical heads
    grows.  Sampling each immutable task block separately preserves the
    requested per-node density without ever constructing a cross-task edge.
    """

    if labels.dtype != torch.long or labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("Block-diagonal generation requires int64 node labels.")
    first_mask = next(iter(task_class_masks.values()), None)
    if not torch.is_tensor(first_mask) or first_mask.ndim != 1:
        raise ValueError("Block-diagonal generation requires task class masks.")
    task_ids, _ = _task_head_rows(
        labels,
        task_class_masks,
        num_classes=int(first_mask.numel()),
    )
    resolved_device = torch.device(device)
    if torch.device(generator.device) != resolved_device:
        raise ValueError("Block-graph RNG and output device must match.")
    local_task_ids = task_ids.to(resolved_device)
    blocks: list[torch.Tensor] = []
    task_values = torch.unique(local_task_ids).detach().cpu().tolist()
    for task_id in sorted(int(value) for value in task_values):
        node_ids = torch.where(local_task_ids == task_id)[0]
        local_edges = random_undirected_edges(
            int(node_ids.numel()),
            directed_edges_per_node=directed_edges_per_node,
            generator=generator,
            device=resolved_device,
        )
        if local_edges.numel():
            blocks.append(node_ids[local_edges])
    if not blocks:
        return torch.empty((2, 0), dtype=torch.long, device=resolved_device)
    combined = torch.cat(blocks, dim=1)
    return canonical_undirected_edges(
        combined, int(labels.shape[0])
    )


def graph_homophily(
    edge_index: torch.Tensor, labels: torch.Tensor
) -> float:
    if labels.dtype != torch.long or labels.ndim != 1:
        raise ValueError("Homophily requires one-dimensional int64 labels.")
    logical = logical_undirected_edges(edge_index, int(labels.shape[0]))
    if logical.shape[1] == 0:
        return 0.0
    values = labels.detach().to(logical.device)
    pairs = logical
    return float((values[pairs[0]] == values[pairs[1]]).float().mean())


def _task_head_rows(
    labels: torch.Tensor,
    task_class_masks: Mapping[int, torch.Tensor],
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve one immutable Task-IL head and row mask for every label."""

    if (
        labels.dtype != torch.long
        or labels.ndim != 1
        or labels.numel() == 0
        or int(labels.min()) < 0
        or int(labels.max()) >= num_classes
    ):
        raise ValueError("Task-aware FedFST labels are invalid.")
    if not isinstance(task_class_masks, Mapping) or not task_class_masks:
        raise ValueError("Task-aware FedFST requires prior task class masks.")
    task_ids = torch.full(
        (labels.shape[0],), -1, dtype=torch.long, device=labels.device
    )
    row_masks = torch.zeros(
        (labels.shape[0], num_classes), dtype=torch.bool, device=labels.device
    )
    class_coverage = torch.zeros(
        (num_classes,), dtype=torch.long, device=labels.device
    )
    for task_id, raw_mask in sorted(task_class_masks.items()):
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
            raise ValueError("FedFST task-mask IDs must be non-negative integers.")
        if (
            not torch.is_tensor(raw_mask)
            or raw_mask.dtype != torch.bool
            or raw_mask.ndim != 1
            or raw_mask.numel() != num_classes
            or not bool(raw_mask.any())
        ):
            raise ValueError("FedFST task class mask does not align with its logits.")
        mask = raw_mask.to(device=labels.device)
        class_coverage += mask.long()
        positions = mask[labels]
        if bool(positions.any()):
            row_indices = torch.nonzero(positions, as_tuple=False).flatten()
            if bool((task_ids.index_select(0, row_indices) >= 0).any()):
                raise ValueError("FedFST labels are covered by multiple task heads.")
            task_ids.index_fill_(0, row_indices, task_id)
            row_masks.index_copy_(
                0,
                row_indices,
                mask.unsqueeze(0).expand(row_indices.numel(), -1),
            )
    if bool((class_coverage > 1).any()):
        raise ValueError("FedFST Task-IL class masks must be disjoint.")
    if bool((task_ids < 0).any()) or not bool(row_masks.any(dim=1).all()):
        raise ValueError("FedFST labels are not covered by exactly one prior task head.")
    return task_ids, row_masks


def task_block_diagonal_edges(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    task_class_masks: Mapping[int, torch.Tensor],
) -> torch.Tensor:
    """Drop Task-IL edges whose endpoints belong to different task heads.

    The returned edge tensor preserves the caller's arc order and device. It
    intentionally does not add self-loops; equations 9--12 add them exactly
    once after the task-block filter has been applied.
    """

    if labels.dtype != torch.long or labels.ndim != 1:
        raise ValueError("Task-block topology requires one-dimensional int64 labels.")
    _validate_edge_index(edge_index, int(labels.shape[0]))
    first_mask = next(iter(task_class_masks.values()), None)
    if not torch.is_tensor(first_mask) or first_mask.ndim != 1:
        raise ValueError("Task-block topology requires prior task class masks.")
    task_ids, _ = _task_head_rows(
        labels,
        task_class_masks,
        num_classes=int(first_mask.numel()),
    )
    if edge_index.shape[1] == 0:
        return edge_index.detach().clone().contiguous()
    endpoints = task_ids.to(edge_index.device)[edge_index]
    keep = endpoints[0] == endpoints[1]
    return edge_index[:, keep].detach().clone().contiguous()


def _remove_edge_type(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    *,
    edge_type: str,
    reduction_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if edge_type not in {"homophilic", "heterophilic"}:
        raise ValueError("Unknown logical edge type.")
    logical = logical_undirected_edges(edge_index, int(labels.shape[0]))
    if logical.shape[1] == 0:
        return edge_index.detach().clone()
    local_labels = labels.detach().to(logical.device)
    same = local_labels[logical[0]] == local_labels[logical[1]]
    candidate_mask = same if edge_type == "homophilic" else ~same
    candidate_indices = torch.where(candidate_mask)[0]
    if candidate_indices.numel() == 0:
        return canonical_undirected_edges(edge_index, int(labels.shape[0]))
    remove_count = max(
        1, int(math.ceil(int(candidate_indices.numel()) * reduction_ratio))
    )
    choice = torch.randperm(
        int(candidate_indices.numel()),
        generator=generator,
        device=logical.device,
    )[:remove_count]
    remove = candidate_indices[choice]
    keep = torch.ones(logical.shape[1], dtype=torch.bool, device=logical.device)
    keep[remove] = False
    return canonical_undirected_edges(
        logical[:, keep], int(labels.shape[0])
    )


def hhkr_edge_type(current_homophily: float, target_homophily: float) -> str:
    """Select the client-side removal direction described below equation 3."""

    return (
        "heterophilic"
        if current_homophily < target_homophily
        else "homophilic"
    )


def spectral_edge_type(current_energy: float, target_energy: float) -> str:
    """Select the server-side removal direction described below equation 8."""

    return "homophilic" if current_energy < target_energy else "heterophilic"


@dataclass(frozen=True)
class TopologyAdjustment:
    edge_index: torch.Tensor
    initial_value: float
    final_value: float
    iterations: int
    converged: bool
    removed_edge_types: tuple[str, ...]


def adjust_homophily(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    *,
    target: float,
    reduction_ratio: float,
    tolerance: float,
    max_iterations: int,
    generator: torch.Generator,
) -> TopologyAdjustment:
    """Match HHKR topology by pruning complete logical edge pairs."""

    if not 0.0 <= target <= 1.0:
        raise ValueError("Homophily target must lie in [0, 1].")
    current_edges = canonical_undirected_edges(edge_index, int(labels.shape[0]))
    initial = graph_homophily(current_edges, labels)
    current = initial
    history: list[str] = []
    best_edges = current_edges
    best_value = current
    best_error = abs(current - target)
    best_history: tuple[str, ...] = ()
    for iteration in range(max_iterations + 1):
        if abs(current - target) <= tolerance:
            return TopologyAdjustment(
                current_edges, initial, current, iteration, True, tuple(history)
            )
        if iteration == max_iterations:
            break
        edge_type = hhkr_edge_type(current, target)
        updated = _remove_edge_type(
            current_edges,
            labels,
            edge_type=edge_type,
            reduction_ratio=reduction_ratio,
            generator=generator,
        )
        if torch.equal(updated, current_edges):
            break
        current_edges = updated
        history.append(edge_type)
        current = graph_homophily(current_edges, labels)
        error = abs(current - target)
        # Strict comparison intentionally retains the first deterministic
        # candidate when two pruning states have the same target error.
        if error < best_error:
            best_edges = current_edges
            best_value = current
            best_error = error
            best_history = tuple(history)
    return TopologyAdjustment(
        best_edges,
        initial,
        best_value,
        len(best_history),
        False,
        best_history,
    )


@dataclass(frozen=True)
class SpectralEnergy:
    feature: float
    structure: float
    total: float
    sampled_feature_indices: tuple[int, ...]
    valid_feature_signals: int
    valid_structure_signals: int


def sampled_feature_indices(
    feature_dim: int, fraction: float, *, generator: torch.Generator
) -> torch.Tensor:
    if feature_dim <= 0 or not 0.0 < fraction <= 1.0:
        raise ValueError("Feature sampling dimensions are invalid.")
    count = max(1, int(feature_dim * fraction))
    return torch.randperm(feature_dim, generator=generator)[:count].sort().values


def high_frequency_energy(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    feature_indices: torch.Tensor,
    compute_device: torch.device | str | None = None,
) -> SpectralEnergy:
    """Compute equations 1, 2, and 7 without an eigendecomposition.

    As the paper leaves zero Rayleigh denominators undefined, this resolves
    them like the frozen reference snapshot: average only ratios whose
    denominator exceeds ``RAYLEIGH_DENOMINATOR_EPSILON``, or return zero when
    no ratio is valid.
    """

    if features.ndim != 2 or not features.is_floating_point():
        raise ValueError("Spectral energy requires floating [nodes, features].")
    num_nodes, feature_dim = (int(features.shape[0]), int(features.shape[1]))
    if num_nodes <= 0:
        raise ValueError("Spectral energy requires at least one node.")
    if (
        feature_indices.dtype != torch.long
        or feature_indices.ndim != 1
        or feature_indices.numel() == 0
        or int(feature_indices.min()) < 0
        or int(feature_indices.max()) >= feature_dim
    ):
        raise ValueError("Feature indices are invalid.")
    # CPU float64 remains the numerical oracle. Production may select CUDA,
    # where fp32 sparse arithmetic avoids host-bound L@X/L@A while preserving
    # the same equations and immutable inputs.
    resolved_device = torch.device(
        "cpu" if compute_device is None else compute_device
    )
    arithmetic_dtype = (
        torch.float64
        if resolved_device.type == "cpu"
        else (
            features.dtype
            if features.dtype in {torch.float32, torch.float64}
            else torch.float32
        )
    )
    values = features.detach().to(
        device=resolved_device, dtype=arithmetic_dtype
    )
    edges = canonical_undirected_edges(
        edge_index.detach().to(resolved_device), num_nodes
    )
    adjacency_values = torch.ones(
        edges.shape[1], dtype=arithmetic_dtype, device=resolved_device
    )
    adjacency = torch.sparse_coo_tensor(
        edges,
        adjacency_values,
        (num_nodes, num_nodes),
        device=resolved_device,
    ).coalesce()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense()
    nodes = torch.arange(num_nodes, dtype=torch.long, device=resolved_device)
    diagonal = torch.stack((nodes, nodes))
    laplacian = torch.sparse_coo_tensor(
        torch.cat((diagonal, edges), dim=1),
        torch.cat((degree, -adjacency_values)),
        (num_nodes, num_nodes),
        device=resolved_device,
    ).coalesce()

    selected = values[:, feature_indices.detach().to(resolved_device)]
    laplacian_features = torch.sparse.mm(laplacian, selected)
    feature_numerator = (selected * laplacian_features).sum(dim=0)
    feature_denominator = selected.square().sum(dim=0)
    valid_feature = feature_denominator > RAYLEIGH_DENOMINATOR_EPSILON
    feature_ratios = feature_numerator[valid_feature] / feature_denominator[
        valid_feature
    ]
    feature_energy = (
        float(feature_ratios.mean()) if feature_ratios.numel() else 0.0
    )

    if edges.shape[1]:
        laplacian_adjacency = torch.sparse.mm(laplacian, adjacency).coalesce()
        masked = laplacian_adjacency.sparse_mask(adjacency).coalesce()
        structure_numerator = torch.zeros(
            num_nodes, dtype=arithmetic_dtype, device=resolved_device
        )
        structure_numerator.scatter_add_(0, masked.indices()[1], masked.values())
        structure_denominator = torch.sparse.sum(adjacency, dim=0).to_dense()
        valid_structure = structure_denominator > RAYLEIGH_DENOMINATOR_EPSILON
        structure_ratios = (
            structure_numerator[valid_structure]
            / structure_denominator[valid_structure]
        )
        structure_energy = (
            float(structure_ratios.mean()) if structure_ratios.numel() else 0.0
        )
    else:
        valid_structure = torch.zeros(
            num_nodes, dtype=torch.bool, device=resolved_device
        )
        structure_energy = 0.0
    total = 0.5 * feature_energy + 0.5 * structure_energy
    if not all(math.isfinite(value) for value in (feature_energy, structure_energy, total)):
        raise RuntimeError("FedFST spectral energy became non-finite.")
    return SpectralEnergy(
        feature=feature_energy,
        structure=structure_energy,
        total=total,
        sampled_feature_indices=tuple(
            int(v) for v in feature_indices.detach().cpu().tolist()
        ),
        valid_feature_signals=int(valid_feature.sum()),
        valid_structure_signals=int(valid_structure.sum()),
    )


def adjust_spectral_energy(
    features: torch.Tensor,
    labels: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    feature_indices: torch.Tensor,
    target: float,
    reduction_ratio: float,
    tolerance: float,
    max_iterations: int,
    generator: torch.Generator,
    compute_device: torch.device | str | None = None,
) -> tuple[TopologyAdjustment, SpectralEnergy]:
    """Match the server graph's equation-7 target using paper edge directions."""

    if not math.isfinite(float(target)) or target < 0.0:
        raise ValueError("Spectral target must be finite and non-negative.")
    current_edges = canonical_undirected_edges(edge_index, int(labels.shape[0]))
    energy = high_frequency_energy(
        features,
        current_edges,
        feature_indices=feature_indices,
        compute_device=compute_device,
    )
    initial = energy.total
    history: list[str] = []
    best_edges = current_edges
    best_energy = energy
    best_error = abs(energy.total - target)
    best_history: tuple[str, ...] = ()
    for iteration in range(max_iterations + 1):
        if abs(energy.total - target) <= tolerance:
            return (
                TopologyAdjustment(
                    current_edges,
                    initial,
                    energy.total,
                    iteration,
                    True,
                    tuple(history),
                ),
                energy,
            )
        if iteration == max_iterations:
            break
        edge_type = spectral_edge_type(energy.total, target)
        updated = _remove_edge_type(
            current_edges,
            labels,
            edge_type=edge_type,
            reduction_ratio=reduction_ratio,
            generator=generator,
        )
        if torch.equal(updated, current_edges):
            break
        current_edges = updated
        history.append(edge_type)
        energy = high_frequency_energy(
            features,
            current_edges,
            feature_indices=feature_indices,
            compute_device=compute_device,
        )
        error = abs(energy.total - target)
        # Keep the first candidate on an exact tie so selection is stable
        # independently of container ordering or later pruning attempts.
        if error < best_error:
            best_edges = current_edges
            best_energy = energy
            best_error = error
            best_history = tuple(history)
    return (
        TopologyAdjustment(
            best_edges,
            initial,
            best_energy.total,
            len(best_history),
            False,
            best_history,
        ),
        best_energy,
    )


@dataclass(frozen=True)
class HHKRLoss:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    feature_kl: torch.Tensor


def hhkr_loss(
    teacher_logits: torch.Tensor,
    generated_features: torch.Tensor,
    matched_real_features: torch.Tensor,
    generated_labels: torch.Tensor,
    *,
    lambda_kl: float,
    task_class_masks: Mapping[int, torch.Tensor] | None = None,
) -> HHKRLoss:
    """Compute equations 4--6, selecting each Task-IL label's source head."""

    if generated_features.shape != matched_real_features.shape:
        raise ValueError("HHKR generated and matched real features must align.")
    if teacher_logits.shape[0] != generated_labels.shape[0]:
        raise ValueError("HHKR logits and generated labels must align.")
    resolved_logits = teacher_logits
    if task_class_masks is not None:
        _, row_masks = _task_head_rows(
            generated_labels,
            task_class_masks,
            num_classes=int(teacher_logits.shape[1]),
        )
        resolved_logits = teacher_logits.masked_fill(~row_masks, -1e12)
    cross_entropy = F.cross_entropy(resolved_logits, generated_labels)
    feature_kl = F.kl_div(
        F.log_softmax(generated_features, dim=1),
        F.softmax(matched_real_features, dim=1),
        reduction="batchmean",
    )
    total = cross_entropy + float(lambda_kl) * feature_kl
    return HHKRLoss(total, cross_entropy, feature_kl)


def weighted_generator_average(
    states: Mapping[int, Mapping[str, torch.Tensor]],
    historical_node_counts: Mapping[int, int],
) -> dict[str, torch.Tensor]:
    """Apply the generator half of equation 8 in client-ID order."""

    if not states or set(states) != set(historical_node_counts):
        raise ValueError("Generator states/counts must cover the same clients.")
    clients = tuple(sorted(states))
    if any(
        isinstance(historical_node_counts[client], bool)
        or not isinstance(historical_node_counts[client], int)
        or historical_node_counts[client] <= 0
        for client in clients
    ):
        raise ValueError("Historical node aggregation weights must be positive.")
    total = sum(historical_node_counts[client] for client in clients)
    keys = tuple(sorted(states[clients[0]]))
    if any(tuple(sorted(states[client])) != keys for client in clients):
        raise ValueError("Generator parameter keys must match across clients.")
    output: dict[str, torch.Tensor] = {}
    for key in keys:
        reference = states[clients[0]][key]
        if not reference.is_floating_point() or not torch.isfinite(reference).all():
            raise ValueError("Generator state must contain finite floating tensors.")
        result = torch.zeros_like(reference)
        for client in clients:
            value = states[client][key]
            if (
                value.shape != reference.shape
                or value.dtype != reference.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError("Generator parameter shapes/dtypes/values must match.")
            result.add_(
                value.to(result.device),
                alpha=float(historical_node_counts[client]) / total,
            )
        output[key] = result.detach().cpu().clone().contiguous()
    return output


def weighted_spectral_target(
    energies: Mapping[int, float], historical_node_counts: Mapping[int, int]
) -> float:
    """Apply the scalar spectral half of equation 8."""

    if not energies or set(energies) != set(historical_node_counts):
        raise ValueError("Spectral energies/counts must cover the same clients.")
    clients = tuple(sorted(energies))
    if any(
        isinstance(historical_node_counts[client], bool)
        or not isinstance(historical_node_counts[client], int)
        or historical_node_counts[client] <= 0
        or isinstance(energies[client], bool)
        or not isinstance(energies[client], (int, float))
        or not math.isfinite(float(energies[client]))
        for client in clients
    ):
        raise ValueError("Spectral targets and weights must be finite/positive.")
    total = sum(historical_node_counts[client] for client in clients)
    return sum(
        float(energies[client]) * historical_node_counts[client] / total
        for client in clients
    )


def transition_matrix(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Materialize equation 9 for small behavioral oracles."""

    resolved = add_self_loops_once(edge_index, num_nodes)
    matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    if resolved.numel():
        matrix[resolved[0].cpu(), resolved[1].cpu()] = 1.0
    degree = matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
    return matrix / degree


def smooth_probabilities(
    probabilities: torch.Tensor, edge_index: torch.Tensor, hops: int
) -> torch.Tensor:
    """Compute equation 10 with row-normalized sparse propagation."""

    if probabilities.ndim != 2 or not probabilities.is_floating_point():
        raise ValueError("Smoothing requires floating [nodes, classes] values.")
    if isinstance(hops, bool) or not isinstance(hops, int) or hops < 0:
        raise ValueError("Smoothing hops must be a non-negative integer.")
    num_nodes = int(probabilities.shape[0])
    resolved = add_self_loops_once(edge_index.to(probabilities.device), num_nodes)
    source, target = resolved
    degree = torch.bincount(source, minlength=num_nodes).to(
        device=probabilities.device, dtype=probabilities.dtype
    )
    weights = degree[source].clamp_min(1.0).reciprocal()
    transition = torch.sparse_coo_tensor(
        resolved,
        weights,
        (num_nodes, num_nodes),
        device=probabilities.device,
    ).coalesce()
    output = probabilities
    for _ in range(hops):
        output = torch.sparse.mm(transition, output)
    return output


def edgewise_low_frequency_kl(
    teacher_probabilities: torch.Tensor,
    student_probabilities: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Compute equation 12: KL(teacher at v || student at u) over E'."""

    if teacher_probabilities.shape != student_probabilities.shape:
        raise ValueError("Teacher and student probability shapes must match.")
    num_nodes = int(student_probabilities.shape[0])
    resolved = add_self_loops_once(edge_index.to(student_probabilities.device), num_nodes)
    source, target = resolved
    epsilon = torch.finfo(student_probabilities.dtype).eps
    teacher = teacher_probabilities.clamp_min(epsilon)
    student = student_probabilities.clamp_min(epsilon)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(epsilon)
    student = student / student.sum(dim=1, keepdim=True).clamp_min(epsilon)
    terms = teacher[target] * (teacher[target].log() - student[source].log())
    return terms.sum(dim=1).mean()


@dataclass(frozen=True)
class HLSTLoss:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    low_frequency_kl: torch.Tensor


def hlst_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    smoothing_hops: int,
    lambda_low: float,
    task_class_masks: Mapping[int, torch.Tensor] | None = None,
) -> HLSTLoss:
    """Compute equations 9--13, using source heads for Task-IL replay."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("HLST teacher/student logits must have matching shapes.")
    resolved_edges = edge_index
    resolved_student_logits = student_logits
    resolved_teacher_logits = teacher_logits
    if task_class_masks is not None:
        _, row_masks = _task_head_rows(
            labels,
            task_class_masks,
            num_classes=int(student_logits.shape[1]),
        )
        resolved_student_logits = student_logits.masked_fill(~row_masks, -1e12)
        resolved_teacher_logits = teacher_logits.masked_fill(~row_masks, -1e12)
        resolved_edges = task_block_diagonal_edges(
            edge_index, labels, task_class_masks
        )
    cross_entropy = F.cross_entropy(resolved_student_logits, labels)
    student = smooth_probabilities(
        F.softmax(resolved_student_logits, dim=1),
        resolved_edges,
        smoothing_hops,
    )
    teacher = smooth_probabilities(
        F.softmax(resolved_teacher_logits, dim=1),
        resolved_edges,
        smoothing_hops,
    )
    low_frequency = edgewise_low_frequency_kl(
        teacher, student, resolved_edges
    )
    total = cross_entropy + float(lambda_low) * low_frequency
    return HLSTLoss(total, cross_entropy, low_frequency)
