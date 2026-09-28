"""Core immutable contracts for scenarios, partitions, and client streams."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Tuple

import torch

from gecko.validation import ScenarioValidationError


TensorMap = Dict[str, torch.Tensor]
TaskSplitMap = Dict[int, Dict[str, torch.Tensor]]


def _clone_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ScenarioSpec:
    """Cursor-free central scenario description.

    This object is central-only. Client-facing views are created by the stream
    builder and never expose the full labels, domains, or ownership mapping.
    """

    dataset_name: str
    problem_type: str
    incremental_type: str
    num_tasks: int
    num_classes: int
    num_features: int
    metrics: Tuple[str, ...]
    edge_index: torch.Tensor
    node_features: torch.Tensor
    query_ids_by_task_split: TaskSplitMap
    query_task_ids: torch.Tensor
    labels: torch.Tensor
    query_endpoints: torch.Tensor | None = None
    task_class_sets: Dict[int, torch.Tensor] = field(default_factory=dict)
    domains: torch.Tensor | None = None
    task_masks: torch.Tensor | None = None
    logical_edge_ids: torch.Tensor | None = None
    logical_edge_to_raw_edges: Dict[int, torch.Tensor] = field(default_factory=dict)
    context_edge_index: torch.Tensor | None = None
    bipartite: bool = False
    node_types: torch.Tensor | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        problem = self.problem_type.upper()
        incremental = self.incremental_type.lower()
        if problem not in {"NC", "LC", "LP"}:
            raise ScenarioValidationError(f"Unknown problem type: {problem}")
        if incremental not in {"task", "class", "domain"}:
            raise ScenarioValidationError(f"Unsupported UEFA incremental type: {incremental}")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ScenarioValidationError("edge_index must have shape [2, num_edges].")
        if self.node_features.ndim < 2:
            raise ScenarioValidationError("node_features must have at least two dimensions.")
        if self.node_features.shape[0] == 0:
            raise ScenarioValidationError("A scenario must contain at least one node.")
        if problem == "NC":
            if self.labels.shape[0] != self.node_features.shape[0]:
                raise ScenarioValidationError("NC labels must align with global nodes.")
        elif self.query_endpoints is None or self.query_endpoints.ndim != 2 or self.query_endpoints.shape[1] != 2:
            raise ScenarioValidationError("LC/LP query_endpoints must have shape [num_queries, 2].")
        if self.query_task_ids.shape[0] != self.labels.shape[0]:
            raise ScenarioValidationError("query_task_ids and labels must align.")
        required_splits = {"train", "val", "test"}
        if set(self.query_ids_by_task_split) != set(range(self.num_tasks)):
            raise ScenarioValidationError("Every immutable global task ID must have query splits.")
        for task_id, split_map in self.query_ids_by_task_split.items():
            if set(split_map) != required_splits:
                raise ScenarioValidationError(f"Task {task_id} does not define train/val/test.")
            for split, indices in split_map.items():
                if indices.dtype != torch.long:
                    raise ScenarioValidationError(f"Task {task_id} {split} indices must be int64.")
                if indices.numel() and (indices.min() < 0 or indices.max() >= self.labels.shape[0]):
                    raise ScenarioValidationError(f"Task {task_id} {split} contains invalid query IDs.")
        if problem in {"NC", "LC"} and self.labels.ndim == 1:
            supervised_ids = torch.cat(
                [
                    indices
                    for split_map in self.query_ids_by_task_split.values()
                    for indices in split_map.values()
                ]
            )
            if supervised_ids.numel():
                supervised_labels = self.labels[torch.unique(supervised_ids)]
                if supervised_labels.min() < 0 or supervised_labels.max() >= self.num_classes:
                    raise ScenarioValidationError(
                        "A supervised classification label is outside [0, num_classes)."
                    )

    def detached_clone(self) -> "ScenarioSpec":
        return ScenarioSpec(**_clone_value(self.__dict__))

    def client_view(self) -> "ClientScenarioView":
        """Return only predictor metadata that is safe to expose to clients."""

        return ClientScenarioView(
            problem_type=self.problem_type,
            incremental_type=self.incremental_type,
            num_classes=self.num_classes,
            num_features=self.num_features,
        )


@dataclass(frozen=True)
class ClientScenarioView:
    """Minimal scenario metadata available inside a federated client."""

    problem_type: str
    incremental_type: str
    num_classes: int
    num_features: int


@dataclass(frozen=True)
class ClientGraph:
    client_id: int
    edge_index: torch.Tensor
    node_features: torch.Tensor
    local_to_global: torch.Tensor
    global_to_local: Dict[int, int]
    boundary_mask: torch.Tensor

    def client_view(self) -> "ClientGraphView":
        """Return a graph view with no global IDs or boundary information."""

        return ClientGraphView(
            client_id=self.client_id,
            edge_index=self.edge_index,
            node_features=self.node_features,
        )


@dataclass(frozen=True)
class ClientGraphView:
    """Strict induced graph exposed to local training code."""

    client_id: int
    edge_index: torch.Tensor
    node_features: torch.Tensor


@dataclass(frozen=True)
class ClientTaskShard:
    client_id: int
    global_task_id: int
    train_queries: torch.Tensor
    train_labels: torch.Tensor
    task_class_mask: torch.Tensor | None = None
    context_edge_index: torch.Tensor | None = None

    @property
    def supervised_query_count(self) -> int:
        return int(self.train_labels.shape[0])

    @property
    def positive_anchor_count(self) -> int:
        if self.train_labels.ndim != 1:
            return self.supervised_query_count
        return int((self.train_labels == 1).sum())


@dataclass(frozen=True)
class CentralEvaluationShard:
    """Held-out query coordinates and IDs owned by the central evaluator."""

    client_id: int
    global_task_id: int
    train_query_ids: torch.Tensor
    validation_queries: torch.Tensor
    validation_query_ids: torch.Tensor
    test_queries: torch.Tensor
    test_query_ids: torch.Tensor
    context_edge_index: torch.Tensor | None = None

    def queries(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        if split == "validation":
            return self.validation_queries, self.validation_query_ids
        if split == "test":
            return self.test_queries, self.test_query_ids
        raise ValueError("Central evaluation split must be 'validation' or 'test'.")


@dataclass(frozen=True)
class PartitionResult:
    node_owner: torch.Tensor
    client_graphs: Dict[int, ClientGraph]
    micro_community_ids: torch.Tensor
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class OrderPlan:
    canonical_order: Tuple[int, ...]
    client_orders: Dict[int, Tuple[int, ...]]
    inverse_client_orders: Dict[int, Dict[int, int]]
    cohort_assignments: Dict[int, int]
    block_size: int
    seed: int
    diagnostics: Dict[str, Any]

    def global_task(self, client_id: int, local_stage: int) -> int:
        return self.client_orders[client_id][local_stage]

    def seen_tasks(self, client_id: int, local_stage: int) -> Tuple[int, ...]:
        return self.client_orders[client_id][: local_stage + 1]


@dataclass(frozen=True)
class ParticipationPlan:
    trace: Dict[int, Dict[int, Tuple[int, ...]]]
    fraction: float
    rounds_per_stage: int
    seed: int


@dataclass
class ClientState:
    client_id: int
    seen_tasks: set[int] = field(default_factory=set)
    seen_class_mask: torch.Tensor | None = None
    continual_state: Dict[str, Any] = field(default_factory=dict)
    local_model_state: Dict[str, torch.Tensor] = field(default_factory=dict)
    strategy_state: Dict[str, Any] = field(default_factory=dict)
    personalized_model_state: Dict[str, torch.Tensor] = field(default_factory=dict)
    topology_overlays: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalUpdateResult:
    client_id: int
    global_task_id: int
    shared_state: Dict[str, torch.Tensor]
    weight: int
    training_loss: float
    communication_bytes: int
    shareable_keys: Tuple[str, ...]
