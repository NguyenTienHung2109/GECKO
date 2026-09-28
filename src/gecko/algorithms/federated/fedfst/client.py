"""Strict-local client history and HHKR training for FedFST."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any
from typing import Mapping

import torch

from gecko.algorithms.federated.fedfst.core import ConditionalFeatureGenerator
from gecko.algorithms.federated.fedfst.core import FedFSTParameters
from gecko.algorithms.federated.fedfst.core import adjust_homophily
from gecko.algorithms.federated.fedfst.core import canonical_undirected_edges
from gecko.algorithms.federated.fedfst.core import generate_balanced_features
from gecko.algorithms.federated.fedfst.core import graph_homophily
from gecko.algorithms.federated.fedfst.core import hhkr_loss
from gecko.algorithms.federated.fedfst.core import high_frequency_energy
from gecko.algorithms.federated.fedfst.core import local_generator
from gecko.algorithms.federated.fedfst.core import random_block_diagonal_edges
from gecko.algorithms.federated.fedfst.core import random_undirected_edges
from gecko.algorithms.federated.fedfst.core import sampled_feature_indices
from gecko.algorithms.federated.fedfst.core import task_block_diagonal_edges


CLIENT_STATE_VERSION = "uefa-fedfst-client-history-v1"


def _empty_history(feature_dim: int) -> dict[str, object]:
    return {
        "version": CLIENT_STATE_VERSION,
        "feature_dim": int(feature_dim),
        "node_ids": torch.empty((0,), dtype=torch.long),
        "features": torch.empty((0, feature_dim), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.long),
        "node_stage_indices": torch.empty((0,), dtype=torch.long),
        "node_task_ids": torch.empty((0,), dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "records": [],
        "last_hhkr": {},
    }


def _validate_history(state: Mapping[str, object]) -> None:
    expected = {
        "version",
        "feature_dim",
        "node_ids",
        "features",
        "labels",
        "node_stage_indices",
        "node_task_ids",
        "edge_index",
        "records",
        "last_hhkr",
    }
    if set(state) != expected or state["version"] != CLIENT_STATE_VERSION:
        raise ValueError("FedFST client history identity mismatch.")
    feature_dim = state["feature_dim"]
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError("FedFST client feature dimension is invalid.")
    node_ids = state["node_ids"]
    features = state["features"]
    labels = state["labels"]
    stages = state["node_stage_indices"]
    tasks = state["node_task_ids"]
    edges = state["edge_index"]
    if not all(torch.is_tensor(value) for value in (node_ids, features, labels, stages, tasks, edges)):
        raise TypeError("FedFST client history tensors are malformed.")
    count = int(node_ids.shape[0])
    if (
        node_ids.dtype != torch.long
        or node_ids.ndim != 1
        or features.ndim != 2
        or tuple(features.shape) != (count, feature_dim)
        or not features.is_floating_point()
        or labels.dtype != torch.long
        or tuple(labels.shape) != (count,)
        or stages.dtype != torch.long
        or tuple(stages.shape) != (count,)
        or tasks.dtype != torch.long
        or tuple(tasks.shape) != (count,)
        or edges.dtype != torch.long
        or edges.ndim != 2
        or edges.shape[0] != 2
    ):
        raise ValueError("FedFST client history tensor shapes/dtypes are invalid.")
    invalid_endpoint = bool(
        edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= count)
    )
    if count and (
        int(node_ids.min()) < 0
        or torch.unique(node_ids).numel() != count
        or not torch.equal(node_ids, node_ids.sort().values)
        or invalid_endpoint
    ):
        raise ValueError("FedFST client history node identity is invalid.")
    if not count and edges.numel():
        raise ValueError("FedFST client history contains a negative endpoint.")
    if not isinstance(state["records"], (list, tuple)) or not isinstance(
        state["last_hhkr"], Mapping
    ):
        raise TypeError("FedFST client history metadata is malformed.")
    if (
        not bool(torch.isfinite(features).all())
        or (labels.numel() and int(labels.min()) < 0)
        or not torch.equal(edges, canonical_undirected_edges(edges, count))
    ):
        raise ValueError("FedFST client history values/topology are invalid.")
    records = state["records"]
    assert isinstance(records, (list, tuple))
    if stages.numel() and (
        bool((stages < 0).any()) or int(stages.max()) >= len(records)
    ):
        raise ValueError("FedFST client history node stages are invalid.")
    for stage_index, record in enumerate(records):
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "stage_index",
                "global_task_id",
                "new_train_nodes",
                "total_train_nodes",
                "classes",
            }
            or type(record["stage_index"]) is not int
            or record["stage_index"] != stage_index
            or type(record["global_task_id"]) is not int
            or record["global_task_id"] < 0
            or type(record["new_train_nodes"]) is not int
            or record["new_train_nodes"] <= 0
            or type(record["total_train_nodes"]) is not int
            or not isinstance(record["classes"], (list, tuple))
        ):
            raise ValueError("FedFST client history record is malformed.")
        stage_mask = stages == stage_index
        stage_classes = sorted(
            {int(value) for value in labels[stage_mask].tolist()}
        )
        if (
            int(stage_mask.sum()) != record["new_train_nodes"]
            or int((stages <= stage_index).sum()) != record["total_train_nodes"]
            or list(record["classes"]) != stage_classes
            or not bool((tasks[stage_mask] == record["global_task_id"]).all())
        ):
            raise ValueError("FedFST client history record/tensor mismatch.")


def get_or_create_history(container: dict[str, object], feature_dim: int) -> dict[str, object]:
    """Return the one client-private FedFST history inside strategy state."""

    if "fedfst" not in container:
        container["fedfst"] = _empty_history(feature_dim)
    history = container["fedfst"]
    if not isinstance(history, dict):
        raise TypeError("FedFST client strategy state must be a mapping.")
    _validate_history(history)
    if history["feature_dim"] != feature_dim:
        raise ValueError("FedFST client feature dimension changed across stages.")
    return history


def _deduplicate_current_labels(
    queries: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(queries, stable=True)
    ordered_queries = queries[order]
    ordered_labels = labels[order]
    unique_queries: list[int] = []
    unique_labels: list[int] = []
    for query, label in zip(ordered_queries.tolist(), ordered_labels.tolist()):
        if unique_queries and query == unique_queries[-1]:
            if label != unique_labels[-1]:
                raise ValueError("One FedFST train node has conflicting labels.")
            continue
        unique_queries.append(int(query))
        unique_labels.append(int(label))
    return (
        torch.tensor(unique_queries, dtype=torch.long),
        torch.tensor(unique_labels, dtype=torch.long),
    )


def append_current_train_history(
    container: dict[str, object], context: Any
) -> None:
    """Append only current train nodes/labels and currently visible local edges."""

    if context.problem_type != "NC" or context.incremental_setting not in {
        "class",
        "task",
    }:
        raise ValueError("FedFST history supports NC Class/Task-IL only.")
    node_features = context.node_features.detach().cpu().contiguous()
    if node_features.ndim != 2 or not node_features.is_floating_point():
        raise ValueError("FedFST requires floating strict-local node features.")
    history = get_or_create_history(container, int(node_features.shape[1]))
    records = list(history["records"])
    if any(int(record["stage_index"]) == int(context.stage_index) for record in records):
        raise RuntimeError("FedFST attempted to consolidate one stage twice.")
    queries = context.train_queries.detach().cpu()
    labels = context.train_labels.detach().cpu()
    if (
        queries.dtype != torch.long
        or queries.ndim != 1
        or labels.dtype != torch.long
        or labels.ndim != 1
        or labels.shape[0] != queries.shape[0]
        or queries.numel() == 0
    ):
        raise ValueError("FedFST requires non-empty scalar NC train labels.")
    queries, labels = _deduplicate_current_labels(queries, labels)
    if int(queries.min()) < 0 or int(queries.max()) >= node_features.shape[0]:
        raise ValueError("FedFST train query is outside the strict-local graph.")

    old_ids = history["node_ids"]
    old_features = history["features"]
    old_labels = history["labels"]
    old_stages = history["node_stage_indices"]
    old_tasks = history["node_task_ids"]
    old_edges = history["edge_index"]
    assert all(
        torch.is_tensor(value)
        for value in (old_ids, old_features, old_labels, old_stages, old_tasks, old_edges)
    )
    old_by_id = {int(node): index for index, node in enumerate(old_ids.tolist())}
    current_by_id = {int(node): int(label) for node, label in zip(queries, labels)}
    for node, label in current_by_id.items():
        if node in old_by_id and int(old_labels[old_by_id[node]]) != label:
            raise ValueError("FedFST historical train label changed across stages.")
    union_ids = torch.tensor(
        sorted(set(old_ids.tolist()) | set(queries.tolist())), dtype=torch.long
    )
    union_position = {int(node): index for index, node in enumerate(union_ids.tolist())}
    feature_rows: list[torch.Tensor] = []
    label_rows: list[int] = []
    stage_rows: list[int] = []
    task_rows: list[int] = []
    for node in union_ids.tolist():
        if node in old_by_id:
            old_index = old_by_id[node]
            feature = old_features[old_index]
            if not torch.equal(feature, node_features[node].to(feature.dtype)):
                raise ValueError("FedFST strict-local features changed across stages.")
            feature_rows.append(feature.detach().clone())
            label_rows.append(int(old_labels[old_index]))
            stage_rows.append(int(old_stages[old_index]))
            task_rows.append(int(old_tasks[old_index]))
        else:
            feature_rows.append(node_features[node].detach().clone())
            label_rows.append(current_by_id[node])
            stage_rows.append(int(context.stage_index))
            task_rows.append(int(context.global_task_id))

    remapped_old = torch.empty((2, 0), dtype=torch.long)
    if old_edges.numel():
        old_endpoint_ids = old_ids[old_edges]
        remapped_old = torch.tensor(
            [
                [union_position[int(node)] for node in old_endpoint_ids[0].tolist()],
                [union_position[int(node)] for node in old_endpoint_ids[1].tolist()],
            ],
            dtype=torch.long,
        )
    visible_edges = context.effective_edge_index.detach().cpu()
    lookup = torch.full((node_features.shape[0],), -1, dtype=torch.long)
    lookup[union_ids] = torch.arange(union_ids.shape[0], dtype=torch.long)
    if visible_edges.numel():
        remapped_visible = lookup[visible_edges]
        keep = (remapped_visible[0] >= 0) & (remapped_visible[1] >= 0)
        remapped_visible = remapped_visible[:, keep]
    else:
        remapped_visible = torch.empty((2, 0), dtype=torch.long)
    merged_edges = canonical_undirected_edges(
        torch.cat((remapped_old, remapped_visible), dim=1), int(union_ids.shape[0])
    )
    if context.incremental_setting == "task" and merged_edges.numel():
        union_tasks = torch.tensor(task_rows, dtype=torch.long)
        endpoint_tasks = union_tasks[merged_edges]
        merged_edges = merged_edges[
            :, endpoint_tasks[0] == endpoint_tasks[1]
        ].contiguous()

    history.update(
        {
            "node_ids": union_ids,
            "features": torch.stack(feature_rows).contiguous(),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "node_stage_indices": torch.tensor(stage_rows, dtype=torch.long),
            "node_task_ids": torch.tensor(task_rows, dtype=torch.long),
            "edge_index": merged_edges,
            "records": records
            + [
                {
                    "stage_index": int(context.stage_index),
                    "global_task_id": int(context.global_task_id),
                    "new_train_nodes": int(
                        sum(node not in old_by_id for node in queries.tolist())
                    ),
                    "total_train_nodes": int(union_ids.shape[0]),
                    "classes": sorted({int(value) for value in labels.tolist()}),
                }
            ],
        }
    )
    _validate_history(history)


def history_payload_bytes(container: Mapping[str, object]) -> int:
    """Count only historical tensor payloads, not Python bookkeeping."""

    history = container.get("fedfst")
    if not isinstance(history, Mapping):
        return 0
    _validate_history(history)
    return sum(
        int(value.numel() * value.element_size())
        for name, value in history.items()
        if name not in {"last_hhkr"} and torch.is_tensor(value)
    )


def _validate_history_boundary(
    history: Mapping[str, object],
    *,
    stage_index: int,
    expected_task_ids: tuple[int, ...],
) -> None:
    """Reject checkpoint history from the current/future stage before HHKR."""

    _validate_history(history)
    if stage_index < 0 or len(expected_task_ids) != stage_index:
        raise ValueError("FedFST historical task boundary is invalid.")
    records = history["records"]
    assert isinstance(records, (list, tuple))
    if len(records) != stage_index:
        raise ValueError("FedFST history must contain exactly the prior stages.")
    for expected_stage, (record, expected_task) in enumerate(
        zip(records, expected_task_ids)
    ):
        if (
            not isinstance(record, Mapping)
            or int(record.get("stage_index", -1)) != expected_stage
            or int(record.get("global_task_id", -1)) != expected_task
        ):
            raise ValueError("FedFST historical record order/identity is invalid.")
    node_stages = history["node_stage_indices"]
    node_tasks = history["node_task_ids"]
    assert torch.is_tensor(node_stages) and torch.is_tensor(node_tasks)
    if node_stages.numel() and (
        bool((node_stages < 0).any()) or bool((node_stages >= stage_index).any())
    ):
        raise ValueError("FedFST history contains current/future-stage nodes.")
    for historical_stage, expected_task in enumerate(expected_task_ids):
        mask = node_stages == historical_stage
        if bool(mask.any()) and not bool((node_tasks[mask] == expected_task).all()):
            raise ValueError("FedFST historical node task IDs are inconsistent.")


def validate_client_history_boundary(
    container: Mapping[str, object],
    *,
    feature_dim: int,
    stage_index: int,
    expected_task_ids: tuple[int, ...],
) -> None:
    """Validate restored client history without creating or mutating it."""

    history = container.get("fedfst")
    if history is None:
        if stage_index == 0:
            return
        raise ValueError("FedFST restored client history is missing.")
    if not isinstance(history, Mapping):
        raise TypeError("FedFST restored client history must be a mapping.")
    if history.get("feature_dim") != feature_dim:
        raise ValueError("FedFST restored client feature dimension is invalid.")
    _validate_history_boundary(
        history,
        stage_index=stage_index,
        expected_task_ids=expected_task_ids,
    )


@dataclass(frozen=True)
class ClientHHKRUpload:
    """Only the paper-required HHKR wire payload; history stays client-local."""

    client_id: int
    generator_state: dict[str, torch.Tensor]
    spectral_energy: float
    historical_node_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.client_id, bool)
            or not isinstance(self.client_id, int)
            or self.client_id < 0
        ):
            raise ValueError("FedFST upload client ID must be non-negative.")
        if not self.generator_state or any(
            not isinstance(name, str)
            or not name
            or not torch.is_tensor(value)
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
            for name, value in self.generator_state.items()
        ):
            raise ValueError("FedFST upload generator state is malformed.")
        object.__setattr__(
            self,
            "generator_state",
            {
                name: value.detach().cpu().clone().contiguous()
                for name, value in self.generator_state.items()
            },
        )
        if (
            isinstance(self.spectral_energy, bool)
            or not isinstance(self.spectral_energy, (int, float))
            or not math.isfinite(float(self.spectral_energy))
            or float(self.spectral_energy) < 0.0
        ):
            raise ValueError("FedFST upload spectral energy is invalid.")
        if (
            isinstance(self.historical_node_count, bool)
            or not isinstance(self.historical_node_count, int)
            or self.historical_node_count <= 0
        ):
            raise ValueError("FedFST upload historical node count is invalid.")
    @property
    def tensor_payload_bytes(self) -> int:
        return sum(
            int(value.numel() * value.element_size())
            for value in self.generator_state.values()
        )


def _clone_task_class_masks(
    task_class_masks: Mapping[int, torch.Tensor] | None,
) -> dict[int, torch.Tensor] | None:
    """Clone optional prior-task masks into workspace-owned CPU state."""

    if task_class_masks is None:
        return None
    if not isinstance(task_class_masks, Mapping):
        raise TypeError("FedFST task class masks must be a mapping.")
    if any(
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id < 0
        for task_id in task_class_masks
    ):
        raise ValueError("FedFST task-mask IDs must be non-negative integers.")
    output: dict[int, torch.Tensor] = {}
    width: int | None = None
    for task_id in sorted(task_class_masks):
        mask = task_class_masks[task_id]
        if (
            not torch.is_tensor(mask)
            or mask.dtype != torch.bool
            or mask.ndim != 1
            or not bool(mask.any())
        ):
            raise ValueError("FedFST task class mask is malformed.")
        if width is None:
            width = int(mask.numel())
        elif mask.numel() != width:
            raise ValueError("FedFST task class masks have inconsistent widths.")
        output[task_id] = mask.detach().cpu().clone().contiguous()
    if output:
        coverage = torch.stack(tuple(output.values())).long().sum(dim=0)
        if bool((coverage > 1).any()):
            raise ValueError("FedFST Task-IL class masks must be disjoint.")
    return output


class FedFSTClientWorkspace:
    """Transactional client-private state and one ephemeral teacher copy."""

    def __init__(
        self,
        client: Any,
        *,
        feature_dim: int,
        stage_index: int,
        expected_task_ids: tuple[int, ...],
        task_class_masks: Mapping[int, torch.Tensor] | None = None,
    ) -> None:
        if (
            isinstance(client.client_id, bool)
            or not isinstance(client.client_id, int)
            or client.client_id < 0
        ):
            raise ValueError("FedFST workspace client ID is invalid.")
        self.client_id = client.client_id
        self._client = client
        self._strategy_state = copy.deepcopy(client.state.strategy_state)
        self._task_class_masks = _clone_task_class_masks(task_class_masks)
        history = get_or_create_history(self._strategy_state, feature_dim)
        _validate_history_boundary(
            history,
            stage_index=stage_index,
            expected_task_ids=expected_task_ids,
        )
        self._teacher: torch.nn.Module | None = None
        self._committed = False

    def bind_teacher(
        self, teacher_model: torch.nn.Module, *, device: torch.device
    ) -> None:
        if self._teacher is not None:
            raise RuntimeError("FedFST client teacher may be delivered only once.")
        self._teacher = copy.deepcopy(teacher_model).to(device)

    def execute_hhkr(
        self,
        *,
        initial_generator_state: Mapping[str, torch.Tensor],
        feature_dim: int,
        conditioning_classes: tuple[int, ...],
        stage_index: int,
        generator_round: int,
        parameters: FedFSTParameters,
        device: torch.device,
    ) -> ClientHHKRUpload:
        if self._teacher is None:
            raise RuntimeError("FedFST client has not received its teacher.")
        return train_hhkr_generator(
            client_id=self.client_id,
            container=self._strategy_state,
            teacher_model=self._teacher,
            initial_generator_state=initial_generator_state,
            feature_dim=feature_dim,
            conditioning_classes=conditioning_classes,
            stage_index=stage_index,
            generator_round=generator_round,
            parameters=parameters,
            device=device,
            task_class_masks=self._task_class_masks,
        )

    def append_current(self, context: Any) -> None:
        if int(context.client_id) != self.client_id:
            raise ValueError("FedFST workspace received another client's context.")
        append_current_train_history(self._strategy_state, context)

    def commit(self) -> None:
        if self._committed:
            raise RuntimeError("FedFST client workspace was already committed.")
        self._client.state.strategy_state = self._strategy_state
        self._committed = True


def _matched_real_features(
    historical_features: torch.Tensor,
    historical_labels: torch.Tensor,
    generated_labels: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample class-matched rows with one device operation per class."""

    if historical_features.device != historical_labels.device:
        raise ValueError("HHKR historical feature/label devices must match.")
    device = historical_features.device
    generator_device = torch.device(generator.device)
    same_rng_device = generator_device.type == device.type and (
        generator_device.index is None
        or device.index is None
        or generator_device.index == device.index
    )
    if generated_labels.device != device or not same_rng_device:
        raise ValueError("HHKR class matching must use one computation device.")
    matched_indices = torch.empty_like(generated_labels)
    for label in torch.unique(generated_labels).detach().cpu().tolist():
        positions = torch.where(generated_labels == int(label))[0]
        candidates = torch.where(historical_labels == int(label))[0]
        if candidates.numel() == 0:
            raise ValueError("HHKR has no historical real feature for a generated class.")
        choices = torch.randint(
            0,
            int(candidates.numel()),
            (int(positions.numel()),),
            generator=generator,
            device=device,
        )
        matched_indices.index_copy_(0, positions, candidates.index_select(0, choices))
    return historical_features.index_select(0, matched_indices)


def train_hhkr_generator(
    *,
    client_id: int,
    container: dict[str, object],
    teacher_model: torch.nn.Module,
    initial_generator_state: Mapping[str, torch.Tensor],
    feature_dim: int,
    conditioning_classes: tuple[int, ...],
    stage_index: int,
    generator_round: int,
    parameters: FedFSTParameters,
    device: torch.device,
    task_class_masks: Mapping[int, torch.Tensor] | None = None,
) -> ClientHHKRUpload:
    """Train one local HHKR generator without exporting historical tensors."""

    history = get_or_create_history(container, feature_dim)
    node_count = int(history["node_ids"].shape[0])
    if node_count <= 0:
        raise ValueError("HHKR requires non-empty prior train history.")
    historical_features = history["features"].detach().to(device)
    historical_labels = history["labels"].detach().to(device)
    historical_edges = history["edge_index"].detach().cpu()
    resolved_historical_edges = (
        historical_edges
        if task_class_masks is None
        else task_block_diagonal_edges(
            historical_edges, history["labels"], task_class_masks
        )
    )
    classes = tuple(
        int(value) for value in torch.unique(history["labels"]).sort().values.tolist()
    )
    if not set(classes).issubset(conditioning_classes):
        raise ValueError(
            "FedFST client history contains a class outside the immutable "
            "historical vocabulary."
        )
    target_homophily = graph_homophily(
        resolved_historical_edges, history["labels"]
    )
    model = ConditionalFeatureGenerator(
        feature_dim=feature_dim,
        class_ids=conditioning_classes,
        noise_dim=parameters.noise_dim,
        dropout=parameters.generator_dropout,
        initialization_seed=local_generator(
            parameters.method_seed, 10, stage_index
        ).initial_seed(),
    ).to(device)
    if set(model.state_dict()) != set(initial_generator_state):
        raise ValueError("HHKR generator state keys do not match the resolution profile.")
    model.load_state_dict(initial_generator_state, strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=parameters.generator_learning_rate
    )
    device_rng = local_generator(
        parameters.method_seed,
        11,
        stage_index,
        generator_round,
        client_id,
        device=device,
    )
    topology_rng = local_generator(
        parameters.method_seed,
        13,
        stage_index,
        generator_round,
        client_id,
        device=device,
    )
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    loss_trace: list[tuple[float, float, float]] = []
    last_adjustment = None
    for _ in range(parameters.generator_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        generated_features, generated_labels = generate_balanced_features(
            model,
            classes,
            parameters.client_nodes_per_class,
            generator=device_rng,
            device=device,
        )
        if task_class_masks is not None and len(task_class_masks) > 1:
            initial_edges = random_block_diagonal_edges(
                generated_labels.detach(),
                task_class_masks,
                directed_edges_per_node=parameters.generated_edges_per_node,
                generator=topology_rng,
                device=device,
            )
        else:
            initial_edges = random_undirected_edges(
                int(generated_labels.shape[0]),
                directed_edges_per_node=parameters.generated_edges_per_node,
                generator=topology_rng,
                device=device,
            )
        last_adjustment = adjust_homophily(
            initial_edges,
            generated_labels.detach(),
            target=target_homophily,
            reduction_ratio=parameters.edge_reduction_ratio,
            tolerance=parameters.topology_tolerance,
            max_iterations=parameters.topology_max_iterations,
            generator=topology_rng,
        )
        generated_edges = last_adjustment.edge_index
        queries = torch.arange(
            generated_labels.shape[0], dtype=torch.long, device=device
        )
        teacher_logits = teacher_model.forward_queries(
            generated_features, generated_edges, queries, "NC"
        )
        real_features = _matched_real_features(
            historical_features,
            historical_labels,
            generated_labels,
            generator=device_rng,
        )
        losses = hhkr_loss(
            teacher_logits,
            generated_features,
            real_features,
            generated_labels,
            lambda_kl=parameters.lambda_kl,
            task_class_masks=task_class_masks,
        )
        if not torch.isfinite(losses.total):
            raise RuntimeError("HHKR generator loss became non-finite.")
        losses.total.backward()
        optimizer.step()
        loss_trace.append(
            (
                float(losses.total.detach().cpu()),
                float(losses.cross_entropy.detach().cpu()),
                float(losses.feature_kl.detach().cpu()),
            )
        )
    assert last_adjustment is not None
    # Equation 8 aggregates one scalar spectral quantity.  Use the same
    # immutable feature-coordinate sample on every client and on the server;
    # otherwise the scalar average mixes different measured quantities.
    feature_rng = local_generator(parameters.method_seed, 21, stage_index)
    indices = sampled_feature_indices(
        feature_dim, parameters.sampled_feature_fraction, generator=feature_rng
    )
    energy = high_frequency_energy(
        historical_features,
        resolved_historical_edges.to(device),
        feature_indices=indices.to(device),
        compute_device=device,
    )
    state = {
        name: value.detach().cpu().clone().contiguous()
        for name, value in model.state_dict().items()
    }
    history["last_hhkr"] = {
        "stage_index": int(stage_index),
        "generator_round": int(generator_round),
        "historical_train_nodes": node_count,
        "historical_classes": list(classes),
        "spectral_energy": float(energy.total),
        "target_homophily": float(target_homophily),
        "final_generated_homophily": float(last_adjustment.final_value),
        "topology_converged": bool(last_adjustment.converged),
        "selected_feature_indices": list(energy.sampled_feature_indices),
        "loss_trace": [list(values) for values in loss_trace],
    }
    if not math.isfinite(energy.total):
        raise RuntimeError("HHKR client spectral energy became non-finite.")
    return ClientHHKRUpload(
        client_id=int(client_id),
        generator_state=state,
        spectral_energy=float(energy.total),
        historical_node_count=node_count,
    )
