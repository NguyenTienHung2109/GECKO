"""Edge-level CaT condensation for LC Class/Task-IL."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from gecko.algorithms.base import ClientContinualAlgorithm
from gecko.algorithms.continual.cat.node import CAT_CONDENSATION_LOSS_SEMANTICS
from gecko.algorithms.continual.cat.node import _derive_condensation_seed
from gecko.algorithms.continual.cat.node import _distribution_matching_loss
from gecko.algorithms.continual.cat.node import _masked_class_logits
from gecko.algorithms.continual.cat.node import _reset_random_encoder


CAT_LC_STATE_FORMAT = "uefa-cat-lc-private-state-v3-task-heads"


def _owned(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone().contiguous()


@dataclass(frozen=True, init=False)
class CondensedEdgeGraphRecord:
    """Synthetic graph and its directed labelled LC query edges."""

    client_id: int
    global_task_id: int
    stage_index: int
    original_node_count: int
    original_query_count: int
    condensation_steps: int
    condensation_seed: int
    condensation_seconds: float
    initial_loss: float
    final_loss: float
    _features: torch.Tensor
    _edge_index: torch.Tensor
    _query_edges: torch.Tensor
    _labels: torch.Tensor
    _class_mask: torch.Tensor

    def __init__(self, **values: object) -> None:
        for name in ("client_id", "global_task_id", "stage_index", "original_node_count",
                     "original_query_count", "condensation_steps", "condensation_seed"):
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, int(value))
        for name in ("condensation_seconds", "initial_loss", "final_loss"):
            value = float(values[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        features, edges = _owned(values["features"]), _owned(values["edge_index"])
        queries, labels = _owned(values["query_edges"]), _owned(values["labels"])
        class_mask = _owned(values["class_mask"])
        if features.ndim != 2 or not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("CaT-LC features must be a finite float matrix.")
        for name, tensor in (("topology", edges), ("queries", queries)):
            if tensor.dtype != torch.long or tensor.ndim != 2 or tensor.shape[0 if name == "topology" else 1] != 2:
                raise ValueError(f"CaT-LC {name} shape is invalid.")
            if tensor.numel() and (int(tensor.min()) < 0 or int(tensor.max()) >= features.shape[0]):
                raise ValueError(f"CaT-LC {name} has an illegal endpoint.")
        if labels.dtype != torch.long or labels.ndim != 1 or labels.numel() != queries.shape[0]:
            raise ValueError("CaT-LC query labels are invalid.")
        if class_mask.dtype != torch.bool or class_mask.ndim != 1:
            raise ValueError("CaT-LC source-task class mask must be one-dimensional bool.")
        if labels.numel() and (
            int(labels.max()) >= class_mask.numel()
            or not bool(class_mask[labels].all())
        ):
            raise ValueError("CaT-LC source-task class mask excludes a condensed label.")
        if queries.shape[0] >= self.original_query_count:
            raise ValueError("CaT-LC memory must contain fewer queries than the original task.")
        object.__setattr__(self, "_features", features)
        object.__setattr__(self, "_edge_index", edges)
        object.__setattr__(self, "_query_edges", queries)
        object.__setattr__(self, "_labels", labels)
        object.__setattr__(self, "_class_mask", class_mask)

    @property
    def features(self) -> torch.Tensor: return self._features.clone()
    @property
    def edge_index(self) -> torch.Tensor: return self._edge_index.clone()
    @property
    def query_edges(self) -> torch.Tensor: return self._query_edges.clone()
    @property
    def labels(self) -> torch.Tensor: return self._labels.clone()
    @property
    def class_mask(self) -> torch.Tensor: return self._class_mask.clone()
    @property
    def payload_bytes(self) -> int:
        return int(sum(x.numel() * x.element_size() for x in
                       (self._features, self._edge_index, self._query_edges,
                        self._labels, self._class_mask)))
    def to_state(self) -> Dict[str, object]:
        names = ("client_id", "global_task_id", "stage_index", "original_node_count",
                 "original_query_count", "condensation_steps", "condensation_seed",
                 "condensation_seconds", "initial_loss", "final_loss")
        return {name: getattr(self, name) for name in names} | {
            "features": self.features, "edge_index": self.edge_index,
            "query_edges": self.query_edges, "labels": self.labels,
            "class_mask": self.class_mask}
    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "CondensedEdgeGraphRecord":
        return cls(**dict(state))


class CaTLCAlgorithm(ClientContinualAlgorithm):
    """CaT whose balanced condensed-memory samples are labelled edges."""

    name = "CaT"
    method_version = "uefa-cat-lc-v3-source-task-heads"

    def __init__(self, *, synthetic_nodes_per_class: int = 2,
                 condensation_steps: int = 32, condensation_lr: float = 1e-3,
                 memory_ceiling_bytes: int = 16 * 1024 * 1024,
                 stage_count: int = 8, feature_initialization: str = "random_choice",
                 **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.problem_type != "LC" or self.incremental_setting not in {"class", "task"}:
            raise ValueError("CaT-LC supports only LC-Class/LC-Task.")
        if self.client_id is None or self.client_id < 0:
            raise ValueError("CaT-LC requires a client_id.")
        for name, value in (("synthetic_nodes_per_class", synthetic_nodes_per_class),
                            ("condensation_steps", condensation_steps),
                            ("memory_ceiling_bytes", memory_ceiling_bytes),
                            ("stage_count", stage_count)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not math.isfinite(float(condensation_lr)) or float(condensation_lr) <= 0:
            raise ValueError("condensation_lr must be positive.")
        if feature_initialization != "random_choice":
            raise ValueError("CaT-LC uses the official randomChoice initializer.")
        self.synthetic_edges_per_class = synthetic_nodes_per_class
        self.condensation_steps, self.condensation_lr = condensation_steps, float(condensation_lr)
        self.memory_ceiling_bytes, self.stage_count = memory_ceiling_bytes, stage_count
        self.feature_initialization = feature_initialization
        self.state.update({"format": CAT_LC_STATE_FORMAT, "records": [],
                           "diagnostics": {"current_raw_loss": 0.0,
                           "training_in_memory_loss": 0.0,
                           "last_condensation_initial_loss": 0.0,
                           "last_condensation_final_loss": 0.0,
                           "last_condensation_seconds": 0.0,
                           "condensation_loss_semantics":
                           CAT_CONDENSATION_LOSS_SEMANTICS}})

    def _validate(self, context: Any) -> None:
        if context.problem_type != "LC" or context.incremental_setting not in {"class", "task"}:
            raise ValueError("CaT-LC received a non LC-Class/LC-Task context.")
        if context.client_id != self.client_id or context.stage_index >= self.stage_count:
            raise ValueError("CaT-LC context identity/stage mismatch.")
        if context.valid_class_mask is None:
            raise ValueError("CaT-LC requires an accumulated class mask.")

    def _records(self) -> Tuple[CondensedEdgeGraphRecord, ...]:
        records = tuple(CondensedEdgeGraphRecord.from_state(x) for x in self.state["records"])
        if sum(x.payload_bytes for x in records) > self.memory_ceiling_bytes:
            raise ValueError("CaT-LC memory ceiling exceeded.")
        return records

    def _initialize(self, context: Any, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries, labels = context.train_queries.cpu(), context.train_labels.cpu()
        features = context.node_features.cpu()
        generator = torch.Generator().manual_seed(seed)
        rows, synthetic_labels = [], []
        for class_id in torch.unique(labels, sorted=True).tolist():
            candidates = (labels == class_id).nonzero().reshape(-1)
            draws = candidates[torch.randint(candidates.numel(),
                (self.synthetic_edges_per_class,), generator=generator)]
            for query_index in draws.tolist():
                source, destination = queries[query_index].tolist()
                rows.extend((features[source], features[destination]))
                synthetic_labels.append(class_id)
        synthetic_features = torch.stack(rows)
        indices = torch.arange(len(synthetic_labels), dtype=torch.long)
        synthetic_queries = torch.stack((2 * indices, 2 * indices + 1), dim=1)
        return synthetic_features, synthetic_queries, torch.tensor(synthetic_labels, dtype=torch.long)

    def _condense(self, model: nn.Module, context: Any) -> CondensedEdgeGraphRecord:
        self._validate(context)
        queries, labels = context.train_queries.cpu(), context.train_labels.cpu()
        if queries.ndim != 2 or queries.shape[1] != 2 or labels.ndim != 1 or labels.numel() == 0:
            raise ValueError("CaT-LC requires aligned directed edge queries.")
        synthetic_count = torch.unique(labels).numel() * self.synthetic_edges_per_class
        if synthetic_count >= queries.shape[0]:
            raise ValueError("CaT-LC condensed query count must be smaller than current data.")
        seed = _derive_condensation_seed(base_seed=self.seed, client_id=self.client_id,
            global_task_id=context.global_task_id, stage_index=context.stage_index)
        initial, synthetic_queries, synthetic_labels = self._initialize(context, seed)
        device = next(model.parameters()).device
        synthetic_features = nn.Parameter(initial.to(device))
        synthetic_queries, synthetic_labels = synthetic_queries.to(device), synthetic_labels.to(device)
        nodes = torch.arange(initial.shape[0], device=device)
        synthetic_topology = torch.stack((nodes, nodes))
        optimizer = torch.optim.Adam([synthetic_features], lr=self.condensation_lr)
        first = last = 0.0
        started = time.perf_counter()
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            for step in range(self.condensation_steps):
                torch.manual_seed((seed + step) % (2**63))
                if device.type == "cuda": torch.cuda.manual_seed_all((seed + step) % (2**63))
                encoder = copy.deepcopy(model).to(device)
                _reset_random_encoder(encoder)
                encoder.eval()
                for parameter in encoder.parameters(): parameter.requires_grad_(False)
                with torch.no_grad():
                    real_logits = context.forward_queries(encoder, queries.to(device))
                synthetic_logits = context.forward_queries(encoder, synthetic_queries,
                    node_features=synthetic_features, edge_index=synthetic_topology)
                loss = _distribution_matching_loss(
                    F.normalize(real_logits, dim=-1), labels.to(device),
                    F.normalize(synthetic_logits, dim=-1), synthetic_labels)
                if step == 0: first = float(loss.detach().cpu())
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            # Use the exact step-0 encoder for the post-optimization audit;
            # losses from independently redrawn encoders are not comparable.
            del loss, synthetic_logits, real_logits, encoder
            torch.manual_seed(seed)
            if device.type == "cuda": torch.cuda.manual_seed_all(seed)
            audit_encoder = copy.deepcopy(model).to(device)
            _reset_random_encoder(audit_encoder)
            audit_encoder.eval()
            for parameter in audit_encoder.parameters(): parameter.requires_grad_(False)
            with torch.no_grad():
                audit_real = context.forward_queries(
                    audit_encoder, queries.to(device)
                )
                audit_synthetic = context.forward_queries(
                    audit_encoder, synthetic_queries,
                    node_features=synthetic_features,
                    edge_index=synthetic_topology,
                )
                audit_loss = _distribution_matching_loss(
                    F.normalize(audit_real, dim=-1), labels.to(device),
                    F.normalize(audit_synthetic, dim=-1), synthetic_labels,
                )
                if not torch.isfinite(audit_loss):
                    raise RuntimeError(
                        "CaT-LC fixed-objective audit produced a non-finite loss."
                    )
                last = float(audit_loss.cpu())
        visible = torch.unique(torch.cat((queries.reshape(-1), context.effective_edge_index.cpu().reshape(-1))))
        return CondensedEdgeGraphRecord(client_id=self.client_id,
            global_task_id=context.global_task_id, stage_index=context.stage_index,
            original_node_count=int(visible.numel()), original_query_count=queries.shape[0],
            condensation_steps=self.condensation_steps, condensation_seed=seed,
            condensation_seconds=time.perf_counter() - started, initial_loss=first, final_loss=last,
            features=synthetic_features.detach(), edge_index=synthetic_topology.cpu(),
            query_edges=synthetic_queries.cpu(), labels=synthetic_labels.cpu(),
            class_mask=context.valid_class_mask)

    def _ensure(self, model: nn.Module, context: Any) -> None:
        if any(r.global_task_id == context.global_task_id for r in self._records()): return
        record = self._condense(model, context)
        if self.replay_payload_bytes() + record.payload_bytes > self.memory_ceiling_bytes:
            raise MemoryError("CaT-LC memory ceiling would be exceeded.")
        self.state["records"].append(record.to_state())
        diagnostics = self.state["diagnostics"]
        diagnostics["last_condensation_initial_loss"] = record.initial_loss
        diagnostics["last_condensation_final_loss"] = record.final_loss
        diagnostics["last_condensation_seconds"] = record.condensation_seconds
        diagnostics["condensation_loss_semantics"] = (
            CAT_CONDENSATION_LOSS_SEMANTICS
        )

    def _memory_loss(self, model: nn.Module, context: Any) -> torch.Tensor:
        device = next(model.parameters()).device
        losses = []
        grouped: Dict[int, list[CondensedEdgeGraphRecord]] = {}
        for record in self._records():
            grouped.setdefault(record.global_task_id, []).append(record)
        for task_id, records in sorted(grouped.items()):
            source_mask = records[0].class_mask
            if any(not torch.equal(record.class_mask, source_mask) for record in records[1:]):
                raise ValueError(f"CaT-LC task {task_id} has inconsistent source-task heads.")
            replay_mask = (
                source_mask
                if self.incremental_setting == "task"
                else context.valid_class_mask
            )
            assert replay_mask is not None
            for record in records:
                if record.stage_index > context.stage_index:
                    raise ValueError("CaT-LC future memory leakage.")
                logits = context.forward_queries(model, record.query_edges.to(device),
                    node_features=record.features.to(device), edge_index=record.edge_index.to(device))
                losses.append(F.cross_entropy(
                    _masked_class_logits(logits, replay_mask), record.labels.to(device)))
        return torch.stack(losses).mean()

    def augment_loss(self, model: nn.Module, context: Any, logits: torch.Tensor,
                     base_loss: torch.Tensor) -> torch.Tensor:
        self._validate(context); self._ensure(model, context)
        memory_loss = self._memory_loss(model, context)
        self.state["diagnostics"]["current_raw_loss"] = float(base_loss.detach().cpu())
        self.state["diagnostics"]["training_in_memory_loss"] = float(memory_loss.detach().cpu())
        return memory_loss + logits.sum() * 0.0

    def consolidate(self, model: nn.Module, context: Any) -> None:
        del model
        if not any(r.global_task_id == context.global_task_id for r in self._records()):
            raise RuntimeError("CaT-LC stage completed without condensation.")

    def replay_samples(self, context: Any) -> Tuple[CondensedEdgeGraphRecord, ...]:
        self._validate(context); return self._records()

    def replay_payload_bytes(self) -> int:
        return sum(r.payload_bytes for r in self._records())

    def diagnostics(self) -> Dict[str, object]:
        records = self._records()
        per_task = {str(r.global_task_id): {
            "stage_index": r.stage_index, "original_nodes": r.original_node_count,
            "original_target_edges": r.original_query_count,
            "synthetic_nodes": r.features.shape[0],
            "synthetic_graph_edges": r.edge_index.shape[1],
            "synthetic_target_edges": r.query_edges.shape[0],
            "synthetic_target_edges_per_class": {
                str(c): int((r.labels == c).sum()) for c in torch.unique(r.labels).tolist()},
            "source_task_active_classes": r.class_mask.nonzero().reshape(-1).tolist(),
            "reduction_ratio": r.query_edges.shape[0] / r.original_query_count,
            "initial_condensation_loss": r.initial_loss,
            "final_condensation_loss": r.final_loss,
            "condensation_loss_semantics": CAT_CONDENSATION_LOSS_SEMANTICS,
            "condensation_objective_delta": r.final_loss - r.initial_loss,
            "condensation_seconds": r.condensation_seconds,
        } for r in records}
        return {**self.state["diagnostics"], "method": self.name,
            "method_version": self.method_version, "condensed_tasks": len(records),
            "synthetic_nodes": sum(r.features.shape[0] for r in records),
            "synthetic_query_edges": sum(r.query_edges.shape[0] for r in records),
            "synthetic_topology_edges": sum(r.edge_index.shape[1] for r in records),
            "original_queries": sum(r.original_query_count for r in records),
            "per_task": per_task,
            "memory_payload_bytes": self.replay_payload_bytes(),
            "memory_ceiling_bytes": self.memory_ceiling_bytes}

    def load_method_state(self, state: Mapping[str, object]) -> None:
        previous = self.save_method_state()
        try:
            super().load_method_state(state)
            if set(self.state) != {"format", "records", "diagnostics"} or self.state["format"] != CAT_LC_STATE_FORMAT:
                raise ValueError("Invalid CaT-LC checkpoint schema.")
            self._records()
            diagnostics = self.state["diagnostics"]
            expected = {
                "current_raw_loss", "training_in_memory_loss",
                "last_condensation_initial_loss", "last_condensation_final_loss",
                "last_condensation_seconds", "condensation_loss_semantics",
            }
            if (
                not isinstance(diagnostics, dict)
                or set(diagnostics) != expected
                or diagnostics["condensation_loss_semantics"]
                != CAT_CONDENSATION_LOSS_SEMANTICS
            ):
                raise ValueError("Invalid CaT-LC diagnostic semantics.")
        except Exception:
            super().load_method_state(previous); raise


__all__ = ["CaTLCAlgorithm", "CondensedEdgeGraphRecord"]
