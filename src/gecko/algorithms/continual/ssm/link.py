"""Edge-centred SSM adapter for strict-local LC Class/Task-IL."""

from __future__ import annotations

from gecko.algorithms.continual.ssm.records import SSM_MAIN_HOP_BUDGETS
from gecko.algorithms.continual.ssm.records import SSM_NODE_ONLY_HOP_BUDGETS
from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT

from gecko.algorithms.continual.ssm.records import SSM_MAIN_HOP_BUDGETS
from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT

import hashlib
import math
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Sequence
from typing import Tuple

import torch
from torch import nn

from gecko.algorithms.base import ClientContinualAlgorithm
from gecko.algorithms.continual.ssm.records import SSM_MAIN_HOP_BUDGETS
from gecko.algorithms.continual.ssm.records import SSM_NODE_ONLY_HOP_BUDGETS
from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT
from gecko.algorithms.continual.ssm.sampling import _PreparedSparseComputationSampler
from gecko.algorithms.continual.ssm.node import _class_balanced_loss_terms


SSM_LC_STATE_FORMAT = "uefa-ssm-lc-private-state-v2-task-heads"


def _owned(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone().contiguous()


def _seed(*values: object) -> int:
    raw = "|".join(str(value) for value in ("ssm-lc", *values)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63)


@dataclass(frozen=True, init=False)
class SSMEdgeRecord:
    """One directed replay edge and its joint sparse computation graph."""

    client_id: int
    global_task_id: int
    stage_index: int
    class_id: int
    source_endpoint: int
    destination_endpoint: int
    source_compact: int
    destination_compact: int
    sampler_mode: str
    hop_budgets: Tuple[int, ...]
    record_seed: int
    _features: torch.Tensor
    _edge_index: torch.Tensor
    _source_local_nodes: torch.Tensor
    _class_mask: torch.Tensor

    def __init__(self, **values: object) -> None:
        scalar_names = (
            "client_id", "global_task_id", "stage_index", "class_id",
            "source_endpoint", "destination_endpoint", "source_compact",
            "destination_compact", "record_seed",
        )
        for name in scalar_names:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
            object.__setattr__(self, name, int(value))
        mode = values["sampler_mode"]
        budgets = tuple(values["hop_budgets"])
        if mode not in {"uniform", "degree"} or budgets not in {
            SSM_MAIN_HOP_BUDGETS, SSM_NODE_ONLY_HOP_BUDGETS
        }:
            raise ValueError("Invalid SSM-LC sampler contract.")
        features = _owned(values["features"])
        edges = _owned(values["edge_index"])
        nodes = _owned(values["source_local_nodes"])
        class_mask = _owned(values["class_mask"])
        if features.ndim != 2 or not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("SSM-LC features must be a finite float matrix.")
        if nodes.dtype != torch.long or nodes.ndim != 1 or nodes.numel() != features.shape[0]:
            raise ValueError("SSM-LC source node map is invalid.")
        if len(set(nodes.tolist())) != nodes.numel():
            raise ValueError("SSM-LC compact nodes must be unique.")
        if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("SSM-LC topology must have shape [2,E].")
        if class_mask.dtype != torch.bool or class_mask.ndim != 1:
            raise ValueError("SSM-LC source-task class mask must be one-dimensional bool.")
        if self.class_id >= class_mask.numel() or not bool(class_mask[self.class_id]):
            raise ValueError("SSM-LC source-task class mask excludes the replay label.")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= nodes.numel()):
            raise ValueError("SSM-LC topology has an illegal endpoint.")
        for compact, original in (
            (self.source_compact, self.source_endpoint),
            (self.destination_compact, self.destination_endpoint),
        ):
            if compact >= nodes.numel() or int(nodes[compact]) != original:
                raise ValueError("SSM-LC target endpoint was lost during sparsification.")
        object.__setattr__(self, "sampler_mode", str(mode))
        object.__setattr__(self, "hop_budgets", budgets)
        object.__setattr__(self, "_features", features)
        object.__setattr__(self, "_edge_index", edges)
        object.__setattr__(self, "_source_local_nodes", nodes)
        object.__setattr__(self, "_class_mask", class_mask)

    @property
    def features(self) -> torch.Tensor: return self._features.clone()
    @property
    def edge_index(self) -> torch.Tensor: return self._edge_index.clone()
    @property
    def source_local_nodes(self) -> torch.Tensor: return self._source_local_nodes.clone()
    @property
    def class_mask(self) -> torch.Tensor: return self._class_mask.clone()
    @property
    def target_query(self) -> torch.Tensor:
        return torch.tensor([[self.source_compact, self.destination_compact]], dtype=torch.long)
    @property
    def payload_bytes(self) -> int:
        return int(sum(value.numel() * value.element_size() for value in (
            self._features, self._edge_index, self._source_local_nodes, self._class_mask
        )))
    def to_state(self) -> Dict[str, object]:
        return {name: getattr(self, name) for name in (
            "client_id", "global_task_id", "stage_index", "class_id",
            "source_endpoint", "destination_endpoint", "source_compact",
            "destination_compact", "sampler_mode", "hop_budgets", "record_seed"
        )} | {"features": self.features, "edge_index": self.edge_index,
               "source_local_nodes": self.source_local_nodes,
               "class_mask": self.class_mask}
    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "SSMEdgeRecord":
        return cls(**dict(state))


def sample_edge_record(
    *, node_features: torch.Tensor, edge_index: torch.Tensor,
    source: int, destination: int, label: int, client_id: int,
    global_task_id: int, stage_index: int, sampler_mode: str,
    hop_budgets: Sequence[int], base_seed: int, class_mask: torch.Tensor,
) -> SSMEdgeRecord:
    """Union deterministic endpoint computation graphs and retain direction."""

    if source == destination:
        raise ValueError("SSM-LC target edge endpoints must be distinct.")
    sampler = _PreparedSparseComputationSampler(
        node_features=node_features, edge_index=edge_index
    )
    record_seed = _seed(base_seed, client_id, global_task_id, stage_index, source, destination)
    endpoint_records = []
    for offset, endpoint in enumerate((source, destination)):
        endpoint_records.append(sampler.sample(
            root_index=endpoint, root_label=label, client_id=client_id,
            global_task_id=global_task_id, stage_index=stage_index,
            sampler_mode=sampler_mode, hop_budgets=hop_budgets,
            rng_seed=(record_seed + offset) % (2**63), rng_base_seed=base_seed,
        ))
    selected = sorted(set(endpoint_records[0].source_local_nodes.tolist()) |
                      set(endpoint_records[1].source_local_nodes.tolist()))
    mapping = {node: index for index, node in enumerate(selected)}
    selected_set = set(selected)
    arcs = sorted({(mapping[int(u)], mapping[int(v)]) for u, v in edge_index.t().tolist()
                   if int(u) in selected_set and int(v) in selected_set})
    compact_edges = (torch.tensor(arcs, dtype=torch.long).t().contiguous()
                     if arcs else torch.empty((2, 0), dtype=torch.long))
    nodes = torch.tensor(selected, dtype=torch.long)
    return SSMEdgeRecord(
        client_id=client_id, global_task_id=global_task_id, stage_index=stage_index,
        class_id=label, source_endpoint=source, destination_endpoint=destination,
        source_compact=mapping[source], destination_compact=mapping[destination],
        sampler_mode=sampler_mode, hop_budgets=tuple(hop_budgets),
        record_seed=record_seed, features=node_features.detach().cpu()[nodes],
        edge_index=compact_edges, source_local_nodes=nodes, class_mask=class_mask,
    )


class SSMLCAlgorithm(ClientContinualAlgorithm):
    """Paper-mechanism SSM adaptation whose replay targets are directed edges."""

    name = "SSM"
    method_version = "uefa-ssm-lc-v2-source-task-heads"

    def __init__(self, *, sampler_mode: str = "degree",
                 hop_budgets: Sequence[int] = SSM_MAIN_HOP_BUDGETS,
                 replay_weight: float = 1.0,
                 replay_ceiling_bytes: int = SSM_REPLAY_CEILING_BYTES,
                 stage_count: int = SSM_STAGE_COUNT, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.problem_type != "LC" or self.incremental_setting not in {"class", "task"}:
            raise ValueError("SSM-LC supports only LC-Class/LC-Task.")
        if self.client_id is None or self.client_id < 0:
            raise ValueError("SSM-LC requires a client_id.")
        if sampler_mode not in {"uniform", "degree"}:
            raise ValueError("Invalid sampler_mode.")
        if tuple(hop_budgets) not in {SSM_MAIN_HOP_BUDGETS, SSM_NODE_ONLY_HOP_BUDGETS}:
            raise ValueError("Invalid fixed SSM hop budgets.")
        if float(replay_weight) != 1.0 or replay_ceiling_bytes != SSM_REPLAY_CEILING_BYTES:
            raise ValueError("SSM-LC preserves the fixed NC replay policy.")
        if stage_count != SSM_STAGE_COUNT:
            raise ValueError("SSM-LC preserves the eight-stage allocation.")
        self.sampler_mode, self.hop_budgets = sampler_mode, tuple(hop_budgets)
        self.replay_ceiling_bytes, self.stage_count = replay_ceiling_bytes, stage_count
        self.state.update({"format": SSM_LC_STATE_FORMAT, "records": [], "allocations": {},
                           "diagnostics": {"current_loss": 0.0, "replay_loss": 0.0}})

    def _records(self) -> Tuple[SSMEdgeRecord, ...]:
        records = tuple(SSMEdgeRecord.from_state(item) for item in self.state["records"])
        if sum(item.payload_bytes for item in records) > self.replay_ceiling_bytes:
            raise ValueError("SSM-LC byte ceiling exceeded.")
        return records

    def _validate(self, context: Any) -> None:
        if context.problem_type != "LC" or context.incremental_setting not in {"class", "task"}:
            raise ValueError("SSM-LC received a non LC-Class/LC-Task context.")
        if context.client_id != self.client_id or context.stage_index >= self.stage_count:
            raise ValueError("SSM-LC context identity/stage mismatch.")
        if context.valid_class_mask is None:
            raise ValueError("SSM-LC requires the accumulated class mask.")

    def replay_samples(self, context: Any) -> Tuple[SSMEdgeRecord, ...]:
        self._validate(context)
        return self._records()

    def augment_loss(self, model: nn.Module, context: Any, logits: torch.Tensor,
                     base_loss: torch.Tensor) -> torch.Tensor:
        del base_loss
        self._validate(context)
        records = self._records()
        counts: Dict[int, int] = {}
        for label in context.train_labels.tolist(): counts[int(label)] = counts.get(int(label), 0) + 1
        for record in records: counts[record.class_id] = counts.get(record.class_id, 0) + 1
        current_mask = context.valid_class_mask
        current_num, current_den = _class_balanced_loss_terms(
            logits, labels=context.train_labels, valid_class_mask=current_mask,
            class_counts=counts)
        replay_num, replay_den = logits.sum() * 0.0, logits.new_zeros(())
        device = next(model.parameters()).device
        grouped: Dict[int, list[SSMEdgeRecord]] = {}
        for record in records:
            grouped.setdefault(record.global_task_id, []).append(record)
        for task_id, task_records in sorted(grouped.items()):
            source_mask = task_records[0].class_mask
            if any(not torch.equal(record.class_mask, source_mask)
                   for record in task_records[1:]):
                raise ValueError(f"SSM-LC task {task_id} has inconsistent source-task heads.")
            replay_mask = (
                source_mask if self.incremental_setting == "task" else current_mask
            )
            for record in task_records:
                values = context.forward_queries(model, record.target_query.to(device),
                    node_features=record.features.to(device), edge_index=record.edge_index.to(device))
                num, den = _class_balanced_loss_terms(values,
                    labels=torch.tensor([record.class_id], device=device),
                    valid_class_mask=replay_mask, class_counts=counts)
                replay_num, replay_den = replay_num + num, replay_den + den
        current_loss = current_num / current_den
        replay_loss = replay_num / replay_den if records else logits.sum() * 0.0
        self.state["diagnostics"] = {"current_loss": float(current_loss.detach().cpu()),
                                     "replay_loss": float(replay_loss.detach().cpu())}
        return (current_num + replay_num) / (current_den + replay_den)

    def consolidate(self, model: nn.Module, context: Any) -> None:
        del model
        self._validate(context)
        queries, labels = context.train_queries.cpu(), context.train_labels.cpu()
        if queries.ndim != 2 or queries.shape[1] != 2 or labels.ndim != 1:
            raise ValueError("SSM-LC requires directed edge queries and scalar labels.")
        candidates = [sample_edge_record(
            node_features=context.node_features, edge_index=context.base_edge_index,
            source=int(pair[0]), destination=int(pair[1]), label=int(label),
            client_id=self.client_id, global_task_id=context.global_task_id,
            stage_index=context.stage_index, sampler_mode=self.sampler_mode,
            hop_budgets=self.hop_budgets, base_seed=self.seed,
            class_mask=context.valid_class_mask,
        ) for pair, label in zip(queries.tolist(), labels.tolist())]
        stage_budget = self.replay_ceiling_bytes // self.stage_count
        classes = sorted(set(labels.tolist()))
        quota = stage_budget // len(classes)
        selected = []
        allocation = {}
        for class_id in classes:
            used = 0
            group = sorted((item for item in candidates if item.class_id == class_id),
                           key=lambda item: (item.source_endpoint, item.destination_endpoint))
            for item in group:
                if used + item.payload_bytes <= quota:
                    selected.append(item); used += item.payload_bytes
            allocation[str(class_id)] = {"candidate_count": len(group),
                                         "inserted_count": sum(x.class_id == class_id for x in selected),
                                         "used_bytes": used, "quota_bytes": quota}
        self.state["records"].extend(item.to_state() for item in selected)
        self.state["allocations"][str(context.stage_index)] = allocation
        if self.replay_payload_bytes() > self.replay_ceiling_bytes:
            raise MemoryError("SSM-LC exceeded its replay ceiling.")

    def replay_payload_bytes(self) -> int:
        return sum(item.payload_bytes for item in self._records())

    def diagnostics(self) -> Dict[str, object]:
        records = self._records()
        per_stage = {}
        for stage in sorted({r.stage_index for r in records}):
            selected = [r for r in records if r.stage_index == stage]
            nodes = sum(r.features.shape[0] for r in selected)
            edges = sum(r.edge_index.shape[1] for r in selected)
            per_stage[str(stage)] = {
                "global_task_id": selected[0].global_task_id,
                "source_task_active_classes":
                    selected[0].class_mask.nonzero().reshape(-1).tolist(),
                "replay_target_edges": len(selected), "stored_nodes": nodes,
                "stored_graph_edges": edges,
                "average_subgraph_nodes": nodes / len(selected),
                "average_subgraph_edges": edges / len(selected),
                "payload_bytes": sum(r.payload_bytes for r in selected),
            }
        return {**self.state["diagnostics"], "method": self.name,
                "method_version": self.method_version, "replay_targets": len(records),
                "replay_nodes": sum(item.features.shape[0] for item in records),
                "replay_edges": sum(item.edge_index.shape[1] for item in records),
                "directed_target_edges": [[r.source_endpoint, r.destination_endpoint] for r in records],
                "class_representation": {str(c): sum(r.class_id == c for r in records)
                                         for c in sorted({r.class_id for r in records})},
                "replay_payload_bytes": self.replay_payload_bytes(),
                "replay_ceiling_bytes": self.replay_ceiling_bytes,
                "per_stage": per_stage,
                "stage_allocations": self.state["allocations"]}

    def load_method_state(self, state: Mapping[str, object]) -> None:
        previous = self.save_method_state()
        try:
            super().load_method_state(state)
            if set(self.state) != {"format", "records", "allocations", "diagnostics"} or self.state["format"] != SSM_LC_STATE_FORMAT:
                raise ValueError("Invalid SSM-LC checkpoint schema.")
            self._records()
        except Exception:
            super().load_method_state(previous)
            raise


__all__ = ["SSMEdgeRecord", "SSMLCAlgorithm", "sample_edge_record"]
