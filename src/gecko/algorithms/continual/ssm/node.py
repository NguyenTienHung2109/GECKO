from __future__ import annotations

from gecko.algorithms.continual.ssm.records import SSM_MAIN_HOP_BUDGETS
from gecko.algorithms.continual.ssm.records import SSM_NODE_ONLY_HOP_BUDGETS
from gecko.algorithms.continual.ssm.records import SSM_RECORD_FORMAT
from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT
from gecko.algorithms.continual.ssm.records import SSM_STATE_FORMAT

from gecko.algorithms.continual.ssm.records import SSM_MAIN_HOP_BUDGETS
from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT

import hashlib
import math
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Sequence
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from gecko.algorithms.base import ClientContinualAlgorithm

def _derive_record_seed(
    *,
    base_seed: int,
    client_id: int,
    global_task_id: int,
    stage_index: int,
    root_index: int,
    sampler_mode: str,
    hop_budgets: Sequence[int],
) -> int:
    from gecko.algorithms.continual.ssm.records import SSM_RECORD_FORMAT
    digest = hashlib.sha256()
    for value in (
        SSM_RECORD_FORMAT,
        str(base_seed),
        str(client_id),
        str(global_task_id),
        str(stage_index),
        str(root_index),
        sampler_mode,
        ",".join(str(value) for value in hop_budgets),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big") % (2**63)


def _model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(model.parameters(), None)
    return fallback if parameter is None else parameter.device


def _class_balanced_loss_terms(
    logits: torch.Tensor,
    *,
    labels: torch.Tensor,
    valid_class_mask: torch.Tensor,
    class_counts: Mapping[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Eq.-(7) inverse-frequency numerator and weight denominator."""

    if logits.ndim != 2 or labels.dtype != torch.long or labels.ndim != 1:
        raise ValueError("SSM class-balanced NC logits/labels have invalid shapes.")
    if logits.shape[0] != labels.shape[0] or labels.numel() == 0:
        raise ValueError("SSM class-balanced logits and root labels must align.")
    mask = valid_class_mask.to(device=logits.device)
    if mask.dtype != torch.bool or mask.ndim != 1 or mask.numel() != logits.shape[1]:
        raise ValueError("SSM valid_class_mask does not match NC logits.")
    device_labels = labels.to(device=logits.device)
    if int(device_labels.min()) < 0 or int(device_labels.max()) >= logits.shape[1]:
        raise ValueError("SSM root label is outside the model output.")
    if not bool(mask[device_labels].all()):
        raise ValueError("SSM root label is excluded by valid_class_mask.")
    active = mask.nonzero(as_tuple=False).reshape(-1)
    remapping = torch.full(
        (logits.shape[1],), -1, dtype=torch.long, device=logits.device
    )
    remapping[active] = torch.arange(active.numel(), device=logits.device)
    mapped_labels = remapping[device_labels]
    unweighted = F.cross_entropy(logits[:, mask], mapped_labels, reduction="none")
    weights = []
    for label in labels.detach().cpu().tolist():
        count = class_counts.get(int(label), 0)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("SSM class-balance counts must be positive integers.")
        weights.append(1.0 / count)
    sample_weights = logits.new_tensor(weights)
    return (unweighted * sample_weights).sum(), sample_weights.sum()


class SSMAlgorithm(ClientContinualAlgorithm):
    """Private NC-Class SSM replay state with the fixed UEFA memory policy."""

    name = "SSM"
    method_version = "uefa-ssm-v2"

    def __init__(
        self,
        *,
        sampler_mode: str = "degree",
        hop_budgets: Sequence[int] = SSM_MAIN_HOP_BUDGETS,
        replay_weight: float = 1.0,
        replay_ceiling_bytes: int = SSM_REPLAY_CEILING_BYTES,
        stage_count: int = SSM_STAGE_COUNT,
        **kwargs: object,
    ) -> None:
        from gecko.algorithms.continual.ssm.replay import SSMReplayStore
        from gecko.algorithms.continual.ssm.records import SSM_MAIN_HOP_BUDGETS
        from gecko.algorithms.continual.ssm.records import SSM_NODE_ONLY_HOP_BUDGETS
        from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
        from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT
        from gecko.algorithms.continual.ssm.records import SSM_STATE_FORMAT
        from gecko.algorithms.continual.ssm.sampling import _PreparedSparseComputationSampler
        super().__init__(**kwargs)
        if self.client_id is None or self.client_id < 0:
            raise ValueError("SSM requires an explicit non-negative client_id.")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not (0 <= self.seed < 2**63)
        ):
            raise ValueError("SSM seed must be a non-negative signed 63-bit integer.")
        if self.problem_type != "NC" or self.incremental_setting not in {"class", "task"}:
            raise ValueError("SSM supports only NC-Class/NC-Task and fails closed.")
        if sampler_mode not in {"uniform", "degree"}:
            raise ValueError("sampler_mode must be 'uniform' or 'degree'.")
        budgets = tuple(hop_budgets)
        if budgets not in {SSM_MAIN_HOP_BUDGETS, SSM_NODE_ONLY_HOP_BUDGETS}:
            raise ValueError(
                "UEFA SSM permits only [10, 25] or the named [0, 0] diagnostic."
            )
        if not math.isfinite(float(replay_weight)) or float(replay_weight) != 1.0:
            raise ValueError(
                "UEFA SSM fixes replay_weight=1.0; it is not a validation axis."
            )
        if (
            isinstance(replay_ceiling_bytes, bool)
            or not isinstance(replay_ceiling_bytes, int)
            or replay_ceiling_bytes != SSM_REPLAY_CEILING_BYTES
        ):
            raise ValueError("UEFA SSM fixes replay_ceiling_bytes at 16 MiB.")
        if (
            isinstance(stage_count, bool)
            or not isinstance(stage_count, int)
            or stage_count != SSM_STAGE_COUNT
        ):
            raise ValueError("UEFA SSM fixes stage_count at eight immutable stages.")
        self.sampler_mode = sampler_mode
        self.hop_budgets = budgets
        self.replay_weight = float(replay_weight)
        store = SSMReplayStore(client_id=self.client_id)
        self.state.update(
            {
                "format": SSM_STATE_FORMAT,
                "method_hyperparameters": self._private_hyperparameters(),
                "replay_store": store.to_state(),
                "consolidated_task_ids": [],
                "task_class_masks": {},
                "diagnostics": {
                    "current_loss": 0.0,
                    "replay_loss": 0.0,
                    "last_inserted_records": 0,
                    "last_rejected_records": 0,
                    "class_balance_counts": {},
                    "class_balance_weights": {},
                },
            }
        )
        self._store_cache: SSMReplayStore | None = store
        self._store_cache_source_id = id(self.state["replay_store"])
        self._sampler_cache_key: tuple[object, ...] | None = None
        self._sampler_cache: _PreparedSparseComputationSampler | None = None
        self._device_record_cache: Dict[
            tuple[str, str],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    def _private_hyperparameters(self) -> Dict[str, object]:
        from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
        from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT
        return {
            "sampler_mode": self.sampler_mode,
            "hop_budgets": list(self.hop_budgets),
            "replay_weight": self.replay_weight,
            "replay_ceiling_bytes": SSM_REPLAY_CEILING_BYTES,
            "stage_count": SSM_STAGE_COUNT,
        }

    def hyperparameters(self) -> Dict[str, object]:
        values = super().hyperparameters()
        values.update(self._private_hyperparameters())
        return values

    def _diagnostic_state(self) -> Dict[str, object]:
        value = self.state.get("diagnostics")
        if not isinstance(value, dict):
            raise RuntimeError("SSM checkpointed diagnostics are malformed.")
        return value

    def _validate_record_provenance(self, records: Sequence[SSMRecord]) -> None:
        from gecko.algorithms.continual.ssm.records import SSMRecord
        for record in records:
            expected_seed = _derive_record_seed(
                base_seed=self.seed,
                client_id=self.client_id,
                global_task_id=record.global_task_id,
                stage_index=record.stage_index,
                root_index=record.source_root_index,
                sampler_mode=self.sampler_mode,
                hop_budgets=self.hop_budgets,
            )
            if (
                record.sampler_mode != self.sampler_mode
                or record.hop_budgets != self.hop_budgets
                or record.rng_algorithm != "torch.Generator.cpu.manual_seed"
                or record.rng_base_seed != self.seed
                or record.rng_record_seed != expected_seed
                or record.sampled_node_count
                != int(record.source_local_nodes.numel() - 1)
            ):
                raise ValueError(
                    "SSM replay record provenance does not match the active method."
                )

    def _store(self) -> SSMReplayStore:
        from gecko.algorithms.continual.ssm.replay import SSMReplayStore
        value = self.state.get("replay_store")
        if not isinstance(value, Mapping):
            raise RuntimeError("SSM private replay store is missing.")
        if (
            self._store_cache is not None
            and self._store_cache_source_id == id(value)
        ):
            return self._store_cache
        store = SSMReplayStore.from_state(value)
        self._validate_record_provenance(store.records())
        self._store_cache = store
        self._store_cache_source_id = id(value)
        return store

    def _prepared_sampler(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> _PreparedSparseComputationSampler:
        from gecko.algorithms.continual.ssm.sampling import _PreparedSparseComputationSampler
        key = (
            id(node_features),
            int(node_features._version),
            tuple(node_features.shape),
            id(edge_index),
            int(edge_index._version),
            tuple(edge_index.shape),
        )
        if self._sampler_cache is None or self._sampler_cache_key != key:
            self._sampler_cache = _PreparedSparseComputationSampler(
                node_features=node_features, edge_index=edge_index
            )
            self._sampler_cache_key = key
        return self._sampler_cache

    def replay_samples(self, context: Any) -> Tuple[SSMRecord, ...]:
        from gecko.algorithms.continual.ssm.records import SSMRecord
        self._validate_context(context)
        return self._store().records()

    def _forward_replay_queries(
        self,
        model: nn.Module,
        context: Any,
        query: torch.Tensor,
        *,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        was_training = bool(model.training)
        if was_training:
            model.eval()
        try:
            return context.forward_queries(
                model,
                query,
                node_features=node_features,
                edge_index=edge_index,
            )
        finally:
            if was_training:
                model.train()

    def _device_record_tensors(
        self, record: SSMRecord, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from gecko.algorithms.continual.ssm.records import SSMRecord
        key = (record.record_id, str(device))
        cached = self._device_record_cache.get(key)
        if cached is None:
            cached = (
                record.features.to(device),
                record.edge_index.to(device),
                torch.tensor([record.root_index], dtype=torch.long, device=device),
                torch.tensor([record.root_label], dtype=torch.long, device=device),
            )
            self._device_record_cache[key] = cached
        return cached

    def _validate_context(self, context: Any) -> None:
        from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT
        if (
            str(context.problem_type).upper() != "NC"
            or str(context.incremental_setting).lower() not in {"class", "task"}
        ):
            raise ValueError("SSM supports only NC-Class/NC-Task contexts.")
        if int(context.client_id) != self.client_id:
            raise ValueError("SSM context belongs to another client.")
        if int(context.stage_index) >= SSM_STAGE_COUNT:
            raise ValueError("SSM context exceeds the fixed eight-stage allocation.")
        class_mask = context.valid_class_mask
        if class_mask is None or class_mask.dtype != torch.bool or class_mask.ndim != 1:
            raise ValueError("SSM NC requires a one-dimensional class mask.")

    def _register_task_mask(self, context: Any) -> torch.Tensor:
        mask = context.valid_class_mask
        assert mask is not None
        owned = mask.detach().cpu().clone().contiguous()
        task = int(context.global_task_id)
        masks = self.state.get("task_class_masks")
        if not isinstance(masks, dict):
            raise RuntimeError("SSM task-class masks are malformed.")
        previous = masks.get(task)
        if previous is not None and not torch.equal(previous, owned):
            raise ValueError("SSM observed two class masks for one global task.")
        masks[task] = owned
        return owned

    def _task_mask(self, task: int) -> torch.Tensor:
        masks = self.state.get("task_class_masks")
        value = masks.get(int(task)) if isinstance(masks, dict) else None
        if not torch.is_tensor(value):
            raise RuntimeError("SSM replay task has no registered class mask.")
        return value

    def augment_loss(
        self,
        model: nn.Module,
        context: Any,
        logits: torch.Tensor,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_context(context)
        current_task_mask = self._register_task_mask(context)
        if not torch.is_tensor(base_loss) or base_loss.ndim != 0:
            raise ValueError("SSM base_loss must be a scalar tensor.")
        records = self._store().records()
        current_labels = context.train_labels
        class_counts: Dict[int, int] = {}
        for label in current_labels.tolist():
            class_counts[int(label)] = class_counts.get(int(label), 0) + 1
        for record in records:
            class_counts[record.root_label] = class_counts.get(record.root_label, 0) + 1
        device = _model_device(model, logits.device)
        current_numerator, current_denominator = _class_balanced_loss_terms(
            logits,
            labels=current_labels,
            valid_class_mask=current_task_mask,
            class_counts=class_counts,
        )
        replay_numerator = logits.sum() * 0.0
        replay_denominator = logits.new_zeros(())
        for record in records:
            features, edges, query, replay_label = self._device_record_tensors(
                record, device
            )
            replay_logits = self._forward_replay_queries(
                model,
                context,
                query,
                node_features=features,
                edge_index=edges,
            )
            numerator, denominator = _class_balanced_loss_terms(
                replay_logits,
                labels=replay_label,
                valid_class_mask=(
                    self._task_mask(record.global_task_id)
                    if self.incremental_setting == "task"
                    else current_task_mask
                ),
                class_counts=class_counts,
            )
            replay_numerator = replay_numerator + numerator
            replay_denominator = replay_denominator + denominator
        current_loss = current_numerator / current_denominator
        replay_loss = (
            replay_numerator / replay_denominator if records else logits.sum() * 0.0
        )
        objective = (current_numerator + replay_numerator) / (
            current_denominator + replay_denominator
        )
        diagnostics = self._diagnostic_state()
        diagnostics["current_loss"] = float(current_loss.detach().cpu())
        diagnostics["replay_loss"] = float(replay_loss.detach().cpu())
        diagnostics["class_balance_counts"] = {
            str(label): count for label, count in sorted(class_counts.items())
        }
        diagnostics["class_balance_weights"] = {
            str(label): 1.0 / count for label, count in sorted(class_counts.items())
        }
        return objective

    def consolidate(self, model: nn.Module, context: Any) -> None:
        from gecko.algorithms.continual.ssm.records import SSMRecord
        del model
        self._validate_context(context)
        self._register_task_mask(context)
        queries = context.train_queries
        labels = context.train_labels
        if queries.dtype != torch.long or queries.ndim != 1:
            raise ValueError("SSM consolidation requires one-dimensional NC queries.")
        if labels.dtype != torch.long or labels.ndim != 1:
            raise ValueError("SSM stores single integer NC root labels only.")
        if labels.shape[0] != queries.shape[0] or queries.numel() == 0:
            raise ValueError("SSM consolidation requires aligned non-empty roots.")
        class_mask = context.valid_class_mask
        assert class_mask is not None
        if (
            int(labels.min()) < 0
            or int(labels.max()) >= class_mask.numel()
            or not bool(class_mask[labels].all())
        ):
            raise ValueError("SSM refuses to persist a future or inactive class label.")
        root_labels: Dict[int, int] = {}
        for raw_root, raw_label in zip(queries.tolist(), labels.tolist()):
            root = int(raw_root)
            label = int(raw_label)
            previous = root_labels.setdefault(root, label)
            if previous != label:
                raise ValueError("One SSM replay root has conflicting current labels.")
        features = context.node_features
        # Deliberately use the immutable strict-local base graph, not an overlay.
        edges = context.base_edge_index
        sampler = self._prepared_sampler(features, edges)
        candidates: list[SSMRecord] = []
        for root in sorted(root_labels):
            record_seed = _derive_record_seed(
                base_seed=self.seed,
                client_id=self.client_id,
                global_task_id=int(context.global_task_id),
                stage_index=int(context.stage_index),
                root_index=root,
                sampler_mode=self.sampler_mode,
                hop_budgets=self.hop_budgets,
            )
            candidates.append(
                sampler.sample(
                    root_index=root,
                    root_label=root_labels[root],
                    client_id=self.client_id,
                    global_task_id=int(context.global_task_id),
                    stage_index=int(context.stage_index),
                    sampler_mode=self.sampler_mode,
                    hop_budgets=self.hop_budgets,
                    rng_seed=record_seed,
                    rng_base_seed=self.seed,
                )
            )
        store = self._store()
        allocation = store.reserve_stage(
            stage_index=int(context.stage_index),
            global_task_id=int(context.global_task_id),
            observed_classes=sorted(set(root_labels.values())),
            candidate_records=candidates,
        )
        self.state["replay_store"] = store.to_state()
        self._store_cache = store
        self._store_cache_source_id = id(self.state["replay_store"])
        task_ids = self.state.get("consolidated_task_ids")
        if not isinstance(task_ids, list):
            raise RuntimeError("SSM consolidated task state is malformed.")
        task_ids.append(int(context.global_task_id))
        class_allocations = allocation["class_allocations"]
        assert isinstance(class_allocations, Mapping)
        diagnostics = self._diagnostic_state()
        diagnostics["last_inserted_records"] = sum(
            int(item["inserted_count"]) for item in class_allocations.values()
        )
        diagnostics["last_rejected_records"] = sum(
            int(item["rejected_count"]) for item in class_allocations.values()
        )

    def replay_payload_bytes(self) -> int:
        """Return exact serialized replay tensors stored in safe method state."""

        return self._store().safe_checkpoint_bytes

    def diagnostics(self) -> Dict[str, object]:
        from gecko.algorithms.continual.ssm.replay import _clone_primitive_tree
        store = self._store()
        records = store.records()
        classes: Dict[str, int] = {}
        tasks: Dict[str, int] = {}
        root_degrees: list[int] = []
        for record in records:
            class_key = str(record.class_id)
            task_key = str(record.global_task_id)
            classes[class_key] = classes.get(class_key, 0) + 1
            tasks[task_key] = tasks.get(task_key, 0) + 1
            edges = record.edge_index
            root_degrees.append(
                int((edges[1] == record.root_index).sum()) if edges.numel() else 0
            )
        allocations = store.allocations()
        diagnostic_allocations: Dict[str, object] = {}
        for stage, allocation in sorted(allocations.items()):
            rendered = _clone_primitive_tree(allocation)
            assert isinstance(rendered, dict)
            class_allocations = rendered["class_allocations"]
            assert isinstance(class_allocations, Mapping)
            rendered["class_allocations"] = {
                str(class_id): value
                for class_id, value in sorted(class_allocations.items())
            }
            diagnostic_allocations[str(stage)] = rendered
        next_stage = max(allocations) + 1 if allocations else 0
        missed_stage_count = next_stage - len(allocations)
        diagnostic_snapshot = _clone_primitive_tree(self._diagnostic_state())
        assert isinstance(diagnostic_snapshot, dict)
        consolidated_task_ids = [
            int(allocation["global_task_id"])
            for _, allocation in sorted(allocations.items())
        ]

        output: Dict[str, object] = {
            **diagnostic_snapshot,
            "method": self.name,
            "method_version": self.method_version,
            "num_consolidated_tasks": len(consolidated_task_ids),
            "consolidated_task_ids": consolidated_task_ids,
            "sampler_mode": self.sampler_mode,
            "hop_budgets": list(self.hop_budgets),
            "replay_targets": len(records),
            "replay_context_nodes": sum(
                record.context_node_count for record in records
            ),
            "replay_edges": sum(record.edge_count for record in records),
            "mean_replay_root_degree": (
                sum(root_degrees) / len(root_degrees) if root_degrees else 0.0
            ),
            "class_representation": dict(sorted(classes.items())),
            "task_representation": dict(sorted(tasks.items())),
            "safe_checkpoint_bytes": store.safe_checkpoint_bytes,
            "replay_ceiling_bytes": store.total_ceiling_bytes,
            "finalized_stage_reserved_bytes": (
                len(allocations) * store.stage_slice_bytes
            ),
            "finalized_stage_unused_bytes": sum(
                int(allocation["unused_bytes"]) for allocation in allocations.values()
            ),
            "missed_stage_reserved_bytes": (
                missed_stage_count * store.stage_slice_bytes
            ),
            "future_stage_reserved_bytes": (
                (store.stage_count - next_stage) * store.stage_slice_bytes
            ),
            "unused_reserved_bytes": store.total_ceiling_bytes
            - store.safe_checkpoint_bytes,
            "stage_allocations": diagnostic_allocations,
        }
        return output

    def load_method_state(self, state: Mapping[str, object]) -> None:
        from gecko.algorithms.continual.ssm.replay import SSMReplayStore
        from gecko.algorithms.continual.ssm.records import SSM_REPLAY_CEILING_BYTES
        from gecko.algorithms.continual.ssm.records import SSM_STAGE_COUNT
        from gecko.algorithms.continual.ssm.records import SSM_STATE_FORMAT
        from gecko.algorithms.continual.ssm.records import _validate_nonnegative_int
        previous = self.save_method_state()
        previous_store_cache = self._store_cache
        previous_store_cache_source_id = self._store_cache_source_id
        try:
            super().load_method_state(state)
            expected = {
                "format",
                "method_hyperparameters",
                "replay_store",
                "consolidated_task_ids",
                "task_class_masks",
                "diagnostics",
            }
            if set(self.state) != expected or self.state["format"] != SSM_STATE_FORMAT:
                raise ValueError("SSM private-state fields do not match the schema.")
            if self.state["method_hyperparameters"] != self._private_hyperparameters():
                raise ValueError(
                    "SSM checkpoint hyperparameters do not match this method."
                )
            replay_state = self.state["replay_store"]
            if not isinstance(replay_state, Mapping):
                raise ValueError("SSM checkpoint replay store is malformed.")
            store = SSMReplayStore.from_state(replay_state)
            if (
                store.client_id != self.client_id
                or store.total_ceiling_bytes != SSM_REPLAY_CEILING_BYTES
                or store.stage_count != SSM_STAGE_COUNT
            ):
                raise ValueError("SSM checkpoint client or allocation policy mismatch.")
            self._validate_record_provenance(store.records())
            diagnostics = self.state["diagnostics"]
            diagnostic_fields = {
                "current_loss",
                "replay_loss",
                "last_inserted_records",
                "last_rejected_records",
                "class_balance_counts",
                "class_balance_weights",
            }
            if (
                not isinstance(diagnostics, dict)
                or set(diagnostics) != diagnostic_fields
            ):
                raise ValueError("SSM checkpoint diagnostics fields are malformed.")
            for name in ("current_loss", "replay_loss"):
                value = diagnostics[name]
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("SSM checkpoint diagnostic losses must be finite.")
            for name in ("last_inserted_records", "last_rejected_records"):
                _validate_nonnegative_int(diagnostics[name], name=name)
            counts = diagnostics["class_balance_counts"]
            weights = diagnostics["class_balance_weights"]
            if not isinstance(counts, dict) or not isinstance(weights, dict):
                raise ValueError(
                    "SSM checkpoint class-balance diagnostics are malformed."
                )
            if set(counts) != set(weights):
                raise ValueError("SSM checkpoint class-balance keys differ.")
            for label, count in counts.items():
                if not isinstance(label, str):
                    raise ValueError(
                        "SSM checkpoint class-balance labels must be strings."
                    )
                checked_count = _validate_nonnegative_int(count, name="class count")
                weight = weights[label]
                if (
                    checked_count == 0
                    or not isinstance(weight, float)
                    or not math.isclose(
                        weight, 1.0 / checked_count, rel_tol=0.0, abs_tol=0.0
                    )
                ):
                    raise ValueError("SSM checkpoint class-balance weight mismatch.")
            task_ids = self.state["consolidated_task_ids"]
            if not isinstance(task_ids, list) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in task_ids
            ):
                raise ValueError("SSM consolidated_task_ids are malformed.")
            allocation_tasks = [
                int(allocation["global_task_id"])
                for _, allocation in sorted(store.allocations().items())
            ]
            if task_ids != allocation_tasks:
                raise ValueError(
                    "SSM task IDs differ from immutable stage allocations."
                )
            masks = self.state["task_class_masks"]
            if not isinstance(masks, dict) or not set(task_ids) <= set(masks):
                raise ValueError("SSM task-class masks omit a consolidated task.")
            for task_id, mask in masks.items():
                if (
                    not torch.is_tensor(mask)
                    or mask.dtype != torch.bool
                    or mask.ndim != 1
                    or mask.device.type != "cpu"
                    or not bool(mask.any())
                ):
                    raise ValueError("SSM task-class mask is malformed.")
                labels = [
                    record.root_label
                    for record in store.records()
                    if record.global_task_id == task_id
                ]
                if labels and not bool(mask[torch.tensor(labels)].all()):
                    raise ValueError("SSM task-class mask excludes replay labels.")
            self._store_cache = store
            self._store_cache_source_id = id(self.state["replay_store"])
            self._sampler_cache = None
            self._sampler_cache_key = None
            self._device_record_cache.clear()
        except Exception:
            super().load_method_state(previous)
            self._store_cache = previous_store_cache
            self._store_cache_source_id = previous_store_cache_source_id
            raise


__all__ = [
    "SSMAlgorithm",
    "SSMRecord",
    "SSMReplayStore",
    "SSM_MAIN_HOP_BUDGETS",
    "SSM_NODE_ONLY_HOP_BUDGETS",
    "SSM_RECORD_FORMAT",
    "SSM_REPLAY_CEILING_BYTES",
    "SSM_STAGE_COUNT",
    "deserialize_ssm_record",
    "sample_sparse_computation_record",
    "serialize_ssm_record",
]




_RELOCATED_EXPORTS = {'SSMRecord': ('gecko.algorithms.continual.ssm.records', 'SSMRecord'), 'SSMReplayStore': ('gecko.algorithms.continual.ssm.replay', 'SSMReplayStore'), 'SSM_MAIN_HOP_BUDGETS': ('gecko.algorithms.continual.ssm.records', 'SSM_MAIN_HOP_BUDGETS'), 'SSM_NODE_ONLY_HOP_BUDGETS': ('gecko.algorithms.continual.ssm.records', 'SSM_NODE_ONLY_HOP_BUDGETS'), 'SSM_RECORD_FORMAT': ('gecko.algorithms.continual.ssm.records', 'SSM_RECORD_FORMAT'), 'SSM_REPLAY_CEILING_BYTES': ('gecko.algorithms.continual.ssm.records', 'SSM_REPLAY_CEILING_BYTES'), 'SSM_STAGE_COUNT': ('gecko.algorithms.continual.ssm.records', 'SSM_STAGE_COUNT'), 'SSM_STATE_FORMAT': ('gecko.algorithms.continual.ssm.records', 'SSM_STATE_FORMAT'), 'SSM_STORE_FORMAT': ('gecko.algorithms.continual.ssm.records', 'SSM_STORE_FORMAT'), '_DTYPES': ('gecko.algorithms.continual.ssm.records', '_DTYPES'), '_FEATURE_DTYPES': ('gecko.algorithms.continual.ssm.records', '_FEATURE_DTYPES'), '_MAX_RECORD_METADATA_BYTES': ('gecko.algorithms.continual.ssm.records', '_MAX_RECORD_METADATA_BYTES'), '_PreparedSparseComputationSampler': ('gecko.algorithms.continual.ssm.sampling', '_PreparedSparseComputationSampler'), '_RECORD_MAGIC': ('gecko.algorithms.continual.ssm.records', '_RECORD_MAGIC'), '_TENSOR_ORDER': ('gecko.algorithms.continual.ssm.records', '_TENSOR_ORDER'), '_canonical_json_bytes': ('gecko.algorithms.continual.ssm.records', '_canonical_json_bytes'), '_clone_primitive_tree': ('gecko.algorithms.continual.ssm.replay', '_clone_primitive_tree'), '_owned_tensor': ('gecko.algorithms.continual.ssm.records', '_owned_tensor'), '_payload_bytes': ('gecko.algorithms.continual.ssm.records', '_payload_bytes'), '_payload_tensor': ('gecko.algorithms.continual.ssm.replay', '_payload_tensor'), '_record_body': ('gecko.algorithms.continual.ssm.records', '_record_body'), '_record_digest': ('gecko.algorithms.continual.ssm.records', '_record_digest'), '_record_metadata': ('gecko.algorithms.continual.ssm.records', '_record_metadata'), '_sample_candidates': ('gecko.algorithms.continual.ssm.sampling', '_sample_candidates'), '_tensor_bytes': ('gecko.algorithms.continual.ssm.records', '_tensor_bytes'), '_tensor_descriptor': ('gecko.algorithms.continual.ssm.records', '_tensor_descriptor'), '_tensor_from_payload': ('gecko.algorithms.continual.ssm.records', '_tensor_from_payload'), '_validate_local_edge_index': ('gecko.algorithms.continual.ssm.records', '_validate_local_edge_index'), '_validate_nonnegative_int': ('gecko.algorithms.continual.ssm.records', '_validate_nonnegative_int'), 'deserialize_ssm_record': ('gecko.algorithms.continual.ssm.records', 'deserialize_ssm_record'), 'sample_sparse_computation_record': ('gecko.algorithms.continual.ssm.sampling', 'sample_sparse_computation_record'), 'serialize_ssm_record': ('gecko.algorithms.continual.ssm.records', 'serialize_ssm_record')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)
