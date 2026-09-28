"""Client-private CaT condensed graph memory for NC Class/Task-IL.

The implementation follows the CaT CGM reference mechanism: current-task
features are optimized by class-wise distribution matching under repeatedly
reinitialized graph encoders, the condensed task is stored with identity
topology, and subsequent classifier updates train only on the accumulated
condensed graph memory.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Sequence
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from gecko.algorithms.base import ClientContinualAlgorithm


CAT_STATE_FORMAT = "uefa-cat-state-v3"
CAT_METHOD_VERSION = "uefa-cat-v3-fixed-objective-diagnostics"
CAT_CONDENSATION_LOSS_SEMANTICS = (
    "first_random_encoder_same_objective_before_after"
)


def _owned_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor.")
    return value.detach().cpu().clone().contiguous()


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _tensor_payload_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def _identity_edges(num_nodes: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    count = _positive_int(num_nodes, name="num_nodes")
    nodes = torch.arange(count, dtype=torch.long, device=device)
    return torch.stack((nodes, nodes), dim=0)


@dataclass(frozen=True, init=False, eq=False)
class CondensedGraphRecord:
    """One immutable current-task graph produced by CaT condensation."""

    client_id: int
    global_task_id: int
    stage_index: int
    original_node_count: int
    condensation_steps: int
    condensation_seed: int
    condensation_seconds: float
    initial_loss: float
    final_loss: float
    _features: torch.Tensor
    _labels: torch.Tensor
    _edge_index: torch.Tensor
    _class_mask: torch.Tensor

    def __init__(
        self,
        *,
        client_id: int,
        global_task_id: int,
        stage_index: int,
        original_node_count: int,
        condensation_steps: int,
        condensation_seed: int,
        condensation_seconds: float,
        initial_loss: float,
        final_loss: float,
        features: torch.Tensor,
        labels: torch.Tensor,
        edge_index: torch.Tensor,
        class_mask: torch.Tensor | None = None,
    ) -> None:
        identifiers = {
            "client_id": client_id,
            "global_task_id": global_task_id,
            "stage_index": stage_index,
            "condensation_seed": condensation_seed,
        }
        checked = {
            name: _nonnegative_int(value, name=name)
            for name, value in identifiers.items()
        }
        original = _positive_int(original_node_count, name="original_node_count")
        steps = _positive_int(condensation_steps, name="condensation_steps")
        seconds = float(condensation_seconds)
        first_loss = float(initial_loss)
        last_loss = float(final_loss)
        if not all(math.isfinite(value) and value >= 0.0 for value in (seconds, first_loss, last_loss)):
            raise ValueError("CaT timing and condensation losses must be finite and non-negative.")

        owned_features = _owned_tensor(features, name="features")
        owned_labels = _owned_tensor(labels, name="labels")
        owned_edges = _owned_tensor(edge_index, name="edge_index")
        owned_class_mask = (
            torch.zeros(int(owned_labels.max()) + 1, dtype=torch.bool)
            if class_mask is None and owned_labels.numel()
            else torch.empty(0, dtype=torch.bool)
            if class_mask is None
            else _owned_tensor(class_mask, name="class_mask")
        )
        if class_mask is None and owned_labels.numel():
            owned_class_mask[owned_labels] = True
        if owned_features.ndim != 2 or not owned_features.is_floating_point():
            raise ValueError("CaT synthetic features must be a two-dimensional float tensor.")
        if owned_features.shape[0] == 0 or not torch.isfinite(owned_features).all():
            raise ValueError("CaT synthetic features must be non-empty and finite.")
        if owned_labels.dtype != torch.long or owned_labels.ndim != 1:
            raise ValueError("CaT synthetic labels must be one-dimensional int64.")
        if owned_labels.shape[0] != owned_features.shape[0]:
            raise ValueError("CaT synthetic labels must align with features.")
        if owned_labels.numel() and int(owned_labels.min()) < 0:
            raise ValueError("CaT synthetic labels must be non-negative.")
        if (
            owned_edges.dtype != torch.long
            or owned_edges.ndim != 2
            or owned_edges.shape[0] != 2
        ):
            raise ValueError("CaT edge_index must have int64 shape [2, num_edges].")
        if owned_edges.numel() and (
            int(owned_edges.min()) < 0
            or int(owned_edges.max()) >= owned_features.shape[0]
        ):
            raise ValueError("CaT edge_index contains an illegal synthetic endpoint.")
        expected_edges = _identity_edges(int(owned_features.shape[0]))
        if not torch.equal(owned_edges, expected_edges):
            raise ValueError("CaT v1 stores the official identity/self-loop topology only.")
        if int(owned_features.shape[0]) >= original:
            raise ValueError("CaT condensed graph must be smaller than its original task data.")
        if owned_class_mask.dtype != torch.bool or owned_class_mask.ndim != 1:
            raise ValueError("CaT task class mask must be one-dimensional bool.")
        if owned_labels.numel() and (
            int(owned_labels.max()) >= owned_class_mask.numel()
            or not bool(owned_class_mask[owned_labels].all())
        ):
            raise ValueError("CaT task class mask excludes a condensed label.")

        for name, value in checked.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "original_node_count", original)
        object.__setattr__(self, "condensation_steps", steps)
        object.__setattr__(self, "condensation_seconds", seconds)
        object.__setattr__(self, "initial_loss", first_loss)
        object.__setattr__(self, "final_loss", last_loss)
        object.__setattr__(self, "_features", owned_features)
        object.__setattr__(self, "_labels", owned_labels)
        object.__setattr__(self, "_edge_index", owned_edges)
        object.__setattr__(self, "_class_mask", owned_class_mask)

    @property
    def features(self) -> torch.Tensor:
        return self._features.clone()

    @property
    def labels(self) -> torch.Tensor:
        return self._labels.clone()

    @property
    def edge_index(self) -> torch.Tensor:
        return self._edge_index.clone()

    @property
    def class_mask(self) -> torch.Tensor:
        return self._class_mask.clone()

    @property
    def synthetic_node_count(self) -> int:
        return int(self._features.shape[0])

    @property
    def synthetic_edge_count(self) -> int:
        return int(self._edge_index.shape[1])

    @property
    def payload_bytes(self) -> int:
        return sum(
            _tensor_payload_bytes(value)
            for value in (self._features, self._labels, self._edge_index, self._class_mask)
        )

    @property
    def classes(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in torch.unique(self._labels, sorted=True).tolist())

    @property
    def class_distribution(self) -> Dict[int, int]:
        return {
            class_id: int((self._labels == class_id).sum())
            for class_id in self.classes
        }

    def to_state(self) -> Dict[str, object]:
        return {
            "client_id": self.client_id,
            "global_task_id": self.global_task_id,
            "stage_index": self.stage_index,
            "original_node_count": self.original_node_count,
            "condensation_steps": self.condensation_steps,
            "condensation_seed": self.condensation_seed,
            "condensation_seconds": self.condensation_seconds,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "features": self.features,
            "labels": self.labels,
            "edge_index": self.edge_index,
            "class_mask": self.class_mask,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "CondensedGraphRecord":
        expected = {
            "client_id",
            "global_task_id",
            "stage_index",
            "original_node_count",
            "condensation_steps",
            "condensation_seed",
            "condensation_seconds",
            "initial_loss",
            "final_loss",
            "features",
            "labels",
            "edge_index",
            "class_mask",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("CaT condensed-record fields do not match the schema.")
        return cls(**dict(state))


def _derive_condensation_seed(
    *, base_seed: int, client_id: int, global_task_id: int, stage_index: int
) -> int:
    message = f"cat|{base_seed}|{client_id}|{global_task_id}|{stage_index}".encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big") % (2**63)


def _reset_random_encoder(model: nn.Module) -> None:
    """Reset leaf trainable modules without invoking parent resets twice."""

    for module in model.modules():
        if any(True for _ in module.children()):
            continue
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()


def _masked_class_logits(logits: torch.Tensor, class_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != class_mask.numel():
        raise ValueError("CaT NC logits do not match the seen-class mask.")
    output = logits.clone()
    output[:, ~class_mask.to(logits.device)] = -1e12
    return output


def _distribution_matching_loss(
    real_embeddings: torch.Tensor,
    real_labels: torch.Tensor,
    synthetic_embeddings: torch.Tensor,
    synthetic_labels: torch.Tensor,
) -> torch.Tensor:
    """Official CaT class-wise normalized embedding-mean objective."""

    if real_embeddings.ndim != 2 or synthetic_embeddings.ndim != 2:
        raise ValueError("CaT distribution matching requires embedding matrices.")
    if real_embeddings.shape[1] != synthetic_embeddings.shape[1]:
        raise ValueError("Real and synthetic CaT embedding widths differ.")
    if real_labels.ndim != 1 or real_labels.shape[0] != real_embeddings.shape[0]:
        raise ValueError("Real CaT labels do not align with embeddings.")
    if synthetic_labels.ndim != 1 or synthetic_labels.shape[0] != synthetic_embeddings.shape[0]:
        raise ValueError("Synthetic CaT labels do not align with embeddings.")
    classes = torch.unique(real_labels, sorted=True)
    if classes.numel() == 0 or not torch.equal(classes, torch.unique(synthetic_labels, sorted=True)):
        raise ValueError("CaT real and synthetic class sets must match exactly.")
    total = real_embeddings.new_zeros(())
    denominator = float(real_labels.numel())
    for class_id in classes.tolist():
        real_mask = real_labels == int(class_id)
        synthetic_mask = synthetic_labels == int(class_id)
        difference = (
            real_embeddings[real_mask].mean(dim=0)
            - synthetic_embeddings[synthetic_mask].mean(dim=0)
        )
        total = total + (float(real_mask.sum()) / denominator) * difference.square().sum()
    return total


class CaTAlgorithm(ClientContinualAlgorithm):
    """CaT condensed graph memory and balanced Training-in-Memory."""

    name = "CaT"
    method_version = CAT_METHOD_VERSION

    def __init__(
        self,
        *,
        synthetic_nodes_per_class: int = 2,
        condensation_steps: int = 32,
        condensation_lr: float = 1.0e-3,
        memory_ceiling_bytes: int = 16 * 1024 * 1024,
        stage_count: int = 8,
        feature_initialization: str = "random_choice",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.client_id is None or self.client_id < 0:
            raise ValueError("CaT requires an explicit non-negative client_id.")
        if self.problem_type != "NC" or self.incremental_setting not in {"class", "task"}:
            raise ValueError("CaT supports only NC-Class/NC-Task and fails closed.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("CaT seed must be a non-negative signed 63-bit integer.")
        self.synthetic_nodes_per_class = _positive_int(
            synthetic_nodes_per_class, name="synthetic_nodes_per_class"
        )
        self.condensation_steps = _positive_int(
            condensation_steps, name="condensation_steps"
        )
        learning_rate = float(condensation_lr)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("condensation_lr must be finite and positive.")
        self.condensation_lr = learning_rate
        self.memory_ceiling_bytes = _positive_int(
            memory_ceiling_bytes, name="memory_ceiling_bytes"
        )
        self.stage_count = _positive_int(stage_count, name="stage_count")
        if feature_initialization != "random_choice":
            raise ValueError("CaT v1 implements the official randomChoice initializer only.")
        self.feature_initialization = feature_initialization
        self.state.update(
            {
                "format": CAT_STATE_FORMAT,
                "method_hyperparameters": self._private_hyperparameters(),
                "records": [],
                "diagnostics": {
                    "current_raw_loss": 0.0,
                    "training_in_memory_loss": 0.0,
                    "last_condensation_initial_loss": 0.0,
                    "last_condensation_final_loss": 0.0,
                    "last_condensation_seconds": 0.0,
                    "condensation_loss_semantics": (
                        CAT_CONDENSATION_LOSS_SEMANTICS
                    ),
                },
            }
        )

    def _private_hyperparameters(self) -> Dict[str, object]:
        return {
            "synthetic_nodes_per_class": self.synthetic_nodes_per_class,
            "condensation_steps": self.condensation_steps,
            "condensation_lr": self.condensation_lr,
            "memory_ceiling_bytes": self.memory_ceiling_bytes,
            "stage_count": self.stage_count,
            "feature_initialization": self.feature_initialization,
        }

    def hyperparameters(self) -> Dict[str, object]:
        output = super().hyperparameters()
        output.update(self._private_hyperparameters())
        return output

    def _validate_context(self, context: Any) -> None:
        if str(context.problem_type).upper() != "NC" or str(context.incremental_setting).lower() not in {"class", "task"}:
            raise ValueError("CaT supports only NC-Class/NC-Task contexts.")
        if int(context.client_id) != self.client_id:
            raise ValueError("CaT context belongs to another client.")
        if int(context.stage_index) >= self.stage_count:
            raise ValueError("CaT context exceeds the configured stage_count.")
        class_mask = context.valid_class_mask
        if class_mask is None or class_mask.dtype != torch.bool or class_mask.ndim != 1:
            raise ValueError("CaT NC requires a one-dimensional task class mask.")

    def _records(self) -> Tuple[CondensedGraphRecord, ...]:
        raw = self.state.get("records")
        if not isinstance(raw, list):
            raise RuntimeError("CaT condensed memory is malformed.")
        records = tuple(CondensedGraphRecord.from_state(value) for value in raw)
        task_ids = [record.global_task_id for record in records]
        stages = [record.stage_index for record in records]
        if len(set(task_ids)) != len(task_ids) or len(set(stages)) != len(stages):
            raise ValueError("CaT memory contains duplicate task or stage records.")
        if any(record.client_id != self.client_id for record in records):
            raise ValueError("CaT memory contains another client's data.")
        if sum(record.payload_bytes for record in records) > self.memory_ceiling_bytes:
            raise ValueError("CaT memory exceeds the configured byte ceiling.")
        return records

    def replay_samples(self, context: Any) -> Tuple[CondensedGraphRecord, ...]:
        self._validate_context(context)
        return self._records()

    def before_task(self, context: Any) -> None:
        self._validate_context(context)
        if any(record.stage_index > int(context.stage_index) for record in self._records()):
            raise ValueError("CaT memory contains a future-stage record.")

    @staticmethod
    def _model_device(model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration as error:
            raise RuntimeError("CaT requires a parameterized graph model.") from error

    def _initialize_features(
        self,
        context: Any,
        *,
        labels: torch.Tensor,
        classes: Sequence[int],
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = context.train_queries.detach().cpu().long()
        real_features = context.node_features.detach().cpu()[queries]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        selected_features = []
        synthetic_labels = []
        for class_id in classes:
            candidates = (labels == class_id).nonzero(as_tuple=False).reshape(-1)
            if candidates.numel() == 0:
                raise ValueError("CaT cannot initialize an absent current-task class.")
            draws = torch.randint(
                int(candidates.numel()),
                (self.synthetic_nodes_per_class,),
                generator=generator,
            )
            selected_features.append(real_features[candidates[draws]])
            synthetic_labels.extend([class_id] * self.synthetic_nodes_per_class)
        return (
            torch.cat(selected_features, dim=0),
            torch.tensor(synthetic_labels, dtype=torch.long),
        )

    def _condense(self, model: nn.Module, context: Any) -> CondensedGraphRecord:
        self._validate_context(context)
        queries = context.train_queries.detach().cpu().long()
        labels = context.train_labels.detach().cpu().long()
        if queries.ndim != 1 or labels.ndim != 1 or queries.shape[0] != labels.shape[0] or labels.numel() == 0:
            raise ValueError("CaT condensation requires aligned non-empty NC training data.")
        if len(set(int(value) for value in queries.tolist())) != queries.numel():
            raise ValueError("CaT current-task queries must not contain duplicates.")
        class_mask = context.valid_class_mask
        assert class_mask is not None
        if int(labels.min()) < 0 or int(labels.max()) >= class_mask.numel() or not bool(class_mask[labels].all()):
            raise ValueError("CaT refuses future or inactive current-task labels.")
        classes = tuple(int(value) for value in torch.unique(labels, sorted=True).tolist())
        synthetic_count = len(classes) * self.synthetic_nodes_per_class
        context_edges = context.effective_edge_index.detach().cpu().long()
        visible_nodes = torch.unique(
            torch.cat((queries, context_edges.reshape(-1))), sorted=True
        )
        original_node_count = int(visible_nodes.numel())
        if synthetic_count >= original_node_count:
            raise ValueError(
                "CaT condensation is not smaller than the task-valid computation "
                "graph; lower synthetic_nodes_per_class."
            )
        seed = _derive_condensation_seed(
            base_seed=self.seed,
            client_id=self.client_id,
            global_task_id=int(context.global_task_id),
            stage_index=int(context.stage_index),
        )
        initial_features, synthetic_labels = self._initialize_features(
            context, labels=labels, classes=classes, seed=seed
        )
        device = self._model_device(model)
        synthetic_features = nn.Parameter(initial_features.to(device))
        labels_device = labels.to(device)
        synthetic_labels_device = synthetic_labels.to(device)
        synthetic_edges = _identity_edges(synthetic_count, device=device)
        optimizer = torch.optim.Adam([synthetic_features], lr=self.condensation_lr)
        initial_loss = 0.0
        final_loss = 0.0
        started = time.perf_counter()
        fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            for step in range(self.condensation_steps):
                step_seed = (seed + step) % (2**63)
                torch.manual_seed(step_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(step_seed)
                encoder = copy.deepcopy(model).to(device)
                _reset_random_encoder(encoder)
                encoder.eval()
                for parameter in encoder.parameters():
                    parameter.requires_grad_(False)
                with torch.no_grad():
                    real_all = context.encode_nodes(encoder)
                    real_embeddings = F.normalize(
                        real_all[queries.to(real_all.device)], p=2, dim=-1
                    )
                synthetic_embeddings = F.normalize(
                    context.encode_nodes(
                        encoder,
                        node_features=synthetic_features,
                        edge_index=synthetic_edges,
                    ),
                    p=2,
                    dim=-1,
                )
                loss = _distribution_matching_loss(
                    real_embeddings,
                    labels_device,
                    synthetic_embeddings,
                    synthetic_labels_device,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("CaT condensation produced a non-finite loss.")
                if step == 0:
                    initial_loss = float(loss.detach().cpu())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # The optimization intentionally redraws a random encoder per step.
            # Comparing step-0 loss with the last-step loss therefore compares
            # different objectives.  Recreate the exact step-0 encoder and audit
            # the final features under that same frozen objective instead.
            del loss, synthetic_embeddings, real_embeddings, real_all, encoder
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            audit_encoder = copy.deepcopy(model).to(device)
            _reset_random_encoder(audit_encoder)
            audit_encoder.eval()
            for parameter in audit_encoder.parameters():
                parameter.requires_grad_(False)
            with torch.no_grad():
                audit_real_all = context.encode_nodes(audit_encoder)
                audit_real = F.normalize(
                    audit_real_all[queries.to(audit_real_all.device)], p=2, dim=-1
                )
                audit_synthetic = F.normalize(
                    context.encode_nodes(
                        audit_encoder,
                        node_features=synthetic_features,
                        edge_index=synthetic_edges,
                    ),
                    p=2,
                    dim=-1,
                )
                audit_loss = _distribution_matching_loss(
                    audit_real,
                    labels_device,
                    audit_synthetic,
                    synthetic_labels_device,
                )
                if not torch.isfinite(audit_loss):
                    raise RuntimeError(
                        "CaT fixed-objective audit produced a non-finite loss."
                    )
                final_loss = float(audit_loss.cpu())
        elapsed = time.perf_counter() - started
        return CondensedGraphRecord(
            client_id=self.client_id,
            global_task_id=int(context.global_task_id),
            stage_index=int(context.stage_index),
            original_node_count=original_node_count,
            condensation_steps=self.condensation_steps,
            condensation_seed=seed,
            condensation_seconds=elapsed,
            initial_loss=initial_loss,
            final_loss=final_loss,
            features=synthetic_features.detach(),
            labels=synthetic_labels,
            edge_index=_identity_edges(synthetic_count),
            class_mask=class_mask,
        )

    def _ensure_current_record(self, model: nn.Module, context: Any) -> None:
        records = self._records()
        task_id = int(context.global_task_id)
        matching = [record for record in records if record.global_task_id == task_id]
        if matching:
            if matching[0].stage_index != int(context.stage_index):
                raise ValueError("CaT task record was created at another stage.")
            return
        if any(record.stage_index >= int(context.stage_index) for record in records):
            raise ValueError("CaT task records are not strictly chronological.")
        record = self._condense(model, context)
        if self.replay_payload_bytes() + record.payload_bytes > self.memory_ceiling_bytes:
            raise MemoryError("CaT condensed memory would exceed memory_ceiling_bytes.")
        raw = self.state["records"]
        assert isinstance(raw, list)
        raw.append(record.to_state())
        diagnostics = self.state["diagnostics"]
        assert isinstance(diagnostics, dict)
        diagnostics["last_condensation_initial_loss"] = record.initial_loss
        diagnostics["last_condensation_final_loss"] = record.final_loss
        diagnostics["last_condensation_seconds"] = record.condensation_seconds
        diagnostics["condensation_loss_semantics"] = (
            CAT_CONDENSATION_LOSS_SEMANTICS
        )

    def _training_in_memory_loss(self, model: nn.Module, context: Any) -> torch.Tensor:
        records = self._records()
        if not records:
            raise RuntimeError("CaT Training-in-Memory requires a current condensed graph.")
        device = self._model_device(model)
        per_task_losses = []
        for record in records:
            if record.stage_index > int(context.stage_index):
                raise ValueError("CaT Training-in-Memory detected future-stage data.")
            features = record.features.to(device)
            labels = record.labels.to(device)
            edges = record.edge_index.to(device)
            queries = torch.arange(record.synthetic_node_count, device=device)
            logits = context.forward_queries(
                model,
                queries,
                node_features=features,
                edge_index=edges,
            )
            replay_mask = (
                record.class_mask
                if self.incremental_setting == "task"
                else context.valid_class_mask
            )
            assert replay_mask is not None
            masked = _masked_class_logits(logits, replay_mask)
            per_task_losses.append(F.cross_entropy(masked, labels))
        return torch.stack(per_task_losses).mean()

    def augment_loss(
        self,
        model: nn.Module,
        context: Any,
        logits: torch.Tensor,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_context(context)
        if not torch.is_tensor(base_loss) or base_loss.ndim != 0:
            raise ValueError("CaT base_loss must be a scalar tensor.")
        self._ensure_current_record(model, context)
        memory_loss = self._training_in_memory_loss(model, context)
        diagnostics = self.state["diagnostics"]
        assert isinstance(diagnostics, dict)
        diagnostics["current_raw_loss"] = float(base_loss.detach().cpu())
        diagnostics["training_in_memory_loss"] = float(memory_loss.detach().cpu())
        # CaT deliberately does not mix the full incoming graph into training.
        return memory_loss + logits.sum() * 0.0

    def consolidate(self, model: nn.Module, context: Any) -> None:
        del model
        self._validate_context(context)
        if not any(
            record.global_task_id == int(context.global_task_id)
            for record in self._records()
        ):
            raise RuntimeError("CaT task completed without current-task condensation.")

    def replay_payload_bytes(self) -> int:
        return sum(record.payload_bytes for record in self._records())

    def diagnostics(self) -> Dict[str, object]:
        records = self._records()
        raw_diagnostics = self.state.get("diagnostics")
        if not isinstance(raw_diagnostics, dict):
            raise RuntimeError("CaT diagnostics are malformed.")
        per_task = {}
        class_distribution: Dict[str, int] = {}
        for record in records:
            distribution = {
                str(key): value for key, value in record.class_distribution.items()
            }
            for key, value in distribution.items():
                class_distribution[key] = class_distribution.get(key, 0) + value
            per_task[str(record.global_task_id)] = {
                "stage_index": record.stage_index,
                "original_nodes": record.original_node_count,
                "synthetic_nodes": record.synthetic_node_count,
                "synthetic_edges": record.synthetic_edge_count,
                "reduction_ratio": record.synthetic_node_count / record.original_node_count,
                "class_distribution": distribution,
                "condensation_steps": record.condensation_steps,
                "condensation_initial_loss": record.initial_loss,
                "condensation_final_loss": record.final_loss,
                "condensation_loss_semantics": (
                    CAT_CONDENSATION_LOSS_SEMANTICS
                ),
                "condensation_objective_delta": (
                    record.final_loss - record.initial_loss
                ),
                "condensation_seconds": record.condensation_seconds,
                "payload_bytes": record.payload_bytes,
            }
        original_nodes = sum(record.original_node_count for record in records)
        synthetic_nodes = sum(record.synthetic_node_count for record in records)
        output = dict(raw_diagnostics)
        output.update(
            {
                "method": self.name,
                "method_version": self.method_version,
                "condensed_tasks": len(records),
                "condensed_task_ids": [record.global_task_id for record in records],
                "original_nodes": original_nodes,
                "synthetic_nodes": synthetic_nodes,
                "synthetic_edges": sum(record.synthetic_edge_count for record in records),
                "reduction_ratio": synthetic_nodes / original_nodes if original_nodes else 0.0,
                "class_distribution": dict(sorted(class_distribution.items())),
                "memory_payload_bytes": self.replay_payload_bytes(),
                "memory_ceiling_bytes": self.memory_ceiling_bytes,
                "peak_memory_payload_bytes": self.replay_payload_bytes(),
                "total_condensation_seconds": sum(record.condensation_seconds for record in records),
                "per_task": per_task,
            }
        )
        return output

    def load_method_state(self, state: Mapping[str, object]) -> None:
        previous = self.save_method_state()
        try:
            super().load_method_state(state)
            if set(self.state) != {"format", "method_hyperparameters", "records", "diagnostics"}:
                raise ValueError("CaT private-state fields do not match the schema.")
            if self.state["format"] != CAT_STATE_FORMAT:
                raise ValueError("Unsupported CaT state format.")
            if self.state["method_hyperparameters"] != self._private_hyperparameters():
                raise ValueError("CaT checkpoint hyperparameters do not match this method.")
            records = self._records()
            if any(record.stage_index >= self.stage_count for record in records):
                raise ValueError("CaT checkpoint record exceeds stage_count.")
            diagnostics = self.state["diagnostics"]
            expected = {
                "current_raw_loss",
                "training_in_memory_loss",
                "last_condensation_initial_loss",
                "last_condensation_final_loss",
                "last_condensation_seconds",
                "condensation_loss_semantics",
            }
            if not isinstance(diagnostics, dict) or set(diagnostics) != expected:
                raise ValueError("CaT checkpoint diagnostics are malformed.")
            if diagnostics["condensation_loss_semantics"] != (
                CAT_CONDENSATION_LOSS_SEMANTICS
            ):
                raise ValueError("CaT checkpoint loss semantics are invalid.")
            numeric_diagnostics = (
                diagnostics["current_raw_loss"],
                diagnostics["training_in_memory_loss"],
                diagnostics["last_condensation_initial_loss"],
                diagnostics["last_condensation_final_loss"],
                diagnostics["last_condensation_seconds"],
            )
            if any(
                not isinstance(value, float) or not math.isfinite(value) or value < 0.0
                for value in numeric_diagnostics
            ):
                raise ValueError("CaT checkpoint diagnostics must be finite non-negative floats.")
        except Exception:
            super().load_method_state(previous)
            raise


__all__ = [
    "CAT_METHOD_VERSION",
    "CAT_STATE_FORMAT",
    "CAT_CONDENSATION_LOSS_SEMANTICS",
    "CaTAlgorithm",
    "CondensedGraphRecord",
    "_distribution_matching_loss",
    "_identity_edges",
]
