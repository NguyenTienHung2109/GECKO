"""Gradient Episodic Memory for leakage-safe UEFA NC/LC clients."""

from __future__ import annotations


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


_STATE_VERSION = "uefa-gem-private-state-v1"


def _supervised_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim > 1:
        valid = labels >= 0
        if not bool(valid.any()):
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(logits[valid], labels[valid].float())
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(
            logits.reshape(-1), labels.float().reshape(-1)
        )
    return F.cross_entropy(logits, labels.long())


def _masked_logits(
    logits: torch.Tensor, class_mask: torch.Tensor | None
) -> torch.Tensor:
    if (
        class_mask is None
        or logits.ndim < 2
        or logits.shape[-1] != class_mask.shape[0]
    ):
        return logits
    output = logits.clone()
    output[..., ~class_mask.to(logits.device)] = -1e12
    return output


def _named_trainable_parameters(
    model: nn.Module,
) -> Tuple[Tuple[str, nn.Parameter], ...]:
    result = tuple(
        sorted(
            (
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.is_floating_point()
            ),
            key=lambda item: item[0],
        )
    )
    if not result:
        raise ValueError("GEM requires floating trainable parameters.")
    return result


def _flatten_autograd(
    objective: torch.Tensor,
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        objective,
        tuple(parameter for _, parameter in named_parameters),
        allow_unused=True,
    )
    return torch.cat(
        tuple(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
            for (_, parameter), gradient in zip(named_parameters, gradients)
        )
    )


def _flatten_parameter_gradients(
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
) -> torch.Tensor:
    return torch.cat(
        tuple(
            torch.zeros_like(parameter).reshape(-1)
            if parameter.grad is None
            else parameter.grad.detach().reshape(-1)
            for _, parameter in named_parameters
        )
    )


def _assign_parameter_gradients(
    named_parameters: Sequence[Tuple[str, nn.Parameter]], gradient: torch.Tensor
) -> None:
    offset = 0
    for _, parameter in named_parameters:
        count = parameter.numel()
        value = gradient[offset : offset + count].reshape_as(parameter)
        parameter.grad = value.to(parameter.device, parameter.dtype).clone()
        offset += count
    if offset != gradient.numel():
        raise ValueError("Projected GEM gradient does not match model parameters.")


def _solve_gem_dual_qp(
    matrix: torch.Tensor,
    linear: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    """Solve BeGin's lower-bound dual QP without a combinatorial oracle."""

    try:
        from qpth.qp import QPFunction
    except ImportError:
        # The canonical UEFA environment provides qpth. SciPy keeps lightweight
        # CPU development environments runnable without changing the QP itself.
        try:
            import numpy as np
            from scipy.optimize import lsq_linear
            from scipy.optimize import minimize
        except ImportError as error:
            raise RuntimeError(
                "GEM requires qpth (canonical) or scipy.optimize (CPU fallback)."
            ) from error
        values = matrix.detach().cpu().double().numpy()
        vector = linear.detach().cpu().double().numpy()
        initial = np.full(vector.shape, float(lower_bound), dtype=np.float64)
        result = minimize(
            lambda candidate: 0.5 * candidate @ values @ candidate
            + vector @ candidate,
            initial,
            jac=lambda candidate: values @ candidate + vector,
            bounds=[(float(lower_bound), None)] * int(vector.size),
            method="L-BFGS-B",
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000},
        )

        def kkt_valid(candidate: Any) -> bool:
            """Validate the sufficient KKT conditions for this convex QP."""

            if (
                np.shape(candidate) != np.shape(vector)
                or not np.isfinite(candidate).all()
            ):
                return False
            bound = float(lower_bound)
            candidate_scale = max(1.0, float(np.abs(candidate).max(initial=0.0)))
            bound_tolerance = 1e-10 * candidate_scale
            if bool((candidate < bound - bound_tolerance).any()):
                return False
            gradient = values @ candidate + vector
            gradient_scale = max(
                1.0,
                float(np.abs(vector).max(initial=0.0)),
                float(np.abs(values @ candidate).max(initial=0.0)),
            )
            gradient_tolerance = 1e-8 * gradient_scale
            active = candidate <= bound + bound_tolerance
            if bool((gradient[active] < -gradient_tolerance).any()):
                return False
            return not bool((np.abs(gradient[~active]) > gradient_tolerance).any())

        if result.success and kkt_valid(result.x):
            return torch.from_numpy(result.x).to(matrix.device, matrix.dtype)

        # L-BFGS-B can report ``ABNORMAL`` when its line search loses precision
        # on a tiny, ill-conditioned GEM dual even though the QP is convex.
        # Cholesky turns the *same* objective into bounded least squares:
        #   Q = L L^T,  A = L^T,  A^T b = -c.
        # BVLS is deterministic for these small per-task duals and does not
        # relax the lower bound or alter the configured projection epsilon.
        try:
            cholesky = np.linalg.cholesky(values)
            design = cholesky.T
            target = np.linalg.solve(cholesky, -vector)
            fallback = lsq_linear(
                design,
                target,
                bounds=(float(lower_bound), np.inf),
                method="bvls",
                tol=1e-12,
                max_iter=10_000,
            )
        except (np.linalg.LinAlgError, ValueError) as error:
            raise RuntimeError(
                "GEM SciPy QP failed in both L-BFGS-B and BVLS: "
                f"L-BFGS-B={result.message}; BVLS={error}"
            ) from error
        if not fallback.success or not kkt_valid(fallback.x):
            raise RuntimeError(
                "GEM SciPy QP failed in both L-BFGS-B and BVLS: "
                f"L-BFGS-B={result.message}; BVLS={fallback.message}"
            )
        return torch.from_numpy(fallback.x).to(matrix.device, matrix.dtype)

    size = int(linear.numel())
    identity = torch.eye(size, device=matrix.device, dtype=matrix.dtype)
    empty = torch.empty(0, device=matrix.device, dtype=matrix.dtype)
    return QPFunction(verbose=False)(
        matrix,
        linear,
        -identity,
        torch.full_like(linear, -float(lower_bound)),
        empty,
        empty,
    )[0].detach()


def project_gem_gradient(
    gradient: torch.Tensor,
    memory_gradients: torch.Tensor,
    *,
    margin: float = 0.5,
    epsilon: float = 1e-3,
    violation_tolerance: float = 1e-10,
) -> torch.Tensor:
    """Project a gradient with the same dual QP as BeGin's GEM reference."""

    current = gradient.detach()
    memories = memory_gradients.detach()
    if current.ndim != 1 or memories.ndim != 2 or memories.shape[1] != current.numel():
        raise ValueError("GEM gradient and memory-gradient shapes do not align.")
    if memories.shape[0] == 0:
        return current.clone()
    if not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(memories).all()):
        raise ValueError("GEM gradients must be finite.")
    products = memories @ current
    if not bool((products < -violation_tolerance).any()):
        return current.clone()

    memory_cpu = memories.cpu().double()
    current_cpu = current.cpu().double()
    count = memory_cpu.shape[0]
    identity = torch.eye(count, dtype=torch.float64)
    quadratic = memory_cpu @ memory_cpu.T
    quadratic = 0.5 * (quadratic + quadratic.T) + epsilon * identity
    linear = memory_cpu @ current_cpu
    dual = _solve_gem_dual_qp(quadratic, linear, margin)
    projected = current_cpu + dual @ memory_cpu
    if not bool(torch.isfinite(projected).all()):
        raise RuntimeError("GEM projection produced a non-finite gradient.")
    post = memory_cpu @ projected
    numerical_tolerance = max(1e-7, 10.0 * epsilon * float(dual.abs().max()))
    if bool((post < -numerical_tolerance).any()):
        raise RuntimeError("GEM projection failed its remembered-gradient oracle.")
    return projected.to(current.device, current.dtype)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class GEMAlgorithm(ClientContinualAlgorithm):
    """Client-private episodic memory with per-task gradient constraints."""

    name = "GEM"
    method_version = "uefa-gem-v1"

    def __init__(
        self,
        *,
        memory_size: int = 100,
        margin: float = 0.5,
        projection_epsilon: float = 1e-3,
        violation_tolerance: float = 1e-10,
        **kwargs: object,
    ) -> None:
        if isinstance(memory_size, bool) or not isinstance(memory_size, int) or memory_size < 1:
            raise ValueError("memory_size must be a positive integer.")
        for name, value in (
            ("margin", margin),
            ("projection_epsilon", projection_epsilon),
            ("violation_tolerance", violation_tolerance),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if projection_epsilon == 0.0:
            raise ValueError("projection_epsilon must be positive.")
        super().__init__(memory_size=memory_size, **kwargs)
        self.margin = float(margin)
        self.projection_epsilon = float(projection_epsilon)
        self.violation_tolerance = float(violation_tolerance)
        self._pending_memory_gradients: torch.Tensor | None = None
        self._pending_parameter_names: Tuple[str, ...] = ()
        self._current_feature_sha256: str | None = None
        self._last_diagnostics: Dict[str, object] = {
            "constraint_count": 0,
            "violated_constraint_count": 0,
            "minimum_dot_before": 0.0,
            "minimum_dot_after": 0.0,
            "projection_applied": False,
        }
        self.state.update(
            {
                "state_version": _STATE_VERSION,
                "method_hyperparameters": self._private_hyperparameters(),
                "memories": {},
                "consolidated_task_ids": [],
                "projection_count": 0,
            }
        )

    def _private_hyperparameters(self) -> Dict[str, object]:
        return {
            "memory_size": self.memory_size,
            "margin": self.margin,
            "projection_epsilon": self.projection_epsilon,
            "violation_tolerance": self.violation_tolerance,
        }

    def hyperparameters(self) -> Dict[str, object]:
        result = super().hyperparameters()
        result.update(self._private_hyperparameters())
        return result

    def _validate_scenario(self, context: Any) -> None:
        if str(context.problem_type).upper() not in {"NC", "LC"}:
            raise ValueError("GEM UEFA support is restricted to NC and LC.")
        if str(context.incremental_setting).lower() not in {"task", "class", "domain"}:
            raise ValueError("GEM requires task, class, or domain incremental data.")

    @staticmethod
    def _task_mask(context: Any) -> torch.Tensor | None:
        if str(context.incremental_setting).lower() != "task":
            return None
        mask = context.valid_class_mask
        return None if mask is None else mask.detach().cpu().bool().clone()

    def _memory_gradients(
        self,
        model: nn.Module,
        context: Any,
        named_parameters: Sequence[Tuple[str, nn.Parameter]],
    ) -> torch.Tensor:
        gradients = []
        current_features = context.node_features
        feature_hash = self._current_feature_sha256 or _tensor_sha256(current_features)
        for task_id in self.state["consolidated_task_ids"]:
            memory = self.state["memories"][task_id]
            queries = memory["queries"]
            if queries.shape[0] == 0:
                continue
            if memory["node_feature_sha256"] != feature_hash:
                raise RuntimeError("GEM client node features changed across tasks.")
            if int(memory["num_nodes"]) != int(current_features.shape[0]):
                raise RuntimeError("GEM client node count changed across tasks.")
            logits = context.forward_queries(
                model,
                queries,
                node_features=current_features,
                edge_index=memory["edge_index"],
            )
            replay_mask = memory["class_mask"]
            if str(context.incremental_setting).lower() == "class":
                # Class-IL replay competes over classes visible now, never over
                # output rows belonging to a future task.
                replay_mask = context.valid_class_mask
            logits = _masked_logits(logits, replay_mask)
            loss = _supervised_loss(
                logits, memory["labels"].to(logits.device)
            )
            gradients.append(_flatten_autograd(loss, named_parameters))
        if not gradients:
            return torch.empty(
                (0, sum(parameter.numel() for _, parameter in named_parameters)),
                device=next(model.parameters()).device,
            )
        return torch.stack(gradients)

    def before_round(self, context: Any) -> None:
        self._validate_scenario(context)
        self._current_feature_sha256 = _tensor_sha256(context.node_features)

    def augment_loss(
        self,
        model: nn.Module,
        context: Any,
        logits: torch.Tensor,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        del logits
        self._validate_scenario(context)
        named = _named_trainable_parameters(model)
        self._pending_memory_gradients = self._memory_gradients(
            model, context, named
        )
        self._pending_parameter_names = tuple(name for name, _ in named)
        return base_loss

    def after_backward(self, model: nn.Module, context: Any) -> None:
        self._validate_scenario(context)
        named = _named_trainable_parameters(model)
        if tuple(name for name, _ in named) != self._pending_parameter_names:
            raise RuntimeError("GEM parameter identity changed within a local update.")
        memories = self._pending_memory_gradients
        self._pending_memory_gradients = None
        self._pending_parameter_names = ()
        if memories is None:
            raise RuntimeError("GEM after_backward requires a preceding augment_loss call.")
        current = _flatten_parameter_gradients(named)
        before = memories @ current if memories.shape[0] else current.new_empty((0,))
        projected = project_gem_gradient(
            current,
            memories,
            margin=self.margin,
            epsilon=self.projection_epsilon,
            violation_tolerance=self.violation_tolerance,
        )
        after = memories @ projected if memories.shape[0] else current.new_empty((0,))
        applied = not torch.equal(projected, current)
        if applied:
            _assign_parameter_gradients(named, projected)
            self.state["projection_count"] += 1
        self._last_diagnostics = {
            "constraint_count": int(memories.shape[0]),
            "violated_constraint_count": int(
                (before < -self.violation_tolerance).sum()
            ),
            "minimum_dot_before": 0.0 if before.numel() == 0 else float(before.min()),
            "minimum_dot_after": 0.0 if after.numel() == 0 else float(after.min()),
            "projection_applied": applied,
        }

    def _sample_indices(self, count: int, task_id: int) -> torch.Tensor:
        digest = hashlib.sha256(
            f"uefa-gem-v1:{self.seed}:{self.client_id}:{task_id}".encode("ascii")
        ).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))
        return torch.randperm(count, generator=generator)

    def _rebalance_existing_memories(self, task_ids: Sequence[int]) -> Dict[int, int]:
        base, remainder = divmod(self.memory_size, len(task_ids))
        quotas = {
            task_id: base + (index < remainder)
            for index, task_id in enumerate(sorted(task_ids))
        }
        for task_id in self.state["consolidated_task_ids"]:
            memory = self.state["memories"][task_id]
            keep = quotas[task_id]
            memory["queries"] = memory["queries"][:keep].clone()
            memory["labels"] = memory["labels"][:keep].clone()
        return quotas

    def consolidate(self, model: nn.Module, context: Any) -> None:
        del model
        self._validate_scenario(context)
        task_id = int(context.global_task_id)
        if task_id in self.state["consolidated_task_ids"]:
            raise ValueError(f"GEM task {task_id} has already been consolidated.")
        task_ids = [*self.state["consolidated_task_ids"], task_id]
        quotas = self._rebalance_existing_memories(task_ids)
        queries = context.train_queries.detach().cpu().long()
        labels = context.train_labels.detach().cpu().clone()
        selected = self._sample_indices(queries.shape[0], task_id)[: quotas[task_id]]
        self.state["memories"][task_id] = {
            "queries": queries[selected].clone(),
            "labels": labels[selected].clone(),
            "class_mask": self._task_mask(context),
            "edge_index": context.effective_edge_index.detach().cpu().long().clone(),
            "num_nodes": int(context.node_features.shape[0]),
            "node_feature_sha256": _tensor_sha256(context.node_features),
        }
        self.state["consolidated_task_ids"].append(task_id)
        self._current_feature_sha256 = None

    def replay_payload_bytes(self) -> int:
        total = 0
        for memory in self.state["memories"].values():
            for value in memory.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total

    def private_state_checksum(self) -> str:
        digest = hashlib.sha256()
        snapshot = super().save_method_state()

        def update(value: Any) -> None:
            if torch.is_tensor(value):
                digest.update(_tensor_sha256(value).encode("ascii"))
            elif isinstance(value, Mapping):
                for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                    update(key)
                    update(value[key])
            elif isinstance(value, (list, tuple)):
                for item in value:
                    update(item)
            else:
                digest.update(repr(value).encode("utf-8"))

        update(snapshot)
        return digest.hexdigest()

    def on_broadcast(self, context: Any, payload: Any) -> None:
        del context, payload
        # No server payload is allowed to mutate client-private episodic memory.
        return None

    def diagnostics(self) -> Dict[str, object]:
        remembered = sum(
            int(memory["queries"].shape[0])
            for memory in self.state["memories"].values()
        )
        return {
            "method": self.name,
            "state_sha256": self.private_state_checksum(),
            "consolidated_task_ids": tuple(self.state["consolidated_task_ids"]),
            "num_consolidated_tasks": len(self.state["consolidated_task_ids"]),
            "remembered_query_count": remembered,
            "replay_payload_bytes": self.replay_payload_bytes(),
            "projection_count": int(self.state["projection_count"]),
            **self._last_diagnostics,
        }

    def _validate_loaded_state(self) -> None:
        required = {
            "state_version",
            "method_hyperparameters",
            "memories",
            "consolidated_task_ids",
            "projection_count",
        }
        if set(self.state) != required or self.state["state_version"] != _STATE_VERSION:
            raise ValueError("GEM checkpoint fields or state version do not match.")
        if self.state["method_hyperparameters"] != self._private_hyperparameters():
            raise ValueError("GEM checkpoint hyperparameters do not match this method.")
        task_ids = self.state["consolidated_task_ids"]
        if (
            not isinstance(task_ids, list)
            or len(task_ids) != len(set(task_ids))
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in task_ids)
        ):
            raise ValueError("GEM consolidated task IDs are invalid.")
        memories = self.state["memories"]
        if not isinstance(memories, Mapping) or set(memories) != set(task_ids):
            raise ValueError("GEM memory task IDs do not match consolidation IDs.")
        if (
            isinstance(self.state["projection_count"], bool)
            or not isinstance(self.state["projection_count"], int)
            or self.state["projection_count"] < 0
        ):
            raise ValueError("GEM projection count is invalid.")
        total = 0
        for task_id in task_ids:
            memory = memories[task_id]
            if not isinstance(memory, Mapping) or set(memory) != {
                "queries",
                "labels",
                "class_mask",
                "edge_index",
                "num_nodes",
                "node_feature_sha256",
            }:
                raise ValueError("GEM memory fields do not match the state schema.")
            queries, labels = memory["queries"], memory["labels"]
            if (
                not torch.is_tensor(queries)
                or queries.dtype != torch.long
                or not torch.is_tensor(labels)
                or labels.shape[0] != queries.shape[0]
                or queries.device.type != "cpu"
                or labels.device.type != "cpu"
            ):
                raise ValueError("GEM replay queries or labels are invalid.")
            num_nodes = memory["num_nodes"]
            if (
                isinstance(num_nodes, bool)
                or not isinstance(num_nodes, int)
                or num_nodes < 1
            ):
                raise ValueError("GEM replay node count is invalid.")
            expected_query_shape = (
                queries.ndim == 1
                if self.problem_type == "NC"
                else queries.ndim == 2 and queries.shape[1] == 2
            )
            if not expected_query_shape or (
                queries.numel()
                and (int(queries.min()) < 0 or int(queries.max()) >= num_nodes)
            ):
                raise ValueError("GEM replay query endpoints are invalid.")
            edge_index = memory["edge_index"]
            if (
                not torch.is_tensor(edge_index)
                or edge_index.dtype != torch.long
                or edge_index.ndim != 2
                or edge_index.shape[0] != 2
                or edge_index.device.type != "cpu"
                or (
                    edge_index.numel()
                    and (
                        int(edge_index.min()) < 0
                        or int(edge_index.max()) >= num_nodes
                    )
                )
            ):
                raise ValueError("GEM replay topology is invalid.")
            class_mask = memory["class_mask"]
            if class_mask is not None and (
                not torch.is_tensor(class_mask)
                or class_mask.dtype != torch.bool
                or class_mask.ndim != 1
                or class_mask.device.type != "cpu"
            ):
                raise ValueError("GEM replay class mask is invalid.")
            feature_hash = memory["node_feature_sha256"]
            if not isinstance(feature_hash, str) or len(feature_hash) != 64:
                raise ValueError("GEM node-feature checksum is invalid.")
            total += int(queries.shape[0])
        if total > self.memory_size:
            raise ValueError("GEM replay state exceeds memory_size.")

    def load_method_state(self, state: Mapping[str, object]) -> None:
        previous = super().save_method_state()
        try:
            super().load_method_state(state)
            self._validate_loaded_state()
            self._pending_memory_gradients = None
            self._pending_parameter_names = ()
            self._current_feature_sha256 = None
        except Exception:
            super().load_method_state(previous)
            raise


__all__ = ["GEMAlgorithm", "project_gem_gradient"]
