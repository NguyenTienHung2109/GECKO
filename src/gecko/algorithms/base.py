"""Stable client continual-algorithm interface."""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Mapping
from typing import Tuple

import torch
from torch import nn


ForwardFunction = Callable[[nn.Module, torch.Tensor], torch.Tensor]


def _clone_method_state(value: Any, *, path: str = "state") -> Any:
    """Clone checkpoint-safe method state and reject opaque Python objects."""

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(
            _clone_method_state(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _clone_method_state(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        output: Dict[Any, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TypeError(
                    f"Method state key at {path} must be str or int, got "
                    f"{type(key).__name__}."
                )
            output[key] = _clone_method_state(item, path=f"{path}.{key}")
        return output
    raise TypeError(
        f"Method state at {path} contains unsupported {type(value).__name__}; "
        "store only tensors and primitive nested containers."
    )


class ClientContinualAlgorithm:
    """Own persistent local CL state independently of server aggregation."""

    name = "Bare"

    method_version = "1"

    def __init__(
        self,
        *,
        memory_size: int = 128,
        regularization: float = 1.0,
        problem_type: str | None = None,
        incremental_setting: str | None = None,
        client_id: int | None = None,
        seed: int = 0,
    ) -> None:
        self.memory_size = memory_size
        self.regularization = regularization
        self.problem_type = None if problem_type is None else problem_type.upper()
        self.incremental_setting = (
            None if incremental_setting is None else incremental_setting.lower()
        )
        self.client_id = client_id
        self.seed = seed
        self.state: Dict[str, object] = {}

    def additional_loss(
        self,
        model: nn.Module,
        forward: ForwardFunction,
        queries: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_task_id: int,
    ) -> torch.Tensor:
        return logits.sum() * 0.0

    def mask_gradients(self, model: nn.Module, global_task_id: int) -> None:
        return None

    def training_loss(
        self,
        model: nn.Module,
        forward: ForwardFunction,
        queries: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        global_task_id: int,
        base_loss: torch.Tensor,
        class_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compose the supervised objective with method-specific behavior."""

        return base_loss + self.additional_loss(
            model, forward, queries, logits, labels, global_task_id
        )

    def after_update(
        self,
        model: nn.Module,
        queries: torch.Tensor,
        labels: torch.Tensor,
        global_task_id: int,
    ) -> None:
        return None

    def after_task(
        self,
        model: nn.Module,
        forward: ForwardFunction,
        queries: torch.Tensor,
        labels: torch.Tensor,
        global_task_id: int,
        class_mask: torch.Tensor | None = None,
        node_features: torch.Tensor | None = None,
    ) -> None:
        """Consolidate client-local continual state once after a task stage."""

        return None

    def hyperparameters(self) -> Dict[str, object]:
        return {
            "method_version": self.method_version,
            "memory_size": self.memory_size,
            "regularization": self.regularization,
            "problem_type": self.problem_type,
            "incremental_setting": self.incremental_setting,
            "seed": self.seed,
        }

    def local_state_keys(self) -> Tuple[str, ...]:
        return tuple(sorted(self.state))

    # The stateful UEFA v2 lifecycle is additive. The legacy training path
    # continues to call the original hooks above; an explicitly configured v2
    # runtime calls these adapters instead.
    def before_task(self, context: Any) -> None:
        """Prepare private state before a client's immutable global task."""

        return None

    def before_round(self, context: Any) -> None:
        """Prepare private state before a local optimization round."""

        return None

    def augment_loss(
        self,
        model: nn.Module,
        context: Any,
        logits: torch.Tensor,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate a v2 loss request to the established algorithm hook."""

        return self.training_loss(
            model,
            context.forward_queries,
            context.train_queries,
            logits,
            context.train_labels,
            context.global_task_id,
            base_loss,
            context.valid_class_mask,
        )

    def after_backward(self, model: nn.Module, context: Any) -> None:
        """Apply legacy task/class gradient masking after full correction."""

        self.mask_gradients(model, context.global_task_id)

    def after_round(self, model: nn.Module, context: Any) -> None:
        """Delegate a v2 round completion to the established update hook."""

        self.after_update(
            model,
            context.train_queries,
            context.train_labels,
            context.global_task_id,
        )

    def replay_samples(self, context: Any) -> Tuple[Any, ...]:
        """Return method-owned replay records; empty for legacy algorithms."""

        return ()

    def consolidate(self, model: nn.Module, context: Any) -> None:
        """Delegate v2 task consolidation to the established task hook."""

        self.after_task(
            model,
            context.forward_queries,
            context.train_queries,
            context.train_labels,
            context.global_task_id,
            context.valid_class_mask,
            context.node_features,
        )

    def on_broadcast(self, context: Any, payload: Any) -> None:
        """Observe a strategy broadcast without mutating legacy private state."""

        return None

    def evaluation_topology(self, context: Any) -> Any:
        """Return a method-owned topology overlay, if one exists."""

        return None

    def replay_payload_bytes(self) -> int:
        """Declare logical replay tensor bytes split from generic client state."""

        return 0

    def topology_overlay_payload_bytes(self) -> int:
        """Declare method-owned overlay tensor bytes split from client state."""

        return 0

    def diagnostics(self) -> Dict[str, object]:
        """Return method diagnostics without exposing private tensor payloads."""

        return {}

    def save_method_state(self) -> Dict[str, object]:
        """Return a detached, CPU, weights-only-compatible state snapshot."""

        return _clone_method_state(self.state)

    def load_method_state(self, state: Mapping[str, object]) -> None:
        """Restore private state while preserving the public state-dict identity."""

        cloned = _clone_method_state(state)
        if not isinstance(cloned, dict):
            raise TypeError("Method checkpoint state must be a mapping.")
        self.state.clear()
        self.state.update(cloned)
