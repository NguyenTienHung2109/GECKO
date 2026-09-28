"""Persistent client lifecycle with reset-per-round optimizer ownership."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict
from typing import Iterable
from typing import Set

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from gecko.config import GECKOConfig
from gecko.algorithms.base import ClientContinualAlgorithm
from gecko.algorithms.context import ClientMethodContext
from gecko.types import ClientGraphView
from gecko.types import ClientScenarioView
from gecko.types import ClientState
from gecko.types import ClientTaskShard
from gecko.types import LocalUpdateResult
from gecko.engine.aggregation import fedprox_penalty
from gecko.engine.parameter_policy import SharedParameterPolicy
from gecko.engine.protocol import ClientUpload
from gecko.engine.protocol import StatefulStrategyProtocol


def supervised_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim > 1:
        valid = labels >= 0
        if not valid.any():
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(logits[valid], labels[valid].float())
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(logits.reshape(-1), labels.float().reshape(-1))
    return F.cross_entropy(logits, labels.long())


@dataclass
class FederatedClient:
    client_id: int
    graph: ClientGraphView
    model: nn.Module
    algorithm: ClientContinualAlgorithm
    config: GECKOConfig
    scenario: ClientScenarioView
    parameter_policy: SharedParameterPolicy
    class_mask_policy: str = "seen"

    def __post_init__(self) -> None:
        if self.class_mask_policy not in {"seen", "full"}:
            raise ValueError("class_mask_policy must be 'seen' or 'full'.")
        self.state = ClientState(self.client_id)
        self.state.continual_state = self.algorithm.state

    def load_shared_state(self, shared_state: Dict[str, torch.Tensor]) -> None:
        self.parameter_policy.load(self.model, shared_state)

    def _class_mask(self, shard: ClientTaskShard) -> torch.Tensor | None:
        if self.class_mask_policy == "full":
            return None
        if self.scenario.incremental_type == "task":
            return shard.task_class_mask
        if self.scenario.incremental_type == "class" and shard.task_class_mask is not None:
            if self.state.seen_class_mask is None:
                return shard.task_class_mask
            return self.state.seen_class_mask | shard.task_class_mask
        return None

    def _forward(
        self,
        model: nn.Module,
        queries: torch.Tensor,
        context_edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        edge_index = (
            self.graph.edge_index
            if context_edge_index is None
            else context_edge_index
        )
        return model.forward_queries(
            self.graph.node_features.to(device),
            edge_index.to(device),
            queries.to(device),
            self.scenario.problem_type,
        )

    def _masked_logits(self, logits: torch.Tensor, shard: ClientTaskShard) -> torch.Tensor:
        mask = self._class_mask(shard)
        if mask is None or logits.ndim < 2 or logits.shape[-1] != mask.shape[0]:
            return logits
        output = logits.clone()
        output[..., ~mask.to(logits.device)] = -1e12
        return output

    def update(
        self,
        shard: ClientTaskShard,
        *,
        global_task_id: int,
        server_state: Dict[str, torch.Tensor] | None,
        strategy: str,
    ) -> LocalUpdateResult:
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        device = next(self.model.parameters()).device
        queries = shard.train_queries.to(device)
        labels = shard.train_labels.to(device)
        if labels.shape[0] == 0:
            shared = self.parameter_policy.extract(self.model)
            return LocalUpdateResult(
                self.client_id, global_task_id, shared, 0, 0.0, 0, tuple(shared)
            )
        last_loss = 0.0
        for _ in range(self.config.training.local_epochs_per_round):
            optimizer.zero_grad()
            forward = lambda model, values: self._forward(
                model, values, shard.context_edge_index
            )
            logits = forward(self.model, queries)
            masked_logits = self._masked_logits(logits, shard)
            base_loss = supervised_loss(masked_logits, labels)
            loss = self.algorithm.training_loss(
                self.model,
                forward,
                queries,
                logits,
                labels,
                global_task_id,
                base_loss,
                self._class_mask(shard),
            )
            if strategy == "fedprox":
                if server_state is None:
                    raise ValueError("FedProx requires a server shared state.")
                if self.config.training.fedprox_mu != 0.0:
                    loss = loss + fedprox_penalty(
                        self.model,
                        server_state,
                        self.parameter_policy.shareable_keys(self.model),
                        self.config.training.fedprox_mu,
                    )
            loss.backward()
            self.algorithm.mask_gradients(self.model, global_task_id)
            optimizer.step()
            last_loss = float(loss.detach())
        self.algorithm.after_update(self.model, queries, labels, global_task_id)
        shared = self.parameter_policy.extract(self.model)
        communication_bytes = sum(tensor.numel() * tensor.element_size() for tensor in shared.values())
        return LocalUpdateResult(
            client_id=self.client_id,
            global_task_id=global_task_id,
            shared_state=shared,
            weight=(
                shard.positive_anchor_count
                if self.config.training.aggregation_weight
                == "current_positive_anchor_count"
                else shard.supervised_query_count
            ),
            training_loss=last_loss,
            communication_bytes=communication_bytes,
            shareable_keys=tuple(shared),
        )


    def build_method_context(
        self,
        shard: ClientTaskShard,
        *,
        global_task_id: int,
        stage_index: int,
        round_index: int,
    ) -> ClientMethodContext:
        """Build an owned strict-local context for an explicit v2 method."""

        if global_task_id != shard.global_task_id:
            raise ValueError("Shard/global task identity mismatch.")
        device = next(self.model.parameters()).device

        def forward_capability(
            model: nn.Module,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
            queries: torch.Tensor,
        ) -> torch.Tensor:
            model_device = next(model.parameters()).device
            return model.forward_queries(
                node_features.to(model_device),
                edge_index.to(model_device),
                queries.to(model_device),
                self.scenario.problem_type,
            )

        def encode_capability(
            model: nn.Module,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
            layer_index: int | None,
        ) -> torch.Tensor:
            if not hasattr(model, "encode_nodes"):
                raise RuntimeError(
                    f"{type(model).__name__} does not expose encode_nodes."
                )
            model_device = next(model.parameters()).device
            return model.encode_nodes(
                node_features.to(model_device),
                edge_index.to(model_device),
                layer_index=layer_index,
            )

        class_mask = self._class_mask(shard)
        return ClientMethodContext(
            client_id=self.client_id,
            global_task_id=global_task_id,
            stage_index=stage_index,
            round_index=round_index,
            problem_type=self.scenario.problem_type,
            incremental_setting=self.scenario.incremental_type,
            train_queries=shard.train_queries.to(device),
            train_labels=shard.train_labels.to(device),
            valid_class_mask=(
                None if class_mask is None else class_mask.to(device)
            ),
            node_features=self.graph.node_features,
            base_edge_index=self.graph.edge_index,
            context_edge_index=shard.context_edge_index,
            forward_queries=forward_capability,
            encode_nodes=encode_capability,
        )

    @staticmethod
    def _masked_logits_for_context(
        logits: torch.Tensor,
        class_mask: torch.Tensor | None,
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

    def begin_task_stateful(self, context: ClientMethodContext) -> None:
        """Invoke the additive task lifecycle without changing legacy runs."""

        self.algorithm.before_task(context)

    def update_stateful(
        self,
        context: ClientMethodContext,
        *,
        server_state: Dict[str, torch.Tensor] | None,
        strategy: StatefulStrategyProtocol,
    ) -> ClientUpload:
        """Run the explicit v2 lifecycle with strategy gradient transforms."""

        if (
            context.client_id != self.client_id
            or context.global_task_id < 0
        ):
            raise ValueError("Method context belongs to a different client/task.")
        self.model.train()
        shared_keys = self.parameter_policy.shareable_keys(self.model)
        strategy_trainable_parameters = getattr(
            strategy, "trainable_parameters", None
        )
        extra_parameters = (
            tuple(strategy_trainable_parameters(self.model, context, shared_keys))
            if callable(strategy_trainable_parameters)
            else ()
        )
        optimizer_learning_rate = float(self.config.training.learning_rate)
        optimizer_weight_decay = float(self.config.training.weight_decay)
        strategy_optimizer_hyperparameters = getattr(
            strategy, "local_optimizer_hyperparameters", None
        )
        if callable(strategy_optimizer_hyperparameters):
            resolved_optimizer_hyperparameters = strategy_optimizer_hyperparameters(
                context,
                default_learning_rate=optimizer_learning_rate,
                default_weight_decay=optimizer_weight_decay,
            )
            if (
                not isinstance(resolved_optimizer_hyperparameters, tuple)
                or len(resolved_optimizer_hyperparameters) != 2
            ):
                raise ValueError(
                    "Strategy local_optimizer_hyperparameters must return "
                    "(learning_rate, weight_decay)."
                )
            optimizer_learning_rate = float(
                resolved_optimizer_hyperparameters[0]
            )
            optimizer_weight_decay = float(
                resolved_optimizer_hyperparameters[1]
            )
            if (
                not math.isfinite(optimizer_learning_rate)
                or optimizer_learning_rate <= 0.0
                or not math.isfinite(optimizer_weight_decay)
                or optimizer_weight_decay < 0.0
            ):
                raise ValueError(
                    "Strategy optimizer learning rate/weight decay are invalid."
                )
        optimizer = torch.optim.Adam(
            tuple(self.model.parameters()) + extra_parameters,
            lr=optimizer_learning_rate,
            weight_decay=optimizer_weight_decay,
        )
        queries = context.train_queries
        labels = context.train_labels
        self.algorithm.before_round(context)
        if labels.shape[0] == 0:
            shared = self.parameter_policy.extract(self.model)
            result = LocalUpdateResult(
                self.client_id,
                context.global_task_id,
                shared,
                0,
                0.0,
                0,
                tuple(shared),
            )
            return strategy.finalize_upload(self, result, context)

        strategy_inner_steps = getattr(
            strategy, "local_optimizer_steps_per_epoch", 1
        )
        if (
            isinstance(strategy_inner_steps, bool)
            or not isinstance(strategy_inner_steps, int)
            or strategy_inner_steps <= 0
        ):
            raise ValueError(
                "Strategy local_optimizer_steps_per_epoch must be a positive integer."
            )
        optimizer_steps = (
            self.config.training.local_epochs_per_round * strategy_inner_steps
        )
        last_loss = 0.0
        for _ in range(optimizer_steps):
            optimizer.zero_grad()
            strategy_forward_queries = getattr(strategy, "forward_queries", None)
            logits = (
                strategy_forward_queries(self.model, context, queries)
                if callable(strategy_forward_queries)
                else context.forward_queries(self.model, queries)
            )
            strategy_training_class_mask = getattr(
                strategy, "training_class_mask", None
            )
            training_class_mask = (
                strategy_training_class_mask(context)
                if callable(strategy_training_class_mask)
                else context.valid_class_mask
            )
            if training_class_mask is not None and (
                not torch.is_tensor(training_class_mask)
                or training_class_mask.dtype != torch.bool
                or training_class_mask.ndim != 1
                or (
                    logits.ndim >= 2
                    and logits.shape[-1] != training_class_mask.numel()
                )
            ):
                raise ValueError(
                    "Strategy training_class_mask must return None or the "
                    "valid one-dimensional bool output mask."
                )
            masked_logits = self._masked_logits_for_context(
                logits, training_class_mask
            )
            base_loss = supervised_loss(masked_logits, labels)
            loss = self.algorithm.augment_loss(
                self.model,
                context,
                logits,
                base_loss,
            )
            strategy_augment_loss = getattr(strategy, "augment_loss", None)
            if callable(strategy_augment_loss):
                loss = strategy_augment_loss(
                    self.model,
                    context,
                    loss,
                    shared_keys,
                )
            if strategy.uses_proximal_objective:
                if server_state is None:
                    raise ValueError("FedProx requires a server shared state.")
                if self.config.training.fedprox_mu != 0.0:
                    loss = loss + fedprox_penalty(
                        self.model,
                        server_state,
                        shared_keys,
                        self.config.training.fedprox_mu,
                    )
            loss.backward()
            strategy_after_backward = getattr(strategy, "after_backward", None)
            if callable(strategy_after_backward):
                strategy_after_backward(
                    self.model,
                    context,
                    shared_keys,
                    self.config.training.learning_rate,
                )
            strategy.transform_gradients(self.model, context, shared_keys)
            if training_class_mask is not None:
                mask_output_gradients = getattr(
                    self.model, "mask_output_gradients", None
                )
                if not callable(mask_output_gradients):
                    raise RuntimeError(
                        "Explicit v2 NC Task/Class execution requires the "
                        "verified output-gradient mask capability."
                    )
                mask_output_gradients(training_class_mask)
            # GEM must see the final current-task masked gradient so its
            # projection may add remembered-task components afterwards.
            self.algorithm.after_backward(self.model, context)
            optimizer.step()
            strategy_after_optimizer_step = getattr(
                strategy, "after_optimizer_step", None
            )
            if callable(strategy_after_optimizer_step):
                strategy_after_optimizer_step(self.model, context, shared_keys)
            last_loss = float(loss.detach())
        self.algorithm.after_round(self.model, context)
        shared = self.parameter_policy.extract(self.model)
        communication_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in shared.values()
        )
        if (
            self.config.training.aggregation_weight
            == "current_positive_anchor_count"
        ):
            weight = (
                int((labels == 1).sum())
                if labels.ndim == 1
                else int(labels.shape[0])
            )
        else:
            weight = int(labels.shape[0])
        result = LocalUpdateResult(
            client_id=self.client_id,
            global_task_id=context.global_task_id,
            shared_state=shared,
            weight=weight,
            training_loss=last_loss,
            communication_bytes=communication_bytes,
            shareable_keys=tuple(shared),
        )
        return strategy.finalize_upload(self, result, context)

    def consolidate_task_stateful(self, context: ClientMethodContext) -> None:
        """Consolidate a v2 method once after the client's completed stage."""

        if context.train_labels.shape[0] == 0:
            return
        self.algorithm.consolidate(self.model, context)


    def control_gradient_at_shared(
        self,
        context: ClientMethodContext,
        shared_state: Dict[str, torch.Tensor],
        shared_keys: tuple[str, ...],
    ) -> Dict[str, torch.Tensor]:
        """Evaluate a projected local-objective gradient at a broadcast state.

        This isolated Option-I pass restores model, method, gradient, mode,
        cache, and global RNG state even when gradient evaluation fails.
        """

        if tuple(shared_keys) != self.parameter_policy.shareable_keys(self.model):
            raise ValueError("Control-gradient keys do not match the parameter policy.")
        model_state = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        dynamic_state = (
            self.model.export_dynamic_state()
            if hasattr(self.model, "export_dynamic_state")
            else {}
        )
        cache_state = {
            name: getattr(self.model, name)
            for name in ("_cached_graph_key", "_cached_graph")
            if hasattr(self.model, name)
        }
        method_state = self.algorithm.save_method_state()
        gradients = {
            name: (
                None
                if parameter.grad is None
                else parameter.grad.detach().clone()
            )
            for name, parameter in self.model.named_parameters()
        }
        training_modes = {
            name: module.training for name, module in self.model.named_modules()
        }
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        numpy_rng_state = (
            numpy_rng_state[0],
            numpy_rng_state[1].copy(),
            numpy_rng_state[2],
            numpy_rng_state[3],
            numpy_rng_state[4],
        )
        torch_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            self.load_shared_state(shared_state)
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            queries = context.train_queries
            labels = context.train_labels
            if labels.shape[0] == 0:
                return {
                    name: torch.zeros_like(
                        dict(self.model.named_parameters())[name],
                        device="cpu",
                    )
                    for name in shared_keys
                }
            logits = context.forward_queries(self.model, queries)
            masked_logits = self._masked_logits_for_context(
                logits, context.valid_class_mask
            )
            base_loss = supervised_loss(masked_logits, labels)
            loss = self.algorithm.augment_loss(
                self.model,
                context,
                logits,
                base_loss,
            )
            loss.backward()
            if context.valid_class_mask is not None:
                mask_output_gradients = getattr(
                    self.model, "mask_output_gradients", None
                )
                if not callable(mask_output_gradients) and self.algorithm.name == "GEM":
                    raise RuntimeError(
                        "Explicit v2 NC/LC Task/Class execution requires the "
                        "verified output-gradient mask capability."
                    )
                if callable(mask_output_gradients):
                    mask_output_gradients(context.valid_class_mask)
            self.algorithm.after_backward(self.model, context)
            parameters = dict(self.model.named_parameters())
            controls = {}
            for name in shared_keys:
                gradient = parameters[name].grad
                controls[name] = (
                    torch.zeros_like(parameters[name], device="cpu")
                    if gradient is None
                    else gradient.detach().cpu().clone()
                )
                if not torch.isfinite(controls[name]).all():
                    raise RuntimeError(
                        f"Non-finite Option-I control gradient for {name}."
                    )
            return controls
        finally:
            self.model.load_state_dict(model_state, strict=True)
            if hasattr(self.model, "load_dynamic_state"):
                self.model.load_dynamic_state(dynamic_state)
            for name, value in cache_state.items():
                setattr(self.model, name, value)
            self.algorithm.load_method_state(method_state)
            for name, parameter in self.model.named_parameters():
                previous = gradients[name]
                parameter.grad = (
                    None
                    if previous is None
                    else previous.to(parameter.device).clone()
                )
            for name, module in self.model.named_modules():
                module.training = training_modes[name]
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)



    def mark_seen(
        self,
        global_task_id: int,
        task_class_mask: torch.Tensor | None = None,
    ) -> None:
        self.state.seen_tasks.add(global_task_id)
        if task_class_mask is not None:
            if self.state.seen_class_mask is None:
                self.state.seen_class_mask = task_class_mask.detach().cpu().clone()
            else:
                self.state.seen_class_mask |= task_class_mask.detach().cpu()

    def consolidate_task(
        self,
        shard: ClientTaskShard,
        *,
        global_task_id: int,
    ) -> None:
        """Update continual state once after this client finishes a task stage."""

        device = next(self.model.parameters()).device
        queries = shard.train_queries.to(device)
        labels = shard.train_labels.to(device)
        if labels.shape[0] == 0:
            return
        forward = lambda model, values: self._forward(
            model, values, shard.context_edge_index
        )
        self.algorithm.after_task(
            self.model,
            forward,
            queries,
            labels,
            global_task_id,
            self._class_mask(shard),
            self.graph.node_features,
        )
