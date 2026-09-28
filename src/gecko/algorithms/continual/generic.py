from __future__ import annotations

from typing import Dict
import torch
from torch import nn
from gecko.algorithms.base import ClientContinualAlgorithm
from gecko.algorithms.continual.mas import MASAlgorithm

class GenericImportanceRegularizationAlgorithm(MASAlgorithm):
    """Experimental importance regularizer without TWP-specific semantics."""

    name = "generic_importance_regularization"


class GenericReplayAlgorithm(ClientContinualAlgorithm):
    """Experimental bounded replay loss without a named-method attribution."""

    name = "generic_replay"

    def additional_loss(self, model, forward, queries, logits, labels, global_task_id):
        from gecko.algorithms.continual._common import _classification_loss
        memory = self.state.get("memory", [])
        if not memory:
            return logits.sum() * 0.0
        losses = []
        for replay_queries, replay_labels, _ in memory:
            replay_logits = forward(model, replay_queries.to(queries.device))
            losses.append(_classification_loss(replay_logits, replay_labels.to(labels.device)))
        return self.regularization * torch.stack(losses).mean()

    def after_update(self, model, queries, labels, global_task_id):
        memory = self.state.setdefault("memory", [])
        count = min(self.memory_size, queries.shape[0])
        memory.append(
            (
                queries[:count].detach().cpu().clone(),
                labels[:count].detach().cpu().clone(),
                global_task_id,
            )
        )
        while sum(item[0].shape[0] for item in memory) > self.memory_size and len(memory) > 1:
            memory.pop(0)


class GenericWeightIsolationAlgorithm(ClientContinualAlgorithm):
    """Experimental task-keyed magnitude mask without a named attribution."""

    name = "generic_weight_isolation"

    def mask_gradients(self, model: nn.Module, global_task_id: int) -> None:
        masks: Dict[int, Dict[str, torch.Tensor]] = self.state.get("task_masks", {})
        if global_task_id not in masks:
            return
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and name in masks[global_task_id]:
                parameter.grad.mul_(masks[global_task_id][name].to(parameter.grad))

    def after_update(self, model, queries, labels, global_task_id):
        masks = self.state.setdefault("task_masks", {})
        if global_task_id not in masks:
            masks[global_task_id] = {
                name: (parameter.detach().abs() >= parameter.detach().abs().median()).float().cpu()
                for name, parameter in model.named_parameters()
            }


