from __future__ import annotations

import torch
from gecko.algorithms.base import ClientContinualAlgorithm

class MASAlgorithm(ClientContinualAlgorithm):
    name = "MAS"
    method_version = "uefa_federated_mas_v1"

    def additional_loss(self, model, forward, queries, logits, labels, global_task_id):
        importances = self.state.get("importances")
        anchor = self.state.get("params")
        if importances is None or anchor is None:
            return logits.sum() * 0.0
        penalty = logits.sum() * 0.0
        for name, parameter in model.named_parameters():
            penalty = penalty + (
                importances[name].to(parameter)
                * (parameter - anchor[name].to(parameter)).square()
            ).sum()
        return self.regularization * penalty

    def after_task(
        self,
        model,
        forward,
        queries,
        labels,
        global_task_id,
        class_mask=None,
        node_features=None,
    ):
        from gecko.algorithms.continual._common import _masked_logits
        self.state["params"] = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }
        cumulative = self.state.setdefault(
            "importances",
            {
                name: torch.zeros_like(parameter, device="cpu")
                for name, parameter in model.named_parameters()
            },
        )
        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = _masked_logits(forward(model, queries), class_mask)
        if logits.ndim == 1:
            objective = logits.square().mean()
        else:
            objective = (torch.linalg.norm(logits, dim=-1) ** 2).mean()
        objective.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                cumulative[name] += parameter.grad.detach().cpu().abs()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        self.state["consolidated_task_ids"] = tuple(
            (*self.state.get("consolidated_task_ids", ()), int(global_task_id))
        )


