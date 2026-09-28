from __future__ import annotations

import torch
from gecko.algorithms.base import ClientContinualAlgorithm

class EWCAlgorithm(ClientContinualAlgorithm):
    name = "EWC"
    method_version = "uefa_federated_ewc_v1"

    def __init__(self, *, regularization: float = 10000.0, **kwargs) -> None:
        super().__init__(regularization=regularization, **kwargs)

    def additional_loss(self, model, forward, queries, logits, labels, global_task_id):
        anchors = self.state.get("anchors", {})
        fishers = self.state.get("fishers", {})
        penalty = logits.sum() * 0.0
        for task_id, anchor in anchors.items():
            fisher = fishers[task_id]
            for name, parameter in model.named_parameters():
                penalty = penalty + (fisher[name].to(parameter) * (parameter - anchor[name].to(parameter)).square()).sum()
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
        from gecko.algorithms.continual._common import _classification_loss
        from gecko.algorithms.continual._common import _masked_logits
        anchors = self.state.setdefault("anchors", {})
        fishers = self.state.setdefault("fishers", {})
        anchors[global_task_id] = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }
        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = _masked_logits(forward(model, queries), class_mask)
        _classification_loss(logits, labels).backward()
        fishers[global_task_id] = {}
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            fishers[global_task_id][name] = (
                torch.zeros_like(parameter, device="cpu")
                if gradient is None
                else gradient.detach().cpu().square().clone()
            )
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    def diagnostics(self):
        anchors = self.state.get("anchors", {})
        fishers = self.state.get("fishers", {})
        anchor_bytes = sum(
            value.numel() * value.element_size()
            for task in anchors.values()
            for value in task.values()
            if torch.is_tensor(value)
        )
        fisher_bytes = sum(
            value.numel() * value.element_size()
            for task in fishers.values()
            for value in task.values()
            if torch.is_tensor(value)
        )
        return {
            "private_state_policy": "client_local_only",
            "consolidated_task_ids": sorted(int(task_id) for task_id in anchors),
            "anchor_bytes": int(anchor_bytes),
            "fisher_bytes": int(fisher_bytes),
            "stores_old_queries": False,
            "stores_old_topology": False,
        }


