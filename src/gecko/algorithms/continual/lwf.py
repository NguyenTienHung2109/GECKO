from __future__ import annotations

import copy
from typing import Dict
import torch
import torch.nn.functional as F
from gecko.algorithms.base import ClientContinualAlgorithm

class LwFAlgorithm(ClientContinualAlgorithm):
    """Paper-style LwF for task- and domain-incremental streams."""

    name = "LwF"
    method_version = "uefa_federated_lwf_v1"

    def __init__(
        self,
        *,
        regularization: float = 1.0,
        temperature: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(regularization=regularization, **kwargs)
        self.temperature = temperature

    def additional_loss(self, model, forward, queries, logits, labels, global_task_id):
        teacher = self.state.get("teacher")
        if teacher is None:
            return logits.sum() * 0.0
        teacher_model = copy.deepcopy(model)
        teacher_model.load_state_dict(teacher)
        teacher_model.eval()
        with torch.no_grad():
            teacher_logits = forward(teacher_model, queries)
        if labels.ndim > 1:
            teacher_targets = torch.sigmoid(
                teacher_logits.detach() / self.temperature
            )
            distillation = F.binary_cross_entropy_with_logits(
                logits / self.temperature,
                teacher_targets,
            ) * (self.temperature ** 2)
            return self.regularization * distillation
        if logits.ndim == 1 or logits.shape[-1] == 1:
            teacher_scores = teacher_logits.reshape(-1).detach()
            student_scores = logits.reshape(-1)
            scale = 2.0 * self.temperature
            distillation = -(
                torch.sigmoid(teacher_scores / scale)
                * F.logsigmoid(student_scores / scale)
                + torch.sigmoid(-teacher_scores / scale)
                * F.logsigmoid(-student_scores / scale)
            ).mean()
            return self.regularization * distillation

        if self.incremental_setting == "task":
            previous_masks = self.state.get("previous_task_masks", ())
        else:
            previous_masks = (self.state["previous_class_mask"],)
        distillation = logits.sum() * 0.0
        for previous_mask in previous_masks:
            mask = previous_mask.to(logits.device)
            teacher_probabilities = F.softmax(
                teacher_logits[..., mask].detach() / self.temperature,
                dim=-1,
            )
            student_log_probabilities = F.log_softmax(
                logits[..., mask] / self.temperature,
                dim=-1,
            )
            distillation = distillation - (
                teacher_probabilities * student_log_probabilities
            ).sum(dim=-1).mean()
        return self.regularization * distillation

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
        logits = forward(model, queries)
        binary_output = logits.ndim == 1 or logits.shape[-1] == 1
        if not binary_output and class_mask is None:
            class_mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        self.state["teacher"] = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        if not binary_output:
            if self.incremental_setting == "task":
                task_masks = list(self.state.get("previous_task_masks", ()))
                task_masks.append(class_mask.detach().cpu().clone())
                self.state["previous_task_masks"] = tuple(task_masks)
            else:
                self.state["previous_class_mask"] = class_mask.detach().cpu().clone()
        self.state["consolidated_task_ids"] = tuple(
            (*self.state.get("consolidated_task_ids", ()), int(global_task_id))
        )

    def hyperparameters(self):
        output = super().hyperparameters()
        output["temperature"] = self.temperature
        return output

    def diagnostics(self):
        teacher = self.state.get("teacher", {})
        teacher_bytes = sum(
            value.numel() * value.element_size()
            for value in teacher.values()
            if torch.is_tensor(value)
        )
        return {
            "private_state_policy": "client_local_only",
            "teacher_present": bool(teacher),
            "teacher_bytes": int(teacher_bytes),
            "consolidated_task_ids": list(
                self.state.get("consolidated_task_ids", ())
            ),
            "stores_old_queries": False,
            "stores_old_topology": False,
        }


class LwFClassILAlgorithm(LwFAlgorithm):
    """Sample-free LwF adaptation for single-head class-incremental learning.

    Softmax distillation restricted to old classes cannot observe a common
    shift of every old-class logit.  Independent Bernoulli distillation keeps
    that direction identifiable while retaining LwF's teacher-only memory.
    """

    name = "LwFClassIL"
    method_version = "uefa_federated_lwf_class_il_v1"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.problem_type not in {None, "NC"}:
            raise ValueError("LwFClassIL is implemented only for node classification.")
        if self.incremental_setting not in {None, "class"}:
            raise ValueError(
                "LwFClassIL requires a single-head class-incremental stream."
            )
        self._last_loss_diagnostics: Dict[str, object] = {
            "distillation_loss": None,
            "weighted_distillation_loss": 0.0,
            "old_class_count": 0,
            "temperature": self.temperature,
            "stores_old_samples": False,
        }

    def additional_loss(self, model, forward, queries, logits, labels, global_task_id):
        teacher = self.state.get("teacher")
        previous_mask = self.state.get("previous_class_mask")
        if teacher is None or previous_mask is None:
            self._last_loss_diagnostics = {
                "distillation_loss": None,
                "weighted_distillation_loss": 0.0,
                "old_class_count": 0,
                "temperature": self.temperature,
                "stores_old_samples": False,
            }
            return logits.sum() * 0.0
        if logits.ndim < 2 or logits.shape[-1] == 1:
            raise RuntimeError("LwFClassIL requires multiclass single-head logits.")

        mask = previous_mask.to(logits.device)
        teacher_model = copy.deepcopy(model)
        teacher_model.load_state_dict(teacher)
        teacher_model.eval()
        with torch.no_grad():
            teacher_logits = forward(teacher_model, queries)
            teacher_targets = torch.sigmoid(
                teacher_logits[..., mask] / self.temperature
            )
        distillation = F.binary_cross_entropy_with_logits(
            logits[..., mask] / self.temperature,
            teacher_targets,
        ) * (self.temperature ** 2)
        weighted = self.regularization * distillation
        self._last_loss_diagnostics = {
            "distillation_loss": float(distillation.detach()),
            "weighted_distillation_loss": float(weighted.detach()),
            "old_class_count": int(mask.sum()),
            "temperature": self.temperature,
            "stores_old_samples": False,
        }
        return weighted

    def diagnostics(self):
        return dict(self._last_loss_diagnostics)


