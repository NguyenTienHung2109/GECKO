"""Faithful client-local ERGNN replay for node-classification streams."""

from __future__ import annotations

import random
from typing import Iterable

import torch
import torch.nn.functional as F

from gecko.algorithms.base import ClientContinualAlgorithm


class ERGNNAlgorithm(ClientContinualAlgorithm):
    """Paper-faithful experience-node replay with CM/MF/random samplers."""

    name = "ERGNN"
    method_version = "uefa_federated_ergnn_paper_v2"

    def __init__(
        self,
        *,
        num_experience_nodes: int = 1,
        sampler_name: str = "CM",
        distance_threshold: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if self.problem_type not in {None, "NC"}:
            raise ValueError("ERGNN is implemented only for node classification.")
        if num_experience_nodes < 1:
            raise ValueError("num_experience_nodes must be positive.")
        if sampler_name not in {"CM", "MF", "random"}:
            raise ValueError(
                "UEFA ERGNN paper v2 supports CM, MF, and random feature samplers; "
                "representation-based plus variants require an ERGNN-specific model."
            )
        self.num_experience_nodes = num_experience_nodes
        self.sampler_name = sampler_name
        self.distance_threshold = distance_threshold
        self._last_loss_diagnostics: dict[str, object] = {
            "current_loss": None,
            "replay_loss": None,
            "total_loss": None,
            "current_weight": 1.0,
            "replay_weight": 0.0,
            "buffer_size": 0,
            "current_sample_count": 0,
        }

    @staticmethod
    def _group_by_class(
        queries: torch.Tensor, labels: torch.Tensor
    ) -> list[list[int]]:
        grouped: list[list[int]] = []
        labels_cpu = labels.detach().cpu().long()
        queries_cpu = queries.detach().cpu().long()
        for label in sorted(int(value) for value in labels_cpu.unique()):
            grouped.append(
                queries_cpu[labels_cpu == label].tolist()
            )
        return grouped

    def _random_sample(
        self, grouped: list[list[int]], rng: random.Random
    ) -> list[int]:
        selected: list[int] = []
        for values in grouped:
            selected.extend(rng.sample(values, min(self.num_experience_nodes, len(values))))
        return selected

    def _mf_sample(
        self, grouped: list[list[int]], features: torch.Tensor
    ) -> list[int]:
        selected: list[int] = []
        for values in grouped:
            vectors = features[values]
            center = vectors.mean(dim=0, keepdim=True)
            distances = torch.linalg.vector_norm(vectors - center, dim=-1)
            rank = torch.argsort(distances)
            chosen = rank[: min(self.num_experience_nodes, len(values))].tolist()
            selected.extend(values[index] for index in chosen)
        return selected

    def _cm_sample(
        self,
        grouped: list[list[int]],
        features: torch.Tensor,
        rng: random.Random,
    ) -> list[int]:
        if len(grouped) < 2:
            return self._random_sample(grouped, rng)
        selected: list[int] = []
        distance_budget = 1000
        vectors = features.detach().cpu().half()
        for class_index, values in enumerate(grouped):
            candidates = (
                values
                if len(values) < distance_budget
                else rng.choices(values, k=distance_budget)
            )
            candidate_vectors = vectors[candidates]
            distances = []
            for other_index, other_values in enumerate(grouped):
                if other_index == class_index:
                    continue
                comparison_ids = rng.choices(
                    other_values, k=min(distance_budget, len(other_values))
                )
                distances.append(
                    torch.cdist(
                        candidate_vectors.float(), vectors[comparison_ids].float()
                    ).half()
                )
            near_other_class = (
                torch.cat(distances, dim=-1) < self.distance_threshold
            ).sum(dim=-1)
            rank = torch.argsort(near_other_class)
            chosen = rank[: min(self.num_experience_nodes, len(rank))].tolist()
            selected.extend(candidates[index] for index in chosen)
        return selected

    def sample_nodes(
        self,
        queries: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        global_task_id: int,
    ) -> list[int]:
        grouped = self._group_by_class(queries, labels)
        client_offset = 0 if self.client_id is None else 1_000_003 * self.client_id
        rng = random.Random(self.seed + client_offset + 9_176 * global_task_id)
        if self.sampler_name == "random":
            return self._random_sample(grouped, rng)
        if self.sampler_name == "MF":
            return self._mf_sample(grouped, features.detach().cpu())
        return self._cm_sample(grouped, features, rng)

    @staticmethod
    def _masked_logits(
        logits: torch.Tensor, class_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if class_mask is None or logits.shape[-1] != class_mask.shape[0]:
            return logits
        output = logits.clone()
        output[..., ~class_mask.to(logits.device)] = -1e12
        return output

    def training_loss(
        self,
        model,
        forward,
        queries,
        logits,
        labels,
        global_task_id,
        base_loss,
        class_mask=None,
    ):
        buffered_nodes = self.state.get("buffered_nodes")
        if buffered_nodes is None or buffered_nodes.numel() == 0:
            current_loss = float(base_loss.detach())
            self._last_loss_diagnostics = {
                "current_loss": current_loss,
                "replay_loss": None,
                "total_loss": current_loss,
                "current_weight": 1.0,
                "replay_weight": 0.0,
                "buffer_size": 0,
                "current_sample_count": int(labels.shape[0]),
            }
            return base_loss
        device = logits.device
        replay_queries = buffered_nodes.to(device)
        replay_labels = self.state["buffered_labels"].to(device)
        replay_logits = self._masked_logits(
            forward(model, replay_queries), class_mask
        )
        replay_loss = F.cross_entropy(replay_logits, replay_labels.long())
        buffer_size = int(replay_labels.shape[0])
        beta = buffer_size / (buffer_size + int(labels.shape[0]))
        total_loss = (1.0 - beta) * base_loss + beta * replay_loss
        self._last_loss_diagnostics = {
            "current_loss": float(base_loss.detach()),
            "replay_loss": float(replay_loss.detach()),
            "total_loss": float(total_loss.detach()),
            "current_weight": 1.0 - beta,
            "replay_weight": beta,
            "buffer_size": buffer_size,
            "current_sample_count": int(labels.shape[0]),
        }
        return total_loss

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
        if node_features is None:
            raise RuntimeError("ERGNN sampling requires client-local node features.")
        selected = self.sample_nodes(
            queries, labels, node_features, global_task_id
        )
        selected_tensor = torch.tensor(selected, dtype=torch.long)
        label_by_node = {
            int(node): int(label)
            for node, label in zip(
                queries.detach().cpu().tolist(), labels.detach().cpu().tolist()
            )
        }
        selected_labels = torch.tensor(
            [label_by_node[node] for node in selected], dtype=torch.long
        )
        existing_nodes = self.state.get(
            "buffered_nodes", torch.empty(0, dtype=torch.long)
        )
        existing_labels = self.state.get(
            "buffered_labels", torch.empty(0, dtype=torch.long)
        )
        self.state["buffered_nodes"] = torch.cat(
            (existing_nodes, selected_tensor)
        )
        self.state["buffered_labels"] = torch.cat(
            (existing_labels, selected_labels)
        )
        history = dict(self.state.get("sampling_history", {}))
        history[int(global_task_id)] = tuple(selected)
        self.state["sampling_history"] = history

    def hyperparameters(self):
        output = super().hyperparameters()
        output.update(
            {
                "num_experience_nodes_per_class_per_task": self.num_experience_nodes,
                "sampler_name": self.sampler_name,
                "distance_threshold": self.distance_threshold,
                "loss_weighting": "paper_eq3_beta_replay",
            }
        )
        return output

    def diagnostics(self):
        output = dict(self._last_loss_diagnostics)
        buffered_nodes = self.state.get("buffered_nodes")
        output["buffer_size"] = (
            0 if buffered_nodes is None else int(buffered_nodes.numel())
        )
        output["loss_weighting"] = "paper_eq3_beta_replay"
        return output

    def replay_payload_bytes(self) -> int:
        """Report the exact private replay tensors retained by this client."""

        return sum(
            int(value.numel() * value.element_size())
            for key in ("buffered_nodes", "buffered_labels")
            if torch.is_tensor(value := self.state.get(key))
        )
