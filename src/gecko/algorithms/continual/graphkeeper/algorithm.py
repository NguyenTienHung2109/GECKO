from __future__ import annotations

from gecko.algorithms.continual.graphkeeper.common import GRAPHKEEPER_STATE_FORMAT

import math
from types import MethodType
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Sequence
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from gecko.algorithms.base import ClientContinualAlgorithm

class GraphKeeperAlgorithm(ClientContinualAlgorithm):
    """Graph domain experts, disentanglement, analytic preservation and routing."""

    name = "GraphKeeper"
    method_version = "uefa-graphkeeper-v3-paper-faithful"
    supported_problem_type = "NC"
    state_format = GRAPHKEEPER_STATE_FORMAT
    prediction_unit = "node"
    router_name = "nearest_frozen_random_gnn_domain_prototype"

    def __init__(
        self,
        *,
        rank: int = 16,
        intra_weight: float = 1.0,
        inter_weight: float = 0.1,
        temperature: float = 0.2,
        feature_drop: float = 0.3,
        edge_drop: float = 0.2,
        adapter_learning_rate: float = 0.01,
        stage_count: int = 8,
        router_projection_dim: int = 2048,
        ridge_lambda: float = 0.1,
        dbscan_eps: float = 0.1,
        dbscan_min_samples: int = 5,
        max_cluster_nodes: int = 2048,
        max_prototypes_per_domain: int = 64,
        pretrain_weight: float = 1.0,
        pretrain_edges: int = 4096,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if (
            self.problem_type != self.supported_problem_type
            or self.incremental_setting != "domain"
        ):
            raise ValueError(
                f"{self.name} supports only "
                f"{self.supported_problem_type}-Domain."
            )
        if self.client_id is None or self.client_id < 0:
            raise ValueError("GraphKeeper requires a client_id.")
        positive_ints = {
            "rank": rank,
            "stage_count": stage_count,
            "router_projection_dim": router_projection_dim,
            "dbscan_min_samples": dbscan_min_samples,
            "max_cluster_nodes": max_cluster_nodes,
            "max_prototypes_per_domain": max_prototypes_per_domain,
            "pretrain_edges": pretrain_edges,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"GraphKeeper {name} must be a positive integer.")
        positive_floats = {
            "intra_weight": intra_weight,
            "inter_weight": inter_weight,
            "temperature": temperature,
            "adapter_learning_rate": adapter_learning_rate,
            "ridge_lambda": ridge_lambda,
            "dbscan_eps": dbscan_eps,
            "pretrain_weight": pretrain_weight,
        }
        for name, value in positive_floats.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"GraphKeeper {name} must be finite and positive.")
        for name, value in (("feature_drop", feature_drop), ("edge_drop", edge_drop)):
            if not 0 <= float(value) < 1:
                raise ValueError(f"GraphKeeper {name} must lie in [0,1).")

        self.rank = rank
        self.stage_count = stage_count
        self.intra_weight = float(intra_weight)
        self.inter_weight = float(inter_weight)
        self.temperature = float(temperature)
        self.feature_drop = float(feature_drop)
        self.edge_drop = float(edge_drop)
        self.adapter_learning_rate = float(adapter_learning_rate)
        self.router_projection_dim = router_projection_dim
        self.ridge_lambda = float(ridge_lambda)
        self.dbscan_eps = float(dbscan_eps)
        self.dbscan_min_samples = dbscan_min_samples
        self.max_cluster_nodes = max_cluster_nodes
        self.max_prototypes_per_domain = max_prototypes_per_domain
        self.pretrain_weight = float(pretrain_weight)
        self.pretrain_edges = pretrain_edges

        self._active_stage: int | None = None
        self._active_layers: list[tuple[nn.Parameter, nn.Parameter]] = []
        self._adapter_optimizer: torch.optim.Optimizer | None = None
        self._patched_model: nn.Module | None = None
        self._expected_route_stage: int | None = None
        self._expected_global_task: int | None = None
        self._router_stage_decisions: Dict[int, Dict[int, int]] = {}
        self._router_global_decisions: Dict[int, Dict[int, int]] = {}
        self._router_query_weighted: Dict[int, Dict[int, int]] = {}
        self._multi_expert_router_stage_decisions: Dict[int, Dict[int, int]] = {}
        self._multi_expert_router_global_decisions: Dict[int, Dict[int, int]] = {}
        self._multi_expert_router_query_weighted: Dict[int, Dict[int, int]] = {}
        self._pretrain_positive_keys: torch.Tensor | None = None
        self._pretrain_key_num_nodes: int | None = None

        self.state.update(
            {
                "format": self.state_format,
                "adapters": {},
                "inter_prototypes": {},
                "routing_projection": torch.empty((0, 0)),
                "routing_prototypes": {},
                "analytic_a": torch.empty((0, 0), dtype=torch.float64),
                "analytic_c": torch.empty((0, 0), dtype=torch.float64),
                "analytic_w": torch.empty((0, 0), dtype=torch.float64),
                "task_to_stage": {},
                "completed_stages": [],
                "diagnostics": {
                    "task_loss": 0.0,
                    "task_loss_semantics": (
                        "pre_consolidation_classifier_on_current_stage"
                    ),
                    "post_consolidation_train_loss": None,
                    "post_consolidation_train_loss_finite": False,
                    "post_consolidation_train_logit_finite_count": 0,
                    "post_consolidation_train_logit_count": 0,
                    "post_consolidation_train_query_count": 0,
                    "contrast_labels_multilabel": False,
                    "contrast_positive_pair_rule": "unobserved",
                    "contrast_positive_pair_count": 0,
                    "contrast_positive_pair_count_semantics": (
                        "ordered_off_diagonal_matrix_entries"
                    ),
                    "contrast_off_diagonal_pair_count": 0,
                    "contrast_positive_pair_density_off_diagonal": 0.0,
                    "contrast_anchors_with_positive_count": 0,
                    "contrast_anchor_count": 0,
                    "intra_loss": 0.0,
                    "inter_loss": 0.0,
                    "pretrain_loss": 0.0,
                    "preserve_loss": 0.0,
                    "total_loss": 0.0,
                    "selected_expert": -1,
                    "router_margin": 0.0,
                    "shared_parameter_count": 0,
                    "new_parameters_per_domain": 0,
                    "classifier_update_norm": 0.0,
                    "dbscan_fallback_domains": [],
                    "inter_prototypes_per_domain": {},
                    "per_stage": {},
                },
            }
        )

    def _validate(self, context: Any) -> None:
        if (
            context.problem_type != self.supported_problem_type
            or context.incremental_setting != "domain"
        ):
            raise ValueError(
                f"{self.name} received a non "
                f"{self.supported_problem_type}-Domain context."
            )
        if context.client_id != self.client_id or context.stage_index >= self.stage_count:
            raise ValueError("GraphKeeper context identity/stage mismatch.")
        if context.valid_class_mask is not None:
            raise ValueError("GraphKeeper Domain-IL must use the shared full output head.")

    def _adapters(self) -> Dict[str, list[Dict[str, torch.Tensor]]]:
        values = self.state["adapters"]
        if not isinstance(values, dict):
            raise ValueError("Malformed GraphKeeper adapters.")
        return values

    def _routing_prototypes(self) -> Dict[str, torch.Tensor]:
        values = self.state["routing_prototypes"]
        if not isinstance(values, dict):
            raise ValueError("Malformed GraphKeeper routing prototypes.")
        return values

    def _inter_prototypes(self) -> Dict[str, list[torch.Tensor]]:
        values = self.state["inter_prototypes"]
        if not isinstance(values, dict):
            raise ValueError("Malformed GraphKeeper inter-domain prototypes.")
        return values

    def _ensure_active_layers(self, model: nn.Module) -> None:
        from gecko.models.backbones import MeanGraphLayer
        from gecko.models.backbones import LegacyGCNAdapter
        from gecko.models.backbones import GECKOGraphModel

        if self._active_stage is None or self._active_layers:
            return
        if isinstance(model, GECKOGraphModel) and all(
            isinstance(layer, MeanGraphLayer) for layer in model.layers
        ):
            dimensions = [
                (
                    int(layer.neighbor_linear.in_features),
                    int(layer.neighbor_linear.out_features),
                )
                for layer in model.layers
            ]
        elif isinstance(model, LegacyGCNAdapter) and model.problem_type in {
            "NC",
            "LP",
        }:
            dimensions = []
            for layer in model.original.convs:
                weight = layer.weight
                if weight is None or weight.ndim != 2:
                    raise RuntimeError(
                        "GraphKeeper requires weighted BeGin GraphConv layers."
                    )
                dimensions.append((int(weight.shape[0]), int(weight.shape[1])))
        else:
            raise RuntimeError(
                "GraphKeeper requires verified UEFA mean-GCN or BeGin NC/LP GCN layers."
            )
        generator = torch.Generator(device="cpu").manual_seed(
            (self.seed + self.client_id * 1009 + self._active_stage * 9176) % (2**63)
        )
        device = next(model.parameters()).device
        for input_size, output_size in dimensions:
            a = torch.randn((input_size, self.rank), generator=generator)
            a = a * (1.0 / math.sqrt(max(input_size, 1)))
            b = torch.zeros((self.rank, output_size))
            self._active_layers.append(
                (nn.Parameter(a.to(device)), nn.Parameter(b.to(device)))
            )

    def _layer_parameters(
        self, model: nn.Module, stage: int
    ) -> Sequence[tuple[torch.Tensor, torch.Tensor]]:
        if self._active_stage == stage:
            self._ensure_active_layers(model)
            return tuple(self._active_layers)
        payload = self._adapters().get(str(stage))
        if payload is None:
            raise ValueError(f"Unknown GraphKeeper expert {stage}.")
        device = next(model.parameters()).device
        return tuple((item["a"].to(device), item["b"].to(device)) for item in payload)

    def _encode_expert(
        self,
        model: nn.Module,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        from gecko.algorithms.continual.graphkeeper.common import _mean_neighbors
        from gecko.algorithms.continual.graphkeeper.common import _symmetric_neighbors
        from gecko.models.backbones import LegacyGCNAdapter
        from gecko.models.backbones import GECKOGraphModel

        parameters = self._layer_parameters(model, stage)
        if isinstance(model, GECKOGraphModel):
            if len(parameters) != len(model.layers):
                raise ValueError("GraphKeeper expert layer count does not match the backbone.")
            hidden = features
            for layer, (a, b) in zip(model.layers, parameters):
                base = layer(hidden, edge_index)
                graph_lora = _mean_neighbors(hidden, edge_index) @ a @ b
                hidden = F.relu(base + graph_lora)
            return hidden
        if isinstance(model, LegacyGCNAdapter) and model.problem_type in {
            "NC",
            "LP",
        }:
            encoder = model.original
            if len(parameters) != len(encoder.convs):
                raise ValueError("GraphKeeper expert layer count does not match the backbone.")
            _, resolved_edges = model._encoder_and_edges(
                edge_index, int(features.shape[0])
            )
            graph = (
                None
                if edge_index.device.type != "cpu"
                else model._graph(edge_index, features.shape[0])
            )
            hidden = encoder.dropout(features)
            for index, (layer, (a, b)) in enumerate(zip(encoder.convs, parameters)):
                if graph is None:
                    weight = layer.weight
                    if weight is None:
                        raise RuntimeError(
                            "GraphKeeper requires weighted BeGin GraphConv layers."
                        )
                    messages = model._symmetric_aggregate(hidden, resolved_edges)
                    base = messages @ weight
                    if layer.bias is not None:
                        base = base + layer.bias
                else:
                    base = layer(graph, hidden)
                    messages = _symmetric_neighbors(hidden, resolved_edges)
                graph_lora = messages @ a @ b
                hidden = encoder.norms[index](base + graph_lora)
                hidden = encoder.activation(hidden)
                hidden = encoder.dropout(hidden)
            return hidden
        raise RuntimeError("GraphKeeper layer-wise experts require a verified NC/LP GCN.")

    def _ensure_routing_projection(self, input_size: int) -> torch.Tensor:
        projection = self.state["routing_projection"]
        if not torch.is_tensor(projection):
            raise ValueError("Malformed GraphKeeper routing projection.")
        if projection.numel():
            if tuple(projection.shape) != (input_size, self.router_projection_dim):
                raise ValueError("GraphKeeper routing projection shape changed.")
            return projection
        generator = torch.Generator(device="cpu").manual_seed(
            (self.seed + self.client_id * 65537 + 0x47524B50) % (2**63)
        )
        projection = torch.randn(
            (input_size, self.router_projection_dim), generator=generator
        ) / math.sqrt(max(input_size, 1))
        self.state["routing_projection"] = projection.contiguous()
        return projection

    def _project_domain(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        from gecko.algorithms.continual.graphkeeper.common import _mean_neighbors
        weight = self._ensure_routing_projection(int(features.shape[1])).to(features.device)
        projected = features @ weight
        return F.relu(projected + _mean_neighbors(projected, edge_index))

    def _domain_routing_prototype(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Return the paper-compatible query prototype used by NC routing."""

        return self._project_domain(features, edge_index)[queries].mean(dim=0)

    def _route(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor,
    ) -> tuple[int, float]:
        prototypes = self._routing_prototypes()
        if not prototypes:
            if self._active_stage is None:
                raise RuntimeError("No GraphKeeper expert exists.")
            return self._active_stage, 0.0
        query_prototype = self._domain_routing_prototype(
            features, edge_index, queries
        )
        distances = sorted(
            (
                float(torch.norm(query_prototype - value.to(query_prototype.device)).detach()),
                int(stage),
            )
            for stage, value in prototypes.items()
        )
        margin = distances[1][0] - distances[0][0] if len(distances) > 1 else 0.0
        return distances[0][1], margin

    def _analytic_logits(self, model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        from gecko.algorithms.continual.graphkeeper.common import _output_size
        output_size = _output_size(model)
        weight = self.state["analytic_w"]
        if not torch.is_tensor(weight):
            raise ValueError("Malformed GraphKeeper analytic classifier.")
        if not weight.numel():
            logits = hidden.new_zeros((hidden.shape[0], output_size))
            return logits.reshape(-1) if self.supported_problem_type == "LP" else logits
        if tuple(weight.shape) != (hidden.shape[1], output_size):
            raise ValueError("GraphKeeper analytic classifier shape changed.")
        logits = hidden @ weight.to(device=hidden.device, dtype=hidden.dtype)
        return logits.reshape(-1) if self.supported_problem_type == "LP" else logits

    def _record_route(self, stage: int, query_count: int) -> None:
        if self._expected_route_stage is None or self._expected_global_task is None:
            return
        expected_stage = int(self._expected_route_stage)
        stage_row = self._router_stage_decisions.setdefault(expected_stage, {})
        stage_row[stage] = stage_row.get(stage, 0) + 1
        weighted_row = self._router_query_weighted.setdefault(expected_stage, {})
        weighted_row[stage] = weighted_row.get(stage, 0) + query_count
        stage_to_task = {
            int(value): int(key) for key, value in self.state["task_to_stage"].items()
        }
        predicted_global = stage_to_task.get(stage)
        if predicted_global is None:
            raise ValueError("Selected GraphKeeper expert has no global task mapping.")
        global_row = self._router_global_decisions.setdefault(
            int(self._expected_global_task), {}
        )
        global_row[predicted_global] = global_row.get(predicted_global, 0) + 1
        if len(self._routing_prototypes()) > 1:
            multi_stage_row = self._multi_expert_router_stage_decisions.setdefault(
                expected_stage, {}
            )
            multi_stage_row[stage] = multi_stage_row.get(stage, 0) + 1
            multi_weighted_row = self._multi_expert_router_query_weighted.setdefault(
                expected_stage, {}
            )
            multi_weighted_row[stage] = (
                multi_weighted_row.get(stage, 0) + query_count
            )
            multi_global_row = self._multi_expert_router_global_decisions.setdefault(
                int(self._expected_global_task), {}
            )
            multi_global_row[predicted_global] = (
                multi_global_row.get(predicted_global, 0) + 1
            )

    def _patch_model(self, model: nn.Module) -> None:
        from gecko.algorithms.continual.graphkeeper.common import _query_embeddings
        if getattr(model, "_graphkeeper_owner", None) is self:
            return
        if getattr(model, "_graphkeeper_owner", None) is not None:
            raise RuntimeError("A model cannot be owned by two GraphKeeper clients.")
        original = model.forward_queries
        algorithm = self

        def forward_queries(
            this: nn.Module,
            features: torch.Tensor,
            edge_index: torch.Tensor,
            queries: torch.Tensor,
            problem_type: str,
        ) -> torch.Tensor:
            if str(problem_type).upper() != algorithm.supported_problem_type:
                return original(features, edge_index, queries, problem_type)
            if this.training and algorithm._active_stage is not None:
                stage, margin = algorithm._active_stage, 0.0
            else:
                stage, margin = algorithm._route(features, edge_index, queries)
                algorithm._record_route(int(stage), int(queries.shape[0]))
            hidden = algorithm._encode_expert(this, features, edge_index, int(stage))
            algorithm.state["diagnostics"]["selected_expert"] = int(stage)
            algorithm.state["diagnostics"]["router_margin"] = float(margin)
            query_hidden = _query_embeddings(
                hidden, queries, algorithm.supported_problem_type
            )
            return algorithm._analytic_logits(this, query_hidden)

        model.forward_queries = MethodType(forward_queries, model)
        model._graphkeeper_owner = self
        model._graphkeeper_original_forward_queries = original
        self._patched_model = model

    def before_task(self, context: Any) -> None:
        self._validate(context)
        stage = int(context.stage_index)
        if stage in self.state["completed_stages"]:
            raise ValueError("GraphKeeper stage was already completed.")
        self._active_stage = stage
        self._active_layers = []
        self._pretrain_positive_keys = None
        self._pretrain_key_num_nodes = None

    def _ensure_active(self, model: nn.Module, context: Any) -> None:
        self._patch_model(model)
        if self._active_stage != context.stage_index:
            self._active_stage = int(context.stage_index)
            self._active_layers = []
        self._ensure_active_layers(model)

    def before_round(self, context: Any) -> None:
        self._validate(context)
        if self._active_layers:
            for a, b in self._active_layers:
                a.grad = None
                b.grad = None
            self._adapter_optimizer = torch.optim.Adam(
                [parameter for pair in self._active_layers for parameter in pair],
                lr=self.adapter_learning_rate,
            )

    def _sample_link_pretraining_edges(
        self, context: Any, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edges = context.effective_edge_index.to(device)
        num_nodes = int(context.node_features.shape[0])
        if not edges.shape[1] or num_nodes < 2:
            return edges[:, :0], edges[:, :0]
        generator = torch.Generator(device=device).manual_seed(
            (
                self.seed
                + self.client_id * 104729
                + context.stage_index * 1009
                + context.round_index * 9176
            )
            % (2**63)
        )
        count = min(self.pretrain_edges, int(edges.shape[1]))
        positive_index = torch.randint(
            int(edges.shape[1]), (count,), generator=generator, device=device
        )
        positives = edges[:, positive_index]
        if (
            self._pretrain_positive_keys is None
            or self._pretrain_key_num_nodes != num_nodes
        ):
            keys = edges[0] * num_nodes + edges[1]
            self._pretrain_positive_keys = torch.sort(torch.unique(keys)).values
            self._pretrain_key_num_nodes = num_nodes
        positive_keys = self._pretrain_positive_keys
        negative_parts: list[torch.Tensor] = []
        remaining = count
        attempts = 0
        while remaining > 0 and attempts < 32:
            attempts += 1
            candidates = torch.randint(
                num_nodes,
                (2, max(remaining * 3, 64)),
                generator=generator,
                device=device,
            )
            source, target = candidates
            keys = source * num_nodes + target
            reverse_keys = target * num_nodes + source
            locations = torch.searchsorted(positive_keys, keys)
            reverse_locations = torch.searchsorted(positive_keys, reverse_keys)
            safe_locations = locations.clamp_max(positive_keys.numel() - 1)
            safe_reverse = reverse_locations.clamp_max(positive_keys.numel() - 1)
            exists = (locations < positive_keys.numel()) & (
                positive_keys[safe_locations] == keys
            )
            reverse_exists = (reverse_locations < positive_keys.numel()) & (
                positive_keys[safe_reverse] == reverse_keys
            )
            keep = (source != target) & ~exists & ~reverse_exists
            accepted = candidates[:, keep][:, :remaining]
            if accepted.shape[1]:
                negative_parts.append(accepted)
                remaining -= int(accepted.shape[1])
        if remaining:
            raise RuntimeError("GraphKeeper could not sample strict non-edge negatives.")
        return positives, torch.cat(negative_parts, dim=1)

    def _pretraining_loss(self, model: nn.Module, context: Any) -> torch.Tensor:
        device = next(model.parameters()).device
        positives, negatives = self._sample_link_pretraining_edges(context, device)
        hidden = context.encode_nodes(model)
        if not positives.shape[1]:
            return hidden.sum() * 0.0
        return self._link_pretraining_loss(hidden, positives, negatives)

    @staticmethod
    def _link_pretraining_loss(
        hidden: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Match GraphKeeper's elementwise endpoint-Hadamard link objective."""

        positive_scores = hidden[positives[0]] * hidden[positives[1]]
        negative_scores = hidden[negatives[0]] * hidden[negatives[1]]
        scores = torch.cat((positive_scores, negative_scores))
        labels = torch.cat(
            (torch.ones_like(positive_scores), torch.zeros_like(negative_scores))
        )
        return F.binary_cross_entropy_with_logits(scores, labels)

    @staticmethod
    def _inter_domain_disentanglement_loss(
        current: torch.Tensor,
        old_prototypes: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Repel each embedding from its nearest old prototype as published."""

        if not old_prototypes:
            return current.sum() * 0.0
        prototype_tensor = torch.stack(
            [prototype.to(current.device) for prototype in old_prototypes]
        )
        nearest_distance = torch.cdist(current, prototype_tensor).min(dim=1).values
        return torch.reciprocal(nearest_distance + 1e-8).mean()

    def augment_loss(
        self,
        model: nn.Module,
        context: Any,
        logits: torch.Tensor,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        from gecko.algorithms.continual.graphkeeper.common import _positive_pairs
        from gecko.algorithms.continual.graphkeeper.common import _query_embeddings
        del logits
        self._validate(context)
        self._ensure_active(model, context)
        features = context.node_features.to(next(model.parameters()).device)
        edges = context.effective_edge_index.to(features.device)
        queries = context.train_queries.to(features.device)
        hidden = self._encode_expert(model, features, edges, context.stage_index)
        current = _query_embeddings(hidden, queries, self.supported_problem_type)

        keep = torch.rand(features.shape[1], device=features.device) >= self.feature_drop
        augmented_features = features * keep
        augmented_edges = edges
        if edges.shape[1]:
            edge_keep = torch.rand(edges.shape[1], device=features.device) >= self.edge_drop
            augmented_edges = edges[:, edge_keep]
        augmented_hidden = self._encode_expert(
            model, augmented_features, augmented_edges, context.stage_index
        )
        augmented = _query_embeddings(
            augmented_hidden, queries, self.supported_problem_type
        )
        contrast_labels = context.train_labels.to(current.device)
        contrast_count = min(int(current.shape[0]), self.max_cluster_nodes)
        if contrast_count < int(current.shape[0]):
            generator = torch.Generator(device=current.device).manual_seed(
                (
                    self.seed
                    + self.client_id * 104729
                    + context.stage_index * 1009
                    + context.round_index * 9176
                    + 0x434F4E54
                )
                % (2**63)
            )
            contrast_indices = torch.randperm(
                current.shape[0], generator=generator, device=current.device
            )[:contrast_count]
            contrast_current = current[contrast_indices]
            contrast_augmented = augmented[contrast_indices]
            contrast_labels = contrast_labels[contrast_indices]
        else:
            contrast_current = current
            contrast_augmented = augmented
        z1 = F.normalize(contrast_current, dim=-1)
        z2 = F.normalize(contrast_augmented, dim=-1)
        similarity = z1 @ z2.t() / self.temperature
        positives = _positive_pairs(contrast_labels)
        positives.fill_diagonal_(False)
        positive_pair_count = int(positives.sum().item())
        off_diagonal_pair_count = contrast_count * max(contrast_count - 1, 0)
        positive_pair_density = (
            positive_pair_count / off_diagonal_pair_count
            if off_diagonal_pair_count
            else 0.0
        )
        anchors_with_positive = int(positives.any(dim=1).sum().item())
        labels_are_multilabel = contrast_labels.ndim > 1
        exp_similarity = torch.exp(similarity - similarity.max(dim=1, keepdim=True).values)
        numerator = (exp_similarity * positives).sum(dim=1)
        denominator = exp_similarity.sum(dim=1).clamp_min(1e-12)
        valid = positives.any(dim=1)
        intra = (
            -torch.log((numerator[valid] / denominator[valid]).clamp_min(1e-12)).mean()
            if valid.any()
            else similarity.sum() * 0.0
        )

        old_prototypes = tuple(
            prototype
            for values in self._inter_prototypes().values()
            for prototype in values
        )
        inter = self._inter_domain_disentanglement_loss(current, old_prototypes)

        pretrain = (
            self._pretraining_loss(model, context)
            if context.stage_index == 0
            else current.sum() * 0.0
        )
        total = (
            self.intra_weight * intra
            + self.inter_weight * inter
            + self.pretrain_weight * pretrain
        )
        with torch.no_grad():
            from gecko.engine.client import supervised_loss

            task_logits = self._analytic_logits(model, current)
            task_loss = supervised_loss(
                task_logits, context.train_labels.to(task_logits.device)
            )
        new_parameters = sum(a.numel() + b.numel() for a, b in self._active_layers)
        self.state["diagnostics"].update(
            {
                "task_loss": float(task_loss.detach().cpu()),
                "intra_loss": float(intra.detach().cpu()),
                "inter_loss": float(inter.detach().cpu()),
                "pretrain_loss": float(pretrain.detach().cpu()),
                "preserve_loss": 0.0,
                "total_loss": float(total.detach().cpu()),
                "shared_parameter_count": sum(p.numel() for p in model.parameters()),
                "new_parameters_per_domain": new_parameters,
                "contrast_query_count": contrast_count,
                "contrast_total_queries": int(current.shape[0]),
                "contrast_labels_multilabel": labels_are_multilabel,
                "contrast_positive_pair_rule": (
                    "multilabel_any_shared_positive_label"
                    if labels_are_multilabel
                    else "single_label_same_class"
                ),
                "contrast_positive_pair_count": positive_pair_count,
                "contrast_positive_pair_count_semantics": (
                    "ordered_off_diagonal_matrix_entries"
                ),
                "contrast_off_diagonal_pair_count": off_diagonal_pair_count,
                "contrast_positive_pair_density_off_diagonal": (
                    positive_pair_density
                ),
                "contrast_anchors_with_positive_count": anchors_with_positive,
                "contrast_anchor_count": contrast_count,
            }
        )
        return total + base_loss * 0.0

    def after_backward(self, model: nn.Module, context: Any) -> None:
        # The first local domain pre-trains the shared GNN. It is immutable
        # thereafter; the recursive analytic classifier is not a model parameter.
        if context.stage_index == 0:
            for name, parameter in model.named_parameters():
                if "node_head" in name or "edge_head" in name:
                    parameter.grad = None
            return
        for parameter in model.parameters():
            parameter.grad = None

    def after_round(self, model: nn.Module, context: Any) -> None:
        del model, context
        if self._adapter_optimizer is None:
            self._adapter_optimizer = torch.optim.Adam(
                [parameter for pair in self._active_layers for parameter in pair],
                lr=self.adapter_learning_rate,
            )
        self._adapter_optimizer.step()
        self._adapter_optimizer = None

    def _cluster_prototypes(
        self, embeddings: torch.Tensor, *, stage: int
    ) -> tuple[list[torch.Tensor], bool]:
        values = embeddings.detach()
        if values.shape[0] > self.max_cluster_nodes:
            generator = torch.Generator(device=values.device).manual_seed(
                (self.seed + self.client_id * 8191 + stage * 131071) % (2**63)
            )
            indices = torch.randperm(
                values.shape[0], generator=generator, device=values.device
            )[: self.max_cluster_nodes]
            values = values[indices]
        if values.shape[0] < self.dbscan_min_samples:
            return [values.mean(dim=0).cpu()], True
        metric_values = self._cluster_metric_values(values)
        neighbors = self._cluster_neighbor_mask(metric_values)
        core = neighbors.sum(dim=1) >= self.dbscan_min_samples
        core_indices = torch.where(core)[0]
        if not core_indices.numel():
            return [values.mean(dim=0).cpu()], True
        core_graph = neighbors[core_indices][:, core_indices]
        labels = torch.arange(core_indices.shape[0], device=values.device)
        sentinel = torch.full_like(labels, core_indices.shape[0])
        for _ in range(int(core_indices.shape[0])):
            neighbor_labels = torch.where(core_graph, labels.unsqueeze(0), sentinel.unsqueeze(0))
            updated = torch.minimum(labels, neighbor_labels.min(dim=1).values)
            if torch.equal(updated, labels):
                break
            labels = updated
        else:
            raise RuntimeError("GPU DBSCAN label propagation did not converge.")
        clusters: list[tuple[int, int, torch.Tensor]] = []
        for root in torch.unique(labels, sorted=True):
            member_core = core_indices[labels == root]
            members = neighbors[:, member_core].any(dim=1)
            count = int(members.sum())
            clusters.append((count, int(root), values[members].mean(dim=0).cpu()))
        clusters.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in clusters[: self.max_prototypes_per_domain]], False

    def _cluster_metric_values(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return the NC embedding geometry used by DBSCAN."""

        return embeddings

    def _cluster_neighbor_mask(self, values: torch.Tensor) -> torch.Tensor:
        """Return the NC Euclidean epsilon-neighborhood graph."""

        return torch.cdist(values, values) <= self.dbscan_eps

    def _update_analytic_classifier(
        self, model: nn.Module, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> None:
        from gecko.algorithms.continual.graphkeeper.common import _output_size
        device = embeddings.device
        x = embeddings.detach().to(dtype=torch.float64)
        output_size = _output_size(model)
        if self.supported_problem_type == "LP":
            y = labels.reshape(-1, 1).to(device=device, dtype=torch.float64)
        elif labels.ndim == 1:
            y = F.one_hot(labels.to(torch.long), num_classes=output_size).to(torch.float64)
        else:
            y = labels.to(device=device, dtype=torch.float64)
        if y.shape[1] != output_size:
            raise ValueError("GraphKeeper analytic labels do not match the shared label space.")
        previous_a = self.state["analytic_a"]
        previous_c = self.state["analytic_c"]
        if not previous_a.numel():
            a = self.ridge_lambda * torch.eye(x.shape[1], device=device, dtype=torch.float64)
            c = torch.zeros((x.shape[1], output_size), device=device, dtype=torch.float64)
            previous_w = torch.zeros_like(c)
        else:
            a = previous_a.to(device)
            c = previous_c.to(device)
            previous_w = self.state["analytic_w"].to(device)
        a = a + x.t() @ x
        c = c + x.t() @ y
        weight = torch.linalg.solve(a, c)
        self.state["analytic_a"] = a.cpu()
        self.state["analytic_c"] = c.cpu()
        self.state["analytic_w"] = weight.cpu()
        self.state["diagnostics"]["classifier_update_norm"] = float(
            torch.norm(weight - previous_w).cpu()
        )

    def _post_consolidation_train_diagnostics(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, object]:
        """Measure the fitted analytic classifier without changing its state."""

        from gecko.engine.client import supervised_loss

        with torch.no_grad():
            logits = self._analytic_logits(model, embeddings)
            loss = supervised_loss(logits, labels.to(logits.device))
        finite = torch.isfinite(logits)
        return {
            "post_consolidation_train_loss": float(loss.detach().cpu()),
            "post_consolidation_train_loss_finite": bool(
                torch.isfinite(loss).item()
            ),
            "post_consolidation_train_logit_finite_count": int(finite.sum().item()),
            "post_consolidation_train_logit_count": int(logits.numel()),
            "post_consolidation_train_query_count": int(labels.shape[0]),
        }

    def consolidate(self, model: nn.Module, context: Any) -> None:
        from gecko.algorithms.continual.graphkeeper.common import _query_embeddings
        self._validate(context)
        self._ensure_active(model, context)
        stage = int(context.stage_index)
        self._adapters()[str(stage)] = [
            {"a": a.detach().cpu().clone(), "b": b.detach().cpu().clone()}
            for a, b in self._active_layers
        ]
        device = next(model.parameters()).device
        features = context.node_features.to(device)
        edges = context.effective_edge_index.to(device)
        queries = context.train_queries.to(device)
        was_training = model.training
        try:
            # Consolidated prototypes and ridge sufficient statistics are later
            # consumed in evaluation mode.  Building them while dropout is active
            # also mutates BatchNorm running buffers in an otherwise read-only
            # boundary pass, so use the same deterministic representation here.
            model.eval()
            with torch.no_grad():
                node_embeddings = self._encode_expert(model, features, edges, stage)
                embeddings = _query_embeddings(
                    node_embeddings, queries, self.supported_problem_type
                )
                routing_prototype = self._domain_routing_prototype(
                    features, edges, queries
                )
            prototypes, fallback = self._cluster_prototypes(
                embeddings, stage=stage
            )
            self._inter_prototypes()[str(stage)] = prototypes
            self._routing_prototypes()[str(stage)] = routing_prototype.cpu()
            self._update_analytic_classifier(
                model, embeddings, context.train_labels.to(device)
            )
            post_consolidation = self._post_consolidation_train_diagnostics(
                model,
                embeddings,
                context.train_labels.to(device),
            )
        finally:
            model.train(was_training)
        self.state["task_to_stage"][str(context.global_task_id)] = stage
        self.state["completed_stages"].append(stage)
        diagnostic = self.state["diagnostics"]
        diagnostic["inter_prototypes_per_domain"][str(stage)] = len(prototypes)
        if fallback:
            diagnostic["dbscan_fallback_domains"].append(stage)
        diagnostic.update(post_consolidation)
        diagnostic["per_stage"][str(stage)] = {
            "global_task_id": int(context.global_task_id),
            "task_loss": float(diagnostic["task_loss"]),
            "task_loss_semantics": diagnostic["task_loss_semantics"],
            **post_consolidation,
            "intra_loss": float(diagnostic["intra_loss"]),
            "inter_loss": float(diagnostic["inter_loss"]),
            "pretrain_loss": float(diagnostic["pretrain_loss"]),
            "preserve_loss": 0.0,
            "total_loss": float(diagnostic["total_loss"]),
            "classifier_update_norm": float(diagnostic["classifier_update_norm"]),
            "inter_prototypes": len(prototypes),
            "dbscan_fallback": bool(fallback),
            "contrast_labels_multilabel": bool(
                diagnostic["contrast_labels_multilabel"]
            ),
            "contrast_positive_pair_rule": diagnostic[
                "contrast_positive_pair_rule"
            ],
            "contrast_positive_pair_count": int(
                diagnostic["contrast_positive_pair_count"]
            ),
            "contrast_positive_pair_count_semantics": diagnostic[
                "contrast_positive_pair_count_semantics"
            ],
            "contrast_off_diagonal_pair_count": int(
                diagnostic["contrast_off_diagonal_pair_count"]
            ),
            "contrast_positive_pair_density_off_diagonal": float(
                diagnostic["contrast_positive_pair_density_off_diagonal"]
            ),
            "contrast_anchors_with_positive_count": int(
                diagnostic["contrast_anchors_with_positive_count"]
            ),
            "contrast_anchor_count": int(diagnostic["contrast_anchor_count"]),
        }
        self._active_layers = []
        self._active_stage = None
        self._pretrain_positive_keys = None
        self._pretrain_key_num_nodes = None

    @staticmethod
    def _accuracy(matrix: Mapping[int, Mapping[int, int]]) -> tuple[float, int]:
        total = sum(sum(row.values()) for row in matrix.values())
        correct = sum(row.get(expected, 0) for expected, row in matrix.items())
        return (correct / total if total else 0.0), total

    @staticmethod
    def _json_matrix(matrix: Mapping[int, Mapping[int, int]]) -> Dict[str, Dict[str, int]]:
        return {
            str(expected): {
                str(predicted): count for predicted, count in sorted(row.items())
            }
            for expected, row in sorted(matrix.items())
        }

    def diagnostics(self) -> Dict[str, object]:
        adapters = self._adapters()
        adapter_parameters = sum(
            item["a"].numel() + item["b"].numel()
            for payload in adapters.values()
            for item in payload
        )
        decision_accuracy, decisions = self._accuracy(self._router_stage_decisions)
        global_accuracy, global_decisions = self._accuracy(self._router_global_decisions)
        weighted_accuracy, weighted_queries = self._accuracy(self._router_query_weighted)
        multi_accuracy, multi_decisions = self._accuracy(
            self._multi_expert_router_stage_decisions
        )
        multi_global_accuracy, multi_global_decisions = self._accuracy(
            self._multi_expert_router_global_decisions
        )
        multi_weighted_accuracy, multi_weighted_queries = self._accuracy(
            self._multi_expert_router_query_weighted
        )
        shared = int(self.state["diagnostics"]["shared_parameter_count"])
        analytic_parameters = int(self.state["analytic_w"].numel())
        projection_elements = int(self.state["routing_projection"].numel())
        return {
            **self.state["diagnostics"],
            "method": self.name,
            "method_version": self.method_version,
            "problem_type": self.supported_problem_type,
            "prediction_unit": self.prediction_unit,
            "domain_id_observable": False,
            "router": self.router_name,
            "num_experts": len(adapters),
            "adapter_rank": self.rank,
            "adapter_parameter_count": adapter_parameters,
            "analytic_parameter_count": analytic_parameters,
            "routing_projection_frozen_elements": projection_elements,
            "total_parameter_count": shared + adapter_parameters + analytic_parameters,
            "preservation_mechanism": "recursive_closed_form_ridge_sufficient_statistics",
            "link_pretraining_objective": "elementwise_endpoint_hadamard_bce",
            "inter_domain_objective": (
                "reciprocal_nearest_euclidean_prototype_distance"
            ),
            "domain_prediction_accuracy": decision_accuracy,
            "domain_prediction_decisions": decisions,
            "global_domain_prediction_accuracy": global_accuracy,
            "global_domain_prediction_decisions": global_decisions,
            "query_weighted_domain_prediction_accuracy": weighted_accuracy,
            "domain_prediction_queries": weighted_queries,
            "multi_expert_only_domain_prediction_accuracy": multi_accuracy,
            "multi_expert_only_domain_prediction_decisions": multi_decisions,
            "multi_expert_only_global_domain_prediction_accuracy": (
                multi_global_accuracy
            ),
            "multi_expert_only_global_domain_prediction_decisions": (
                multi_global_decisions
            ),
            "multi_expert_only_query_weighted_domain_prediction_accuracy": (
                multi_weighted_accuracy
            ),
            "multi_expert_only_domain_prediction_queries": multi_weighted_queries,
            "domain_confusion_matrix": self._json_matrix(self._router_stage_decisions),
            "global_domain_confusion_matrix": self._json_matrix(
                self._router_global_decisions
            ),
            "query_weighted_domain_confusion_matrix": self._json_matrix(
                self._router_query_weighted
            ),
            "multi_expert_only_domain_confusion_matrix": self._json_matrix(
                self._multi_expert_router_stage_decisions
            ),
            "multi_expert_only_global_domain_confusion_matrix": self._json_matrix(
                self._multi_expert_router_global_decisions
            ),
            "multi_expert_only_query_weighted_domain_confusion_matrix": (
                self._json_matrix(self._multi_expert_router_query_weighted)
            ),
            "completed_stages": list(self.state["completed_stages"]),
        }

    def evaluation_topology(self, context: Any) -> None:
        """Expose truth only to post-selection audit counters, never routing."""

        self._expected_global_task = int(context.global_task_id)
        self._expected_route_stage = self.state["task_to_stage"].get(
            str(context.global_task_id)
        )
        return None

    def local_state_keys(self) -> Tuple[str, ...]:
        return (
            "adapters",
            "inter_prototypes",
            "routing_projection",
            "routing_prototypes",
            "analytic_a",
            "analytic_c",
            "analytic_w",
            "task_to_stage",
            "completed_stages",
        )

    def load_method_state(self, state: Mapping[str, object]) -> None:
        previous = self.save_method_state()
        try:
            super().load_method_state(state)
            expected = {
                "format",
                "adapters",
                "inter_prototypes",
                "routing_projection",
                "routing_prototypes",
                "analytic_a",
                "analytic_c",
                "analytic_w",
                "task_to_stage",
                "completed_stages",
                "diagnostics",
            }
            if set(self.state) != expected or self.state["format"] != self.state_format:
                raise ValueError("Invalid GraphKeeper checkpoint schema.")
            stages = self.state["completed_stages"]
            if stages != sorted(set(stages)) or any(x >= self.stage_count for x in stages):
                raise ValueError("Invalid GraphKeeper stage chronology.")
            stage_keys = {str(value) for value in stages}
            if (
                set(self._adapters()) != stage_keys
                or set(self._inter_prototypes()) != stage_keys
                or set(self._routing_prototypes()) != stage_keys
            ):
                raise ValueError("GraphKeeper expert/prototype stages differ.")
            if set(self.state["task_to_stage"].values()) != set(stages):
                raise ValueError("GraphKeeper task/expert routing map is invalid.")
            analytic = (
                self.state["analytic_a"],
                self.state["analytic_c"],
                self.state["analytic_w"],
            )
            if not all(torch.is_tensor(value) and value.ndim == 2 for value in analytic):
                raise ValueError("GraphKeeper analytic state is malformed.")
            if any(value.numel() for value in analytic) and not all(
                value.numel() for value in analytic
            ):
                raise ValueError("GraphKeeper analytic state is incomplete.")
            projection = self.state["routing_projection"]
            if not torch.is_tensor(projection) or projection.ndim != 2:
                raise ValueError("GraphKeeper routing projection is malformed.")
            self._active_stage = None
            self._active_layers = []
            self._adapter_optimizer = None
            self._pretrain_positive_keys = None
            self._pretrain_key_num_nodes = None
        except Exception:
            super().load_method_state(previous)
            raise




_RELOCATED_EXPORTS = {'GRAPHKEEPER_LP_STATE_FORMAT': ('gecko.algorithms.continual.graphkeeper.common', 'GRAPHKEEPER_LP_STATE_FORMAT'), 'GRAPHKEEPER_STATE_FORMAT': ('gecko.algorithms.continual.graphkeeper.common', 'GRAPHKEEPER_STATE_FORMAT'), 'GraphKeeperLPAlgorithm': ('gecko.algorithms.continual.graphkeeper.lp', 'GraphKeeperLPAlgorithm'), '__all__': ('gecko.algorithms.continual.graphkeeper.lp', '__all__'), '_mean_neighbors': ('gecko.algorithms.continual.graphkeeper.common', '_mean_neighbors'), '_output_size': ('gecko.algorithms.continual.graphkeeper.common', '_output_size'), '_positive_pairs': ('gecko.algorithms.continual.graphkeeper.common', '_positive_pairs'), '_query_embeddings': ('gecko.algorithms.continual.graphkeeper.common', '_query_embeddings'), '_symmetric_neighbors': ('gecko.algorithms.continual.graphkeeper.common', '_symmetric_neighbors')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)
