"""Synchronous-stage single-process federated coordinator."""

from __future__ import annotations


import hashlib
import logging
import math
import os
import random
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping

import torch
import numpy as np
from torch import nn

from gecko.evaluation.metrics import summarize_continual_matrix
from gecko.evaluation.evaluator import FederatedEvaluator
from gecko.algorithms.catalog import CompatibilityEntry
from gecko.algorithms.catalog import MethodRegistry
from gecko.models.backbones import LegacyGCNAdapter
from gecko.models.backbones import GECKOGraphModel
from gecko.models.fedfst_gat import FedFSTGAT
from gecko.models.registry import ModelRegistry
from gecko.data.streams.builder import StreamBundle
from gecko.engine.aggregation import weighted_average
from gecko.engine.client import FederatedClient
from gecko.engine.client import supervised_loss
from gecko.engine.parameter_policy import SharedParameterPolicy
from gecko.evaluation.references import NCReferenceView
from gecko.algorithms.federated.catalog import STRATEGIES

LOGGER = logging.getLogger(__name__)


class NullRunLogger:
    def log(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        return None

    def finish(self) -> None:
        return None


@dataclass
class FederatedCoordinator:
    stream: StreamBundle
    strategy_name: str = "fedavg"
    algorithm_name: str = "Bare"
    model_name: str = "auto"
    model_factory: Callable[[], nn.Module] | None = None
    run_logger: Any = None
    allow_experimental_placeholder: bool = False
    device: str | torch.device = "cpu"
    reference_mode: str | None = None
    model_seed: int | None = None
    method_config: Mapping[str, Any] | None = None
    algorithm_parameters: Mapping[str, Any] | None = None
    checkpoint_dir: str | Path | None = None
    checkpoint_every: int | None = None
    resume_from: str | Path | None = None
    evaluation_model: str = "strategy"
    stream_scientific_fingerprint: str | None = None
    source_sha: str | None = None
    diagnostic_resume_override: str | None = None
    class_mask_policy: str = "seen"

    def __post_init__(self) -> None:
        self.algorithm_parameters = dict(self.algorithm_parameters or {})
        self.class_mask_policy = str(self.class_mask_policy).lower().replace("-", "_")
        if self.class_mask_policy not in {"seen", "full"}:
            raise ValueError("class_mask_policy must be 'seen' or 'full'.")
        if self.strategy_name == "joint_oracle":
            LOGGER.warning(
                "joint_oracle is a legacy alias; use centralized_shard_oracle."
            )
            self.strategy_name = "centralized_shard_oracle"
        if self.strategy_name not in STRATEGIES:
            raise ValueError(f"Unknown server strategy: {self.strategy_name}")
        self.strategy = STRATEGIES[self.strategy_name]
        if self.strategy.requires_method_config and self.method_config is None:
            raise ValueError(f"Strategy {self.strategy_name!r} requires method_config.")
        self.v2_method_resolution = None
        if self.method_config is not None:
            from gecko.algorithms.method_config import validate_method_config

            self.v2_method_resolution = validate_method_config(
                self.method_config,
                expected_strategy=self.strategy_name,
                expected_continual_method=self.algorithm_name,
                problem_type=self.stream.scenario.problem_type,
                incremental_setting=self.stream.scenario.incremental_type,
            )
        if self.reference_mode is not None and not self.strategy.oracle:
            raise ValueError("reference_mode requires centralized_shard_oracle.")
        self.reference_view = (
            NCReferenceView(self.stream, self.reference_mode)
            if self.reference_mode is not None
            else None
        )
        self.method_registry = MethodRegistry()
        if (
            self.v2_method_resolution is not None
            and self.v2_method_resolution.name != "legacy_adapter_v1"
        ):
            resolved = self.v2_method_resolution
            self.method_compatibility = CompatibilityEntry(
                algorithm=self.algorithm_name,
                server_strategy=self.strategy_name,
                problem_type=self.stream.scenario.problem_type,
                incremental_setting=self.stream.scenario.incremental_type,
                model_family="uefa_graph_model",
                support_status=resolved.support_status,
                reason=resolved.reason,
                test_coverage=resolved.test_coverage,
                runnable=resolved.runnable,
                benchmark_eligible=resolved.benchmark_eligible,
                scientific_fidelity=resolved.scientific_fidelity,
            )
        else:
            self.method_compatibility = self.method_registry.validate(
                self.algorithm_name,
                self.strategy_name,
                self.stream.scenario.problem_type,
                self.stream.scenario.incremental_type,
                allow_experimental_placeholder=self.allow_experimental_placeholder,
            )
        if str(self.device) == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA execution was requested but CUDA is unavailable.")
        self.resolved_model_seed = (
            self.stream.config.seed if self.model_seed is None else self.model_seed
        )
        random.seed(self.resolved_model_seed)
        np.random.seed(self.resolved_model_seed % (2**32))
        torch.manual_seed(self.resolved_model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.resolved_model_seed)
        if self.device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
        self.deterministic_execution = {
            "python_seed": self.resolved_model_seed,
            "numpy_seed": self.resolved_model_seed % (2**32),
            "torch_seed": self.resolved_model_seed,
            "cuda_deterministic_algorithms": self.device.type == "cuda",
            "graph_aggregation": "sorted_target_segment_reduce_chunked_v1",
        }
        self.parameter_policy = SharedParameterPolicy()
        self.model_registry = ModelRegistry()
        base_factory = self.model_factory or self._default_model
        if self.method_config is None:
            factory = base_factory
        else:
            from gecko.models.capabilities import attach_v2_model_capabilities

            def factory() -> nn.Module:
                return attach_v2_model_capabilities(base_factory())

        self.global_model = factory().to(self.device)
        self.resolved_model_name = (
            "begin_gcn"
            if isinstance(self.global_model, LegacyGCNAdapter)
            else "fedfst_gat"
            if isinstance(self.global_model, FedFSTGAT)
            else "uefa_gcn"
            if isinstance(self.global_model, GECKOGraphModel)
            else type(self.global_model).__name__
        )
        try:
            self.model_compatibility = self.model_registry.entry(
                self.resolved_model_name
            )
        except ValueError:
            self.model_compatibility = None
        self.model_benchmark_eligible = bool(
            self.model_compatibility is not None
            and self.model_compatibility.runnable_by_default
            and self.model_compatibility.benchmark_eligible
        )
        self.global_state = self.parameter_policy.extract(self.global_model)
        self.initial_model_state_digest = self._state_digest(self.global_state)
        explicit_v2_family = None
        continual_method_parameters: Dict[str, object] = {}
        if self.v2_method_resolution is not None:
            continual_method_parameters = dict(
                self.v2_method_resolution.continual_method_parameters
            )
            if (
                self.v2_method_resolution.continual_method_name
                in self.method_registry.explicit_v2_names()
                and self.v2_method_resolution.name != "legacy_adapter_v1"
            ):
                explicit_v2_family = self.v2_method_resolution.name
        if continual_method_parameters and self.algorithm_parameters:
            raise ValueError(
                "Use method-config continual parameters or algorithm_parameters, not both."
            )
        continual_method_parameters.update(self.algorithm_parameters)
        self.clients = {
            client_id: FederatedClient(
                client_id=client_id,
                graph=graph.client_view(),
                model=factory().to(self.device),
                algorithm=self.method_registry.create(
                    self.algorithm_name,
                    allow_experimental_placeholder=self.allow_experimental_placeholder,
                    explicit_v2_family=explicit_v2_family,
                    problem_type=self.stream.scenario.problem_type,
                    incremental_setting=self.stream.scenario.incremental_type,
                    client_id=client_id,
                    seed=self.stream.config.seed,
                    **continual_method_parameters,
                ),
                config=self.stream.config,
                scenario=self.stream.scenario.client_view(),
                parameter_policy=self.parameter_policy,
                class_mask_policy=self.class_mask_policy,
            )
            for client_id, graph in self.stream.partition.client_graphs.items()
        }
        for client in self.clients.values():
            client.load_shared_state(self.global_state)
        self.run_logger = self.run_logger or NullRunLogger()

        self._stateful_runtime = None
        if self.method_config is not None:
            from gecko.engine.runtime import StatefulFederatedRuntime

            self._stateful_runtime = StatefulFederatedRuntime(
                self,
                method_config=self.method_config,
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_every=self.checkpoint_every,
                resume_from=self.resume_from,
                evaluation_model=self.evaluation_model,
                stream_scientific_fingerprint=self.stream_scientific_fingerprint,
                source_sha=self.source_sha,
                diagnostic_resume_override=self.diagnostic_resume_override,
            )

    @staticmethod
    def _state_digest(state: Dict[str, torch.Tensor]) -> str:
        digest = hashlib.sha256()
        for key, value in sorted(state.items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _state_delta_l2(
        before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]
    ) -> float:
        squared = 0.0
        for key in before:
            before_value = before[key].detach()
            after_value = after[key].detach()
            if after_value.device.type == "cuda":
                compute_device = after_value.device
            elif before_value.device.type == "cuda":
                compute_device = before_value.device
            else:
                compute_device = after_value.device
            difference = after_value.to(
                device=compute_device, dtype=torch.float64
            ) - before_value.to(device=compute_device, dtype=torch.float64)
            squared += float(difference.square().sum())
        return math.sqrt(squared)

    def _stream_tensor_versions(self) -> Dict[str, int]:
        tensors: Dict[str, torch.Tensor] = {
            "scenario.edge_index": self.stream.scenario.edge_index,
            "scenario.node_features": self.stream.scenario.node_features,
            "scenario.labels": self.stream.scenario.labels,
            "partition.node_owner": self.stream.partition.node_owner,
        }
        if self.stream.scenario.query_endpoints is not None:
            tensors["scenario.query_endpoints"] = self.stream.scenario.query_endpoints
        candidate_groups = self.stream.scenario.metadata.get("candidate_group_ids")
        if torch.is_tensor(candidate_groups):
            tensors["scenario.candidate_group_ids"] = candidate_groups
        for client_id, graph in self.stream.partition.client_graphs.items():
            tensors[f"client.{client_id}.edge_index"] = graph.edge_index
            tensors[f"client.{client_id}.node_features"] = graph.node_features
            for task_id, shard in self.stream.shards[client_id].items():
                tensors[f"client.{client_id}.task.{task_id}.queries"] = (
                    shard.train_queries
                )
                tensors[f"client.{client_id}.task.{task_id}.labels"] = (
                    shard.train_labels
                )
                if shard.context_edge_index is not None:
                    tensors[f"client.{client_id}.task.{task_id}.context_edge_index"] = (
                        shard.context_edge_index
                    )
        return {name: tensor._version for name, tensor in tensors.items()}

    def _shard_weight(self, shard: Any) -> int:
        if (
            self.stream.config.training.aggregation_weight
            == "current_positive_anchor_count"
        ):
            return shard.positive_anchor_count
        return shard.supervised_query_count

    def _primary_evaluation_support(self, query_ids: torch.Tensor) -> int:
        """Return the averaging support of one primary evaluation cell."""

        primary = self.stream.scenario.metrics[0].lower()
        if primary.startswith("hits@"):
            return int((self.stream.scenario.labels[query_ids] == 1).sum())
        return int(query_ids.numel())

    def _default_model(self) -> nn.Module:
        output_size = self.stream.scenario.num_classes
        if self.stream.scenario.problem_type == "LP":
            output_size = 1
        kwargs = {
            "input_size": self.stream.scenario.num_features,
            "output_size": output_size,
            "hidden_size": self.stream.config.training.hidden_size,
            "num_layers": self.stream.config.training.num_layers,
            "problem_type": self.stream.scenario.problem_type,
        }
        if self.model_name == "fedfst_gat":
            if self.stream.scenario.problem_type != "NC":
                raise ValueError("fedfst_gat supports node classification only.")
            return FedFSTGAT(
                input_size=self.stream.scenario.num_features,
                output_size=self.stream.scenario.num_classes,
                hidden_size=FedFSTGAT.paper_hidden_size,
                num_layers=FedFSTGAT.paper_num_layers,
                problem_type="NC",
                dropout=FedFSTGAT.paper_dropout,
            )
        if self.model_name in {"auto", "begin_gcn"}:
            try:
                return LegacyGCNAdapter(
                    **kwargs,
                    incremental_type=self.stream.scenario.incremental_type,
                )
            except RuntimeError:
                if self.model_name == "begin_gcn":
                    raise
                LOGGER.info(
                    "Original BeGin GCN dependencies unavailable; using uefa_gcn."
                )
        if self.model_name not in {"auto", "uefa_gcn"}:
            raise ValueError(
                f"Model {self.model_name!r} is registered for discovery but has no "
                "UEFA execution adapter."
            )
        return GECKOGraphModel(**kwargs)

    def _evaluate_stage(
        self,
        evaluator: FederatedEvaluator,
        matrix: torch.Tensor,
        query_counts: torch.Tensor,
        stage: int,
    ) -> Dict[str, Any]:
        evaluation_start = time.perf_counter()
        evaluator.clear_reference_cache()
        primary = self.stream.scenario.metrics[0]
        stage_values = []
        stage_weights = []
        validation_values = []
        validation_weights = []
        primary_by_client: Dict[int, List[float]] = {}
        graph_values: Dict[str, List[float]] = {}
        final_stage = stage == self.stream.scenario.num_tasks - 1
        for client_id, client in self.clients.items():
            if self.strategy.oracle:
                model = self.global_model
            else:
                if self.strategy.aggregates:
                    client.load_shared_state(self.global_state)
                model = client.model
            for task_id in sorted(client.state.seen_tasks):
                metrics = evaluator.evaluate(
                    model,
                    client_id,
                    task_id,
                    client.state.seen_tasks,
                    final_stage=final_stage,
                )
                value = metrics[primary]
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"Non-finite {primary} for client={client_id}, "
                        f"stage={stage}, task={task_id}."
                    )
                matrix[client_id, stage, task_id] = value
                central = self.stream.evaluation_shards[client_id][task_id]
                test_support = self._primary_evaluation_support(
                    central.test_query_ids
                )
                query_counts[client_id, task_id] = test_support
                if value == value:
                    stage_values.append(value)
                    stage_weights.append(test_support)
                    primary_by_client.setdefault(int(client_id), []).append(value)
                validation_metrics = evaluator.evaluate(
                    model,
                    client_id,
                    task_id,
                    client.state.seen_tasks,
                    split="validation",
                    final_stage=final_stage,
                )
                validation_value = validation_metrics[primary]
                if validation_value == validation_value:
                    validation_values.append(validation_value)
                    validation_weights.append(
                        self._primary_evaluation_support(
                            central.validation_query_ids
                        )
                    )
                for name, metric_value in metrics.items():
                    if name != primary and metric_value == metric_value:
                        graph_values.setdefault(name, []).append(metric_value)
        expected_primary_cells = sum(
            len(client.state.seen_tasks) for client in self.clients.values()
        )
        positive_query_micro = primary.lower().startswith("hits@")

        def aggregate(values: List[float], weights: List[int]) -> float | None:
            if not values:
                return None
            if not positive_query_micro:
                return sum(values) / len(values)
            denominator = sum(weights)
            if denominator <= 0:
                return None
            return sum(value * weight for value, weight in zip(values, weights)) / denominator

        output = {
            "validation_metric": (
                aggregate(validation_values, validation_weights)
            ),
            "stage_test_metric": (
                aggregate(stage_values, stage_weights)
            ),
            "diagnostic_macro_cell_stage_test_metric": (
                sum(stage_values) / len(stage_values) if stage_values else None
            ),
            "diagnostic_macro_client_stage_test_metric": (
                sum(sum(values) / len(values) for values in primary_by_client.values())
                / len(primary_by_client)
                if primary_by_client
                else None
            ),
            "stage": float(stage),
            "primary_metric_expected_cell_count": expected_primary_cells,
            "primary_metric_finite_cell_count": len(stage_values),
            "primary_metric_support_coverage": (
                len(stage_values) / expected_primary_cells
                if expected_primary_cells
                else 0.0
            ),
            "validation_primary_metric_expected_cell_count": (
                expected_primary_cells
            ),
            "validation_primary_metric_finite_cell_count": len(
                validation_values
            ),
            "validation_primary_metric_support_coverage": (
                len(validation_values) / expected_primary_cells
                if expected_primary_cells
                else 0.0
            ),
        }
        output.update(
            {
                name: sum(values) / len(values)
                for name, values in graph_values.items()
                if values
            }
        )
        output["evaluation_runtime_seconds"] = time.perf_counter() - evaluation_start
        return output

    def joint_oracle_objective(self, stage: int) -> torch.Tensor:
        """Return the query-count-weighted objective over disjoint client graphs."""

        if self.reference_view is not None and self.reference_view.uses_full_topology:
            return self._full_topology_reference_objective(stage)
        weighted_losses = []
        weights = []
        for client_id, client in self.clients.items():
            global_task_id = self.stream.orders.global_task(client_id, stage)
            shard = self.stream.shards[client_id][global_task_id]
            shard_weight = self._shard_weight(shard)
            if shard_weight == 0:
                continue
            device = next(self.global_model.parameters()).device
            if self.reference_view is not None:
                features, edge_index, queries = self.reference_view.device_inputs(
                    client_id, shard.train_queries, device
                )
            else:
                features, edge_index, queries = self._oracle_inputs(
                    client_id,
                    shard.train_queries,
                    shard.context_edge_index,
                )
                features, edge_index, queries = (
                    features.to(device),
                    edge_index.to(device),
                    queries.to(device),
                )
            logits = self.global_model.forward_queries(
                features,
                edge_index,
                queries,
                self.stream.scenario.problem_type,
            )
            mask = client._class_mask(shard)
            if (
                mask is not None
                and logits.ndim > 1
                and logits.shape[-1] == mask.shape[0]
            ):
                logits = logits.clone()
                logits[..., ~mask.to(device)] = -1e12
            loss = supervised_loss(logits, shard.train_labels.to(device))
            weighted_losses.append(loss * shard_weight)
            weights.append(shard_weight)
        if not weighted_losses:
            raise RuntimeError(
                "JointOracle received no supervised query at this stage."
            )
        return torch.stack(weighted_losses).sum() / sum(weights)

    def _full_topology_reference_objective(self, stage: int) -> torch.Tensor:
        entries = []
        for client_id, client in self.clients.items():
            task_id = self.stream.orders.global_task(client_id, stage)
            shard = self.stream.shards[client_id][task_id]
            weight = self._shard_weight(shard)
            if weight <= 0:
                continue
            _, _, queries = self.reference_view.inputs(client_id, shard.train_queries)
            entries.append((client, shard, weight, queries))
        if not entries:
            raise RuntimeError("Full-topology reference has no supervised query.")
        device = next(self.global_model.parameters()).device
        features, edge_index, _ = self.reference_view.all_node_device_inputs(0, device)
        combined_queries = torch.cat([entry[3] for entry in entries])
        combined_logits = self.global_model.forward_queries(
            features,
            edge_index,
            combined_queries.to(device),
            self.stream.scenario.problem_type,
        )
        weighted_losses = []
        cursor = 0
        for client, shard, weight, queries in entries:
            logits = combined_logits[cursor : cursor + queries.shape[0]]
            cursor += queries.shape[0]
            mask = client._class_mask(shard)
            if mask is not None and logits.ndim > 1:
                logits = logits.clone()
                logits[..., ~mask.to(device)] = -1e12
            weighted_losses.append(
                supervised_loss(logits, shard.train_labels.to(device)) * weight
            )
        return torch.stack(weighted_losses).sum() / sum(entry[2] for entry in entries)

    def _oracle_inputs(
        self,
        client_id: int,
        queries: torch.Tensor,
        context_edge_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reference_view is not None:
            if context_edge_index is not None:
                raise ValueError(
                    "NC-Domain references do not accept task edge contexts."
                )
            return self.reference_view.inputs(client_id, queries)
        graph = self.stream.partition.client_graphs[client_id]
        edge_index = (
            graph.edge_index if context_edge_index is None else context_edge_index
        )
        return graph.node_features, edge_index, queries

    def _backward_centralized_shard_objective(self, stage: int) -> float:
        """Backpropagate the exact shard objective without retaining all graphs."""

        if self.reference_view is not None and self.reference_view.uses_full_topology:
            objective = self._full_topology_reference_objective(stage)
            objective.backward()
            return float(objective.detach())
        task_ids = {
            client_id: self.stream.orders.global_task(client_id, stage)
            for client_id in self.clients
        }
        raw_weights = {
            client_id: self._shard_weight(self.stream.shards[client_id][task_id])
            for client_id, task_id in task_ids.items()
        }
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            raise RuntimeError(
                "CentralizedShardOracle received no supervised query at this stage."
            )
        objective_value = 0.0
        device = next(self.global_model.parameters()).device
        for client_id, client in self.clients.items():
            task_id = task_ids[client_id]
            shard = self.stream.shards[client_id][task_id]
            if self.reference_view is not None:
                features, edge_index, queries = self.reference_view.device_inputs(
                    client_id, shard.train_queries, device
                )
            else:
                features, edge_index, queries = self._oracle_inputs(
                    client_id,
                    shard.train_queries,
                    shard.context_edge_index,
                )
                features, edge_index, queries = (
                    features.to(device),
                    edge_index.to(device),
                    queries.to(device),
                )
            logits = self.global_model.forward_queries(
                features,
                edge_index,
                queries,
                self.stream.scenario.problem_type,
            )
            mask = client._class_mask(shard)
            if mask is not None and logits.ndim > 1:
                logits = logits.clone()
                logits[..., ~mask.to(device)] = -1e12
            loss = supervised_loss(logits, shard.train_labels.to(device))
            weighted = loss * (raw_weights[client_id] / total_weight)
            weighted.backward()
            objective_value += float(weighted.detach())
        return objective_value

    def _run_oracle_round(self, stage: int, round_id: int) -> Dict[str, Any]:
        """Optimize a centralized weighted objective on disjoint client graphs."""

        round_start = time.perf_counter()
        self.global_model.train()
        optimizer = torch.optim.Adam(
            self.global_model.parameters(),
            lr=self.stream.config.training.learning_rate,
            weight_decay=self.stream.config.training.weight_decay,
        )
        before_state = {key: value.clone() for key, value in self.global_state.items()}
        participant_ids = sorted(self.clients)
        participant_tasks = {
            client_id: int(self.stream.orders.global_task(client_id, stage))
            for client_id in participant_ids
        }
        raw_weights = {
            client_id: int(
                self._shard_weight(
                    self.stream.shards[client_id][participant_tasks[client_id]]
                )
            )
            for client_id in participant_ids
        }
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            raise RuntimeError(
                "CentralizedShardOracle received no supervised query at this stage."
            )
        normalized_weights = {
            client_id: weight / total_weight
            for client_id, weight in raw_weights.items()
        }
        last_loss = 0.0
        for _ in range(self.stream.config.training.local_epochs_per_round):
            optimizer.zero_grad()
            last_loss = self._backward_centralized_shard_objective(stage)
            optimizer.step()
        if not math.isfinite(last_loss):
            raise RuntimeError("CentralizedShardOracle produced non-finite loss.")
        self.global_state = self.parameter_policy.extract(self.global_model)
        return {
            "stage": stage,
            "round": round_id,
            "participants": len(self.clients),
            "participant_ids": participant_ids,
            "participant_global_task_ids": participant_tasks,
            "raw_aggregation_weights": raw_weights,
            "normalized_aggregation_weights": normalized_weights,
            "normalized_aggregation_weight_sum": sum(normalized_weights.values()),
            "empty_updates": 0,
            "mean_training_loss": last_loss,
            "round_runtime_seconds": time.perf_counter() - round_start,
            "communication_payload_bytes": 0,
            "oracle": True,
            "oracle_semantics": (
                self.reference_view.mode.semantics
                if self.reference_view is not None
                else "centralized_shard_query_weighted_objective"
            ),
            "server_parameter_delta_l2": self._state_delta_l2(
                before_state, self.global_state
            ),
        }

    def run(self) -> Dict[str, Any]:
        if self._stateful_runtime is not None:
            return self._stateful_runtime.run()
        num_clients = self.stream.config.partition.num_clients
        num_tasks = self.stream.scenario.num_tasks
        matrix = torch.full((num_clients, num_tasks, num_tasks), float("nan"))
        query_counts = torch.zeros(num_clients, num_tasks)
        evaluator = FederatedEvaluator(
            self.stream,
            reference_view=self.reference_view,
            class_mask_policy=self.class_mask_policy,
        )
        round_records: List[Dict[str, Any]] = []
        stage_records: List[Dict[str, float]] = []
        total_communication = 0
        global_step = 0
        start_time = time.perf_counter()
        initial_stream_versions = self._stream_tensor_versions()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for stage in range(num_tasks):
            stage_start = time.perf_counter()
            stage_participants: set[int] = set()
            for round_id in range(self.stream.config.training.rounds_per_stage):
                if self.strategy.oracle:
                    record = self._run_oracle_round(stage, round_id)
                    round_records.append(record)
                    self.run_logger.log(record, step=global_step)
                    global_step += 1
                    continue
                participants = self.stream.participation.trace[stage][round_id]
                stage_participants.update(int(value) for value in participants)
                participant_tasks = {
                    int(client_id): int(
                        self.stream.orders.global_task(client_id, stage)
                    )
                    for client_id in participants
                }
                if (
                    self.stream.config.order.profile == "synchronized"
                    and len(set(participant_tasks.values())) != 1
                ):
                    raise RuntimeError(
                        "Synchronized order assigned different global tasks in one round."
                    )
                updates = []
                client_deltas = {}
                round_start = time.perf_counter()
                broadcast_bytes = 0
                if self.strategy.aggregates:
                    broadcast_bytes = len(participants) * sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in self.global_state.values()
                    )
                for client_id in participants:
                    client = self.clients[client_id]
                    global_task_id = self.stream.orders.global_task(client_id, stage)
                    before_client_state = self.parameter_policy.extract(client.model)
                    shard = self.stream.shards[client_id][global_task_id]
                    if self.strategy_name in {"fedavg", "fedprox"}:
                        client.load_shared_state(self.global_state)
                    update = client.update(
                        shard,
                        global_task_id=global_task_id,
                        server_state=self.global_state,
                        strategy=self.strategy_name,
                    )
                    updates.append(update)
                    if update.global_task_id != global_task_id:
                        raise RuntimeError(
                            "Client update reported the wrong global task ID."
                        )
                    if not math.isfinite(update.training_loss):
                        raise RuntimeError(
                            f"Non-finite training loss for client={client_id}, "
                            f"stage={stage}, task={global_task_id}."
                        )
                    client_deltas[int(client_id)] = self._state_delta_l2(
                        before_client_state, update.shared_state
                    )
                positive_updates = [update for update in updates if update.weight > 0]
                if len(positive_updates) != len(updates):
                    empty_clients = [
                        update.client_id for update in updates if update.weight <= 0
                    ]
                    raise RuntimeError(
                        f"Active clients produced empty updates: {empty_clients}."
                    )
                raw_weights = {
                    int(update.client_id): int(update.weight) for update in updates
                }
                total_weight = sum(raw_weights.values())
                normalized_weights = {
                    client_id: weight / total_weight
                    for client_id, weight in raw_weights.items()
                }
                normalized_weight_sum = sum(normalized_weights.values())
                if not math.isclose(normalized_weight_sum, 1.0, abs_tol=1e-12):
                    raise RuntimeError(
                        "Normalized aggregation weights do not sum to one."
                    )
                server_before = {
                    key: value.clone() for key, value in self.global_state.items()
                }
                if self.strategy.aggregates and positive_updates:
                    self.global_state = weighted_average(updates)
                if self.strategy_name in {"fedavg", "fedprox"}:
                    self.parameter_policy.load(self.global_model, self.global_state)
                upload_bytes = (
                    sum(update.communication_bytes for update in updates)
                    if self.strategy.aggregates
                    else 0
                )
                round_payload = broadcast_bytes + upload_bytes
                total_communication += round_payload
                record = {
                    "stage": stage,
                    "round": round_id,
                    "participants": len(participants),
                    "participant_ids": [int(value) for value in participants],
                    "participant_global_task_ids": participant_tasks,
                    "raw_aggregation_weights": raw_weights,
                    "normalized_aggregation_weights": normalized_weights,
                    "normalized_aggregation_weight_sum": normalized_weight_sum,
                    "empty_updates": len(updates) - len(positive_updates),
                    "mean_training_loss": sum(
                        update.training_loss for update in updates
                    )
                    / max(1, len(updates)),
                    "round_runtime_seconds": time.perf_counter() - round_start,
                    "communication_payload_bytes": round_payload,
                    "client_parameter_delta_l2": client_deltas,
                    "server_parameter_delta_l2": (
                        self._state_delta_l2(server_before, self.global_state)
                        if self.strategy.aggregates
                        else None
                    ),
                }
                round_records.append(record)
                self.run_logger.log(record, step=global_step)
                global_step += 1
            if not self.strategy.oracle:
                for client_id in sorted(stage_participants):
                    client = self.clients[client_id]
                    global_task_id = self.stream.orders.global_task(client_id, stage)
                    client.consolidate_task(
                        self.stream.shards[client_id][global_task_id],
                        global_task_id=global_task_id,
                    )
            for client_id, client in self.clients.items():
                global_task_id = self.stream.orders.global_task(client_id, stage)
                client.mark_seen(
                    global_task_id,
                    self.stream.shards[client_id][global_task_id].task_class_mask,
                )
            stage_metrics = self._evaluate_stage(evaluator, matrix, query_counts, stage)
            stage_metrics["stage_runtime_seconds"] = time.perf_counter() - stage_start
            stage_records.append(stage_metrics)
            self.run_logger.log(stage_metrics, step=global_step)
        summary = summarize_continual_matrix(
            matrix,
            self.stream.orders,
            query_counts,
            base_metric=self.stream.scenario.metrics[0],
        )
        for client_id in range(num_clients):
            order = self.stream.orders.client_orders[client_id]
            for stage in range(num_tasks):
                expected = set(order[: stage + 1])
                for task_id in range(num_tasks):
                    populated = bool(torch.isfinite(matrix[client_id, stage, task_id]))
                    if populated != (task_id in expected):
                        raise RuntimeError(
                            "A[k,s,t] population mismatch for "
                            f"client={client_id}, stage={stage}, task={task_id}."
                        )
        final_stream_versions = self._stream_tensor_versions()
        if initial_stream_versions != final_stream_versions:
            changed = sorted(
                key
                for key in initial_stream_versions
                if initial_stream_versions[key] != final_stream_versions.get(key)
            )
            raise RuntimeError(f"In-memory stream tensors were mutated: {changed}.")
        result = {
            "benchmark_name": "UEFA",
            "stream_id": self.stream.stream_id,
            "stream_hash": self.stream.stream_hash,
            "strategy": self.strategy_name,
            "client_continual_algorithm": self.algorithm_name,
            "algorithm_parameters": dict(self.algorithm_parameters),
            "method_support_status": self.method_compatibility.support_status,
            "model_support_status": (
                self.model_compatibility.release_status
                if self.model_compatibility is not None
                else "unregistered_custom_model"
            ),
            "model_benchmark_tier": (
                self.model_compatibility.benchmark_tier
                if self.model_compatibility is not None
                else "unsupported_custom"
            ),
            "model_scientific_fidelity": (
                self.model_compatibility.scientific_fidelity
                if self.model_compatibility is not None
                else "not_verified"
            ),
            "model_benchmark_eligible": self.model_benchmark_eligible,
            "benchmark_eligible": (
                self.method_compatibility.benchmark_eligible
                and self.model_benchmark_eligible
                and not self.strategy.oracle
            ),
            "strategy_benchmark_eligible": not self.strategy.oracle,
            "scientific_fidelity": self.method_compatibility.scientific_fidelity,
            "model": self.model_name,
            "resolved_model": self.resolved_model_name,
            "execution_device": str(self.device),
            "deterministic_execution": dict(self.deterministic_execution),
            "initial_model_state_digest": self.initial_model_state_digest,
            "model_seed": self.resolved_model_seed,
            "training_budget": {
                "optimizer": "Adam",
                "learning_rate": self.stream.config.training.learning_rate,
                "weight_decay": self.stream.config.training.weight_decay,
                "rounds_per_stage": (self.stream.config.training.rounds_per_stage),
                "local_epochs_per_round": (
                    self.stream.config.training.local_epochs_per_round
                ),
            },
            "strategy_hyperparameters": {
                "fedprox_mu": (
                    self.stream.config.training.fedprox_mu
                    if self.strategy_name == "fedprox"
                    else None
                ),
            },
            "client_algorithm_hyperparameters": {
                client_id: client.algorithm.hyperparameters()
                for client_id, client in self.clients.items()
            },
            "client_algorithm_local_state_keys": {
                client_id: client.algorithm.local_state_keys()
                for client_id, client in self.clients.items()
            },
            "oracle": self.strategy.oracle,
            "oracle_semantics": (
                self.reference_view.mode.semantics
                if self.reference_view is not None
                else "centralized_shard_query_weighted_objective"
                if self.strategy.oracle
                else None
            ),
            "diagnostic_reference": (
                self.reference_view.metadata()
                if self.reference_view is not None
                else None
            ),
            "base_metric": self.stream.scenario.metrics[0],
            "client_stage_task_matrix": matrix,
            "A[k,s,t]": matrix,
            "query_counts": query_counts,
            "client_stage_task_matrix_population_verified": True,
            "in_memory_stream_tensors_unchanged": True,
            "summary": summary,
            "primary_metric_aggregation": (
                "positive_query_micro"
                if self.stream.scenario.metrics[0].lower().startswith("hits@")
                else "macro_cell"
            ),
            "macro_client_metric_role": (
                "diagnostic_only"
                if self.stream.scenario.metrics[0].lower().startswith("hits@")
                else "co_primary_summary"
            ),
            "rounds": round_records,
            "stages": stage_records,
            "graph_specific_report": {
                **self.stream.partition.diagnostics,
                **(
                    {
                        key: value
                        for key, value in stage_records[-1].items()
                        if key.startswith(("boundary_", "interior_"))
                    }
                    if stage_records
                    else {}
                ),
            },
            "communication_payload_bytes": total_communication,
            "runtime_seconds": time.perf_counter() - start_time,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else None
            ),
            "model_parameter_count": sum(
                parameter.numel() for parameter in self.global_model.parameters()
            ),
        }
        self.run_logger.log(summary, step=global_step + 1)
        return result
