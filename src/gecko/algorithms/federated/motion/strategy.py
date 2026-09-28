"""MOTION strategy for leakage-safe UEFA NC-Class execution."""

from __future__ import annotations

import math
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from gecko.task_masks import task_aware_cross_entropy

from gecko.engine.accounting import ResourceLedger
from gecko.engine.accounting import tensor_payload_bytes
from gecko.algorithms.federated.motion.gepae import motion_gepae_aggregate
from gecko.algorithms.federated.motion.gtmsc import MotionGraphMemory
from gecko.algorithms.federated.motion.gtmsc import MotionReservoir
from gecko.algorithms.federated.motion.gtmsc import motion_coarsen_graph
from gecko.algorithms.federated.motion.gtmsc import motion_memory_from_state
from gecko.algorithms.federated.motion.gtmsc import motion_merge_observed_graph
from gecko.algorithms.federated.motion.gtmsc import motion_update_reservoir
from gecko.engine.protocol import AggregationResult
from gecko.engine.protocol import BroadcastPayload
from gecko.engine.protocol import BroadcastReason
from gecko.engine.protocol import ClientUpload
from gecko.engine.protocol import EvaluationSelection
from gecko.engine.protocol import FrozenTensorMap
from gecko.engine.protocol import ParameterManifest
from gecko.engine.protocol import RoundContext
from gecko.engine.protocol import TensorState


class MotionStrategy:
    """Client-private G-TMSC replay with server-only G-EPAE aggregation."""

    strategy_version = "uefa-motion-strategy-v3"
    upstream_commit = "24e8402f9a17e289ff48577e6dc7a47d075aed88"
    name = "motion"
    aggregates = True
    oracle = False
    uses_proximal_objective = False

    def __init__(
        self,
        *,
        buffer_size: int = 200,
        expert_select: int = 3,
        k_list: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
        node_reduction_rate: float = 0.5,
        pcb_ratio: float = 0.1,
        pcb_min_ratio: float = 0.0001,
        pcb_max_ratio: float = 0.0001,
        replay_ceiling_bytes: int = 16 * 1024 * 1024,
        replay_weight: float = 1.0,
        similarity_threshold: float = 0.7,
        use_node_positional: bool = True,
        use_node_mmd: bool = True,
        use_node_mahalanobis: bool = True,
    ) -> None:
        if (
            isinstance(buffer_size, bool)
            or not isinstance(buffer_size, int)
            or buffer_size <= 0
        ):
            raise ValueError("MOTION buffer_size must be a positive integer.")
        if (
            isinstance(expert_select, bool)
            or not isinstance(expert_select, int)
            or expert_select <= 0
        ):
            raise ValueError("MOTION expert_select must be a positive integer.")
        ratios = tuple(float(value) for value in k_list)
        if (
            not ratios
            or expert_select > len(ratios)
            or any(
                not math.isfinite(value) or not 0.0 < value < 1.0
                for value in ratios
            )
        ):
            raise ValueError("MOTION k_list/expert_select controls are invalid.")
        fixed_units = {
            "node_reduction_rate": node_reduction_rate,
            "pcb_ratio": pcb_ratio,
            "similarity_threshold": similarity_threshold,
        }
        for name, value in fixed_units.items():
            if not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0:
                raise ValueError(f"MOTION {name} must lie in (0, 1).")
        for name, value in {
            "pcb_min_ratio": pcb_min_ratio,
            "pcb_max_ratio": pcb_max_ratio,
        }.items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0:
                raise ValueError(f"MOTION {name} must lie in [0, 1).")
        if float(pcb_min_ratio) + float(pcb_max_ratio) >= 1.0:
            raise ValueError("MOTION PCB clamp ratios must sum to less than one.")
        if replay_ceiling_bytes != 16 * 1024 * 1024:
            raise ValueError("MOTION replay ceiling must equal 16 MiB.")
        if not math.isfinite(float(replay_weight)) or float(replay_weight) <= 0.0:
            raise ValueError("MOTION replay_weight must be finite and positive.")
        for name, value in {
            "use_node_positional": use_node_positional,
            "use_node_mmd": use_node_mmd,
            "use_node_mahalanobis": use_node_mahalanobis,
        }.items():
            if not isinstance(value, bool):
                raise TypeError(f"MOTION {name} must be boolean.")

        self.buffer_size = buffer_size
        self.expert_select = expert_select
        self.k_list = ratios
        self.node_reduction_rate = float(node_reduction_rate)
        self.pcb_ratio = float(pcb_ratio)
        self.pcb_min_ratio = float(pcb_min_ratio)
        self.pcb_max_ratio = float(pcb_max_ratio)
        self.replay_ceiling_bytes = replay_ceiling_bytes
        self.replay_weight = float(replay_weight)
        self.similarity_threshold = float(similarity_threshold)
        self.use_node_positional = use_node_positional
        self.use_node_mmd = use_node_mmd
        self.use_node_mahalanobis = use_node_mahalanobis
        self._manifest: ParameterManifest | None = None
        self._compute_device = torch.device("cpu")
        self._client_ids: tuple[int, ...] = ()
        self._shared_state = FrozenTensorMap()
        self._active_graphs: dict[int, MotionGraphMemory] = {}
        self._last_replay_losses: dict[int, float] = {}
        self._task_class_masks: dict[int, torch.Tensor] = {}
        self._task_aware: bool | None = None
        self._client_memory_nodes: dict[int, int] = {}
        self._client_reservoir_nodes: dict[int, int] = {}
        self._client_coarsening_counts: dict[int, int] = {}
        self._last_scale_nonzero_fraction = 0.0
        self._last_scale_mean = 0.0
        self._completed_rounds = 0
        self._initialized = False

    def bind_runtime(
        self,
        *,
        model_template: torch.nn.Module,
        feature_dim: int,
        num_classes: int,
        model_seed: int,
        device: torch.device,
    ) -> None:
        """Bind the explicit tensor compute device before initialization."""

        del model_template, feature_dim, num_classes, model_seed
        self._compute_device = torch.device(device)
        if self._compute_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MOTION CUDA compute was requested but unavailable.")

    def _require_initialized(self) -> None:
        if not self._initialized or self._manifest is None:
            raise RuntimeError("MOTION has not been initialized.")

    def _validate_state(
        self, state: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        self._require_initialized()
        assert self._manifest is not None
        expected = tuple(self._manifest.shared_trainable)
        if set(state) != set(expected):
            raise ValueError("MOTION state keys do not match shared trainables.")
        reference = self._shared_state.materialize()
        output: dict[str, torch.Tensor] = {}
        for key in expected:
            value = state[key]
            if (
                not torch.is_tensor(value)
                or value.shape != reference[key].shape
                or value.dtype != reference[key].dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"MOTION state tensor {key!r} is invalid.")
            output[key] = value.detach().clone().contiguous()
        return output

    def initialize(
        self,
        shared_state: TensorState,
        parameter_manifest: ParameterManifest,
        client_ids: Iterable[int],
    ) -> None:
        if self._initialized:
            raise RuntimeError("MOTION cannot be initialized twice.")
        clients = tuple(sorted(int(client_id) for client_id in client_ids))
        if (
            not clients
            or len(set(clients)) != len(clients)
            or any(client_id < 0 for client_id in clients)
        ):
            raise ValueError("MOTION client IDs must be unique non-negative integers.")
        if not tuple(parameter_manifest.shared_trainable) or set(shared_state) != set(
            parameter_manifest.shared_trainable
        ):
            raise ValueError("MOTION requires every shared trainable parameter.")
        self._manifest = parameter_manifest
        self._client_ids = clients
        self._shared_state = FrozenTensorMap(shared_state)
        self._initialized = True

    @property
    def shared_state(self) -> dict[str, torch.Tensor]:
        self._require_initialized()
        return self._shared_state.materialize()

    def prepare_payload(
        self, context: RoundContext, client_id: int, reason: BroadcastReason
    ) -> BroadcastPayload:
        self._require_initialized()
        if client_id not in context.participant_ids or client_id not in self._client_ids:
            raise ValueError("MOTION payload client is not a known participant.")
        resources = ResourceLedger()
        if reason == "initialization":
            resources.add(
                initialization_model_downlink_bytes=self._shared_state.payload_bytes
            )
        elif reason == "training":
            resources.add(training_model_downlink_bytes=self._shared_state.payload_bytes)
        elif reason == "evaluation":
            resources.add(evaluation_sync_bytes=self._shared_state.payload_bytes)
        elif reason != "resume":
            raise ValueError(f"Unknown MOTION broadcast reason {reason!r}.")
        return BroadcastPayload(
            client_id=client_id,
            round_context=context,
            reason=reason,
            model_state=self._shared_state,
            metadata={
                "strategy_version": self.strategy_version,
                "client_graph_payload": False,
            },
            resources=resources,
        )

    @staticmethod
    def _motion_client_state(client: Any) -> dict[str, Any]:
        root = client.state.strategy_state
        state = root.get("motion")
        if state is None:
            state = {
                "strategy_version": MotionStrategy.strategy_version,
                "sampled_tasks": (),
                "coarsened_tasks": (),
                "reservoir": {},
                "memory": {},
                "last_coarsening": {},
            }
            root["motion"] = state
        if (
            not isinstance(state, dict)
            or state.get("strategy_version") != MotionStrategy.strategy_version
            or set(state)
            != {
                "strategy_version",
                "sampled_tasks",
                "coarsened_tasks",
                "reservoir",
                "memory",
                "last_coarsening",
            }
        ):
            raise ValueError("MOTION client-private state is malformed.")
        return state

    @staticmethod
    def _reservoir_from_state(state: Mapping[str, Any]) -> MotionReservoir | None:
        if not state:
            return None
        if set(state) != {"node_ids", "labels", "seen_samples"}:
            raise ValueError("MOTION reservoir state fields do not match.")
        return MotionReservoir(
            node_ids=state["node_ids"].detach().clone().to(dtype=torch.long),
            labels=state["labels"].detach().clone().to(dtype=torch.long),
            seen_samples=int(state["seen_samples"]),
        )

    def client_receive(
        self, client: Any, payload: BroadcastPayload, method_context: Any
    ) -> None:
        if int(client.client_id) != payload.client_id:
            raise ValueError("MOTION payload was delivered to the wrong client.")
        client.load_shared_state(payload.model_state.materialize())
        client.algorithm.on_broadcast(method_context, payload)
        if payload.reason != "training":
            return
        if method_context.problem_type != "NC":
            raise ValueError("Initial MOTION scope supports NC only.")
        is_task = str(method_context.incremental_setting).lower() == "task"
        if self._task_aware is not None and self._task_aware != is_task:
            raise ValueError("MOTION cannot mix Class-IL and Task-IL contexts.")
        self._task_aware = is_task
        if is_task:
            mask = method_context.valid_class_mask
            if mask is None or mask.dtype != torch.bool or mask.ndim != 1:
                raise ValueError("MOTION Task-IL requires a class mask.")
            task = int(method_context.global_task_id)
            owned = mask.detach().cpu().clone().contiguous()
            previous = self._task_class_masks.get(task)
            if previous is not None and not torch.equal(previous, owned):
                raise ValueError("MOTION observed two masks for one global task.")
            self._task_class_masks[task] = owned

        state = self._motion_client_state(client)
        sampled_tasks = tuple(int(task) for task in state["sampled_tasks"])
        task_id = int(method_context.global_task_id)
        memory = (
            None
            if not state["memory"]
            else motion_memory_from_state(state["memory"])
        )
        reservoir = self._reservoir_from_state(state["reservoir"])
        if task_id not in sampled_tasks:
            reservoir = motion_update_reservoir(
                reservoir,
                method_context.train_queries,
                method_context.train_labels,
                capacity=self.buffer_size,
                seed=(int(client.client_id) + 1) * 1_000_003 + task_id,
            )
            num_classes = int(client.scenario.num_classes)
            memory = motion_merge_observed_graph(
                memory,
                node_features=method_context.node_features.to(self._compute_device),
                edge_index=method_context.effective_edge_index.to(self._compute_device),
                train_queries=method_context.train_queries.to(self._compute_device),
                train_labels=method_context.train_labels.to(self._compute_device),
                num_classes=num_classes,
            )
            sampled_tasks = sampled_tasks + (task_id,)
            state["sampled_tasks"] = sampled_tasks
            state["reservoir"] = {
                "node_ids": reservoir.node_ids.detach().clone(),
                "labels": reservoir.labels.detach().clone(),
                "seen_samples": int(reservoir.seen_samples),
            }
        if memory is None or reservoir is None:
            raise RuntimeError("MOTION failed to initialize client graph memory.")
        state["memory"] = memory.state_dict()
        self._active_graphs[int(client.client_id)] = memory
        self._client_memory_nodes[int(client.client_id)] = int(memory.features.shape[0])
        self._client_reservoir_nodes[int(client.client_id)] = int(
            reservoir.node_ids.numel()
        )

    def augment_loss(
        self,
        model: torch.nn.Module,
        method_context: Any,
        loss: torch.Tensor,
        shared_keys: tuple[str, ...],
    ) -> torch.Tensor:
        """Add loss on the client's private merged/coarsened graph."""

        del shared_keys
        client_id = int(method_context.client_id)
        graph = self._active_graphs.get(client_id)
        if graph is None or not bool(graph.train_mask.any()):
            return loss
        replay_queries = torch.nonzero(graph.train_mask, as_tuple=True)[0]
        replay_labels = graph.labels[replay_queries]
        logits = method_context.forward_queries(
            model,
            replay_queries,
            node_features=graph.features,
            edge_index=graph.edge_index,
        )
        replay_labels = replay_labels.to(device=logits.device, dtype=torch.long)
        if self._task_aware:
            replay_loss = task_aware_cross_entropy(
                logits, replay_labels, self._task_class_masks
            )
        else:
            class_mask = method_context.valid_class_mask
            if class_mask is not None and logits.ndim == 2 and logits.shape[1] == class_mask.shape[0]:
                logits = logits.clone()
                logits[:, ~class_mask.to(logits.device)] = -1e12
            replay_loss = F.cross_entropy(logits, replay_labels)
        self._last_replay_losses[client_id] = float(replay_loss.detach())
        # Upstream DYGRA_reservior.observe optimizes only the merged graph.
        # Adding the base current-task CE would count current nodes twice.
        return self.replay_weight * replay_loss

    def transform_gradients(
        self, model: torch.nn.Module, method_context: Any, shared_keys: tuple[str, ...]
    ) -> None:
        return None

    def finalize_upload(
        self, client: Any, local_result: Any, method_context: Any
    ) -> ClientUpload:
        """Upload only model parameters; G-TMSC runs once at task boundary."""

        self._require_initialized()
        client_id = int(client.client_id)
        if method_context.problem_type != "NC":
            raise ValueError("Initial MOTION scope supports NC only.")
        if (
            local_result.client_id != client_id
            or local_result.global_task_id != int(method_context.global_task_id)
            or client_id not in self._client_ids
        ):
            raise ValueError("MOTION local result identity mismatch.")
        model_state = self._validate_state(local_result.shared_state)
        graph = self._active_graphs.get(client_id)
        if graph is None:
            raise RuntimeError("MOTION client graph was not prepared before training.")
        state = self._motion_client_state(client)
        reservoir = self._reservoir_from_state(state["reservoir"])
        if reservoir is None:
            raise RuntimeError("MOTION reservoir is missing during upload.")
        return ClientUpload(
            client_id=client_id,
            global_task_id=local_result.global_task_id,
            weight=local_result.weight,
            training_loss=local_result.training_loss,
            model_state=model_state,
            auxiliary_state={},
            diagnostics={
                "strategy_version": self.strategy_version,
                "gtmsc_active_nodes": int(graph.features.shape[0]),
                "gtmsc_runs_this_task": 0,
                "reservoir_nodes": int(reservoir.node_ids.numel()),
                "client_private_graph_uploaded": False,
            },
            resources=ResourceLedger(
                training_model_uplink_bytes=FrozenTensorMap(model_state).payload_bytes
            ),
        )

    def consolidate_client(self, client: Any, method_context: Any) -> None:
        """Coarsen the merged evolving graph once after all task rounds."""

        self._require_initialized()
        client_id = int(client.client_id)
        state = self._motion_client_state(client)
        task_id = int(method_context.global_task_id)
        coarsened_tasks = tuple(int(value) for value in state["coarsened_tasks"])
        if task_id in coarsened_tasks:
            raise RuntimeError("MOTION attempted to coarsen one task more than once.")
        graph = self._active_graphs.get(client_id)
        if graph is None:
            graph = motion_memory_from_state(state["memory"])
        reservoir = self._reservoir_from_state(state["reservoir"])
        if reservoir is None:
            raise RuntimeError("MOTION reservoir is missing during coarsening.")
        client.model.eval()
        with torch.no_grad():
            hidden = method_context.encode_nodes(
                client.model,
                node_features=graph.features,
                edge_index=graph.edge_index,
            )
        coarsened = motion_coarsen_graph(
            graph,
            hidden.detach(),
            protected_raw_nodes=reservoir.node_ids,
            reduction_rate=self.node_reduction_rate,
            k_list=self.k_list,
            expert_select=self.expert_select,
            use_node_positional=self.use_node_positional,
            use_node_mmd=self.use_node_mmd,
            use_node_mahalanobis=self.use_node_mahalanobis,
            similarity_threshold=self.similarity_threshold,
        )
        memory = coarsened.graph
        if memory.payload_bytes > self.replay_ceiling_bytes:
            raise ValueError("MOTION client graph memory exceeds the 16 MiB ceiling.")
        state["memory"] = memory.state_dict()
        state["coarsened_tasks"] = coarsened_tasks + (task_id,)
        state["last_coarsening"] = {
            "global_task_id": task_id,
            "stage_index": int(method_context.stage_index),
            "round_index": int(method_context.round_index),
            "input_nodes": int(graph.features.shape[0]),
            "coarsened_nodes": int(memory.features.shape[0]),
            "protected_nodes": int(reservoir.node_ids.numel()),
        }
        self._active_graphs[client_id] = memory
        self._client_memory_nodes[client_id] = int(memory.features.shape[0])
        self._client_reservoir_nodes[client_id] = int(reservoir.node_ids.numel())
        self._client_coarsening_counts[client_id] = len(state["coarsened_tasks"])

    def client_replay_payload_bytes(self, client: Any) -> int:
        state = self._motion_client_state(client)
        return tensor_payload_bytes(state["memory"]) + tensor_payload_bytes(
            state["reservoir"]
        )

    def aggregate(
        self, context: RoundContext, uploads: tuple[ClientUpload, ...]
    ) -> AggregationResult:
        self._require_initialized()
        upload_ids = tuple(upload.client_id for upload in uploads)
        if len(set(upload_ids)) != len(upload_ids) or set(upload_ids) != set(
            context.participant_ids
        ):
            raise ValueError("MOTION requires exactly one upload per participant.")
        if any(upload.auxiliary_state for upload in uploads):
            raise ValueError("MOTION clients must not upload graph-memory tensors.")
        states = {
            upload.client_id: self._validate_state(upload.model_state)
            for upload in uploads
        }
        weights = {upload.client_id: int(upload.weight) for upload in uploads}
        new_state, scales = motion_gepae_aggregate(
            self._shared_state.materialize(),
            states,
            weights,
            pcb_ratio=self.pcb_ratio,
            pcb_min_ratio=self.pcb_min_ratio,
            pcb_max_ratio=self.pcb_max_ratio,
        )
        resources = ResourceLedger()
        for upload in uploads:
            resources.merge(upload.resources)
        self._shared_state = FrozenTensorMap(new_state)
        self._completed_rounds += 1
        self._last_scale_nonzero_fraction = float((scales > 0).float().mean())
        self._last_scale_mean = float(scales.mean())
        return AggregationResult(
            shared_state=self._shared_state,
            diagnostics={
                "strategy_version": self.strategy_version,
                "aggregation": "motion_gepae_pcb",
                "participants": list(sorted(context.participant_ids)),
                "scale_nonzero_fraction": self._last_scale_nonzero_fraction,
                "scale_mean": self._last_scale_mean,
                "client_auxiliary_uplink_bytes": 0,
            },
            resources=resources,
        )

    def personalize(
        self, context: RoundContext, result: AggregationResult
    ) -> AggregationResult:
        return result

    def select_evaluation(
        self, client_id: int, stage_index: int
    ) -> EvaluationSelection:
        self._require_initialized()
        if client_id not in self._client_ids or stage_index < 0:
            raise ValueError("Unknown MOTION evaluation client or stage.")
        return EvaluationSelection(
            client_id=client_id,
            source="shared",
            model_state=self._shared_state,
            count_evaluation_sync=True,
        )

    def diagnostics(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "upstream_commit": self.upstream_commit,
            "fidelity": "clean_room_leakage_safe_mechanism_adaptation",
            "supported_scope": "NC-Class/NC-Task",
            "gtmsc": "dynamic_merge_multi_expert_similarity_coarsening",
            "gtmsc_lifecycle": "merge_once_and_coarsen_once_per_task",
            "local_objective": "merged_graph_cross_entropy",
            "gepae": "parameter_compatibility_balance",
            "client_memory_is_local_only": True,
            "client_graph_or_feature_uplink": False,
            "compute_device": str(self._compute_device),
            "gpu_tensor_kernels": self._compute_device.type == "cuda",
            "buffer_size": self.buffer_size,
            "expert_select": self.expert_select,
            "expert_select_contract": "paper_s3_official_code_hardcodes_s2",
            "k_list": list(self.k_list),
            "node_reduction_rate": self.node_reduction_rate,
            "similarity_threshold": self.similarity_threshold,
            "replay_ceiling_bytes": self.replay_ceiling_bytes,
            "replay_weight": self.replay_weight,
            "pcb_ratio": self.pcb_ratio,
            "pcb_min_ratio": self.pcb_min_ratio,
            "pcb_max_ratio": self.pcb_max_ratio,
            "completed_rounds": self._completed_rounds,
            "clients_with_memory": sorted(self._client_memory_nodes),
            "client_memory_nodes": {
                f"client-{client_id}": nodes
                for client_id, nodes in sorted(self._client_memory_nodes.items())
            },
            "client_reservoir_nodes": {
                f"client-{client_id}": nodes
                for client_id, nodes in sorted(self._client_reservoir_nodes.items())
            },
            "client_coarsening_counts": {
                f"client-{client_id}": count
                for client_id, count in sorted(self._client_coarsening_counts.items())
            },
            "clients_with_replay_loss": sorted(self._last_replay_losses),
            "last_replay_losses": {
                f"client-{client_id}": value
                for client_id, value in sorted(self._last_replay_losses.items())
            },
            "last_scale_nonzero_fraction": self._last_scale_nonzero_fraction,
            "last_scale_mean": self._last_scale_mean,
            "task_aware": bool(self._task_aware),
            "task_class_masks": sorted(self._task_class_masks),
        }

    def state_dict(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "client_ids": self._client_ids,
            "shared_state": self._shared_state.materialize(),
            "last_replay_losses": dict(self._last_replay_losses),
            "client_memory_nodes": dict(self._client_memory_nodes),
            "client_reservoir_nodes": dict(self._client_reservoir_nodes),
            "client_coarsening_counts": dict(self._client_coarsening_counts),
            "last_scale_nonzero_fraction": self._last_scale_nonzero_fraction,
            "last_scale_mean": self._last_scale_mean,
            "completed_rounds": self._completed_rounds,
            "task_aware": self._task_aware,
            "task_class_masks": {
                task: mask.detach().cpu().clone()
                for task, mask in self._task_class_masks.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._require_initialized()
        expected = {
            "strategy_version",
            "client_ids",
            "shared_state",
            "last_replay_losses",
            "client_memory_nodes",
            "client_reservoir_nodes",
            "client_coarsening_counts",
            "last_scale_nonzero_fraction",
            "last_scale_mean",
            "completed_rounds",
            "task_aware",
            "task_class_masks",
        }
        if set(state) != expected or state["strategy_version"] != self.strategy_version:
            raise ValueError("MOTION checkpoint identity mismatch.")
        if tuple(state["client_ids"]) != self._client_ids:
            raise ValueError("MOTION checkpoint client IDs mismatch.")
        completed = state["completed_rounds"]
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise ValueError("MOTION checkpoint completed rounds are invalid.")
        self._shared_state = FrozenTensorMap(self._validate_state(state["shared_state"]))
        self._last_replay_losses = {
            int(key): float(value)
            for key, value in dict(state["last_replay_losses"]).items()
        }
        self._client_memory_nodes = {
            int(key): int(value)
            for key, value in dict(state["client_memory_nodes"]).items()
        }
        self._client_reservoir_nodes = {
            int(key): int(value)
            for key, value in dict(state["client_reservoir_nodes"]).items()
        }
        self._client_coarsening_counts = {
            int(key): int(value)
            for key, value in dict(state["client_coarsening_counts"]).items()
        }
        self._last_scale_nonzero_fraction = float(state["last_scale_nonzero_fraction"])
        self._last_scale_mean = float(state["last_scale_mean"])
        self._completed_rounds = completed
        task_aware = state["task_aware"]
        if task_aware not in {None, True, False}:
            raise ValueError("MOTION checkpoint task-aware flag is invalid.")
        masks = state["task_class_masks"]
        if not isinstance(masks, Mapping):
            raise ValueError("MOTION checkpoint task masks are invalid.")
        self._task_aware = task_aware
        self._task_class_masks = {
            int(task): mask.detach().cpu().clone().bool()
            for task, mask in masks.items()
        }
        self._active_graphs = {}
