"""Stateful FedFST strategy: FedAvg followed by HHKR and HLST."""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Mapping

import torch

from gecko.engine.accounting import ResourceLedger
from gecko.engine.accounting import tensor_payload_bytes
from gecko.engine.protocol import AggregationResult
from gecko.engine.protocol import FrozenTensorMap
from gecko.engine.protocol import ParameterManifest
from gecko.engine.protocol import RoundContext
from gecko.engine.protocol import TensorState
from gecko.algorithms.federated.legacy import LegacyStrategyAdapter
from gecko.algorithms.context import ClientMethodContext

from gecko.algorithms.federated.fedfst.client import ClientHHKRUpload
from gecko.algorithms.federated.fedfst.client import FedFSTClientWorkspace
from gecko.algorithms.federated.fedfst.client import history_payload_bytes
from gecko.algorithms.federated.fedfst.client import validate_client_history_boundary
from gecko.algorithms.federated.fedfst.core import FEDFST_IMPLEMENTATION_VERSION
from gecko.algorithms.federated.fedfst.core import PAPER_DOI
from gecko.algorithms.federated.fedfst.core import REFERENCE_CODE_COMMIT
from gecko.algorithms.federated.fedfst.core import ConditionalFeatureGenerator
from gecko.algorithms.federated.fedfst.core import FedFSTParameters
from gecko.algorithms.federated.fedfst.core import adjust_spectral_energy
from gecko.algorithms.federated.fedfst.core import derive_seed
from gecko.algorithms.federated.fedfst.core import generate_balanced_features
from gecko.algorithms.federated.fedfst.core import hlst_loss
from gecko.algorithms.federated.fedfst.core import local_generator
from gecko.algorithms.federated.fedfst.core import random_block_diagonal_edges
from gecko.algorithms.federated.fedfst.core import random_undirected_edges
from gecko.algorithms.federated.fedfst.core import sampled_feature_indices
from gecko.algorithms.federated.fedfst.core import task_block_diagonal_edges
from gecko.algorithms.federated.fedfst.core import weighted_generator_average
from gecko.algorithms.federated.fedfst.core import weighted_spectral_target


def _tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _clone_json(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("FedFST metadata contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        return {str(key): _clone_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clone_json(item) for item in value]
    raise TypeError(f"FedFST metadata contains unsupported {type(value).__name__}.")


def _validated_tensor_map(
    value: object,
    reference: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> FrozenTensorMap:
    """Validate a checkpoint tensor map without mutating live strategy state."""

    if not isinstance(value, Mapping) or set(value) != set(reference):
        raise ValueError(f"FedFST checkpoint {label} keys mismatch.")
    prepared: dict[str, torch.Tensor] = {}
    for name, expected in reference.items():
        actual = value[name]
        if (
            not torch.is_tensor(actual)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not actual.is_floating_point()
            or not bool(torch.isfinite(actual).all())
        ):
            raise ValueError(
                f"FedFST checkpoint {label} tensor {name!r} is invalid."
            )
        prepared[name] = actual.detach().cpu().clone().contiguous()
    return FrozenTensorMap(prepared)


class FedFSTStrategy(LegacyStrategyAdapter):
    """Paper-equation FedFST strategy under strict UEFA visibility rules."""

    strategy_version = FEDFST_IMPLEMENTATION_VERSION

    def __init__(
        self, **parameters: int | float | str | tuple[int, ...]
    ) -> None:
        super().__init__("fedavg")
        self.name = "fedfst"
        self.parameters = FedFSTParameters(**parameters)
        self._model_template: torch.nn.Module | None = None
        self._device = torch.device("cpu")
        self._feature_dim = 0
        self._num_classes = 0
        self._model_seed: int | None = None
        self._bound = False
        self._previous_task_state = FrozenTensorMap()
        self._global_generator_state = FrozenTensorMap()
        self._global_generator_classes = torch.empty((0,), dtype=torch.long)
        self._global_spectral_target: float | None = None
        self._task_order: tuple[int, ...] = ()
        self._client_task_orders: dict[int, tuple[int, ...]] = {}
        self._historical_classes_by_stage: tuple[tuple[int, ...], ...] = ()
        self._incremental_setting: str | None = None
        self._task_class_masks: dict[int, torch.Tensor] = {}
        self._stream_adaptation_digest = ""
        self._stage_records: list[dict[str, object]] = []
        self._teacher_snapshot_stages: list[int] = []

    def training_class_mask(
        self, method_context: ClientMethodContext
    ) -> torch.Tensor | None:
        """Resolve the paper-faithful client-loss output mask.

        The linked FedFST implementation applies cross entropy to the complete
        classifier head in class-incremental training.  This changes neither
        the labels available to a client nor UEFA's seen-class evaluator.  The
        task-incremental adaptation retains its immutable task-head mask.
        """

        if (
            method_context.incremental_setting == "class"
            and self.parameters.class_il_output_training_policy == "paper_full"
        ):
            return None
        return method_context.valid_class_mask

    def local_optimizer_hyperparameters(
        self,
        method_context: ClientMethodContext,
        *,
        default_learning_rate: float,
        default_weight_decay: float,
    ) -> tuple[float, float]:
        """Use the linked FedFST client Adam hyperparameters at fixed budget."""

        del method_context, default_learning_rate, default_weight_decay
        return (
            float(self.parameters.client_learning_rate),
            float(self.parameters.client_weight_decay),
        )

    def bind_runtime(
        self,
        *,
        model_template: torch.nn.Module,
        feature_dim: int,
        num_classes: int,
        model_seed: int,
        device: torch.device | str,
    ) -> None:
        """Bind public model capabilities without retaining a stream object."""

        if self._bound or self._initialized:
            raise RuntimeError("FedFST runtime may be bound only once before initialize.")
        if feature_dim <= 0 or num_classes <= 1:
            raise ValueError("FedFST model dimensions are invalid.")
        if isinstance(model_seed, bool) or not isinstance(model_seed, int):
            raise ValueError("FedFST model seed must be an integer.")
        forward = getattr(model_template, "forward_queries", None)
        if not callable(forward):
            raise RuntimeError("FedFST requires a verified NC forward_queries model.")
        self._model_template = copy.deepcopy(model_template).cpu()
        self._device = torch.device(device)
        self._feature_dim = int(feature_dim)
        self._num_classes = int(num_classes)
        self._model_seed = model_seed
        self._bound = True

    def validate_stream(self, stream: Any) -> None:
        """Fail closed on schedules not studied by the published method."""

        scenario = stream.scenario
        incremental_setting = str(scenario.incremental_type).lower()
        if scenario.problem_type != "NC" or incremental_setting not in {
            "class",
            "task",
        }:
            raise ValueError(
                "FedFST supports single-label NC-Class/Task streams "
                "only."
            )
        if self._bound and (
            scenario.num_features != self._feature_dim
            or scenario.num_classes != self._num_classes
        ):
            raise ValueError("FedFST stream/model dimensions disagree.")
        if scenario.labels.ndim != 1:
            raise ValueError("FedFST does not support multi-label node targets.")
        orders = tuple(
            tuple(int(task) for task in stream.orders.client_orders[client_id])
            for client_id in sorted(stream.orders.client_orders)
        )
        if not orders or any(
            len(order) != scenario.num_tasks
            or set(order) != set(range(scenario.num_tasks))
            for order in orders
        ):
            raise ValueError("FedFST requires one complete task permutation per client.")
        self._client_task_orders = dict(zip(sorted(stream.orders.client_orders), orders))
        self._task_order = orders[0]
        self._incremental_setting = incremental_setting
        task_masks = scenario.task_masks
        if (
            not torch.is_tensor(task_masks)
            or task_masks.dtype != torch.bool
            or tuple(task_masks.shape)
            != (scenario.num_tasks, scenario.num_classes)
        ):
            raise ValueError("FedFST requires explicit immutable class-task masks.")
        self._task_class_masks = {
            task_id: task_masks[task_id].detach().cpu().clone().contiguous()
            for task_id in range(scenario.num_tasks)
        }
        historical: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for task_id in orders[0]:
            task_classes = {
                int(value)
                for value in torch.where(task_masks[int(task_id)])[0].tolist()
            }
            if not task_classes or seen.intersection(task_classes):
                raise ValueError(
                    "FedFST requires non-empty disjoint class-incremental tasks."
                )
            seen.update(task_classes)
        for stage_index in range(scenario.num_tasks):
            prior_classes = {
                int(label)
                for task_id in self._prior_tasks(stage_index)
                for label in torch.where(task_masks[task_id])[0].tolist()
            }
            historical.append(tuple(sorted(prior_classes)))
        self._historical_classes_by_stage = tuple(historical)
        global_train_classes: dict[int, set[int]] = {
            task_id: set() for task_id in self._task_order
        }
        for client_id in sorted(stream.orders.client_orders):
            client_shards = stream.shards[client_id]
            for task_id in self._task_order:
                shard = client_shards[task_id]
                queries = shard.train_queries
                labels = shard.train_labels
                if (
                    not torch.is_tensor(queries)
                    or queries.dtype != torch.long
                    or queries.ndim != 1
                    or queries.numel() == 0
                    or not torch.is_tensor(labels)
                    or labels.dtype != torch.long
                    or labels.ndim != 1
                    or labels.shape[0] != queries.shape[0]
                ):
                    raise ValueError(
                        "FedFST requires a non-empty scalar NC train shard for "
                        "every client/task."
                    )
                task_classes = set(
                    int(value)
                    for value in torch.where(task_masks[int(task_id)])[0].tolist()
                )
                observed = {int(value) for value in labels.tolist()}
                if not observed.issubset(task_classes):
                    raise ValueError(
                        "FedFST train labels disagree with the immutable task mask."
                    )
                global_train_classes[int(task_id)].update(observed)
        for task_id in self._task_order:
            expected_classes = set(
                int(value)
                for value in torch.where(task_masks[int(task_id)])[0].tolist()
            )
            if global_train_classes[int(task_id)] != expected_classes:
                raise ValueError(
                    "FedFST requires global train support for every scheduled class."
                )
        all_clients = set(range(stream.config.partition.num_clients))
        raw_trace = stream.participation.trace
        stages = (
            sorted(raw_trace.items())
            if isinstance(raw_trace, Mapping)
            else enumerate(raw_trace)
        )
        observed_stages = 0
        for stage_index, stage in stages:
            observed_stages += 1
            rounds = (
                sorted(stage.items())
                if isinstance(stage, Mapping)
                else enumerate(stage)
            )
            stage_union: set[int] = set()
            for round_index, participants in rounds:
                resolved = {int(value) for value in participants}
                if not resolved or not resolved.issubset(all_clients):
                    raise ValueError(
                        "FedFST fixed participation trace contains an empty or "
                        "out-of-range round at "
                        f"stage={stage_index}, round={round_index}."
                    )
                stage_union.update(resolved)
            if stage_union != all_clients:
                raise ValueError(
                    "FedFST requires the immutable ordinary participation trace "
                    "to cover every client at least once per stage; the stage-union "
                    f"is incomplete at stage={stage_index}."
                )
        if observed_stages != scenario.num_tasks:
            raise ValueError("FedFST participation trace has the wrong stage count.")

        adaptation = hashlib.sha256()
        adaptation.update(incremental_setting.encode("ascii"))
        adaptation.update(b"fixed_trace_stage_union")
        adaptation.update(str(sorted(self._client_task_orders.items())).encode("ascii"))
        for task_id in self._task_order:
            adaptation.update(str(task_id).encode("ascii"))
            adaptation.update(
                self._task_class_masks[task_id].numpy().tobytes()
            )
        self._stream_adaptation_digest = adaptation.hexdigest()

    def _prior_tasks(
        self, stage_index: int, client_id: int | None = None
    ) -> tuple[int, ...]:
        """Resolve client history or the server union using immutable task IDs."""
        if client_id is not None:
            return self._client_task_orders[client_id][:stage_index]
        return tuple(dict.fromkeys(
            task_id
            for client in sorted(self._client_task_orders)
            for task_id in self._client_task_orders[client][:stage_index]
        ))

    def _prior_head_masks(
        self, stage_index: int, client_id: int | None = None
    ) -> Mapping[int, torch.Tensor] | None:
        """Return only logits whose semantics existed in the frozen teacher.

        Task-IL retains one immutable mask per prior task.  Class-IL uses one
        union mask over the historical vocabulary.  The latter is essential:
        full-head CE on old-only synthetic labels would explicitly suppress
        current and future classifier rows that the previous teacher never
        defined.
        """

        if stage_index <= 0:
            return None
        prior_tasks = self._prior_tasks(stage_index, client_id)
        if self._incremental_setting == "task":
            return {
                task_id: self._task_class_masks[task_id]
                for task_id in prior_tasks
            }
        historical = torch.zeros((self._num_classes,), dtype=torch.bool)
        for task_id in prior_tasks:
            historical |= self._task_class_masks[task_id]
        if not bool(historical.any()):
            raise RuntimeError("FedFST historical Class-IL head is empty.")
        return {0: historical.contiguous()}

    def _prior_topology_masks(
        self, stage_index: int
    ) -> Mapping[int, torch.Tensor] | None:
        """Return prior Task-IL blocks; Class-IL has one shared topology."""

        if self._incremental_setting != "task" or stage_index <= 0:
            return None
        return {
            task_id: self._task_class_masks[task_id]
            for task_id in self._prior_tasks(stage_index)
        }

    def initialize(
        self,
        shared_state: TensorState,
        parameter_manifest: ParameterManifest,
        client_ids: Iterable[int],
    ) -> None:
        if not self._bound:
            raise RuntimeError("FedFST must bind its runtime before initialization.")
        super().initialize(shared_state, parameter_manifest, client_ids)
        assert self._model_template is not None
        trainable = {
            name
            for name, parameter in self._model_template.named_parameters()
            if parameter.requires_grad
        }
        if trainable != set(parameter_manifest.shared_trainable):
            raise RuntimeError(
                "FedFST requires every trainable forward parameter to be shared."
            )
        manifest_buffers = set(parameter_manifest.local_buffers)
        model_buffers = dict(self._model_template.named_buffers())
        if manifest_buffers != set(model_buffers):
            raise RuntimeError("FedFST server/local buffer policy is inconsistent.")
        if any(
            value.is_floating_point() and not bool(torch.isfinite(value).all())
            for value in model_buffers.values()
        ):
            raise RuntimeError("FedFST model template contains a non-finite buffer.")

    def _make_model(self, shared_state: Mapping[str, torch.Tensor]) -> torch.nn.Module:
        if self._model_template is None or self._manifest is None:
            raise RuntimeError("FedFST model template/manifest is unavailable.")
        if set(shared_state) != set(self._manifest.shared_trainable):
            raise ValueError("FedFST shared model keys do not match the manifest.")
        model = copy.deepcopy(self._model_template).to(self._device)
        full_state = model.state_dict()
        for name, value in shared_state.items():
            if name not in full_state or full_state[name].shape != value.shape:
                raise ValueError(f"FedFST shared model tensor {name!r} is invalid.")
            full_state[name] = value.detach().to(
                device=full_state[name].device, dtype=full_state[name].dtype
            )
        model.load_state_dict(full_state, strict=True)
        return model

    def _extract_shared(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        if self._manifest is None:
            raise RuntimeError("FedFST parameter manifest is unavailable.")
        state = model.state_dict()
        output = {
            name: state[name].detach().cpu().clone().contiguous()
            for name in self._manifest.shared_trainable
        }
        if any(not torch.isfinite(value).all() for value in output.values()):
            raise RuntimeError("FedFST server model became non-finite.")
        return output

    def _new_generator(
        self, stage_index: int, classes: tuple[int, ...]
    ) -> ConditionalFeatureGenerator:
        return ConditionalFeatureGenerator(
            feature_dim=self._feature_dim,
            class_ids=classes,
            noise_dim=self.parameters.noise_dim,
            dropout=self.parameters.generator_dropout,
            initialization_seed=local_generator(
                self.parameters.method_seed, 10, stage_index
            ).initial_seed(),
        )

    @staticmethod
    def _upload_metadata_bytes(upload: ClientHHKRUpload) -> int:
        # float64 S_h and int64 historical train-node count.
        return 8 + 8

    def _run_generator_federation(
        self,
        *,
        stage_index: int,
        conditioning_classes: tuple[int, ...],
        workspaces: Mapping[int, FedFSTClientWorkspace],
        teacher: torch.nn.Module,
        resources: ResourceLedger,
    ) -> tuple[
        dict[str, torch.Tensor],
        float,
        tuple[int, ...],
        dict[int, ClientHHKRUpload],
    ]:
        if not conditioning_classes:
            raise ValueError("FedFST HHKR requires historical conditioning classes.")
        generator_model = self._new_generator(stage_index, conditioning_classes)
        global_state = {
            name: value.detach().cpu().clone()
            for name, value in generator_model.state_dict().items()
        }
        final_uploads: dict[int, ClientHHKRUpload] = {}
        resources.add(
            training_auxiliary_downlink_bytes=(
                self._previous_task_state.payload_bytes * len(workspaces)
            )
        )
        for client_id in sorted(workspaces):
            workspaces[client_id].bind_teacher(teacher, device=self._device)
        for generator_round in range(self.parameters.generator_rounds):
            resources.add(
                training_auxiliary_downlink_bytes=(
                    (
                        tensor_payload_bytes(global_state)
                        + 8 * len(conditioning_classes)
                    )
                    * len(workspaces)
                )
            )
            uploads: dict[int, ClientHHKRUpload] = {}
            for client_id in sorted(workspaces):
                upload = workspaces[client_id].execute_hhkr(
                    initial_generator_state=global_state,
                    feature_dim=self._feature_dim,
                    conditioning_classes=conditioning_classes,
                    stage_index=stage_index,
                    generator_round=generator_round,
                    parameters=self.parameters,
                    device=self._device,
                )
                if upload.client_id != client_id:
                    raise RuntimeError(
                        "FedFST HHKR upload/client identity mismatch."
                    )
                uploads[client_id] = upload
                resources.add(
                    training_auxiliary_uplink_bytes=(
                        upload.tensor_payload_bytes
                        + self._upload_metadata_bytes(upload)
                    )
                )
            counts = {
                client_id: upload.historical_node_count
                for client_id, upload in uploads.items()
            }
            global_state = weighted_generator_average(
                {
                    client_id: upload.generator_state
                    for client_id, upload in uploads.items()
                },
                counts,
            )
            final_uploads = uploads
        counts = {
            client_id: upload.historical_node_count
            for client_id, upload in final_uploads.items()
        }
        target = weighted_spectral_target(
            {
                client_id: upload.spectral_energy
                for client_id, upload in final_uploads.items()
            },
            counts,
        )
        return global_state, target, conditioning_classes, final_uploads

    def _server_synthetic_graph(
        self,
        *,
        stage_index: int,
        generator_state: Mapping[str, torch.Tensor],
        classes: tuple[int, ...],
        spectral_target: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        model = self._new_generator(stage_index, classes).to(self._device)
        model.load_state_dict(generator_state, strict=True)
        model.eval()
        device_rng = local_generator(
            self.parameters.method_seed, 20, stage_index, device=self._device
        )
        topology_rng = local_generator(
            self.parameters.method_seed, 22, stage_index, device=self._device
        )
        with torch.no_grad():
            features, labels = generate_balanced_features(
                model,
                classes,
                self.parameters.server_nodes_per_class,
                generator=device_rng,
                device=self._device,
            )
        # The paper leaves the initial server adjacency density unspecified.
        # A fixed Cora-scale density of two arcs/node cannot reach the
        # unnormalized-Laplacian energy observed on OGBN-Arxiv.  Start at the
        # target energy scale so the paper's pruning-only optimizer can move
        # downward toward the target instead of being structurally unable to
        # create the missing density.
        requested_density = self.parameters.generated_edges_per_node
        if self.parameters.server_initial_edge_policy == "target_scaled":
            requested_density = max(
                requested_density,
                int(math.ceil(float(spectral_target))),
            )
        elif self.parameters.server_initial_edge_policy == "target_scaled_capped":
            requested_density = min(
                self.parameters.server_max_edges_per_node,
                max(
                    requested_density,
                    int(math.ceil(float(spectral_target))),
                ),
            )
        topology_masks = self._prior_topology_masks(stage_index)
        if topology_masks is None:
            edges = random_undirected_edges(
                int(labels.shape[0]),
                directed_edges_per_node=requested_density,
                generator=topology_rng,
                device=self._device,
            )
            construction_policy = (
                "paper_fixed_global_random"
                if self.parameters.server_initial_edge_policy == "paper_fixed"
                else (
                    "target_scaled_capped_global_random"
                    if self.parameters.server_initial_edge_policy
                    == "target_scaled_capped"
                    else "target_scaled_global_random"
                )
            )
        else:
            edges = random_block_diagonal_edges(
                labels.detach(),
                topology_masks,
                directed_edges_per_node=requested_density,
                generator=topology_rng,
                device=self._device,
            )
            construction_policy = (
                "paper_fixed_task_block_random"
                if self.parameters.server_initial_edge_policy == "paper_fixed"
                else (
                    "target_scaled_capped_task_block_random"
                    if self.parameters.server_initial_edge_policy
                    == "target_scaled_capped"
                    else "target_scaled_task_block_random"
                )
            )
        raw_directed_edges = int(edges.shape[1])
        feature_rng = local_generator(self.parameters.method_seed, 21, stage_index)
        feature_indices = sampled_feature_indices(
            self._feature_dim,
            self.parameters.sampled_feature_fraction,
            generator=feature_rng,
        )
        adjustment, final_energy = adjust_spectral_energy(
            features.detach(),
            labels.detach(),
            edges,
            feature_indices=feature_indices.to(self._device),
            target=spectral_target,
            reduction_ratio=self.parameters.edge_reduction_ratio,
            tolerance=self.parameters.topology_tolerance,
            max_iterations=self.parameters.topology_max_iterations,
            generator=topology_rng,
            compute_device=self._device,
        )
        resolved_edges = adjustment.edge_index
        metadata = {
            "target_spectral_energy": float(spectral_target),
            "initial_spectral_energy": float(adjustment.initial_value),
            "final_spectral_energy": float(adjustment.final_value),
            "feature_spectral_energy": float(final_energy.feature),
            "structure_spectral_energy": float(final_energy.structure),
            "topology_iterations": int(adjustment.iterations),
            "topology_converged": bool(adjustment.converged),
            "removed_edge_types": list(adjustment.removed_edge_types),
            "sampled_feature_indices": list(final_energy.sampled_feature_indices),
            "task_block_diagonal": topology_masks is not None,
            "cross_task_directed_edges_removed": 0,
            "construction_policy": construction_policy,
            "configured_minimum_directed_edges_per_node": int(
                self.parameters.generated_edges_per_node
            ),
            "requested_initial_directed_edges_per_node": int(requested_density),
            "realized_initial_directed_edges_per_node": (
                float(raw_directed_edges) / float(labels.shape[0])
            ),
            "initial_target_absolute_error": abs(
                float(adjustment.initial_value) - float(spectral_target)
            ),
            "final_target_absolute_error": abs(
                float(adjustment.final_value) - float(spectral_target)
            ),
        }
        return features, labels, resolved_edges, metadata

    def _distill(
        self,
        *,
        student_state: Mapping[str, torch.Tensor],
        teacher: torch.nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor,
        edge_index: torch.Tensor,
        task_class_masks: Mapping[int, torch.Tensor] | None,
        validation_evaluator: Callable[
            [Mapping[str, torch.Tensor]], Mapping[str, float | int]
        ],
    ) -> tuple[
        dict[str, torch.Tensor],
        tuple[tuple[float, float, float], ...],
        dict[str, object],
    ]:
        student = self._make_model(student_state)
        student.train()
        # UEFA keeps BatchNorm and other persistent buffers client-local.  HLST
        # therefore optimizes shared parameters with every buffer-owning module
        # frozen in evaluation mode, while buffer-free Dropout modules retain
        # the paper's train-mode behavior.  Snapshot/assert below fails closed
        # if an unknown module still mutates a discarded local buffer.
        frozen_buffers = {
            name: value.detach().cpu().clone()
            for name, value in student.named_buffers()
        }
        for module in student.modules():
            if any(value is not None for value in module._buffers.values()):
                # Do not call ``eval()`` here: it recurses into children and a
                # buffer-owning parent could accidentally disable descendant
                # Dropout.  Only the module that owns the persistent buffer is
                # frozen; buffer-free stochastic layers retain train mode.
                module.training = False
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        optimizer = torch.optim.Adam(
            (parameter for parameter in student.parameters() if parameter.requires_grad),
            lr=self.parameters.distillation_learning_rate,
        )
        queries = torch.arange(labels.shape[0], dtype=torch.long, device=self._device)
        trace: list[tuple[float, float, float]] = []
        cuda_devices: list[int] = []
        if self._device.type == "cuda":
            cuda_devices.append(
                torch.cuda.current_device()
                if self._device.index is None
                else self._device.index
            )
        best_state: dict[str, torch.Tensor] | None = None
        best_score = -math.inf
        best_epoch = -1
        validation_trace: list[dict[str, float | int]] = []
        stopped_early = False

        def score_checkpoint(epoch: int) -> tuple[float, float, float]:
            nonlocal best_epoch, best_score, best_state
            candidate = self._extract_shared(student)
            raw = validation_evaluator(candidate)
            required = {
                "mean_seen_validation",
                "mean_current_validation",
                "mean_previous_validation",
                "finite_cells",
                "expected_cells",
            }
            if set(raw) != required:
                raise RuntimeError(
                    "FedFST validation selector returned an invalid aggregate schema."
                )
            mean_seen = float(raw["mean_seen_validation"])
            mean_current = float(raw["mean_current_validation"])
            mean_previous = float(raw["mean_previous_validation"])
            finite_cells = int(raw["finite_cells"])
            expected_cells = int(raw["expected_cells"])
            if (
                not math.isfinite(mean_seen)
                or not math.isfinite(mean_current)
                or not math.isfinite(mean_previous)
                or finite_cells <= 0
                or finite_cells != expected_cells
            ):
                raise RuntimeError(
                    "FedFST validation checkpoint has incomplete/non-finite coverage."
                )
            validation_trace.append(
                {
                    "epoch": epoch,
                    "mean_seen_validation": mean_seen,
                    "mean_current_validation": mean_current,
                    "mean_previous_validation": mean_previous,
                    "finite_cells": finite_cells,
                    "expected_cells": expected_cells,
                }
            )
            # Strict improvement preserves the earliest checkpoint on ties.
            if mean_seen > best_score:
                best_score = mean_seen
                best_epoch = epoch
                best_state = candidate
            return mean_seen, mean_current, mean_previous

        with torch.random.fork_rng(devices=cuda_devices):
            seed = derive_seed(
                self.parameters.method_seed, 30, len(self._stage_records)
            )
            cpu_rng = torch.Generator(device="cpu")
            cpu_rng.manual_seed(seed)
            torch.set_rng_state(cpu_rng.get_state())
            if self._device.type == "cuda":
                cuda_rng = torch.Generator(device=self._device)
                cuda_rng.manual_seed(seed)
                torch.cuda.set_rng_state(cuda_rng.get_state(), self._device)
            score_checkpoint(0)
            checkpoints = set(self.parameters.distillation_validation_checkpoints)
            for epoch_index in range(self.parameters.distillation_epochs):
                optimizer.zero_grad(set_to_none=True)
                student_logits = student.forward_queries(
                    features, edge_index, queries, "NC"
                )
                with torch.no_grad():
                    teacher_logits = teacher.forward_queries(
                        features, edge_index, queries, "NC"
                    )
                losses = hlst_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    edge_index,
                    smoothing_hops=self.parameters.smoothing_hops,
                    lambda_low=self.parameters.lambda_low,
                    task_class_masks=task_class_masks,
                )
                if not torch.isfinite(losses.total):
                    raise RuntimeError("FedFST HLST loss became non-finite.")
                losses.total.backward()
                optimizer.step()
                trace.append(
                    (
                        float(losses.total.detach().cpu()),
                        float(losses.cross_entropy.detach().cpu()),
                        float(losses.low_frequency_kl.detach().cpu()),
                    )
                )
                completed_epoch = epoch_index + 1
                if completed_epoch in checkpoints:
                    _, current_score, previous_score = score_checkpoint(
                        completed_epoch
                    )
                    if (
                        self.parameters.distillation_early_stop_policy
                        == "author_balance_crossing"
                        and completed_epoch >= 2
                        and previous_score > current_score
                    ):
                        stopped_early = True
                        break
        if any(
            not torch.equal(before, dict(student.named_buffers())[name].detach().cpu())
            for name, before in frozen_buffers.items()
        ):
            raise RuntimeError("FedFST HLST mutated a client-local model buffer.")
        if best_state is None or best_epoch < 0:
            raise RuntimeError("FedFST validation selector produced no checkpoint.")
        selection = {
            "policy": "validation_only_mean_seen_best_earliest_tie",
            "selected_epoch": best_epoch,
            "selected_mean_seen_validation": best_score,
            "stopped_early": stopped_early,
            "evaluated_checkpoints": validation_trace,
            "method_received_validation_labels_or_features": False,
            "method_received_test_metrics": False,
        }
        return best_state, tuple(trace), selection

    def finalize_stage(
        self,
        context: RoundContext,
        clients: Mapping[int, Any],
        method_contexts: Mapping[int, Any],
        *,
        validation_evaluator: Callable[
            [Mapping[str, torch.Tensor]], Mapping[str, float | int]
        ] | None = None,
    ) -> AggregationResult:
        """Run post-task HHKR/HLST, then make current train data historical."""

        self._require_initialized()
        if set(clients) != set(context.participant_ids) or set(method_contexts) != set(
            context.participant_ids
        ):
            raise ValueError("FedFST stage clients/contexts do not match participants.")
        if set(context.participant_ids) != set(self._client_ids):
            raise ValueError("FedFST stage finalization requires every client.")
        stage_index = int(context.stage_index)
        if len(self._stage_records) != stage_index:
            raise RuntimeError("FedFST stage finalization is out of order.")
        if (
            not self._task_order
            or len(self._historical_classes_by_stage) != len(self._task_order)
            or stage_index >= len(self._task_order)
        ):
            raise RuntimeError("FedFST immutable stream schedule is unavailable.")
        expected_round = int(context.round_index)
        for client_id in sorted(clients):
            expected_task = self._client_task_orders[client_id][stage_index]
            if (
                isinstance(clients[client_id].client_id, bool)
                or not isinstance(clients[client_id].client_id, int)
                or clients[client_id].client_id != client_id
            ):
                raise ValueError(
                    "FedFST client mapping key/identity mismatch."
                )
            method_context = method_contexts[client_id]
            if (
                int(method_context.client_id) != client_id
                or int(method_context.stage_index) != stage_index
                or int(method_context.round_index) != expected_round
                or int(method_context.global_task_id) != expected_task
                or context.task_for(client_id) != expected_task
                or method_context.problem_type != "NC"
                or method_context.incremental_setting != self._incremental_setting
            ):
                raise ValueError(
                    "FedFST client context does not match the immutable stage/task "
                    "boundary."
                )

        # Every client and server mutation below is prepared on private state.
        # The live strategy/client states are assigned only after HHKR, HLST, and
        # every history append have succeeded.
        workspaces = {
            client_id: FedFSTClientWorkspace(
                clients[client_id],
                feature_dim=self._feature_dim,
                stage_index=stage_index,
                expected_task_ids=self._prior_tasks(stage_index, client_id),
                task_class_masks=self._prior_head_masks(stage_index, client_id),
            )
            for client_id in sorted(clients)
        }
        candidate_shared = FrozenTensorMap(self._shared_state)
        candidate_generator = FrozenTensorMap(self._global_generator_state)
        candidate_generator_classes = self._global_generator_classes.detach().clone()
        candidate_spectral_target = self._global_spectral_target
        resources = ResourceLedger()
        stage_record: dict[str, object] = {
            "stage_index": stage_index,
            "hhkr_applied": False,
            "hlst_applied": False,
            "contains_raw_feature_upload": False,
            "contains_raw_label_upload": False,
            "contains_raw_edge_upload": False,
            "incremental_setting": self._incremental_setting,
            "task_aware": self._incremental_setting == "task",
            "participation_policy": "fixed_trace_stage_union",
        }
        if stage_index > 0:
            if validation_evaluator is None:
                raise RuntimeError(
                    "FedFST HLST requires the UEFA validation-only checkpoint evaluator."
                )
            if not self._previous_task_state:
                raise RuntimeError("FedFST has no immutable previous-task teacher.")
            teacher = self._make_model(self._previous_task_state.materialize())
            teacher_before = {
                name: value.detach().cpu().clone()
                for name, value in teacher.state_dict().items()
            }
            generator_state, spectral_target, classes, uploads = (
                self._run_generator_federation(
                    stage_index=stage_index,
                    conditioning_classes=self._historical_classes_by_stage[
                        stage_index
                    ],
                    workspaces=workspaces,
                    teacher=teacher,
                    resources=resources,
                )
            )
            features, labels, edges, topology = self._server_synthetic_graph(
                stage_index=stage_index,
                generator_state=generator_state,
                classes=classes,
                spectral_target=spectral_target,
            )
            distilled, distillation_trace, distillation_selection = self._distill(
                student_state=self._shared_state.materialize(),
                teacher=teacher,
                features=features,
                labels=labels,
                edge_index=edges,
                task_class_masks=self._prior_head_masks(stage_index),
                validation_evaluator=validation_evaluator,
            )
            if any(
                not torch.equal(value, teacher.state_dict()[name].detach().cpu())
                for name, value in teacher_before.items()
            ):
                raise RuntimeError("FedFST mutated its frozen previous-task teacher.")
            candidate_shared = FrozenTensorMap(distilled)
            candidate_generator = FrozenTensorMap(generator_state)
            candidate_generator_classes = torch.tensor(classes, dtype=torch.long)
            candidate_spectral_target = float(spectral_target)
            synthetic_bytes = (
                int(features.numel() * features.element_size())
                + int(labels.numel() * labels.element_size())
                + int(edges.numel() * edges.element_size())
            )
            resources.add(synthetic_artifact_bytes=synthetic_bytes)
            stage_record.update(
                {
                    "hhkr_applied": True,
                    "hlst_applied": True,
                    "historical_classes": list(classes),
                    "historical_task_ids": list(self._prior_tasks(stage_index)),
                    "synthetic_nodes": int(labels.shape[0]),
                    "synthetic_directed_edges": int(edges.shape[1]),
                    "synthetic_digest": _tensor_digest(
                        features, labels, edges
                    ),
                    "generator_rounds": self.parameters.generator_rounds,
                    "client_hhkr": {
                        f"client-{client_id}": {
                            "historical_train_nodes": upload.historical_node_count,
                            "spectral_energy": upload.spectral_energy,
                        }
                        for client_id, upload in sorted(uploads.items())
                    },
                    "server_topology": topology,
                    "distillation_epochs": self.parameters.distillation_epochs,
                    "distillation_validation_selection": distillation_selection,
                    "final_hlst_total_loss": distillation_trace[-1][0],
                    "final_hlst_ce_loss": distillation_trace[-1][1],
                    "final_hlst_low_frequency_kl": distillation_trace[-1][2],
                }
            )

        # Construct the post-HLST teacher before current data enters history.
        # History appends still happen only on private workspaces at this point.
        candidate_teacher = FrozenTensorMap(candidate_shared)
        for client_id in sorted(workspaces):
            workspaces[client_id].append_current(method_contexts[client_id])
        stage_record["teacher_snapshot_stage"] = stage_index
        stage_record["history_appended_after_teacher_snapshot"] = True

        # Commit consists only of assignments of fully validated owned state.
        self._shared_state = candidate_shared
        self._previous_task_state = candidate_teacher
        self._global_generator_state = candidate_generator
        self._global_generator_classes = candidate_generator_classes
        self._global_spectral_target = candidate_spectral_target
        for client_id in sorted(workspaces):
            workspaces[client_id].commit()
        self._teacher_snapshot_stages = [
            *self._teacher_snapshot_stages,
            stage_index,
        ]
        self._stage_records = [*self._stage_records, stage_record]
        return AggregationResult(
            shared_state=candidate_shared,
            diagnostics=stage_record,
            resources=resources,
        )

    def client_replay_payload_bytes(self, client: Any) -> int:
        return history_payload_bytes(client.state.strategy_state)

    def validate_restored_clients(
        self, clients: Mapping[int, Any], completed_stages: int
    ) -> None:
        """Cross-check client-private checkpoint history against stream progress."""

        if (
            type(completed_stages) is not int
            or not 0 <= completed_stages <= len(self._task_order)
            or set(clients) != set(self._client_ids)
        ):
            raise ValueError("FedFST restored client progress is invalid.")
        for client_id in self._client_ids:
            client = clients[client_id]
            if client.client_id != client_id:
                raise ValueError("FedFST restored client identity mismatch.")
            validate_client_history_boundary(
                client.state.strategy_state,
                feature_dim=self._feature_dim,
                stage_index=completed_stages,
                expected_task_ids=self._prior_tasks(completed_stages, client_id),
            )

    def diagnostics(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "fidelity": "paper_equations_1_through_13_train_only_adaptation",
            "paper_doi": PAPER_DOI,
            "reference_code_commit": REFERENCE_CODE_COMMIT,
            "parameters_digest": self.parameters.digest,
            "parameters": self.parameters.to_dict(),
            "model_seed": self._model_seed,
            "support_scope": "client_order_stage_union_nc_class_task",
            "incremental_setting": self._incremental_setting,
            "stream_adaptation_digest": self._stream_adaptation_digest,
            "participation_policy": "ordinary_round_trace_plus_fixed_stage_union",
            "task_head_policy": (
                "prior_heads_only_block_diagonal_synthetic_topology"
                if self._incremental_setting == "task"
                else (
                    "paper_full_output_training_seen_class_evaluation"
                    if self.parameters.class_il_output_training_policy
                    == "paper_full"
                    else "benchmark_seen_output_training_and_evaluation"
                )
            ),
            "history_visibility": "prior_strict_local_train_only",
            "client_task_orders": {
                str(k): list(v) for k, v in self._client_task_orders.items()
            },
            "auxiliary_upload_schema": (
                "generator_state_spectral_energy_train_count"
            ),
            "previous_teacher_delivery": "explicit_accounted_ephemeral_downlink",
            "hlst_student_mode": "train_with_isolated_seeded_rng",
            "server_model_state_policy": (
                "all_trainable_shared_client_local_buffers_frozen_during_hlst"
            ),
            "teacher_selection_uses_validation_or_test": "validation_only_aggregate",
            "distillation_early_stop_policy": (
                self.parameters.distillation_early_stop_policy
            ),
            "hlst_kl_direction": "teacher_destination_to_student_source",
            "completed_stages": len(self._stage_records),
            "hhkr_hlst_stages": [
                int(record["stage_index"])
                for record in self._stage_records
                if bool(record["hhkr_applied"])
            ],
            "teacher_snapshot_stages": list(self._teacher_snapshot_stages),
            "global_generator_classes": self._global_generator_classes.tolist(),
            "global_spectral_target": self._global_spectral_target,
            "stage_records": _clone_json(self._stage_records),
        }

    def _validated_checkpoint_records(
        self, value: object
    ) -> list[dict[str, object]]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("FedFST checkpoint stage metadata is malformed.")
        if len(value) > len(self._task_order):
            raise ValueError("FedFST checkpoint has more stages than the stream.")
        cloned = _clone_json(value)
        assert isinstance(cloned, list)
        base_fields = {
            "stage_index",
            "hhkr_applied",
            "hlst_applied",
            "contains_raw_feature_upload",
            "contains_raw_label_upload",
            "contains_raw_edge_upload",
            "teacher_snapshot_stage",
            "history_appended_after_teacher_snapshot",
            "incremental_setting",
            "task_aware",
            "participation_policy",
        }
        auxiliary_fields = {
            "historical_classes",
            "historical_task_ids",
            "synthetic_nodes",
            "synthetic_directed_edges",
            "synthetic_digest",
            "generator_rounds",
            "client_hhkr",
            "server_topology",
            "distillation_epochs",
            "final_hlst_total_loss",
            "final_hlst_ce_loss",
            "final_hlst_low_frequency_kl",
        }
        for stage_index, record in enumerate(cloned):
            expected_auxiliary = stage_index > 0
            expected_fields = (
                base_fields | auxiliary_fields
                if expected_auxiliary
                else base_fields
            )
            if not isinstance(record, dict) or set(record) != expected_fields:
                raise ValueError("FedFST checkpoint stage record is malformed.")
            if (
                type(record["stage_index"]) is not int
                or record["stage_index"] != stage_index
                or type(record["teacher_snapshot_stage"]) is not int
                or record["teacher_snapshot_stage"] != stage_index
                or record["hhkr_applied"] is not expected_auxiliary
                or record["hlst_applied"] is not expected_auxiliary
                or record["contains_raw_feature_upload"] is not False
                or record["contains_raw_label_upload"] is not False
                or record["contains_raw_edge_upload"] is not False
                or record["history_appended_after_teacher_snapshot"] is not True
                or record["incremental_setting"] != self._incremental_setting
                or record["task_aware"] is not (self._incremental_setting == "task")
                or record["participation_policy"] != "fixed_trace_stage_union"
            ):
                raise ValueError("FedFST checkpoint stage record identity is invalid.")
            if not expected_auxiliary:
                continue
            expected_classes = list(
                self._historical_classes_by_stage[stage_index]
            )
            client_hhkr = record.get("client_hhkr")
            if (
                record.get("historical_classes") != expected_classes
                or record.get("historical_task_ids") != list(self._prior_tasks(stage_index))
                or not isinstance(client_hhkr, dict)
                or set(client_hhkr)
                != {f"client-{client_id}" for client_id in self._client_ids}
            ):
                raise ValueError(
                    "FedFST checkpoint HHKR stage metadata is inconsistent."
                )
            for client_record in client_hhkr.values():
                if (
                    not isinstance(client_record, dict)
                    or set(client_record)
                    != {"historical_train_nodes", "spectral_energy"}
                ):
                    raise ValueError("FedFST checkpoint client HHKR record is invalid.")
                node_count = client_record.get("historical_train_nodes")
                energy = client_record.get("spectral_energy")
                if (
                    type(node_count) is not int
                    or node_count <= 0
                    or type(energy) not in {int, float}
                    or not math.isfinite(float(energy))
                    or float(energy) < 0.0
                ):
                    raise ValueError(
                        "FedFST checkpoint client HHKR values are invalid."
                    )
            topology = record.get("server_topology")
            topology_fields = {
                "target_spectral_energy",
                "initial_spectral_energy",
                "final_spectral_energy",
                "feature_spectral_energy",
                "structure_spectral_energy",
                "topology_iterations",
                "topology_converged",
                "removed_edge_types",
                "sampled_feature_indices",
                "task_block_diagonal",
                "cross_task_directed_edges_removed",
                "construction_policy",
                "configured_minimum_directed_edges_per_node",
                "requested_initial_directed_edges_per_node",
                "realized_initial_directed_edges_per_node",
                "initial_target_absolute_error",
                "final_target_absolute_error",
            }
            if not isinstance(topology, dict) or set(topology) != topology_fields:
                raise ValueError("FedFST checkpoint topology record is malformed.")
            spectral_values = (
                topology["target_spectral_energy"],
                topology["initial_spectral_energy"],
                topology["final_spectral_energy"],
                topology["feature_spectral_energy"],
                topology["structure_spectral_energy"],
            )
            removed = topology["removed_edge_types"]
            sampled = topology["sampled_feature_indices"]
            iterations = topology["topology_iterations"]
            task_block_diagonal = topology["task_block_diagonal"]
            cross_task_removed = topology["cross_task_directed_edges_removed"]
            construction_policy = topology["construction_policy"]
            configured_density = topology[
                "configured_minimum_directed_edges_per_node"
            ]
            requested_density = topology[
                "requested_initial_directed_edges_per_node"
            ]
            realized_density = topology[
                "realized_initial_directed_edges_per_node"
            ]
            initial_error = topology["initial_target_absolute_error"]
            final_error = topology["final_target_absolute_error"]
            if (
                any(
                    type(value) not in {int, float}
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in spectral_values
                )
                or type(iterations) is not int
                or not 0 <= iterations <= self.parameters.topology_max_iterations
                or type(topology["topology_converged"]) is not bool
                or not isinstance(removed, list)
                or len(removed) != iterations
                or any(
                    value not in {"homophilic", "heterophilic"}
                    for value in removed
                )
                or not isinstance(sampled, list)
                or not sampled
                or any(
                    type(value) is not int
                    or not 0 <= value < self._feature_dim
                    for value in sampled
                )
                or sampled != sorted(set(sampled))
                or task_block_diagonal is not (
                    self._incremental_setting == "task"
                )
                or type(cross_task_removed) is not int
                or cross_task_removed < 0
                or (not task_block_diagonal and cross_task_removed != 0)
                or construction_policy
                not in {
                    "paper_fixed_global_random",
                    "paper_fixed_task_block_random",
                    "target_scaled_capped_global_random",
                    "target_scaled_capped_task_block_random",
                    "target_scaled_global_random",
                    "target_scaled_task_block_random",
                }
                or type(configured_density) is not int
                or configured_density != self.parameters.generated_edges_per_node
                or type(requested_density) is not int
                or requested_density < configured_density
                or (
                    self.parameters.server_initial_edge_policy == "paper_fixed"
                    and requested_density != configured_density
                )
                or (
                    self.parameters.server_initial_edge_policy == "target_scaled"
                    and not construction_policy.startswith("target_scaled_")
                )
                or (
                    self.parameters.server_initial_edge_policy
                    == "target_scaled_capped"
                    and not construction_policy.startswith(
                        "target_scaled_capped_"
                    )
                )
                or (
                    self.parameters.server_initial_edge_policy
                    == "target_scaled_capped"
                    and requested_density
                    > self.parameters.server_max_edges_per_node
                )
                or (
                    self.parameters.server_initial_edge_policy == "paper_fixed"
                    and not construction_policy.startswith("paper_fixed_")
                )
                or type(realized_density) not in {int, float}
                or not math.isfinite(float(realized_density))
                or float(realized_density) < 0.0
                or type(initial_error) not in {int, float}
                or not math.isfinite(float(initial_error))
                or float(initial_error) < 0.0
                or type(final_error) not in {int, float}
                or not math.isfinite(float(final_error))
                or float(final_error) < 0.0
                or task_block_diagonal
                is not construction_policy.endswith("task_block_random")
            ):
                raise ValueError("FedFST checkpoint topology values are invalid.")
            digest = record.get("synthetic_digest")
            final_losses = (
                record.get("final_hlst_total_loss"),
                record.get("final_hlst_ce_loss"),
                record.get("final_hlst_low_frequency_kl"),
            )
            if (
                type(record.get("synthetic_nodes")) is not int
                or record["synthetic_nodes"] <= 0
                or type(record.get("synthetic_directed_edges")) is not int
                or record["synthetic_directed_edges"] < 0
                or type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or record.get("generator_rounds")
                != self.parameters.generator_rounds
                or record.get("distillation_epochs")
                != self.parameters.distillation_epochs
                or any(
                    type(value) not in {int, float}
                    or not math.isfinite(float(value))
                    for value in final_losses
                )
            ):
                raise ValueError("FedFST checkpoint synthetic/HLST record is invalid.")
        return cloned

    def state_dict(self) -> Mapping[str, object]:
        self._require_initialized()
        return {
            "strategy_version": self.strategy_version,
            "name": self.name,
            "client_ids": self._client_ids,
            "parameters_digest": self.parameters.digest,
            "stream_adaptation_digest": self._stream_adaptation_digest,
            "shared_state": self._shared_state.materialize(),
            "previous_task_state": self._previous_task_state.materialize(),
            "global_generator_state": self._global_generator_state.materialize(),
            "global_generator_classes": self._global_generator_classes.detach()
            .cpu()
            .clone(),
            "global_spectral_target": self._global_spectral_target,
            "stage_records": _clone_json(self._stage_records),
            "teacher_snapshot_stages": list(self._teacher_snapshot_stages),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._require_initialized()
        expected = {
            "strategy_version",
            "name",
            "client_ids",
            "parameters_digest",
            "stream_adaptation_digest",
            "shared_state",
            "previous_task_state",
            "global_generator_state",
            "global_generator_classes",
            "global_spectral_target",
            "stage_records",
            "teacher_snapshot_stages",
        }
        raw_client_ids = state.get("client_ids")
        if (
            set(state) != expected
            or state.get("strategy_version") != self.strategy_version
            or state.get("name") != self.name
            or state.get("parameters_digest") != self.parameters.digest
            or state.get("stream_adaptation_digest") != self._stream_adaptation_digest
            or not isinstance(raw_client_ids, (list, tuple))
            or any(type(value) is not int for value in raw_client_ids)
            or tuple(raw_client_ids) != self._client_ids
        ):
            raise ValueError("FedFST checkpoint identity mismatch.")
        shared = state["shared_state"]
        previous = state["previous_task_state"]
        generator = state["global_generator_state"]
        classes = state["global_generator_classes"]
        records = state["stage_records"]
        snapshots = state["teacher_snapshot_stages"]
        target = state["global_spectral_target"]
        if not torch.is_tensor(classes) or classes.dtype != torch.long or classes.ndim != 1:
            raise ValueError("FedFST checkpoint class vocabulary is malformed.")
        if not isinstance(snapshots, (list, tuple)) or any(
            type(value) is not int for value in snapshots
        ):
            raise TypeError("FedFST checkpoint stage metadata is malformed.")
        if target is not None and (
            type(target) not in {int, float}
            or not math.isfinite(float(target))
            or float(target) < 0.0
        ):
            raise ValueError("FedFST checkpoint spectral target is invalid.")
        candidate_records = self._validated_checkpoint_records(records)
        completed_stages = len(candidate_records)
        if len(snapshots) != completed_stages or list(snapshots) != list(
            range(completed_stages)
        ):
            raise ValueError("FedFST checkpoint stage sequence is invalid.")
        current_reference = self._shared_state.materialize()
        candidate_shared = _validated_tensor_map(
            shared, current_reference, label="shared model"
        )
        if completed_stages:
            candidate_previous = _validated_tensor_map(
                previous, current_reference, label="previous-task teacher"
            )
        else:
            candidate_previous = _validated_tensor_map(
                previous, {}, label="previous-task teacher"
            )

        if completed_stages < 2:
            expected_classes: tuple[int, ...] = ()
            generator_reference: Mapping[str, torch.Tensor] = {}
            if target is not None:
                raise ValueError(
                    "FedFST checkpoint has a spectral target before HHKR."
                )
        else:
            generator_stage = completed_stages - 1
            expected_classes = self._historical_classes_by_stage[generator_stage]
            generator_reference = self._new_generator(
                generator_stage, expected_classes
            ).state_dict()
            if target is None:
                raise ValueError("FedFST checkpoint is missing its spectral target.")
            last_topology = candidate_records[-1]["server_topology"]
            assert isinstance(last_topology, dict)
            if float(last_topology["target_spectral_energy"]) != float(target):
                raise ValueError(
                    "FedFST checkpoint spectral target disagrees with its audit record."
                )
        expected_class_tensor = torch.tensor(expected_classes, dtype=torch.long)
        if not torch.equal(classes.detach().cpu(), expected_class_tensor):
            raise ValueError(
                "FedFST checkpoint generator vocabulary disagrees with the stream."
            )
        candidate_generator = _validated_tensor_map(
            generator, generator_reference, label="global generator"
        )
        candidate_classes = classes.detach().cpu().clone().contiguous()
        candidate_target = None if target is None else float(target)
        candidate_snapshots = list(snapshots)

        # All potentially failing construction/validation is complete. Restore
        # the owned candidates together using assignment-only commit steps.
        self._shared_state = candidate_shared
        self._previous_task_state = candidate_previous
        self._global_generator_state = candidate_generator
        self._global_generator_classes = candidate_classes
        self._global_spectral_target = candidate_target
        self._stage_records = candidate_records
        self._teacher_snapshot_stages = candidate_snapshots
