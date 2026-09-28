from __future__ import annotations

from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_CANDIDATE_K
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_EPOCHS
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_HEADS
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_LAMBDA
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_LR
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_TAU
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_TOP_N
from gecko.algorithms.continual.dslr.records import DSLR_DIAGNOSTIC_VARIANTS
from gecko.algorithms.continual.dslr.records import DSLR_LINK_REDUCTIONS
from gecko.algorithms.continual.dslr.records import DSLR_REPLAY_CEILING_BYTES
from gecko.algorithms.continual.dslr.records import DSLR_REPLAY_FRACTION
from gecko.algorithms.continual.dslr.records import DSLR_SELECTION_MODES
from gecko.algorithms.continual.dslr.records import DSLR_STRUCTURE_MODES
from gecko.algorithms.continual.dslr.records import _CHECKPOINT_VERSION
from gecko.algorithms.continual.dslr.records import _STATE_VERSION

from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_CANDIDATE_K
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_EPOCHS
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_HEADS
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_LAMBDA
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_LR
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_TAU
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_TOP_N
from gecko.algorithms.continual.dslr.records import DSLR_REPLAY_CEILING_BYTES
from gecko.algorithms.continual.dslr.records import DSLR_REPLAY_FRACTION

import hashlib
import math
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from gecko.algorithms.base import ClientContinualAlgorithm
from gecko.algorithms.topology import TopologyOverlay
from gecko.algorithms.topology import edge_index_sha256

def _derived_seed(
    *, base_seed: int, client_id: int, global_task_id: int, purpose: str
) -> int:
    digest = hashlib.sha256()
    for value in (base_seed, client_id, global_task_id, purpose):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big") % (2**63)


def _visible_context_nodes(context: Any) -> torch.Tensor:
    """Return endpoints explicitly visible in the current safe graph context."""
    from gecko.algorithms.continual.dslr.records import _validate_local_edges
    from gecko.algorithms.continual.dslr.records import _validate_local_indices

    features = context.node_features
    edges = _validate_local_edges(
        context.effective_edge_index,
        num_nodes=features.shape[0],
        name="effective_edge_index",
    )
    queries = _validate_local_indices(
        context.train_queries,
        num_nodes=features.shape[0],
        name="train_queries",
    )
    values = set(int(value) for value in queries.tolist())
    values.update(int(value) for value in edges.reshape(-1).tolist())
    if not values:
        raise ValueError("DSLR current strict-local context has no visible nodes.")
    return torch.tensor(sorted(values), dtype=torch.long)


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_mask: torch.Tensor | None,
) -> torch.Tensor:
    if class_mask is not None:
        mask = class_mask.to(device=logits.device, dtype=torch.bool)
        if mask.ndim != 1 or mask.shape[0] != logits.shape[-1]:
            raise ValueError("valid_class_mask does not align with DSLR logits.")
        logits = logits.clone()
        logits[..., ~mask] = -1e12
    return F.cross_entropy(logits, labels.to(logits.device).long())


def _stored_overlay_digest(
    *,
    client_id: int,
    global_task_id: int,
    base_edge_sha256: str,
    added_edge_index: torch.Tensor,
    deleted_edge_index: torch.Tensor,
    undirected: bool,
) -> str:
    """Recompute the TopologyOverlay identity without retaining its base graph."""

    digest = hashlib.sha256()
    for value in (
        str(client_id),
        str(global_task_id),
        "DSLR",
        base_edge_sha256,
        "undirected" if undirected else "directed",
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    digest.update(added_edge_index.detach().cpu().contiguous().numpy().tobytes())
    digest.update(deleted_edge_index.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class DSLRAlgorithm(ClientContinualAlgorithm):
    """Private DSLR replay, structure learner, and per-task overlay state.

    The explicit v2 lifecycle lazily trains a fresh private structure learner
    before the first optimization step of each task after replay exists. The
    resulting task-visible overlay drives equation-12 training and evaluation.
    """

    name = "DSLR"
    method_version = "uefa-dslr-core-v3-explicit-link-reduction"

    def __init__(
        self,
        *,
        beta: float = 0.1,
        radius: float = 0.15,
        structure_lambda: float = DSLR_DEFAULT_STRUCTURE_LAMBDA,
        top_n: int = DSLR_DEFAULT_TOP_N,
        candidate_k: int = DSLR_DEFAULT_CANDIDATE_K,
        tau: float = DSLR_DEFAULT_TAU,
        structure_epochs: int = DSLR_DEFAULT_STRUCTURE_EPOCHS,
        structure_learning_rate: float = DSLR_DEFAULT_STRUCTURE_LR,
        structure_hidden_dim: int = 64,
        structure_heads: int = DSLR_DEFAULT_STRUCTURE_HEADS,
        selection_mode: str = "coverage_diversity",
        structure_mode: str = "full",
        replay_fraction: float = DSLR_REPLAY_FRACTION,
        replay_ceiling_bytes: int = DSLR_REPLAY_CEILING_BYTES,
        undirected: bool = True,
        link_reduction: str = "sum",
        **kwargs: object,
    ) -> None:
        from gecko.algorithms.continual.dslr.replay import DSLRReplayStore
        from gecko.algorithms.continual.dslr.structure import DSLRStructureLearner
        from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_CANDIDATE_K
        from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_EPOCHS
        from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_HEADS
        from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_LAMBDA
        from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_LR
        from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_TAU
        from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_TOP_N
        from gecko.algorithms.continual.dslr.records import DSLR_DIAGNOSTIC_VARIANTS
        from gecko.algorithms.continual.dslr.records import DSLR_LINK_REDUCTIONS
        from gecko.algorithms.continual.dslr.records import DSLR_REPLAY_CEILING_BYTES
        from gecko.algorithms.continual.dslr.records import DSLR_REPLAY_FRACTION
        from gecko.algorithms.continual.dslr.records import DSLR_SELECTION_MODES
        from gecko.algorithms.continual.dslr.records import DSLR_STRUCTURE_MODES
        from gecko.algorithms.continual.dslr.records import _STATE_VERSION
        from gecko.algorithms.continual.dslr.records import _validate_nonnegative_int
        from gecko.algorithms.continual.dslr.records import _validate_positive_float
        from gecko.algorithms.continual.dslr.records import _validate_positive_int
        from gecko.algorithms.continual.dslr.records import _validate_probability
        super().__init__(**kwargs)
        if self.client_id is None:
            raise ValueError("DSLR requires an explicit client_id.")
        self.client_id = _validate_nonnegative_int(self.client_id, name="client_id")
        self.seed = _validate_nonnegative_int(self.seed, name="seed")
        if self.seed >= 2**63:
            raise ValueError("DSLR seed must fit signed 63 bits.")
        self.beta = _validate_probability(beta, name="beta")
        self.radius = _validate_positive_float(radius, name="radius")
        self.structure_lambda = _validate_probability(
            structure_lambda, name="structure_lambda"
        )
        self.top_n = _validate_positive_int(top_n, name="top_n")
        self.candidate_k = _validate_positive_int(candidate_k, name="candidate_k")
        self.tau = _validate_probability(tau, name="tau")
        self.structure_epochs = _validate_positive_int(
            structure_epochs, name="structure_epochs"
        )
        self.structure_learning_rate = _validate_positive_float(
            structure_learning_rate, name="structure_learning_rate"
        )
        self.structure_hidden_dim = _validate_positive_int(
            structure_hidden_dim, name="structure_hidden_dim"
        )
        self.structure_heads = _validate_positive_int(
            structure_heads, name="structure_heads"
        )
        if selection_mode not in DSLR_SELECTION_MODES:
            raise ValueError(f"selection_mode must be one of {DSLR_SELECTION_MODES}.")
        if structure_mode not in DSLR_STRUCTURE_MODES:
            raise ValueError(f"structure_mode must be one of {DSLR_STRUCTURE_MODES}.")
        pair = (selection_mode, structure_mode)
        if (
            pair != ("coverage_diversity", "full")
            and pair not in DSLR_DIAGNOSTIC_VARIANTS
        ):
            raise ValueError("Unknown DSLR diagnostic component combination.")
        self.selection_mode = selection_mode
        self.structure_mode = structure_mode
        self.diagnostic_variant = DSLR_DIAGNOSTIC_VARIANTS.get(pair)
        self.name = "DSLR" if self.diagnostic_variant is None else "DSLRDiagnostic"
        self.replay_fraction = _validate_probability(
            replay_fraction, name="replay_fraction", open_upper=True
        )
        if self.replay_fraction == 0.0:
            raise ValueError("replay_fraction must be positive.")
        self.replay_ceiling_bytes = _validate_positive_int(
            replay_ceiling_bytes, name="replay_ceiling_bytes"
        )
        self.undirected = bool(undirected)
        if link_reduction not in DSLR_LINK_REDUCTIONS:
            raise ValueError(
                f"link_reduction must be one of {DSLR_LINK_REDUCTIONS}."
            )
        self.link_reduction = link_reduction
        if self.link_reduction == "mean":
            if self.diagnostic_variant is not None:
                raise ValueError(
                    "DSLR-Normalized is available only for CD plus full structure."
                )
            self.name = "DSLR-Normalized"
        self._store = DSLRReplayStore(
            client_id=self.client_id,
            ceiling_bytes=self.replay_ceiling_bytes,
        )
        self._structure_learner: DSLRStructureLearner | None = None
        self._last_diagnostics: Dict[str, object] = {
            "requested_replay_nodes": 0,
            "achieved_replay_nodes": 0,
            "replay_payload_bytes": 0,
            "replay_ceiling_bytes": self.replay_ceiling_bytes,
            "selection_metadata_bytes": 0,
            "overlay_payload_bytes": 0,
            "structure_parameter_bytes": 0,
            "structure_initialization_id": None,
            "structure_epochs": 0,
            "structure_visible_nodes": 0,
            "positive_logical_edges": 0,
            "negative_logical_edges_per_epoch": 0,
            "final_link_loss": 0.0,
            "final_link_loss_per_pair": 0.0,
            "final_node_loss": 0.0,
            "final_structure_loss": 0.0,
            "link_loss_pair_count": 0,
            "link_loss_reduction": self.link_reduction,
            "weighted_link_contribution": 0.0,
            "weighted_node_contribution": 0.0,
            "weighted_link_to_node_ratio": 0.0,
            "pre_broadcast_state_sha256": None,
            "post_broadcast_state_sha256": None,
        }
        self.state.update(
            {
                "state_version": _STATE_VERSION,
                "method_hyperparameters": self._private_hyperparameters(),
                "base_edge_sha256": None,
                "num_nodes": None,
                "feature_dim": None,
                "seen_node_ids": torch.empty(0, dtype=torch.long),
                "seen_labels": torch.empty(0, dtype=torch.long),
                "seen_global_task_ids": torch.empty(0, dtype=torch.long),
                "seen_stage_indices": torch.empty(0, dtype=torch.long),
                "task_class_masks": {},
                "replay_payloads": [],
                "consolidated_task_ids": [],
                "prepared_task_ids": [],
                "phi_metadata": {},
                "phi_state": {},
                "phi_initialization_ids": {},
                "overlays": {},
            }
        )
        self._private_state_digest_cache: str | None = None

    def _private_hyperparameters(self) -> Dict[str, object]:
        return {
            "beta": self.beta,
            "radius": self.radius,
            "structure_lambda": self.structure_lambda,
            "top_n": self.top_n,
            "candidate_k": self.candidate_k,
            "tau": self.tau,
            "structure_epochs": self.structure_epochs,
            "structure_learning_rate": self.structure_learning_rate,
            "structure_hidden_dim": self.structure_hidden_dim,
            "structure_heads": self.structure_heads,
            "selection_mode": self.selection_mode,
            "structure_mode": self.structure_mode,
            "replay_fraction": self.replay_fraction,
            "replay_ceiling_bytes": self.replay_ceiling_bytes,
            "undirected": self.undirected,
            "link_reduction": self.link_reduction,
            "seed": self.seed,
            "client_id": self.client_id,
        }

    @staticmethod
    def _validate_context(context: Any) -> None:
        if (
            str(context.problem_type).upper() != "NC"
            or str(context.incremental_setting).lower() not in {"class", "task"}
        ):
            raise ValueError("DSLR core supports NC Class/Task-IL only.")
        mask = context.valid_class_mask
        if mask is None or mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError("DSLR NC requires a one-dimensional class mask.")

    def _register_task_mask(self, context: Any) -> torch.Tensor:
        mask = context.valid_class_mask
        assert mask is not None
        owned = mask.detach().cpu().clone().contiguous()
        task = int(context.global_task_id)
        masks = self.state.get("task_class_masks")
        if not isinstance(masks, dict):
            raise RuntimeError("DSLR task-class masks are malformed.")
        previous = masks.get(task)
        if previous is not None and not torch.equal(previous, owned):
            raise ValueError("DSLR observed two class masks for one global task.")
        if previous is None:
            self._private_state_digest_cache = None
        masks[task] = owned
        return owned

    def _task_mask(self, task: int) -> torch.Tensor:
        masks = self.state.get("task_class_masks")
        value = masks.get(int(task)) if isinstance(masks, dict) else None
        if not torch.is_tensor(value):
            raise RuntimeError("DSLR replay task has no registered class mask.")
        return value

    def _validate_graph_identity(self, context: Any) -> None:
        features = context.node_features
        base = context.base_edge_index
        digest = edge_index_sha256(base, num_nodes=features.shape[0])
        stored_digest = self.state["base_edge_sha256"]
        if stored_digest is None:
            self.state["base_edge_sha256"] = digest
            self.state["num_nodes"] = int(features.shape[0])
            self.state["feature_dim"] = int(features.shape[1])
        elif (
            stored_digest != digest
            or self.state["num_nodes"] != int(features.shape[0])
            or self.state["feature_dim"] != int(features.shape[1])
        ):
            raise ValueError("DSLR strict-local graph identity changed across tasks.")

    def initialize_structure_learner(
        self,
        *,
        input_dim: int,
        num_classes: int,
        global_task_id: int,
        device: torch.device | str = "cpu",
    ) -> DSLRStructureLearner:
        from gecko.algorithms.continual.dslr.structure import DSLRStructureLearner
        from gecko.algorithms.continual.dslr.records import _validate_nonnegative_int
        task = _validate_nonnegative_int(global_task_id, name="global_task_id")
        seed = _derived_seed(
            base_seed=self.seed,
            client_id=self.client_id,
            global_task_id=task,
            purpose="dslr-phi-initialization",
        )
        learner = DSLRStructureLearner(
            input_dim=input_dim,
            hidden_dim=self.structure_hidden_dim,
            num_classes=num_classes,
            initialization_seed=seed,
            heads=self.structure_heads,
        ).to(device)
        self._structure_learner = learner
        initialization_id = learner.state_sha256()
        self.state["phi_metadata"] = {
            "input_dim": int(input_dim),
            "hidden_dim": self.structure_hidden_dim,
            "heads": self.structure_heads,
            "num_classes": int(num_classes),
            "global_task_id": task,
            "initialization_seed": seed,
        }
        self.state["phi_state"] = {
            name: value.detach().cpu().clone()
            for name, value in learner.state_dict().items()
        }
        self.state["phi_initialization_ids"][task] = initialization_id
        self._last_diagnostics["structure_initialization_id"] = initialization_id
        self._last_diagnostics["structure_parameter_bytes"] = sum(
            value.numel() * value.element_size()
            for value in learner.state_dict().values()
        )
        return learner

    def _seen_mapping(self) -> Dict[int, tuple[int, int, int]]:
        tensors = (
            self.state["seen_node_ids"],
            self.state["seen_labels"],
            self.state["seen_global_task_ids"],
            self.state["seen_stage_indices"],
        )
        return {
            int(node): (int(label), int(task), int(stage))
            for node, label, task, stage in zip(
                *(tensor.tolist() for tensor in tensors)
            )
        }

    def _consolidate_impl(self, model: nn.Module, context: Any) -> None:
        """Select the cumulative 5% replay target with equations 3--5 and 11."""
        from gecko.algorithms.continual.dslr.records import DSLRReplaySnapshot
        from gecko.algorithms.continual.dslr.selection import allocate_class_quotas
        from gecko.algorithms.continual.dslr.selection import greedy_coverage_selection
        from gecko.algorithms.continual.dslr.selection import mean_feature_selection
        from gecko.algorithms.continual.dslr.selection import select_prior_candidates

        self._validate_context(context)
        self._register_task_mask(context)
        if int(context.client_id) != self.client_id:
            raise ValueError("DSLR context belongs to another client.")
        task = int(context.global_task_id)
        if task in self.state["consolidated_task_ids"]:
            raise ValueError(f"DSLR task {task} has already been consolidated.")
        self._validate_graph_identity(context)
        queries = context.train_queries
        labels = context.train_labels
        if queries.dtype != torch.long or queries.ndim != 1:
            raise ValueError("DSLR NC train queries must be one-dimensional int64.")
        if (
            labels.dtype != torch.long
            or labels.ndim != 1
            or labels.shape[0] != queries.shape[0]
        ):
            raise ValueError("DSLR NC labels must be one-dimensional int64.")
        if labels.numel() and int(labels.min()) < 0:
            raise ValueError("DSLR cannot consolidate unknown training labels.")
        seen = self._seen_mapping()
        for node, label in zip(queries.tolist(), labels.tolist()):
            node_id = int(node)
            class_id = int(label)
            if node_id in seen and seen[node_id][0] != class_id:
                raise ValueError(
                    "DSLR received conflicting labels for a local train node."
                )
            seen.setdefault(node_id, (class_id, task, int(context.stage_index)))
        ordered_nodes = sorted(seen)
        self.state["seen_node_ids"] = torch.tensor(ordered_nodes, dtype=torch.long)
        self.state["seen_labels"] = torch.tensor(
            [seen[node][0] for node in ordered_nodes], dtype=torch.long
        )
        self.state["seen_global_task_ids"] = torch.tensor(
            [seen[node][1] for node in ordered_nodes], dtype=torch.long
        )
        self.state["seen_stage_indices"] = torch.tensor(
            [seen[node][2] for node in ordered_nodes], dtype=torch.long
        )
        class_counts: Dict[int, int] = {}
        for node in ordered_nodes:
            class_counts[seen[node][0]] = class_counts.get(seen[node][0], 0) + 1
        # ``replay_fraction`` specifies the requested cumulative budget, but
        # Class-IL replay has a stricter coverage invariant: every class seen
        # by this client must retain one representative.  Small Dirichlet
        # shards can otherwise make the fractional budget smaller than the
        # number of seen classes and turn a valid stream into a runtime error.
        requested = max(
            int(math.ceil(self.replay_fraction * len(ordered_nodes))),
            len(class_counts),
        )
        quotas = allocate_class_quotas(class_counts, requested_count=requested)
        embeddings = context.encode_nodes(model).detach().cpu()
        features = context.node_features.detach().cpu()
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != features.shape[0]
            or not bool(torch.isfinite(embeddings).all())
        ):
            raise RuntimeError(
                "DSLR downstream encoder returned invalid node embeddings."
            )
        all_labels = torch.full((features.shape[0],), -1, dtype=torch.long)
        for node in ordered_nodes:
            all_labels[node] = seen[node][0]
        candidate_indices = torch.tensor(ordered_nodes, dtype=torch.long)
        if self.selection_mode == "coverage_diversity":
            selected = greedy_coverage_selection(
                embeddings,
                all_labels,
                candidate_indices,
                class_quotas=quotas,
                radius=self.radius,
            )
        else:
            selected = mean_feature_selection(
                features,
                all_labels,
                candidate_indices,
                class_quotas=quotas,
            )
        available = _visible_context_nodes(context)
        if not set(ordered_nodes) <= set(int(value) for value in available.tolist()):
            raise ValueError(
                "DSLR cumulative labeled nodes are not all visible in the current context."
            )
        snapshots = []
        for node in selected:
            label, provenance_task, provenance_stage = seen[node]
            snapshots.append(
                DSLRReplaySnapshot(
                    client_id=self.client_id,
                    global_task_id=provenance_task,
                    stage_index=provenance_stage,
                    class_id=label,
                    source_local_index=node,
                    feature=features[node],
                    embedding=embeddings[node],
                    candidate_local_indices=(
                        torch.empty(0, dtype=torch.long)
                        if self.structure_mode == "none"
                        else select_prior_candidates(
                            embeddings,
                            replay_index=node,
                            available_indices=available,
                            candidate_k=self.candidate_k,
                        )
                    ),
                )
            )
        report = self._store.replace(
            snapshots,
            seen_classes=class_counts,
            requested_count=requested,
        )
        self.state["replay_payloads"] = list(self._store.payloads())
        self.state["consolidated_task_ids"].append(task)
        metadata_tensors = (
            self.state["seen_node_ids"],
            self.state["seen_labels"],
            self.state["seen_global_task_ids"],
            self.state["seen_stage_indices"],
        )
        self._last_diagnostics.update(report)
        self._last_diagnostics["selection_metadata_bytes"] = sum(
            tensor.numel() * tensor.element_size() for tensor in metadata_tensors
        )

    def consolidate(self, model: nn.Module, context: Any) -> None:
        """Atomically consolidate one task private replay state."""

        previous_state = super().save_method_state()
        previous_payloads = self._store.payloads()
        previous_diagnostics = dict(self._last_diagnostics)
        previous_digest = self._private_state_digest_cache
        self._private_state_digest_cache = None
        try:
            self._consolidate_impl(model, context)
        except Exception:
            super().load_method_state(previous_state)
            self._store.load_payloads(previous_payloads)
            self._last_diagnostics = previous_diagnostics
            self._private_state_digest_cache = previous_digest
            raise

    def replay_samples(self, context: Any) -> Tuple[DSLRReplaySnapshot, ...]:
        from gecko.algorithms.continual.dslr.records import DSLRReplaySnapshot
        del context
        return self._store.snapshots()

    def replay_payload_bytes(self) -> int:
        return self._store.used_bytes

    def topology_overlay_payload_bytes(self) -> int:
        return sum(
            int(values[name].numel() * values[name].element_size())
            for values in self.state["overlays"].values()
            for name in ("added_edge_index", "deleted_edge_index")
        )

    def private_state_checksum(self) -> str:
        """Return a canonical digest of broadcast-immutable private state."""
        from gecko.algorithms.continual.dslr.records import _update_digest

        if self._private_state_digest_cache is not None:
            return self._private_state_digest_cache
        digest = hashlib.sha256()
        _update_digest(digest, self.state)
        self._private_state_digest_cache = digest.hexdigest()
        return self._private_state_digest_cache

    def on_broadcast(self, context: Any, payload: Any) -> None:
        """Record that a model broadcast left every DSLR private field unchanged."""

        del context, payload
        checksum = self.private_state_checksum()
        self._last_diagnostics["pre_broadcast_state_sha256"] = checksum
        self._last_diagnostics["post_broadcast_state_sha256"] = checksum

    def _prepare_structure_impl(
        self, model: nn.Module, context: Any
    ) -> TopologyOverlay:
        """Train fresh phi_t and derive a task-visible method overlay."""
        from gecko.algorithms.continual.dslr.structure import build_dslr_overlay
        from gecko.algorithms.continual.dslr.structure import fit_structure_learner

        model_parameter = next(model.parameters(), None)
        if model_parameter is None:
            raise ValueError(
                "DSLR requires a parameterized client model to select its device."
            )
        learner_device = model_parameter.device
        # The paper's phi_t is separate from theta_t, but its temporary
        # strict-local tensors must execute on the client model device.
        self._validate_context(context)
        self._register_task_mask(context)
        if int(context.client_id) != self.client_id:
            raise ValueError("DSLR context belongs to another client.")
        task = int(context.global_task_id)
        if self.structure_mode == "none":
            raise ValueError("This DSLR diagnostic disables structure learning.")
        if task in self.state["prepared_task_ids"]:
            raise ValueError(f"DSLR task {task} structure has already been prepared.")
        self._validate_graph_identity(context)
        snapshots = self._store.snapshots()
        if not snapshots:
            raise ValueError("DSLR structure learning requires prior replay snapshots.")
        current_indices = context.train_queries
        current_labels = context.train_labels
        if current_indices.numel() == 0 or current_labels.numel() == 0:
            raise ValueError("DSLR structure learning requires current train labels.")
        replay_indices = torch.tensor(
            [snapshot.source_local_index for snapshot in snapshots], dtype=torch.long
        )
        replay_labels = torch.tensor(
            [snapshot.class_id for snapshot in snapshots], dtype=torch.long
        )
        replay_task_ids = torch.tensor(
            [snapshot.global_task_id for snapshot in snapshots], dtype=torch.long
        )
        if set(current_indices.tolist()) & set(replay_indices.tolist()):
            raise ValueError("DSLR current and replay training nodes must be disjoint.")
        features = context.node_features
        edge_index = context.effective_edge_index
        visible = _visible_context_nodes(context)
        visible_set = set(int(value) for value in visible.tolist())
        supervised = set(int(value) for value in current_indices.tolist()) | set(
            int(value) for value in replay_indices.tolist()
        )
        if not supervised <= visible_set:
            raise ValueError("DSLR supervision contains a future or hidden local node.")
        for snapshot in snapshots:
            if snapshot.source_local_index >= features.shape[0] or not torch.equal(
                snapshot.feature, features[snapshot.source_local_index].detach().cpu()
            ):
                raise ValueError(
                    "DSLR replay feature no longer matches the strict-local graph."
                )
            if snapshot.candidate_local_indices.numel() > self.candidate_k:
                raise ValueError("DSLR snapshot exceeds the configured Eq.-(11) K.")
            if any(
                int(value) not in visible_set
                for value in snapshot.candidate_local_indices.tolist()
            ):
                raise ValueError(
                    "DSLR replay candidate is not visible in the current task."
                )
        class_mask = context.valid_class_mask
        assert class_mask is not None
        learner = self.initialize_structure_learner(
            input_dim=features.shape[1],
            num_classes=int(class_mask.numel()),
            global_task_id=task,
            device=learner_device,
        )
        report = fit_structure_learner(
            learner,
            node_features=features,
            edge_index=edge_index,
            allowed_nodes=visible,
            current_indices=current_indices,
            current_labels=current_labels,
            replay_indices=replay_indices,
            replay_labels=replay_labels,
            current_class_mask=context.valid_class_mask,
            replay_task_ids=(
                replay_task_ids
                if self.incremental_setting == "task"
                else torch.zeros_like(replay_task_ids)
            ),
            task_class_masks=(
                {
                    task_id: self._task_mask(task_id)
                    for task_id in sorted(set(replay_task_ids.tolist()))
                }
                if self.incremental_setting == "task"
                else {0: class_mask}
            ),
            beta=self.beta,
            structure_lambda=(
                1.0
                if self.structure_mode == "link_only"
                else 0.0
                if self.structure_mode == "node_only"
                else self.structure_lambda
            ),
            epochs=self.structure_epochs,
            learning_rate=self.structure_learning_rate,
            link_reduction=self.link_reduction,
            rng_seed=_derived_seed(
                base_seed=self.seed,
                client_id=self.client_id,
                global_task_id=task,
                purpose="dslr-negative-sampling",
            ),
        )
        with torch.no_grad():
            structure_embeddings = (
                learner.encode(
                    features.to(next(learner.parameters()).device),
                    edge_index,
                )
                .detach()
                .cpu()
            )
        overlay = build_dslr_overlay(
            client_id=self.client_id,
            global_task_id=task,
            base_edge_index=edge_index,
            allowed_nodes=visible,
            structure_embeddings=structure_embeddings,
            replay_snapshots=snapshots,
            top_n=self.top_n,
            tau=self.tau,
            undirected=self.undirected,
        )
        self.state["phi_state"] = {
            name: value.detach().cpu().clone()
            for name, value in learner.state_dict().items()
        }
        self.state["prepared_task_ids"].append(task)
        self.state["overlays"][task] = {
            "client_id": overlay.client_id,
            "global_task_id": overlay.global_task_id,
            "base_edge_sha256": overlay.base_edge_sha256,
            "overlay_id": overlay.overlay_id,
            "undirected": overlay.undirected,
            "added_edge_index": overlay.added_edge_index,
            "deleted_edge_index": overlay.deleted_edge_index,
        }
        self._last_diagnostics.update(report)
        self._last_diagnostics["overlay_payload_bytes"] = overlay.payload_bytes
        return overlay

    def prepare_structure(self, model: nn.Module, context: Any) -> TopologyOverlay:
        """Atomically train phi_t and install a task topology overlay."""

        previous_state = super().save_method_state()
        previous_payloads = self._store.payloads()
        previous_diagnostics = dict(self._last_diagnostics)
        previous_learner = self._structure_learner
        previous_digest = self._private_state_digest_cache
        self._private_state_digest_cache = None
        try:
            return self._prepare_structure_impl(model, context)
        except Exception:
            super().load_method_state(previous_state)
            self._store.load_payloads(previous_payloads)
            self._last_diagnostics = previous_diagnostics
            self._structure_learner = previous_learner
            self._private_state_digest_cache = previous_digest
            raise

    def evaluation_topology(self, context: Any) -> TopologyOverlay | None:
        self._validate_context(context)
        if int(context.client_id) != self.client_id:
            raise ValueError("DSLR context belongs to another client.")
        task = int(context.global_task_id)
        values = self.state["overlays"].get(task)
        if values is None:
            return None
        overlay = TopologyOverlay(
            client_id=values["client_id"],
            global_task_id=values["global_task_id"],
            method_name="DSLR",
            num_nodes=context.node_features.shape[0],
            base_edge_index=context.effective_edge_index,
            added_edge_index=values["added_edge_index"],
            deleted_edge_index=values["deleted_edge_index"],
            undirected=values["undirected"],
        )
        if (
            overlay.base_edge_sha256 != values["base_edge_sha256"]
            or overlay.overlay_id != values["overlay_id"]
        ):
            raise ValueError("DSLR checkpoint overlay checksum mismatch.")
        return overlay

    def augment_loss(
        self,
        model: nn.Module,
        context: Any,
        logits: torch.Tensor,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Activate equations 6--12 on the explicit stateful training path."""

        del logits
        self._validate_context(context)
        if not torch.is_tensor(base_loss) or base_loss.ndim != 0:
            raise ValueError("DSLR base_loss must be a scalar tensor.")
        snapshots = self._store.snapshots()
        if not snapshots:
            return base_loss
        if self.structure_mode == "none":
            return self.downstream_loss(model, context)
        task = int(context.global_task_id)
        if task not in self.state["prepared_task_ids"]:
            self.prepare_structure(model, context)
        return self.downstream_loss(model, context)

    def downstream_loss(self, model: nn.Module, context: Any) -> torch.Tensor:
        """Evaluate downstream node classification on the refined graph (Eq. 12)."""
        from gecko.algorithms.continual.dslr.structure import downstream_classification_loss

        self._validate_context(context)
        overlay = self.evaluation_topology(context)
        edges = (
            context.effective_edge_index
            if overlay is None
            else overlay.apply(context.effective_edge_index)
        )
        current_logits = context.forward_queries(
            model, context.train_queries, edge_index=edges
        )
        current = _classification_loss(
            current_logits, context.train_labels, context.valid_class_mask
        )
        snapshots = self._store.snapshots()
        if not snapshots:
            return current
        replay_indices = torch.tensor(
            [snapshot.source_local_index for snapshot in snapshots], dtype=torch.long
        )
        replay_labels = torch.tensor(
            [snapshot.class_id for snapshot in snapshots], dtype=torch.long
        )
        replay_logits = context.forward_queries(model, replay_indices, edge_index=edges)
        replay_terms = []
        for replay_task in sorted({snapshot.global_task_id for snapshot in snapshots}):
            positions = torch.tensor(
                [
                    index
                    for index, snapshot in enumerate(snapshots)
                    if snapshot.global_task_id == replay_task
                ],
                dtype=torch.long,
                device=replay_logits.device,
            )
            replay_terms.append(
                (
                    _classification_loss(
                        replay_logits[positions],
                        replay_labels[positions.cpu()],
                        (
                            self._task_mask(replay_task)
                            if self.incremental_setting == "task"
                            else context.valid_class_mask
                        ),
                    ),
                    int(positions.numel()),
                )
            )
        replay = sum(loss * count for loss, count in replay_terms) / sum(
            count for _, count in replay_terms
        )
        return downstream_classification_loss(current, replay, beta=self.beta)

    def diagnostics(self) -> Dict[str, object]:
        snapshots = self._store.snapshots()
        return {
            "method": self.name,
            "state_sha256": self.private_state_checksum(),
            "fidelity_status": (
                "mechanism_adaptation"
                if self.diagnostic_variant is None
                else "diagnostic_only"
            ),
            "benchmark_eligible": False,
            "selection_mode": self.selection_mode,
            "structure_mode": self.structure_mode,
            "diagnostic_variant": self.diagnostic_variant,
            "topology_overlay_payload_bytes": self.topology_overlay_payload_bytes(),
            "seen_classes": tuple(
                sorted(set(int(value) for value in self.state["seen_labels"].tolist()))
            ),
            "consolidated_task_ids": tuple(self.state["consolidated_task_ids"]),
            "prepared_task_ids": tuple(self.state["prepared_task_ids"]),
            "replay_snapshot_ids": tuple(
                snapshot.snapshot_id for snapshot in snapshots
            ),
            **self._last_diagnostics,
        }

    def save_method_state(self) -> Dict[str, object]:
        from gecko.algorithms.continual.dslr.records import _CHECKPOINT_VERSION
        return {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "private_state": super().save_method_state(),
            "diagnostics": dict(self._last_diagnostics),
        }

    def _validate_loaded_state(self) -> None:
        from gecko.algorithms.continual.dslr.structure import DSLRStructureLearner
        from gecko.algorithms.continual.dslr.records import _STATE_VERSION
        from gecko.algorithms.continual.dslr.records import _validate_local_edges
        required = {
            "state_version",
            "method_hyperparameters",
            "base_edge_sha256",
            "num_nodes",
            "feature_dim",
            "seen_node_ids",
            "seen_labels",
            "seen_global_task_ids",
            "seen_stage_indices",
            "task_class_masks",
            "replay_payloads",
            "consolidated_task_ids",
            "prepared_task_ids",
            "phi_metadata",
            "phi_state",
            "phi_initialization_ids",
            "overlays",
        }
        if set(self.state) != required:
            raise ValueError("DSLR checkpoint fields do not match the state schema.")
        if self.state["state_version"] != _STATE_VERSION:
            raise ValueError("Unsupported DSLR private-state version.")
        if self.state["method_hyperparameters"] != self._private_hyperparameters():
            raise ValueError(
                "DSLR checkpoint hyperparameters do not match this method."
            )
        tensors = [
            self.state[name]
            for name in (
                "seen_node_ids",
                "seen_labels",
                "seen_global_task_ids",
                "seen_stage_indices",
            )
        ]
        if (
            any(
                not torch.is_tensor(value)
                or value.dtype != torch.long
                or value.ndim != 1
                or value.device.type != "cpu"
                for value in tensors
            )
            or len({value.shape for value in tensors}) != 1
        ):
            raise ValueError("DSLR seen-node checkpoint tensors are malformed.")
        nodes, labels, tasks, stages = tensors
        if nodes.numel() and (
            int(nodes.min()) < 0
            or nodes.tolist() != sorted(set(int(value) for value in nodes.tolist()))
            or int(labels.min()) < 0
            or int(tasks.min()) < 0
            or int(stages.min()) < 0
        ):
            raise ValueError("DSLR seen-node checkpoint metadata is invalid.")
        num_nodes = self.state["num_nodes"]
        feature_dim = self.state["feature_dim"]
        digest = self.state["base_edge_sha256"]
        if nodes.numel():
            if (
                isinstance(num_nodes, bool)
                or not isinstance(num_nodes, int)
                or num_nodes <= int(nodes.max())
                or isinstance(feature_dim, bool)
                or not isinstance(feature_dim, int)
                or feature_dim <= 0
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise ValueError("DSLR graph identity checkpoint fields are invalid.")
        elif any(value is not None for value in (num_nodes, feature_dim, digest)):
            raise ValueError("Empty DSLR state must not carry graph identity.")
        for name in ("consolidated_task_ids", "prepared_task_ids"):
            values = self.state[name]
            if (
                not isinstance(values, list)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in values
                )
                or len(values) != len(set(values))
            ):
                raise ValueError(f"DSLR {name} checkpoint field is invalid.")
        masks = self.state["task_class_masks"]
        if not isinstance(masks, dict) or not set(self.state["consolidated_task_ids"]) <= set(masks):
            raise ValueError("DSLR task-class masks omit a consolidated task.")
        for task_id, mask in masks.items():
            if (
                isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id < 0
                or not torch.is_tensor(mask)
                or mask.dtype != torch.bool
                or mask.ndim != 1
                or mask.device.type != "cpu"
                or not bool(mask.any())
            ):
                raise ValueError("DSLR task-class mask is malformed.")
        payloads = self.state["replay_payloads"]
        if not isinstance(payloads, list):
            raise TypeError("DSLR replay_payloads must be a list.")
        self._store.load_payloads(payloads)
        seen = self._seen_mapping()
        snapshots = self._store.snapshots()
        for snapshot in snapshots:
            expected = seen.get(snapshot.source_local_index)
            if (
                expected is None
                or expected
                != (
                    snapshot.class_id,
                    snapshot.global_task_id,
                    snapshot.stage_index,
                )
                or snapshot.feature.numel() != feature_dim
                or snapshot.source_local_index >= num_nodes
                or snapshot.candidate_local_indices.numel() > self.candidate_k
                or (
                    snapshot.candidate_local_indices.numel()
                    and int(snapshot.candidate_local_indices.max()) >= num_nodes
                )
            ):
                raise ValueError(
                    "DSLR replay snapshot disagrees with checkpoint metadata."
                )
        if self.state["consolidated_task_ids"]:
            represented = {snapshot.class_id for snapshot in snapshots}
            if represented != set(int(value) for value in labels.tolist()):
                raise ValueError("DSLR checkpoint replay omits a seen class.")
        metadata = self.state["phi_metadata"]
        phi_state = self.state["phi_state"]
        if bool(metadata) != bool(phi_state):
            raise ValueError("DSLR phi metadata and tensor state must coexist.")
        self._structure_learner = None
        if metadata:
            if not self.state["prepared_task_ids"]:
                raise ValueError("DSLR phi state exists without a prepared task.")
            if not isinstance(metadata, Mapping) or set(metadata) != {
                "input_dim",
                "hidden_dim",
                "heads",
                "num_classes",
                "global_task_id",
                "initialization_seed",
            }:
                raise ValueError("DSLR phi metadata is malformed.")
            if (
                metadata["heads"] != self.structure_heads
                or metadata["global_task_id"] != self.state["prepared_task_ids"][-1]
            ):
                raise ValueError(
                    "DSLR phi metadata disagrees with prepared-task state."
                )
            learner = DSLRStructureLearner(
                input_dim=metadata["input_dim"],
                hidden_dim=metadata["hidden_dim"],
                heads=metadata["heads"],
                num_classes=metadata["num_classes"],
                initialization_seed=metadata["initialization_seed"],
            )
            if not isinstance(phi_state, Mapping):
                raise TypeError("DSLR phi_state must be a mapping.")
            learner.load_state_dict(phi_state, strict=True)
            self._structure_learner = learner
        ids = self.state["phi_initialization_ids"]
        if not isinstance(ids, Mapping) or any(
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id < 0
            or not isinstance(value, str)
            or len(value) != 64
            for task_id, value in ids.items()
        ):
            raise ValueError("DSLR phi initialization IDs are malformed.")
        overlays = self.state["overlays"]
        if not isinstance(overlays, Mapping):
            raise TypeError("DSLR overlays must be a mapping.")
        if set(overlays) != set(self.state["prepared_task_ids"]) or set(ids) != set(
            self.state["prepared_task_ids"]
        ):
            raise ValueError("DSLR overlay/phi task identities are inconsistent.")
        overlay_fields = {
            "client_id",
            "global_task_id",
            "base_edge_sha256",
            "overlay_id",
            "undirected",
            "added_edge_index",
            "deleted_edge_index",
        }
        for task_id, overlay in overlays.items():
            if not isinstance(overlay, Mapping) or set(overlay) != overlay_fields:
                raise ValueError("DSLR overlay checkpoint fields are malformed.")
            if (
                overlay["client_id"] != self.client_id
                or overlay["global_task_id"] != task_id
                or not isinstance(overlay["base_edge_sha256"], str)
                or len(overlay["base_edge_sha256"]) != 64
                or not isinstance(overlay["overlay_id"], str)
                or len(overlay["overlay_id"]) != 64
                or not isinstance(overlay["undirected"], bool)
            ):
                raise ValueError("DSLR overlay checkpoint identity is invalid.")
            added = _validate_local_edges(
                overlay["added_edge_index"],
                num_nodes=num_nodes,
                name="added overlay",
            )
            deleted = _validate_local_edges(
                overlay["deleted_edge_index"],
                num_nodes=num_nodes,
                name="deleted overlay",
            )
            if set(map(tuple, added.t().tolist())) & set(
                map(tuple, deleted.t().tolist())
            ):
                raise ValueError("DSLR overlay adds and deletes the same arc.")
            expected_overlay_id = _stored_overlay_digest(
                client_id=overlay["client_id"],
                global_task_id=overlay["global_task_id"],
                base_edge_sha256=overlay["base_edge_sha256"],
                added_edge_index=added,
                deleted_edge_index=deleted,
                undirected=overlay["undirected"],
            )
            if expected_overlay_id != overlay["overlay_id"]:
                raise ValueError("DSLR overlay checkpoint checksum is invalid.")

    @staticmethod
    def _validate_diagnostics(values: Mapping[str, object]) -> None:
        from gecko.algorithms.continual.dslr.records import DSLR_LINK_REDUCTIONS
        required = {
            "requested_replay_nodes",
            "achieved_replay_nodes",
            "replay_payload_bytes",
            "replay_ceiling_bytes",
            "selection_metadata_bytes",
            "overlay_payload_bytes",
            "structure_parameter_bytes",
            "structure_initialization_id",
            "structure_epochs",
            "structure_visible_nodes",
            "positive_logical_edges",
            "negative_logical_edges_per_epoch",
            "link_loss_pair_count",
            "link_loss_reduction",
            "final_link_loss",
            "final_link_loss_per_pair",
            "final_node_loss",
            "final_structure_loss",
            "weighted_link_contribution",
            "weighted_node_contribution",
            "weighted_link_to_node_ratio",
            "pre_broadcast_state_sha256",
            "post_broadcast_state_sha256",
        }
        if set(values) != required:
            raise ValueError("DSLR checkpoint diagnostics do not match the schema.")
        for name, value in values.items():
            if name == "structure_initialization_id":
                if value is not None and (
                    not isinstance(value, str) or len(value) != 64
                ):
                    raise ValueError("DSLR structure initialization ID is invalid.")
            elif name in {
                "pre_broadcast_state_sha256",
                "post_broadcast_state_sha256",
            }:
                if value is not None and (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(f"DSLR diagnostic checksum {name!r} is invalid.")
            elif name == "link_loss_reduction":
                if value not in DSLR_LINK_REDUCTIONS:
                    raise ValueError("DSLR link-loss reduction is invalid.")
            elif name.startswith("final_"):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(f"DSLR diagnostic {name} is invalid.")
            elif name.startswith("weighted_"):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(f"DSLR diagnostic {name} is invalid.")
            elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"DSLR diagnostic {name} is invalid.")

    def load_method_state(self, state: Mapping[str, object]) -> None:
        from gecko.algorithms.continual.dslr.records import _CHECKPOINT_VERSION
        previous_state = super().save_method_state()
        previous_diagnostics = dict(self._last_diagnostics)
        previous_store_payloads = self._store.payloads()
        previous_learner = self._structure_learner
        previous_digest = self._private_state_digest_cache
        try:
            if not isinstance(state, Mapping) or set(state) != {
                "checkpoint_version",
                "private_state",
                "diagnostics",
            }:
                raise ValueError(
                    "DSLR method checkpoint fields do not match the schema."
                )
            if state["checkpoint_version"] != _CHECKPOINT_VERSION:
                raise ValueError("Unsupported DSLR method checkpoint version.")
            private = state["private_state"]
            diagnostics = state["diagnostics"]
            if not isinstance(private, Mapping) or not isinstance(diagnostics, Mapping):
                raise TypeError(
                    "DSLR checkpoint state and diagnostics must be mappings."
                )
            self._validate_diagnostics(diagnostics)
            super().load_method_state(private)
            self._validate_loaded_state()
            if (
                diagnostics["replay_payload_bytes"] != self._store.used_bytes
                or diagnostics["replay_ceiling_bytes"] != self.replay_ceiling_bytes
                or diagnostics["achieved_replay_nodes"] != len(self._store.snapshots())
            ):
                raise ValueError(
                    "DSLR checkpoint diagnostics disagree with replay state."
                )
            self._last_diagnostics = dict(diagnostics)
            self._private_state_digest_cache = None
        except Exception:
            super().load_method_state(previous_state)
            self._store.load_payloads(previous_store_payloads)
            self._structure_learner = previous_learner
            self._last_diagnostics = previous_diagnostics
            self._private_state_digest_cache = previous_digest
            raise


__all__ = [
    "DSLRAlgorithm",
    "DSLRReplaySnapshot",
    "DSLRReplayStore",
    "DSLRStructureLearner",
    "DSLR_DEFAULT_CANDIDATE_K",
    "DSLR_DEFAULT_STRUCTURE_EPOCHS",
    "DSLR_DEFAULT_STRUCTURE_LAMBDA",
    "DSLR_DEFAULT_STRUCTURE_LR",
    "DSLR_DEFAULT_STRUCTURE_HEADS",
    "DSLR_DEFAULT_TAU",
    "DSLR_DEFAULT_TOP_N",
    "DSLR_DIAGNOSTIC_VARIANTS",
    "DSLR_REPLAY_CEILING_BYTES",
    "DSLR_REPLAY_FRACTION",
    "DSLR_SELECTION_MODES",
    "DSLR_STRUCTURE_MODES",
    "allocate_class_quotas",
    "build_dslr_overlay",
    "cosine_link_scores",
    "coverage_sets",
    "deserialize_dslr_snapshot",
    "downstream_classification_loss",
    "fit_structure_learner",
    "greedy_coverage_selection",
    "link_prediction_loss",
    "mean_feature_selection",
    "node_supervision_loss",
    "sample_strict_local_negative_edges",
    "select_prior_candidates",
    "serialize_dslr_snapshot",
    "structure_learning_loss",
]




_RELOCATED_EXPORTS = {'DSLRReplaySnapshot': ('gecko.algorithms.continual.dslr.records', 'DSLRReplaySnapshot'), 'DSLRReplayStore': ('gecko.algorithms.continual.dslr.replay', 'DSLRReplayStore'), 'DSLRStructureLearner': ('gecko.algorithms.continual.dslr.structure', 'DSLRStructureLearner'), 'DSLR_DEFAULT_CANDIDATE_K': ('gecko.algorithms.continual.dslr.records', 'DSLR_DEFAULT_CANDIDATE_K'), 'DSLR_DEFAULT_STRUCTURE_EPOCHS': ('gecko.algorithms.continual.dslr.records', 'DSLR_DEFAULT_STRUCTURE_EPOCHS'), 'DSLR_DEFAULT_STRUCTURE_HEADS': ('gecko.algorithms.continual.dslr.records', 'DSLR_DEFAULT_STRUCTURE_HEADS'), 'DSLR_DEFAULT_STRUCTURE_LAMBDA': ('gecko.algorithms.continual.dslr.records', 'DSLR_DEFAULT_STRUCTURE_LAMBDA'), 'DSLR_DEFAULT_STRUCTURE_LR': ('gecko.algorithms.continual.dslr.records', 'DSLR_DEFAULT_STRUCTURE_LR'), 'DSLR_DEFAULT_TAU': ('gecko.algorithms.continual.dslr.records', 'DSLR_DEFAULT_TAU'), 'DSLR_DEFAULT_TOP_N': ('gecko.algorithms.continual.dslr.records', 'DSLR_DEFAULT_TOP_N'), 'DSLR_DIAGNOSTIC_VARIANTS': ('gecko.algorithms.continual.dslr.records', 'DSLR_DIAGNOSTIC_VARIANTS'), 'DSLR_LINK_REDUCTIONS': ('gecko.algorithms.continual.dslr.records', 'DSLR_LINK_REDUCTIONS'), 'DSLR_REPLAY_CEILING_BYTES': ('gecko.algorithms.continual.dslr.records', 'DSLR_REPLAY_CEILING_BYTES'), 'DSLR_REPLAY_FRACTION': ('gecko.algorithms.continual.dslr.records', 'DSLR_REPLAY_FRACTION'), 'DSLR_SELECTION_MODES': ('gecko.algorithms.continual.dslr.records', 'DSLR_SELECTION_MODES'), 'DSLR_STRUCTURE_MODES': ('gecko.algorithms.continual.dslr.records', 'DSLR_STRUCTURE_MODES'), '_CHECKPOINT_VERSION': ('gecko.algorithms.continual.dslr.records', '_CHECKPOINT_VERSION'), '_DTYPES': ('gecko.algorithms.continual.dslr.records', '_DTYPES'), '_FLOAT_DTYPES': ('gecko.algorithms.continual.dslr.records', '_FLOAT_DTYPES'), '_MAX_METADATA_BYTES': ('gecko.algorithms.continual.dslr.records', '_MAX_METADATA_BYTES'), '_SNAPSHOT_FORMAT': ('gecko.algorithms.continual.dslr.records', '_SNAPSHOT_FORMAT'), '_SNAPSHOT_MAGIC': ('gecko.algorithms.continual.dslr.records', '_SNAPSHOT_MAGIC'), '_STATE_VERSION': ('gecko.algorithms.continual.dslr.records', '_STATE_VERSION'), '_TENSOR_ORDER': ('gecko.algorithms.continual.dslr.records', '_TENSOR_ORDER'), '_canonical_json_bytes': ('gecko.algorithms.continual.dslr.records', '_canonical_json_bytes'), '_complement_ordinals_to_pair_ranks': ('gecko.algorithms.continual.dslr.structure', '_complement_ordinals_to_pair_ranks'), '_decode_tensor': ('gecko.algorithms.continual.dslr.records', '_decode_tensor'), '_logical_positive_universe': ('gecko.algorithms.continual.dslr.structure', '_logical_positive_universe'), '_owned_tensor': ('gecko.algorithms.continual.dslr.records', '_owned_tensor'), '_payload_bytes': ('gecko.algorithms.continual.dslr.records', '_payload_bytes'), '_payload_tensor': ('gecko.algorithms.continual.dslr.replay', '_payload_tensor'), '_sample_unique_ordinals': ('gecko.algorithms.continual.dslr.structure', '_sample_unique_ordinals'), '_snapshot_body': ('gecko.algorithms.continual.dslr.records', '_snapshot_body'), '_snapshot_digest': ('gecko.algorithms.continual.dslr.records', '_snapshot_digest'), '_snapshot_metadata': ('gecko.algorithms.continual.dslr.records', '_snapshot_metadata'), '_tensor_bytes': ('gecko.algorithms.continual.dslr.records', '_tensor_bytes'), '_tensor_descriptor': ('gecko.algorithms.continual.dslr.records', '_tensor_descriptor'), '_update_digest': ('gecko.algorithms.continual.dslr.records', '_update_digest'), '_validate_local_edges': ('gecko.algorithms.continual.dslr.records', '_validate_local_edges'), '_validate_local_indices': ('gecko.algorithms.continual.dslr.records', '_validate_local_indices'), '_validate_nonnegative_int': ('gecko.algorithms.continual.dslr.records', '_validate_nonnegative_int'), '_validate_positive_float': ('gecko.algorithms.continual.dslr.records', '_validate_positive_float'), '_validate_positive_int': ('gecko.algorithms.continual.dslr.records', '_validate_positive_int'), '_validate_probability': ('gecko.algorithms.continual.dslr.records', '_validate_probability'), 'allocate_class_quotas': ('gecko.algorithms.continual.dslr.selection', 'allocate_class_quotas'), 'build_dslr_overlay': ('gecko.algorithms.continual.dslr.structure', 'build_dslr_overlay'), 'cosine_link_scores': ('gecko.algorithms.continual.dslr.structure', 'cosine_link_scores'), 'coverage_sets': ('gecko.algorithms.continual.dslr.selection', 'coverage_sets'), 'deserialize_dslr_snapshot': ('gecko.algorithms.continual.dslr.records', 'deserialize_dslr_snapshot'), 'downstream_classification_loss': ('gecko.algorithms.continual.dslr.structure', 'downstream_classification_loss'), 'fit_structure_learner': ('gecko.algorithms.continual.dslr.structure', 'fit_structure_learner'), 'greedy_coverage_selection': ('gecko.algorithms.continual.dslr.selection', 'greedy_coverage_selection'), 'link_prediction_loss': ('gecko.algorithms.continual.dslr.structure', 'link_prediction_loss'), 'mean_feature_selection': ('gecko.algorithms.continual.dslr.selection', 'mean_feature_selection'), 'node_supervision_loss': ('gecko.algorithms.continual.dslr.structure', 'node_supervision_loss'), 'sample_strict_local_negative_edges': ('gecko.algorithms.continual.dslr.structure', 'sample_strict_local_negative_edges'), 'select_prior_candidates': ('gecko.algorithms.continual.dslr.selection', 'select_prior_candidates'), 'serialize_dslr_snapshot': ('gecko.algorithms.continual.dslr.records', 'serialize_dslr_snapshot'), 'structure_learning_loss': ('gecko.algorithms.continual.dslr.structure', 'structure_learning_loss')}

def __getattr__(name: str):
    from importlib import import_module
    if name not in _RELOCATED_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _RELOCATED_EXPORTS[name]
    return getattr(import_module(module), symbol)
