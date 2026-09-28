from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from gecko.algorithms.continual.cat.node import CAT_CONDENSATION_LOSS_SEMANTICS
from gecko.algorithms.continual.cat.node import CaTAlgorithm
from gecko.algorithms.continual.cat.node import CondensedGraphRecord
from gecko.algorithms.continual.cat.node import _distribution_matching_loss
from gecko.algorithms.continual.cat.node import _identity_edges
from gecko.algorithms.continual.cat.node import _reset_random_encoder
from gecko.algorithms.context import ClientMethodContext


class TinyGraphModel(nn.Module):
    def __init__(self, *, classes: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 5, bias=False)
        self.head = nn.Linear(5, classes)

    def encode(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        aggregate = torch.zeros_like(features)
        degree = torch.zeros(features.shape[0], device=features.device)
        if edge_index.numel():
            aggregate.index_add_(0, edge_index[1], features[edge_index[0]])
            degree.index_add_(
                0,
                edge_index[1],
                torch.ones(edge_index.shape[1], device=features.device),
            )
        mixed = features + aggregate / degree.clamp_min(1).unsqueeze(1)
        return F.relu(self.encoder(mixed))

    def logits(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(features, edge_index))


def _context(
    *,
    stage: int = 0,
    task: int = 0,
    labels: torch.Tensor | None = None,
    valid_class_mask: torch.Tensor | None = None,
) -> ClientMethodContext:
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.9, 0.1, 0.1],
            [0.7, 0.3, 0.1],
            [0.0, 1.0, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.9, 0.2],
            [0.2, 0.7, 0.1],
        ],
        dtype=torch.float32,
    )
    edges = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 5, 6],
            [1, 2, 3, 0, 5, 6, 7, 4, 0, 0, 4, 4],
        ],
        dtype=torch.long,
    )
    task_labels = (
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        if labels is None
        else labels
    )
    class_mask = (
        torch.tensor([True, True, False, False])
        if valid_class_mask is None
        else valid_class_mask
    )

    def forward(
        model: nn.Module,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(model, TinyGraphModel)
        return model.logits(node_features, edge_index)[queries]

    def encode(
        model: nn.Module,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        layer_index: int | None,
    ) -> torch.Tensor:
        assert layer_index is None
        assert isinstance(model, TinyGraphModel)
        return model.encode(node_features, edge_index)

    return ClientMethodContext(
        client_id=2,
        global_task_id=task,
        stage_index=stage,
        round_index=0,
        problem_type="NC",
        incremental_setting="class",
        train_queries=torch.arange(8),
        train_labels=task_labels,
        valid_class_mask=class_mask,
        node_features=features,
        base_edge_index=edges,
        context_edge_index=edges,
        forward_queries=forward,
        encode_nodes=encode,
    )


def _algorithm(**overrides: object) -> CaTAlgorithm:
    parameters = {
        "client_id": 2,
        "seed": 19,
        "problem_type": "NC",
        "incremental_setting": "class",
        "synthetic_nodes_per_class": 1,
        "condensation_steps": 2,
        "condensation_lr": 0.01,
        "memory_ceiling_bytes": 1_000_000,
        "stage_count": 3,
        "feature_initialization": "random_choice",
    }
    parameters.update(overrides)
    return CaTAlgorithm(**parameters)


def _run_first_task(
    algorithm: CaTAlgorithm, model: TinyGraphModel, context: ClientMethodContext
) -> torch.Tensor:
    logits = context.forward_queries(model)
    masked = logits.clone()
    class_mask = context.valid_class_mask
    assert class_mask is not None
    masked[:, ~class_mask] = -1e12
    base_loss = F.cross_entropy(masked, context.train_labels)
    return algorithm.augment_loss(model, context, logits, base_loss)


def test_cat_memory_starts_empty_and_updates_with_real_condensed_topology() -> None:
    algorithm = _algorithm()
    context = _context()
    assert algorithm.replay_samples(context) == ()
    assert algorithm.replay_payload_bytes() == 0

    loss = _run_first_task(algorithm, TinyGraphModel(), context)
    assert torch.isfinite(loss)
    records = algorithm.replay_samples(context)
    assert len(records) == 1
    record = records[0]
    assert record.synthetic_node_count == 2
    assert record.synthetic_node_count < record.original_node_count
    assert record.synthetic_edge_count == 2
    assert torch.equal(record.edge_index, _identity_edges(2))
    assert record.class_distribution == {0: 1, 1: 1}


def test_cat_record_rejects_invalid_graph_features_and_non_reduction() -> None:
    common = {
        "client_id": 0,
        "global_task_id": 0,
        "stage_index": 0,
        "original_node_count": 4,
        "condensation_steps": 1,
        "condensation_seed": 1,
        "condensation_seconds": 0.1,
        "initial_loss": 1.0,
        "final_loss": 0.5,
        "features": torch.ones(2, 3),
        "labels": torch.tensor([0, 1]),
        "edge_index": _identity_edges(2),
    }
    record = CondensedGraphRecord(**common)
    assert record.payload_bytes > 0

    invalid_edges = dict(common)
    invalid_edges["edge_index"] = torch.tensor([[0], [1]])
    with pytest.raises(ValueError, match="identity"):
        CondensedGraphRecord(**invalid_edges)

    invalid_features = dict(common)
    invalid_features["features"] = torch.tensor([[float("nan"), 0.0, 0.0], [1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        CondensedGraphRecord(**invalid_features)

    no_reduction = dict(common)
    no_reduction["original_node_count"] = 2
    with pytest.raises(ValueError, match="smaller"):
        CondensedGraphRecord(**no_reduction)


def test_cat_condensation_objective_decreases_on_controlled_features() -> None:
    real = F.normalize(torch.tensor([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0], [0.0, 1.5]]), dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    synthetic = nn.Parameter(torch.tensor([[0.2, 1.0], [1.0, 0.2]]))
    synthetic_labels = torch.tensor([0, 1])
    optimizer = torch.optim.Adam([synthetic], lr=0.05)
    initial = float(
        _distribution_matching_loss(
            real, labels, F.normalize(synthetic, dim=-1), synthetic_labels
        ).detach()
    )
    for _ in range(50):
        loss = _distribution_matching_loss(
            real, labels, F.normalize(synthetic, dim=-1), synthetic_labels
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = float(
        _distribution_matching_loss(
            real, labels, F.normalize(synthetic, dim=-1), synthetic_labels
        ).detach()
    )
    assert final < initial


def test_cat_reported_losses_use_one_fixed_random_encoder_objective() -> None:
    algorithm = _algorithm(condensation_steps=3)
    model = TinyGraphModel()
    context = _context()
    _run_first_task(algorithm, model, context)
    record = algorithm.replay_samples(context)[0]
    labels = context.train_labels.detach().cpu().long()
    classes = tuple(int(value) for value in torch.unique(labels).tolist())
    initial_features, initial_labels = algorithm._initialize_features(
        context,
        labels=labels,
        classes=classes,
        seed=record.condensation_seed,
    )

    def fixed_objective(features: torch.Tensor) -> float:
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(record.condensation_seed)
            encoder = copy.deepcopy(model)
            _reset_random_encoder(encoder)
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
            with torch.no_grad():
                real_all = context.encode_nodes(encoder)
                real = F.normalize(real_all[context.train_queries], p=2, dim=-1)
                synthetic = F.normalize(
                    context.encode_nodes(
                        encoder,
                        node_features=features,
                        edge_index=_identity_edges(features.shape[0]),
                    ),
                    p=2,
                    dim=-1,
                )
                return float(
                    _distribution_matching_loss(
                        real,
                        labels,
                        synthetic,
                        initial_labels,
                    )
                )

    assert record.initial_loss == pytest.approx(fixed_objective(initial_features))
    assert record.final_loss == pytest.approx(fixed_objective(record.features))
    task = algorithm.diagnostics()["per_task"]["0"]
    assert task["condensation_loss_semantics"] == (
        CAT_CONDENSATION_LOSS_SEMANTICS
    )
    assert task["condensation_objective_delta"] == pytest.approx(
        record.final_loss - record.initial_loss
    )


def test_cat_is_deterministic_and_does_not_perturb_global_rng() -> None:
    torch.manual_seed(91)
    base_model = TinyGraphModel()
    model_a = copy.deepcopy(base_model)
    model_b = copy.deepcopy(base_model)
    context = _context()
    algorithm_a = _algorithm()
    algorithm_b = _algorithm()

    rng_before = torch.get_rng_state().clone()
    _run_first_task(algorithm_a, model_a, context)
    rng_after = torch.get_rng_state().clone()
    _run_first_task(algorithm_b, model_b, context)

    assert torch.equal(rng_before, rng_after)
    first = algorithm_a.replay_samples(context)[0]
    second = algorithm_b.replay_samples(context)[0]
    assert torch.equal(first.features, second.features)
    assert torch.equal(first.labels, second.labels)
    assert torch.equal(first.edge_index, second.edge_index)
    assert first.condensation_seed == second.condensation_seed


def test_cat_accumulates_balanced_task_memory_and_uses_memory_only() -> None:
    algorithm = _algorithm()
    model = TinyGraphModel()
    first_context = _context()
    first_loss = _run_first_task(algorithm, model, first_context)
    logits = first_context.forward_queries(model)
    alternate = algorithm.augment_loss(
        model, first_context, logits, logits.sum() * 0.0 + 999.0
    )
    assert torch.allclose(first_loss, alternate)

    second_context = _context(
        stage=1,
        task=1,
        labels=torch.tensor([2, 2, 2, 2, 3, 3, 3, 3]),
        valid_class_mask=torch.tensor([True, True, True, True]),
    )
    _run_first_task(algorithm, model, second_context)
    records = algorithm.replay_samples(second_context)
    assert [record.global_task_id for record in records] == [0, 1]
    assert [record.synthetic_node_count for record in records] == [2, 2]
    assert algorithm.diagnostics()["class_distribution"] == {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 1,
    }


def test_cat_leakage_guard_uses_local_stage_not_permuted_global_task_id() -> None:
    algorithm = _algorithm()
    model = TinyGraphModel()
    first_context = _context(stage=0, task=5)
    _run_first_task(algorithm, model, first_context)

    later_local_stage = _context(
        stage=1,
        task=3,
        labels=torch.tensor([2, 2, 2, 2, 3, 3, 3, 3]),
        valid_class_mask=torch.tensor([True, True, True, True]),
    )
    algorithm.before_task(later_local_stage)
    _run_first_task(algorithm, model, later_local_stage)
    assert [record.global_task_id for record in algorithm.replay_samples(later_local_stage)] == [5, 3]

    tampered = algorithm.save_method_state()
    tampered["records"][0]["stage_index"] = 2
    algorithm.load_method_state(tampered)
    with pytest.raises(ValueError, match="future-stage"):
        algorithm.before_task(later_local_stage)


def test_cat_rejects_future_labels_and_memory_ceiling_overflow() -> None:
    future_context = _context(
        labels=torch.tensor([0, 0, 0, 0, 3, 3, 3, 3]),
        valid_class_mask=torch.tensor([True, True, False, False]),
    )
    with pytest.raises(ValueError, match="future or inactive"):
        _run_first_task(_algorithm(), TinyGraphModel(), future_context)

    with pytest.raises(MemoryError, match="memory_ceiling"):
        _run_first_task(
            _algorithm(memory_ceiling_bytes=1), TinyGraphModel(), _context()
        )


def test_cat_checkpoint_roundtrip_preserves_memory_and_rejects_mismatch() -> None:
    algorithm = _algorithm()
    model = TinyGraphModel()
    context = _context()
    _run_first_task(algorithm, model, context)
    state = algorithm.save_method_state()

    restored = _algorithm()
    restored.load_method_state(state)
    assert torch.equal(
        restored.replay_samples(context)[0].features,
        algorithm.replay_samples(context)[0].features,
    )
    assert restored.replay_payload_bytes() == algorithm.replay_payload_bytes()

    incompatible = _algorithm(condensation_steps=3)
    with pytest.raises(ValueError, match="hyperparameters"):
        incompatible.load_method_state(state)


def test_cat_configuration_supports_nc_task_and_fails_closed_outside_nc() -> None:
    assert _algorithm(incremental_setting="task")
    with pytest.raises(ValueError, match="NC-Class"):
        _algorithm(problem_type="LC")
