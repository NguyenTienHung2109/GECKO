"""Paper-equation and strict-local behavioral tests for DSLR."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from gecko.algorithms.continual.dslr import algorithm as dslr_module
from gecko.algorithms.context import ClientMethodContext
from gecko.algorithms.continual.dslr.records import DSLR_DEFAULT_STRUCTURE_EPOCHS
from gecko.algorithms.continual.dslr.algorithm import DSLRAlgorithm
from gecko.algorithms.continual.dslr.records import DSLRReplaySnapshot
from gecko.algorithms.continual.dslr.replay import DSLRReplayStore
from gecko.algorithms.continual.dslr.structure import DSLRStructureLearner
from gecko.algorithms.continual.dslr.structure import build_dslr_overlay
from gecko.algorithms.continual.dslr.records import deserialize_dslr_snapshot
from gecko.algorithms.continual.dslr.structure import downstream_classification_loss
from gecko.algorithms.continual.dslr.structure import fit_structure_learner
from gecko.algorithms.continual.dslr.selection import greedy_coverage_selection
from gecko.algorithms.continual.dslr.selection import mean_feature_selection
from gecko.algorithms.continual.dslr.structure import sample_strict_local_negative_edges
from gecko.algorithms.continual.dslr.records import serialize_dslr_snapshot
from gecko.algorithms.continual.dslr.structure import structure_learning_loss
from gecko.algorithms.topology import edge_index_sha256


def _cuda_kernels_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in torch.cuda.get_arch_list()


def _chain(size: int) -> torch.Tensor:
    arcs = [
        pair
        for node in range(size - 1)
        for pair in ((node, node + 1), (node + 1, node))
    ]
    return torch.tensor(arcs, dtype=torch.long).t().contiguous()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(37)
            self.encoder = nn.Linear(4, 6)
            self.classifier = nn.Linear(6, 3)

    def encode_nodes(
        self,
        features: torch.Tensor,
        edges: torch.Tensor,
        *,
        layer_index: int | None = None,
    ) -> torch.Tensor:
        del layer_index
        aggregate = features.clone()
        degree = features.new_ones(features.shape[0])
        edges = edges.to(features.device)
        aggregate.index_add_(0, edges[1], features[edges[0]])
        degree.index_add_(0, edges[1], features.new_ones(edges.shape[1]))
        return F.relu(self.encoder(aggregate / degree.unsqueeze(-1)))

    def forward_queries(
        self,
        features: torch.Tensor,
        edges: torch.Tensor,
        queries: torch.Tensor,
        problem_type: str,
    ) -> torch.Tensor:
        assert problem_type == "NC"
        return self.classifier(self.encode_nodes(features, edges))[queries]


def _context(
    features: torch.Tensor,
    base: torch.Tensor,
    *,
    task: int,
    visible: int,
) -> ClientMethodContext:
    queries = torch.arange(task * 20, (task + 1) * 20)
    mask = torch.zeros(3, dtype=torch.bool)
    mask[: task + 1] = True

    def forward(model, node_features, edge_index, values):
        return model.forward_queries(node_features, edge_index, values, "NC")

    def encode(model, node_features, edge_index, layer_index):
        return model.encode_nodes(node_features, edge_index, layer_index=layer_index)

    return ClientMethodContext(
        client_id=0,
        global_task_id=task,
        stage_index=task,
        round_index=0,
        problem_type="NC",
        incremental_setting="class",
        train_queries=queries,
        train_labels=torch.full((20,), task, dtype=torch.long),
        valid_class_mask=mask,
        node_features=features,
        base_edge_index=base,
        context_edge_index=_chain(visible),
        forward_queries=forward,
        encode_nodes=encode,
    )


def _method(**overrides: object) -> DSLRAlgorithm:
    values: dict[str, object] = {
        "client_id": 0,
        "seed": 11,
        "problem_type": "NC",
        "incremental_setting": "class",
        "beta": 0.1,
        "radius": 0.15,
        "top_n": 2,
        "candidate_k": 5,
        "structure_epochs": 2,
        "structure_hidden_dim": 4,
        "structure_heads": 2,
        "replay_fraction": 0.05,
    }
    values.update(overrides)
    return DSLRAlgorithm(**values)


def _snapshot(candidates: list[int]) -> DSLRReplaySnapshot:
    return DSLRReplaySnapshot(
        client_id=0,
        global_task_id=0,
        stage_index=0,
        class_id=0,
        source_local_index=0,
        feature=torch.tensor([1.0, 0.0]),
        embedding=torch.tensor([1.0, 0.0]),
        candidate_local_indices=torch.tensor(candidates),
    )


def test_equation_closed_forms_coverage_removal_and_negative_universe():
    assert structure_learning_loss(
        link_loss=torch.tensor(2.0),
        node_loss=torch.tensor(4.0),
        structure_lambda=0.5,
    ).item() == pytest.approx(3.0)
    assert downstream_classification_loss(
        torch.tensor(2.0), torch.tensor(4.0), beta=0.1
    ).item() == pytest.approx(3.8)
    embeddings = torch.tensor([[0.0], [0.05], [0.1], [10.0]])
    labels = torch.zeros(4, dtype=torch.long)
    assert greedy_coverage_selection(
        embeddings,
        labels,
        torch.arange(4),
        class_quotas={0: 2},
        radius=0.2,
    ) == (0, 3)
    with pytest.raises(ValueError, match="coverage exhausted"):
        greedy_coverage_selection(
            embeddings,
            labels,
            torch.arange(4),
            class_quotas={0: 3},
            radius=0.2,
        )
    negatives = sample_strict_local_negative_edges(
        torch.tensor([[0, 1], [1, 0]]),
        num_nodes=4,
        count=2,
        generator=torch.Generator().manual_seed(9),
        allowed_nodes=torch.tensor([0, 1, 2]),
    )
    assert {tuple(sorted(pair)) for pair in negatives.t().tolist()} == {(0, 2), (1, 2)}


def test_negative_sampling_is_deterministic_unique_and_has_full_small_support():
    edges = torch.tensor([[1, 3, 3], [3, 1, 3]], dtype=torch.long)
    allowed = torch.tensor([1, 3, 7, 9], dtype=torch.long)
    first = sample_strict_local_negative_edges(
        edges,
        num_nodes=10,
        count=3,
        generator=torch.Generator().manual_seed(151),
        allowed_nodes=allowed,
    )
    second = sample_strict_local_negative_edges(
        edges,
        num_nodes=10,
        count=3,
        generator=torch.Generator().manual_seed(151),
        allowed_nodes=allowed,
    )
    assert torch.equal(first, second)
    pairs = [tuple(pair) for pair in first.t().tolist()]
    assert len(pairs) == len(set(pairs)) == 3
    assert set(pairs) <= {(1, 7), (1, 9), (3, 7), (3, 9), (7, 9)}

    observed = {
        tuple(
            sample_strict_local_negative_edges(
                torch.empty((2, 0), dtype=torch.long),
                num_nodes=4,
                count=1,
                generator=torch.Generator().manual_seed(seed),
            )[:, 0].tolist()
        )
        for seed in range(128)
    }
    assert observed == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}


@pytest.mark.gpu
@pytest.mark.skipif(
    not _cuda_kernels_supported(),
    reason="CUDA kernels for the active GPU are unavailable",
)
def test_negative_sampling_keeps_runtime_tensors_on_cuda():
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(151)
    negatives = sample_strict_local_negative_edges(
        torch.tensor([[1, 3, 3], [3, 1, 3]], dtype=torch.long, device=device),
        num_nodes=10,
        count=3,
        generator=generator,
        allowed_nodes=torch.tensor([1, 3, 7, 9], dtype=torch.long),
        device=device,
    )
    assert negatives.device == device
    pairs = [tuple(pair) for pair in negatives.detach().cpu().t().tolist()]
    assert len(pairs) == len(set(pairs)) == 3
    assert set(pairs) <= {(1, 7), (1, 9), (3, 7), (3, 9), (7, 9)}


def test_negative_sampling_returns_complete_complement_and_scales_sparsely():
    edges = torch.tensor([[1, 3, 3], [3, 1, 3]], dtype=torch.long)
    complete = sample_strict_local_negative_edges(
        edges,
        num_nodes=10,
        count=5,
        generator=torch.Generator().manual_seed(5),
        allowed_nodes=torch.tensor([1, 3, 7, 9]),
    )
    assert {tuple(pair) for pair in complete.t().tolist()} == {
        (1, 7),
        (1, 9),
        (3, 7),
        (3, 9),
        (7, 9),
    }

    sparse = sample_strict_local_negative_edges(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        num_nodes=50_000,
        count=4,
        generator=torch.Generator().manual_seed(31),
    )
    sparse_pairs = [tuple(pair) for pair in sparse.t().tolist()]
    assert len(sparse_pairs) == len(set(sparse_pairs)) == 4
    assert all(0 <= source < target < 50_000 for source, target in sparse_pairs)
    assert (0, 1) not in sparse_pairs


def test_mean_feature_control_selects_nodes_closest_to_class_feature_center():
    features = torch.tensor([[0.0], [2.0], [10.0], [100.0], [104.0]])
    labels = torch.tensor([0, 0, 0, 1, 1])
    assert mean_feature_selection(
        features,
        labels,
        torch.arange(5),
        class_quotas={0: 2, 1: 1},
    ) == (1, 0, 3)


@pytest.mark.parametrize(
    ("selection", "structure", "variant"),
    [
        ("mean_feature", "none", "mf_no_structure"),
        ("coverage_diversity", "none", "cd_selection_only"),
        ("mean_feature", "full", "mf_full_structure"),
        ("coverage_diversity", "link_only", "cd_link_only"),
        ("coverage_diversity", "node_only", "cd_node_only"),
    ],
)
def test_named_ablation_pairs_are_diagnostic_only(
    selection: str, structure: str, variant: str
):
    method = _method(selection_mode=selection, structure_mode=structure)
    assert method.name == "DSLRDiagnostic"
    assert method.diagnostic_variant == variant
    assert method.diagnostics()["fidelity_status"] == "diagnostic_only"
    with pytest.raises(ValueError, match="Unknown DSLR diagnostic"):
        _method(selection_mode="mean_feature", structure_mode="link_only")
    assert _method().name == "DSLR"
    assert _method().diagnostic_variant is None


def test_overlay_filters_connected_nodes_before_top_n_and_fails_hidden_candidates():
    base = torch.tensor([[0, 1], [1, 0]])
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
    overlay = build_dslr_overlay(
        client_id=0,
        global_task_id=1,
        base_edge_index=base,
        allowed_nodes=torch.arange(4),
        structure_embeddings=embeddings,
        replay_snapshots=[_snapshot([1, 2, 3])],
        top_n=2,
    )
    assert set(map(tuple, overlay.added_edge_index.t().tolist())) == {
        (0, 2),
        (2, 0),
        (0, 3),
        (3, 0),
    }
    with pytest.raises(ValueError, match="outside the visible context"):
        build_dslr_overlay(
            client_id=0,
            global_task_id=1,
            base_edge_index=base,
            allowed_nodes=torch.tensor([0, 1, 2]),
            structure_embeddings=embeddings,
            replay_snapshots=[_snapshot([1, 2, 3])],
        )


def test_gat_training_is_deterministic_exact_budget_and_rejects_hidden_positives():
    before = torch.get_rng_state().clone()
    first = DSLRStructureLearner(
        input_dim=2,
        hidden_dim=3,
        heads=2,
        num_classes=2,
        initialization_seed=101,
    )
    second = DSLRStructureLearner(
        input_dim=2,
        hidden_dim=3,
        heads=2,
        num_classes=2,
        initialization_seed=101,
    )
    assert torch.equal(torch.get_rng_state(), before)
    kwargs = {
        "node_features": torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10,
        "edge_index": _chain(4),
        "allowed_nodes": torch.arange(4),
        "current_indices": torch.tensor([2, 3]),
        "current_labels": torch.tensor([1, 1]),
        "replay_indices": torch.tensor([0, 1]),
        "replay_labels": torch.tensor([0, 0]),
        "beta": 0.1,
        "epochs": 3,
        "rng_seed": 17,
    }
    assert fit_structure_learner(first, **kwargs) == fit_structure_learner(
        second, **kwargs
    )
    assert first.state_sha256() == second.state_sha256()
    assert torch.equal(torch.get_rng_state(), before)
    assert DSLR_DEFAULT_STRUCTURE_EPOCHS == 99
    with pytest.raises(ValueError, match="positive topology"):
        fit_structure_learner(
            DSLRStructureLearner(
                input_dim=2,
                hidden_dim=2,
                heads=1,
                num_classes=2,
                initialization_seed=3,
            ),
            node_features=torch.ones((4, 2)),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            allowed_nodes=torch.tensor([0, 1, 2]),
            current_indices=torch.tensor([2]),
            current_labels=torch.tensor([1]),
            replay_indices=torch.tensor([0]),
            replay_labels=torch.tensor([0]),
            beta=0.1,
            epochs=1,
            rng_seed=5,
        )


def test_snapshot_checksum_and_whole_record_ceiling_are_exact():
    snapshot = _snapshot([1, 2])
    payload = serialize_dslr_snapshot(snapshot)
    assert deserialize_dslr_snapshot(payload).snapshot_id == snapshot.snapshot_id
    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        deserialize_dslr_snapshot(corrupted)
    store = DSLRReplayStore(client_id=0, ceiling_bytes=snapshot.serialized_bytes)
    report = store.replace([snapshot], seen_classes=[0], requested_count=1)
    assert report["replay_payload_bytes"] == snapshot.serialized_bytes
    with pytest.raises(ValueError, match="one complete snapshot"):
        DSLRReplayStore(
            client_id=0, ceiling_bytes=snapshot.serialized_bytes - 1
        ).replace([snapshot], seen_classes=[0], requested_count=1)


def test_no_structure_diagnostics_replay_without_preparing_an_overlay():
    features = torch.arange(240, dtype=torch.float32).reshape(60, 4) / 100
    base = _chain(60)
    context_zero = _context(features, base, task=0, visible=20)
    method = _method(selection_mode="mean_feature", structure_mode="none")
    model = _Model()
    method.consolidate(model, context_zero)
    assert method.replay_samples(context_zero)[0].candidate_local_indices.numel() == 0
    context_one = _context(features, base, task=1, visible=40)
    logits = context_one.forward_queries(model)
    base_loss = F.cross_entropy(logits, context_one.train_labels)
    loss = method.augment_loss(model, context_one, logits, base_loss)
    assert torch.isfinite(loss) and not torch.allclose(loss, base_loss)
    diagnostics = method.diagnostics()
    assert diagnostics["prepared_task_ids"] == ()
    assert diagnostics["structure_epochs"] == 0
    assert method.evaluation_topology(context_one) is None


@pytest.mark.parametrize(
    ("structure_mode", "expected_lambda"),
    [("link_only", 1.0), ("node_only", 0.0)],
)
def test_link_and_node_only_diagnostics_select_exact_equation8_weight(
    monkeypatch, structure_mode: str, expected_lambda: float
):
    captured = []
    from gecko.algorithms.continual.dslr import structure as structure_module
    original = structure_module.fit_structure_learner

    def wrapped(*args, **kwargs):
        captured.append(kwargs["structure_lambda"])
        return original(*args, **kwargs)

    monkeypatch.setattr(structure_module, "fit_structure_learner", wrapped)
    features = torch.arange(240, dtype=torch.float32).reshape(60, 4) / 100
    base = _chain(60)
    context_zero = _context(features, base, task=0, visible=20)
    context_one = _context(features, base, task=1, visible=40)
    method = _method(structure_mode=structure_mode)
    model = _Model()
    method.consolidate(model, context_zero)
    logits = context_one.forward_queries(model)
    method.augment_loss(
        model,
        context_one,
        logits,
        F.cross_entropy(logits, context_one.train_labels),
    )
    assert captured == [expected_lambda]


def test_lifecycle_excludes_future_nodes_activates_loss_and_roundtrips_state():
    features = torch.arange(240, dtype=torch.float32).reshape(60, 4) / 100
    future_perturbed = features.clone()
    future_perturbed[20:] += 10_000
    base = _chain(60)
    context_zero = _context(features, base, task=0, visible=20)
    perturbed_zero = _context(future_perturbed, base, task=0, visible=20)
    model, twin_model = _Model(), _Model()
    method, twin = _method(), _method()
    method.consolidate(model, context_zero)
    twin.consolidate(twin_model, perturbed_zero)
    snapshot = method.replay_samples(context_zero)[0]
    assert serialize_dslr_snapshot(snapshot) == serialize_dslr_snapshot(
        twin.replay_samples(perturbed_zero)[0]
    )
    assert int(snapshot.candidate_local_indices.max()) < 20

    context_one = _context(features, base, task=1, visible=40)
    logits = context_one.forward_queries(model)
    base_loss = F.cross_entropy(logits, context_one.train_labels)
    loss = method.augment_loss(model, context_one, logits, base_loss)
    assert torch.isfinite(loss) and not torch.allclose(loss, base_loss)
    loss.backward()
    diagnostics = method.diagnostics()
    assert diagnostics["prepared_task_ids"] == (1,)
    assert diagnostics["structure_epochs"] == 2
    assert diagnostics["structure_visible_nodes"] == 40
    assert method.evaluation_topology(context_zero) is None
    overlay = method.evaluation_topology(context_one)
    assert overlay is not None
    assert overlay.base_edge_sha256 == edge_index_sha256(
        context_one.effective_edge_index, num_nodes=60
    )
    overlay.apply(context_one.effective_edge_index)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        overlay.apply(context_one.base_edge_index)

    state = method.save_method_state()
    restored = _method()
    restored.load_method_state(state)
    assert restored.diagnostics() == diagnostics
    assert restored.evaluation_topology(context_one).overlay_id == overlay.overlay_id
    tampered = copy.deepcopy(state)
    tampered["private_state"]["overlays"][1]["overlay_id"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        _method().load_method_state(tampered)
    tampered_diagnostics = copy.deepcopy(state)
    tampered_diagnostics["diagnostics"]["pre_broadcast_state_sha256"] = "invalid"
    with pytest.raises(ValueError, match="diagnostic checksum"):
        _method().load_method_state(tampered_diagnostics)
