from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from torch import nn
from gecko.algorithms.base import ClientContinualAlgorithm
from gecko.engine.parameter_manifest import build_parameter_manifest
from gecko.engine.parameter_policy import SharedParameterPolicy
from gecko.models.backbones import GECKOGraphModel
from gecko.models.capabilities import attach_v2_model_capabilities

from gecko.models.registry import ModelRegistry


class RecordingAlgorithm(ClientContinualAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, int]] = []

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
        self.calls.append(("loss", global_task_id))
        return base_loss + logits.sum() * 0.0

    def mask_gradients(self, model, global_task_id):
        self.calls.append(("mask", global_task_id))

    def after_update(self, model, queries, labels, global_task_id):
        self.calls.append(("round", global_task_id))

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
        self.calls.append(("task", global_task_id))


def _context(model: GECKOGraphModel):
    features = torch.randn(5, 3)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    queries = torch.tensor([0, 2])
    labels = torch.tensor([0, 1])
    return SimpleNamespace(
        global_task_id=7,
        train_queries=queries,
        train_labels=labels,
        valid_class_mask=torch.tensor([True, True]),
        node_features=features,
        forward_queries=lambda current, values: current.forward_queries(
            features, edges, values, "NC"
        ),
    )


def test_v2_lifecycle_adapters_delegate_to_legacy_hooks_exactly_once():
    model = GECKOGraphModel(3, 2, hidden_size=4, num_layers=2)
    algorithm = RecordingAlgorithm()
    context = _context(model)
    logits = context.forward_queries(model, context.train_queries)
    base_loss = logits.sum()

    actual = algorithm.augment_loss(model, context, logits, base_loss)
    assert torch.equal(actual, base_loss)
    algorithm.after_backward(model, context)
    algorithm.after_round(model, context)
    algorithm.consolidate(model, context)

    assert algorithm.calls == [
        ("loss", 7),
        ("mask", 7),
        ("round", 7),
        ("task", 7),
    ]


def test_method_state_round_trip_preserves_state_mapping_identity_and_clones():
    algorithm = ClientContinualAlgorithm()
    algorithm.state.update(
        {"anchors": {1: torch.tensor([1.0])}, "history": (1, "task")}
    )
    mapping_id = id(algorithm.state)
    snapshot = algorithm.save_method_state()
    snapshot["anchors"][1].add_(10)
    assert algorithm.state["anchors"][1].item() == 1.0

    algorithm.load_method_state({"restored": [torch.tensor([3.0])]})
    assert id(algorithm.state) == mapping_id
    assert algorithm.state["restored"][0].item() == 3.0
    with pytest.raises(TypeError, match="unsupported object"):
        algorithm.load_method_state({"bad": object()})


def test_uefa_gcn_capabilities_are_functional_and_do_not_change_forward():
    torch.manual_seed(4)
    model = attach_v2_model_capabilities(
        GECKOGraphModel(3, 2, hidden_size=4, num_layers=2)
    )
    features = torch.randn(6, 3)
    edges = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    queries = torch.tensor([0, 2, 5])
    before = model.forward_queries(features, edges, queries, "NC")
    parameter_names = tuple(dict(model.named_parameters()))

    embeddings = model.encode_nodes(features, edges, layer_index=0)
    scores = model.collect_topology_scores(features, edges, layer_index=0)
    scores.square().sum().backward()
    # Topology scores exercise only the encoder, so the classifier gradients
    # are legitimately absent until supplied by an NC objective/correction.
    model.node_head.weight.grad = torch.ones_like(model.node_head.weight)
    model.node_head.bias.grad = torch.ones_like(model.node_head.bias)
    model.mask_output_gradients(torch.tensor([True, False]))
    assert torch.equal(model.node_head.weight.grad[0], torch.ones(4))
    assert torch.equal(model.node_head.weight.grad[1], torch.zeros(4))
    assert model.node_head.bias.grad.tolist() == [1.0, 0.0]
    after = model.forward_queries(features, edges, queries, "NC")

    assert embeddings.shape == (6, 4)
    assert scores.shape == (edges.shape[1],)
    assert scores.requires_grad
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert tuple(dict(model.named_parameters())) == parameter_names
    assert torch.equal(before, after)


def test_uefa_lc_output_gradient_mask_targets_edge_head_only():
    model = attach_v2_model_capabilities(
        GECKOGraphModel(3, 3, hidden_size=4, num_layers=2, problem_type="LC")
    )
    model.edge_head.weight.grad = torch.ones_like(model.edge_head.weight)
    model.edge_head.bias.grad = torch.ones_like(model.edge_head.bias)
    model.node_head.weight.grad = torch.full_like(model.node_head.weight, 2.0)
    model.mask_output_gradients(torch.tensor([True, False, True]))
    assert torch.equal(model.edge_head.weight.grad[0], torch.ones(8))
    assert torch.equal(model.edge_head.weight.grad[1], torch.zeros(8))
    assert model.edge_head.bias.grad.tolist() == [1.0, 0.0, 1.0]
    assert torch.equal(
        model.node_head.weight.grad,
        torch.full_like(model.node_head.weight, 2.0),
    )


class ManifestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 4)
        self.local_head = nn.Linear(4, 2)
        self.register_buffer("running_marker", torch.ones(1))


def test_parameter_manifest_is_exhaustive_disjoint_and_legacy_compatible():
    model = ManifestModel()
    policy = SharedParameterPolicy()
    manifest = build_parameter_manifest(model, policy)

    assert manifest.shared_trainable == policy.shareable_keys(model)
    assert set(manifest.local_trainable) == {
        "local_head.weight",
        "local_head.bias",
    }
    assert manifest.local_buffers == ("running_marker",)
    classified = (
        set(manifest.shared_trainable)
        | set(manifest.local_trainable)
        | set(manifest.local_buffers)
        | set(manifest.dynamic_metadata)
    )
    assert classified == (
        set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
    )
    assert sum(
        len(values)
        for values in (
            manifest.shared_trainable,
            manifest.local_trainable,
            manifest.local_buffers,
            manifest.dynamic_metadata,
        )
    ) == len(classified)


def test_parameter_manifest_fails_closed_for_frozen_named_parameters():
    model = ManifestModel()
    model.shared.weight.requires_grad_(False)
    policy = SharedParameterPolicy()

    assert "shared.weight" in policy.shareable_keys(model)
    with pytest.raises(
        ValueError,
        match=r"frozen parameters.*shared\.weight",
    ):
        build_parameter_manifest(model, policy)


@pytest.mark.integration
def test_begin_gcn_manifest_and_dynamic_metadata_round_trip():
    model = ModelRegistry().create(
        "begin_gcn",
        input_size=3,
        output_size=3,
        hidden_size=4,
        num_layers=2,
        problem_type="NC",
        incremental_type="class",
    )
    model = attach_v2_model_capabilities(model)
    policy = SharedParameterPolicy()
    manifest = build_parameter_manifest(model, policy)
    assert manifest.shared_trainable == policy.shareable_keys(model)
    assert set(manifest.local_buffers) == set(dict(model.named_buffers()))
    assert "original.classifier.observed" in manifest.dynamic_metadata
    assert "original.classifier.output_masks" in manifest.dynamic_metadata
    cache_entries = {
        entry.name: entry
        for entry in manifest.entries
        if entry.name.startswith("_cached")
    }
    assert cache_entries
    assert all(not entry.checkpoint_persistent for entry in cache_entries.values())

    expected = model.export_dynamic_state()
    classifier = model.original.classifier
    classifier.observed.zero_()
    classifier.output_masks = None
    model._cached_graph_key = ("stale",)
    model._cached_graph = object()
    model.load_dynamic_state(expected)

    assert classifier.observed.all()
    assert classifier.output_masks is not None
    assert classifier.output_masks[0].all()
    assert model._cached_graph_key is None
    assert model._cached_graph is None
    classifier.lin.weight.grad = torch.ones_like(classifier.lin.weight)
    classifier.lin.bias.grad = torch.ones_like(classifier.lin.bias)
    model.mask_output_gradients(torch.tensor([True, False, True]))
    assert torch.equal(classifier.lin.weight.grad[0], torch.ones(4))
    assert torch.equal(classifier.lin.weight.grad[1], torch.zeros(4))
    assert classifier.lin.bias.grad.tolist() == [1.0, 0.0, 1.0]
