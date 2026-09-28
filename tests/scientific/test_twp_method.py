from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from gecko.algorithms.continual.twp import TWPAlgorithm
from gecko.algorithms.continual.twp import normalized_nonparametric_attention
from gecko.algorithms.continual.twp import topology_attention_squared_norm
from gecko.algorithms.context import ClientMethodContext
from gecko.models.backbones import LegacyGCNAdapter
from gecko.models.backbones import GECKOGraphModel
from gecko.models.capabilities import attach_v2_model_capabilities
from gecko.models.capabilities import encode_model_nodes
from gecko.models.capabilities import project_model_nodes_for_twp


class _ToyGraphModel(nn.Module):
    twp_num_encoder_layers = 3

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(3)])
        self.classifier = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.layers[0].weight.copy_(torch.tensor([[0.7, -0.2], [0.3, 0.6]]))
            self.layers[1].weight.copy_(torch.tensor([[0.4, 0.5], [-0.6, 0.8]]))
            self.layers[2].weight.copy_(torch.tensor([[0.9, -0.3], [0.2, 0.7]]))
            self.classifier.weight.copy_(torch.tensor([[0.5, -0.4], [-0.3, 0.6]]))
            self.classifier.bias.copy_(torch.tensor([0.1, -0.2]))

    def encode_until(self, features: torch.Tensor, layer_index: int) -> torch.Tensor:
        hidden = features
        for index, layer in enumerate(self.layers):
            hidden = torch.tanh(layer(hidden))
            if index == layer_index:
                return hidden
        raise IndexError(layer_index)

    def project_for_twp(
        self, features: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        hidden = features
        for index, layer in enumerate(self.layers):
            if index == layer_index:
                return layer(hidden)
            hidden = torch.tanh(layer(hidden))
        raise IndexError(layer_index)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_until(features, len(self.layers) - 1)
        return self.classifier(hidden)


class _ToyContext:
    def __init__(self, *, task_id: int = 7, forbid_encoding: bool = False) -> None:
        self.client_id = 2
        self.global_task_id = task_id
        self.stage_index = 0
        self.round_index = 0
        self.problem_type = "NC"
        self.incremental_setting = "task"
        self.train_queries = torch.tensor([0, 1, 2], dtype=torch.long)
        self.train_labels = torch.tensor([0, 1, 0], dtype=torch.long)
        self.valid_class_mask = torch.tensor([True, True])
        self.node_features = torch.tensor(
            [[0.2, 1.0], [1.1, -0.4], [-0.7, 0.6], [0.9, 0.3]],
            dtype=torch.float32,
        )
        # Two centers each have three incoming neighbors, keeping the
        # equation-(10) softmax differentiable and nontrivial.
        self.effective_edge_index = torch.tensor(
            [[0, 1, 3, 0, 1, 2], [2, 2, 2, 3, 3, 3]],
            dtype=torch.long,
        )
        self.forbid_encoding = forbid_encoding
        self.encoded_layers: list[int] = []

    def forward_queries(
        self,
        model: _ToyGraphModel,
        queries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected = self.train_queries if queries is None else queries
        return model.logits(self.node_features)[selected]

    def encode_nodes(
        self,
        model: _ToyGraphModel,
        *,
        layer_index: int | None = None,
    ) -> torch.Tensor:
        if self.forbid_encoding:
            raise AssertionError("topology capability must not be used")
        assert layer_index is not None
        self.encoded_layers.append(layer_index)
        return model.encode_until(self.node_features, layer_index)

    def twp_project_nodes(
        self,
        model: _ToyGraphModel,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        if self.forbid_encoding:
            raise AssertionError("topology capability must not be used")
        self.encoded_layers.append(layer_index)
        return model.project_for_twp(self.node_features, layer_index)


class _ToyLCContext(_ToyContext):
    def __init__(self, *, task_id: int = 7) -> None:
        super().__init__(task_id=task_id)
        self.problem_type = "LC"
        self.train_queries = torch.tensor([[2, 3]], dtype=torch.long)
        self.train_labels = torch.tensor([0], dtype=torch.long)

    def forward_queries(
        self,
        model: _ToyGraphModel,
        queries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pairs = self.train_queries if queries is None else queries
        node_logits = model.logits(self.node_features)
        return 0.5 * (node_logits[pairs[:, 0]] + node_logits[pairs[:, 1]])


def _named_parameters(model: nn.Module):
    return tuple(sorted(model.named_parameters(), key=lambda item: item[0]))


def test_nonparametric_attention_matches_equation_10_and_incoming_orientation():
    embeddings = torch.tensor(
        [[0.2, -0.1], [0.7, 0.3], [-0.4, 0.8], [0.5, -0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edges = torch.tensor([[0, 1, 3, 0, 2], [2, 2, 2, 3, 3]], dtype=torch.long)
    coefficients, retained = normalized_nonparametric_attention(
        embeddings,
        edges,
        center_nodes=torch.tensor([2]),
    )
    raw = torch.stack(
        [
            embeddings[2].dot(torch.tanh(embeddings[0])),
            embeddings[2].dot(torch.tanh(embeddings[1])),
            embeddings[2].dot(torch.tanh(embeddings[3])),
        ]
    )
    assert torch.equal(retained, torch.tensor([0, 1, 2]))
    assert torch.allclose(coefficients, torch.softmax(raw, dim=0))
    assert float(coefficients.sum().detach()) == pytest.approx(1.0)

    all_coefficients, _ = normalized_nonparametric_attention(embeddings, edges)
    assert float(all_coefficients[:3].sum().detach()) == pytest.approx(1.0)
    assert float(all_coefficients[3:].sum().detach()) == pytest.approx(1.0)
    assert torch.allclose(
        topology_attention_squared_norm(embeddings, edges),
        all_coefficients.square().sum(),
    )
    assert not torch.allclose(
        topology_attention_squared_norm(embeddings, edges),
        torch.linalg.vector_norm(all_coefficients),
    )


def test_nonparametric_attention_edge_budget_is_deterministic_and_bounded():
    embeddings = torch.tensor(
        [[0.2, -0.1], [0.7, 0.3], [-0.4, 0.8], [0.5, -0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edges = torch.tensor(
        [[0, 1, 3, 0, 1], [2, 2, 2, 2, 2]], dtype=torch.long
    )
    coefficients, retained = normalized_nonparametric_attention(
        embeddings,
        edges,
        center_nodes=torch.tensor([2]),
        max_retained_edges=3,
    )
    assert torch.equal(retained, torch.tensor([0, 2, 4]))
    raw = torch.stack(
        [
            embeddings[2].dot(torch.tanh(embeddings[0])),
            embeddings[2].dot(torch.tanh(embeddings[3])),
            embeddings[2].dot(torch.tanh(embeddings[1])),
        ]
    )
    assert torch.allclose(coefficients, torch.softmax(raw, dim=0))
    assert topology_attention_squared_norm(
        embeddings,
        edges,
        center_nodes=torch.tensor([2]),
        max_retained_edges=3,
    ).requires_grad


def test_topology_importance_uses_training_query_centers_only():
    first_model = _ToyGraphModel()
    second_model = copy.deepcopy(first_model)
    first_context = _ToyContext(task_id=19)
    second_context = _ToyContext(task_id=19)
    first_context.train_queries = torch.tensor([2], dtype=torch.long)
    first_context.train_labels = torch.tensor([0], dtype=torch.long)
    second_context.train_queries = first_context.train_queries.clone()
    second_context.train_labels = first_context.train_labels.clone()
    # The first three arcs into training center 2 are identical. Only arcs
    # centered on held-out/context-only node 3 differ.
    second_context.effective_edge_index[:, 3:] = torch.tensor(
        [[3, 3, 3], [3, 3, 3]], dtype=torch.long
    )
    first = TWPAlgorithm(lambda_l=0.0, lambda_t=1.0, beta=0.0)
    second = TWPAlgorithm(lambda_l=0.0, lambda_t=1.0, beta=0.0)

    first.consolidate(first_model, first_context)
    second.consolidate(second_model, second_context)

    for name in first.state["topology_importance"][19]:
        assert torch.equal(
            first.state["topology_importance"][19][name],
            second.state["topology_importance"][19][name],
        )


def test_lc_topology_importance_uses_unique_training_edge_endpoints_only():
    first_model = _ToyGraphModel()
    second_model = copy.deepcopy(first_model)
    first_context = _ToyLCContext(task_id=23)
    second_context = _ToyLCContext(task_id=23)
    # Append arcs centered on node 1, which is not an endpoint of query (2, 3).
    first_context.effective_edge_index = torch.cat(
        (
            first_context.effective_edge_index,
            torch.tensor([[0, 2], [1, 1]], dtype=torch.long),
        ),
        dim=1,
    )
    second_context.effective_edge_index = torch.cat(
        (
            second_context.effective_edge_index,
            torch.tensor([[3, 3], [1, 1]], dtype=torch.long),
        ),
        dim=1,
    )
    first = TWPAlgorithm(lambda_l=0.0, lambda_t=1.0, beta=0.0)
    second = TWPAlgorithm(lambda_l=0.0, lambda_t=1.0, beta=0.0)
    first.consolidate(first_model, first_context)
    second.consolidate(second_model, second_context)
    for name in first.state["topology_importance"][23]:
        assert torch.equal(
            first.state["topology_importance"][23][name],
            second.state["topology_importance"][23][name],
        )


def test_consolidation_matches_independent_equations_3_and_6_without_squaring():
    model = _ToyGraphModel()
    reference = copy.deepcopy(model)
    context = _ToyContext(task_id=17)
    algorithm = TWPAlgorithm(lambda_l=3.0, lambda_t=5.0, beta=0.0)

    reference_named = _named_parameters(reference)
    loss = F.cross_entropy(context.forward_queries(reference), context.train_labels)
    expected_loss_gradients = torch.autograd.grad(
        loss, tuple(parameter for _, parameter in reference_named), allow_unused=True
    )
    embeddings = context.twp_project_nodes(reference, layer_index=1)
    topology = topology_attention_squared_norm(
        embeddings,
        context.effective_edge_index,
        center_nodes=context.train_queries,
    )
    expected_topology_gradients = torch.autograd.grad(
        topology,
        tuple(parameter for _, parameter in reference_named),
        allow_unused=True,
    )

    prior_gradients = {}
    for name, parameter in model.named_parameters():
        parameter.grad = torch.full_like(parameter, 0.123)
        prior_gradients[name] = parameter.grad.clone()
    context.encoded_layers.clear()
    algorithm.consolidate(model, context)

    assert context.encoded_layers == [1]
    assert algorithm.state["consolidated_task_ids"] == [17]
    for (name, parameter), loss_gradient, topology_gradient in zip(
        reference_named,
        expected_loss_gradients,
        expected_topology_gradients,
    ):
        expected_loss = (
            torch.zeros_like(parameter)
            if loss_gradient is None
            else loss_gradient.abs()
        )
        expected_topology = (
            torch.zeros_like(parameter)
            if topology_gradient is None
            else topology_gradient.abs()
        )
        assert torch.allclose(
            algorithm.state["loss_importance"][17][name], expected_loss
        )
        assert torch.allclose(
            algorithm.state["topology_importance"][17][name], expected_topology
        )
        assert torch.equal(algorithm.state["anchors"][17][name], parameter.detach())
        assert torch.equal(
            dict(model.named_parameters())[name].grad, prior_gradients[name]
        )

    nonbinary = [
        value
        for value in algorithm.state["loss_importance"][17].values()
        if bool(((value != 0) & (value != 1)).any())
    ]
    assert nonbinary
    assert any(
        gradient is not None
        and not torch.allclose(
            algorithm.state["loss_importance"][17][name], gradient.square()
        )
        for (name, _), gradient in zip(reference_named, expected_loss_gradients)
    )


def test_old_task_penalty_matches_equations_7_and_8_exactly():
    model = _ToyGraphModel()
    context = _ToyContext(task_id=4)
    algorithm = TWPAlgorithm(lambda_l=2.0, lambda_t=7.0, beta=0.0)
    algorithm.consolidate(model, context)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.add_(0.01 * (index + 1))

    logits = context.forward_queries(model)
    base_loss = F.cross_entropy(logits, context.train_labels)
    actual = algorithm.augment_loss(model, context, logits, base_loss)
    expected_penalty = base_loss.new_zeros(())
    for name, parameter in model.named_parameters():
        importance = (
            2.0 * algorithm.state["loss_importance"][4][name]
            + 7.0 * algorithm.state["topology_importance"][4][name]
        )
        expected_penalty = (
            expected_penalty
            + (
                importance * (parameter - algorithm.state["anchors"][4][name]).square()
            ).sum()
        )
    assert torch.allclose(actual, base_loss + expected_penalty)
    assert algorithm.diagnostics()["penalty"] == pytest.approx(
        float(expected_penalty.detach())
    )


def test_beta_topology_importance_keeps_create_graph_and_changes_gradient():
    model = _ToyGraphModel()
    base_model = copy.deepcopy(model)
    context = _ToyContext()
    algorithm = TWPAlgorithm(lambda_l=0.0, lambda_t=2.0, beta=0.3)

    logits = context.forward_queries(model)
    base_loss = F.cross_entropy(logits, context.train_labels)
    augmented = algorithm.augment_loss(model, context, logits, base_loss)
    actual_gradients = torch.autograd.grad(
        augmented, tuple(model.parameters()), allow_unused=True
    )

    base_logits = context.forward_queries(base_model)
    base_only = F.cross_entropy(base_logits, context.train_labels)
    topology = topology_attention_squared_norm(
        context.twp_project_nodes(base_model, layer_index=1),
        context.effective_edge_index,
        center_nodes=context.train_queries,
    )
    topology_gradients = torch.autograd.grad(
        topology,
        tuple(base_model.parameters()),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    current_importance = sum(
        (
            torch.zeros_like(parameter) if gradient is None else 2.0 * gradient.abs()
        ).sum()
        for parameter, gradient in zip(base_model.parameters(), topology_gradients)
    )
    expected = base_only + 0.3 * current_importance
    expected_gradients = torch.autograd.grad(
        expected, tuple(base_model.parameters()), allow_unused=True
    )
    assert torch.allclose(augmented, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        if actual_gradient is None or expected_gradient is None:
            assert actual_gradient is expected_gradient
        else:
            assert torch.allclose(actual_gradient, expected_gradient)

    plain_model = copy.deepcopy(model)
    plain_logits = context.forward_queries(plain_model)
    plain_loss = F.cross_entropy(plain_logits, context.train_labels)
    plain_gradients = torch.autograd.grad(
        plain_loss, tuple(plain_model.parameters()), allow_unused=True
    )
    assert any(
        actual is not None and plain is not None and not torch.allclose(actual, plain)
        for actual, plain in zip(actual_gradients, plain_gradients)
    )


def test_lambda_t_zero_never_calls_topology_capability():
    model = _ToyGraphModel()
    context = _ToyContext(forbid_encoding=True)
    algorithm = TWPAlgorithm(lambda_l=2.0, lambda_t=0.0, beta=0.2)
    logits = context.forward_queries(model)
    base = F.cross_entropy(logits, context.train_labels)
    actual = algorithm.augment_loss(model, context, logits, base)
    assert actual > base
    algorithm.consolidate(model, context)
    assert all(
        torch.count_nonzero(value) == 0
        for value in algorithm.state["topology_importance"][7].values()
    )


def test_all_zero_coefficients_are_an_exact_bare_diagnostic():
    model = _ToyGraphModel()
    context = _ToyContext(forbid_encoding=True)
    algorithm = TWPAlgorithm(lambda_l=0.0, lambda_t=0.0, beta=0.4)
    algorithm.consolidate(model, context)
    logits = context.forward_queries(model)
    base = F.cross_entropy(logits, context.train_labels)
    actual = algorithm.augment_loss(model, context, logits, base)
    assert actual is base
    assert all(
        torch.count_nonzero(value) == 0
        for field in ("loss_importance", "topology_importance")
        for value in algorithm.state[field][7].values()
    )


def test_broadcast_is_private_state_noop_and_global_task_ids_are_immutable():
    model = _ToyGraphModel()
    algorithm = TWPAlgorithm(lambda_l=1.0, lambda_t=1.0, beta=0.0)
    first = _ToyContext(task_id=11)
    algorithm.consolidate(model, first)
    before = algorithm.private_state_checksum()
    algorithm.on_broadcast(
        first, {"shared_state": {"layers.0.weight": torch.ones(2, 2)}}
    )
    assert algorithm.private_state_checksum() == before
    assert algorithm.diagnostics()["pre_broadcast_state_sha256"] == before
    assert algorithm.diagnostics()["post_broadcast_state_sha256"] == before

    second = _ToyContext(task_id=3)
    algorithm.consolidate(model, second)
    assert algorithm.state["consolidated_task_ids"] == [11, 3]
    assert set(algorithm.state["anchors"]) == {11, 3}
    with pytest.raises(ValueError, match="already been consolidated"):
        algorithm.consolidate(model, _ToyContext(task_id=11))


def test_private_state_round_trip_is_owned_strict_and_atomic():
    model = _ToyGraphModel()
    algorithm = TWPAlgorithm(
        lambda_l=2.0,
        lambda_t=3.0,
        beta=0.1,
        middle_layer_index=1,
    )
    algorithm.consolidate(model, _ToyContext(task_id=5))
    algorithm.on_broadcast(_ToyContext(task_id=5), {})
    expected_diagnostics = algorithm.diagnostics()
    saved = algorithm.save_method_state()
    expected_checksum = algorithm.private_state_checksum()

    restored = TWPAlgorithm(
        lambda_l=2.0,
        lambda_t=3.0,
        beta=0.1,
        middle_layer_index=1,
    )
    restored.load_method_state(saved)
    assert restored.private_state_checksum() == expected_checksum
    assert restored.diagnostics() == expected_diagnostics
    saved["private_state"]["anchors"][5]["layers.0.weight"].add_(100.0)
    assert restored.private_state_checksum() == expected_checksum

    malformed = restored.save_method_state()
    malformed["private_state"].pop("topology_importance")
    before_rejected_load = restored.private_state_checksum()
    diagnostics_before_rejected_load = restored.diagnostics()
    with pytest.raises(ValueError, match="checkpoint fields"):
        restored.load_method_state(malformed)
    assert restored.private_state_checksum() == before_rejected_load
    assert restored.diagnostics() == diagnostics_before_rejected_load

    malformed = restored.save_method_state()
    malformed["private_state"]["loss_importance"][5]["layers.0.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="Invalid TWP tensor"):
        restored.load_method_state(malformed)
    assert restored.private_state_checksum() == before_rejected_load

    malformed = restored.save_method_state()
    malformed["diagnostics"]["penalty"] = float("nan")
    with pytest.raises(ValueError, match="diagnostic"):
        restored.load_method_state(malformed)
    assert restored.diagnostics() == diagnostics_before_rejected_load

    mismatched = TWPAlgorithm(
        lambda_l=2.0,
        lambda_t=4.0,
        beta=0.1,
        middle_layer_index=1,
    )
    with pytest.raises(ValueError, match="hyperparameters"):
        mismatched.load_method_state(restored.save_method_state())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_l": -1.0},
        {"lambda_t": float("nan")},
        {"beta": float("inf")},
        {"middle_layer_index": -1},
        {"topology_max_edges": 0},
    ],
)
def test_twp_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        TWPAlgorithm(**kwargs)


def test_middle_layer_fails_closed_when_depth_is_unknown_or_index_is_outside():
    model = _ToyGraphModel()
    context = _ToyContext()
    outside = TWPAlgorithm(lambda_l=0.0, lambda_t=1.0, beta=0.1, middle_layer_index=3)
    logits = context.forward_queries(model)
    base = F.cross_entropy(logits, context.train_labels)
    with pytest.raises(ValueError, match="outside"):
        outside.augment_loss(model, context, logits, base)

    class _UnknownDepth(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 2))

    unknown = _UnknownDepth()
    algorithm = TWPAlgorithm(lambda_l=0.0, lambda_t=1.0, beta=0.1)
    detached_base = (unknown.weight.square()).sum()
    with pytest.raises(RuntimeError, match="cannot infer"):
        algorithm.augment_loss(
            unknown,
            context,
            detached_base.reshape(1),
            detached_base,
        )


def test_twp_uses_real_leakage_safe_context_and_uefa_middle_layer_capability():
    torch.manual_seed(29)
    model = GECKOGraphModel(
        input_size=3,
        output_size=2,
        hidden_size=4,
        num_layers=3,
        problem_type="NC",
    )
    attach_v2_model_capabilities(model)
    features = torch.tensor(
        [
            [0.2, 0.1, -0.4],
            [0.7, -0.3, 0.5],
            [-0.2, 0.8, 0.6],
            [0.9, 0.4, -0.1],
        ]
    )
    edges = torch.tensor([[0, 1, 3, 0, 1, 2], [2, 2, 2, 3, 3, 3]], dtype=torch.long)
    context = ClientMethodContext(
        client_id=1,
        global_task_id=23,
        stage_index=0,
        round_index=0,
        problem_type="NC",
        incremental_setting="class",
        train_queries=torch.tensor([0, 1, 2]),
        train_labels=torch.tensor([0, 1, 0]),
        valid_class_mask=torch.tensor([True, True]),
        node_features=features,
        base_edge_index=edges,
        context_edge_index=None,
        forward_queries=lambda local_model, x, arcs, roots: local_model.forward_queries(
            x, arcs, roots, "NC"
        ),
        encode_nodes=lambda local_model, x, arcs, layer: encode_model_nodes(
            local_model, x, arcs, layer_index=layer
        ),
    )
    reference = copy.deepcopy(model)
    previous_hidden = F.relu(reference.layers[0](features, edges))
    expected_projection = reference.layers[1].neighbor_linear(previous_hidden)
    direct_projection = project_model_nodes_for_twp(
        model,
        features,
        edges,
        layer_index=1,
    )
    context_projection = context.twp_project_nodes(model, layer_index=1)
    assert torch.allclose(direct_projection, expected_projection)
    assert torch.allclose(context_projection, expected_projection)

    reference_parameters = _named_parameters(reference)
    expected_topology = topology_attention_squared_norm(
        expected_projection,
        edges,
        center_nodes=context.train_queries,
    )
    expected_topology_gradients = torch.autograd.grad(
        expected_topology,
        tuple(parameter for _, parameter in reference_parameters),
        allow_unused=True,
    )
    algorithm = TWPAlgorithm(lambda_l=1.0, lambda_t=1.0, beta=0.05)
    logits = context.forward_queries(model)
    base = F.cross_entropy(logits, context.train_labels)
    augmented = algorithm.augment_loss(model, context, logits, base)
    augmented.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert algorithm.diagnostics()["resolved_middle_layer_index"] == 1

    model.zero_grad(set_to_none=True)
    algorithm.consolidate(model, context)
    assert algorithm.state["consolidated_task_ids"] == [23]
    for (name, parameter), expected_gradient in zip(
        reference_parameters,
        expected_topology_gradients,
    ):
        expected_importance = (
            torch.zeros_like(parameter)
            if expected_gradient is None
            else expected_gradient.abs()
        )
        assert torch.allclose(
            algorithm.state["topology_importance"][23][name], expected_importance
        )
    assert not hasattr(context, "ownership_map")
    assert not hasattr(context, "test_labels")


def test_twp_context_moves_projection_inputs_to_the_model_device():
    class DeviceCheckingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(1, device="meta"))

        def twp_project_nodes(
            self,
            features: torch.Tensor,
            edge_index: torch.Tensor,
            *,
            layer_index: int,
        ) -> torch.Tensor:
            assert features.device == self.weight.device
            assert edge_index.device == self.weight.device
            assert layer_index == 0
            return features

    context = ClientMethodContext(
        client_id=0,
        global_task_id=0,
        stage_index=0,
        round_index=0,
        problem_type="NC",
        incremental_setting="class",
        train_queries=torch.tensor([0]),
        train_labels=torch.tensor([0]),
        valid_class_mask=torch.tensor([True]),
        node_features=torch.ones(2, 2),
        base_edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        context_edge_index=None,
        forward_queries=lambda model, features, edges, roots: features[roots],
        encode_nodes=lambda model, features, edges, layer: features,
    )
    projected = context.twp_project_nodes(DeviceCheckingModel(), layer_index=0)
    assert projected.device.type == "meta"


def test_begin_gcn_twp_projection_uses_torch_graphconv_path(monkeypatch):
    try:
        model = LegacyGCNAdapter(
            input_size=3,
            output_size=2,
            hidden_size=4,
            num_layers=3,
            problem_type="NC",
            incremental_type="task",
        )
    except (ImportError, OSError, RuntimeError) as error:
        pytest.skip(f"begin_gcn dependencies are unavailable: {error}")
    model.eval()
    features = torch.randn(4, 3)
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [0, 0, 1, 1, 2, 2, 3, 3]],
        dtype=torch.long,
    )

    def reject_dgl_graph(*_args, **_kwargs):
        raise AssertionError("TWP projection must not invoke CUDA-incompatible DGL")

    monkeypatch.setattr(model, "_graph", reject_dgl_graph)
    projected = project_model_nodes_for_twp(
        model, features, edges, layer_index=1
    )
    projected.square().sum().backward()

    assert projected.shape == (4, 4)
    assert model.original.convs[0].weight.grad is not None
    assert model.original.convs[1].weight.grad is not None


def test_begin_gcn_twp_projection_and_topology_gradient_match_equation_10():
    try:
        __import__("dgl")
        model = LegacyGCNAdapter(
            input_size=3,
            output_size=2,
            hidden_size=4,
            num_layers=3,
            problem_type="NC",
            incremental_type="class",
        )
        reference = LegacyGCNAdapter(
            input_size=3,
            output_size=2,
            hidden_size=4,
            num_layers=3,
            problem_type="NC",
            incremental_type="class",
        )
    except (ImportError, OSError, RuntimeError) as error:
        pytest.skip(f"begin_gcn dependencies are unavailable: {error}")

    torch.manual_seed(41)
    reference.load_state_dict(model.state_dict(), strict=True)
    attach_v2_model_capabilities(model)
    model.eval()
    reference.eval()
    features = torch.tensor(
        [
            [0.2, 0.1, -0.4],
            [0.7, -0.3, 0.5],
            [-0.2, 0.8, 0.6],
            [0.9, 0.4, -0.1],
        ]
    )
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [0, 0, 1, 1, 2, 2, 3, 3]],
        dtype=torch.long,
    )
    context = ClientMethodContext(
        client_id=0,
        global_task_id=31,
        stage_index=0,
        round_index=0,
        problem_type="NC",
        incremental_setting="task",
        train_queries=torch.tensor([0, 1, 2]),
        train_labels=torch.tensor([0, 1, 0]),
        valid_class_mask=torch.tensor([True, True]),
        node_features=features,
        base_edge_index=edges,
        context_edge_index=None,
        forward_queries=lambda local_model, x, arcs, roots: local_model.forward_queries(
            x, arcs, roots, "NC"
        ),
        encode_nodes=lambda local_model, x, arcs, layer: encode_model_nodes(
            local_model, x, arcs, layer_index=layer
        ),
    )

    encoder = reference.original
    graph = reference._graph(edges, features.shape[0])
    previous_hidden = encoder.dropout(features)
    previous_hidden = encoder.convs[0](graph, previous_hidden)
    previous_hidden = encoder.norms[0](previous_hidden)
    previous_hidden = encoder.activation(previous_hidden)
    previous_hidden = encoder.dropout(previous_hidden)
    expected_projection = torch.matmul(previous_hidden, encoder.convs[1].weight)
    actual_projection = context.twp_project_nodes(model, layer_index=1)
    assert torch.allclose(actual_projection, expected_projection)

    reference_parameters = _named_parameters(reference)
    expected_topology = topology_attention_squared_norm(
        expected_projection,
        edges,
        center_nodes=context.train_queries,
    )
    expected_gradients = torch.autograd.grad(
        expected_topology,
        tuple(parameter for _, parameter in reference_parameters),
        allow_unused=True,
    )
    algorithm = TWPAlgorithm(
        lambda_l=0.0,
        lambda_t=1.0,
        beta=0.0,
        middle_layer_index=1,
    )
    algorithm.consolidate(model, context)
    for (name, parameter), expected_gradient in zip(
        reference_parameters,
        expected_gradients,
    ):
        expected_importance = (
            torch.zeros_like(parameter)
            if expected_gradient is None
            else expected_gradient.abs()
        )
        assert torch.allclose(
            algorithm.state["topology_importance"][31][name],
            expected_importance,
        )
