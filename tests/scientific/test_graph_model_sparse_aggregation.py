from __future__ import annotations

import copy

import torch

from gecko.models.backbones import MeanGraphLayer


def _dense_edge_gather_reference(
    layer: MeanGraphLayer,
    features: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    source, target = edge_index
    aggregate = torch.zeros_like(features)
    degree = torch.zeros(features.shape[0], dtype=features.dtype)
    aggregate.index_add_(0, target, features[source])
    degree.index_add_(0, target, torch.ones_like(target, dtype=features.dtype))
    aggregate = aggregate / degree.clamp_min(1).unsqueeze(1)
    return layer.self_linear(features) + layer.neighbor_linear(aggregate)


def test_sparse_mean_graph_layer_matches_edge_gather_outputs_and_gradients():
    torch.manual_seed(7)
    sparse_layer = MeanGraphLayer(4, 3)
    reference_layer = copy.deepcopy(sparse_layer)
    sparse_features = torch.randn(6, 4, requires_grad=True)
    reference_features = sparse_features.detach().clone().requires_grad_(True)
    # Includes duplicate arcs and a zero-in-degree node.
    edges = torch.tensor(
        [[0, 0, 1, 2, 2, 4], [1, 1, 2, 0, 3, 3]],
        dtype=torch.long,
    )

    sparse_output = sparse_layer(sparse_features, edges)
    reference_output = _dense_edge_gather_reference(
        reference_layer,
        reference_features,
        edges,
    )
    assert torch.allclose(sparse_output, reference_output)

    sparse_output.square().sum().backward()
    reference_output.square().sum().backward()
    assert torch.allclose(sparse_features.grad, reference_features.grad)
    for sparse_parameter, reference_parameter in zip(
        sparse_layer.parameters(),
        reference_layer.parameters(),
    ):
        assert torch.allclose(sparse_parameter.grad, reference_parameter.grad)
