from __future__ import annotations

import pytest
import torch

from gecko.models.cuda_legacy_gcn import forward_nc_torch_graphconv
from gecko.models.cuda_legacy_gcn import install_nc_cuda_graphconv_compatibility
from gecko.models.backbones import LegacyGCNAdapter


def _model() -> LegacyGCNAdapter:
    torch.manual_seed(17)
    model = LegacyGCNAdapter(
        input_size=4,
        output_size=3,
        hidden_size=5,
        num_layers=2,
        problem_type="NC",
        incremental_type="class",
    )
    model.eval()
    return model


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
    # Bidirected path plus an isolated node exercises clamp_min(1).
    edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    return features, edges


def test_native_torch_graphconv_matches_legacy_dgl_cpu_forward():
    model = _model()
    features, edges = _fixture()
    graph = model._graph(edges, features.shape[0])
    expected = model.original(graph, features)
    actual = forward_nc_torch_graphconv(model.original, features, edges)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.gpu
def test_begin_gcn_cuda_path_runs_forward_and_backward_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the compatibility-path test.")
    install_nc_cuda_graphconv_compatibility()
    model = _model().cuda()
    features, edges = _fixture()
    features = features.cuda().requires_grad_(True)
    edges = edges.cuda()
    queries = torch.arange(features.shape[0], device="cuda")
    logits = model.forward_queries(features, edges, queries, "NC")
    assert logits.is_cuda
    assert logits.shape == (6, 3)
    logits.square().mean().backward()
    assert features.grad is not None
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

