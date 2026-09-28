from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from gecko.engine.aggregation import weighted_average
from gecko.models.fedfst_gat import FedFSTGAT
from gecko.models.registry import ModelRegistry
from gecko.types import LocalUpdateResult


class _PaperLinkedGAT(nn.Module):
    """Independent executable form of the paper-linked model contract."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            (GATConv(input_size, 64), GATConv(64, output_size))
        )

    def forward(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, (edge_index, _) = self.layers[0](
            features, edge_index, return_attention_weights=True
        )
        hidden = F.dropout(F.relu(hidden), p=0.5, training=self.training)
        logits, _ = self.layers[1](
            hidden, edge_index, return_attention_weights=True
        )
        return hidden, logits


@pytest.mark.parametrize("training", (False, True))
def test_fedfst_gat_matches_paper_linked_pyg_forward(training: bool) -> None:
    torch.manual_seed(17)
    oracle = _PaperLinkedGAT(5, 7)
    model = FedFSTGAT(5, 7)
    model.load_state_dict(oracle.state_dict(), strict=True)
    features = torch.randn(11, 5)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 3, 7, 8, 9], [1, 2, 0, 4, 5, 8, 9, 7]],
        dtype=torch.long,
    )
    oracle.train(training)
    model.train(training)

    torch.manual_seed(91)
    expected_hidden, expected_logits = oracle(features, edge_index)
    torch.manual_seed(91)
    actual_hidden = model.encode(features, edge_index)
    torch.manual_seed(91)
    actual_logits = model.forward_node(features, edge_index)

    assert torch.equal(actual_hidden, expected_hidden)
    assert torch.equal(actual_logits, expected_logits)


def test_fedfst_gat_snapshot_is_complete_and_registry_is_promoted() -> None:
    registry = ModelRegistry()
    entry = registry.entry("fedfst_gat")
    model = registry.create(
        "fedfst_gat", input_size=4, output_size=6, problem_type="NC"
    )
    assert tuple(model.named_buffers()) == ()
    assert entry.has_factory is True
    assert entry.runnable_by_default is True
    assert entry.benchmark_eligible is True
    assert entry.release_status == "supported"
    restored = copy.deepcopy(model)
    restored.load_state_dict(model.state_dict(), strict=True)
    features = torch.randn(8, 4)
    edges = torch.tensor([[0, 1, 2, 4], [1, 2, 0, 5]], dtype=torch.long)
    model.eval()
    restored.eval()
    assert torch.equal(
        model.forward_node(features, edges), restored.forward_node(features, edges)
    )


def test_fedfst_gat_forward_backward_and_parameter_aggregation_are_safe() -> None:
    model = FedFSTGAT(4, 6)
    features = torch.randn(9, 4)
    edges = torch.tensor([[0, 1, 2, 6], [1, 2, 0, 7]], dtype=torch.long)
    queries = torch.tensor([0, 3, 8])
    logits = model.forward_queries(features, edges, queries, "NC")
    logits.square().mean().backward()
    assert logits.shape == (3, 6)
    assert all(parameter.grad is not None for parameter in model.parameters())
    model.mask_output_gradients(
        torch.tensor([True, True, True, False, False, False])
    )
    output = model.layers[-1]
    projection = getattr(output, "lin", None)
    if projection is None:
        projection = output.lin_src
    assert torch.count_nonzero(projection.weight.grad[3:]) == 0
    assert torch.count_nonzero(output.att_src.grad[..., 3:]) == 0
    assert torch.count_nonzero(output.att_dst.grad[..., 3:]) == 0

    states = []
    for client_id, delta in enumerate((0.0, 0.25)):
        state = {
            name: value.detach().clone() + delta
            for name, value in model.named_parameters()
        }
        states.append(
            LocalUpdateResult(
                client_id, 0, state, 1, 0.0, 0, tuple(state)
            )
        )
    averaged = weighted_average(states)
    for name, parameter in model.named_parameters():
        assert torch.allclose(averaged[name], parameter.detach() + 0.125)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fedfst_gat_cuda_forward_backward_has_no_host_buffer() -> None:
    model = FedFSTGAT(4, 6).cuda()
    features = torch.randn(9, 4, device="cuda")
    edges = torch.tensor([[0, 1, 2, 6], [1, 2, 0, 7]], device="cuda")
    logits = model.forward_queries(features, edges, torch.tensor([0, 3, 8], device="cuda"), "NC")
    logits.square().mean().backward()
    assert logits.is_cuda
    assert tuple(model.named_buffers()) == ()
