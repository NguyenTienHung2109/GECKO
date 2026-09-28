"""Buffer-free GAT matching the model profile used by FedFST."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FedFSTGAT(nn.Module):
    """Two-layer, one-head PyG GAT used by the FedFST reference code.

    The paper-linked implementation uses ``GATConv`` with its defaults: one
    head, concatenated output, learned bias, and automatically added self
    loops.  It applies ReLU and dropout only between graph-attention layers.
    This adapter deliberately contains no persistent buffers, so a server
    parameter snapshot completely determines its evaluation state.
    """

    paper_hidden_size = 64
    paper_num_layers = 2
    paper_dropout = 0.5

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        problem_type: str = "NC",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GATConv
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError(
                "fedfst_gat requires the pinned torch-geometric dependency."
            ) from error
        if str(problem_type).upper() != "NC":
            raise ValueError("fedfst_gat supports node classification only.")
        if (
            hidden_size != self.paper_hidden_size
            or num_layers != self.paper_num_layers
        ):
            raise ValueError(
                "fedfst_gat is a fixed paper profile: hidden_size=64 and num_layers=2."
            )
        if dropout != self.paper_dropout:
            raise ValueError("fedfst_gat is a fixed paper profile: dropout=0.5.")
        self.problem_type = "NC"
        self.dropout = float(dropout)
        self.layers = nn.ModuleList(
            (
                GATConv(input_size, hidden_size),
                GATConv(hidden_size, output_size),
            )
        )

    def encode(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return the exact hidden tensor consumed by the output GAT layer."""

        hidden = F.relu(self.layers[0](features, edge_index))
        return F.dropout(hidden, p=self.dropout, training=self.training)

    def encode_nodes(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        layer_index: int | None = None,
    ) -> torch.Tensor:
        """Expose the paper model's sole hidden representation."""

        if layer_index not in (None, 0, -1):
            raise ValueError("fedfst_gat exposes only its hidden GAT layer.")
        return self.encode(features, edge_index)

    def forward_node(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers[-1](self.encode(features, edge_index), edge_index)

    def forward_queries(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        queries: torch.Tensor,
        problem_type: str,
    ) -> torch.Tensor:
        if str(problem_type).upper() != "NC":
            raise ValueError("fedfst_gat supports node classification only.")
        return self.forward_node(features, edge_index)[queries]

    def mask_output_gradients(self, class_mask: torch.Tensor) -> None:
        """Keep unseen output coordinates fixed under UEFA class masking."""

        if (
            not torch.is_tensor(class_mask)
            or class_mask.dtype != torch.bool
            or class_mask.ndim != 1
        ):
            raise ValueError("class_mask must be a one-dimensional bool tensor.")
        output = self.layers[-1]
        if output.out_channels != class_mask.numel() or output.heads != 1:
            raise RuntimeError("fedfst_gat output width does not match the class mask.")
        projection = getattr(output, "lin", None)
        if projection is None:
            projection = getattr(output, "lin_src", None)
        if projection is None:
            raise RuntimeError(
                "Unsupported torch-geometric GATConv projection layout."
            )
        inactive = ~class_mask.to(projection.weight.device)
        if projection.weight.grad is not None:
            projection.weight.grad[inactive] = 0
        if output.bias is not None and output.bias.grad is not None:
            output.bias.grad[inactive] = 0
        for attention in (output.att_src, output.att_dst):
            if attention.grad is not None:
                attention.grad[..., inactive] = 0
