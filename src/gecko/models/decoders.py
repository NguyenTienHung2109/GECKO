"""Reusable link decoders."""

from __future__ import annotations

import torch
from torch import nn


class DotProductDecoder(nn.Module):
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (source * target).sum(dim=-1)


class ProductMLPDecoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.linear(source * target)
