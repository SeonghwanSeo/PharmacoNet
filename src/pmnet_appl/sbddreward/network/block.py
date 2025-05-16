from __future__ import annotations

import torch
from torch import nn

from .layers.pair_transition import PairTransition
from .layers.triangular_attention import TriangleAttention
from .layers.triangular_multiplicative_update import DirectTriangleMultiplicativeUpdate


class ComplexFormerBlock(nn.Module):
    def __init__(
        self,
        c_hidden: int,
        c_head: int,
        n_heads: int,
        n_transition: int,
        dropout: float,
    ):
        super().__init__()
        self.tri_mul_update = DirectTriangleMultiplicativeUpdate(c_hidden, c_hidden)
        self.tri_attention = TriangleAttention(c_hidden, c_head, n_heads)
        self.transition = PairTransition(c_hidden, n_transition)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(
        self,
        z_complex: torch.Tensor,
        zpair_protein: torch.Tensor,
        mask_complex: torch.Tensor,
    ) -> torch.Tensor:
        z_complex = z_complex + self.dropout(self.tri_mul_update.forward(z_complex, zpair_protein, mask_complex))
        z_complex = z_complex + self.dropout(self.tri_attention.forward(z_complex, mask_complex))
        z_complex = self.transition.forward(z_complex, mask_complex)
        return z_complex
