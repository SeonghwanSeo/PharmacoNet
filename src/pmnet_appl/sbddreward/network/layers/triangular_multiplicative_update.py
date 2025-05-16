from __future__ import annotations

import torch
from torch import nn


class DirectTriangleMultiplicativeUpdate(nn.Module):
    def __init__(self, c_in: int, c_hidden: int):
        super().__init__()
        self.linear_b = nn.Linear(c_in, c_hidden)
        self.linear_b_g = nn.Sequential(nn.Linear(c_in, c_hidden), nn.Sigmoid())

        self.layernorm_z = nn.LayerNorm(c_in)
        self.linear_z = nn.Linear(c_in, c_hidden)
        self.linear_z_g = nn.Sequential(nn.Linear(c_in, c_hidden), nn.Sigmoid())

        self.linear_o = nn.Linear(c_hidden, c_in)
        self.linear_o_g = nn.Sequential(nn.Linear(c_hidden, c_in), nn.Sigmoid())

    def forward(self, z, b, z_mask):
        """
        z: [N, A, B, C]
        b: [N, B, B, C]
        z_mask: [N, A, B]
        a -> b
        """
        b = self.linear_b(b) * self.linear_b_g(b)
        z = self.layernorm_z(z)
        _z = self.linear_z(z) * self.linear_z_g(z) * z_mask.unsqueeze(-1)

        message = torch.einsum("bikc,bjkc->bijc", _z, b)

        z = self.linear_o_g(z) * self.linear_o(message) * z_mask.unsqueeze(-1)
        return z
