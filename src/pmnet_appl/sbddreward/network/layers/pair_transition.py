from __future__ import annotations

from torch import nn


class PairTransition(nn.Module):
    def __init__(self, c_hidden, expand: int = 4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_hidden)
        self.linear_1 = nn.Linear(c_hidden, expand * c_hidden)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(expand * c_hidden, c_hidden)

    def forward(self, z, mask):
        z = self.layer_norm(z)
        z = self.linear_1(z)
        z = self.relu(z)
        z = self.linear_2(z)
        z = z * mask.unsqueeze(-1)
        return z
