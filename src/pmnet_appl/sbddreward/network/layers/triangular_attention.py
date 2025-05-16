from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class TriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, num_heads, inf=1e9):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            num_heads:
                Number of attention heads
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.inf = inf

        self.layer_norm = nn.LayerNorm(self.c_in)
        self.mha = Attention(self.c_in, self.c_in, self.c_in, self.c_hidden, self.num_heads)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        else:
            mask = mask.float()

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        biases = []

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        biases.append(mask_bias)

        # [*, H, I, J]
        # triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # [*, 1, H, I, J]
        # triangle_bias = triangle_bias.unsqueeze(-4)
        # biases.append(triangle_bias)

        x = self.mha(q_x=x, kv_x=x, biases=biases)

        return x


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        num_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q: Input dimension of query data
            c_k: Input dimension of key data
            c_v: Input dimension of value data
            c_hidden: Per-head hidden dimension
            num_heads: Number of attention heads
            gating: Whether the output should be gated using query data
        """
        super().__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.num_heads = num_heads

        self.linear_q = nn.Linear(self.c_q, self.c_hidden * self.num_heads, bias=False)
        self.linear_k = nn.Linear(self.c_k, self.c_hidden * self.num_heads, bias=False)
        self.linear_v = nn.Linear(self.c_v, self.c_hidden * self.num_heads, bias=False)

        self.linear_o = nn.Linear(self.c_hidden * self.num_heads, self.c_q)
        if gating:
            self.linear_g = nn.Linear(self.c_q, self.c_hidden * self.num_heads)
        else:
            self.linear_g = None

        self.sigmoid = nn.Sigmoid()

    def init_weight(self):
        for module in [self.linear_q, self.linear_k, self.linear_v]:
            nn.init.xavier_uniform_(module.weight, gain=1)
        with torch.no_grad():
            for module in [self.linear_o, self.linear_g]:
                if module is not None:
                    module.weight.fill_(0.0)

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        k = k.view(k.shape[:-1] + (self.num_heads, -1))
        v = v.view(v.shape[:-1] + (self.num_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.num_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        q, k, v = self._prep_qkv(q_x, kv_x)
        if biases is None:
            biases = []
        o = attention(q, k, v, biases)
        o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, [1, 0])

    # [*, H, Q, K]
    a = torch.matmul(query, key)
    for b in biases:
        a = a + b
    a = a.softmax(-1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


def permute_final_dims(tensor: torch.Tensor, inds: Sequence[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])
