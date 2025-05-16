from __future__ import annotations

import torch
from torch import nn


class OneHotEncoding(nn.Module):
    def __init__(
        self,
        bin_min: float = 0.0,
        bin_max: float = 15.0,
        num_classes: int = 16,
        rounding_mode: str = "floor",
    ):
        super().__init__()
        assert num_classes > 1
        self.bin_min: float = bin_min
        self.bin_size: int = int((bin_max - bin_min) / (num_classes - 1))
        self.bin_max: float = bin_max + (self.bin_size / 2)  # to prevent float error.
        self.num_classes: int = num_classes
        self.rounding_mode: str = rounding_mode

    def forward(self, x) -> torch.Tensor:
        x = x.clip(self.bin_min, self.bin_max)
        idx = torch.div(x - self.bin_min, self.bin_size, rounding_mode=self.rounding_mode).long()
        out = torch.nn.functional.one_hot(idx, num_classes=self.num_classes).float()
        return out
