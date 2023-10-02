from torch import nn, Tensor

from typing import List, Tuple
from timm.models.layers import to_3tuple

from ..builder import NECK


@NECK.register()
class MultipleCenterCrop(nn.Module):
    def __init__(self, crop_sizes: List[int]):
        super().__init__()
        self.crop_sizes: List[Tuple[int, int, int]] = [to_3tuple(size) for size in crop_sizes]

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        outputs: List[Tensor] = []
        for size, tensor in zip(self.crop_sizes, inputs):
            Dc, Hc, Wc = size
            _, _, D, H, W = tensor.size()
            d, h, w = (D - Dc) // 2, (H - Hc) // 2, (W - Wc) // 2
            assert d >= 0 and h >= 0 and w >= 0
            if d == 0 and h == 0 and w == 0:
                outputs.append(tensor)
            else:
                outputs.append(tensor[:, :, d:D - d, h:H - h, w:W - w].contiguous())
        return outputs

    def initialize_weights(self):
        pass


@NECK.register()
class CenterCrop(nn.Module):
    def __init__(self, crop_size: int):
        super().__init__()
        self.crop_size: Tuple[int, int, int] = to_3tuple(crop_size)

    def forward(self, input: Tensor) -> Tensor:
        Dc, Hc, Wc = self.crop_size
        _, _, D, H, W = input.size()
        d, h, w = (D - Dc) // 2, (H - Hc) // 2, (W - Wc) // 2
        assert d >= 0 and h >= 0 and w >= 0
        if d != 0 or h != 0 or w != 0:
            return input[:, :, d:D - d, h:H - h, w:W - w].contiguous()
        else:
            return input

    def initialize_weights(self):
        pass
