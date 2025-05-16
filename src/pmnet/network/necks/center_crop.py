from torch import Tensor, nn


class MultipleCenterCrop(nn.Module):
    def __init__(self, crop_sizes: list[int]):
        super().__init__()
        self.crop_sizes: list[tuple[int, int, int]] = [(size, size, size) for size in crop_sizes]

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        outputs: list[Tensor] = []
        for size, tensor in zip(self.crop_sizes, inputs, strict=True):
            Dc, Hc, Wc = size
            _, _, D, H, W = tensor.size()
            d, h, w = (D - Dc) // 2, (H - Hc) // 2, (W - Wc) // 2
            assert d >= 0 and h >= 0 and w >= 0
            if d == 0 and h == 0 and w == 0:
                outputs.append(tensor)
            else:
                outputs.append(tensor[:, :, d : D - d, h : H - h, w : W - w].contiguous())
        return outputs

    def initialize_weights(self):
        pass


class CenterCrop(nn.Module):
    def __init__(self, crop_size: int):
        super().__init__()
        self.crop_size: tuple[int, int, int] = (crop_size, crop_size, crop_size)

    def forward(self, input: Tensor) -> Tensor:
        Dc, Hc, Wc = self.crop_size
        _, _, D, H, W = input.size()
        d, h, w = (D - Dc) // 2, (H - Hc) // 2, (W - Wc) // 2
        assert d >= 0 and h >= 0 and w >= 0
        if d != 0 or h != 0 or w != 0:
            return input[:, :, d : D - d, h : H - h, w : W - w].contiguous()
        else:
            return input

    def initialize_weights(self):
        pass
