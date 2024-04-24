import torch
from torch import nn
from torch.nn import functional as F

from typing import Tuple, Union, List
from torch import Tensor


def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


def to_3tuple(value):
    if not isinstance(value, tuple):
        return (value, value, value)
    else:
        return value


class GaussianSmoothing(nn.Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        sigma: Union[float, Tuple[float, float, float]],
    ):
        super(GaussianSmoothing, self).__init__()
        kernel_size = to_3tuple(kernel_size)
        sigma = to_3tuple(sigma)

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float) for size in kernel_size], indexing='ij'
        )
        kernel: torch.Tensor = None
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            # _kernel = 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
            _kernel = torch.exp(-((mgrid - mean) / (std)) ** 2 / 2)     # omit constant part
            if kernel is None:
                kernel = _kernel
            else:
                kernel *= _kernel

        # Make sure sum of values in gaussian kernel equals 1.
        kernel /= torch.sum(kernel)                     # (Kd, Kh, Kw), Kd, Kh, Kw: kernel_size

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())      # (1, 1, Kd, Kh, Kw)
        kernel = kernel.repeat(1, 1, 1, 1, 1)           # (1, 1, Kd, Kh, Kw)

        self.register_buffer('weight', kernel)
        self.pad: Tuple[int, int, int, int, int, int] = \
            (kernel_size[0] // 2, kernel_size[0] // 2,
             kernel_size[1] // 2, kernel_size[1] // 2,
             kernel_size[2] // 2, kernel_size[2] // 2,)

    @torch.no_grad()
    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        x = F.pad(x, self.pad, mode='constant', value=0.)
        weight = self.weight.repeat(x.shape[-4], 1, 1, 1, 1)
        return F.conv3d(x, weight=weight, groups=x.shape[-4])
