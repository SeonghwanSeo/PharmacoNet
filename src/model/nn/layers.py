from torch import nn

from typing import Optional, Type


class BaseConv3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = None,
            dilation: int = 1,
            groups: int = 1,
            norm_layer: Optional[Type[nn.Module]] = nn.BatchNorm3d,
            act_layer: Optional[Type[nn.Module]] = nn.ReLU,
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        bias = (norm_layer is None)
        self._conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self._norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        self._act = act_layer() if act_layer is not None else nn.Identity()

    def initialize_weights(self):
        if isinstance(self._act, nn.LeakyReLU):
            a = self._act.negative_slope
            nn.init.kaiming_normal_(self._conv.weight, a, mode='fan_out', nonlinearity='leaky_relu')
        else:
            nn.init.kaiming_normal_(self._conv.weight, mode='fan_out', nonlinearity='relu')
        if self._conv.bias is not None:
            nn.init.constant_(self._conv.bias, 0.)
        if not isinstance(self._norm, nn.Identity):
            nn.init.constant_(self._norm.weight, 1.0)

    def forward(self, x):
        return self._act(self._norm(self._conv(x)))
