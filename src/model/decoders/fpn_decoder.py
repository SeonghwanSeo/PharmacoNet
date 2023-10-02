import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Optional, Type, List
from torch import Tensor

from ..nn.layers import BaseConv3d
from ..builder import DECODER


@DECODER.register()
class FPNDecoder(nn.Module):
    """
    Modified FPN Structure [https://arxiv.org/abs/1807.10221]
    feature_channels: Bottom-Up Manner.
    """

    def __init__(
        self,
        feature_channels: Sequence[int],
        num_convs: Sequence[int],
        channels: int = 64,
        interpolate_mode: str = 'nearest',
        align_corners: bool = False,
        norm_layer: Optional[Type[nn.Module]] = nn.BatchNorm3d,
        act_layer: Optional[Type[nn.Module]] = nn.ReLU,
    ):
        super(FPNDecoder, self).__init__()
        self.feature_channels = feature_channels
        self.interpolate_mode = interpolate_mode
        if interpolate_mode == 'trilinear':
            self.align_corners = align_corners
        else:
            self.align_corners = None
        self.channels = channels

        lateral_conv_list = []
        fpn_convs_list = []
        for level, (channels, num_conv) in enumerate(zip(self.feature_channels, num_convs)):
            if level == (len(self.feature_channels) - 1):  # Lowest-Resolution Channels (Top)
                lateral_conv = nn.Identity()
                fpn_convs = nn.Sequential(*[
                    BaseConv3d(
                        channels if i == 0 else self.channels, self.channels,
                        kernel_size=3, norm_layer=norm_layer, act_layer=act_layer,
                    ) for i in range(num_conv)
                ])
            else:
                lateral_conv = BaseConv3d(channels, self.channels, kernel_size=1, norm_layer=norm_layer, act_layer=act_layer)
                fpn_convs = nn.Sequential(*[
                    BaseConv3d(
                        self.channels, self.channels,
                        kernel_size=3, norm_layer=norm_layer, act_layer=act_layer,
                    ) for _ in range(num_conv)
                ])
            lateral_conv_list.append(lateral_conv)
            fpn_convs_list.append(fpn_convs)

        self.lateral_conv_list = nn.ModuleList(lateral_conv_list)
        self.fpn_convs_list = nn.ModuleList(fpn_convs_list)

    def initialize_weights(self):
        for m in self.lateral_conv_list:
            if isinstance(m, BaseConv3d):
                m.initialize_weights()
        for seqm in self.fpn_convs_list:
            for m in seqm.children():
                m.initialize_weights()

    def forward(self, features: Sequence[Tensor]) -> List[Tensor]:
        """Forward function.
        Args:
            features: Bottom-Up, [Highest-Resolution Feature Map, ..., Lowest-Resolution Feature Map]
        Returns:
            features: Top-Down, [Lowest-Resolution Feature Map, ..., Highest-Resolution Feature Map]
        """
        num_levels = len(features)
        assert num_levels == len(self.feature_channels)
        fpn = None
        multi_scale_features = []
        for level in range(num_levels - 1, -1, -1):
            feature = features[level]
            lateral_conv = self.lateral_conv_list[level]
            fpn_convs = self.fpn_convs_list[level]
            current_fpn = lateral_conv(feature)
            if level == (num_levels - 1):    # Top
                assert fpn is None
                fpn = current_fpn
            else:
                assert fpn is not None
                fpn = current_fpn + F.interpolate(fpn, size=current_fpn.size()[-3:], mode=self.interpolate_mode, align_corners=self.align_corners)
            fpn = fpn_convs(fpn)
            multi_scale_features.append(fpn)
        return multi_scale_features
