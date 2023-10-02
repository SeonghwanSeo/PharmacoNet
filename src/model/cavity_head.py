from torch import nn

from functools import partial

from typing import Tuple, Type, Optional
from torch import Tensor

from .builder import HEAD
from .nn import BaseConv3d


@HEAD.register()
class CavityHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        norm_layer: Optional[Type[nn.Module]] = nn.BatchNorm3d,
        act_layer: Optional[Type[nn.Module]] = partial(nn.ReLU, inplace=True),
    ):
        super(CavityHead, self).__init__()

        self.short_head = nn.Sequential(
            BaseConv3d(feature_dim, hidden_dim, kernel_size=3, norm_layer=norm_layer, act_layer=act_layer),
            BaseConv3d(hidden_dim, 1, kernel_size=1, norm_layer=None, act_layer=None),
        )
        self.long_head = nn.Sequential(
            BaseConv3d(feature_dim, hidden_dim, kernel_size=3, norm_layer=norm_layer, act_layer=act_layer),
            BaseConv3d(hidden_dim, 1, kernel_size=1, norm_layer=None, act_layer=None),
        )

    def initialize_weights(self):
        for m in self.short_head.children():
            m.initialize_weights()
        for m in self.long_head.children():
            m.initialize_weights()

    def forward(
        self,
        features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Pocket Extraction Function

        Args:
            features: FloatTensor [N, F, D, H, W]

        Returns:
            focus_area_short: FloatTensor [N, 1, D, H, W]
            focus_area_long: FloatTensor [N, 1, D, H, W]
        """
        focus_area_short = self.short_head(features)
        focus_area_long = self.long_head(features)
        return focus_area_short, focus_area_long
