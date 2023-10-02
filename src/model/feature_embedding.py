from torch import nn

from typing import Sequence, Optional
from torch import Tensor

from .builder import EMBEDDING


@EMBEDDING.register()
class FeaturePyramidNetwork(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        decoder: nn.Module,
        neck: Optional[nn.Module] = None,
        feature_indices: Optional[Sequence[int]] = None,
        set_input_to_bottom: bool = True,
    ):
        super(FeaturePyramidNetwork, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.feature_indices = feature_indices
        self.input_is_bottom = set_input_to_bottom

        if neck is not None:
            self.with_neck = True
            self.neck = neck
        else:
            self.with_neck = False

    def initialize_weights(self):
        self.backbone.initialize_weights()
        self.decoder.initialize_weights()
        if self.with_neck:
            self.neck.initialize_weights()

    def forward(self, in_image: Tensor) -> Sequence[Tensor]:
        """Feature Pyramid Network -> return multi-scale feature maps
        Args:
            in_image: (N, C, D, H, W)
        Returns:
            multi-scale features (top_down): [(N, F, D, H, W)]
        """
        bottom_up_features: Sequence[Tensor] = self.backbone(in_image)
        if self.feature_indices is not None:
            bottom_up_features = [bottom_up_features[index] for index in self.feature_indices]
        if self.input_is_bottom:
            bottom_up_features = [in_image, *bottom_up_features]
        if self.with_neck:
            bottom_up_features = self.neck(bottom_up_features)
        top_down_features = self.decoder(bottom_up_features)
        return top_down_features
