from collections.abc import Sequence

from torch import Tensor, nn

from pmnet.network.backbones.swinv2 import SwinTransformerV2
from pmnet.network.decoders.fpn_decoder import FPNDecoder


class FeaturePyramidNetwork(nn.Module):
    def __init__(
        self,
        backbone: SwinTransformerV2,
        decoder: FPNDecoder,
        neck: nn.Module | None = None,
        feature_indices: tuple[int, ...] = (0, 1, 2, 3),
        set_input_to_bottom: bool = True,
    ):
        super().__init__()
        self.backbone: SwinTransformerV2 = backbone
        self.decoder: FPNDecoder = decoder
        self.feature_indices: tuple[int, ...] = feature_indices
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
