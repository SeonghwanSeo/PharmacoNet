from collections.abc import Sequence

import torch.nn as nn
from torch import Tensor

from pmnet.typing import MultiScaleFeature

from .cavity_head import CavityHead
from .feature_embedding import FeaturePyramidNetwork
from .mask_head import MaskHead
from .token_head import TokenHead


class PharmacoNetModel(nn.Module):
    def __init__(
        self,
        embedding: FeaturePyramidNetwork,
        cavity_head: CavityHead,
        token_head: TokenHead,
        mask_head: MaskHead,
        num_interactions: int,
    ):
        super().__init__()
        self.num_interactions = num_interactions
        self.embedding = embedding
        self.cavity_head = cavity_head
        self.token_head = token_head
        self.mask_head = mask_head

    def initialize_weights(self):
        self.embedding.initialize_weights()
        self.token_head.initialize_weights()
        self.mask_head.initialize_weights()

    def setup_train(self, criterion: nn.Module):
        self.criterion = criterion

    def forward_feature(self, in_image: Tensor) -> MultiScaleFeature:
        """Feature Embedding
        Args:
            in_image: FloatTensor [N, C, Din, Hin, Win]
        Returns:
            multi-scale features: [FloatTensor [N, F, D, H, W]]
        """
        return MultiScaleFeature(*self.embedding.forward(in_image))

    def forward_cavity_extraction(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Cavity Extraction
        Args:
            features: FloatTensor [N, F, Dout, Hout, Wout]
        Returns:
            cavity_narrow: FloatTensor [N, 1, Dout, Hout, Wout]
            cavity_wide: FloatTensor [N, 1, Dout, Hout, Wout]
        """
        return self.cavity_head.forward(features)

    def forward_token_prediction(
        self,
        features: Tensor,
        tokens_list: Sequence[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """token Selection Network

        Args:
            features: FloatTensor [N, F, Dout, Hout, Wout]
            tokens_list: List[Tensor [Ntoken, 4] - (x, y, z, i)]

        Returns:
            token_scores_list: List[FloatTensor [Ntoken,] $\\in$ [0, 1]]
            token_features_list: List[FloatTensor [Ntoken, F]]
        """
        token_scores_list, token_features_list = self.token_head.forward(features, tokens_list)
        return token_scores_list, token_features_list

    def forward_segmentation(
        self,
        multi_scale_features: MultiScaleFeature,
        box_tokens_list: Sequence[Tensor],
        box_token_features_list: Sequence[Tensor],
        return_aux: bool = False,
    ) -> tuple[list[Tensor], list[list[Tensor]] | None]:
        """Mask Prediction

        Args:
            multi_scales_features: List[FloatTensor [N, F, D_scale, H_scale, W_scale]]
            box_tokens_list: List[Tensor [Nbox, 4] - (x, y, z, i)]
            box_token_features_list: List[FloatTensor [Nbox, F]]

        Returns:
            box_masks_list: List[FloatTensor [Nbox, D, H, W]]
            aux_box_masks_list: List[List[FloatTensor [Nbox, D_scale, H_scale, W_scale]]]
        """
        return self.mask_head.forward(multi_scale_features, box_tokens_list, box_token_features_list, return_aux)
