import torch
from torch import nn

from typing import Sequence, List, Optional, Tuple
from torch import Tensor

from .builder import HEAD


@HEAD.register()
class MaskHead(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        token_feature_dim: int,
    ):
        super(MaskHead, self).__init__()
        feature_channels_list: List[int] = decoder.feature_channels
        self.point_mlp_list = nn.ModuleList(
            [nn.Linear(token_feature_dim, channels) for channels in feature_channels_list]
        )
        self.background_mlp_list = nn.ModuleList(
            [nn.Linear(token_feature_dim, channels) for channels in feature_channels_list]
        )
        self.decoder = decoder
        self.conv_logits = nn.Conv3d(decoder.channels, 1, kernel_size=1)

    def initialize_weights(self):
        def _init_weight(m):
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in [self.conv_logits, self.point_mlp_list, self.background_mlp_list]:
            m.apply(_init_weight)

        self.decoder.initialize_weights()

    def forward(
        self,
        multi_scale_features: Tensor,
        tokens_list: Sequence[Tensor],
        token_features_list: Sequence[Tensor],
        return_aux: bool = False,
    ) -> Tuple[List[Tensor], Optional[List[List[Tensor]]]]:
        """Box Predicting Function

        Args:
            multi_scale_features: Top-Down, List[FloatTensor [N, F_scale, D_scale, H_scale, W_scale]]
            tokens_list: List[IntTensor [Nbox, 4] - (x, y, z, i)]
            token_features_list: List[FloatTensor[Nbox, Ftoken]

        Returns:
            masks_list: List[FloatTensor [Nbox, D, H, W]]
            aux_masks_list(optional): List[List[FloatTensor [Nbox, D, H, W]]]
        """
        num_images = len(tokens_list)
        assert len(multi_scale_features[0]) == num_images
        multi_scale_features = multi_scale_features[::-1]   # Top-Down -> Bottom-Up
        if return_aux:
            out_masks_list = []
            aux_masks_list = []
            for image_idx in range(num_images):
                out = self.do_predict_w_aux(
                    [multi_scale_features[level][image_idx] for level in range(len(multi_scale_features))],
                    tokens_list[image_idx],
                    token_features_list[image_idx],
                )
                out_masks_list.append(out[-1])
                aux_masks_list.append(out[:-1])
        else:
            aux_masks_list = None
            out_masks_list = [
                self.do_predict_single(
                    [multi_scale_features[level][image_idx] for level in range(len(multi_scale_features))],
                    tokens_list[image_idx],
                    token_features_list[image_idx],
                ) for image_idx in range(num_images)
            ]
        return out_masks_list, aux_masks_list

    def do_predict_w_aux(
        self,
        multi_scale_features: Sequence[Tensor],
        tokens: Tensor,
        token_features: Tensor,
    ) -> List[Tensor]:
        """Box Predicting Function

        Args:
            multi_scale_features: Bottom-Up, List[FloatTensor [F_scale, D_scale, H_scale, W_scale]]
            token_features: FloatTensor [Nbox, Ftoken]
            tokens: IntTensor [Nbox, 4] - (x, y, z, i)

        Returns:
            multi_scale_box_masks: List[FloatTensor [Nbox, D_scale, H_scale, W_scale]]
        """
        Nbox = tokens.size(0)
        multi_scale_size = [features.size()[1:] for features in multi_scale_features]
        if Nbox > 0:
            Dout, Hout, Wout = multi_scale_size[0]
            token_indices = torch.split(tokens, 1, dim=1)                           # (x_list, y_list, z_list, i_list)
            xs, ys, zs, _ = token_indices

            bottom_up_box_features = []
            for level in range(len(multi_scale_features)):
                features = multi_scale_features[level]
                _, D, H, W = features.shape
                _xs = torch.div(xs, Dout // D, rounding_mode='trunc')
                _ys = torch.div(ys, Hout // H, rounding_mode='trunc')
                _zs = torch.div(zs, Wout // W, rounding_mode='trunc')
                box_features = self.get_box_features(features, (_xs, _ys, _zs), token_features, level)
                bottom_up_box_features.append(box_features)

            top_down_features = self.decoder(bottom_up_box_features)
            top_down_box_masks = [self.conv_logits(features).squeeze(1) for features in top_down_features]
            return top_down_box_masks
        else:
            return [torch.empty((0, *size), dtype=multi_scale_features[0].dtype, device=tokens.device) for size in multi_scale_size[::-1]]

    def do_predict_single(
        self,
        multi_scale_features: Sequence[Tensor],
        tokens: Tensor,
        token_features: Tensor,
    ) -> Tensor:
        """Box Predicting Function

        Args:
            multi_scale_features: Bottom-Up, List[FloatTensor [F_scale, D_scale, H_scale, W_scale]]
            token_features: FloatTensor [Nbox, Ftoken]
            tokens: IntTensor [Nbox, 4] - (x, y, z, i)

        Returns:
            box_masks: FloatTensor [Nbox, D_out, H_out, W_out]
        """
        Nbox = tokens.size(0)
        multi_scale_size = [features.size()[1:] for features in multi_scale_features]
        Dout, Hout, Wout = multi_scale_size[0]
        if Nbox > 0:
            token_indices = torch.split(tokens, 1, dim=1)                           # (x_list, y_list, z_list, i_list)
            xs, ys, zs, _ = token_indices

            bottom_up_box_features = []
            for level in range(len(multi_scale_features)):
                features = multi_scale_features[level]
                _, D, H, W = features.shape
                _xs = torch.div(xs, Dout // D, rounding_mode='trunc')
                _ys = torch.div(ys, Hout // H, rounding_mode='trunc')
                _zs = torch.div(zs, Wout // W, rounding_mode='trunc')
                box_features = self.get_box_features(features, (_xs, _ys, _zs), token_features, level)
                bottom_up_box_features.append(box_features)

            top_down_features = self.decoder(bottom_up_box_features)
            return self.conv_logits(top_down_features[-1]).squeeze(1)
        else:
            return torch.empty((0, Dout, Hout, Wout), dtype=multi_scale_features[0].dtype, device=tokens.device)

    def get_box_features(
        self,
        features: Tensor,
        token_indices: Tuple[Tensor, Tensor, Tensor],
        token_features: Tensor,
        level: int
    ) -> Tensor:
        """Extract token features

        Args:
            features: FloatTensor [F_scale, D_scale, H_scale, W_scale]
            token_indices: Tuple[IntTensor [Nbox,], IntTensor [Nbox,], IntTensor[Nbox,]] - (xs, ys, zs)
            token_features: FloatTensor [Nbox, Ftoken]

        Returns:
            box_features: FloatTensor [Nbox, F_scale, D_scale, H_scale, W_scale]
        """
        F, D, H, W = features.shape
        xs, ys, zs = token_indices
        Nbox = token_features.size(0)
        Nboxs = torch.arange(Nbox, dtype=xs.dtype, device=xs.device)
        background_features = self.background_mlp_list[level](token_features)               # [Nbox, F]
        point_features = self.point_mlp_list[level](token_features)                         # [Nbox, F]
        box_features = background_features.view(Nbox, F, 1, 1, 1).repeat(1, 1, D, H, W)     # [Nbox, F, D, H, W]
        box_features[Nboxs, :, xs, ys, zs] += point_features
        features = features.unsqueeze(0) + box_features
        return features
