from pmnet.network.backbones.swinv2 import SwinTransformerV2
from pmnet.network.cavity_head import CavityHead
from pmnet.network.decoders.fpn_decoder import FPNDecoder
from pmnet.network.detector import PharmacoNetModel
from pmnet.network.feature_embedding import FeaturePyramidNetwork
from pmnet.network.mask_head import MaskHead
from pmnet.network.token_head import TokenHead


def build_model(config: dict) -> PharmacoNetModel:
    # embedding
    embedding = FeaturePyramidNetwork(
        backbone=SwinTransformerV2(
            in_channels=33,
            image_size=64,
            patch_size=2,
            embed_dim=96,
            depths=(2, 6, 2, 2),
            num_heads=(3, 6, 12, 24),
            window_size=4,
            out_indices=(0, 1, 2, 3),
        ),
        decoder=FPNDecoder(
            feature_channels=(33, 96, 192, 384, 768),
            num_convs=(1, 2, 2, 2, 2),
            channels=96,
            interpolate_mode="nearest",
        ),
    )
    cavity_head = CavityHead(
        feature_dim=96,
        hidden_dim=96,
    )

    token_head = TokenHead(
        feature_dim=96,
        num_interactions=10,
        token_feature_dim=192,
        num_feature_mlp_layers=3,
        num_score_mlp_layers=3,
    )

    mask_head = MaskHead(
        token_feature_dim=192,
        decoder=FPNDecoder(
            feature_channels=(96, 96, 96, 96, 96),
            num_convs=(1, 2, 2, 2, 2),
            channels=96,
        ),
    )

    return PharmacoNetModel(embedding, cavity_head, token_head, mask_head, num_interactions=10)
