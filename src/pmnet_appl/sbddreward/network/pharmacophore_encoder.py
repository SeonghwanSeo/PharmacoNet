from __future__ import annotations
import torch
from torch import nn
from torch import Tensor
from pmnet.api.typing import MultiScaleFeature, HotspotInfo


class PharmacophoreEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.multi_scale_dims = (96, 96, 96, 96, 96)
        self.hotspot_dim = 192
        self.hidden_dim = hidden_dim
        self.hotspot_mlp: nn.Module = nn.Sequential(nn.SiLU(), nn.Linear(self.hotspot_dim, hidden_dim))
        self.pocket_mlp_list: nn.ModuleList = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Conv3d(channels, hidden_dim, 3)) for channels in self.multi_scale_dims]
        )
        self.pocket_layer: nn.Module = nn.Sequential(
            nn.SiLU(), nn.Linear(5 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def init_weight(self):
        def _init_weight(m):
            if isinstance(m, nn.Linear | nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weight)

    def forward(self, pmnet_attr: tuple[MultiScaleFeature, list[HotspotInfo]]) -> tuple[Tensor, Tensor, Tensor]:
        multi_scale_features, hotspot_infos = pmnet_attr
        dev = multi_scale_features[0].device
        if len(hotspot_infos) > 0:
            hotspot_positions = torch.tensor([info["hotspot_position"] for info in hotspot_infos], device=dev)
            hotspot_features = torch.stack([info["hotspot_feature"] for info in hotspot_infos])
            hotspot_features = self.hotspot_mlp(hotspot_features)
        else:
            hotspot_positions = torch.zeros((0, 3), device=dev)
            hotspot_features = torch.zeros((0, self.hidden_dim), device=dev)
        pocket_features: Tensor = torch.cat(
            [mlp(feat.squeeze(0)).mean((-1, -2, -3)) for mlp, feat in zip(self.pocket_mlp_list, multi_scale_features, strict=False)],
            dim=-1,
        )
        pocket_features = self.pocket_layer(pocket_features)
        return hotspot_features, hotspot_positions, pocket_features
