from dataclasses import dataclass
from typing import NamedTuple

import torch
from typing_extensions import Self


class MultiScaleFeature(NamedTuple):
    size4: torch.Tensor  # [96, 4, 4, 4]
    size8: torch.Tensor  # [96, 8, 8, 8]
    size16: torch.Tensor  # [96, 16, 16, 16]
    size32: torch.Tensor  # [96, 32, 32, 32]
    size64: torch.Tensor  # [96, 64, 64, 64]


@dataclass(frozen=True, slots=True)
class HotspotInfo:
    type: str
    features: torch.Tensor
    position: tuple[float, float, float]
    score: float
    nci_type: str
    density_type: str
    density_map: torch.Tensor | None = None

    def to_state(self) -> dict:
        state = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k in state.keys():
            if isinstance(state[k], torch.Tensor):
                state[k] = state[k].cpu()
        return state

    @classmethod
    def from_state(cls, state: dict) -> Self:
        return cls(**state)


@dataclass(frozen=True, slots=True)
class PMNetAttr:
    multi_scale_features: MultiScaleFeature
    hotspots: list[HotspotInfo]

    def to_state(self) -> dict:
        multi_scale_features = tuple(v.cpu() for v in self.multi_scale_features)
        hotspots = [info.to_state() for info in self.hotspots]
        return {"multi_scale_features": multi_scale_features, "hotspots": hotspots}

    @classmethod
    def from_state(cls, state: dict) -> Self:
        multi_scale_features = MultiScaleFeature(*state["multi_scale_features"])
        hotspots = [HotspotInfo.from_state(info) for info in state["hotspot_infos"]]
        return cls(multi_scale_features, hotspots)
