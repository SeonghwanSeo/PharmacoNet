# NOTE: For DL Model Training
__all__ = ["PharmacoNet", "ProteinParser", "get_pmnet_dev", "MultiScaleFeature", "HotspotInfo"]

import torch
from pmnet.module import PharmacoNet
from pmnet.data.parser import ProteinParser
from . import typing


def get_pmnet_dev(
    device: str | torch.device = "cpu", score_threshold: float = 0.5, molvoxel_library: str = "numpy"
) -> PharmacoNet:
    """
    device: 'cpu' | 'cuda'
    score_threshold: float | dict[str, float] | None
        custom threshold to identify hotspots.
        For feature extraction, recommended value is '0.5'
    molvoxel_library: str
        If you want to use PharmacoNet in DL model training, recommend to use 'numpy'
    """
    return PharmacoNet(device, score_threshold, False, molvoxel_library)
