# NOTE: For DL Model Training
__all__ = ["PharmacoNet", "ProteinParser", "get_pmnet_dev"]

import torch

from pmnet.data.parser import ProteinParser
from pmnet.module import PharmacoNet


def get_pmnet_dev(
    device: str | torch.device = "cpu",
    score_threshold: float = 0.5,
    molvoxel_library: str = "numpy",
    compile: bool = False,
) -> PharmacoNet:
    """
    device: 'cpu' | 'cuda'
    score_threshold: float | dict[str, float] | None
        custom threshold to identify hotspots.
        For feature extraction, recommended value is '0.5'
    molvoxel_library: str
        If you want to use PharmacoNet in DL model training, recommend to use 'numpy'
    compile: bool
        torch.compile
    """
    pm_net: PharmacoNet = PharmacoNet(device, score_threshold, False, molvoxel_library)
    if compile:
        assert torch.__version__ >= "2.0.0"
        pm_net.run_extraction = torch.compile(pm_net.run_extraction)
    return pm_net
