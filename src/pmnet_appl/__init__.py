"""
Copyright: if you use this script, please cite:
```
@article{seo2023pharmaconet,
  title = {PharmacoNet: Accelerating Large-Scale Virtual Screening by Deep Pharmacophore Modeling},
  author = {Seo, Seonghwan and Kim, Woo Youn},
  journal = {arXiv preprint arXiv:2310.00681},
  year = {2023},
  url = {https://arxiv.org/abs/2310.00681},
}
```
"""

from __future__ import annotations

from pathlib import Path

import torch

from pmnet_appl.base import BaseProxy

ALLOWED_MODEL_LIST = ["TacoGFN_Reward", "SBDDReward"]
ALLOWED_DOCKING_LIST = ["QVina", "UniDock_Vina"]


def get_docking_proxy(
    model: str,
    docking: str,
    train_dataset: str,
    db: str | Path | None,
    device: str | torch.device,
) -> BaseProxy:
    """Get Docking Proxy Model

    Parameters
    ----------
    model : str
        Model name (Currently: ['TacoGFN_Reward', 'SBDDReward'])
    docking : str
        Docking program name
    train_dataset : str
        Dataset for model training
    db : Path | str | None
        cache database path ('train' | 'test' | 'all' | custom cache database path)
        - 'train': CrossDocked2020 training pockets (15,201)
        - 'test': CrossDocked2020 test pockets (100)
        - 'all': train + test
    device : str | torch.device
        cuda | spu

    Returns
    -------
    Proxy Model: BaseProxy
    """

    assert model in ("TacoGFN_Reward", "SBDDReward"), f"model({model}) is not allowed"
    if model == "TacoGFN_Reward":
        from pmnet_appl.tacogfn_reward import TacoGFN_Proxy

        assert docking in ["QVina"], f"docking({docking}) is not allowed"
        assert train_dataset in ["ZINCDock15M", "CrossDocked2020"], f"train_dataset({train_dataset}) is not allowed"
        return TacoGFN_Proxy.load(docking, train_dataset, db, device)
    elif model == "SBDDReward":
        from pmnet_appl.sbddreward import SBDDReward_Proxy

        assert docking in ["UniDock_Vina"], f"docking({docking}) is not allowed"
        assert train_dataset in ["ZINC"], f"train_dataset({train_dataset}) is not allowed"
        return SBDDReward_Proxy.load(docking, train_dataset, db, device)
    else:
        raise ValueError(docking)
