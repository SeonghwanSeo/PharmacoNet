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
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor

from pmnet.typing import HotspotInfo, MultiScaleFeature
from pmnet_appl.base.proxy import BaseProxy
from pmnet_appl.sbddreward.data import NUM_ATOM_FEATURES, NUM_BOND_FEATURES, smi2graph
from pmnet_appl.sbddreward.network import (
    AffinityHead,
    GraphEncoder,
    PharmacophoreEncoder,
)

Cache = tuple[Tensor, Tensor, Tensor, float, float]


class SBDDReward_Proxy(BaseProxy):
    root_dir = Path(__file__).parent

    def _setup_model(self):
        self.model = _RewardNetwork()

    def _get_cache(self, pmnet_attr: tuple[MultiScaleFeature, list[HotspotInfo]]) -> Cache:
        return self.model.get_cache(pmnet_attr)

    @torch.no_grad()
    def _scoring_list(self, cache: Cache, smiles_list: list[str], return_sigma: bool = False) -> Tensor:
        cache = (
            cache[0].to(self.device),
            cache[1].to(self.device),
            cache[2].to(self.device),
            cache[3],
            cache[4],
        )

        ligand_graphs = []
        flag = []
        for smi in smiles_list:
            try:
                graph = smi2graph(smi)
            except Exception:
                flag.append(False)
            else:
                flag.append(True)
                ligand_graphs.append(graph)
        if not any(flag):
            return torch.zeros(len(smiles_list), dtype=torch.float32, device=self.device)
        ligand_batch: gd.Batch = gd.Batch.from_data_list(ligand_graphs).to(self.device)
        if all(flag):
            return self.model.scoring(cache, ligand_batch, return_sigma)
        else:
            result = torch.zeros(len(smiles_list), dtype=torch.float32, device=self.device)
            result[flag] = self.model.scoring(cache, ligand_batch, return_sigma)
            return result

    @classmethod
    def load(
        cls,
        docking: str,
        train_dataset: str,
        db: Path | str | None,
        device: torch.device | str = "cpu",
    ):
        """Load Pretrained Proxy Model

        Parameters
        ----------
        docking : str
            docking program (currently: ['UniDock_Vina'])
        train_dataset : str
            training dataset name (currently: ['ZINC'])
        db : Path | str | None
            cache database path ('train' | 'test' | 'all' | custom cache database path)
            - 'train': CrossDocked2020 training pockets (15,201)
            - 'test': CrossDocked2020 test pockets (100)
            - 'all': train + test
        device : str
            cuda | spu
        """
        assert docking in ["UniDock_Vina"]
        assert train_dataset in ["ZINC"]
        return super().load(docking, train_dataset, db, device)

    def scoring(self, target: str, smiles: str, return_sigma: bool = False) -> Tensor:
        """Scoring single molecule with its SMILES

        Parameters
        ----------
        target : str
            target key
        smiles : str
            molecule smiles
        return_sigma : bool (default = False)
            if True, return sigma instead of absolute affinity

        Returns
        -------
        Tensor [1,]
            Esimated Docking Score (or Simga)

        """
        return self._scoring_list(self._cache[target], [smiles], return_sigma)

    def scoring_list(self, target: str, smiles_list: list[str], return_sigma: bool = False) -> Tensor:
        """Scoring multiple molecules with their SMILES

        Parameters
        ----------
        target : str
            target key
        smiles_list : list[str]
            molecule smiles list
        return_sigma : bool (default = False)
            if True, return sigma instead of absolute affinity

        Returns
        -------
        Tensor [N,]
            Esimated Docking Scores (or Simga)

        """
        return self._scoring_list(self._cache[target], smiles_list, return_sigma)

    def get_statistic(self, target: str) -> tuple[float, float]:
        cache: Cache = self._cache[target]
        return cache[-2], cache[-1]


class _RewardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pharmacophore_encoder: PharmacophoreEncoder = PharmacophoreEncoder(128)
        self.ligand_encoder: GraphEncoder = GraphEncoder(NUM_ATOM_FEATURES, NUM_BOND_FEATURES, 128, 128, 4)
        self.head: AffinityHead = AffinityHead(128, 3)

    def get_cache(self, pmnet_attr) -> Cache:
        X_protein, pos_protein, Z_protein = self.pharmacophore_encoder.forward(pmnet_attr)
        mu, std = self.head.cal_mu(Z_protein), self.head.cal_std(Z_protein)
        return (
            X_protein.cpu(),
            pos_protein.cpu(),
            Z_protein.cpu(),
            mu.item(),
            std.item(),
        )

    def scoring(self, cache: Cache, ligand_batch: gd.Batch, return_sigma: bool = False):
        X_protein, pos_protein, Z_protein, mu, std = cache
        X_ligand, Z_ligand = self.ligand_encoder.forward(ligand_batch)
        sigma = self.head.cal_sigma(X_protein, pos_protein, Z_protein, X_ligand, Z_ligand, ligand_batch.batch)
        if return_sigma:
            return sigma
        else:
            return sigma * std + mu

    def get_info(self, cache: Cache, ligand_batch: gd.Batch) -> tuple[float, float, Tensor]:
        X_protein, pos_protein, Z_protein, mu, std = cache
        X_ligand, Z_ligand = self.ligand_encoder.forward(ligand_batch)
        sigma = self.head.cal_sigma(X_protein, pos_protein, Z_protein, X_ligand, Z_ligand, ligand_batch.batch)
        return mu, std, sigma


if __name__ == "__main__":
    print("start!")
    proxy = SBDDReward_Proxy.load("UniDock_Vina", "ZINC", "test", "cpu")
    print("proxy is loaded")
    print(proxy.scoring("14gs_A", "c1ccccc1"))
    print(proxy.scoring("14gs_A", "c11"))
    print(proxy.scoring_list("14gs_A", ["c1ccccc1", "C1CCCCC1", "c11"]))
