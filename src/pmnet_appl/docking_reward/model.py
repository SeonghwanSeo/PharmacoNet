import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from pathlib import Path
from numpy.typing import NDArray
from omegaconf import DictConfig
from typing import NewType

from pmnet.api import PharmacoNet, get_pmnet_dev

from .network.pharmacophore_encoder import PharmacophoreEncoder
from .network.ligand_encoder import GraphEncoder
from .network.head import AffinityHead
from .utils import NUM_ATOM_FEATURES, NUM_BOND_FEATURES, smi2graph
from .config import Config

Cache = NewType("Cache", tuple[Tensor, Tensor, Tensor])


class AffinityModel(nn.Module):
    def __init__(self, config: Config | DictConfig, device: str = "cuda"):
        super().__init__()
        self.pmnet: PharmacoNet = get_pmnet_dev(device)
        self.global_cfg = config
        self.cfg = config.model
        self.pharmacophore_encoder: PharmacophoreEncoder = PharmacophoreEncoder(self.cfg.hidden_dim)
        self.ligand_encoder: GraphEncoder = GraphEncoder(
            NUM_ATOM_FEATURES, NUM_BOND_FEATURES, self.cfg.hidden_dim, self.cfg.hidden_dim, self.cfg.ligand_num_convs
        )
        self.head: AffinityHead = AffinityHead(self.cfg.hidden_dim)
        self.l2_loss: nn.MSELoss = nn.MSELoss()
        self.to(device)
        self.initialize_weights()

    def initialize_weights(self):
        self.pharmacophore_encoder.initialize_weights()
        self.ligand_encoder.initialize_weights()
        self.head.initialize_weights()

    # NOTE: Model training
    def forward_train(self, batch) -> Tensor:
        if self.pmnet.device != self.device:
            self.pmnet.to(self.device)

        loss_list = []
        for pharmacophore_info, ligand_graphs in batch:
            # NOTE: Run PharmacoNet Feature Extraction
            # (Model is freezed; method `run_extraction` is decorated by torch.no_grad())
            pmnet_attr = self.pmnet.run_extraction(pharmacophore_info)
            del pharmacophore_info

            # NOTE: Binding Affinity Prediction
            x_protein, pos_protein, Z_protein = self.pharmacophore_encoder.forward(pmnet_attr)
            x_ligand = self.ligand_encoder.forward(ligand_graphs.to(self.device))
            affinity = self.head.forward(x_protein, x_ligand, ligand_graphs.batch, ligand_graphs.num_graphs)

            loss_list.append(self.l2_loss.forward(affinity, ligand_graphs.affinity))
        loss = torch.stack(loss_list).mean()
        return loss

    # NOTE: Python API
    @torch.no_grad()
    def feature_extraction(
        self,
        protein_pdb_path: str | Path,
        ref_ligand_path: str | Path | None = None,
        center: tuple[float, float, float] | NDArray | None = None,
    ) -> Cache:
        multi_scale_features, hotspot_infos = self.pmnet.feature_extraction(protein_pdb_path, ref_ligand_path, center)
        return self.pharmacophore_encoder(multi_scale_features, hotspot_infos)

    def scoring(self, target: str, smiles: str) -> Tensor:
        return self._scoring(self.cache[target], smiles)

    def scoring_list(self, target: str, smiles_list: list[str]) -> Tensor:
        return self._scoring_list(self.cache[target], smiles_list)

    @torch.no_grad()
    def _scoring(self, cache: Cache, smiles: str) -> Tensor:
        return self._scoring_list(cache, [smiles])

    @torch.no_grad()
    def _scoring_list(self, cache: Cache, smiles_list: list[str]) -> Tensor:
        Z_protein, X_protein, pos_protein = cache
        Z_protein = Z_protein.to(self.device)
        X_protein = X_protein.to(self.device)
        pos_protein = pos_protein.to(self.device)
        ligand_batch = Batch.from_data_list([smi2graph(smiles) for smiles in smiles_list]).to(self.device)
        X_ligand, Z_ligand = self.ligand_encoder(ligand_batch)
        return self.head.scoring(X_protein, pos_protein, Z_protein, X_ligand, Z_ligand, ligand_batch.batch)

    def to(self, device: str | torch.device):
        super().to(device)
        if self.pmnet is not None:
            if self.pmnet.device != self.device:
                self.pmnet.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
