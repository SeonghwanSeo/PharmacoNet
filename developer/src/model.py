import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from pmnet.api import PharmacoNet, get_pmnet_dev

from .config import Config
from .data import NUM_ATOM_FEATURES, NUM_BOND_FEATURES
from .network import AffinityHead, GraphEncoder, PharmacophoreEncoder

Cache = tuple[Tensor, Tensor, Tensor]


class AffinityModel(nn.Module):
    def __init__(self, config: Config | DictConfig):
        super().__init__()
        self.pmnet: PharmacoNet = get_pmnet_dev()
        self.global_cfg = config
        self.cfg = config.model
        self.pharmacophore_encoder: PharmacophoreEncoder = PharmacophoreEncoder(self.cfg.hidden_dim)
        self.ligand_encoder: GraphEncoder = GraphEncoder(
            NUM_ATOM_FEATURES,
            NUM_BOND_FEATURES,
            self.cfg.hidden_dim,
            self.cfg.hidden_dim,
            self.cfg.ligand_num_convs,
        )
        self.head: AffinityHead = AffinityHead(self.cfg.hidden_dim)
        self.l2_loss: nn.MSELoss = nn.MSELoss()
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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
