from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch

from .block import ComplexFormerBlock
from .layers.one_hot import OneHotEncoding


class AffinityHead(nn.Module):
    def __init__(self, hidden_dim: int, n_blocks: int, p_dropout: float = 0.1):
        super().__init__()
        # Complex Embedding
        self.interaction_mlp: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.one_hot = OneHotEncoding(0, 30, 16)
        self.protein_pair_embedding = nn.Linear(16, hidden_dim)
        self.blocks = nn.ModuleList(
            [ComplexFormerBlock(hidden_dim, hidden_dim // 4, 4, 4, 0.1) for _ in range(n_blocks)]
        )

        self.mlp_mu: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.mlp_std: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.mlp_sigma_bias: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mlp_sigma: nn.Module = nn.Linear(hidden_dim, 1)
        self.gate_sigma: nn.Module = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.linear_distance = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p_dropout)

    def scoring(
        self,
        X_protein: Tensor,
        pos_protein: Tensor,
        Z_protein: Tensor,
        X_ligand: Tensor,
        Z_ligand: Tensor,
        ligand_batch: Tensor,
        return_sigma: bool = False,
    ) -> Tensor:
        sigma = self.cal_sigma(X_protein, pos_protein, Z_protein, X_ligand, Z_ligand, ligand_batch)
        if return_sigma:
            return sigma
        mu, std = self.cal_mu(Z_protein), self.cal_std(Z_protein)
        return sigma * std + mu

    def cal_mu(self, Z_protein) -> torch.Tensor:
        return self.mlp_mu(self.dropout(Z_protein)).view(1) * -15

    def cal_std(self, Z_protein) -> torch.Tensor:
        return self.mlp_std(self.dropout(Z_protein)).view(1) * 5

    def cal_sigma(self, X_protein, pos_protein, Z_protein, X_ligand, Z_ligand, ligand_batch) -> torch.Tensor:
        Z_complex, mask_complex = self._embedding(X_protein, pos_protein, X_ligand, ligand_batch, Z_ligand.shape[0])
        Z_protein, Z_ligand, Z_complex = (
            self.dropout(Z_protein),
            self.dropout(Z_ligand),
            self.dropout(Z_complex),
        )
        z_sigma = self.mlp_sigma(Z_complex) * self.gate_sigma(Z_complex)
        sigma = (z_sigma.squeeze(-1) * mask_complex).sum((1, 2))
        bias = self.mlp_sigma_bias(torch.cat([Z_protein.view(1, -1).repeat(Z_ligand.size(0), 1), Z_ligand], dim=-1))
        return sigma.view(-1) + bias.view(-1)

    def _embedding(self, X_protein, pos_protein, X_ligand, ligand_batch, num_ligands) -> tuple[Tensor, Tensor]:
        Z_complex = torch.einsum("ik,jk->ijk", X_ligand, X_protein)  # [Vlig, Vprot, Fh]
        Z_complex = self.interaction_mlp(self.dropout(Z_complex))
        Z_complex, mask_complex = to_dense_batch(Z_complex, ligand_batch, batch_size=num_ligands)

        mask_complex = mask_complex.unsqueeze(-1)  # [N, Vlig, 1]
        if X_protein.shape[0] > 0:
            pdist_protein = torch.cdist(pos_protein, pos_protein, compute_mode="donot_use_mm_for_euclid_dist")
            pdist_protein = self.one_hot(pdist_protein).unsqueeze(0)
            Zpair_protein = self.protein_pair_embedding(pdist_protein.to(device=X_ligand.device, dtype=torch.float))
            Z_complex_init = Z_complex
            for block in self.blocks:
                Z_complex = block(Z_complex, Zpair_protein, mask_complex)
            Z_complex = Z_complex_init + Z_complex
        return Z_complex, mask_complex
