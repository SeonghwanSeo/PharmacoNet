import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch


class AffinityHead(nn.Module):
    def __init__(self, hidden_dim: int, p_dropout: float = 0.1):
        super().__init__()
        self.interaction_mlp: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.mlp_affinity: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(p_dropout)

    def initialize_weights(self):
        def _init_weight(m):
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weight)

    def forward(
        self,
        x_protein: Tensor,
        x_ligand: Tensor,
        ligand_batch: Tensor,
        num_ligands: int,
    ) -> Tensor:
        """
        affinity predict header for (single protein - multi ligands)
        output: (N_ligand,)
        """
        Z_complex = torch.einsum("ik,jk->ijk", x_ligand, x_protein)  # [Vlig, Vprot, Fh]
        Z_complex, mask_complex = to_dense_batch(Z_complex, ligand_batch, batch_size=num_ligands)
        mask_complex = mask_complex.unsqueeze(-1)  # [N, Vlig, 1]
        Z_complex = self.interaction_mlp(self.dropout(Z_complex))
        pair_affinity = self.mlp_affinity(Z_complex).squeeze(-1) * mask_complex
        return pair_affinity.sum((1, 2))
