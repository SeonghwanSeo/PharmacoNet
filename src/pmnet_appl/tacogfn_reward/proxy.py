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
@article{shen2024tacogfn,
  title = {TacoGFN: Target-conditioned GFlowNet for Structure-based Drug Design},
  author = {Shen, Tony and Seo, Seonghwan and Lee, Grayson and Pandey, Mohit and Smith, Jason R and Cherkasov, Artem and Kim, Woo Youn and Ester, Martin},
  journal = {arXiv preprint arXiv:2310.03223},
  year = {2024},
  url = {https://arxiv.org/abs/2310.03223},
}
```
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch_geometric.data as gd
import torch_geometric.nn as pygnn
from torch import Tensor, nn
from torch_scatter import scatter_mean, scatter_sum

from pmnet.typing import PMNetAttr
from pmnet_appl.base.proxy import BaseProxy
from pmnet_appl.tacogfn_reward.data import smi2graph

Cache = tuple[Tensor, Tensor]


class TacoGFN_Proxy(BaseProxy):
    root_dir = Path.home() / ".local" / "share" / "pmnet" / "tacogfn_proxy"
    cache_gdrive_link: dict[tuple[str, str], str] = {
        ("QVina-ZINCDock15M", "train"): "1VibvAjhir5oXx5cmzfE0F2UVTSDsGH3v",
        ("QVina-ZINCDock15M", "test"): "1F05JjkJuc6FwU4h8MLUEan34ovewGPLz",
        ("QVina-CrossDocked2020", "train"): "1-5he-ItdtcZvlGqyI_rVU0XIk0XGzFC-",
        ("QVina-CrossDocked2020", "test"): "1Ps3-Mj2GHH_FLtnjAD1riYiRyK01C_T8",
    }
    model_gdrive_link: dict[str, str] = {
        "QVina-ZINCDock15M": "1lrH79-6YI2CfEP5sWIWzgboZsONXWkIZ",
        "QVina-CrossDocked2020": "1Kjn4xNc8458Ibf-ckWjtuUotuhGFLiHK",
    }

    def _setup_model(self):
        self.model: AffinityHead = AffinityHead()

    def _load_checkpoint(self, ckpt_path: str | Path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

    def _get_cache(self, pmnet_attr: PMNetAttr) -> Cache:
        multi_scale_features, hotspots = pmnet_attr.multi_scale_features, pmnet_attr.hotspots
        if len(hotspots) > 0:
            hotspot_features = torch.stack([node.features for node in hotspots])
        else:
            hotspot_features = torch.zeros((0, 192), device=self.device)
        pocket_features_list, hotspot_features_list = self.model.ready_to_calculate(
            multi_scale_features, [hotspot_features]
        )
        return pocket_features_list[0].cpu(), hotspot_features_list[0].cpu()

    @torch.no_grad()
    def _scoring_list(self, cache: Cache, smiles_list: list[str]) -> Tensor:
        pocket_features, hotspot_features = cache[0].to(self.device), cache[1].to(self.device)
        if len(smiles_list) == 1:
            ligand_graph = smi2graph(smiles_list[0]).to(self.device)
        else:
            ligand_graph = gd.Batch.from_data_list([smi2graph(smiles) for smiles in smiles_list]).to(self.device)
        return self.model._calculate_affinity_single(pocket_features, hotspot_features, ligand_graph)

    @classmethod
    def load(
        cls,
        docking: str,
        train_dataset: str,
        db: Path | str | None,
        device: str | torch.device = "cpu",
    ):
        """Load Pretrained Proxy Model

        Parameters
        ----------
        docking : str
            docking program name (currently: ['QVina'])
        train_dataset : str
            training dataset name (currently: ['ZINCDock15M', 'CrossDocked2020'])
        db : Path | str | None
            cache database path ('train' | 'test' | 'all' | custom cache database path)
            - 'train': CrossDocked2020 training pockets (15,201)
            - 'test': CrossDocked2020 test pockets (100)
            - 'all': train + test
        device : str | torch.device
            cuda | spu
        """
        assert docking in ["QVina", "QuickVina"], f"Unsupported docking program: {docking}"
        assert train_dataset in ["ZINCDock15M", "CrossDocked2020"], f"Unsupported train dataset: {train_dataset}"
        return super().load("QVina", train_dataset, db, device)


class AffinityHead(nn.Module):
    def __init__(self):
        super().__init__()
        feature_channels: Sequence[int] = [96, 96, 96, 96, 96]
        token_feature_dim: int = 192
        hidden_dim: int = 256

        self.hidden_dim = hidden_dim

        # Ready-To-Affinity-Calculation
        self.token_mlp: nn.Module = nn.Sequential(
            nn.SiLU(),
            nn.Linear(token_feature_dim, hidden_dim),
        )
        self.pocket_mlp_list: nn.ModuleList = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Conv3d(channels, hidden_dim, 3)) for channels in feature_channels]
        )
        self.pocket_mlp: nn.Module = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim * len(feature_channels), hidden_dim),
        )

        self.concat_layer: nn.Module = nn.Linear(3 * hidden_dim, hidden_dim)
        self.concat_gate: nn.Module = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.ligand_encoder = GraphEncoder(10, 5, 128, 256, 4)
        ligand_atom_channels: int = self.ligand_encoder.atom_channels
        ligand_graph_channels: int = self.ligand_encoder.graph_channels

        if ligand_atom_channels != hidden_dim:
            self.ligand_layer_atom = nn.Linear(ligand_atom_channels, hidden_dim)
        else:
            self.ligand_layer_atom = nn.Identity()
        if ligand_graph_channels != hidden_dim:
            self.ligand_layer_graph = nn.Linear(ligand_graph_channels, hidden_dim)
        else:
            self.ligand_layer_graph = nn.Identity()

        self.energy_bias_mlp: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.interaction_mlp: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.pair_energy_layer: nn.Module = nn.Linear(hidden_dim, 1)
        self.pair_energy_gate: nn.Module = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        multi_scale_features: Sequence[Tensor],
        token_features_list: Sequence[Tensor],
        ligand_graph_list: Sequence[gd.Data] | Sequence[gd.Batch],
    ) -> Tensor:
        """Affinity Prediction

        Args:
            multi_scale_features: Top-Down, List[FloatTensor [N, F_scale, D_scale, H_scale, W_scale]]   - size: N_scale
            token_features_list: List[FloatTensor[Nbox, Ftoken]                                         - size: N
            ligand_graph_list: Union[Sequence[gd.Data], Sequence[gd.Batch]]                             - size: N

        Returns:
            affinity: FloatTensor [N, Ngraph]
        """
        assert len(token_features_list) == len(ligand_graph_list)

        pocket_features, token_features_list = self.ready_to_calculate(multi_scale_features, token_features_list)
        num_ligands = ligand_graph_list[0].y.size(0)
        pocket_features = pocket_features.unsqueeze(1).repeat(1, num_ligands, 1)
        return self.calculate_affinity(pocket_features, token_features_list, ligand_graph_list)

    def ready_to_calculate(
        self,
        multi_scale_features: Sequence[Tensor],
        token_features_list: Sequence[Tensor],
    ) -> tuple[Tensor, Sequence[Tensor]]:
        """Affinity Prediction

        Args:
            multi_scale_features: Top-Down, List[FloatTensor [N, F_scale, D_scale, H_scale, W_scale]]
            token_features_list: List[FloatTensor[Nbox, Ftoken]

        Returns:
            pocket_features: FloatTensor [N, F_hidden]
            token_features_list: List[FloatTensor [Nbox, F_hidden]]
        """
        multi_scale_features = multi_scale_features[::-1]  # Top-Down -> Bottom-Up
        multi_scale_features = [
            layer(feature) for layer, feature in zip(self.pocket_mlp_list, multi_scale_features, strict=False)
        ]
        pocket_features: Tensor = self.pocket_mlp(
            torch.cat(
                [feature.mean(dim=(-1, -2, -3)) for feature in multi_scale_features],
                dim=-1,
            )
        )  # [N, Fh]

        token_features_list = [
            self.token_mlp(feature) for feature in token_features_list
        ]  # List[FloatTensor[Nbox, Fh]]
        out_token_features_list = []
        for feature in token_features_list:
            if feature.size(0) == 0:
                feature = torch.zeros((2 * self.hidden_dim,), dtype=feature.dtype, device=feature.device)
            else:
                feature = torch.cat([feature.sum(0), feature.mean(0)])
            out_token_features_list.append(feature)
        token_features = torch.stack(out_token_features_list)  # [N, 2 * Fh]
        pocket_features = torch.cat([pocket_features, token_features], dim=-1)  # [N, 3 * Fh]
        pocket_features = self.concat_layer(pocket_features) * self.concat_gate(pocket_features)  # [N, Fh]
        return pocket_features, token_features_list

    def calculate_affinity(
        self,
        pocket_features: Tensor,
        token_features_list: Sequence[Tensor],
        ligand_graph_list: Sequence[gd.Data] | Sequence[gd.Batch],
    ) -> Tensor:
        """
        pred: [N, Ngraph]    # Ngraph: mini-batch-size, number of ligands per receptor
        """
        num_images = len(token_features_list)
        total_pred_list = [
            self._calculate_affinity_single(
                pocket_features[image_idx],
                token_features_list[image_idx],
                ligand_graph_list[image_idx],
            )
            for image_idx in range(num_images)
        ]
        return torch.stack(total_pred_list)

    def _calculate_affinity_single(
        self,
        pocket_features: Tensor,
        token_features: Tensor,
        ligand_graph: gd.Data | gd.Batch,
    ) -> Tensor:
        X, Z = self.ligand_encoder(ligand_graph)
        ligand_atom_features = self.ligand_layer_atom(X)  # [Natom, Fh]
        interaction_map = torch.einsum("ik,jk->ijk", ligand_atom_features, token_features)  # [Natom, Nbox, Fh]
        interaction_map = self.interaction_mlp(interaction_map)

        # Element-Wise Calculation
        pair_energy = self.pair_energy_layer(interaction_map) * self.pair_energy_gate(
            interaction_map
        )  # [Natom, Nbox, 1]
        if isinstance(ligand_graph, gd.Batch):
            pair_energy = pair_energy.sum((1, 2))  # [Natom,]
            pair_energy = scatter_sum(pair_energy, ligand_graph.batch)  # [N,]
        else:
            pair_energy = pair_energy.sum()

        # Graph-Wise Calculation
        ligand_graph_features = self.ligand_layer_graph(Z)  # [N, Fh]
        pocket_features = pocket_features.repeat(Z.shape[0], 1)
        bias = self.energy_bias_mlp(torch.cat([pocket_features, ligand_graph_features], dim=-1))  # [N, 1]

        return pair_energy.view(-1) + bias.view(-1)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        input_node_dim: int = 10,
        input_edge_dim: int = 5,
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_convs: int = 4,
    ):
        super().__init__()
        self.graph_channels: int = out_dim
        self.atom_channels: int = out_dim

        # Ligand Encoding
        self.node_layer = nn.Embedding(input_node_dim, hidden_dim)
        self.edge_layer = nn.Embedding(input_edge_dim, hidden_dim)
        self.conv_list = nn.ModuleList(
            [
                pygnn.GINEConv(
                    nn=nn.Sequential(pygnn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()),
                    edge_dim=hidden_dim,
                )
                for _ in range(num_convs)
            ]
        )
        self.readout_layer = nn.Linear(hidden_dim * 2, out_dim)
        self.readout_gate = nn.Linear(hidden_dim * 2, out_dim)

        self.head = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, data: gd.Data | gd.Batch) -> tuple[Tensor, Tensor]:
        """Affinity Prediction

        Args:
            x: Node Feature
            edge_attr: Edge Feature
            edge_index: Edge Index

        Returns:
            updated_data: Union[gd.Data, gd.Batch]
        """
        x: Tensor = self.node_layer(data.x)
        edge_attr: Tensor = self.edge_layer(data.edge_attr)
        skip_x = x
        edge_index = data.edge_index
        for layer in self.conv_list:
            x = layer(x, edge_index, edge_attr)
        x = skip_x + x
        X = self.head(x)
        if isinstance(data, gd.Batch):
            Z1 = scatter_sum(x, data.batch, dim=0, dim_size=data.num_graphs)  # V, Fh -> N, Fh
            Z2 = scatter_mean(x, data.batch, dim=0, dim_size=data.num_graphs)  # V, Fh -> N, Fh
        else:
            Z1 = x.sum(0, keepdim=True)  # V, Fh -> 1, Fh
            Z2 = x.mean(0, keepdim=True)  # V, Fh -> 1, Fh
        Z = torch.cat([Z1, Z2], dim=-1)
        Z = self.readout_gate(Z) * self.readout_layer(Z)  # [N, Fh]
        return X, Z


if __name__ == "__main__":
    print("start!")
    proxy = TacoGFN_Proxy.load("QVina", "ZINCDock15M", "all", "cpu")
    print("load TacoGFN QVina-ZINCDock15M Proxy!")
    print(proxy.scoring("14gs_A", "c1ccccc1"))
    print(proxy.scoring_list("14gs_A", ["c1ccccc1", "C1CCCCC1"]))

    proxy = TacoGFN_Proxy.load("QVina", "CrossDocked2020", "all", "cpu")
    print("load TacoGFN QVina-CrossDocked2020 Proxy!")
    print(proxy.scoring("14gs_A", "c1ccccc1"))
    print(proxy.scoring_list("14gs_A", ["c1ccccc1", "C1CCCCC1"]))
