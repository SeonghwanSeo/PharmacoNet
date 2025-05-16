from __future__ import annotations

import torch
import torch_geometric.nn as pygnn
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_mean, scatter_sum


class GraphEncoder(nn.Module):
    def __init__(
        self,
        input_node_dim: int,
        input_edge_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_convs: int,
    ):
        super().__init__()
        self.graph_channels: int = out_dim
        self.atom_channels: int = out_dim

        # Ligand Encoding
        self.node_layer = nn.Linear(input_node_dim, hidden_dim)
        self.edge_layer = nn.Linear(input_edge_dim, hidden_dim)
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

    def init_weight(self):
        def _init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weight)

    def forward(
        self,
        data: Data | Batch,
    ) -> tuple[Tensor, Tensor]:
        """Affinity Prediction

        Args:
            x: Node Feature
            edge_attr: Edge Feature
            edge_index: Edge Index

        Returns:
            updated_data: Union[Data, Batch]
        """
        x: Tensor = self.node_layer(data.x)
        edge_attr: Tensor = self.edge_layer(data.edge_attr)

        skip_x = x
        edge_index = data.edge_index
        for layer in self.conv_list:
            x = layer(x, edge_index, edge_attr)

        x = skip_x + x
        X = self.head(x)

        if isinstance(data, Batch):
            Z1 = scatter_sum(x, data.batch, dim=0, dim_size=data.num_graphs)  # V, Fh -> N, Fh
            Z2 = scatter_mean(x, data.batch, dim=0, dim_size=data.num_graphs)  # V, Fh -> N, Fh
        else:
            Z1 = x.sum(0, keepdim=True)  # V, Fh -> 1, Fh
            Z2 = x.mean(0, keepdim=True)  # V, Fh -> 1, Fh
        Z = torch.cat([Z1, Z2], dim=-1)
        Z = self.readout_gate(Z) * self.readout_layer(Z)  # [N, Fh]
        return X, Z
