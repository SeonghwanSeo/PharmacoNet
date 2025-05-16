import torch_geometric.nn as gnn
from torch import Tensor, nn
from torch_geometric.data import Batch, Data


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
                gnn.GINEConv(
                    nn=nn.Sequential(gnn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()),
                    edge_dim=hidden_dim,
                )
                for _ in range(num_convs)
            ]
        )

        self.head = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim))

    def initialize_weights(self):
        def _init_weight(m):
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-1, 1)

        self.apply(_init_weight)

    def forward(self, data: Data | Batch) -> Tensor:
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
        return self.head(x)
