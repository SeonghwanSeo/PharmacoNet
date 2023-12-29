from __future__ import annotations

import pickle
import json
import os
from openbabel import pybel
from rdkit.Chem import Mol

from typing import Dict, List, Tuple, Set, Iterable, Optional
from numpy.typing import NDArray

from .density_map import DensityMapGraph, DensityMapNode, DensityMapNodeCluster, DensityMapEdge


INTERACTION_TO_PHARMACOPHORE = {
    'Hydrophobic': 'Hydrophobic',
    'PiStacking_P': 'Aromatic',
    'PiStacking_T': 'Aromatic',
    'PiCation_lring': 'Aromatic',
    'PiCation_pring': 'Cation',
    'HBond_pdon': 'HBond_acceptor',
    'HBond_ldon': 'HBond_donor',
    'SaltBridge_pneg': 'Cation',
    'SaltBridge_lneg': 'Anion',
    'XBond': 'Halogen',
}


# NOTE: Pickle-Friendly Object
class PharmacophoreModel():
    def __init__(self):
        self.nodes: List[ModelNode]
        self.edges: List[ModelEdge]
        self.node_dict: Dict[str, List[ModelNode]]
        self.node_cluster_dict: Dict[str, List[ModelNodeCluster]]
        self.node_clusters: List[ModelNodeCluster]

    @classmethod
    def create(
        cls,
        center: Tuple[float, float, float],
        resolution: float,
        size: int,
        density_maps: List[dict],
    ):
        graph = DensityMapGraph(center, resolution, size)
        for node in density_maps:
            graph.add_node(node['type'], node['position'], node['score'], node['map'])
        graph.setup()

        model = cls()
        model.nodes = [ModelNode.create(model, node) for node in graph.nodes]
        model.edges = [ModelEdge.create(model, edge) for edge in graph.edges]
        for node in model.nodes:
            node.setup()
        model.node_dict = {
            typ: [model.nodes[node.index] for node in node_list]
            for typ, node_list in graph.node_dict.items()
        }
        model.node_cluster_dict = {
            typ: [ModelNodeCluster.create(model, cluster) for cluster in cluster_list]
            for typ, cluster_list in graph.node_cluster_dict.items()
        }
        model.node_clusters = []
        for node_cluster_list in model.node_cluster_dict.values():
            model.node_clusters.extend(node_cluster_list)
        del graph
        return model

    def scoring(
        self,
        ligand_pbmol: pybel.Molecule,
        ligand_rbmol: Optional[Mol] = None,
        atom_positions: Optional[NDArray] = None,
        conformer_axis: Optional[int] = None,
    ) -> float:
        """Scoring Function

        Args:
            ligand_pbmol: pybel.Molecule
            ligand_rdmol: Chem.Mol | None
            atom_positions: List[NDArray[np.float32]] | NDArray[np.float32] | None
            conformer_axis: Optional[int]

            case: atom_positions: NDArray[np.float32]
                i) conformer_axis is 0 or None
                    atom_positions: (N_conformers, N_atoms, 3)
                ii) conformer_axis is 1
                    atom_positions: (N_atoms, N_conformers, 3)
            case: atom_positions: None
                Using RDKit Conformer informations
        """
        from .ligand import Ligand
        from .graph_match import GraphMatcher
        ligand = Ligand(ligand_pbmol, ligand_rbmol, atom_positions, conformer_axis)
        matcher = GraphMatcher(self, ligand)
        return matcher.scoring()

    def save(self, save_path: str):
        extension = os.path.splitext(save_path)[-1]
        state = self.__getstate__()
        if extension == '.pkl':
            with open(save_path, 'wb') as w:
                pickle.dump(state, w)
        elif extension == '.json':
            with open(save_path, 'w') as w:
                json.dump(state, w, indent=2)
        else:
            raise NotImplementedError

    @classmethod
    def load(cls, save_path: str):
        extension = os.path.splitext(save_path)[-1]
        if extension == '.pkl':
            with open(save_path, 'rb') as f:
                state = pickle.load(f)
        elif extension == '.json':
            with open(save_path) as f:
                state = json.load(f)
        else:
            raise NotImplementedError
        model = cls()
        model.__setstate__(state)
        return model

    def __getstate__(self):
        state = dict(
            nodes=[node.get_kwargs() for node in self.nodes],
            edges=[edge.get_kwargs() for edge in self.edges],
            node_cluster_dict={typ: [cluster.get_kwargs() for cluster in cluster_list] for typ, cluster_list in self.node_cluster_dict.items()},
            node_dict={typ: [node.index for node in nodes] for typ, nodes in self.node_dict.items()},
        )
        return state

    def __setstate__(self, state):
        self.nodes = [ModelNode(self, **kwargs) for kwargs in state['nodes']]
        self.edges = [ModelEdge(self, **kwargs) for kwargs in state['edges']]
        for node in self.nodes:
            node.setup()
        self.node_dict = {
            typ: [self.nodes[index] for index in indices] for typ, indices in state['node_dict'].items()
        }
        self.node_cluster_dict = {
            typ: [ModelNodeCluster(self, **kwargs) for kwargs in cluster_list]
            for typ, cluster_list in state['node_cluster_dict'].items()
        }
        self.node_clusters: List[ModelNodeCluster] = []
        for node_cluster_list in self.node_cluster_dict.values():
            self.node_clusters.extend(node_cluster_list)


class ModelNodeCluster():
    def __init__(
        self,
        graph: PharmacophoreModel,
        cluster_type: str,
        node_indices: Iterable[int],
        node_types: Iterable[str],
        center: Tuple[float, float, float],
        size: float,
    ):
        self.type: str = cluster_type
        self.nodes: Set[ModelNode] = {graph.nodes[index] for index in node_indices}
        self.node_indices: Set[int] = set(node_indices)
        self.node_types: Set[str] = set(node_types)

        self.center: Tuple[float, float, float] = center
        self.size: float = size

    @classmethod
    def create(cls, graph: PharmacophoreModel, cluster: DensityMapNodeCluster) -> ModelNodeCluster:
        return cls(
            graph,
            cluster.type,
            {node.index for node in cluster.nodes},
            {INTERACTION_TO_PHARMACOPHORE[node.type] for node in cluster.nodes},
            cluster.center,
            cluster.size
        )

    def __repr__(self):
        return f'ModelCluster({self.type})[{self.nodes}]'

    def get_kwargs(self):
        return dict(
            cluster_type=self.type,
            node_indices=tuple(self.node_indices),
            node_types=tuple(self.node_types),
            center=self.center,
            size=self.size,
        )


class ModelNode():
    def __init__(
        self,
        graph: PharmacophoreModel,
        index: int,
        type: str,
        interaction_type: str,
        hotspot_position: Tuple[float, float, float],
        score: float,
        center: Tuple[float, float, float],
        radius: float,
        neighbor_edge_dict: Dict[int, int],
        overlapped_nodes: List[int],
    ):
        self.graph: PharmacophoreModel = graph
        self.index: int = index
        self.type: str = type
        self.interaction_type: str = interaction_type
        self.hotspot_position: Tuple[float, float, float] = hotspot_position
        self.score: float = score
        self.center: Tuple[float, float, float] = center
        self.radius: float = radius

        self._neighbor_edge_dict: Dict[int, int] = neighbor_edge_dict
        self._overlapped_nodes: List[int] = overlapped_nodes
        self.neighbor_edge_dict: Dict[ModelNode, ModelEdge]
        self.overlapped_nodes: List[ModelNode]

    def setup(self):
        self.neighbor_edge_dict = {
            self.graph.nodes[int(node_index)]: self.graph.edges[edge_index]     # json save key as str, so type conversion is needed.
            for node_index, edge_index in self._neighbor_edge_dict.items()
        }
        self.overlapped_nodes = [self.graph.nodes[node_index] for node_index in self._overlapped_nodes]

    @classmethod
    def create(cls, graph: PharmacophoreModel, node: DensityMapNode) -> ModelNode:
        x, y, z = node.center.tolist()
        center = (x, y, z)
        return cls(
            graph,
            node.index,
            INTERACTION_TO_PHARMACOPHORE[node.type],
            node.type,
            node.hotspot_position,
            node.score,
            center,
            node.radius,
            {neighbor.index: edge.index for neighbor, edge in node.neighbor_edge_dict.items()},
            [node.index for node in node.overlapped_nodes],
        )

    def __hash__(self):
        return self.index

    def get_kwargs(self):
        return dict(
            index=self.index,
            type=self.type,
            interaction_type=self.interaction_type,
            hotspot_position=self.hotspot_position,
            score=self.score,
            center=self.center,
            radius=self.radius,
            neighbor_edge_dict=self._neighbor_edge_dict,
            overlapped_nodes=self._overlapped_nodes
        )

    def __repr__(self):
        return f'ModelNode({self.index})[{self.type}]'


class ModelEdge():
    def __init__(
        self,
        graph: PharmacophoreModel,
        index: int,
        node_indices: Tuple[int, int],
        edge_type: Tuple[str, str],
        distance_mean: float,
        distance_std: float,
    ):
        self.graph: PharmacophoreModel = graph
        self.index: int = index
        self.nodes: Tuple[ModelNode, ModelNode] = (self.graph.nodes[node_indices[0]], self.graph.nodes[node_indices[1]])
        self.node_indices: Tuple[int, int] = node_indices
        self.type: Tuple[str, str] = edge_type
        self.distance_mean: float = distance_mean
        self.distance_std: float = distance_std

    @classmethod
    def create(cls, graph: PharmacophoreModel, edge: DensityMapEdge) -> ModelEdge:
        return cls(
            graph,
            edge.index,
            edge.node_indices,
            edge.type,
            edge.distance_mean,
            edge.distance_std,
        )

    def __hash__(self):
        return self.index

    def get_kwargs(self):
        return dict(
            index=self.index,
            node_indices=self.node_indices,
            edge_type=self.type,
            distance_mean=self.distance_mean,
            distance_std=self.distance_std,
        )
