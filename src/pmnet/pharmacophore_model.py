from __future__ import annotations

import json
import os
import pickle
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from openbabel import pybel

from pmnet.scoring.graph_match import GraphMatcher
from pmnet.scoring.ligand import Ligand
from pmnet.typing import HotspotInfo
from pmnet.utils.density_map import (
    DensityMapEdge,
    DensityMapGraph,
    DensityMapNode,
    DensityMapNodeCluster,
)

INTERACTION_TO_PHARMACOPHORE: dict[str, str] = {
    "Hydrophobic": "Hydrophobic",
    "PiStacking_P": "Aromatic",
    "PiStacking_T": "Aromatic",
    "PiCation_lring": "Aromatic",
    "PiCation_pring": "Cation",
    "HBond_pdon": "HBond_acceptor",
    "HBond_ldon": "HBond_donor",
    "SaltBridge_pneg": "Cation",
    "SaltBridge_lneg": "Anion",
    "XBond": "Halogen",
}


INTERACTION_TO_HOTSPOT: dict[str, str] = {
    "Hydrophobic": "Hydrophobic",
    "PiStacking_P": "Aromatic",
    "PiStacking_T": "Aromatic",
    "PiCation_lring": "Cation",
    "PiCation_pring": "Aromatic",
    "HBond_pdon": "HBond_donor",
    "HBond_ldon": "HBond_acceptor",
    "SaltBridge_pneg": "Anion",
    "SaltBridge_lneg": "Cation",
    "XBond": "Halogen",
}


# NOTE: Pickle-Friendly Object
class PharmacophoreModel:
    def __init__(self):
        self.pdbblock: str
        self.nodes: list[ModelNode]
        self.edges: list[ModelEdge]
        self.node_dict: dict[str, list[ModelNode]]
        self.node_cluster_dict: dict[str, list[ModelNodeCluster]]
        self.node_clusters: list[ModelNodeCluster]

    def scoring_pbmol(
        self,
        ligand_pbmol: pybel.Molecule,
        atom_positions: list[NDArray[np.float32]] | NDArray[np.float32],
        conformer_axis: int | None = None,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Scoring Function

        Args:
            ligand_pbmol: pybel.Molecule
            atom_positions: list[NDArray[np.float32]] | NDArray[np.float32] | None
            conformer_axis: int | None
            weights: dict[str, float] | None

            i) conformer_axis is 0 or None
                atom_positions: (N_conformers, N_atoms, 3)
            ii) conformer_axis is 1
                atom_positions: (N_atoms, N_conformers, 3)
        """
        ligand = Ligand(ligand_pbmol, atom_positions, conformer_axis)
        return self._scoring(ligand, weights)

    def scoring_file(
        self,
        ligand_file: str | Path,
        weights: dict[str, float] | None = None,
        num_conformers: int | None = None,
    ) -> float:
        ligand = Ligand.load_from_file(ligand_file, num_conformers)
        return self._scoring(ligand, weights)

    def scoring_smiles(
        self,
        ligand_smiles: str,
        num_conformers: int,
        weights: dict[str, float] | None = None,
    ) -> float:
        ligand = Ligand.load_from_smiles(ligand_smiles, num_conformers)
        return self._scoring(ligand, weights)

    def _scoring(
        self,
        ligand: Ligand,
        weights: dict[str, float] | None = None,
    ) -> float:
        return GraphMatcher(self, ligand, weights).run()

    @classmethod
    def create(
        cls,
        pdbblock: str,
        center: tuple[float, float, float] | NDArray,
        hotspot_infos: list[HotspotInfo],
        resolution: float = 0.5,
        size: int = 64,
    ):
        assert len(center) == 3
        if not isinstance(center, tuple):
            x, y, z = center.tolist()
            center = (x, y, z)
        graph = DensityMapGraph(center, resolution, size)
        for node in hotspot_infos:
            x, y, z = node.position
            assert node.density_map is not None, "Density map must be provided for each hotspot."
            graph.add_node(
                node.nci_type,
                (x, y, z),
                float(node.score),
                node.density_map.numpy(),
            )
        graph.setup()

        model = cls()
        model.pdbblock = pdbblock
        model.nodes = [ModelNode.create(model, node) for node in graph.nodes]
        model.edges = [ModelEdge.create(model, edge) for edge in graph.edges]
        for node in model.nodes:
            node.setup()
        model.node_dict = {
            typ: [model.nodes[node.index] for node in node_list] for typ, node_list in graph.node_dict.items()
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

    def save(self, save_path: str | Path):
        extension = os.path.splitext(save_path)[-1]
        state = self.__getstate__()
        if extension == ".pm":
            with open(save_path, "wb") as w:
                pickle.dump(state, w)
        elif extension == ".json":
            with open(save_path, "w") as w:
                json.dump(state, w, indent=2)
        else:
            raise NotImplementedError

    @classmethod
    def load(cls, save_path: str | Path):
        extension = os.path.splitext(save_path)[-1]
        if extension == ".pm":
            with open(save_path, "rb") as f:
                state = pickle.load(f)
        elif extension == ".json":
            with open(save_path) as f:
                state = json.load(f)
        else:
            raise NotImplementedError
        model = cls()
        model.__setstate__(state)
        return model

    def __getstate__(self):
        state = dict(
            pdbblock=self.pdbblock,
            nodes=[node.get_kwargs() for node in self.nodes],
            edges=[edge.get_kwargs() for edge in self.edges],
            node_cluster_dict={
                typ: [cluster.get_kwargs() for cluster in cluster_list]
                for typ, cluster_list in self.node_cluster_dict.items()
            },
            node_dict={typ: [node.index for node in nodes] for typ, nodes in self.node_dict.items()},
        )
        return state

    def __setstate__(self, state):
        self.pdbblock = state.get("pdbblock")
        self.nodes = [ModelNode(self, **kwargs) for kwargs in state["nodes"]]
        self.edges = [ModelEdge(self, **kwargs) for kwargs in state["edges"]]
        for node in self.nodes:
            node.setup()
        self.node_dict = {typ: [self.nodes[index] for index in indices] for typ, indices in state["node_dict"].items()}
        self.node_cluster_dict = {
            typ: [ModelNodeCluster(self, **kwargs) for kwargs in cluster_list]
            for typ, cluster_list in state["node_cluster_dict"].items()
        }
        self.node_clusters = []
        for node_cluster_list in self.node_cluster_dict.values():
            self.node_clusters.extend(node_cluster_list)


class ModelNodeCluster:
    def __init__(
        self,
        graph: PharmacophoreModel,
        cluster_type: str,
        node_indices: Iterable[int],
        node_types: Iterable[str],
        center: tuple[float, float, float],
        size: float,
    ):
        self.type: str = cluster_type
        self.nodes: set[ModelNode] = {graph.nodes[index] for index in node_indices}
        self.node_indices: set[int] = set(node_indices)
        self.node_types: set[str] = set(node_types)

        self.center: tuple[float, float, float] = center
        self.size: float = size

    @classmethod
    def create(cls, graph: PharmacophoreModel, cluster: DensityMapNodeCluster) -> ModelNodeCluster:
        return cls(
            graph,
            cluster.type,
            {node.index for node in cluster.nodes},
            {INTERACTION_TO_PHARMACOPHORE[node.type] for node in cluster.nodes},
            cluster.center,
            cluster.size,
        )

    def __repr__(self):
        return f"ModelCluster({self.type})[{self.nodes}]"

    def get_kwargs(self):
        return dict(
            cluster_type=self.type,
            node_indices=tuple(self.node_indices),
            node_types=tuple(self.node_types),
            center=self.center,
            size=self.size,
        )


class ModelNode:
    def __init__(
        self,
        graph: PharmacophoreModel,
        index: int,
        type: str,
        interaction_type: str,
        hotspot_position: tuple[float, float, float],
        score: float,
        center: tuple[float, float, float],
        radius: float,
        neighbor_edge_dict: dict[int, int],
        overlapped_nodes: list[int],
    ):
        self.graph: PharmacophoreModel = graph
        self.index: int = index
        self.type: str = type
        self.interaction_type: str = interaction_type
        self.hotspot_position: tuple[float, float, float] = hotspot_position
        self.score: float = score
        self.center: tuple[float, float, float] = center
        self.radius: float = radius

        self._neighbor_edge_dict: dict[int, int] = neighbor_edge_dict
        self._overlapped_nodes: list[int] = overlapped_nodes
        self.neighbor_edge_dict: dict[ModelNode, ModelEdge]
        self.overlapped_nodes: list[ModelNode]

    def setup(self):
        self.neighbor_edge_dict = {
            self.graph.nodes[int(node_index)]: self.graph.edges[
                edge_index
            ]  # json save key as str, so type conversion is needed.
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
            overlapped_nodes=self._overlapped_nodes,
        )

    def __repr__(self):
        return f"ModelNode({self.index})[{self.type}]"


class ModelEdge:
    def __init__(
        self,
        graph: PharmacophoreModel,
        index: int,
        node_indices: tuple[int, int],
        edge_type: tuple[str, str],
        distance_mean: float,
        distance_std: float,
    ):
        self.graph: PharmacophoreModel = graph
        self.index: int = index
        self.nodes: tuple[ModelNode, ModelNode] = (
            self.graph.nodes[node_indices[0]],
            self.graph.nodes[node_indices[1]],
        )
        self.node_indices: tuple[int, int] = node_indices
        self.type: tuple[str, str] = edge_type
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
