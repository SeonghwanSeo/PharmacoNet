from __future__ import annotations

import itertools
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from openbabel import pybel
from openbabel.pybel import ob

from .ligand_utils import PharmacophoreNode, get_pharmacophore_nodes


class Ligand:
    def __init__(
        self,
        pbmol: pybel.Molecule,
        atom_positions: list[NDArray[np.float32]] | NDArray[np.float32],
        conformer_axis: int | None = None,
        _unsafe: bool = False,
    ):
        """Ligand Object

        Args:
            pbmol: pybel.Molecule Object
            atom_positions: list[NDArray[np.float32]] | NDArray[np.float32] | None
                case: NDArray[np.float32]
                    i) conformer_axis is 0 or None
                        atom_positions: (N_conformers, N_atoms, 3)
                    ii) conformer_axis is 1
                        atom_positions: (N_atoms, N_conformers, 3)
                case: None
                    Using RDKit Conformer informations
        """
        self.pbmol: pybel.Molecule = pbmol if _unsafe else pbmol.clone
        self.pbmol.removeh()
        self.obmol: ob.OBMol = self.pbmol.OBMol
        self.obatoms: list[ob.OBAtom] = [self.obmol.GetAtom(i + 1) for i in range(self.obmol.NumAtoms())]

        self.num_atoms: int = len(self.obatoms)
        self.num_rotatable_bonds: int = pbmol.OBMol.NumRotors()

        self.atom_positions: NDArray[np.float32]  # [N_atoms, N_conformers, 3]
        if isinstance(atom_positions, list):
            self.atom_positions = np.stack(atom_positions, axis=1, dtype=np.float32)
        elif isinstance(atom_positions, np.ndarray):
            self.atom_positions = np.asarray(atom_positions, dtype=np.float32)
            if conformer_axis in [0, None]:
                self.atom_positions = np.ascontiguousarray(np.moveaxis(self.atom_positions, 0, 1))

        assert self.num_atoms == self.atom_positions.shape[0]
        self.num_conformers: int = self.atom_positions.shape[1]

        self.pharmacophore_nodes: dict[str, list[PharmacophoreNode]] = get_pharmacophore_nodes(self.pbmol)
        self.pharmacophore_list: list[tuple[str, PharmacophoreNode]] = []
        for typ, node_list in self.pharmacophore_nodes.items():
            self.pharmacophore_list.extend((typ, node) for node in node_list)

        self.graph = LigandGraph(self)

    @classmethod
    def load_from_file(cls, filename: str | Path, num_conformers: int | None = None) -> Ligand:
        assert filename is not None
        extension = os.path.splitext(filename)[1]
        assert extension in [".sdf", ".pdb", ".mol2"]
        if num_conformers is None:
            pbmol_list: list[pybel.Molecule] = list(pybel.readfile(extension[1:], str(filename)))
        else:
            assert num_conformers > 0
            pbmol_list: list[pybel.Molecule] = list(
                itertools.islice(pybel.readfile(extension[1:], str(filename)), num_conformers)
            )

        base_pbmol = pbmol_list[0]
        base_pbmol.removeh()
        num_atoms = len(base_pbmol.atoms)
        atom_positions = []
        for pbmol in pbmol_list:
            pbmol.removeh()
            assert len(pbmol.atoms) == num_atoms
            atom_positions.append([atom.coords for atom in pbmol.atoms])
        return cls(base_pbmol, atom_positions, _unsafe=True)

    @classmethod
    def load_from_smiles(cls, smiles: str, num_conformers: int) -> Ligand:
        import tempfile

        from rdkit import Chem
        from rdkit.Chem import rdDistGeom

        fn = tempfile.NamedTemporaryFile(suffix=".sdf").name
        try:
            rdmol: Chem.rdchem.Mol = Chem.MolFromSmiles(smiles)
            rdmol = Chem.AddHs(rdmol)
            rdDistGeom.EmbedMultipleConfs(rdmol, num_conformers, params=rdDistGeom.srETKDGv3())
            with Chem.SDWriter(fn) as w:
                for i in range(rdmol.GetNumConformers()):
                    w.write(rdmol, confId=i)
            out = cls.load_from_file(fn)
            os.unlink(fn)
        except Exception as e:
            print(e)
            os.unlink(fn)
            raise e
        return out


class LigandGraph:
    def __init__(self, ligand_obj: Ligand):
        self.nodes: list[LigandNode] = []
        self.edges: list[LigandEdge] = []
        self.node_dict: dict[str, list[LigandNode]] = {}
        self.edge_dict: dict[tuple[LigandNode, LigandNode], LigandEdge] = {}
        self.node_clusters: list[LigandNodeCluster]
        self.node_cluster_dict: dict[str, list[LigandNodeCluster]] = dict(
            Cation=[], Anion=[], HBond=[], Aromatic=[], Hydrophobic=[], Halogen=[]
        )
        self.__add_nodes(ligand_obj)
        self.__setup_conformer(ligand_obj)
        self.__group_nodes(ligand_obj)
        self.__setup_cluster()

    def __setup_conformer(self, obj):
        assert obj.num_conformers > 0
        self.atom_positions = obj.atom_positions
        self.num_conformers = obj.num_conformers
        for node in self.nodes:
            node.set_positions()
        for edge in self.edges:
            edge.set_distances()

    def __add_nodes(self, obj):
        atom_indices_dict: dict[int | Sequence[int], LigandNode] = dict()
        for pharmacophore_type, pharmacophore_node in obj.pharmacophore_list:
            node = atom_indices_dict.get(pharmacophore_node.atom_indices, None)
            if node is not None:
                node.types.append(pharmacophore_type)
                self.node_dict.setdefault(pharmacophore_type, []).append(node)
            else:
                new_node = LigandNode(
                    self,
                    len(self.nodes),
                    pharmacophore_node.atom_indices,
                    pharmacophore_node.center_indices,
                    pharmacophore_type,
                )
                self.nodes.append(new_node)
                self.node_dict.setdefault(pharmacophore_type, []).append(new_node)
                for node in self.nodes[:-1]:
                    edge = node.add_neighbors(new_node)
                    self.edges.append(edge)
                    self.edge_dict[(node, new_node)] = edge
                    self.edge_dict[(new_node, node)] = edge
                atom_indices_dict[pharmacophore_node.atom_indices] = new_node

    def __group_nodes(self, obj):
        hbond_pharmacophores: dict[int, list[LigandNode]] = {}
        hydrop_pharmacophores: dict[int, list[LigandNode]] = {}

        for node in self.nodes:
            pharmacophore_types = node.types
            if "HBond_acceptor" in pharmacophore_types or "HBond_donor" in pharmacophore_types:
                # NOTE: Clustering Nodes in Same Functional Groups (such as -PO3)
                assert len(node.atom_indices) == 1
                atom_index = next(iter(node.atom_indices))
                obatom = obj.obatoms[atom_index]
                neighbors = [
                    neighbor.GetIdx() - 1 for neighbor in ob.OBAtomAtomIter(obatom) if neighbor.GetAtomicNum() != 1
                ]
                if len(neighbors) == 1:
                    group_nodes = hbond_pharmacophores.setdefault(neighbors[0], [])
                    for _node in group_nodes:
                        node.group_nodes.add(_node)
                        _node.group_nodes.add(node)
                    group_nodes.append(node)

            elif "Hydrophobic" in pharmacophore_types:
                # NOTE: Clustering Nodes in Same Functional Groups (such as -C(CH3)3)
                assert len(node.atom_indices) == 1
                atom_index = next(iter(node.atom_indices))
                obatom = obj.obatoms[atom_index]
                neighbors = [
                    neighbor.GetIdx() - 1 for neighbor in ob.OBAtomAtomIter(obatom) if neighbor.GetAtomicNum() != 1
                ]
                if len(neighbors) == 1:
                    group_nodes = hydrop_pharmacophores.setdefault(neighbors[0], [])
                    for _node in group_nodes:
                        node.group_nodes.add(_node)
                        _node.group_nodes.add(node)
                    group_nodes.append(node)

        hydrophobic_nodes = self.node_dict.get("Hydrophobic", [])
        index_to_node = {next(iter(node.atom_indices)): node for node in hydrophobic_nodes}
        while len(index_to_node) > 0:
            index, node = index_to_node.popitem()
            group_nodes = [node] + list(node.group_nodes)
            group_index = [next(iter(node.atom_indices)) for node in group_nodes]
            for atom_index in group_index:
                obatom = obj.obatoms[atom_index]
                neighbors = [
                    neighbor.GetIdx() - 1 for neighbor in ob.OBAtomAtomIter(obatom) if neighbor.GetAtomicNum() == 6
                ]
                for neighbor_index in neighbors:
                    neighbor_node = index_to_node.pop(neighbor_index, None)
                    if neighbor_node is None or neighbor_index in group_nodes:
                        continue
                    group_index.append(neighbor_index)
                    for node in group_nodes:
                        node.group_nodes.add(neighbor_node)
                        neighbor_node.group_nodes.add(node)
                    group_nodes.append(neighbor_node)

    def __setup_cluster(self):
        # NOTE: set Cluster
        # Cluster: list[LigandNode]
        # The order inside the cluster follows priority.
        in_cluster: set[LigandNode] = set()
        node_cluster_dict: dict[LigandNode, LigandNodeCluster] = {}
        for pharmacophore_type in ["Aromatic", "Cation", "Anion", "Halogen"]:
            for node in self.node_dict.get(pharmacophore_type, []):
                if node in in_cluster:
                    continue
                in_cluster.add(node)
                cluster = LigandNodeCluster(pharmacophore_type)
                cluster.add_new_node(node, "high")
                node_cluster_dict[node] = cluster

        for pharmacophore_type in ["Hydrophobic", "HBond_donor", "HBond_acceptor"]:
            for node in self.node_dict.get(pharmacophore_type, []):
                if node in in_cluster:
                    continue
                in_cluster.add(node)
                add_new_cluster: bool = True
                if len(node.dependence_nodes) > 0:
                    # NOTE: Added to Dependence Nodes (Aromatic Ring, Charged Atoms)
                    cluster = node_cluster_dict[min(node.dependence_nodes)]
                    cluster.add_new_node(node, "low")
                    add_new_cluster = False
                elif len(node.group_nodes) > 0:
                    # NOTE: Clustering Nodes in Same Functional Groups
                    for group_node in node.group_nodes:
                        if group_node in node_cluster_dict:
                            cluster = node_cluster_dict[group_node]
                            cluster.add_new_node(node, "low")
                            add_new_cluster = False
                            break
                if add_new_cluster:
                    if pharmacophore_type.startswith("HBond"):
                        cluster = LigandNodeCluster("HBond")
                    else:
                        cluster = LigandNodeCluster("Hydrophobic")
                    cluster.add_new_node(node, "low")
                    node_cluster_dict[node] = cluster

        self.node_clusters = list(node_cluster_dict.values())
        for cluster in self.node_clusters:
            self.node_cluster_dict[cluster.type].append(cluster)


class LigandNode:
    def __init__(
        self,
        graph: LigandGraph,
        node_index: int,
        atom_indices: int | Sequence[int],
        center_indices: int | Sequence[int],
        node_type: str,
    ):
        """Ligand Pharmacophore Node

        Args:
            graph: root graph
            index: node index
            node_type: node pharmacophoretypes
        """
        self.graph: LigandGraph = graph
        self.index: int = node_index
        self.types: list[str] = [node_type]
        self.atom_indices: set[int] = {atom_indices} if isinstance(atom_indices, int) else set(atom_indices)
        self.center_indices: int | Sequence[int] = center_indices

        self.neighbor_edge_dict: dict[LigandNode, LigandEdge] = {}
        self.group_nodes: set[LigandNode] = set()
        self.dependence_nodes: set[LigandNode] = set()

        self.positions: NDArray[np.float32]  # (N, 3)

    def __repr__(self):
        return f"LigandNode({self.index}){self.types}"

    def set_positions(self):
        if isinstance(self.center_indices, int):
            self.positions = np.asarray(self.graph.atom_positions[self.center_indices], dtype=np.float32)
        else:
            self.positions = np.mean(
                self.graph.atom_positions[self.center_indices, :],
                axis=0,
                dtype=np.float32,
            )

    def add_neighbors(self, neighbor: LigandNode) -> LigandEdge:
        edge = self.neighbor_edge_dict.get(neighbor, None)
        if edge is not None:
            return edge
        edge = LigandEdge(self.graph, self, neighbor)
        self.neighbor_edge_dict[neighbor] = edge
        neighbor.neighbor_edge_dict[self] = edge

        node_types = self.types
        neighbor_node_types = neighbor.types

        def check_type(node_types, *keys):
            return any(typ.startswith(keys) for typ in node_types)

        if check_type(node_types, "Hydrophobic") and check_type(neighbor_node_types, "Aromatic"):
            if self.atom_indices.issubset(neighbor.atom_indices):
                self.dependence_nodes.add(neighbor)
        elif check_type(node_types, "Aromatic") and check_type(neighbor_node_types, "Hydrophobic"):
            if neighbor.atom_indices.issubset(self.atom_indices):
                neighbor.dependence_nodes.add(self)
        elif check_type(node_types, "HBond") and check_type(neighbor_node_types, "Cation", "Anion"):
            if self.atom_indices.issubset(neighbor.atom_indices):
                self.dependence_nodes.add(neighbor)
        elif check_type(node_types, "Cation", "Anion") and check_type(neighbor_node_types, "HBond"):
            if neighbor.atom_indices.issubset(self.atom_indices):
                neighbor.dependence_nodes.add(self)
        return edge

    def __gt__(self, other):
        return self.index > other.index

    def __le__(self, other):
        return self.index < other.index


class LigandEdge:
    def __init__(self, graph, node1, node2):
        self.graph = graph
        self.index = len(self.graph.edges)
        if node2.index < node1.index:
            node1, node2 = node2, node1
        self.indices: tuple[int, int] = (node1.index, node2.index)
        self.nodes: tuple[LigandNode, LigandNode] = (node1, node2)

        self.distances: NDArray[np.float32]  # (N_conformer,)

    def set_distances(self):
        node1, node2 = self.nodes
        self.distances = np.linalg.norm(node1.positions - node2.positions, axis=-1)


class LigandNodeCluster:
    """Ligand Node Cluster
    Property
    - center_node
    - cluster_nodes

    Case
    - one high-priority node:
        center_node = high-priority node
        cluster_nodes = []
    - one high-priority node + multi low-priority nodes:
        center_node = high-priority node
        cluster_nodes = low-priority nodes
    - one low-priority node
        center_node = low-priority node
        cluster_nodes = []
    - multi low-priority nodes:
        center_node = None
        cluster_nodes = low-priority nodes
    """

    def __init__(self, cluster_type: str):
        self.type: str = cluster_type
        self._high_priority_node: LigandNode | None = None
        self._low_priority_nodes: list[LigandNode] = []

        self._node_types: set[str] | None = None
        self._node_indices: list[int] | None = None

        self._positions: NDArray[np.float32] | None = None  # [N_conf, N_node, 3]
        self._center: NDArray[np.float32] | None = None  # [N_conf, 3]
        self._size: NDArray[np.float32] | None = None  # [N_conf,]

    def __iter__(self) -> Iterator[LigandNode]:
        if self._high_priority_node is not None:
            if len(self._low_priority_nodes) > 0:
                yield self._high_priority_node
                yield from self._low_priority_nodes
            else:
                yield self._high_priority_node
        else:
            yield from iter(self._low_priority_nodes)

    def add_new_node(self, node: LigandNode, priority: str):
        assert priority in ("low", "high")
        if priority.startswith("high"):
            self._high_priority_node = node
        else:
            self._low_priority_nodes.append(node)
        self._node_types = None
        self._node_indices = None

    def __repr__(self):
        if self.center_node is not None and len(self.cluster_nodes) > 0:
            return f"LigandNodeCluster({self.type})[ Center: {self.center_node} Cluster: {self.cluster_nodes} ]"
        elif self.center_node is not None:
            return f"LigandNodeCluster({self.type})[ Center: {self.center_node} ]"
        else:
            return f"LigandNodeCluster({self.type})[ Cluster: {self.cluster_nodes} ]"

    @property
    def nodes(self) -> list[LigandNode]:
        return list(iter(self))

    @property
    def node_types(self) -> set[str]:
        if self._node_types is None:
            self._node_types = set()
            for node in self:
                self._node_types.update(node.types)
        return self._node_types

    @property
    def node_indices(self) -> set[str]:
        if self._node_types is None:
            self._node_types = set()
            for node in self:
                self._node_types.update(node.types)
        return self._node_types

    @property
    def center_node(self) -> LigandNode | None:
        if self._high_priority_node is not None:
            return self._high_priority_node
        elif len(self._low_priority_nodes) == 1:
            return self._low_priority_nodes[0]
        else:
            return None

    @property
    def cluster_nodes(self) -> list[LigandNode]:
        if self._high_priority_node is not None:
            return self._low_priority_nodes
        elif len(self._low_priority_nodes) > 1:
            return self._low_priority_nodes
        else:
            return []

    @property
    def positions(self) -> NDArray[np.float32]:
        if self._positions is None:
            self._positions = np.stack([node.positions for node in self.nodes], axis=1)
        return self._positions

    @property
    def center(self) -> NDArray[np.float32]:
        if self._center is None:
            self._center = np.mean(self.positions, axis=1)
        assert self._center is not None
        return self._center

    @property
    def size(self) -> NDArray[np.float32]:
        if self._size is None:
            self._size = np.max(
                np.linalg.norm(self.positions - self.center.reshape(-1, 1, 3), axis=-1),
                axis=-1,
            )
        assert self._size is not None
        return self._size
