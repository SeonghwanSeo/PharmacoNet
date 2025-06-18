from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from pmnet.data.constant import INTERACTION_LIST

OVERLAP_DISTANCE = 1.5
CLUSTER_DISTANCE = 3.0


def coords_to_position(coords, center, resolution, size) -> tuple[float, float, float]:
    x_center, y_center, z_center = center
    x, y, z = coords
    x_origin = x_center - (resolution * (size - 1) / 2)
    y_origin = y_center - (resolution * (size - 1) / 2)
    z_origin = z_center - (resolution * (size - 1) / 2)
    x_pos = float(x_origin + x * resolution)
    y_pos = float(y_origin + y * resolution)
    z_pos = float(z_origin + z * resolution)
    return (x_pos, y_pos, z_pos)


class DensityMapGraph:
    def __init__(
        self,
        center: tuple[float, float, float],
        resolution: float = 0.5,
        size: int = 64,
    ):
        self.center: tuple[float, float, float] = center
        self.resolution: float = resolution
        self.size: int = size

        self.nodes: list[DensityMapNode] = []
        self.edges: list[DensityMapEdge] = []
        self.node_dict: dict[str, list[DensityMapNode]] = {typ: [] for typ in INTERACTION_LIST}
        self.edge_dict_nodes: dict[tuple[DensityMapNode, DensityMapNode], DensityMapEdge] = {}
        self.edge_dict_indices: dict[tuple[int, int], DensityMapEdge] = {}

        self.node_clusters: list[DensityMapNodeCluster] = []
        self.node_cluster_dict: dict[str, list[DensityMapNodeCluster]] = dict(
            Cation=[], Anion=[], HBond=[], Aromatic=[], Hydrophobic=[], Halogen=[]
        )

    def add_node(
        self,
        node_type: str,
        hotspot_position: tuple[float, float, float],
        score: float,
        mask: NDArray[np.float32 | np.float64],
    ) -> list[DensityMapNode]:
        new_node_list = []
        for grids, grid_scores in self.__extract_pharmacophores(mask):
            grids, grid_scores = np.array(grids), np.array(grid_scores)
            if len(grids) < 8:
                continue
            new_node = DensityMapNode(self, hotspot_position, node_type, score, grids, grid_scores)
            self.nodes.append(new_node)
            self.node_dict[node_type].append(new_node)
            new_node_list.append(new_node)
            for node in self.nodes:  # Add self loop
                edge = new_node.add_neighbors(node)
                self.edges.append(edge)
                self.edge_dict_nodes[(node, new_node)] = edge
                self.edge_dict_nodes[(new_node, node)] = edge
                self.edge_dict_indices[(node.index, new_node.index)] = edge
                self.edge_dict_indices[(new_node.index, node.index)] = edge
        return new_node_list

    def setup(self):
        self.__clustering()

    @staticmethod
    def __extract_pharmacophores(
        mask,
    ) -> Iterator[tuple[list[tuple[int, int, int]], list[float]]]:
        """Get Node From Mask by Clustering Algorithm

        Args:
            mask (NDArray): FloatArray[D, H, W]

        Yields:
            coordinates: list[tuple[int, int, int]] - [(x, y, z)]
            grid_scores: list[float] - [score]
        """
        x_indices, y_indices, z_indices = np.where(mask > 0.0)
        points = {(int(x), int(y), int(z)) for x, y, z in zip(x_indices, y_indices, z_indices, strict=False)}
        while len(points) > 0:
            point = x, y, z = points.pop()
            cluster = [point]
            scores = [float(mask[x, y, z])]
            search_center = cluster
            for x, y, z in search_center:
                new_center = []
                for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    new_point = _x, _y, _z = (x + dx, y + dy, z + dz)
                    if new_point in points:
                        new_center.append(new_point)
                        scores.append(float(mask[_x, _y, _z]))
                        points.remove(new_point)
                cluster += new_center
                search_center = new_center
            yield cluster, scores

    def __clustering(self):
        def are_nodes_close(node1: DensityMapNode, node2: DensityMapNode):
            edge = self.edge_dict_nodes[node1, node2]
            return edge.distance_mean < CLUSTER_DISTANCE

        # NOTE: Cation, Anion, Aromatic Aromatic
        GROUP_CLUSTER_CONFIGS = [
            {
                "name": "Cation",
                "major_type": ("SaltBridge_pneg", "PiCation_pring"),
                "minor_type": "HBond",
            },
            {"name": "Anion", "major_type": "SaltBridge_lneg", "minor_type": "HBond"},
            {
                "name": "Aromatic",
                "major_type": ("PiStacking", "PiCation_lring"),
                "minor_type": "Hydrophobic",
            },
        ]
        used_nodes = set()
        for node in self.nodes:
            if node in used_nodes:
                continue
            for CONFIG in GROUP_CLUSTER_CONFIGS:
                if node.type.startswith(CONFIG["major_type"]):
                    cluster_nodes = {node}
                    # NOTE: Add Overlapped Nodes
                    cluster_nodes.update(
                        overlapped_node
                        for overlapped_node in node.overlapped_nodes
                        if overlapped_node.type.startswith(CONFIG["major_type"])
                    )
                    # NOTE: Add Dependency Nodes
                    cluster_nodes.update(
                        _node
                        for _node in self.nodes
                        if _node.type.startswith(CONFIG["minor_type"])
                        and any(are_nodes_close(_node, cluster_node) for cluster_node in cluster_nodes)
                    )
                    used_nodes.update(cluster_nodes)
                    # NOTE: Add New Cluster
                    cluster = DensityMapNodeCluster(self, cluster_nodes, CONFIG["name"])
                    self.node_cluster_dict[CONFIG["name"]].append(cluster)
                    break

        # NOTE: HBond, Hydrophobic, Halogen
        SINGLE_CLUSTER_CONFIGS = [
            {"name": "HBond", "type": "HBond"},
            {"name": "Hydrophobic", "type": "Hydrophobic"},
            {"name": "Halogen", "type": "XBond"},
        ]
        for node in self.nodes:
            if node in used_nodes:
                continue
            for CONFIG in SINGLE_CLUSTER_CONFIGS:
                if node.type.startswith(CONFIG["type"]):
                    # NOTE: Create New Cluster - [Node, Close Nodes]
                    cluster_nodes = {
                        _node
                        for _node in self.nodes
                        if _node.type.startswith(CONFIG["type"]) and are_nodes_close(node, _node)
                    }
                    cluster_nodes.add(node)
                    cluster = DensityMapNodeCluster(self, cluster_nodes, CONFIG["name"])
                    used_nodes.update(cluster_nodes)
                    self.node_cluster_dict[CONFIG["name"]].append(cluster)
                    break

        for clusters in self.node_cluster_dict.values():
            self.node_clusters.extend(clusters)


class DensityMapNodeCluster:
    def __init__(
        self,
        graph: DensityMapGraph,
        nodes: set[DensityMapNode],
        cluster_type: str,
    ):
        self.graph = graph
        self.type: str = cluster_type
        self.nodes: set[DensityMapNode] = nodes
        positions = np.array([node.center for node in self.nodes])
        radii = np.array([node.radius * 2 for node in self.nodes])
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center.reshape(1, 3), axis=-1) + radii
        x, y, z = center.tolist()
        self.center: tuple[float, float, float] = (x, y, z)
        self.size: float = np.max(distances).item()

    def __repr__(self):
        return f"DensityMapNodeCluster[{self.type}] [ {self.nodes} ]"


class DensityMapNode:
    def __init__(
        self,
        graph: DensityMapGraph,
        hotspot_position: tuple[float, float, float],
        node_type: str,
        score: float,
        grids: NDArray[np.int_],
        grid_scores: NDArray[np.float64],
    ):
        self.graph: DensityMapGraph = graph
        self.index: int = len(self.graph.nodes)
        self.type: str = node_type
        self.grids = grids

        self.hotspot_position: tuple[float, float, float] = hotspot_position

        self.score: float = score
        center_coords = np.average(grids, axis=0, weights=grid_scores)
        self.center: NDArray[np.float32] = np.array(
            coords_to_position(center_coords, self.graph.center, self.graph.resolution, self.graph.size),
            dtype=np.float32,
        )
        self.radius = (grids.shape[0] / (4 * math.pi / 3)) ** (1 / 3) * self.graph.resolution

        self.neighbor_edge_dict: dict[DensityMapNode, DensityMapEdge] = {}
        self.overlapped_nodes: list[DensityMapNode] = []

    def __hash__(self):
        return self.index

    def __repr__(self):
        return f"DensityMapNode({self.index})[{self.type}]"

    def add_neighbors(self, neighbor: DensityMapNode) -> DensityMapEdge:
        assert neighbor not in self.neighbor_edge_dict
        edge = DensityMapEdge(self.graph, self, neighbor)
        self.neighbor_edge_dict[neighbor] = edge
        neighbor.neighbor_edge_dict[self] = edge

        if edge.overlapped:
            self.overlapped_nodes.append(neighbor)
            neighbor.overlapped_nodes.append(self)

        return edge


class DensityMapEdge:
    """Density Map Edge

    Attributes:
        graph: Parent Graph
        index: Edge Index
        node_indices: (Node1.index, Node2.index)
        nodes: (Node1, Node2)
        type: (Node1.type, Node2.type)
        center_distance: distance between map center
        overlapped: end nodes are overlapped
    """

    def __init__(self, graph, node1, node2):
        self.graph = graph
        self.index = len(self.graph.edges)
        if node2.index < node1.index:
            node1, node2 = node2, node1
        self.node_indices: tuple[int, int] = (node1.index, node2.index)
        self.nodes: tuple[DensityMapNode, DensityMapNode] = (node1, node2)
        type1, type2 = node1.type, node2.type

        self.type: tuple[str, str] = (min(type1, type2), max(type1, type2))
        self.distance_mean: float = np.linalg.norm(node1.center - node2.center).item()
        self.distance_std: float = math.sqrt(node1.radius**2 + node2.radius**2)
        self.overlapped: bool = self.distance_mean < OVERLAP_DISTANCE
