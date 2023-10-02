from __future__ import annotations

import itertools

import numpy as np
import math

from typing import Dict, List, Tuple, Optional
from numpy.typing import NDArray

from .ligand import Ligand, LigandGraph, LigandNode, LigandNodeCluster
from .pharmacophore_model import PharmacophoreModel, ModelNodeCluster, ModelNode
from .tree import ClusterMatchTreeRoot
try:
    from .match_utils_numba import scoring_matching_pair, scoring_matching_self
except:
    from .match_utils import scoring_matching_pair, scoring_matching_self


# NOTE: TYPE
LigandClusterPair = Tuple[LigandNodeCluster, LigandNodeCluster]
ModelClusterPair = Tuple[ModelNodeCluster, ModelNodeCluster]


# NOTE: Constant
WEIGHTS = dict(
    Cation=8,
    Anion=8,
    Aromatic=8,
    HBond_donor=4,
    HBond_acceptor=4,
    Halogen=4,
    Hydrophobic=1,
)
MAX_DEPTH = 20


def priority_fn(cluster: LigandNodeCluster):
    cluster_size_priority = -len(cluster.nodes)
    cluster_type = cluster.type
    atom_index = min(cluster.nodes[0].atom_indices)
    if cluster_type.startswith('Aromatic'):
        return (0, cluster_size_priority, 0, atom_index)
    elif cluster_type.startswith('Cation'):
        return (0, cluster_size_priority, 1, atom_index)
    elif cluster_type.startswith('Anion'):
        return (0, cluster_size_priority, 2, atom_index)
    elif cluster_type.startswith('HBond'):
        return (1, cluster_size_priority, 0, atom_index)
    elif cluster_type.startswith('Halogen'):
        return (1, cluster_size_priority, 1, atom_index)
    elif cluster_type.startswith('Hydrophobic'):
        return (1, cluster_size_priority, 2, atom_index)
    else:
        raise NotImplementedError


class GraphMatcher():
    def __init__(
        self,
        model: PharmacophoreModel,
        ligand: Ligand,
        max_depth: Optional[int] = MAX_DEPTH,
    ):
        self.model_graph: PharmacophoreModel = model
        self.ligand_graph: LigandGraph = ligand.graph
        self.num_conformers = self.ligand_graph.num_conformers
        self.max_depth = max_depth
        self.cluster_match_dict: Dict[LigandNodeCluster, List[ModelNodeCluster]]
        self.ligand_cluster_list: List[LigandNodeCluster]
        self.node_match_dict: Dict[Tuple[LigandNodeCluster, ModelNodeCluster], List[Tuple[LigandNode, List[ModelNode], NDArray[np.float32]]]]

    def setup(self):
        self.cluster_match_dict = self._get_cluster_match_dict()
        self.ligand_cluster_list = sorted(self.cluster_match_dict.keys(), key=priority_fn)
        if self.max_depth is not None:
            self.ligand_cluster_list = self.ligand_cluster_list[:self.max_depth]
        self.node_match_dict = self._get_node_match_dict()
        self.matching_pair_scores_dict: Dict[LigandClusterPair, Dict[ModelClusterPair, Tuple[float, ...]]] = self._get_pair_scores()

    def evaluate(self):
        self.setup()
        root_tree = self._run()
        return list(root_tree.iteration())

    def scoring(self) -> float:
        self.setup()
        root_tree = self._run()
        return max(leaf.score for leaf in root_tree.iteration())

    def _run(self) -> ClusterMatchTreeRoot:
        root_tree = ClusterMatchTreeRoot(
            self.ligand_cluster_list,
            self.cluster_match_dict,
            self.matching_pair_scores_dict,
            self.ligand_graph.num_conformers
        )
        root_tree.run()
        return root_tree

    def _get_cluster_match_dict(self) -> Dict[LigandNodeCluster, List[ModelNodeCluster]]:
        cluster_match_dict: Dict[LigandNodeCluster, List[ModelNodeCluster]] = {}
        ligand_graph, model_graph = self.ligand_graph, self.model_graph
        for ligand_cluster in ligand_graph.node_clusters:
            match_model_clusters = [
                model_cluster for model_cluster in model_graph.node_clusters
                if len(ligand_cluster.node_types.intersection(model_cluster.node_types)) > 0
            ]
            if len(match_model_clusters) > 0:
                cluster_match_dict[ligand_cluster] = match_model_clusters
        return cluster_match_dict

    def _get_node_match_dict(self) -> Dict[Tuple[LigandNodeCluster, ModelNodeCluster], List[Tuple[LigandNode, List[ModelNode], NDArray[np.float32]]]]:
        def __get_node_match(ligand_node: LigandNode, model_cluster: ModelNodeCluster) -> Tuple[LigandNode, List[ModelNode], NDArray[np.float32]]:
            match_model_nodes = [model_node for model_node in model_cluster.nodes if model_node.type in ligand_node.types]
            weights = np.array([WEIGHTS[model_node.type] * model_node.score for model_node in match_model_nodes], dtype=np.float32)
            return (ligand_node, match_model_nodes, weights)

        node_match_dict = {
            (ligand_cluster, model_cluster): [__get_node_match(ligand_node, model_cluster) for ligand_node in ligand_cluster.nodes]
            for ligand_cluster, model_cluster_list in self.cluster_match_dict.items()
            for model_cluster in model_cluster_list
        }
        node_match_dict = {
            key: [(ligand_node, model_node_list, weights) for ligand_node, model_node_list, weights in node_matches if len(model_node_list) > 0]
            for key, node_matches in node_match_dict.items()
        }
        return node_match_dict

    # NOTE: (Not Use) Code with Readability - same to _get_pair_scores().
    def _get_pair_scores_readability(self) -> Dict[LigandClusterPair, Dict[ModelClusterPair, Tuple[float, ...]]]:
        matching_pair_scores_dict: Dict[LigandClusterPair, Dict[ModelClusterPair, Tuple[float, ...]]] = {
            (ligand_cluster1, ligand_cluster2): {} for ligand_cluster1, ligand_cluster2 in itertools.combinations(self.ligand_cluster_list, 2)
        }

        NO_MATCH_SCORE = (-1,) * self.num_conformers
        for ligand_cluster1, ligand_cluster2 in itertools.combinations(self.ligand_cluster_list, 2):
            ligand_cluster_distance = np.linalg.norm(ligand_cluster1.center - ligand_cluster2.center, axis=-1)
            ligand_cluster_size = ligand_cluster1.size + ligand_cluster2.size

            model_cluster_list1, model_cluster_list2 = self.cluster_match_dict[ligand_cluster1], self.cluster_match_dict[ligand_cluster2]
            for model_cluster1, model_cluster2 in itertools.product(model_cluster_list1, model_cluster_list2):
                (x1, y1, z1), (x2, y2, z2) = model_cluster1.center, model_cluster2.center
                model_cluster_distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
                model_cluster_size = model_cluster1.size + model_cluster2.size

                if min(np.abs(ligand_cluster_distance - model_cluster_distance) - ligand_cluster_size) > model_cluster_size:
                    pair_score = NO_MATCH_SCORE
                else:
                    node_match_list1 = self.node_match_dict[ligand_cluster1, model_cluster1]
                    node_match_list2 = self.node_match_dict[ligand_cluster2, model_cluster2]
                    pair_score = scoring_matching_pair(node_match_list1, node_match_list2, self.num_conformers)
                matching_pair_scores_dict[ligand_cluster1, ligand_cluster2][model_cluster1, model_cluster2] = pair_score

        for ligand_cluster in self.ligand_cluster_list:
            for model_cluster in self.cluster_match_dict[ligand_cluster]:
                node_match_list = self.node_match_dict[ligand_cluster, model_cluster]
                self_pair_score = scoring_matching_self(node_match_list, self.num_conformers)
                matching_pair_scores_dict[ligand_cluster, ligand_cluster][model_cluster, model_cluster] = self_pair_score

        return matching_pair_scores_dict

    # NOTE: Efficient but bad-readable code
    def _get_pair_scores(self) -> Dict[LigandClusterPair, Dict[ModelClusterPair, Tuple[float, ...]]]:
        NO_MATCH_SCORE = (-1,) * self.num_conformers

        def __get_score_dict_outer(ligand_cluster_pair: LigandClusterPair) -> Dict[ModelClusterPair, Tuple[float, ...]]:
            ligand_cluster1, ligand_cluster2 = ligand_cluster_pair
            if ligand_cluster1 is ligand_cluster2:
                return {
                    (model_cluster, model_cluster): scoring_matching_self(self.node_match_dict[ligand_cluster1, model_cluster], self.num_conformers)
                    for model_cluster in self.cluster_match_dict[ligand_cluster1]
                }
            else:
                ligand_cluster_distance = np.linalg.norm(ligand_cluster1.center - ligand_cluster2.center, axis=-1)
                ligand_cluster_size = ligand_cluster1.size + ligand_cluster2.size
                return {
                    model_cluster_pair: __get_score_dict_inner(ligand_cluster_pair, model_cluster_pair, ligand_cluster_distance, ligand_cluster_size)
                    for model_cluster_pair in itertools.product(self.cluster_match_dict[ligand_cluster1], self.cluster_match_dict[ligand_cluster2])
                }

        def __get_score_dict_inner(
            ligand_cluster_pair: LigandClusterPair,
            model_cluster_pair: ModelClusterPair,
            ligand_cluster_distance: NDArray[np.float32],
            ligand_cluster_size: NDArray[np.float32],
        ) -> Tuple[float, ...]:
            ligand_cluster1, ligand_cluster2 = ligand_cluster_pair
            model_cluster1, model_cluster2 = model_cluster_pair
            (x1, y1, z1), (x2, y2, z2) = model_cluster1.center, model_cluster2.center
            model_cluster_distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            model_cluster_size = model_cluster1.size + model_cluster2.size

            if min(np.abs(ligand_cluster_distance - model_cluster_distance) - ligand_cluster_size) > model_cluster_size:
                pair_score = NO_MATCH_SCORE
            else:
                node_match_list1 = self.node_match_dict[ligand_cluster1, model_cluster1]
                node_match_list2 = self.node_match_dict[ligand_cluster2, model_cluster2]
                pair_score = scoring_matching_pair(node_match_list1, node_match_list2, self.num_conformers)
            return pair_score

        matching_pair_scores_dict: Dict[LigandClusterPair, Dict[ModelClusterPair, Tuple[float, ...]]] = {
            ligand_cluster_pair: __get_score_dict_outer(ligand_cluster_pair)
            for ligand_cluster_pair in itertools.combinations_with_replacement(self.ligand_cluster_list, 2)
        }
        return matching_pair_scores_dict
