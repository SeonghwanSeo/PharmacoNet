from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Iterator

from .ligand import LigandNodeCluster
from .pharmacophore_model import ModelNodeCluster


# NOTE: TYPE
LigandClusterPair = Tuple[LigandNodeCluster, LigandNodeCluster]
ModelClusterPair = Tuple[ModelNodeCluster, ModelNodeCluster]


class ClusterMatchTree():
    def __init__(
        self,
        model_cluster: Optional[ModelNodeCluster],
        pair_scores: Optional[Dict[int, float]],
        parent: ClusterMatchTree,
    ):
        self.level: int = parent.level + 1
        self.num_matches: int = parent.num_matches + (model_cluster is not None)
        self.parent: ClusterMatchTree = parent
        self.root: ClusterMatchTreeRoot = parent.root
        self.children: List[ClusterMatchTree] = []

        ligand_cluster: LigandNodeCluster = self.root.ligand_cluster_list[self.level]
        self.ligand_cluster: LigandNodeCluster = ligand_cluster
        self.model_cluster: Optional[ModelNodeCluster] = model_cluster

        self.pair_scores: Dict[int, float]
        if model_cluster is not None:
            assert pair_scores is not None
            self_pair_scores = self.root.matching_pair_scores_dict[ligand_cluster, ligand_cluster][model_cluster, model_cluster]
            self.pair_scores = {
                conformer_id: parent.pair_scores[conformer_id] + self_pair_scores[conformer_id] + score
                for conformer_id, score in pair_scores.items()
            }
        else:
            self.pair_scores = parent.pair_scores

    @property
    def score(self) -> float:
        return self.edge_score

    @property
    def edge_score(self) -> float:
        if self.num_matches == 0:
            return 0.
        return max(self.pair_scores.values())

    @property
    def conformer_ids(self):
        return self.pair_scores.keys()

    def dfs_run(
        self,
        match_dict: Dict[LigandNodeCluster, Dict[ModelNodeCluster, Dict[int, float]]]
    ) -> int:
        """recursive function

        Args:
            level: level of new node
            ligand_cluster: ligand cluster according to the level
            model_cluster_dict: candidate model cluster
                ModelCluster: {conformer_id: accumulate_score}
        """
        upd_match_dict: Dict[LigandNodeCluster, Dict[ModelNodeCluster, Dict[int, float]]] = {}
        if self.model_cluster is not None:
            for ligand_cluster, model_cluster_dict in match_dict.items():
                upd_model_cluster_dict = {}
                matching_pair_scores_dict = self.root.matching_pair_scores_dict[self.ligand_cluster, ligand_cluster]
                for model_cluster, conformer_pair_score_dict in model_cluster_dict.items():
                    pair_score_list: Tuple[float, ...] = matching_pair_scores_dict[self.model_cluster, model_cluster]
                    # NOTE: Update Model Cluster List accoring to Validity of Pair (Use only Valid Conformer)
                    upd_conformer_pair_score_dict: Dict[int, float] = {
                        conformer_id: total_score + pair_score_list[conformer_id]
                        for conformer_id, total_score in conformer_pair_score_dict.items()
                        if conformer_id in self.conformer_ids and pair_score_list[conformer_id] > 0
                    }
                    if len(upd_conformer_pair_score_dict) > 0:
                        upd_model_cluster_dict[model_cluster] = upd_conformer_pair_score_dict
                upd_match_dict[ligand_cluster] = upd_model_cluster_dict
        else:
            upd_match_dict = match_dict.copy()

        # NOTE: Add Child
        if self.level < len(self.root.ligand_cluster_list) - 1:
            child_ligand_cluster = self.root.ligand_cluster_list[self.level + 1]
            model_cluster_dict = upd_match_dict.pop(child_ligand_cluster)
            max_num_matches = 0
            for model_cluster, conformer_pair_score_dict in model_cluster_dict.items():
                child = self.add_child(model_cluster, conformer_pair_score_dict)
                num_matches = child.dfs_run(upd_match_dict)
                max_num_matches = max(num_matches, max_num_matches)
            if len(model_cluster_dict) == 0 or (self.num_matches + max_num_matches) < 5:
                child = self.add_child(None, None)
                num_matches = child.dfs_run(upd_match_dict)
                max_num_matches = max(num_matches, max_num_matches)
            return max_num_matches + int(self.model_cluster is not None)
        else:
            return int(self.model_cluster is not None)

    def add_child(self, model_cluster: Optional[ModelNodeCluster], pair_score_dict: Optional[Dict[int, float]]):
        child = ClusterMatchTree(model_cluster, pair_score_dict, self)
        self.children.append(child)
        return child

    def delete(self):
        assert self.level >= 0
        self.parent.children.remove(self)
        del self

    @property
    def size(self) -> int:
        if len(self.children) == 0:
            return 1
        size = 0
        for node in self.children:
            size += node.size
        return size

    @property
    def key(self) -> List[Optional[ModelNodeCluster]]:
        key = []
        node: ClusterMatchTree = self
        while node is not self.root:
            key.append(node.model_cluster)
            node = node.parent
        key.reverse()
        return key

    @property
    def item(self) -> Dict[LigandNodeCluster, Optional[ModelNodeCluster]]:
        node: ClusterMatchTree = self
        graph_match: Dict[LigandNodeCluster, Optional[ModelNodeCluster]] = {}
        while node is not self.root:
            graph_match[node.ligand_cluster] = node.model_cluster
            node = node.parent
        return graph_match

    def iteration(self, level: Optional[int] = None) -> Iterator[ClusterMatchTree]:
        if level is not None:
            yield from self.iteration_level(level)
        else:
            yield from self.iteration_leaf()

    def iteration_level(self, level: int) -> Iterator[ClusterMatchTree]:
        assert self.level <= level
        if self.level < level:
            for node in self.children:
                yield from node.iteration_level(level)
        elif self.level == level:
            yield self

    def iteration_leaf(self) -> Iterator[ClusterMatchTree]:
        if len(self.children) > 0:
            for node in self.children:
                yield from node.iteration_leaf()
        else:
            yield self

    def __tree_repr__(self):
        repr = '  ' * (self.level + 1) + f'- {self.model_cluster}'
        if len(self.children) > 0:
            repr += '  ' * (self.level + 2)
            repr += f'(level {self.level + 1}) {self.children[0].ligand_cluster}\n'
            for child in self.children:
                repr += child.__tree_repr__()
            repr += '\n'
        return repr

    def __repr__(self):
        repr = ''
        tree = self
        while tree is not self.root:
            repr = f'({tree.ligand_cluster}, {tree.model_cluster})\n' + repr
            tree = tree.parent
        return repr


class ClusterMatchTreeRoot(ClusterMatchTree):
    def __init__(
        self,
        ligand_cluster_list: List[LigandNodeCluster],
        cluster_match_dict: Dict[LigandNodeCluster, List[ModelNodeCluster]],
        matching_pair_score_dict: Dict[LigandClusterPair, Dict[ModelClusterPair, Tuple[float, ...]]],
        num_conformers: int,
    ):
        self.root = self
        self.level: int = -1
        self.num_matches: int = 0
        self.num_conformers: int = num_conformers
        self.children: List[ClusterMatchTree] = []
        self.ligand_cluster_list: List[LigandNodeCluster] = ligand_cluster_list
        self.cluster_match_dict: Dict[LigandNodeCluster, List[ModelNodeCluster]] = cluster_match_dict
        self.matching_pair_scores_dict: Dict[Tuple[LigandNodeCluster, LigandNodeCluster], Dict[Tuple[ModelNodeCluster, ModelNodeCluster], Tuple[float, ...]]] = matching_pair_score_dict

        self.model_cluster = None
        self.pair_scores: Dict[int, float] = {conformer_id: 0. for conformer_id in range(num_conformers)}

    def __repr__(self):
        repr = 'Root\n'
        if len(self.children) > 0:
            repr += f'(level {self.level + 1}) {self.children[0].ligand_cluster}\n'
        for child in self.children:
            repr += child.__tree_repr__()
        return repr

    def run(self):
        match_dict = {
            ligand_cluster: {
                model_cluster: {conformer_id: 0. for conformer_id in range(self.num_conformers)}
                for model_cluster in self.cluster_match_dict[ligand_cluster]
            }
            for ligand_cluster in self.ligand_cluster_list
        }
        self.dfs_run(match_dict)
