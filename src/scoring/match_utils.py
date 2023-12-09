import itertools
import numpy as np
import math

from typing import Tuple, List, Tuple
from numpy.typing import NDArray

from .ligand import LigandNode
from .pharmacophore_model import ModelNode


DISTANCE_SIGMA_THRESHOLD = 2.
PASS_THRESHOLD = 0.5


def scoring_matching_pair(
    cluster_node_match_list1: List[Tuple[LigandNode, List[ModelNode], NDArray[np.float32]]],
    cluster_node_match_list2: List[Tuple[LigandNode, List[ModelNode], NDArray[np.float32]]],
    num_conformers: int,
) -> Tuple[float, ...]:
    match_scores = np.zeros((num_conformers,), dtype=np.float32)
    num_fails = np.zeros((num_conformers,), dtype=np.int16)

    match_threshold = len(cluster_node_match_list1) * len(cluster_node_match_list2) * (1 - PASS_THRESHOLD)

    num_pass = np.empty((num_conformers,), dtype=np.int16)
    likelihood = np.empty((num_conformers,), dtype=np.float32)
    for cluster_node_match1, cluster_node_match2 in itertools.product(cluster_node_match_list1, cluster_node_match_list2):
        ligand_node1, model_node_list1, weights1 = cluster_node_match1
        ligand_node2, model_node_list2, weights2 = cluster_node_match2
        ligand_edge = ligand_node1.neighbor_edge_dict[ligand_node2]
        distances = ligand_edge.distances

        num_match = len(model_node_list1) * len(model_node_list2)
        means = np.array(
            [
                [model_node1.neighbor_edge_dict[model_node2].distance_mean]
                for model_node1, model_node2 in itertools.product(model_node_list1, model_node_list2)
            ],
            dtype=np.float32
        )   # [M*N, 1]
        stds = np.array(
            [
                [model_node1.neighbor_edge_dict[model_node2].distance_std]
                for model_node1, model_node2 in itertools.product(model_node_list1, model_node_list2)
            ],
            dtype=np.float32
        )   # [M*N, 1]
        weights = (weights1.reshape(-1, 1) * weights2.reshape(1, -1)).reshape(-1)   # [M * N]

        weights_sum = sum(weights)
        normalize_coeff = 1 / weights_sum   # / (math.sqrt(2 * math.pi) (skip)
        score_coeff = weights_sum / num_match

        distance_sigma_array = (distances.reshape(1, num_conformers) - means) / stds
        np.sum(np.abs(distance_sigma_array) < DISTANCE_SIGMA_THRESHOLD, axis=0, out=num_pass)
        num_fails += (num_pass < (num_match * PASS_THRESHOLD))
        if min(num_fails) > match_threshold:
            return (-1,) * num_conformers
        np.dot(weights / stds.reshape(-1), np.exp(-0.5 * distance_sigma_array ** 2), out=likelihood)
        match_scores += likelihood * normalize_coeff * score_coeff

    return tuple(float(score) if num_fail <= match_threshold else -1 for score, num_fail in zip(match_scores, num_fails))


def scoring_matching_self(
    cluster_node_match_list: List[Tuple[LigandNode, List[ModelNode], NDArray[np.float32]]],
    num_conformers: int,
) -> Tuple[float, ...]:
    match_scores = np.zeros((num_conformers,), dtype=np.float32)
    likelihood = np.empty((num_conformers,), dtype=np.float32)
    for cluster_node_match1, cluster_node_match2 in itertools.combinations(cluster_node_match_list, 2):
        ligand_node1, model_node_list1, weights1 = cluster_node_match1
        ligand_node2, model_node_list2, weights2 = cluster_node_match2
        ligand_edge = ligand_node1.neighbor_edge_dict[ligand_node2]
        distances = ligand_edge.distances

        num_match = len(model_node_list1) * len(model_node_list2)
        means = np.array(
            [
                [model_node1.neighbor_edge_dict[model_node2].distance_mean]
                for model_node1, model_node2 in itertools.product(model_node_list1, model_node_list2)
            ],
            dtype=np.float32
        )   # [M*N, 1]
        stds = np.array(
            [
                [model_node1.neighbor_edge_dict[model_node2].distance_std]
                for model_node1, model_node2 in itertools.product(model_node_list1, model_node_list2)
            ],
            dtype=np.float32
        )   # [M*N, 1]
        weights = (weights1.reshape(-1, 1) * weights2.reshape(1, -1)).reshape(-1)   # [M*N]
        weights_sum = sum(weights)
        normalize_coeff = 1 / weights_sum   # / (math.sqrt(2 * math.pi) (skip)
        score_coeff = weights_sum / num_match

        distance_sigma_array = (distances.reshape(1, num_conformers) - means) / stds
        np.dot(weights / stds.reshape(-1), np.exp(-0.5 * distance_sigma_array ** 2), out=likelihood)

        match_scores += likelihood * normalize_coeff * score_coeff

    return tuple(match_scores.tolist())
