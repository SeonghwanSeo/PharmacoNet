import math
import numpy as np
import numba as nb
import itertools

from typing import Tuple, List, Tuple
from numpy.typing import NDArray

from .ligand import LigandNode
from .pharmacophore_model import ModelNode


DISTANCE_SIGMA_THRESHOLD = 2.
PASS_THRESHOLD = 0.5


@nb.njit('void(float32[::1],float32[:, :, ::1],float32[::1],float32[::1],float32[::1],int16[::1])', fastmath=True, cache=True)
def __numba_run(
    distances: NDArray[np.float32],
    mean_stds: NDArray[np.float32],
    weights1: NDArray[np.float32],
    weights2: NDArray[np.float32],
    score_array: NDArray[np.float32],
    fail_array: NDArray[np.int16],
):
    """Scoring Function

    Args:
        distances: [C,]
        mean_stds: [M, N, 2]
        weights1: [M,]
        weights2: [N,]
        score_array: [C,]
        fail_array: [C,]
    """
    assert mean_stds.shape[0] == weights1.shape[0]
    assert mean_stds.shape[1] == weights2.shape[0]
    assert distances.shape[0] == score_array.shape[0] == fail_array.shape[0]

    num_match: int
    pass_threshold: float

    W1: float
    W1: float
    normalize_coeff: float
    score_coeff: float

    num_pass: int
    sigma_sq: float
    likelihood: float
    _likelihood: float

    M = weights1.shape[0]
    N = weights2.shape[0]
    C = distances.shape[0]

    num_match = M * N
    pass_threshold = num_match * 0.5    # PASS_THRESHOLD

    # NOTE: Coefficient Calculation
    W1 = sum(weights1)
    W2 = sum(weights2)
    normalize_coeff = 1 / (W1 * W2)
    score_coeff = (W1 * W2) / num_match

    for c in range(C):
        d = distances[c]
        num_pass = 0
        likelihood = 0.
        for m in range(M):
            w1 = weights1[m]
            _likelihood = 0.
            for n in range(N):
                w2 = weights2[n]
                mu_std = mean_stds[m, n]
                mu = mu_std[0]
                std = mu_std[1]
                sigma_sq = ((d - mu) / std) ** 2
                num_pass += (sigma_sq < 4.)  # abs(sigma) < 2.0 A (DISTANCE_SIGMA_THRESHOLD)
                _likelihood += w2 / std * math.exp(-0.5 * sigma_sq)
            likelihood += w1 * _likelihood

        score_array[c] += likelihood * normalize_coeff * score_coeff
        fail_array[c] += num_pass < pass_threshold


def __get_distance_mean_std(model_node1: ModelNode, model_node2: ModelNode) -> Tuple[float, float]:
    edge = model_node1.neighbor_edge_dict[model_node2]
    return edge.distance_mean, edge.distance_std


def scoring_matching_pair(
    cluster_node_match_list1: List[Tuple[LigandNode, List[ModelNode], NDArray[np.float32]]],
    cluster_node_match_list2: List[Tuple[LigandNode, List[ModelNode], NDArray[np.float32]]],
    num_conformers: int,
) -> Tuple[float, ...]:

    match_threshold = len(cluster_node_match_list1) * len(cluster_node_match_list2) * (1 - PASS_THRESHOLD)

    match_scores = np.zeros((num_conformers,), dtype=np.float32)
    num_fails = np.zeros((num_conformers,), dtype=np.int16)
    for ligand_node1, model_node_list1, weights1 in cluster_node_match_list1:
        for ligand_node2, model_node_list2, weights2 in cluster_node_match_list2:
            ligand_edge = ligand_node1.neighbor_edge_dict[ligand_node2]
            distances = ligand_edge.distances

            mean_stds = np.array([
                [__get_distance_mean_std(model_node1, model_node2) for model_node2 in model_node_list2]
                for model_node1 in model_node_list1
            ], dtype=np.float32)   # [M, N, 2]
            __numba_run(
                distances,
                mean_stds,
                weights1,
                weights2,
                match_scores,
                num_fails
            )
            if min(num_fails) > match_threshold:
                return (-1,) * num_conformers

    return tuple(float(score) if num_fail <= match_threshold else -1 for score, num_fail in zip(match_scores, num_fails))


def scoring_matching_self(
    cluster_node_match_list: List[Tuple[LigandNode, List[ModelNode], NDArray[np.float32]]],
    num_conformers: int,
) -> Tuple[float, ...]:
    match_scores = np.zeros((num_conformers,), dtype=np.float32)
    num_fails = np.zeros((num_conformers,), dtype=np.int16)
    for match1, match2 in itertools.combinations(cluster_node_match_list, 2):
        ligand_node1, model_node_list1, weights1 = match1
        ligand_node2, model_node_list2, weights2 = match2

        ligand_edge = ligand_node1.neighbor_edge_dict[ligand_node2]
        distances = ligand_edge.distances

        mean_stds = np.array([
            [__get_distance_mean_std(model_node1, model_node2) for model_node2 in model_node_list2]
            for model_node1 in model_node_list1
        ], dtype=np.float32)   # [M, N, 2]
        __numba_run(
            distances,
            mean_stds,
            weights1,
            weights2,
            match_scores,
            num_fails
        )

    return tuple(match_scores.tolist())
