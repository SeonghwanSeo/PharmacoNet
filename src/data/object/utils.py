import numpy as np
import math

from openbabel.pybel import ob

from typing import Sequence, Tuple, Union
from numpy.typing import NDArray


def check_in_cutoff(coords, neighbor_coords_list, cutoff: float):
    """
    coords: (3,)
    neighbor_coords: (N, 3)
    cutoff: scalar
    """
    x1, y1, z1 = coords
    cutoff_square = cutoff**2
    for neighbor_coords in neighbor_coords_list:
        x2, y2, z2 = neighbor_coords
        distance_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
        if distance_sq < cutoff_square:
            return True
    return False


def angle_btw_vectors(vec1: NDArray, vec2: NDArray, degree=True, normalized=False) -> float:
    if np.array_equal(vec1, vec2):
        return 0.0
    if normalized:
        cosval = np.dot(vec1, vec2)
    else:
        cosval = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = math.acos(np.clip(cosval, -1, 1))
    return math.degrees(angle) if degree else angle


def vector(p1: Union[Sequence[float], NDArray], p2: Union[Sequence[float], NDArray]) -> NDArray:
    return np.subtract(p2, p1)


def euclidean3d(p1: Union[Sequence[float], NDArray], p2: Union[Sequence[float], NDArray]) -> float:
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))


def normalize(vec: NDArray) -> NDArray:
    norm = np.linalg.norm(vec)
    assert norm > 0, "vector size is zero"
    return vec / norm


def projection(point: Union[Sequence[float], NDArray], origin: Union[Sequence[float], NDArray],
               normal: NDArray) -> NDArray:
    """
    point: point to be projected
    normal, orig: normal vector & origin of projection plane
    """
    c = np.dot(normal, np.subtract(point, origin))
    return np.subtract(point, c * normal)


def ob_coords(obatom: ob.OBAtom) -> Tuple[float, float, float]:
    return (obatom.x(), obatom.y(), obatom.z())
