import numpy as np
import math

from typing import Tuple, List
from numpy.typing import NDArray, ArrayLike

from .objects import Protein
from . import constant as C


def get_token_informations(
    protein_obj: Protein,
) -> Tuple[NDArray[np.float32], NDArray[np.int16]]:
    """get token information

    Args:
        protein_obj: Union[Protein]

    Returns:
        token_positions: [float, (N, 3)] token center positions
        token_classes: [int, (N,)] token interaction type
    """
    num_tokens = \
        len(protein_obj.hydrophobic_atoms_all) + \
        len(protein_obj.rings_all) * 3 + \
        len(protein_obj.hbond_donors_all) + \
        len(protein_obj.hbond_acceptors_all) + \
        len(protein_obj.pos_charged_atoms_all) * 2 + \
        len(protein_obj.neg_charged_atoms_all) + \
        len(protein_obj.xbond_acceptors_all)

    positions: List[Tuple[float, float, float]] = []
    classes: List[int] = []

    # NOTE: Hydrophobic
    positions.extend(tuple(hydrop.coords) for hydrop in protein_obj.hydrophobic_atoms_all)
    classes.extend([C.HYDROPHOBIC] * len(protein_obj.hydrophobic_atoms_all))

    # NOTE: PiStacking_P
    positions.extend(tuple(ring.center) for ring in protein_obj.rings_all)
    classes.extend([C.PISTACKING_P] * len(protein_obj.rings_all))

    # NOTE: PiStacking_T
    positions.extend(tuple(ring.center) for ring in protein_obj.rings_all)
    classes.extend([C.PISTACKING_T] * len(protein_obj.rings_all))

    # NOTE: PiCation_lring
    positions.extend(tuple(cation.center) for cation in protein_obj.pos_charged_atoms_all)
    classes.extend([C.PICATION_LRING] * len(protein_obj.pos_charged_atoms_all))

    # NOTE: PiCation_pring
    positions.extend(tuple(ring.center) for ring in protein_obj.rings_all)
    classes.extend([C.PICATION_PRING] * len(protein_obj.rings_all))

    # NOTE: HBond_ldon
    positions.extend(tuple(acceptor.coords) for acceptor in protein_obj.hbond_acceptors_all)
    classes.extend([C.HBOND_LDON] * len(protein_obj.hbond_acceptors_all))

    # NOTE: HBond_pdon
    positions.extend(tuple(donor.coords) for donor in protein_obj.hbond_donors_all)
    classes.extend([C.HBOND_PDON] * len(protein_obj.hbond_donors_all))

    # NOTE: Saltbridge_lneg
    positions.extend(tuple(cation.center) for cation in protein_obj.pos_charged_atoms_all)
    classes.extend([C.SALTBRIDGE_LNEG] * len(protein_obj.pos_charged_atoms_all))

    # NOTE: Saltbridge_pneg
    positions.extend(tuple(anion.center) for anion in protein_obj.neg_charged_atoms_all)
    classes.extend([C.SALTBRIDGE_PNEG] * len(protein_obj.neg_charged_atoms_all))

    # NOTE: XBond
    positions.extend(tuple(acceptor.O_coords) for acceptor in protein_obj.xbond_acceptors_all)
    classes.extend([C.XBOND] * len(protein_obj.xbond_acceptors_all))

    assert len(positions) == len(classes) == num_tokens
    return (
        np.array(positions, dtype=np.float32),
        np.array(classes, dtype=np.int16),
    )


def get_token_and_filter(
    positions: NDArray[np.float32],
    classes: NDArray[np.int16],
    center: NDArray[np.float32],
    resolution: float,
    dimension: int,
) -> Tuple[NDArray[np.int16], NDArray[np.int16]]:
    """Create token and Filtering valid instances

    Args:
        positions: [float, (N, 3)] token center positions
        classes: [int, (N,)] token interaction type
        center: [float, (3,)] voxel image center
        resolution: voxel image resolution
        dimension: voxel imzge dimension (size)

    Returns:
        token: [int, (N_token, 4)]
        filter: [int, (N_token,)]
    """
    filter = []
    tokens = []
    x_center, y_center, z_center = center
    x_start = x_center - (dimension / 2) * resolution
    y_start = y_center - (dimension / 2) * resolution
    z_start = z_center - (dimension / 2) * resolution
    for i, ((x, y, z), c) in enumerate(zip(positions, classes)):
        _x = int((x - x_start) // resolution)
        _y = int((y - y_start) // resolution)
        _z = int((z - z_start) // resolution)
        if (0 <= _x < dimension) and (0 <= _y < dimension) and (0 <= _z < dimension):
            filter.append(i)
            tokens.append((_x, _y, _z, c))

    return np.array(tokens, dtype=np.int16), np.array(filter, dtype=np.int16)


def get_box_area(
    tokens: ArrayLike,
    pharmacophore_size: float,
    resolution: float,
    dimension: int,
) -> NDArray[np.bool_]:
    """Create Box Area

    Args:
        tokens: [Ntoken, 4]
        resolution: float, default = 0.5
        dimension: int, default = 64,

    Returns:
        box_areas: BoolArray [Ntoken, D, H, W] D=H=W=dimension
    """
    num_tokens = len(tokens)
    box_areas = np.zeros((num_tokens, dimension, dimension, dimension), dtype=np.bool_)
    grids = np.stack(np.meshgrid(np.arange(dimension), np.arange(dimension), np.arange(dimension), indexing='ij'), 3)
    for i, (x, y, z, t) in enumerate(tokens):
        x, y, z, t = int(x), int(y), int(z), int(t)
        distances = np.linalg.norm(grids - np.array([[x, y, z]]), axis=-1)
        threshold = math.ceil((C.INTERACTION_DIST[int(t)] + pharmacophore_size) / resolution)
        box_areas[i] = distances < threshold
    return box_areas
