import numpy as np
from typing import Tuple, Sequence
from openbabel.pybel import ob
from numpy.typing import NDArray

from .objects import Protein


protein_atom_num_list = (6, 7, 8, 16, -1)
protein_atom_symbol_list = ('C', 'N', 'O', 'S', 'UNK_ATOM')
protein_aminoacid_list = (
    'GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
    'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS',
    'UNK_AA',
)
protein_interactable_list = ('HydrophobicAtom', 'Ring', 'HBondDonor', 'HBondAcceptor', 'Cation', 'Anion', 'XBondAcceptor')

NUM_PROTEIN_ATOMIC_NUM = len(protein_atom_num_list)
NUM_PROTEIN_AMINOACID_NUM = len(protein_aminoacid_list)
NUM_PROTEIN_INTERACTABLE_NUM = len(protein_interactable_list)

PROTEIN_CHANNEL_LIST: Sequence[str] = protein_atom_symbol_list + protein_aminoacid_list + protein_interactable_list
NUM_PROTEIN_CHANNEL = len(PROTEIN_CHANNEL_LIST)


def get_position(obatom: ob.OBAtom) -> Tuple[float, float, float]:
    return (obatom.x(), obatom.y(), obatom.z())


def protein_atom_function(atom: ob.OBAtom, out: NDArray, **kwargs) -> NDArray[np.float32]:
    atomicnum = atom.GetAtomicNum()
    if atomicnum in protein_atom_num_list:
        out[protein_atom_num_list.index(atomicnum)] = 1
    else:
        out[NUM_PROTEIN_ATOMIC_NUM - 1] = 1
    residue_type = atom.GetResidue().GetName()
    if residue_type in protein_aminoacid_list:
        out[NUM_PROTEIN_ATOMIC_NUM + protein_aminoacid_list.index(residue_type)] = 1
    else:
        out[NUM_PROTEIN_ATOMIC_NUM + NUM_PROTEIN_AMINOACID_NUM - 1] = 1
    return out


def get_protein_pointcloud(pocket_obj: Protein) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    positions = np.array(
        [(obatom.x(), obatom.y(), obatom.z()) for obatom in pocket_obj.obatoms],
        dtype=np.float32
    )

    channels = np.zeros((pocket_obj.num_heavyatoms, NUM_PROTEIN_CHANNEL), dtype=np.float32)
    for i, atom in enumerate(pocket_obj.obatoms):
        protein_atom_function(atom, channels[i])

    offset = NUM_PROTEIN_ATOMIC_NUM + NUM_PROTEIN_AMINOACID_NUM
    for hydrop in pocket_obj.hydrophobic_atoms_all:
        channels[hydrop.index, offset] = 1
    for ring in pocket_obj.rings_all:
        channels[ring.indices, offset + 1] = 1
    for donor in pocket_obj.hbond_donors_all:
        channels[donor.index, offset + 2] = 1
    for acceptor in pocket_obj.hbond_acceptors_all:
        channels[acceptor.index, offset + 3] = 1
    for cation in pocket_obj.pos_charged_atoms_all:
        channels[cation.indices, offset + 4] = 1
    for anion in pocket_obj.neg_charged_atoms_all:
        channels[anion.indices, offset + 5] = 1
    for acceptor in pocket_obj.xbond_acceptors_all:
        channels[acceptor.indices, offset + 6] = 1
    return positions, channels
