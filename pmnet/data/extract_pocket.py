import os
import numpy as np

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.PDBIO import Select

from numpy.typing import ArrayLike

import warnings
warnings.filterwarnings("ignore")

AMINO_ACID = [
    'GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
    'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS',
    'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU', 'PTR', 'GLV', 'CYT', 'SEP',
    'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'MSE', 'CSO', 'KCX',
    'CSD', 'MLY', 'PCA', 'LLP'
]


class DistSelect(Select):
    def __init__(self, center, cutoff: float = 40.0):
        self.center = np.array(center).reshape(1, 3)
        self.cutoff = cutoff

    def accept_residue(self, residue):
        if super().accept_residue(residue) == 0:
            return 0
        if residue.get_resname() not in AMINO_ACID:
            return 0
        residue_positions = np.array([
            list(atom.get_vector())
            for atom in residue.get_atoms()
            if "H" not in atom.get_id()
        ])
        if residue_positions.shape[0] == 0:
            return 0
        min_dis = np.min(np.linalg.norm(residue_positions - self.center, axis=-1))
        if min_dis < self.cutoff:
            return 1
        else:
            return 0


def extract_pocket(
    protein_pdb_path: str,
    out_pocket_pdb_path: str,
    center: ArrayLike,
    cutoff: float
):
    parser = PDBParser()
    structure = parser.get_structure("protein", protein_pdb_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pocket_pdb_path, DistSelect(center, cutoff))
    command = f"obabel {out_pocket_pdb_path} -O {out_pocket_pdb_path} -d 2>/dev/null"
    os.system(command)
