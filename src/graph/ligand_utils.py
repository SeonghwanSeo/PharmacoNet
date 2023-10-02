import numpy as np
from openbabel import pybel
from openbabel.pybel import ob

from typing import List, Union, Tuple, Dict
from numpy.typing import NDArray


class PharmacophoreNode():
    def __init__(self, atom_indices: Union[int, Tuple[int, ...]], center_indices: Union[None, int, Tuple[int, ...]] = None):
        if center_indices is None:
            center_indices = atom_indices
        self.atom_indices: Union[int, Tuple[int, ...]] = atom_indices
        self.center_indices: Union[int, Tuple[int, ...]] = center_indices

    def get_center(self, atom_positions: NDArray) -> NDArray:
        if isinstance(self.center_indices, int):
            return atom_positions[self.center_indices]
        else:
            return np.mean(atom_positions[self.center_indices, :], axis=0)


def get_pharmacophore_nodes(pbmol: pybel.Molecule) -> Dict[str, List[PharmacophoreNode]]:
    obmol = pbmol.OBMol
    obatoms: List[ob.OBAtom] = list(ob.OBMolAtomIter(obmol))
    pbmol_hyd = pbmol.clone
    pbmol_hyd.OBMol.AddPolarHydrogens()
    obmol_hyd = pbmol_hyd.OBMol
    num_heavy_atoms = len(obatoms)
    obatoms_hyd: List[ob.OBAtom] = list(ob.OBMolAtomIter(obmol_hyd))[:num_heavy_atoms]

    hydrophobics = [
        PharmacophoreNode(idx) for idx, obatom in enumerate(obatoms)
        if obatom.GetAtomicNum() == 6
        and all(neigh.GetAtomicNum() in (1, 6) for neigh in ob.OBAtomAtomIter(obatom))
    ]
    hbond_acceptors = [
        PharmacophoreNode(idx) for idx, obatom in enumerate(obatoms)
        if obatom.GetAtomicNum() not in [9, 17, 35, 53]
        and obatom.IsHbondAcceptor()
    ]
    hbond_donors = [
        PharmacophoreNode(idx) for idx, obatom in enumerate(obatoms_hyd)
        if obatom.IsHbondDonor()
    ]
    rings = [
        PharmacophoreNode(tuple(sorted(idx - 1 for idx in ring._path)))     # start from 1 -> minus
        for ring in pbmol.sssr if ring.IsAromatic()
    ]
    rings.sort(key=lambda ring: ring.atom_indices)

    pos_charged = [
        PharmacophoreNode(idx) for idx, obatom in enumerate(obatoms)
        if is_quartamine_N(obatom)
        or is_tertamine_N(obatom)
        or is_sulfonium_S(obatom)
    ]
    neg_charged = []

    for idx, obatom in enumerate(obatoms):
        if is_guanidine_C(obatom):
            nitrogens = tuple(
                neigh.GetIdx() - 1 for neigh in ob.OBAtomAtomIter(obatom)
                if neigh.GetAtomicNum() == 7
            )
            pos_charged.append(PharmacophoreNode((idx,) + nitrogens, idx))

        elif is_phosphate_P(obatom) or is_sulfate_S(obatom):
            neighbors = tuple(neigh.GetIdx() - 1 for neigh in ob.OBAtomAtomIter(obatom))
            neg_charged.append(PharmacophoreNode((idx,) + neighbors, idx))

        elif is_sulfonicacid_S(obatom):
            oxygens = tuple(
                neigh.GetIdx() - 1 for neigh in ob.OBAtomAtomIter(obatom)
                if neigh.GetAtomicNum() == 8
            )
            neg_charged.append(PharmacophoreNode((idx,) + oxygens, idx))

        elif is_carboxylate_C(obatom):
            oxygens = tuple(
                neigh.GetIdx() - 1 for neigh in ob.OBAtomAtomIter(obatom)
                if neigh.GetAtomicNum() == 8
            )
            neg_charged.append(PharmacophoreNode((idx,) + oxygens, oxygens))

    xbond_donors = [
        PharmacophoreNode(idx) for idx, obatom in enumerate(obatoms)
        if is_halocarbon_X(obatom)
    ]

    return {
        'Hydrophobic': hydrophobics,
        'Aromatic': rings,
        'Cation': pos_charged,
        'Anion': neg_charged,
        'HBond_donor': hbond_donors,
        'HBond_acceptor': hbond_acceptors,
        'Halogen': xbond_donors
    }


""" FUNCTIONAL GROUP """


def is_quartamine_N(obatom: ob.OBAtom):
    # It's a nitrogen, so could be a protonated amine or quaternary ammonium
    if obatom.GetAtomicNum() != 7:  # Nitrogen
        return False
    if obatom.GetExplicitDegree() != 4:
        return False
    for neigh in ob.OBAtomAtomIter(obatom):
        if neigh.GetAtomicNum() == 1:  # It's a quat. ammonium (N with 4 residues != H)
            return False
    return True


def is_tertamine_N(obatom: ob.OBAtom):  # Nitrogen
    return (
        obatom.GetAtomicNum() == 7
        and obatom.GetHyb() == 3
        and obatom.GetHvyDegree() == 3
    )


def is_sulfonium_S(obatom: ob.OBAtom):
    if obatom.GetAtomicNum() != 16:  # Sulfur
        return False
    if obatom.GetExplicitDegree() != 3:
        return False
    for neigh in ob.OBAtomAtomIter(obatom):
        if neigh.GetAtomicNum() == 1:  # It's a sulfonium (S with 3 residues != H)
            return False
    return True


def is_guanidine_C(obatom: ob.OBAtom):
    if obatom.GetAtomicNum() != 6:  # It's a carbon atom
        return False
    numNs = 0
    numN_with_only_C = 0
    for neigh in ob.OBAtomAtomIter(obatom):
        if neigh.GetAtomicNum() == 7:
            numNs += 1
            if neigh.GetHvyDegree() == 1:
                numN_with_only_C += 1
        else:
            return False
    return numNs == 3 and numN_with_only_C > 0


def is_sulfonicacid_S(obatom: ob.OBAtom):
    if obatom.GetAtomicNum() != 16:  # Sulfur
        return False
    numOs = 0
    for neigh in ob.OBAtomAtomIter(obatom):
        if neigh.GetAtomicNum() == 8:
            numOs += 1
    return numOs == 3


def is_sulfate_S(obatom: ob.OBAtom):
    if obatom.GetAtomicNum() != 16:  # Sulfur
        return False
    numOs = 0
    for neigh in ob.OBAtomAtomIter(obatom):
        if neigh.GetAtomicNum() == 8:
            numOs += 1
    return numOs == 4


def is_phosphate_P(obatom: ob.OBAtom):
    if obatom.GetAtomicNum() != 15:  # Phosphor
        return False
    for neigh in ob.OBAtomAtomIter(obatom):
        if neigh.GetAtomicNum() != 8:  # It's a phosphate, only O
            return False
    return True


def is_carboxylate_C(obatom: ob.OBAtom):
    if obatom.GetAtomicNum() != 6:  # It's a carbon atom
        return False
    numOs = numCs = 0
    for neigh in ob.OBAtomAtomIter(obatom):
        neigh_z = neigh.GetAtomicNum()
        if neigh_z == 8:
            numOs += 1
        elif neigh_z == 6:
            numCs += 1
    return numOs == 2 and numCs == 1


def is_halocarbon_X(obatom: ob.OBAtom) -> bool:
    if obatom.GetAtomicNum() not in [9, 17, 35, 53]:  # Halogen atoms
        return False
    for neigh in ob.OBAtomAtomIter(obatom):
        if neigh.GetAtomicNum() == 6:
            return True
    return False
