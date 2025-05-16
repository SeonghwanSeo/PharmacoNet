import torch
from openbabel import pybel
from openbabel.pybel import ob
from torch_geometric.data import Data as Data

ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)  # 0: None


ATOM_DICT = {
    6: 0,  # C
    7: 1,  # N
    8: 2,  # O
    9: 3,  # F
    15: 4,  # P
    16: 5,  # S
    17: 6,  # Cl
    35: 7,  # Br
    53: 8,  # I
    -1: 9,  # UNKNOWN
}
_NUM_ATOM_TYPES = 10
_NUM_ATOM_CHIRAL = 2
_NUM_ATOM_CHARGE = 2
NUM_ATOM_FEATURES = _NUM_ATOM_TYPES + _NUM_ATOM_CHIRAL + _NUM_ATOM_CHARGE

BOND_DICT = {
    1: 0,
    2: 1,
    3: 2,
    1.5: 3,  # AROMATIC
    -1: 4,  # UNKNOWN
}
_NUM_BOND_TYPES = 5
NUM_BOND_FEATURES = _NUM_BOND_TYPES


def get_atom_features(pbmol: pybel.Molecule) -> list[list[float]]:
    facade = pybel.ob.OBStereoFacade(pbmol.OBMol)
    features = []
    for atom in pbmol.atoms:
        feat = [0] * NUM_ATOM_FEATURES
        feat[ATOM_DICT.get(atom.atomicnum, 9)] = 1

        offset = _NUM_ATOM_TYPES
        mid = atom.OBAtom.GetId()
        if facade.HasTetrahedralStereo(mid):
            stereo = facade.GetTetrahedralStereo(mid).GetConfig().winding
            if stereo == pybel.ob.OBStereo.Clockwise:
                feat[offset + 0] = 1
            else:
                feat[offset + 1] = 1

        offset += _NUM_ATOM_CHIRAL
        charge = atom.formalcharge
        if charge > 0:
            feat[offset + 0] = 1
        elif charge < 0:
            feat[offset + 1] = 1
        features.append(feat)
    return features


def get_bond_features(
    pbmol: pybel.Molecule,
) -> tuple[list[list[float]], tuple[list[int], list[int]]]:
    edge_index_row = []
    edge_index_col = []
    edge_attr = []
    obmol: ob.OBMol = pbmol.OBMol
    for obbond in ob.OBMolBondIter(obmol):
        obbond: ob.OBBond
        edge_index_row.append(obbond.GetBeginAtomIdx() - 1)
        edge_index_col.append(obbond.GetEndAtomIdx() - 1)

        feat = [0] * NUM_BOND_FEATURES
        if obbond.IsAromatic():
            feat[3] = 1
        else:
            feat[BOND_DICT.get(obbond.GetBondOrder(), 4)] = 1
        edge_attr.append(feat)
    edge_index = (edge_index_row, edge_index_col)
    return edge_attr, edge_index


def smi2graph(smiles: str) -> Data:
    pbmol = pybel.readstring("smi", smiles)
    atom_features = get_atom_features(pbmol)
    edge_attr, edge_index = get_bond_features(pbmol)
    return Data(
        x=torch.FloatTensor(atom_features),
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_attr),
    )
