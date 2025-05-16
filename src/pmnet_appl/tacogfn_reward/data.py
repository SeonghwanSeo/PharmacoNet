import torch
import torch_geometric.data as gd
from openbabel import pybel
from openbabel.pybel import ob

ATOM_DICT = {
    6: 0,
    7: 1,
    8: 2,
    9: 3,
    15: 4,
    16: 5,
    17: 6,
    35: 7,
    53: 8,
    -1: 9,  # UNKNOWN
}

BOND_DICT = {
    1: 0,
    2: 1,
    3: 2,
    1.5: 3,  # AROMATIC
    -1: 4,  # UNKNOWN
}


def smi2graph(smiles: str) -> gd.Data:
    pbmol = pybel.readstring("smi", smiles)
    obmol: ob.OBMol = pbmol.OBMol
    atom_features = []
    pos = []
    for pbatom in pbmol.atoms:
        atom_features.append(ATOM_DICT.get(pbatom.atomicnum, 9))
        pos.append(pbatom.coords)

    edge_index = []
    edge_type = []
    for obbond in ob.OBMolBondIter(obmol):
        obbond: ob.OBBond
        edge_index.append((obbond.GetBeginAtomIdx() - 1, obbond.GetEndAtomIdx() - 1))
        if obbond.IsAromatic():
            edge_type.append(3)
        else:
            edge_type.append(BOND_DICT.get(obbond.GetBondOrder(), 4))

    return gd.Data(
        x=torch.LongTensor(atom_features),
        edge_index=torch.LongTensor(edge_index).T,
        edge_attr=torch.LongTensor(edge_type),
    )
