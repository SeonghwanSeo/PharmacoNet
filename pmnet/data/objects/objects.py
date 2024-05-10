from __future__ import annotations
from openbabel import pybel
from openbabel.pybel import ob

from typing import List, Tuple

from .atom_classes import (
    HydrophobicAtom_P,
    HBondAcceptor_P,
    HBondDonor_P,
    Ring_P,
    PosCharged_P,
    NegCharged_P,
    XBondAcceptor_P,
)


class Protein():
    def __init__(
            self,
            pbmol: pybel.Molecule,
            addh: bool = True,
    ):
        """
        pbmol: Pybel Mol
        addh: if True, call OBMol.AddPolarHydrogens()
        setup: if True, find interactable parts
        ligand: if ligand is not None, extract pocket
        """

        self.addh: bool = addh

        self.pbmol = pbmol.clone
        self.pbmol.removeh()
        self.obmol = self.pbmol.OBMol
        self.obatoms: List[ob.OBAtom] = list(ob.OBMolAtomIter(self.obmol))
        self.num_heavyatoms = len(self.obatoms)

        self.pbmol_hyd: pybel.Molecule
        if addh:
            self.pbmol_hyd = self.pbmol.clone
            self.pbmol_hyd.OBMol.AddPolarHydrogens()
        else:
            self.pbmol_hyd = pbmol
        self.obmol_hyd = self.pbmol_hyd.OBMol
        self.obatoms_hyd: List[ob.OBAtom] = list(ob.OBMolAtomIter(self.obmol_hyd))[:self.num_heavyatoms]
        self.obatoms_hyd_nonwater: List[ob.OBAtom] = [
            obatom for obatom in self.obatoms_hyd
            if obatom.GetResidue().GetName() != 'HOH'
            and obatom.GetAtomicNum() in [6, 7, 8, 16]
        ]
        self.obresidues_hyd: List[ob.OBResidue] = list(ob.OBResidueIter(self.obmol_hyd))

        self.hydrophobic_atoms_all: List[HydrophobicAtom_P]
        self.rings_all: List[Ring_P]
        self.pos_charged_atoms_all: List[PosCharged_P]
        self.neg_charged_atoms_all: List[NegCharged_P]
        self.hbond_donors_all: List[HBondDonor_P]
        self.hbond_acceptors_all: List[HBondAcceptor_P]
        self.xbond_acceptors_all: List[XBondAcceptor_P]

        self.hydrophobic_atoms_all = self.__find_hydrophobic_atoms()
        self.rings_all = self.__find_rings()
        self.pos_charged_atoms_all, self.neg_charged_atoms_all = self.__find_charged_atoms()
        self.hbond_donors_all = self.__find_hbond_donors()
        self.hbond_acceptors_all = self.__find_hbond_acceptors()
        self.xbond_acceptors_all = self.__find_xbond_acceptors()

    @classmethod
    def from_pdbfile(cls, path, addh=True, **kwargs):
        pbmol = next(pybel.readfile('pdb', path))
        return cls(pbmol, addh, **kwargs)

    # Search Interactable Part
    def __find_hydrophobic_atoms(self) -> List[HydrophobicAtom_P]:
        hydrophobics = [HydrophobicAtom_P(obatom) for obatom in self.obatoms_hyd_nonwater
                        if obatom.GetAtomicNum() == 6
                        and all(neigh.GetAtomicNum() in (1, 6) for neigh in ob.OBAtomAtomIter(obatom))
                        ]
        return hydrophobics

    def __find_hbond_acceptors(self) -> List[HBondAcceptor_P]:
        acceptors = [HBondAcceptor_P(obatom) for obatom in self.obatoms_hyd_nonwater
                     if obatom.IsHbondAcceptor()
                     ]
        return acceptors

    def __find_hbond_donors(self) -> List[HBondDonor_P]:
        donors = [HBondDonor_P(obatom) for obatom in self.obatoms_hyd_nonwater
                  if obatom.IsHbondDonor()
                  ]
        return donors

    def __find_rings(self) -> List[Ring_P]:
        rings = []
        ring_candidates = self.pbmol_hyd.sssr
        for ring in ring_candidates:
            if not 4 < len(ring._path) <= 6:
                continue
            obatoms = [self.obatoms_hyd[idx - 1] for idx in sorted(ring._path)]
            residue = obatoms[0].GetResidue()
            if residue.GetName() not in ["TYR", "TRP", "HIS", "PHE"]:
                continue
            rings.append(Ring_P(obatoms))
        return rings

    def __find_charged_atoms(self) -> Tuple[List[PosCharged_P], List[NegCharged_P]]:
        pos_charged = []
        neg_charged = []

        for obresidue in self.obresidues_hyd:
            obresname = obresidue.GetName()
            if obresname in ("ARG", "HIS", "LYS"):
                obatoms = [obatom for obatom in ob.OBResidueAtomIter(obresidue)
                           if obatom.GetAtomicNum() == 7
                           and obresidue.GetAtomProperty(obatom, ob.SIDECHAIN)
                           ]
                if len(obatoms) > 0:
                    pos_charged.append(PosCharged_P(obatoms))

            elif obresname in ("GLU", "ASP"):
                obatoms = [obatom for obatom in ob.OBResidueAtomIter(obresidue)
                           if obatom.GetAtomicNum() == 8
                           and obresidue.GetAtomProperty(obatom, ob.SIDECHAIN)
                           ]
                if len(obatoms) > 0:
                    neg_charged.append(NegCharged_P(obatoms))

        return pos_charged, neg_charged

    def __find_xbond_acceptors(self) -> List[XBondAcceptor_P]:
        """Look for halogen bond acceptors (Y-{O|N|S}, with Y=N,C)"""
        acceptors = []
        for obatom in self.obatoms_hyd_nonwater:
            if obatom.GetAtomicNum() not in [8, 7, 16]:
                continue
            neighbors = [neigh for neigh in ob.OBAtomAtomIter(obatom)
                         if neigh.GetAtomicNum() in [6, 7, 16]
                         ]
            if len(neighbors) == 1:
                O, Y = obatom, neighbors[0]
                acceptors.append(XBondAcceptor_P(O, Y))
        return acceptors
