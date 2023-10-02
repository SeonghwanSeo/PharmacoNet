from dataclasses import dataclass, field
from openbabel.pybel import ob
import numpy as np

from collections.abc import Sequence
from typing import Sequence, List, Tuple
from numpy.typing import NDArray
from functools import cached_property

from . import utils

Tuple3D = Tuple[float, float, float]


@dataclass
class Point3D(Sequence):
    x: float
    y: float
    z: float

    @classmethod
    def from_obatom(cls, obatom: ob.OBAtom):
        return cls(*utils.ob_coords(obatom))

    @classmethod
    def from_array(cls, array: NDArray):
        x, y, z = array
        return cls(x, y, z)

    def __array__(self):
        return np.array((self.x, self.y, self.z))

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, idx: int):
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        elif idx == 2:
            return self.z
        raise ValueError

    def __len__(self):
        return 3


@dataclass
class BaseInteractablePart():

    @property
    def small(self):
        if self.__small is None:
            self.__small = self.to_small()
        return self.__small

    def to_small(self):
        raise NotImplemented


@dataclass
class BaseHydrophobicAtom(BaseInteractablePart):
    obatom: ob.OBAtom
    coords: Point3D = field(init=False)

    def __post_init__(self):
        self.coords = Point3D.from_obatom(self.obatom)

    @property
    def index(self) -> int:
        return self.obatom.GetIdx() - 1


@dataclass
class BaseHBondAcceptor(BaseInteractablePart):
    obatom: ob.OBAtom
    coords: Point3D = field(init=False)

    def __post_init__(self):
        self.coords = Point3D.from_obatom(self.obatom)

    @property
    def index(self) -> int:
        return self.obatom.GetIdx() - 1


@dataclass
class BaseHBondDonor(BaseInteractablePart):
    obatom: ob.OBAtom
    coords: Point3D = field(init=False)
    hydrogens: Sequence[ob.OBAtom] = field(init=False)
    hydrogen_coords_list: Sequence[Point3D] = field(init=False)

    def __post_init__(self):
        self.coords = Point3D.from_obatom(self.obatom)
        hydrogens = [neigh for neigh in ob.OBAtomAtomIter(self.obatom)
                     if neigh.GetAtomicNum() == 1
                     ]
        self.hydrogens = hydrogens
        self.hydrogen_coords_list = [Point3D.from_obatom(h) for h in hydrogens]

    @property
    def index(self) -> int:
        return self.obatom.GetIdx() - 1


@dataclass
class BaseRing(BaseInteractablePart):
    obatoms: Sequence[ob.OBAtom]
    center: Point3D = field(init=False)
    normal: NDArray = field(init=False)

    def __post_init__(self):
        coords_list = np.array(
            [utils.ob_coords(obatom) for obatom in self.obatoms]
        )
        self.center = Point3D.from_array(np.mean(coords_list, axis=0))
        p1, p2, p3 = coords_list[0], coords_list[2], coords_list[4]
        v1, v2 = utils.vector(p1, p2), utils.vector(p1, p3)
        self.normal = utils.normalize(np.cross(v1, v2))

    @property
    def indices(self) -> List[int]:
        return [obatom.GetIdx() - 1 for obatom in self.obatoms]


@dataclass
class BaseCharged(BaseInteractablePart):
    obatoms: Sequence[ob.OBAtom]
    center: Point3D = None

    def __post_init__(self):
        if self.center is None:
            if len(self.obatoms) == 1:
                self.center = Point3D.from_obatom(self.obatoms[0])
            else:
                coords_list = np.array([utils.ob_coords(obatom) for obatom in self.obatoms])
                self.center = Point3D.from_array(np.mean(coords_list, axis=0))

    @property
    def indices(self) -> List[int]:
        return [obatom.GetIdx() - 1 for obatom in self.obatoms]


@dataclass
class BasePosCharged(BaseCharged):
    pass


@dataclass
class BaseNegCharged(BaseCharged):
    pass


@dataclass
class BaseXBondDonor(BaseInteractablePart):
    X: ob.OBAtom
    C: ob.OBAtom = field(init=False)
    X_coords: Point3D = field(init=False)
    C_coords: Point3D = field(init=False)

    def __post_init__(self):
        for neigh in ob.OBAtomAtomIter(self.X):
            if neigh.GetAtomicNum() == 6:
                self.C = neigh
                break
        assert self.C is not None
        self.X_coords = Point3D.from_obatom(self.X)
        self.C_coords = Point3D.from_obatom(self.C)

    @property
    def X_index(self) -> int:
        return self.X.GetIdx() - 1

    @property
    def C_index(self) -> int:
        return self.C.GetIdx() - 1

    @property
    def indices(self) -> List[int]:
        return [self.X_index, self.C_index]


@dataclass
class BaseXBondAcceptor(BaseInteractablePart):
    O: ob.OBAtom
    Y: ob.OBAtom
    O_coords: Point3D = field(init=False)
    Y_coords: Point3D = field(init=False)

    def __post_init__(self):
        self.O_coords = Point3D.from_obatom(self.O)
        self.Y_coords = Point3D.from_obatom(self.Y)

    @property
    def O_index(self) -> int:
        return self.O.GetIdx() - 1

    @property
    def Y_index(self) -> int:
        return self.Y.GetIdx() - 1

    @property
    def indices(self) -> List[int]:
        return [self.O_index, self.Y_index]


""" PROTEIN """


class ProteinAtom:
    @cached_property
    def obresidue_index(self) -> int:
        return self.obresidue.GetIdx() - 1

    @cached_property
    def obresidue_name(self) -> str:
        return self.obresidue.GetName()

    @cached_property
    def obresidue(self) -> ob.OBResidue:
        if hasattr(self, "obatom"):
            obatom = self.obatom
        elif hasattr(self, "obatoms"):
            obatom = self.obatoms[0]
        else:
            obatom = self.Y
        return obatom.GetResidue()


@dataclass
class HydrophobicAtom_P(ProteinAtom, BaseHydrophobicAtom):
    pass


@dataclass
class HBondAcceptor_P(ProteinAtom, BaseHBondAcceptor):
    pass


@dataclass
class HBondDonor_P(ProteinAtom, BaseHBondDonor):
    pass


@dataclass
class Ring_P(ProteinAtom, BaseRing):
    pass


@dataclass
class PosCharged_P(ProteinAtom, BasePosCharged):
    pass


@dataclass
class NegCharged_P(ProteinAtom, BaseNegCharged):
    pass


@dataclass
class XBondAcceptor_P(ProteinAtom, BaseXBondAcceptor):
    pass
