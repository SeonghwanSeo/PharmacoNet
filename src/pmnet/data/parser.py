import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from molvoxel import Voxelizer, create_voxelizer
from numpy.typing import NDArray
from openbabel import pybel
from torch import Tensor

from pmnet.data import pointcloud, token_inference
from pmnet.data.extract_pocket import extract_pocket
from pmnet.data.objects import Protein


class ProteinParser:
    def __init__(
        self,
        center_noise: float = 0.0,
        pocket_extract: bool = True,
        molvoxel_library: str = "numpy",
    ):
        """
        center_noise: for data augmentation
        pocket_extract: if True, we read pocket instead of entire protein. (faster)
        """
        self.voxelizer = create_voxelizer(0.5, 64, sigma=1 / 3, library=molvoxel_library)
        self.noise: float = center_noise
        self.extract: bool = pocket_extract

        ob_log_handler = pybel.ob.OBMessageHandler()
        ob_log_handler.SetOutputLevel(0)  # 0: None

    def __call__(
        self,
        protein_pdb_path: str | Path,
        ref_ligand_path: str | Path | None = None,
        center: NDArray[np.float32] | tuple[float, float, float] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.parse(protein_pdb_path, ref_ligand_path, center)

    def parse(
        self,
        protein_pdb_path: str | Path,
        ref_ligand_path: str | Path | None = None,
        center: NDArray[np.float32] | tuple[float, float, float] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        assert (ref_ligand_path is not None) or (center is not None)
        _center = self.get_center(ref_ligand_path, center)
        return parse_protein(self.voxelizer, protein_pdb_path, _center, self.noise, self.extract)

    @staticmethod
    def get_center(
        ref_ligand_path: str | Path | None = None,
        center: tuple[float, float, float] | NDArray | None = None,
    ) -> tuple[float, float, float]:
        if center is not None:
            assert len(center) == 3
            x, y, z = center
        else:
            assert ref_ligand_path is not None
            extension = os.path.splitext(ref_ligand_path)[1]
            assert extension in [".sdf", ".pdb", ".mol2"]
            ref_ligand = next(pybel.readfile(extension[1:], str(ref_ligand_path)))
            x, y, z = np.mean([atom.coords for atom in ref_ligand.atoms], axis=0, dtype=np.float32).tolist()
        return float(x), float(y), float(z)


def parse_protein(
    voxelizer: Voxelizer,
    protein_pdb_path: str | Path,
    center: NDArray[np.float32] | tuple[float, float, float],
    center_noise: float = 0.0,
    pocket_extract: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if isinstance(center, tuple):
        center = np.array(center, dtype=np.float32)
    if center_noise > 0:
        center = center + (np.random.rand(3) * 2 - 1) * center_noise

    if pocket_extract:
        with tempfile.TemporaryDirectory() as dirname:
            pocket_path = os.path.join(dirname, "pocket.pdb")
            extract_pocket(protein_pdb_path, pocket_path, center)
            protein_obj: Protein = Protein.from_pdbfile(pocket_path)
    else:
        protein_obj: Protein = Protein.from_pdbfile(protein_pdb_path)

    token_positions, token_classes = token_inference.get_token_informations(protein_obj)
    tokens, filter = token_inference.get_token_and_filter(token_positions, token_classes, center)
    token_positions = token_positions[filter]

    protein_positions, protein_features = pointcloud.get_protein_pointcloud(protein_obj)
    protein_image = np.asarray(
        voxelizer.forward_features(protein_positions, center, protein_features, radii=1.5),
        np.float32,
    )
    mask = np.logical_not(np.asarray(voxelizer.forward_single(protein_positions, center, radii=1.0), np.bool_))
    del protein_obj
    return (
        torch.from_numpy(protein_image).to(torch.float),
        torch.from_numpy(mask).to(torch.bool),
        torch.from_numpy(token_positions).to(torch.float),
        torch.from_numpy(tokens).to(torch.long),
    )
