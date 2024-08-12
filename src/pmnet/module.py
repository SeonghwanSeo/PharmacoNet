from __future__ import annotations
import os
import tempfile
import logging
from pathlib import Path
from importlib.util import find_spec

import tqdm
from openbabel import pybel
import torch
import numpy as np
from omegaconf import OmegaConf

from typing import Any
from torch import Tensor
from numpy.typing import NDArray

from molvoxel import create_voxelizer, BaseVoxelizer

from pmnet.network import build_model
from pmnet.network.detector import PharmacoFormer
from pmnet.data import token_inference, pointcloud
from pmnet.data import constant as C
from pmnet.data import INTERACTION_LIST
from pmnet.data.objects import Protein
from pmnet.data.extract_pocket import extract_pocket
from pmnet.utils.smoothing import GaussianSmoothing
from pmnet.utils.download_weight import download_pretrained_model
from pmnet.pharmacophore_model import PharmacophoreModel, INTERACTION_TO_PHARMACOPHORE, INTERACTION_TO_HOTSPOT

DEFAULT_FOCUS_THRESHOLD = 0.5
DEFAULT_BOX_THRESHOLD = 0.5
DEFAULT_SCORE_THRESHOLD = {
    "PiStacking_P": 0.7,  # Top 30%
    "PiStacking_T": 0.7,
    "SaltBridge_lneg": 0.7,
    "SaltBridge_pneg": 0.7,
    "PiCation_lring": 0.7,
    "PiCation_pring": 0.7,
    "XBond": 0.85,  # Top 15%
    "HBond_ldon": 0.85,
    "HBond_pdon": 0.85,
    "Hydrophobic": 0.85,
}


class PharmacoNet:
    def __init__(
        self,
        device: str = "cpu",
        score_threshold: float | dict[str, float] | None = DEFAULT_SCORE_THRESHOLD,
        verbose: bool = True,
        molvoxel_library: str = "numba",
    ):
        """
        device: 'cpu' | 'cuda'
        score_threshold: float | dict[str, float] | None
            custom threshold to identify hotspots.
            For feature extraction, recommended value is '0.5'
        molvoxel_library: str
            If you want to use PharmacoNet in DL model training, recommend to use 'numpy'
        """
        assert molvoxel_library in ["numpy", "numba"]
        if molvoxel_library == "numba" and (not find_spec("numba")):
            molvoxel_library = "numpy"

        running_path = Path(__file__)
        weight_path = running_path.parent / "weights" / "model.tar"
        if not weight_path.exists():
            download_pretrained_model(weight_path, verbose)
        checkpoint = torch.load(weight_path, map_location="cpu")
        config = OmegaConf.create(checkpoint["config"])
        model = build_model(config.MODEL)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.model: PharmacoFormer = model.to(device)
        self.device = device
        self.score_distributions = {
            typ: np.array(distribution["focus"]) for typ, distribution in checkpoint["score_distributions"].items()
        }
        del checkpoint

        self.focus_threshold: float = DEFAULT_FOCUS_THRESHOLD
        self.box_threshold: float = DEFAULT_BOX_THRESHOLD
        self.score_threshold: dict[str, float]
        if isinstance(score_threshold, dict):
            self.score_threshold = score_threshold
        elif isinstance(score_threshold, float):
            self.score_threshold = {typ: score_threshold for typ in INTERACTION_LIST}
        else:
            self.score_threshold = DEFAULT_SCORE_THRESHOLD

        self.resolution = 0.5
        self.size = 64
        self.voxelizer: BaseVoxelizer = create_voxelizer(
            self.resolution, self.size, sigma=(1 / 3), library=molvoxel_library
        )
        self.smoothing = GaussianSmoothing(kernel_size=5, sigma=0.5).to(device)
        if verbose:
            self.logger = logging.getLogger("PharmacoNet")
        else:
            self.logger = None

    @torch.no_grad()
    def run(
        self,
        protein_pdb_path: str | Path,
        ref_ligand_path: str | Path | None = None,
        center: tuple[float, float, float] | NDArray | None = None,
    ) -> PharmacophoreModel:
        assert (ref_ligand_path is not None) or (center is not None)
        center = self.get_center(ref_ligand_path, center)
        protein_data = parse_protein(self.voxelizer, protein_pdb_path, center, 0.0, True)
        hotspot_infos = self.create_density_maps(protein_data)
        with open(protein_pdb_path) as f:
            pdbblock: str = "\n".join(f.readlines())
        return PharmacophoreModel.create(pdbblock, center, hotspot_infos)

    @torch.no_grad()
    def feature_extraction(
        self,
        protein_pdb_path: str | Path,
        ref_ligand_path: str | Path | None = None,
        center: tuple[float, float, float] | NDArray | None = None,
    ) -> tuple[list[Tensor], list[dict[str, Any]]]:
        assert (ref_ligand_path is not None) or (center is not None)
        center = self.get_center(ref_ligand_path, center)
        protein_data = parse_protein(self.voxelizer, protein_pdb_path, center, 0.0, True)
        return self.run_extraction(protein_data)

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

    @torch.no_grad()
    def create_density_maps(self, protein_data: tuple[Tensor, Tensor, Tensor, Tensor]):
        protein_image, mask, token_positions, tokens = protein_data
        protein_image = protein_image.to(device=self.device, dtype=torch.float)
        token_positions = token_positions.to(device=self.device, dtype=torch.float)
        tokens = tokens.to(device=self.device, dtype=torch.long)
        mask = mask.to(device=self.device, dtype=torch.bool)

        self.print_log(
            "debug",
            f"Protein-based Pharmacophore Modeling... (device: {self.device})",
        )
        protein_image = protein_image.unsqueeze(0)
        multi_scale_features = self.model.forward_feature(protein_image)  # List[[1, D, H, W, F]]
        bottom_features = multi_scale_features[-1]

        token_scores, token_features = self.model.forward_token_prediction(bottom_features, [tokens])
        token_scores = token_scores[0].sigmoid()  # [Ntoken,]
        token_features = token_features[0]  # [Ntoken, F]

        cavity_narrow, cavity_wide = self.model.forward_cavity_extraction(bottom_features)
        cavity_narrow = cavity_narrow[0].sigmoid() > self.focus_threshold  # [1, D, H, W]
        cavity_wide = cavity_wide[0].sigmoid() > self.focus_threshold  # [1, D, H, W]

        num_tokens = tokens.shape[0]
        indices = []
        relative_scores = []
        for i in range(num_tokens):
            x, y, z, typ = tokens[i].tolist()
            # NOTE: Check the token score
            absolute_score = token_scores[i].item()
            relative_score = float((self.score_distributions[INTERACTION_LIST[int(typ)]] < absolute_score).mean())
            if relative_score < self.score_threshold[INTERACTION_LIST[int(typ)]]:
                continue
            # NOTE: Check the token exists in cavity
            if typ in C.LONG_INTERACTION:
                if not cavity_wide[0, x, y, z]:
                    continue
            else:
                if not cavity_narrow[0, x, y, z]:
                    continue
            indices.append(i)
            relative_scores.append(relative_score)
        selected_indices = torch.tensor(indices, device=self.device, dtype=torch.long)  # [Ntoken',]
        hotspots = tokens[selected_indices]  # [Ntoken',]
        hotspot_positions = token_positions[selected_indices]  # [Ntoken', 3]
        hotspot_features = token_features[selected_indices]  # [Ntoken', F]
        del protein_image, tokens, token_positions, token_features

        density_maps_list = []
        if self.device == "cpu":
            step = 1
        else:
            step = 4
        with tqdm.tqdm(
            desc="hotspots",
            total=hotspots.size(0),
            leave=False,
            disable=(self.logger is None),
        ) as pbar:
            for idx in range(0, hotspots.size(0), step):
                _hotspots, _hotspot_features = [hotspots[idx : idx + step]], [hotspot_features[idx : idx + step]]
                density_maps = self.model.forward_segmentation(multi_scale_features, _hotspots, _hotspot_features)[0]
                density_maps = density_maps[0].sigmoid()  # [4, D, H, W]
                density_maps_list.append(density_maps)
                pbar.update(len(_hotspots))

        density_maps = torch.cat(density_maps_list, dim=0)  # [Ntoken', D, H, W]

        box_area = token_inference.get_box_area(hotspots)
        box_area = torch.from_numpy(box_area).to(device=self.device, dtype=torch.bool)  # [Ntoken', D, H, W]
        unavailable_area = ~(box_area & mask & cavity_narrow)  # [Ntoken', D, H, W]

        # NOTE: masking should be performed before smoothing - masked area is not trained.
        density_maps.masked_fill_(unavailable_area, 0.0)
        density_maps = self.smoothing(density_maps)
        density_maps.masked_fill_(unavailable_area, 0.0)
        density_maps[density_maps < self.box_threshold] = 0.0

        hotspot_infos = []
        assert len(hotspots) == len(relative_scores)
        for hotspot, score, position, map in zip(hotspots, relative_scores, hotspot_positions, density_maps):
            if torch.all(map < 1e-6):
                continue
            interaction_type = INTERACTION_LIST[int(hotspot[3])]
            hotspot_infos.append(
                {
                    "nci_type": interaction_type,
                    "hotspot_type": INTERACTION_TO_HOTSPOT[interaction_type],
                    "hotspot_position": position,
                    "hotspot_score": score,
                    "point_type": INTERACTION_TO_PHARMACOPHORE[interaction_type],
                    "point_map": map.cpu().numpy(),
                }
            )
        self.print_log(
            "debug",
            f"Protein-based Pharmacophore Modeling finish (Total {len(hotspot_infos)} protein hotspots are detected)",
        )
        return hotspot_infos

    @torch.no_grad()
    def run_extraction(
        self, protein_data: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> tuple[list[Tensor], list[dict[str, Any]]]:
        protein_image, mask, token_positions, tokens = protein_data
        protein_image = protein_image.to(device=self.device, dtype=torch.float)
        token_positions = token_positions.to(device=self.device, dtype=torch.float)
        tokens = tokens.to(device=self.device, dtype=torch.long)
        mask = mask.to(device=self.device, dtype=torch.bool)

        protein_image = protein_image.unsqueeze(0)
        multi_scale_features = self.model.forward_feature(protein_image)  # List[[1, D, H, W, F]]
        bottom_features = multi_scale_features[-1]

        token_scores, token_features = self.model.forward_token_prediction(bottom_features, [tokens])
        token_scores = token_scores[0].sigmoid()  # [Ntoken,]
        token_features = token_features[0]  # [Ntoken, F]

        cavity_narrow, cavity_wide = self.model.forward_cavity_extraction(bottom_features)
        cavity_narrow = cavity_narrow[0].sigmoid() > self.focus_threshold  # [1, D, H, W]
        cavity_wide = cavity_wide[0].sigmoid() > self.focus_threshold  # [1, D, H, W]

        num_tokens = tokens.shape[0]
        indices = []
        relative_scores = []
        for i in range(num_tokens):
            x, y, z, typ = tokens[i].tolist()
            # NOTE: Check the token score
            absolute_score = token_scores[i].item()
            relative_score = float((self.score_distributions[INTERACTION_LIST[int(typ)]] < absolute_score).mean())
            if relative_score < self.score_threshold[INTERACTION_LIST[int(typ)]]:
                continue
            # NOTE: Check the token exists in cavity
            _cavity = cavity_wide if typ in C.LONG_INTERACTION else cavity_narrow
            if not _cavity[0, x, y, z]:
                continue
            indices.append(i)
            relative_scores.append(relative_score)
        hotspots = tokens[indices]  # [Ntoken',]
        hotspot_positions = token_positions[indices]  # [Ntoken', 3]
        hotspot_features = token_features[indices]  # [Ntoken', F]

        hotspot_infos = []
        assert len(hotspots) == len(relative_scores)
        for hotspot, score, position, feature in zip(hotspots, relative_scores, hotspot_positions, hotspot_features):
            interaction_type = INTERACTION_LIST[int(hotspot[3])]
            hotspot_infos.append(
                {
                    "nci_type": interaction_type,
                    "hotspot_type": INTERACTION_TO_HOTSPOT[interaction_type],
                    "hotspot_feature": feature,
                    "hotspot_position": tuple(position.tolist()),
                    "hotspot_score": float(score),
                    "point_type": INTERACTION_TO_PHARMACOPHORE[interaction_type],
                }
            )
        del protein_image, mask, token_positions, tokens
        return multi_scale_features, hotspot_infos

    def print_log(self, level, log):
        if self.logger is None:
            return None
        if level == "debug":
            self.logger.debug(log)
        elif level == "info":
            self.logger.info(log)


# NOTE: For DL Model Training
def parse_protein(
    voxelizer: BaseVoxelizer,
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
        voxelizer.forward_features(protein_positions, center, protein_features, radii=1.5), np.float32
    )
    mask = np.logical_not(np.asarray(voxelizer.forward_single(protein_positions, center, radii=1.0), np.bool_))
    del protein_obj
    return (
        torch.from_numpy(protein_image),
        torch.from_numpy(mask),
        torch.from_numpy(token_positions),
        torch.from_numpy(tokens),
    )
