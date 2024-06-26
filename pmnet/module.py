import os
import tempfile
import math
import logging
from pathlib import Path

import tqdm
from omegaconf import OmegaConf
from openbabel import pybel
import torch
import numpy as np
from typing import Any

from torch import Tensor
from numpy.typing import NDArray, ArrayLike

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
from pmnet.utils.density_map import DensityMapGraph

from pmnet.pharmacophore_model import PharmacophoreModel, INTERACTION_TO_PHARMACOPHORE

from importlib.util import find_spec

MOLVOXEL_LIBRARY = "numba" if find_spec("numba") is not None else "numpy"

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
    ):
        running_path = Path(__file__)
        weight_path = running_path.parent / "weights" / "model.tar"
        download_pretrained_model(weight_path)
        checkpoint = torch.load(weight_path, map_location="cpu")
        self.config = config = OmegaConf.create(checkpoint["config"])
        self.device = device
        model = build_model(config.MODEL)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.model: PharmacoFormer = model.to(device)
        self.smoothing = GaussianSmoothing(kernel_size=5, sigma=0.5).to(device)
        self.focus_threshold: float = DEFAULT_FOCUS_THRESHOLD
        self.box_threshold: float = DEFAULT_BOX_THRESHOLD
        self.score_distributions = {
            typ: np.array(distribution["focus"])
            for typ, distribution in checkpoint["score_distributions"].items()
        }

        self.score_threshold: dict[str, float]
        if isinstance(score_threshold, dict):
            self.score_threshold = score_threshold
        else:
            self.score_threshold = {typ: score_threshold for typ in INTERACTION_LIST}
        del checkpoint

        in_resolution = config.VOXEL.IN.RESOLUTION
        in_size = config.VOXEL.IN.SIZE
        self.in_voxelizer: BaseVoxelizer = create_voxelizer(
            in_resolution, in_size, sigma=(1 / 3), library=MOLVOXEL_LIBRARY
        )
        self.pocket_cutoff = (in_resolution * in_size * math.sqrt(3) / 2) + 5.0
        self.out_resolution = config.VOXEL.OUT.RESOLUTION
        self.out_size = config.VOXEL.OUT.SIZE

        if verbose:
            self.logger = logging.getLogger("PharmacoNet")
        else:
            self.logger = None

    def run(
        self,
        protein_pdb_path: str,
        ref_ligand_path: str | None = None,
        center: ArrayLike | None = None,
    ) -> PharmacophoreModel:
        assert (ref_ligand_path is not None) or (center is not None)
        if center is not None:
            center_array = np.array(center, dtype=np.float32)
        else:
            assert ref_ligand_path is not None
            extension = os.path.splitext(ref_ligand_path)[1]
            assert extension in [".sdf", ".pdb", ".mol2"]
            ref_ligand = next(pybel.readfile(extension[1:], str(ref_ligand_path)))
            center_array = np.mean(
                [atom.coords for atom in ref_ligand.atoms], axis=0, dtype=np.float32
            )
        assert center_array is not None
        assert center_array.shape == (3,)
        return self._run(protein_pdb_path, center_array)

    @torch.no_grad()
    def feature_extraction(
        self,
        protein_pdb_path: str,
        ref_ligand_path: str | None = None,
        center: ArrayLike | None = None,
    ) -> list[dict[str, Any]]:
        if center is not None:
            center_array = np.array(center, dtype=np.float32)
        else:
            assert ref_ligand_path is not None
            extension = os.path.splitext(ref_ligand_path)[1]
            assert extension in [".sdf", ".pdb", ".mol2"]
            ref_ligand = next(pybel.readfile(extension[1:], str(ref_ligand_path)))
            center_array = np.mean(
                [atom.coords for atom in ref_ligand.atoms], axis=0, dtype=np.float32
            )
        assert center_array is not None
        assert center_array.shape == (3,)
        return self._feature_extraction(protein_pdb_path, tuple(center_array.tolist()))

    @torch.no_grad()
    def _run(
        self,
        protein_pdb_path: str,
        center: NDArray[np.float32],
    ):
        protein_image, non_protein_area, token_positions, tokens = self._parse_protein(
            protein_pdb_path, center
        )
        with open(protein_pdb_path) as f:
            pdbblock: str = "\n".join(f.readlines())

        density_maps = self._create_density_maps(
            torch.from_numpy(protein_image),
            (
                torch.from_numpy(non_protein_area)
                if non_protein_area is not None
                else None
            ),
            torch.from_numpy(token_positions),
            torch.from_numpy(tokens),
        )
        return self._create_pharmacophore_model(pdbblock, center, density_maps)

    def _create_pharmacophore_model(
        self, pdbblock: str, center: NDArray[np.float32], density_maps
    ) -> PharmacophoreModel:
        x, y, z = center.tolist()
        return PharmacophoreModel.create(
            pdbblock, (x, y, z), self.out_resolution, self.out_size, density_maps
        )

    def _parse_protein(
        self,
        protein_pdb_path: str,
        center: NDArray[np.float32],
    ) -> tuple[NDArray, NDArray | None, NDArray, NDArray]:
        self.print_log("debug", "Extract Pocket...")
        with tempfile.TemporaryDirectory() as dirname:
            pocket_path = os.path.join(dirname, "pocket.pdb")
            extract_pocket(
                protein_pdb_path, pocket_path, center, self.pocket_cutoff
            )  # root(3)
            protein_obj: Protein = Protein.from_pdbfile(pocket_path)
        self.print_log("debug", "Extract Pocket Finish")

        token_positions, token_classes = token_inference.get_token_informations(
            protein_obj
        )
        tokens, filter = token_inference.get_token_and_filter(
            token_positions, token_classes, center, self.out_resolution, self.out_size
        )
        token_positions = token_positions[filter]

        protein_positions, protein_features = pointcloud.get_protein_pointcloud(
            protein_obj
        )

        self.print_log("debug", "MolVoxel:Voxelize Pocket...")
        protein_image = np.asarray(
            self.in_voxelizer.forward_features(
                protein_positions,
                center,
                protein_features,
                radii=self.config.VOXEL.RADII.PROTEIN,
            ),
            np.float32,
        )
        if self.config.VOXEL.RADII.PROTEIN_MASKING > 0:
            non_protein_area = np.logical_not(
                np.asarray(
                    self.in_voxelizer.forward_single(
                        protein_positions,
                        center,
                        radii=self.config.VOXEL.RADII.PROTEIN_MASKING,
                    ),
                    np.bool_,
                )
            )
        else:
            non_protein_area = None
        self.print_log("debug", "MolVoxel:Voxelize Pocket Finish")

        return protein_image, non_protein_area, token_positions, tokens

    def _create_density_maps(
        self,
        protein_image: Tensor,
        non_protein_area: Tensor | None,
        token_positions: Tensor,
        tokens: Tensor,
    ):
        protein_image = protein_image.to(device=self.device, dtype=torch.float)
        token_positions = token_positions.to(device=self.device, dtype=torch.float)
        tokens = tokens.to(device=self.device, dtype=torch.long)
        non_protein_area = (
            non_protein_area.to(device=self.device, dtype=torch.bool)
            if non_protein_area is not None
            else None
        )

        with torch.amp.autocast(self.device, enabled=self.config.AMP_ENABLE):
            self.print_log(
                "debug",
                f"Protein-based Pharmacophore Modeling... (device: {self.device})",
            )
            protein_image = protein_image.unsqueeze(0)
            multi_scale_features = self.model.forward_feature(
                protein_image
            )  # List[[1, D, H, W, F]]
            bottom_features = multi_scale_features[-1]

            token_scores, token_features = self.model.forward_token_prediction(
                bottom_features, [tokens]
            )
            token_scores = token_scores[0].sigmoid()  # [Ntoken,]
            token_features = token_features[0]  # [Ntoken, F]

            cavity_narrow, cavity_wide = self.model.forward_cavity_extraction(
                bottom_features
            )
            cavity_narrow = (
                cavity_narrow[0].sigmoid() > self.focus_threshold
            )  # [1, D, H, W]
            cavity_wide = (
                cavity_wide[0].sigmoid() > self.focus_threshold
            )  # [1, D, H, W]

            num_tokens = tokens.shape[0]
            indices = []
            relative_scores = []
            for i in range(num_tokens):
                x, y, z, typ = tokens[i].tolist()
                # NOTE: Check the token score
                absolute_score = token_scores[i].item()
                relative_score = float(
                    (
                        self.score_distributions[INTERACTION_LIST[int(typ)]]
                        < absolute_score
                    ).mean()
                )
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
            selected_indices = torch.tensor(
                indices, device=self.device, dtype=torch.long
            )  # [Ntoken',]
            hotspots = tokens[selected_indices]  # [Ntoken',]
            hotspot_positions = token_positions[selected_indices]  # [Ntoken', 3]
            hotspot_features = token_features[selected_indices]  # [Ntoken', F]
            del tokens
            del token_positions
            del token_features

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
                    _hotspots, _hotspot_features = (
                        hotspots[idx : idx + step],
                        hotspot_features[idx : idx + step],
                    )
                    density_maps = self.model.forward_segmentation(
                        multi_scale_features, [_hotspots], [_hotspot_features]
                    )[
                        0
                    ]  # [[4, D, H, W]]

                    density_maps = density_maps[0].sigmoid()  # [4, D, H, W]
                    density_maps_list.append(density_maps)
                    pbar.update(len(_hotspots))

            density_maps = torch.cat(density_maps_list, dim=0)  # [Ntoken', D, H, W]

            box_area = token_inference.get_box_area(
                hotspots,
                self.config.VOXEL.RADII.PHARMACOPHORE,
                self.out_resolution,
                self.out_size,
            )
            box_area = torch.from_numpy(box_area).to(
                device=self.device, dtype=torch.bool
            )  # [Ntoken', D, H, W]
            unavailable_area = ~(
                box_area & non_protein_area & cavity_narrow
            )  # [Ntoken', D, H, W]

            # NOTE: masking should be performed before smoothing - masked area is not trained.
            density_maps.masked_fill_(unavailable_area, 0.0)
            density_maps = self.smoothing(density_maps)
            density_maps.masked_fill_(unavailable_area, 0.0)
            density_maps[density_maps < self.box_threshold] = 0.0

        out = []
        assert len(hotspots) == len(relative_scores)
        for token, score, position, map in zip(
            hotspots, relative_scores, hotspot_positions, density_maps
        ):
            if torch.all(map < 1e-6):
                continue
            out.append(
                {
                    "coords": tuple(token[:3].tolist()),
                    "type": INTERACTION_LIST[int(token[3])],
                    "position": tuple(position.tolist()),
                    "score": float(score),
                    "map": map.cpu().numpy(),
                }
            )
        self.print_log(
            "debug",
            f"Protein-based Pharmacophore Modeling finish (Total {len(out)} protein hotspots are detected)",
        )
        return out

    def _feature_extraction(
        self, protein_pdb_path: str, center: tuple[float, float, float]
    ):
        center_array = np.array(center, dtype=np.float32)
        protein_image, non_protein_area, token_positions, tokens = self._parse_protein(
            protein_pdb_path, center_array
        )
        density_maps = self._create_density_maps_feature(
            torch.from_numpy(protein_image),
            (
                torch.from_numpy(non_protein_area)
                if non_protein_area is not None
                else None
            ),
            torch.from_numpy(token_positions),
            torch.from_numpy(tokens),
        )
        graph = DensityMapGraph(center, self.out_resolution, self.out_size)
        features = []
        for map in density_maps:
            node_list = graph.add_node(
                map["type"], map["position"], map["score"], map["map"]
            )
            for node in node_list:
                features.append(
                    {
                        "type": INTERACTION_TO_PHARMACOPHORE[node.type],
                        "nci_type": node.type,
                        "hotspot_position": node.hotspot_position,
                        "priority_score": node.score,
                        "center": tuple(node.center.tolist()),
                        "radius": node.radius,
                        "feature": map["feature"],
                    }
                )
        return features

    def _create_density_maps_feature(
        self,
        protein_image: Tensor,
        non_protein_area: Tensor | None,
        token_positions: Tensor,
        tokens: Tensor,
    ):
        protein_image = protein_image.to(device=self.device, dtype=torch.float)
        token_positions = token_positions.to(device=self.device, dtype=torch.float)
        tokens = tokens.to(device=self.device, dtype=torch.long)
        non_protein_area = (
            non_protein_area.to(device=self.device, dtype=torch.bool)
            if non_protein_area is not None
            else None
        )

        with torch.amp.autocast(self.device, enabled=self.config.AMP_ENABLE):
            self.print_log(
                "debug",
                f"Protein-based Pharmacophore Modeling... (device: {self.device})",
            )
            protein_image = protein_image.unsqueeze(0)
            multi_scale_features = self.model.forward_feature(
                protein_image
            )  # List[[1, D, H, W, F]]
            bottom_features = multi_scale_features[-1]

            token_scores, token_features = self.model.forward_token_prediction(
                bottom_features, [tokens]
            )
            token_scores = token_scores[0].sigmoid()  # [Ntoken,]
            token_features = token_features[0]  # [Ntoken, F]

            cavity_narrow, cavity_wide = self.model.forward_cavity_extraction(
                bottom_features
            )
            cavity_narrow = (
                cavity_narrow[0].sigmoid() > self.focus_threshold
            )  # [1, D, H, W]
            cavity_wide = (
                cavity_wide[0].sigmoid() > self.focus_threshold
            )  # [1, D, H, W]

            num_tokens = tokens.shape[0]
            indices = []
            relative_scores = []
            for i in range(num_tokens):
                x, y, z, typ = tokens[i].tolist()
                # NOTE: Check the token score
                absolute_score = token_scores[i].item()
                relative_score = float(
                    (
                        self.score_distributions[INTERACTION_LIST[int(typ)]]
                        < absolute_score
                    ).mean()
                )
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
            selected_indices = torch.tensor(
                indices, device=self.device, dtype=torch.long
            )  # [Ntoken',]

            hotspots = tokens[selected_indices]  # [Ntoken',]
            hotspot_positions = token_positions[selected_indices]  # [Ntoken', 3]
            hotspot_features = token_features[selected_indices]  # [Ntoken', F]
            del tokens
            del token_positions
            del token_features

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
                    _hotspots, _hotspot_features = (
                        hotspots[idx : idx + step],
                        hotspot_features[idx : idx + step],
                    )
                    density_maps = self.model.forward_segmentation(
                        multi_scale_features, [_hotspots], [_hotspot_features]
                    )[
                        0
                    ]  # [[4, D, H, W]]
                    density_maps = density_maps[0].sigmoid()  # [4, D, H, W]
                    density_maps_list.append(density_maps)
                    pbar.update(len(_hotspots))

            density_maps = torch.cat(density_maps_list, dim=0)  # [Ntoken', D, H, W]

            box_area = token_inference.get_box_area(
                hotspots,
                self.config.VOXEL.RADII.PHARMACOPHORE,
                self.out_resolution,
                self.out_size,
            )
            box_area = torch.from_numpy(box_area)  # [Ntoken', D, H, W]
            box_area = box_area.to(device=self.device, dtype=torch.bool)
            unavailable_area = ~(
                box_area & non_protein_area & cavity_narrow
            )  # [Ntoken', D, H, W]

            # NOTE: masking should be performed before smoothing - masked area is not trained.
            density_maps.masked_fill_(unavailable_area, 0.0)
            density_maps = self.smoothing(density_maps)
            density_maps.masked_fill_(unavailable_area, 0.0)
            density_maps[density_maps < self.box_threshold] = 0.0

        out = []
        assert len(hotspots) == len(relative_scores)
        for token, score, position, feature, map in zip(
            hotspots, relative_scores, hotspot_positions, hotspot_features, density_maps
        ):
            if torch.all(map < 1e-6):
                continue
            out.append(
                {
                    "coords": tuple(token[:3].tolist()),
                    "type": INTERACTION_LIST[int(token[3])],
                    "position": tuple(position.tolist()),
                    "score": float(score),
                    "feature": feature.cpu().numpy(),
                    "map": map.cpu().numpy(),
                }
            )
        return out

    def print_log(self, level, log):
        if self.logger is None:
            return None
        if level == "debug":
            self.logger.debug(log)
        elif level == "info":
            self.logger.info(log)
