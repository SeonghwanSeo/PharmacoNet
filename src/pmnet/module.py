from __future__ import annotations

import logging
import os
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray
from omegaconf import OmegaConf
from openbabel import pybel
from torch import Tensor

from pmnet.data import constant as C
from pmnet.data import token_inference
from pmnet.data.parser import ProteinParser
from pmnet.network import build_model
from pmnet.network.detector import PharmacoNetModel
from pmnet.pharmacophore_model import (
    INTERACTION_TO_HOTSPOT,
    INTERACTION_TO_PHARMACOPHORE,
    PharmacophoreModel,
)
from pmnet.typing import HotspotInfo, MultiScaleFeature, PMNetAttr
from pmnet.utils.download_weight import download_pretrained_model
from pmnet.utils.smoothing import GaussianSmoothing

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
        device: str | torch.device = "cpu",
        score_threshold: float | dict[str, float] | None = DEFAULT_SCORE_THRESHOLD,
        verbose: bool = True,
        molvoxel_library: str = "numba",
        weight_path: str | Path | None = None,
    ):
        """
        device: 'cpu' | 'cuda'
        score_threshold: float | dict[str, float] | None
            custom threshold to identify hotspots.
            For feature extraction, recommended value is '0.5'
        molvoxel_library: str
            If you want to use PharmacoNet in DL model training, recommend to use 'numpy'
        """
        # load parser
        assert molvoxel_library in ["numpy", "numba"]
        if molvoxel_library == "numba" and (not find_spec("numba")):
            molvoxel_library = "numpy"
        self.parser: ProteinParser = ProteinParser(molvoxel_library=molvoxel_library)

        # download model
        if weight_path is None:
            weight_path = Path.home() / ".local" / "share" / "pmnet" / "pmnet.tar"
            if not weight_path.exists():
                weight_path.parent.mkdir(exist_ok=True, parents=True)
                download_pretrained_model(weight_path, verbose)
        else:
            weight_path = Path(weight_path)

        # build model
        checkpoint = torch.load(weight_path, map_location="cpu")
        config = OmegaConf.create(checkpoint["config"])
        model = build_model(config.MODEL)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model: PharmacoNetModel = model.to(device)
        self.smoothing = GaussianSmoothing(kernel_size=5, sigma=0.5).to(device)
        self.score_distributions = {
            typ: np.array(distribution["focus"]) for typ, distribution in checkpoint["score_distributions"].items()
        }
        del checkpoint

        # thresholds
        self.focus_threshold: float = DEFAULT_FOCUS_THRESHOLD
        self.box_threshold: float = DEFAULT_BOX_THRESHOLD
        self.score_threshold: dict[str, float]
        if isinstance(score_threshold, dict):
            self.score_threshold = score_threshold
        elif isinstance(score_threshold, float):
            self.score_threshold = {typ: score_threshold for typ in C.INTERACTION_LIST}
        else:
            self.score_threshold = DEFAULT_SCORE_THRESHOLD

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
        protein_data = self.parser.parse(protein_pdb_path, center=center)
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
    ) -> PMNetAttr:
        protein_data = self.parser.parse(protein_pdb_path, ref_ligand_path, center)
        return self.run_extraction(protein_data)

    @torch.no_grad()
    def run_extraction(self, protein_data: tuple[Tensor, Tensor, Tensor, Tensor]) -> PMNetAttr:
        protein_image, mask, token_pos, tokens = protein_data
        protein_image = protein_image.to(device=self.device)
        token_pos = token_pos.to(device=self.device)
        tokens = tokens.to(device=self.device)
        mask = mask.to(device=self.device)

        multi_scale_features: MultiScaleFeature = self.model.forward_feature(
            protein_image.unsqueeze(0)
        )  # List[[1, D, H, W, F]]
        token_scores, token_features = self.model.forward_token_prediction(multi_scale_features[-1], [tokens])
        token_scores = token_scores[0].sigmoid()  # [Ntoken,]
        token_features = token_features[0]  # [Ntoken, F]
        cavity_narrow, cavity_wide = self.model.forward_cavity_extraction(multi_scale_features[-1])
        cavity_narrow = cavity_narrow[0].sigmoid() > self.focus_threshold  # [1, D, H, W]
        cavity_wide = cavity_wide[0].sigmoid() > self.focus_threshold  # [1, D, H, W]

        indices = []
        rel_scores = []
        for i in range(tokens.shape[0]):
            x, y, z, typ = tokens[i].tolist()
            # NOTE: Check the token score
            absolute_score = token_scores[i].item()
            relative_score = float((self.score_distributions[C.INTERACTION_LIST[int(typ)]] < absolute_score).mean())
            if relative_score < self.score_threshold[C.INTERACTION_LIST[int(typ)]]:
                continue
            # NOTE: Check the token exists in cavity
            _cavity = cavity_wide if typ in C.LONG_INTERACTION else cavity_narrow
            if not _cavity[0, x, y, z]:
                continue
            indices.append(i)
            rel_scores.append(relative_score)
        hotspots = tokens[indices]  # [Ntoken',]
        hotpsot_pos = token_pos[indices]  # [Ntoken', 3]
        hotspot_features = token_features[indices]  # [Ntoken', F]
        del protein_image, mask, token_pos, tokens

        hotspot_infos: list[HotspotInfo] = []
        for hotspot, score, position, features in zip(hotspots, rel_scores, hotpsot_pos, hotspot_features, strict=True):
            interaction_type = C.INTERACTION_LIST[int(hotspot[3])]
            hotspot_infos.append(
                HotspotInfo(
                    type=INTERACTION_TO_HOTSPOT[interaction_type],
                    score=float(score),
                    features=features,
                    position=tuple(position.tolist()),
                    nci_type=interaction_type,
                    density_type=INTERACTION_TO_PHARMACOPHORE[interaction_type],
                )
            )
        return PMNetAttr(multi_scale_features, hotspot_infos)

    def print_log(self, level, log):
        if self.logger is None:
            return None
        if level == "debug":
            self.logger.debug(log)
        elif level == "info":
            self.logger.info(log)

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
            center = np.mean([atom.coords for atom in ref_ligand.atoms], axis=0, dtype=np.float32)
            x, y, z = center.tolist()
        return float(x), float(y), float(z)

    @torch.no_grad()
    def create_density_maps(self, protein_data: tuple[Tensor, Tensor, Tensor, Tensor]):
        protein_image, mask, token_pos, tokens = protein_data
        protein_image = protein_image.to(device=self.device, dtype=torch.float)
        token_pos = token_pos.to(device=self.device, dtype=torch.float)
        tokens = tokens.to(device=self.device, dtype=torch.long)
        mask = mask.to(device=self.device, dtype=torch.bool)

        self.print_log(
            "debug",
            f"Protein-based Pharmacophore Modeling... (device: {self.device})",
        )
        multi_scale_features = self.model.forward_feature(protein_image.unsqueeze(0))  # List[[1, D, H, W, F]]
        token_scores, token_features = self.model.forward_token_prediction(multi_scale_features[-1], [tokens])
        token_scores = token_scores[0].sigmoid()  # [Ntoken,]
        token_features = token_features[0]  # [Ntoken, F]
        cavity_narrow, cavity_wide = self.model.forward_cavity_extraction(multi_scale_features[-1])
        cavity_narrow = cavity_narrow[0].sigmoid() > self.focus_threshold  # [1, D, H, W]
        cavity_wide = cavity_wide[0].sigmoid() > self.focus_threshold  # [1, D, H, W]

        num_tokens = tokens.shape[0]
        indices = []
        rel_scores = []
        for i in range(num_tokens):
            x, y, z, typ = tokens[i].tolist()
            # NOTE: Check the token score
            absolute_score = token_scores[i].item()
            relative_score = float((self.score_distributions[C.INTERACTION_LIST[int(typ)]] < absolute_score).mean())
            if relative_score < self.score_threshold[C.INTERACTION_LIST[int(typ)]]:
                continue
            # NOTE: Check the token exists in cavity
            if typ in C.LONG_INTERACTION:
                if not cavity_wide[0, x, y, z]:
                    continue
            else:
                if not cavity_narrow[0, x, y, z]:
                    continue
            indices.append(i)
            rel_scores.append(relative_score)
        selected_indices = torch.tensor(indices, device=self.device, dtype=torch.long)  # [Ntoken',]
        hotspots = tokens[selected_indices]  # [Ntoken',]
        hotpsot_pos = token_pos[selected_indices]  # [Ntoken', 3]
        hotspot_features = token_features[selected_indices]  # [Ntoken', F]
        del protein_image, tokens, token_pos, token_features

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
        for hotspot, score, position, features, map in zip(
            hotspots, rel_scores, hotpsot_pos, hotspot_features, density_maps, strict=True
        ):
            if torch.all(map < 1e-6):
                continue
            interaction_type = C.INTERACTION_LIST[int(hotspot[3])]
            hotspot_infos.append(
                HotspotInfo(
                    type=INTERACTION_TO_HOTSPOT[interaction_type],
                    features=features.cpu(),
                    position=tuple(position.tolist()),
                    score=score,
                    nci_type=interaction_type,
                    density_type=INTERACTION_TO_PHARMACOPHORE[interaction_type],
                    density_map=map.cpu(),
                )
            )
        self.print_log(
            "debug",
            f"Protein-based Pharmacophore Modeling finish (Total {len(hotspot_infos)} protein hotspots are detected)",
        )
        return hotspot_infos

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, device):
        self.model = self.model.to(device)

    def cuda(self):
        self.model = self.model.cuda()

    def cpu(self):
        self.model = self.model.cpu()
