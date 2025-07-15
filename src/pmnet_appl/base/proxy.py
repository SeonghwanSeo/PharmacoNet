"""
Copyright: if you use this script, please cite:
```
@article{seo2023pharmaconet,
  title = {PharmacoNet: Accelerating Large-Scale Virtual Screening by Deep Pharmacophore Modeling},
  author = {Seo, Seonghwan and Kim, Woo Youn},
  journal = {arXiv preprint arXiv:2310.00681},
  year = {2023},
  url = {https://arxiv.org/abs/2310.00681},
}
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gdown
import torch
import torch.nn as nn
import tqdm
from numpy.typing import NDArray
from torch import Tensor

from pmnet.api import PharmacoNet, get_pmnet_dev
from pmnet.typing import PMNetAttr

Cache = Any


class BaseProxy(nn.Module):
    root_dir = Path.home() / ".local" / "share" / "pmnet" / "base_proxy"
    cache_gdrive_link: dict[tuple[str, str], str] = {}
    model_gdrive_link: dict[str, str] = {}

    def __init__(
        self,
        ckpt_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        compile_pmnet: bool = False,
    ):
        super().__init__()
        self.pmnet = None  # NOTE: Lazy
        self.ckpt_path: str | Path | None = ckpt_path
        self._cache = {}
        self._setup_model()
        self.eval()
        self.to(device)
        if self.ckpt_path is not None:
            self._load_checkpoint(self.ckpt_path)
        self.compile_pmnet: bool = compile_pmnet

    # NOTE: Implement Here!
    def _setup_model(self):
        pass

    def _load_checkpoint(self, ckpt_path: str | Path):
        self.load_state_dict(torch.load(ckpt_path, self.device))

    @torch.no_grad()
    def _scoring_list(self, cache: Cache, smiles_list: list[str]) -> Tensor:
        raise NotImplementedError

    def _get_cache(self, pmnet_attr: PMNetAttr) -> Cache:
        raise NotImplementedError

    @classmethod
    def _download_model(cls, suffix: str):
        weight_dir = cls.root_dir
        weight_dir.mkdir(parents=True, exist_ok=True)
        model_path = weight_dir / f"model-{suffix}.ckpt"
        if not model_path.exists():
            id = cls.model_gdrive_link[suffix]
            gdown.download(f"https://drive.google.com/uc?id={id}", str(model_path))

    @classmethod
    def _download_cache(cls, suffix: str, label: str):
        weight_dir = cls.root_dir
        weight_dir.mkdir(parents=True, exist_ok=True)
        cache_path = weight_dir / f"cache-{label}-{suffix}.pt"
        if not cache_path.exists():
            id = cls.cache_gdrive_link[(suffix, label)]
            gdown.download(f"https://drive.google.com/uc?id={id}", str(cache_path))

    # NOTE: Python Method
    @classmethod
    def load(
        cls,
        docking: str,
        train_dataset: str,
        db: Path | str | None,
        device: str | torch.device = "cpu",
    ):
        """Load Pretrained Proxy Model

        Parameters
        ----------
        docking : str
            docking program name
        train_dataset : str
            training dataset name
        db : Path | str | None
            cache database path ('train' | 'test' | 'all' | custom cache database path)
            - 'train': CrossDocked2020 training pockets (15,201)
            - 'test': CrossDocked2020 test pockets (100)
            - 'all': train + test
        device : str | torch.device
            cuda | spu
        """
        weight_dir = cls.root_dir
        suffix = f"{docking}-{train_dataset}"
        ckpt_path = weight_dir / f"model-{suffix}.ckpt"
        cls._download_model(suffix)

        train_cache_path = weight_dir / f"cache-train-{suffix}.pt"
        test_cache_path = weight_dir / f"cache-test-{suffix}.pt"
        if db is None:
            cache_dict = {}
        elif db == "all":
            cls._download_cache(suffix, "train")
            cls._download_cache(suffix, "test")
            cache_dict = torch.load(train_cache_path, "cpu") | torch.load(test_cache_path, "cpu")
        elif db == "train":
            cls._download_cache(suffix, "train")
            cache_dict = torch.load(train_cache_path, "cpu")
        elif db == "test":
            cls._download_cache(suffix, "test")
            cache_dict = torch.load(test_cache_path, "cpu")
        else:
            cache_dict = torch.load(db, "cpu")

        model = cls(ckpt_path, device)
        model.update_cache(cache_dict)
        return model

    def scoring(self, target: str, smiles: str) -> Tensor:
        """Scoring single molecule with its SMILES

        Parameters
        ----------
        target : str
            target key
        smiles : str
            molecule smiles

        Returns
        -------
        Tensor [1,]
            Esimated Docking Score (or Simga)

        """
        return self._scoring_list(self._cache[target], [smiles])

    def scoring_list(self, target: str, smiles_list: list[str]) -> Tensor:
        """Scoring multiple molecules with their SMILES

        Parameters
        ----------
        target : str
            target key
        smiles_list : list[str]
            molecule smiles list

        Returns
        -------
        Tensor [N,]
            Esimated Docking Scores (or Simga)

        """
        return self._scoring_list(self._cache[target], smiles_list)

    def put_cache(self, key: str, cache: Cache):
        """Add Cache

        Parameters
        ----------
        key : str
            Pocket Key
        cache : Cache
            Pocket Feature Cache
        """
        self._cache[key] = cache

    def update_cache(self, cache_dict: dict[str, Cache]):
        """Add Multiple Cache

        Parameters
        ----------
        cache_dict : dict[str, Cache]
            Pocket Key - Cache Dictionary
        """
        self._cache.update(cache_dict)

    def get_cache_database(
        self,
        pocket_info: dict[str, tuple[str | Path, str | Path | tuple[float, float, float] | NDArray]],
        save_path: str | Path | None = None,
        verbose: bool = True,
    ) -> dict[str, Cache]:
        """Get Cache Database

        Parameters
        ----------
        pocket_info : dict[str, tuple[str | Path, str | Path | tuple[float, float, float] | NDArray]]
            Key: Pocket Identification Key
            Item: (Protein Path, Pocket Center Information(ref_ligand_path or center coordinates))
                - Protein Path: str | Path
                - Pocket Center Information: str | Path | tuple[float, float, float] | NDArray]
                    - if str | Path: ref_ligand_path
                    - if tuple[float, float, float] | NDArray: center coordinates

        save_path: str | Path | None (default = None)
            if save_path is not None, save database at the input path.

        verbose: bool (default=True)
            if True, use tqdm.

        Returns
        -------
        cache_dict: dict[str, Cache]
            Cache Database
        """
        cache_dict: dict[str, Cache] = {}
        for key, (protein_pdb_path, pocket_center) in tqdm.tqdm(pocket_info.items(), disable=not (verbose)):
            try:
                if isinstance(pocket_center, str | Path):
                    cache = self.get_cache(protein_pdb_path, ref_ligand_path=pocket_center)
                else:
                    cache = self.get_cache(protein_pdb_path, center=pocket_center)
            except Exception as e:
                print(key, e)
            else:
                cache_dict[key] = cache
        if save_path is not None:
            torch.save(cache_dict, save_path)
        return cache_dict

    @torch.no_grad()
    def get_cache(
        self,
        protein_pdb_path: str | Path,
        ref_ligand_path: str | Path | None = None,
        center: tuple[float, float, float] | NDArray | None = None,
    ) -> Cache:
        """Calculate Cache

        Parameters
        ----------
        protein_pdb_path : str | Path
            Protein PDB Path
        ref_ligand_path : str | Path | None
            Reference Ligand Path (None if center is not None)
        center : tuple[float, float, float] | NDArray | None
            Pocket Center Coordinates (None if ref_ligand_path is not None)

        Returns
        -------
        cache: Cache
            Pocket Information Cache (device: 'cpu')

        """
        self.setup_pmnet()
        assert self.pmnet is not None
        pmnet_attr = self.pmnet.feature_extraction(protein_pdb_path, ref_ligand_path, center)
        cache = self._get_cache(pmnet_attr)
        cache = [v.cpu() if isinstance(v, Tensor) else v for v in cache]
        return cache

    def setup_pmnet(self):
        # NOTE: Lazy Load
        if self.pmnet is None:
            self.pmnet: PharmacoNet | None = get_pmnet_dev(self.device, compile=self.compile_pmnet)
        if self.pmnet.device != self.device:
            self.pmnet.to(self.device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
