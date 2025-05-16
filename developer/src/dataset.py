import pickle
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from pmnet.api import ProteinParser

from .data import smi2graphdata


class BaseDataset(Dataset):
    def __init__(
        self,
        code_list: list[str],
        protein_info: dict[str, tuple[float, float, float]],
        protein_dir: Path | str,
        ligand_path: Path | str,
        center_noise: float = 0.0,
    ):
        self.parser: ProteinParser = ProteinParser(center_noise)

        self.code_list: list[str] = code_list
        self.protein_info = protein_info
        self.protein_dir = Path(protein_dir)
        self.center_noise = center_noise
        with open(ligand_path, "rb") as f:
            self.ligand_data: dict[str, list[tuple[str, str, float]]] = pickle.load(f)

    def __len__(self):
        return len(self.code_list)

    def __getitem__(self, index: int) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], Batch]:
        code = self.code_list[index]
        protein_path: str = str(self.protein_dir / f"{code}.pdb")
        center: tuple[float, float, float] = self.protein_info[code]
        pharmacophore_info = self.parser(protein_path, center=center)
        ligands = self.ligand_data[code]
        ligand_graphs: Batch = Batch.from_data_list(list(map(self.get_ligand_data, ligands)))
        return pharmacophore_info, ligand_graphs

    @staticmethod
    def get_ligand_data(args: tuple[str, str, float]) -> Data:
        ligand_id, smiles, affinity = args
        data = smi2graphdata(smiles)
        x, edge_index, edge_attr = data["x"], data["edge_index"], data["edge_attr"]
        affinity = min(float(affinity), 0.0)
        return Data(
            x,
            edge_index,
            edge_attr,
            affinity=torch.FloatTensor([affinity]),
        )
