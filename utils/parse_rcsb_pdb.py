import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pymol
from openbabel import pybel

PathLike = str | Path


@dataclass
class LigandInform:
    order: int
    id: str
    pdbchain: str
    authchain: str
    residx: int
    center: tuple[float, float, float]
    file_path: PathLike
    name: str | None
    synonyms: str | None

    def __str__(self) -> str:
        x, y, z = self.center
        string = (
            f"Ligand {self.order}\n"
            f"- ID      : {self.id} (Chain: {self.pdbchain} [auth {self.authchain}])\n"
            f"- Center  : {x:.3f}, {y:.3f}, {z:.3f}"
        )
        if self.name is not None:
            string += f"\n- Name    : {self.name}"
        if self.synonyms is not None:
            string += f"\n- Synonyms: {self.synonyms}"
        return string


def download_pdb(pdb_code: str, output_file: PathLike):
    url = f"https://files.rcsb.org/download/{pdb_code.lower()}.pdb"
    try:
        with urlopen(url) as response:
            content = response.read().decode("utf-8")
            with open(output_file, "w") as file:
                file.write(content)
    except Exception as e:
        print(f"Error downloading PDB file: {e}")


def parse_pdb(pdb_code: str, protein_path: PathLike, save_dir: PathLike) -> list[LigandInform]:
    protein: pybel.Molecule = next(pybel.readfile("pdb", str(protein_path)))

    if "HET" not in protein.data.keys():
        return []
    het_lines: list[str] = protein.data["HET"].split("\n")
    hetnam_lines: list[str] = protein.data["HETNAM"].split("\n")
    if "HETSYN" in protein.data.keys():
        hetsyn_lines = protein.data["HETSYN"].split("\n")
    else:
        hetsyn_lines = []

    het_id_list = tuple(line.strip().split()[0] for line in het_lines)

    ligand_name_dict = {}
    for line in hetnam_lines:
        line = line.strip()
        if line.startswith(het_id_list):
            key, *strings = line.split()
            assert key not in ligand_name_dict
            ligand_name_dict[key] = " ".join(strings)
        else:
            _, key, *strings = line.split()
            assert key in ligand_name_dict
            if ligand_name_dict[key][-1] == "-":
                ligand_name_dict[key] += " ".join(strings)
            else:
                ligand_name_dict[key] += " " + " ".join(strings)

    ligand_syn_dict = {}
    for line in hetsyn_lines:
        line: str = line.strip()
        if line.startswith(het_id_list):
            key, *strings = line.split()
            assert key not in ligand_syn_dict
            ligand_syn_dict[key] = " ".join(strings)
        else:
            _, key, *strings = line.split()
            assert key in ligand_syn_dict
            if ligand_syn_dict[key][-1] == "-":
                ligand_syn_dict[key] += " ".join(strings)
            else:
                ligand_syn_dict[key] += " " + " ".join(strings)

    pymol.finish_launching(["pymol", "-cq"])
    pymol.cmd.load(str(protein_path))

    ligand_inform_list = []
    last_chain = protein.data["SEQRES"].split("\n")[-1].split()[1]
    for idx, line in enumerate(het_lines):
        vs = line.strip().split()
        if len(vs) == 4:
            ligid, authchain, residue_idx, _ = vs
        else:
            (
                ligid,
                authchain,
                residue_idx,
            ) = (
                vs[0],
                vs[1][0],
                vs[1][1:],
            )
        pdbchain = chr(ord(last_chain) + idx + 1)
        identify_key = f"{pdb_code}_{pdbchain}_{ligid}"
        ligand_path = os.path.join(save_dir, f"{identify_key}.pdb")

        if not os.path.exists(ligand_path):
            pymol.cmd.select(
                identify_key,
                f"resn {ligid} and resi {residue_idx} and chain {authchain}",
            )
            pymol.cmd.save(ligand_path, identify_key)

        ligand = next(pybel.readfile("pdb", ligand_path))
        x, y, z = np.mean([atom.coords for atom in ligand.atoms], axis=0).tolist()

        inform = LigandInform(
            idx + 1,
            ligid,
            pdbchain,
            authchain,
            int(residue_idx),
            (x, y, z),
            ligand_path,
            ligand_name_dict.get(ligid),
            ligand_syn_dict.get(ligid),
        )
        ligand_inform_list.append(inform)
    # pymol.cmd.quit()
    return ligand_inform_list
