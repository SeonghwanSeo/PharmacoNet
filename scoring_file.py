import argparse

from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing
import functools

from src.scoring import PharmacophoreModel


class Scoring_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('scoring')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.add_argument('-p', '--pharmacophore_model', type=str, help='path of pharmacophore model (.pkl)', required=True)
        self.add_argument('-s', '--smiles_path', type=str, help='molecules SMILES file', required=True)
        self.add_argument('--num_conformers', type=int, help='number of RDKit conformer to use', default=10)
        self.add_argument('--num_cpus', type=int, help='number of cpu cores. default: (try to detect the number of CPUs)')


def scoring(smiles: str, model: PharmacophoreModel, num_conformers: int = 10):
    try:
        pbmol = pybel.readstring('smi', smiles)

        # NOTE: Create Conformers
        rdmol = Chem.MolFromSmiles(smiles)
        rdmol = Chem.AddHs(rdmol)
        AllChem.EmbedMultipleConfs(rdmol, num_conformers, AllChem.srETKDGv3())
        assert rdmol.GetNumConformers() > 0

        # NOTE: Scoring
        return model.scoring(pbmol, rdmol)

    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception:
        return None


if __name__ == '__main__':
    parser = Scoring_ArgParser()
    args = parser.parse_args()
    model = PharmacophoreModel.load(args.pharmacophore_model)

    with open(args.smiles_path) as f:
        lines = f.readlines()
    key_list = [line.strip().split(',')[0] for line in lines]
    smiles_list = [line.strip().split(',')[1] for line in lines]

    num_cpus = args.num_cpus if args.num_cpus is not None else multiprocessing.cpu_count()
    partial_func = functools.partial(scoring, model=model, num_conformers=args.num_conformers)
    with multiprocessing.Pool(num_cpus) as p:
        scores = p.map(partial_func, smiles_list)
    for key, smiles, score in zip(key_list, smiles_list, scores):
        print(key, smiles, score)
