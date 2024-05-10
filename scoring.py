import argparse
from pmnet import PharmacophoreModel


class Scoring_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('scoring')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.add_argument('-p', '--pharmacophore_model', type=str, help='path of pharmacophore model (.pm | .json)', required=True)
        self.add_argument('-s', '--smiles', type=str, help='molecule SMILES', required=True)
        self.add_argument('--num_conformers', type=int, help='number of RDKit conformer to use', default=8)


def scoring(smiles: str, model: PharmacophoreModel, num_conformers: int = 10):
    # NOTE: Scoring
    if True:
        # NOTE: Scoring with RDKit Object (Using rdmol.GetConformers())
        return model.scoring_smiles(smiles, num_conformers)
    else:
        # NOTE: Scoring with custom ligand conformers.
        return model.scoring(pbmol, atom_positions=atom_positions)


if __name__ == '__main__':
    parser = Scoring_ArgParser()
    args = parser.parse_args()
    model = PharmacophoreModel.load(args.pharmacophore_model)
    score = scoring(args.smiles, model, args.num_conformers)
    print(score)
