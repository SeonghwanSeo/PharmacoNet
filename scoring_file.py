import argparse

from pmnet import PharmacophoreModel


class Scoring_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('scoring')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.add_argument('-p', '--pharmacophore_model', type=str, help='path of pharmacophore model (.pm | .json)', required=True)
        self.add_argument('-l', '--ligand_path', type=str, help='ligand file')
        self.add_argument('--num_conformers', type=int, help='number of RDKit conformer to use', default=10)
        self.add_argument('--num_cpus', type=int, help='number of cpu cores. default: (try to detect the number of CPUs)')


if __name__ == '__main__':
    parser = Scoring_ArgParser()
    args = parser.parse_args()
    model = PharmacophoreModel.load(args.pharmacophore_model)
    score = model.scoring_file(args.ligand_path)
    print(score)
