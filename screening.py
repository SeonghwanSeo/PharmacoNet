import argparse
from pathlib import Path
import multiprocessing
from functools import partial

from pmnet import PharmacophoreModel


class Screening_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('scoring')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        cfg_args = self.add_argument_group('config')
        cfg_args.add_argument('-p', '--pharmacophore_model', type=str, help='path of pharmacophore model (.pm | .json)', required=True)
        cfg_args.add_argument('--library', type=str, help='molecular library path', required=True)
        cfg_args.add_argument('--out', type=str, help='result file path', required=True)
        cfg_args.add_argument('--cpus', type=int, help='number of cpus', default=1)

        param_args = self.add_argument_group('parameter')
        param_args.add_argument('--hydrophobic', type=float, help='weight for hydrophobic carbon', default=1.)
        param_args.add_argument('--aromatic', type=float, help='weight for aromatic ring', default=4.)
        param_args.add_argument('--hba', type=float, help='weight for hbond acceptor', default=4.)
        param_args.add_argument('--hbd', type=float, help='weight for hbond donor', default=4.)
        param_args.add_argument('--halogen', type=float, help='weight for halogen atom', default=4.)
        param_args.add_argument('--anion', type=float, help='weight for anion', default=8.)
        param_args.add_argument('--cation', type=float, help='weight for cation', default=8.)


def func(file, model, weight):
    return file.stem, model.scoring_file(file, weight)


if __name__ == '__main__':
    parser = Screening_ArgParser()
    args = parser.parse_args()
    model = PharmacophoreModel.load(args.pharmacophore_model)
    weight = dict(
        Cation=args.cation,
        Anion=args.anion,
        Aromatic=args.aromatic,
        HBond_donor=args.hbd,
        HBond_acceptor=args.hba,
        Halogen=args.halogen,
        Hydrophobic=args.hydrophobic,
    )
    library_path = Path(args.library)
    file_list = list(library_path.rglob('*.sdf')) + list(library_path.rglob('*.mol2'))
    f = partial(func, model=model, weight=weight)

    with multiprocessing.Pool(args.cpus) as pool:
        result = pool.map(f, file_list)

    result.sort(key=lambda x: x[1], reverse=True)

    with open(args.out, 'w') as w:
        w.write(f'path,score\n')
        for filename, score in result:
            w.write(f'{filename},{score}\n')
