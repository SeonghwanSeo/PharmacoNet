import argparse
from src.modeling import ModelingModule


class Modeling_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('pharmacophore modeling script')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        # config
        cfg_args = self.add_argument_group('config')
        cfg_args.add_argument('--model_path', type=str, help='path of embedded model (.tar)', default='./weights/model.tar')
        cfg_args.add_argument('-r', '--receptor', type=str, help='path of receptor pdb file (.pdb)', required=True)
        cfg_args.add_argument('-p', '--pharmacophore_model', type=str, help='path to save pharmacophore model (.pkl)', required=True)

        cfg_args.add_argument('--autobox_ligand', type=str, help='path of ligand to define the center of box (.sdf, .pdb, .mol2)')
        cfg_args.add_argument('--center', nargs='+', type=float, help='coordinate of the center')

        # system config
        env_args = self.add_argument_group('environment')
        env_args.add_argument('--cuda', action='store_true', help='use gpu acceleration with CUDA')


def main(args):
    if args.autobox_ligand is not None:
        print(f'Using center of {args.autobox_ligand} as center of box')
    else:
        assert args.center is not None, \
            'No Center!. Enter the input `--autobox_ligand <LIGAND_PATH>` or `--center x y z`'
        assert len(args.center) == 3, \
            'Wrong Center!. The arguments for center coordinates should be 3. (ex. --center 1.00 2.00 -1.50)'
        print(f'Using center {tuple(args.center)}')
    module = ModelingModule(args.model_path, 'cuda' if args.cuda else 'cpu')

    if args.autobox_ligand is not None:
        pharmacophore_model = module.run(args.receptor, ref_ligand_path=args.autobox_ligand)
    else:
        pharmacophore_model = module.run(args.receptor, center=tuple(args.center))
    pharmacophore_model.save(args.pharmacophore_model)


if __name__ == '__main__':
    parser = Modeling_ArgParser()
    args = parser.parse_args()
    main(args)
