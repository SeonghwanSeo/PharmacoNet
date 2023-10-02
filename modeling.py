import argparse
from src.inference import InferenceModule


class Modeling_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('pharmacophore modeling script')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        # config
        cfg_args = self.add_argument_group('config')
        cfg_args.add_argument('--model_path', type=str, help='path of embedded model (.tar)', default='./weights/model.tar')
        cfg_args.add_argument('-r', '--receptor', type=str, help='path of receptor pdb file (.pdb)', required=True)
        cfg_args.add_argument('-p', '--pm_path', type=str, help='path to save pharmacophore model (.pkl)', required=True)

        cfg_args.add_argument('--autobox_ligand', type=str, help='path of ligand for binding site detection (.sdf, .pdb, .mol2)')
        cfg_args.add_argument('--center_x', type=float, help='x coordinate of the center')
        cfg_args.add_argument('--center_y', type=float, help='y coordinate of the center')
        cfg_args.add_argument('--center_z', type=float, help='z coordinate of the center')

        # system config
        env_args = self.add_argument_group('environment')
        env_args.add_argument('--device', type=str, help='device', choices=['cuda', 'cpu'], default='cpu')


def main(args):
    if args.autobox_ligand is not None:
        print(f'Using center of {args.autobox_ligand} as center of box')
    else:
        assert args.center_x is not None and args.center_y is not None and args.center_z is not None, \
            'No Center!. Enter the input --autobox_ligand or (--center_x, --center_y, --center_z)'
        print(f'Using center {(args.center_x, args.center_y, args.center_z)}')
    module = InferenceModule(args.model_path, args.device)

    if args.autobox_ligand is not None:
        pharmacophore_model = module.run(args.receptor, ref_ligand_path=args.autobox_ligand)
    else:
        pharmacophore_model = module.run(args.receptor, center=(args.center_x, args.center_y, args.center_z))
    pharmacophore_model.save(args.pm_path)


if __name__ == '__main__':
    parser = Modeling_ArgParser()
    args = parser.parse_args()
    main(args)
