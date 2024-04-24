import argparse
from pathlib import Path
import pmnet
from pmnet.module import PharmacoNet
import logging
import os

from utils.download_weight import download_pretrained_model
from utils.parse_rcsb_pdb import download_pdb, parse_pdb


class Modeling_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('pharmacophore modeling script')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        # config
        cfg_args = self.add_argument_group('config')
        cfg_args.add_argument('pdb', type=str, help='RCSB PDB code')
        cfg_args.add_argument('-l', '--ligand_id', type=str, help='RCSB ligand code')
        cfg_args.add_argument('-p', '--protein', type=str, help='custom path of protein pdb file (.pdb)')
        cfg_args.add_argument('-c', '--chain', type=str, help='Chain')
        cfg_args.add_argument('-a', '--all', action='store_true', help='use all binding sites')
        cfg_args.add_argument('-o', '--out', type=str, help='custom path to save pharmacophore model (.json | .pkl)')

        # system config
        env_args = self.add_argument_group('environment')
        env_args.add_argument('--exp_dir', type=str, help='custom directorh path. default: `./result/{PDBID}`')
        env_args.add_argument('--cuda', action='store_true', help='use gpu acceleration with CUDA')
        env_args.add_argument('--force', action='store_true', help='force to save the pharmacophore model')
        env_args.add_argument('-v', '--verbose', action='store_true', help='verbose')

        # config
        adv_args = self.add_argument_group('Advanced Setting')
        adv_args.add_argument('--autobox_ligand', type=str, help='path of ligand to define the center of box (.sdf, .pdb, .mol2)')
        adv_args.add_argument('--center', nargs='+', type=float, help='coordinate of the center')


def main(args):
    # NOTE: Setting
    if args.exp_dir is None:
        SAVE_DIR = Path('./result') / args.pdb
    else:
        SAVE_DIR = Path(args.exp_dir)
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    if args.out is not None and os.path.splitext(args.out)[-1] not in ['.json', '.pkl']:
        logging.error(f'Wrong extension, it should be json or pkl - {args.out}')
        return

    # NOTE: Load PharmacoNet
    logging.debug(f'Load PharmacoNet...')
    running_path = Path(pmnet.__file__)
    weight_path = running_path.parent.parent / 'weights' / 'model.tar'
    download_pretrained_model(weight_path)
    module = PharmacoNet(str(weight_path), 'cuda' if args.cuda else 'cpu')
    logging.debug(f'Load PharmacoNet finish')

    # NOTE: Set Protein
    if isinstance(args.protein, str):
        protein_path: str = args.protein
    else:  # NOTE: Download PDB
        protein_path: str = str(SAVE_DIR / f'{args.pdb}.pdb')
    if not os.path.exists(protein_path):
        logging.debug(f'Download {args.pdb} to {protein_path}')
        download_pdb(args.pdb, protein_path)
    else:
        logging.debug(f'Load {protein_path}')

    def run_pmnet(save_path, ligand_path=None, center=None):
        if (not args.force) and os.path.exists(save_path):
            logging.warning(f'Pass - {save_path} exists\n')
        else:
            pharmacophore_model = module.run(protein_path, ref_ligand_path=ligand_path, center=center)
            pharmacophore_model.save(str(save_path))
            logging.info(f'Save Pharmacophore Model to {save_path}\n')

    def run_pmnet_autobox_ligand(ligand_path):
        logging.info(f'Using center of {ligand_path} as center of box')
        save_path = SAVE_DIR / f'{args.pdb}_{Path(ligand_path).stem}_model.json' if args.out is None else args.out
        run_pmnet(save_path, ligand_path=ligand_path)

    def run_pmnet_center(center):
        x, y, z = center
        logging.info(f'Using center {(x, y, z)}')
        save_path = SAVE_DIR / f'{args.pdb}_{x}_{y}_{z}_model.json' if args.out is None else args.out
        run_pmnet(save_path, center=(x, y, z))

    def run_pmnet_inform(inform):
        logging.info(f"Running {inform.order}th Ligand...")
        save_path = SAVE_DIR / f'{args.pdb}_{inform.pdbchain}_{inform.id}_model.json' if args.out is None else args.out
        run_pmnet(save_path, center=inform.center)

    # NOTE: Case 1 With Custom Autobox Ligand Center
    if args.autobox_ligand is not None:
        assert os.path.exists(args.autobox_ligand), \
            f'Wrong Path!. The arguments for autobox ligand does not exist ({args.autobox_ligand})'
        run_pmnet_autobox_ligand(args.autobox_ligand)
        return

    # NOTE: Case 2: With Custom Center
    elif args.center is not None:
        assert len(args.center) == 3, \
            'Wrong Center!. The arguments for center coordinates should be 3. (ex. --center 1.00 2.00 -1.50)'
        run_pmnet_center(args.center)
        return

    # NOTE: Case 3: With Detected Ligand(s) Center
    else:
        inform_list = parse_pdb(args.pdb, protein_path, SAVE_DIR)
        if len(inform_list) == 0:
            logging.warning('No ligand is detected!')
            logging.info('Enter the center of binding site manually:')
            x = float(input('x: '))
            y = float(input('y: '))
            z = float(input('z: '))
            run_pmnet_center((x, y, z))
            return

        if len(inform_list) == 1:
            logging.info('A total of 1 ligand is detected!')
        else:
            logging.info(f'A total of {len(inform_list)} ligands are detected!')

        filtered_inform_list = []
        for inform in inform_list:
            if args.ligand_id is not None and args.ligand_id.upper() != inform.id:
                continue
            if args.chain is not None and args.chain.upper() not in [inform.pdbchain, inform.authchain]:
                continue
            filtered_inform_list.append(inform)
        if len(filtered_inform_list) == 0:
            inform_text_list = '\n'.join(str(inform) for inform in inform_list)
            logging.warning(f'Ligand List:\n{inform_text_list}\n')
            logging.warning(f'No matching pattern - ligand id: {args.ligand_id}, chain: {args.chain}')
            return
        inform_list = filtered_inform_list
        del filtered_inform_list

        if len(inform_list) == 1:
            run_pmnet_inform(inform_list[0])
            return

        inform_text_list = '\n\n'.join(str(inform) for inform in inform_list)
        logging.info(f'Ligand List:\n{inform_text_list}\n')
        if not args.all:
            dic = {str(inform.order): inform for inform in inform_list}
            logging.info(
                f'Select the ligand number(s) (ex. {inform_list[-1].order} ; {inform_list[0].order},{inform_list[-1].order} ; manual ; all ; exit)'
            )
            flag = False
            while not flag:
                answer = input('ligand number: ')
                if answer == 'exit':
                    return
                if answer == 'manual':
                    logging.info('Enter the center of binding site manually:')
                    x = float(input('x: '))
                    y = float(input('y: '))
                    z = float(input('z: '))
                    run_pmnet_center((x, y, z))
                    return
                if answer == 'all':
                    filtered_inform_list = inform_list
                    break
                number_list = answer.split(',')
                filtered_inform_list = []
                for number in number_list:
                    if number.strip() not in dic:
                        flag = False
                        logging.warning(f'Invalid number: {number}')
                        break
                    else:
                        flag = True
                        filtered_inform_list.append(dic[number.strip()])
                if len(filtered_inform_list) == 0:
                    flag = False
        inform_list = filtered_inform_list
        del filtered_inform_list

        if len(inform_list) > 1 and args.out is not None:
            logging.warning(f'Multiple ligand selected - Omit argument `out` ({args.out})')
            args.out = None

        for inform in inform_list:
            run_pmnet_inform(inform)


if __name__ == '__main__':
    parser = Modeling_ArgParser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    main(args)
