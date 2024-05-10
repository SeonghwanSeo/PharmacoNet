import argparse
from pathlib import Path
import logging
import os

import pmnet
from pmnet.module import PharmacoNet
from pmnet import PharmacophoreModel
from utils.download_weight import download_pretrained_model
from utils.parse_rcsb_pdb import download_pdb, parse_pdb
from utils import visualize


SUCCESS = 0
EXIT = 1
FAIL = 2


class Modeling_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('pharmacophore modeling script')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        # config
        cfg_args = self.add_argument_group('config')
        cfg_args.add_argument('--pdb', type=str, help='RCSB PDB code')
        cfg_args.add_argument('-l', '--ligand_id', type=str, help='RCSB ligand code')
        cfg_args.add_argument('-p', '--protein', type=str, help='custom path of protein pdb file (.pdb)')
        cfg_args.add_argument('-c', '--chain', type=str, help='Chain')
        cfg_args.add_argument('-a', '--all', action='store_true', help='use all binding sites')
        cfg_args.add_argument('--out_dir', type=str, help='custom directorh path. default: `./result/{PDBID | prefix}`')
        cfg_args.add_argument('--prefix', type=str, help='task name. default: {PDBID}')
        cfg_args.add_argument('--suffix', choices=('pm', 'json'), type=str, help='extension of pharmacophore model (pm (default) | json)', default='pm')

        # system config
        env_args = self.add_argument_group('environment')
        env_args.add_argument('--cuda', action='store_true', help='use gpu acceleration with CUDA')
        env_args.add_argument('--force', action='store_true', help='force to save the pharmacophore model')
        env_args.add_argument('-v', '--verbose', action='store_true', help='verbose')

        # config
        adv_args = self.add_argument_group('Advanced Setting')
        adv_args.add_argument('--ref_ligand', type=str, help='path of ligand to define the center of box (.sdf, .pdb, .mol2)')
        adv_args.add_argument('--center', nargs='+', type=float, help='coordinate of the center')


def main(args):
    logging.info(pmnet.__description__)
    assert (args.prefix is not None or args.pdb is not None), 'MISSING PREFIX: `--prefix` or `--pdb`'
    PREFIX = args.prefix if args.prefix else args.pdb

    # NOTE: Setting
    if args.out_dir is None:
        SAVE_DIR = Path('./result') / PREFIX
    else:
        SAVE_DIR = Path(args.out_dir)
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    # NOTE: Load PharmacoNet
    running_path = Path(pmnet.__file__)
    weight_path = running_path.parent.parent / 'weights' / 'model.tar'
    download_pretrained_model(weight_path)
    module = PharmacoNet(str(weight_path), 'cuda' if args.cuda else 'cpu')
    logging.info(f'Load PharmacoNet finish')

    # NOTE: Set Protein
    if isinstance(args.pdb, str):
        protein_path: str = str(SAVE_DIR / f'{PREFIX}.pdb')
        if not os.path.exists(protein_path):
            logging.info(f'Download {args.pdb} to {protein_path}')
            download_pdb(args.pdb, protein_path)
        else:
            logging.info(f'Load {protein_path}')
    elif isinstance(args.protein, str):
        protein_path: str = args.protein
        assert os.path.exists(protein_path)
        logging.info(f'Load {protein_path}')
    else:
        raise Exception('Missing protein: `--pdb` or `--protein`')

    # NOTE: Functions
    def run_pmnet(filename, ligand_path=None, center=None) -> PharmacophoreModel:
        model_path = SAVE_DIR / f'{filename}.{args.suffix}'
        pymol_path = SAVE_DIR / f'{filename}.pse'
        if (not args.force) and os.path.exists(model_path):
            logging.warning(f'Modeling Pass - {model_path} exists')
            pharmacophore_model = PharmacophoreModel.load(str(model_path))
        else:
            pharmacophore_model = module.run(protein_path, ref_ligand_path=ligand_path, center=center)
            pharmacophore_model.save(str(model_path))
            logging.info(f'Save Pharmacophore Model to {model_path}')
        if (not args.force) and os.path.exists(pymol_path):
            logging.warning(f'Visualizing Pass - {pymol_path} exists\n')
        else:
            visualize.visualize_single(pharmacophore_model, protein_path, ligand_path, PREFIX, str(pymol_path))
            logging.info(f'Save Pymol Visualization Session to {pymol_path}\n')
        return pharmacophore_model

    def run_pmnet_ref_ligand(ligand_path) -> PharmacophoreModel:
        logging.info(f'Using center of {ligand_path} as center of box')
        return run_pmnet(f'{PREFIX}_{Path(ligand_path).stem}_model', ligand_path)

    def run_pmnet_center(center) -> PharmacophoreModel:
        x, y, z = center
        logging.info(f'Using center {(x, y, z)}')
        return run_pmnet(f'{PREFIX}_{x}_{y}_{z}_model', center=(x, y, z))

    def run_pmnet_inform(inform) -> PharmacophoreModel:
        logging.info(f"Running {inform.order}th Ligand...\n{str(inform)}")
        return run_pmnet(f'{PREFIX}_{inform.pdbchain}_{inform.id}_model', inform.file_path, inform.center)

    def run_pmnet_manual_center():
        logging.info('Enter the center of binding site manually:')
        x = float(input('x: '))
        y = float(input('y: '))
        z = float(input('z: '))
        return run_pmnet_center((x, y, z))

    ############
    # NOTE: Run!!

    # NOTE: Case 1 With Custom Autobox Ligand Center
    if args.ref_ligand is not None:
        assert os.path.exists(args.ref_ligand), \
            f'Wrong Path!. The arguments for reference ligand does not exist ({args.ref_ligand})'
        run_pmnet_ref_ligand(args.ref_ligand)
        return SUCCESS

    # NOTE: Case 2: With Custom Center
    if args.center is not None:
        assert len(args.center) == 3, \
            'Wrong Center!. The arguments for center coordinates should be 3. (ex. --center 1.00 2.00 -1.50)'
        run_pmnet_center(args.center)
        return SUCCESS

    # NOTE: Case 3: With Detected Ligand(s) Center
    # NOTE: Ligand Detection
    inform_list = parse_pdb(PREFIX, protein_path, SAVE_DIR)

    # NOTE: Case 3-1: No detected Ligand
    if len(inform_list) == 0:
        logging.warning('No ligand is detected!')
        run_pmnet_manual_center()
        return SUCCESS

    # NOTE: Case 3-2: with `all` option
    if args.all:
        logging.info(f'Use All Binding Site (-a | --all)')
        model_dict = {}
        for inform in inform_list:
            model_dict[f'{PREFIX}_{inform.pdbchain}_{inform.id}'] = (run_pmnet_inform(inform), inform.file_path)
        pymol_path = SAVE_DIR / f'{PREFIX}.pse'
        logging.info(f"Visualize all pharmacophore models...")
        if (not args.force) and os.path.exists(pymol_path):
            logging.warning(f'Visualizing Pass - {pymol_path} exists\n')
        else:
            visualize.visualize_multiple(model_dict, protein_path, PREFIX, str(pymol_path))
            logging.info(f'Save Pymol Visualization Session to {pymol_path}\n')
        return

    inform_list_text = '\n\n'.join(str(inform) for inform in inform_list)
    logging.info(f'A total of {len(inform_list)} ligand(s) are detected!\n{inform_list_text}\n')

    # NOTE: Case 3-3: pattern matching
    if args.ligand_id is not None or args.chain is not None:
        logging.info(f'Filtering with matching pattern - ligand id: {args.ligand_id}, chain: {args.chain}')
        filtered_inform_list = []
        for inform in inform_list:
            if args.ligand_id is not None and args.ligand_id.upper() != inform.id:
                continue
            if args.chain is not None and args.chain.upper() not in [inform.pdbchain, inform.authchain]:
                continue
            filtered_inform_list.append(inform)
        inform_list = filtered_inform_list
        del filtered_inform_list

        if len(inform_list) == 0:
            logging.warning(f'No matching pattern!')
            return FAIL
        if len(inform_list) > 1:
            inform_list_text = '\n\n'.join(str(inform) for inform in inform_list)
            logging.info(f'A total of {len(inform_list)} ligands are selected!\n{inform_list_text}\n')

    if len(inform_list) == 1:
        run_pmnet_inform(inform_list[0])
        return SUCCESS

    logging.info(
        f'Select the ligand number(s) (ex. {inform_list[-1].order} ; {inform_list[0].order},{inform_list[-1].order} ; manual ; all ; exit)'
    )
    inform_dic = {str(inform.order): inform for inform in inform_list}
    answer = ask_prompt(inform_dic)
    if answer == 'exit':
        return EXIT
    if answer == 'manual':
        run_pmnet_manual_center()
        return SUCCESS
    if answer == 'all':
        filtered_inform_list = inform_list
    else:
        number_list = answer.split(',')
        filtered_inform_list = []
        for number in number_list:
            filtered_inform_list.append(inform_dic[number.strip()])
    for inform in filtered_inform_list:
        run_pmnet_inform(inform)
    return SUCCESS


def ask_prompt(number_dic):
    flag = False
    while not flag:
        answer = input('ligand number: ')
        if answer in ['all', 'exit', 'manual']:
            break
        number_list = answer.split(',')
        for number in number_list:
            if number.strip() not in number_dic:
                flag = False
                logging.warning(f'Invalid number: {number}')
                break
            else:
                flag = True
    return answer


if __name__ == '__main__':
    parser = Modeling_ArgParser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    main(args)
