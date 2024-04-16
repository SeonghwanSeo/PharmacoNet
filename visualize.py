import os
import argparse

import tempfile
import pymol
from pymol import cmd

import sys
sys.path.append(".")
sys.path.append("..")
from src.scoring.pharmacophore_model import PharmacophoreModel


class Visualize_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('scoring')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.add_argument('-p', '--pharmacophore_model', type=str, help='path to save pharmacophore model (.json | .pkl)', required=True)
        self.add_argument('-r', '--receptor', type=str, help='path of receptor file')
        self.add_argument('-l', '--ligand', type=str, help='path of ligand file')
        self.add_argument('-o', '--out', type=str, help='path of pymol session file (.pse)', required=True)
        self.add_argument('--prefix', type=str, help='prefix')


PHARMACOPHORE_COLOR_DICT = {
    'Hydrophobic': 'orange',
    'Aromatic': 'deeppurple',
    'Cation': 'blue',
    'Anion': 'red',
    'HBond_acceptor': 'magenta',
    'HBond_donor': 'cyan',
    'Halogen': 'yellow',
}

INTERACTION_COLOR_DICT = {
    'Hydrophobic': 'orange',
    'PiStacking_P': 'deeppurple',
    'PiStacking_T': 'deeppurple',
    'PiCation_lring': 'blue',
    'PiCation_pring': 'deeppurple',
    'HBond_ldon': 'magenta',
    'HBond_pdon': 'cyan',
    'SaltBridge_lneg': 'blue',
    'SaltBridge_pneg': 'red',
    'XBond': 'yellow',
}


if __name__ == '__main__':
    parser = Visualize_ArgParser()
    args = parser.parse_args()
    RECEPTOR_PATH = args.receptor
    LIGAND_PATH = args.ligand
    PHARMACOPHORE_MODEL_PATH = args.pharmacophore_model
    SAVE_PATH = args.out
    prefix = f'{args.prefix}_' if args.prefix else ''

    pymol.pymol_argv = ['pymol', '-pcq']
    pymol.finish_launching(args=['pymol', '-pcq', '-K'])
    cmd.reinitialize()
    cmd.feedback('disable', 'all', 'everything')

    model = PharmacophoreModel.load(PHARMACOPHORE_MODEL_PATH)

    # NOTE: Draw Molecule
    if RECEPTOR_PATH:
        cmd.load(RECEPTOR_PATH)
        cmd.set_name(os.path.splitext(os.path.basename(RECEPTOR_PATH))[0], f'{prefix}Protein')
        cmd.color('gray90', f'{prefix}Protein')
    else:
        with tempfile.TemporaryDirectory() as direc:
            RECEPTOR_PATH = f'{direc}/pocket.pdb'
            with open(RECEPTOR_PATH, 'w') as w:
                w.write(model.pocket_pdbblock)
            cmd.load(RECEPTOR_PATH)
            cmd.set_name(os.path.splitext(os.path.basename(RECEPTOR_PATH))[0], f'{prefix}Protein')
            cmd.color('gray90', f'{prefix}Protein')
    if LIGAND_PATH:
        cmd.load(LIGAND_PATH)
        cmd.set_name(os.path.splitext(os.path.basename(LIGAND_PATH))[0], f'{prefix}Ligand')
        cmd.color('green', f'{prefix}Ligand')

    # NOTE: Pharmacophore Model
    nci_dict = {}
    for node in model.nodes:
        protein_color = INTERACTION_COLOR_DICT[node.interaction_type]
        pharmacophore_color = PHARMACOPHORE_COLOR_DICT[node.type]
        protein_id = f'{prefix}hotspot{node.index}'
        cmd.pseudoatom(protein_id, pos=node.hotspot_position, color=protein_color)
        cmd.set('sphere_color', protein_color, protein_id)
        cmd.set('sphere_scale', 0.3, protein_id)

        pharmacophore_id = f'{prefix}point{node.index}'
        cmd.pseudoatom(pharmacophore_id, pos=node.center, color=protein_color)
        cmd.set('sphere_color', pharmacophore_color, pharmacophore_id)
        cmd.set('sphere_scale', node.radius, pharmacophore_id)

        interaction_id = f'{prefix}interaction{node.index}'
        cmd.distance(interaction_id, protein_id, pharmacophore_id)
        cmd.set('dash_color', pharmacophore_color, interaction_id)
        cmd.set('dash_gap', 0.2, interaction_id)
        cmd.set('dash_length', 0.4, interaction_id)

        result_id = f'{prefix}NCI{node.index}'
        cmd.group(result_id, f'{protein_id} {pharmacophore_id} {interaction_id}')
        nci_dict.setdefault(node.interaction_type, []).append(result_id)

    for interaction_type, nci_list in nci_dict.items():
        cmd.group(f'{prefix}{interaction_type}', " ".join(nci_list))
        cmd.group(f'{prefix}Model', f'{prefix}{interaction_type}')

    cmd.enable('all')
    cmd.bg_color('white')
    cmd.hide('everything', 'all')
    if LIGAND_PATH:
        cmd.show('sticks', f'{prefix}Ligand')
    cmd.show('sticks', f'{prefix}Protein')
    cmd.show('cartoon', f'{prefix}Protein')
    cmd.show('sphere', f'{prefix}Model')
    cmd.show('dash', f'{prefix}Model')
    cmd.set('sphere_transparency', 0.5, f'{prefix}point*')
    cmd.set('stick_transparency', 0.5, f'{prefix}Protein')
    cmd.set('cartoon_transparency', 0.7, f'{prefix}Protein')

    cmd.save(f'{SAVE_PATH}')
