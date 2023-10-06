import os
import argparse

import pymol
from pymol import cmd

import sys
sys.path.append(".")
sys.path.append("..")
from src.graph.pharmacophore_model import PharmacophoreModel


class Visualize_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('scoring')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.add_argument('-p', '--pharmacophore_model', type=str, help='path to save pharmacophore model (.pkl)', required=True)
        self.add_argument('-r', '--receptor', type=str, help='path of receptor file')
        self.add_argument('-l', '--ligand', type=str, help='path of ligand file')
        self.add_argument('-o', '--out', type=str, help='path of pymol session file (.pse)', required=True)


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

    pymol.pymol_argv = ['pymol', '-pcq']
    pymol.finish_launching(args=['pymol', '-pcq', '-K'])
    cmd.reinitialize()
    cmd.feedback('disable', 'all', 'everything')

    # NOTE: Draw Molecule
    if RECEPTOR_PATH:
        cmd.load(RECEPTOR_PATH)
        cmd.set_name(os.path.splitext(os.path.basename(RECEPTOR_PATH))[0], 'Protein')
        cmd.color('gray90', 'Protein')
    if LIGAND_PATH:
        cmd.load(LIGAND_PATH)
        cmd.set_name(os.path.splitext(os.path.basename(LIGAND_PATH))[0], 'Ligand')
        cmd.color('green', 'Ligand')

    # NOTE: Pharmacophore Model
    model = PharmacophoreModel.load(PHARMACOPHORE_MODEL_PATH)
    for node in model.nodes:
        protein_color = INTERACTION_COLOR_DICT[node.interaction_type]
        pharmacophore_color = PHARMACOPHORE_COLOR_DICT[node.type]
        protein_id = f'spot{node.index}'
        cmd.pseudoatom(protein_id, pos=node.hotspot_position, color=protein_color)
        cmd.set('sphere_color', protein_color, protein_id)
        cmd.set('sphere_scale', 0.5, protein_id)

        pharmacophore_id = f'pharmacophore{node.index}_{node.type}'
        cmd.pseudoatom(pharmacophore_id, pos=node.center, color=protein_color)
        cmd.set('sphere_color', pharmacophore_color, pharmacophore_id)
        cmd.set('sphere_scale', node.radius, pharmacophore_id)

        interaction_id = f'interaction{node.index}_{node.interaction_type}'
        cmd.distance(interaction_id, protein_id, pharmacophore_id)
        cmd.set('dash_color', pharmacophore_color, interaction_id)
        cmd.set('dash_gap', 0.2, interaction_id)
        cmd.set('dash_length', 0.4, interaction_id)

        result_id = f'result{node.index}_{node.interaction_type}'
        cmd.group(result_id, protein_id + ' ' + pharmacophore_id + ' ' + interaction_id)

    cmd.group('Result', 'result*')

    cmd.enable('all')
    cmd.hide('everything', 'all')
    if RECEPTOR_PATH:
        cmd.show('sticks', 'Protein')
    if LIGAND_PATH:
        cmd.show('sticks', 'Ligand')
    cmd.show('sphere', 'Result')
    cmd.show('dash', 'Result')

    cmd.bg_color('white')
    if RECEPTOR_PATH:
        cmd.set('stick_transparency', 0.5, 'Protein')
    if LIGAND_PATH:
        cmd.set('stick_transparency', 0.5, 'Ligand')

    cmd.set('sphere_transparency', 0.5, 'pharmacophore*')
    cmd.save(f'{SAVE_PATH}')
