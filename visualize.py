import os
import pymol
from pymol import cmd

import sys
sys.path.append(".")
sys.path.append("..")
from src.graph.pharmacophore_model import PharmacophoreModel

PHARMACOPHORE_MODEL_PATH = sys.argv[1]
PROTEIN_PATH = sys.argv[2]
LIGAND_PATH = sys.argv[3]
SAVE_PATH = sys.argv[4]


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

pymol.pymol_argv = ['pymol', '-pcq']
pymol.finish_launching(args=['pymol', '-pcq', '-K'])
cmd.reinitialize()
cmd.feedback('disable', 'all', 'everything')

# NOTE: Draw Molecule
cmd.load(LIGAND_PATH)
cmd.load(PROTEIN_PATH)
cmd.set_name(os.path.splitext(os.path.basename(LIGAND_PATH))[0], 'Ligand')
cmd.set_name(os.path.splitext(os.path.basename(PROTEIN_PATH))[0], 'Protein')
cmd.color('green', 'Ligand')
cmd.color('gray90', 'Protein')

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
cmd.show('sticks', 'Ligand')
cmd.show('sticks', 'Protein')
cmd.show('sphere', 'Result')
cmd.show('dash', 'Result')
cmd.util.cnc('all')

cmd.bg_color('white')
cmd.set('stick_transparency', 0.5, 'Protein')
cmd.set('sphere_transparency', 0.5, 'pharmacophore*')
cmd.save(f'{SAVE_PATH}')
