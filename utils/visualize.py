import os
import argparse
import logging

import tempfile
import pymol
from pymol import cmd

from pmnet import PharmacophoreModel


from typing import Optional, Dict, Tuple


class Visualize_ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('scoring')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.add_argument('model', type=str, help='path to save pharmacophore model (.json | .pkl)')
        self.add_argument('-p', '--protein', type=str, help='path of protein file')
        self.add_argument('-l', '--ligand', type=str, help='path of reference ligand file')
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


def visualize_single(
    model: PharmacophoreModel,
    protein_path: Optional[str],
    ligand_path: Optional[str],
    prefix: str,
    save_path: str,
):
    pymol.pymol_argv = ['pymol', '-pcq']
    pymol.finish_launching(args=['pymol', '-pcq', '-K'])
    pymol.cmd.reinitialize()
    pymol.cmd.feedback('disable', 'all', 'everything')

    prefix = f'{prefix}_' if prefix else ''

    # NOTE: Draw Molecule
    if protein_path:
        cmd.load(protein_path)
    else:
        with tempfile.TemporaryDirectory() as direc:
            protein_path = f'{direc}/pocket.pdb'
            with open(protein_path, 'w') as w:
                w.write(model.pocket_pdbblock)
            cmd.load(protein_path)
    cmd.set_name(os.path.splitext(os.path.basename(protein_path))[0], f'{prefix}Protein')
    cmd.remove('hetatm')

    if ligand_path:
        cmd.load(ligand_path)
        cmd.set_name(os.path.splitext(os.path.basename(ligand_path))[0], f'{prefix}Ligand')

    # NOTE: Pharmacophore Model
    nci_dict = {}
    for node in model.nodes:
        hotspot_color = INTERACTION_COLOR_DICT[node.interaction_type]
        pharmacophore_color = PHARMACOPHORE_COLOR_DICT[node.type]
        hotspot_id = f'{prefix}hotspot{node.index}'
        cmd.pseudoatom(hotspot_id, pos=node.hotspot_position, color=hotspot_color)
        cmd.set('sphere_color', hotspot_color, hotspot_id)

        pharmacophore_id = f'{prefix}point{node.index}'
        cmd.pseudoatom(pharmacophore_id, pos=node.center, color=hotspot_color)
        cmd.set('sphere_color', pharmacophore_color, pharmacophore_id)
        cmd.set('sphere_scale', node.radius, pharmacophore_id)

        interaction_id = f'{prefix}interaction{node.index}'
        cmd.distance(interaction_id, hotspot_id, pharmacophore_id)
        cmd.set('dash_color', pharmacophore_color, interaction_id)

        result_id = f'{prefix}NCI{node.index}'
        cmd.group(result_id, f'{hotspot_id} {pharmacophore_id} {interaction_id}')
        nci_dict.setdefault(node.interaction_type, []).append(result_id)

    for interaction_type, nci_list in nci_dict.items():
        cmd.group(f'{prefix}{interaction_type}', " ".join(nci_list))
        cmd.group(f'{prefix}Model', f'{prefix}{interaction_type}')

    cmd.set('stick_transparency', 0.6, f'{prefix}Protein')
    cmd.set('cartoon_transparency', 0.6, f'{prefix}Protein')
    cmd.color('gray90', f'{prefix}Protein and (name C*)')

    cmd.set('sphere_scale', 0.3, '*hotspot*')
    cmd.set('sphere_transparency', 0.2, f'*point*')
    cmd.set('dash_gap', 0.2, '*interaction*')
    cmd.set('dash_length', 0.4, '*interaction*')
    cmd.hide('label', '*interaction*')

    cmd.bg_color('white')
    cmd.show('sticks', f'{prefix}Protein')
    cmd.show('sphere', f'{prefix}Model')
    cmd.show('dash', f'{prefix}Model')
    cmd.disable(f'{prefix}Protein')
    cmd.enable(f'{prefix}Protein')

    cmd.save(save_path)


def visualize_multiple(
    model_dict: Dict[str, Tuple[PharmacophoreModel, str]],
    protein_path: str,
    pdb: str,
    save_path: str,
):
    pymol.pymol_argv = ['pymol', '-pcq']
    pymol.finish_launching(args=['pymol', '-pcq', '-K'])
    cmd.reinitialize()
    cmd.feedback('disable', 'all', 'everything')

    # NOTE: Draw Molecule
    cmd.load(protein_path)
    cmd.set_name(os.path.splitext(os.path.basename(protein_path))[0], pdb)
    cmd.remove('hetatm')

    for prefix, (model, ligand_path) in model_dict.items():
        if ligand_path:
            cmd.load(ligand_path)
            cmd.set_name(os.path.splitext(os.path.basename(ligand_path))[0], f'{prefix}_Ligand')

        # NOTE: Pharmacophore Model
        nci_dict = {}
        for node in model.nodes:
            hotspot_color = INTERACTION_COLOR_DICT[node.interaction_type]
            pharmacophore_color = PHARMACOPHORE_COLOR_DICT[node.type]
            hotspot_id = f'{prefix}_hotspot{node.index}'
            cmd.pseudoatom(hotspot_id, pos=node.hotspot_position, color=hotspot_color)
            cmd.set('sphere_color', hotspot_color, hotspot_id)

            pharmacophore_id = f'{prefix}_point{node.index}'
            cmd.pseudoatom(pharmacophore_id, pos=node.center, color=hotspot_color)
            cmd.set('sphere_color', pharmacophore_color, pharmacophore_id)
            cmd.set('sphere_scale', node.radius, pharmacophore_id)

            interaction_id = f'{prefix}_interaction{node.index}'
            cmd.distance(interaction_id, hotspot_id, pharmacophore_id)
            cmd.set('dash_color', pharmacophore_color, interaction_id)

            result_id = f'{prefix}_NCI{node.index}'
            cmd.group(result_id, f'{hotspot_id} {pharmacophore_id} {interaction_id}')
            nci_dict.setdefault(node.interaction_type, []).append(result_id)

        for interaction_type, nci_list in nci_dict.items():
            cmd.group(f'{prefix}_{interaction_type}', " ".join(nci_list))
            cmd.group(f'{prefix}_Model', f'{prefix}_{interaction_type}')
        cmd.group(prefix, f'{prefix}_Model {prefix}_Ligand')

    cmd.set('stick_transparency', 0.6, pdb)
    cmd.set('cartoon_transparency', 0.6, pdb)
    cmd.color('gray90', f'{pdb} and (name C*)')

    cmd.set('sphere_scale', 0.3, '*hotspot*')
    cmd.set('sphere_transparency', 0.2, '*point*')
    cmd.set('dash_gap', 0.2, '*interaction*')
    cmd.set('dash_length', 0.4, '*interaction*')
    cmd.hide('label', '*interaction*')

    cmd.bg_color('white')
    cmd.show('sphere', '*Model')
    cmd.show('dash', '*Model')
    cmd.show('sticks', pdb)
    cmd.disable(pdb)
    cmd.enable(pdb)
    cmd.save(save_path)


if __name__ == '__main__':
    parser = Visualize_ArgParser()
    args = parser.parse_args()
    visualize_single(
        PharmacophoreModel.load(args.model),
        args.protein,
        args.ligand,
        args.prefix,
        args.out
    )
