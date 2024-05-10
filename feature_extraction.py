import os
from pathlib import Path
import argparse
import numpy as np
from openbabel import pybel
import torch
import pickle

import pmnet
from pmnet.module import PharmacoNet
from pmnet.utils.density_map import DensityMapGraph
from utils.download_weight import download_pretrained_model


DEFAULT_SCORE_THRESHOLD = {
    'PiStacking_P': 0.6,    # Top 40%
    'PiStacking_T': 0.6,
    'SaltBridge_lneg': 0.6,
    'SaltBridge_pneg': 0.6,
    'PiCation_lring': 0.6,
    'PiCation_pring': 0.6,
    'XBond': 0.8,           # Top 20%
    'HBond_ldon': 0.8,
    'HBond_pdon': 0.8,
    'Hydrophobic': 0.8,
}


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__('pharmacophore modeling script')
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        # config
        self.add_argument('-p', '--protein', type=str, help='custom path of protein pdb file (.pdb)', required=True)
        self.add_argument('--out', type=str, help='save path of features (.pkl)', required=True)
        self.add_argument('--ref_ligand', type=str, help='path of ligand to define the center of box (.sdf, .pdb, .mol2)')
        self.add_argument('--center', nargs='+', type=float, help='coordinate of the center')
        self.add_argument('--cuda', action='store_true', help='use gpu acceleration with CUDA')
        self.add_argument('-v', '--verbose', action='store_true', help='verbose')


def feature_extraction(
    protein_pdb_path,
    center,
    focus_threshold=0.5,
    box_threshold=0.5,
    score_threshold=DEFAULT_SCORE_THRESHOLD,
    device='cpu'
):
    center_array = np.array(center, dtype=np.float32)

    weight_path = Path(pmnet.__file__).parent.parent / 'weights' / 'model.tar'
    download_pretrained_model(weight_path)
    module = PharmacoNet(
        model_path=str(weight_path),
        device=device,
        molvoxel_library='numpy',
        focus_threshold=focus_threshold,
        box_threshold=box_threshold,
        score_threshold=score_threshold,
    )

    with torch.no_grad():
        _, protein_image, non_protein_area, token_positions, tokens = module._parse_protein(protein_pdb_path, center_array)
        density_maps = module._create_density_maps_feature(
            torch.from_numpy(protein_image),
            torch.from_numpy(non_protein_area) if non_protein_area is not None else None,
            torch.from_numpy(token_positions),
            torch.from_numpy(tokens),
        )
    graph = DensityMapGraph(center, module.out_resolution, module.out_size)
    features = []
    for map in density_maps:
        node_list = graph.add_node(map['type'], map['position'], map['score'], map['map'])
        for node in node_list:
            features.append({
                'type': node.type,
                'hotspot_position': node.hotspot_position,
                'score': node.score,
                'center': node.center,
                'radius': node.radius,
                'feature': map['feature'],
            })
    return features


if __name__ == '__main__':
    parser = ArgParser()
    args = parser.parse_args()
    if args.center is not None:
        center = args.center
    elif args.ref_ligand is not None:
        extension = os.path.splitext(args.ref_ligand)[1]
        assert extension in ['.sdf', '.pdb', '.mol2']
        ref_ligand = next(pybel.readfile(extension[1:], str(args.ref_ligand)))
        center = np.mean([atom.coords for atom in ref_ligand.atoms], axis=0, dtype=np.float32)
    else:
        raise Exception('Missing center - `--ref_ligand` or `--center`')
    features = feature_extraction(args.protein, center, device='cuda' if args.cuda else 'cpu')
    with open(args.out, 'wb') as w:
        pickle.dump(features, w)
