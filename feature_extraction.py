import argparse

import torch

from pmnet.api import get_pmnet_dev
from pmnet.module import PharmacoNet
from pmnet.typing import PMNetAttr


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__("PharmacoNet Feature Extraction Script")
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.add_argument(
            "-p",
            "--protein",
            type=str,
            help="custom path of protein pdb file (.pdb)",
            required=True,
        )
        self.add_argument(
            "-o",
            "--out",
            type=str,
            help="save path of features (torch object)",
            required=True,
        )
        self.add_argument(
            "--ref_ligand",
            type=str,
            help="path of ligand to define the center of box (.sdf, .pdb, .mol2)",
        )
        self.add_argument("--center", nargs="+", type=float, help="coordinate of the center")
        self.add_argument("--cuda", action="store_true", help="use gpu acceleration with CUDA")


def main(args):
    """
    return tuple[multi_scale_features, hotspot_info]
        multi_scale_features: list[torch.Tensor]:
            - [96, 4, 4, 4], [96, 8, 8, 8], [96, 16, 16, 16], [96, 32, 32, 32], [96, 64, 64, 64]
        hotspot_info
            - hotspot_feature: torch.Tensor (192,)
            - hotspot_position: tuple[float, float, float] - (x, y, z)
            - hotspot_score: float in [0, 1]

            - nci_type: str (10 types)
                'Hydrophobic': Hydrophobic interaction
                'PiStacking_P': PiStacking (Parallel)
                'PiStacking_T': PiStacking (T-shaped)
                'PiCation_lring': Interaction btw Protein Cation & Ligand Aromatic Ring
                'PiCation_pring': Interaction btw Protein Aromatic Ring & Ligand Cation
                'SaltBridge_pneg': SaltBridge btw Protein Anion & Ligand Cation
                'SaltBridge_lneg': SaltBridge btw Protein Cation & Ligand Anion
                'XBond': Halogen Bond
                'HBond_pdon': Hydrogen Bond btw Protein Donor & Ligand Acceptor
                'HBond_ldon': Hydrogen Bond btw Protein Acceptor & Ligand Donor

            - hotspot_type: str (7 types)
                {'Hydrophobic', 'Aromatic', 'Cation', 'Anion',
                'Halogen', 'HBond_donor', 'HBond_acceptor'}
                *** `type` is obtained from `nci_type`.
            - point_type: str (7 types)
                {'Hydrophobic', 'Aromatic', 'Cation', 'Anion',
                'Halogen', 'HBond_donor', 'HBond_acceptor'}
                *** `type` is obtained from `nci_type`.
    ]
    """
    device = "cuda" if args.cuda else "cpu"
    module: PharmacoNet = get_pmnet_dev(device)
    pmnet_attr: PMNetAttr = module.feature_extraction(args.protein, args.ref_ligand, args.center)
    state = pmnet_attr.to_state()
    torch.save(state, args.out)

    pmnet_attr = PMNetAttr.from_state(state)


if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse_args()
    main(args)
