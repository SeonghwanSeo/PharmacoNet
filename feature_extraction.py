import argparse
import pickle
from pmnet.module import PharmacoNet


"""
PHARMACOPHORE_POINT_FEATURE_LIST: list[dict[str, Any]]
    PHARMACOPHORE_POINT_FEATURE
        - type: str (7 types)
            {'Hydrophobic', 'Aromatic', 'Cation', 'Anion',
             'Halogen', 'HBond_donor', 'HBond_acceptor'}
            *** `type` is obtained from `nci_type`.

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

        - hotspot_position: tuple[float, float, float] - (x, y, z)
        - priority_score: str in [0, 1]
        - center: tuple[float, float, float] - (x, y, z) 
        - radius: float
        - feature: NDArray[np.float32]
"""


DEFAULT_SCORE_THRESHOLD = {
    "PiStacking_P": 0.7,  # Top 30%
    "PiStacking_T": 0.7,
    "SaltBridge_lneg": 0.7,
    "SaltBridge_pneg": 0.7,
    "PiCation_lring": 0.7,
    "PiCation_pring": 0.7,
    "XBond": 0.85,  # Top 15%
    "HBond_ldon": 0.85,
    "HBond_pdon": 0.85,
    "Hydrophobic": 0.85,
}

RECOMMEND_SCORE_THRESHOLD = {
    "PiStacking_P": 0.5,
    "PiStacking_T": 0.5,
    "SaltBridge_lneg": 0.5,
    "SaltBridge_pneg": 0.5,
    "PiCation_lring": 0.5,
    "PiCation_pring": 0.5,
    "XBond": 0.5,
    "HBond_ldon": 0.5,
    "HBond_pdon": 0.5,
    "Hydrophobic": 0.5,
}


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__("PharmacoNet Feature Extraction Script")
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.add_argument("-p", "--protein", type=str, help="custom path of protein pdb file (.pdb)", required=True)
        self.add_argument("--out", type=str, help="save path of features (.pkl)", required=True)
        self.add_argument(
            "--ref_ligand", type=str, help="path of ligand to define the center of box (.sdf, .pdb, .mol2)"
        )
        self.add_argument("--center", nargs="+", type=float, help="coordinate of the center")
        self.add_argument("--cuda", action="store_true", help="use gpu acceleration with CUDA")


if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse_args()
    module = PharmacoNet(
        device="cuda" if args.cuda else "cpu",
        focus_threshold=0.5,
        box_threshold=0.5,
        score_threshold=RECOMMEND_SCORE_THRESHOLD,
    )
    pharmacophore_point_feature_list = module.feature_extraction(args.protein, args.ref_ligand, args.center)
    for key, item in pharmacophore_point_feature_list[0].items():
        print(key)
        print(type(item))
    with open(args.out, "wb") as w:
        pickle.dump(pharmacophore_point_feature_list, w)
