import numpy as np
from pmnet.module import PharmacoNet
from pmnet.utils.density_map import DensityMapGraph
import torch
import pickle


DEFAULT_SCORE_THRESHOLD = {
    "PiStacking_P": 0.6,  # Top 40%
    "PiStacking_T": 0.6,
    "SaltBridge_lneg": 0.6,
    "SaltBridge_pneg": 0.6,
    "PiCation_lring": 0.6,
    "PiCation_pring": 0.6,
    "XBond": 0.8,  # Top 20%
    "HBond_ldon": 0.8,
    "HBond_pdon": 0.8,
    "Hydrophobic": 0.8,
}


def main():
    protein_pdb_path = "./examples/6OIM_protein.pdb"
    save_path = "./examples/6OIM_MOV.pkl"
    center = (1.872, -8.260, -1.361)
    device = "cuda"
    features = feature_extraction(protein_pdb_path, center, device=device, verbose=True)
    with open(save_path, "wb") as w:
        pickle.dump(features, w)


def feature_extraction(
    protein_pdb_path,
    center,
    focus_threshold=0.5,
    box_threshold=0.5,
    score_threshold=DEFAULT_SCORE_THRESHOLD,
    device="cpu",
    verbose=True,
):
    module = PharmacoNet(
        device=device,
        focus_threshold=focus_threshold,
        box_threshold=box_threshold,
        score_threshold=score_threshold,
        verbose=verbose,
    )
    with torch.no_grad():
        center_array = np.array(center, dtype=np.float32)
        _, protein_image, non_protein_area, token_positions, tokens = module._parse_protein(
            protein_pdb_path, center_array
        )
        density_maps = module._create_density_maps_feature(
            torch.from_numpy(protein_image),
            torch.from_numpy(non_protein_area) if non_protein_area is not None else None,
            torch.from_numpy(token_positions),
            torch.from_numpy(tokens),
        )
    graph = DensityMapGraph(center, module.out_resolution, module.out_size)
    features = []
    for map in density_maps:
        node_list = graph.add_node(map["type"], map["position"], map["score"], map["map"])
        for node in node_list:
            features.append(
                {
                    "type": node.type,
                    "hotspot_position": node.hotspot_position,
                    "score": node.score,
                    "center": node.center,
                    "radius": node.radius,
                    "feature": map["feature"],
                }
            )
    return features


if __name__ == "__main__":
    main()
