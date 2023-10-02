import os
from openbabel import pybel
from typing import Dict, Union, Sequence

import numpy as np
from numpy.typing import NDArray


def load_ligand(ligand_path: str) -> pybel.Molecule:
    extension = os.path.splitext(ligand_path)[1]
    assert extension in ['.sdf', '.pdb', '.mol2']
    return next(pybel.readfile(extension[1:], ligand_path))


def get_score_threshold(
    score_distributions: Dict[str, Dict[str, NDArray[np.float32]]],
    relative_score_threshold: Union[float, Dict[str, float]],
    interaction_list: Sequence[str],
) -> Dict[str, float]:
    abs_score_threshold = {}
    for interaction_type in interaction_list:
        focus_scores = score_distributions[interaction_type]['focus']
        if isinstance(relative_score_threshold, float):
            rel_threshold = relative_score_threshold
        else:
            rel_threshold = relative_score_threshold[interaction_type]
        if rel_threshold == 0.0:
            abs_score_threshold[interaction_type] = float('inf')
        else:
            abs_score_threshold[interaction_type] = focus_scores[int(len(focus_scores) * (1 - rel_threshold))]
    return abs_score_threshold
