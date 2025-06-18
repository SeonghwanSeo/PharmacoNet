# PharmacoNet: Open-source Protein-based Pharmacophore Modeling

[![DOI](https://zenodo.org/badge/699273873.svg)](https://zenodo.org/doi/10.5281/zenodo.12168474)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**Before using PharmacoNet, consider using [OpenPharmaco](https://github.com/SeonghwanSeo/OpenPharmaco): GUI powered by PharmacoNet.**

**Chemical Science (Open Access)** [[paper](https://doi.org/10.1039/D4SC04854G)]

Official Github for **_PharmacoNet: deep learning-guided pharmacophore modeling for ultra-large-scale virtual screening_** by Seonghwan Seo\* and Woo Youn Kim.

PharmacoNet is an extremely rapid yet reasonably accurate ligand evaluation tool with high generation ability:

1. Fully automated protein-based pharmacophore modeling based on image instance segmentation modeling
2. Coarse-grained graph matching at the pharmacophore level for high throughput virtual screening
3. Pharmacophore-aware scoring function with parameterized analytical function for robust generalization ability
4. Better pocket representation for deep learning developer ([section](#pharmacophore-feature-extraction))

If you have any problems or need help with the code, please add an github issue or contact [shwan0106@kaist.ac.kr](mailto:shwan0106@kaist.ac.kr).

\* You can read the previous NeurIPS 2023 Workshop version at [arXiv](https://arxiv.org/abs/2310.00681).

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Pharmacophore Modeling](#pharmacophore-modeling)
- [Virtual Screening](#virtual-screening)
- [Pharmacophore Feature Extraction](#pharmacophore-feature-extraction)
- [Pre-trained Docking Proxy](#pretrained-docking-proxy)
- [Citation](#citation)

## Quick Start

```bash
# Pharmacophore Modeling
python modeling.py --pdb <PDB ID>   # RCSB PDB importing
python modeling.py --protein <PROTEIN_PATH> --prefix <EXP_NAME> --cuda  # CUDA acceleration
python modeling.py --protein <PROTEIN_PATH> --prefix <EXP_NAME> --ref_ligand <REF_LIGAND_PATH>

# Virtual Screening
python screening.py -p <MODEL_PATH> --library <LIBRARY_DIR> --out <RESULT_PATH> --cpus <NCPU>

# Feature Extraction for Deep Learning Researcher
python feature_extraction.py --protein <PROTEIN_PATH> --ref_ligand <REF_LIGAND_PATH> --out <SAVE_PKL_PATH>
python feature_extraction.py --protein <PROTEIN_PATH> --center <X> <Y> <Z> --out <SAVE_PKL_PATH> --cuda
```

## Installation

- Using `environment.yml`
  For various environment including Linux, MacOS and Window, the script installs **cpu-only version of PyTorch** by default. You can install a cuda-available version by modifying `environment.yml` or installing PyTorch manually.

  ```bash
  conda create -f environment.yml
  conda activate pmnet
  pip install .
  ```

- Manual Installation

  ```bash
  # Required python>=3.9, Best Performance at higher version. (3.9, 3.10, 3.11, 3.12(best))
  conda create --name pmnet python=3.12 pymol-open-source
  conda activate pmnet

  pip install torch # 1.13<=torch, CUDA acceleration is available. 1min for 1 cpu, 10s for 1 gpu
  pip install rdkit biopython omegaconf tdqm numba # Numba is optional, but recommended.
  pip install molvoxel # Molecular voxelization tools with minimal dependencies (https://github.com/SeonghwanSeo/molvoxel.git)
  ```

- Installation for feature extractions (For Deep learning developer)

  ```bash
  # in your project
  pip install pharmaconet @ git+https://github.com/SeonghwanSeo/PharmacoNet.git
  ```

## Pharmacophore Modeling

You can run `model.py` for automated protein-based pharmacophore modeling with RCSB PDB code or custom protein path (`--protein`). With protein path, you should enter `--prefix`.

#### Example with RCSB PDB Code

The pharmacophore model file is `result/6oim/6oim_D_MOV_model.pm` and the pymol session file is `result/6oim/6oim_D_MOV_model.pse`

```bash
# Pharmacophore Modeling for KRAS(G12C) - PDBID: 6OIM
> python modeling.py --pdb 6oim
INFO:root:Load PharmacoNet finish
INFO:root:Download 6oim to result/6oim/6oim.pdb
==============================

INFO:root:A total of 3 ligand(s) are detected!
Ligand 1
- ID      : MG (Chain: B [auth A])
- Center  : -2.512, 2.588, 0.220
- Name    : MAGNESIUM ION

Ligand 2
- ID      : GDP (Chain: C [auth A])
- Center  : -6.125, 3.588, 7.310
- Name    : GUANOSINE-5-DIPHOSPHATE

Ligand 3
- ID      : MOV (Chain: D [auth A])
- Center  : 1.872, -8.260, -1.361
- Name    : AMG 510 (BOUND FORM)
- Synonyms: 6-FLUORO-7-(2-FLUORO-6-HYDROXYPHENYL)-4-[(2S)-2-METHYL-4-PROPANOYLPIPERAZIN-1-YL]-1-[4-METHYL-2-(PROPAN-2-YL)PYRIDIN-3-YL]PYRIDO[2,3-D]PYRIMIDIN-2(1H)-ONE

INFO:root:Select the ligand number(s) (ex. 3 ; 1,3 ; manual ; all ; exit)
ligand number:3 # USER INPUT: Enter the ligand number for binding site detection
INFO:root:Running 3th Ligand...
Ligand 3
- ID      : MOV (Chain: D [auth A])
- Center  : 1.872, -8.260, -1.361
- Name    : AMG 510 (BOUND FORM)
- Synonyms: 6-FLUORO-7-(2-FLUORO-6-HYDROXYPHENYL)-4-[(2S)-2-METHYL-4-PROPANOYLPIPERAZIN-1-YL]-1-[4-METHYL-2-(PROPAN-2-YL)PYRIDIN-3-YL]PYRIDO[2,3-D]PYRIMIDIN-2(1H)-ONE
INFO:root:Save Pharmacophore Model to result/6oim/6oim_D_MOV_model.pm
INFO:root:Save Pymol Visualization Session to result/6oim/6oim_D_MOV_model.pse
```

#### Example with custom protein

```bash
# With reference ligand.
> python modeling.py --protein ./examples/6OIM_protein.pdb --ref_ligand ./examples/6OIM_D_MOV.pdb --prefix 6oim
INFO:root:Load PharmacoNet finish
INFO:root:Load examples/6OIM_protein.pdb
INFO:root:Using center of examples/6oim_D_MOV.pdb as center of box
INFO:root:Save Pharmacophore Model to result/6oim/6oim_6oim_D_MOV_model.pm
INFO:root:Save Pymol Visualization Session to result/6oim/6oim_6oim_D_MOV_model.pse

# Without reference ligand -> center is required.
> python modeling.py --protein ./examples/6OIM_protein.pdb --prefix 6oim
INFO:root:Load PharmacoNet finish
INFO:root:Load examples/6OIM_protein.pdb
WARNING:root:No ligand is detected!
INFO:root:Enter the center of binding site manually:
x: 2 # USER INPUT: Enter x
y: -8 # USER INPUT: Enter y
z: -1 # USER INPUT: Enter z
INFO:root:Using center (2.0, -8.0, -1.0)
INFO:root:Save Pharmacophore Model to result/6OIM/6OIM_2.0_-8.0_-1.0_model.pm
INFO:root:Save Pymol Visualization Session to result/6OIM/6OIM_2.0_-8.0_-1.0_model.pse
```

#### Example with custom model weight file (offline)

PharmacoNet's weight file is automatically downloaded during `modeling.py`.
If your environment is offline, you can download the weight files from [Google Drive](https://drive.google.com/uc?id=1gzjdM7bD3jPm23LBcDXtkSk18nETL04p).

```bash
> python modeling.py --pdb 6oim --weight_path <WEIGHT_PATH>
```

## Virtual Screening

We provide the simple script for screening.

```bash
# Default Parameter Setting (Cation/Anion: 8, Aromatic/Halogen/HBA/HBD: 4, Hydrophobic: 1)
python screening.py -p <MODEL_PATH> --library <LIBRARY_DIR> --out <RESULT_PATH> --cpus <NCPU>

# Custom Parameters Setting
python screening.py -p <MODEL_PATH> --library <LIBRARY_DIR> --out <RESULT_PATH> --cpus <NCPU> \
  --anion <ANION> --cation <CATION> --aromatic <AROMATIC> \
  --hbd <HBD> --hba <HBA> --halogen <HALOGEN> --hydrophobic <HYDROPHOBIC>

# Example
python screening.py -p ./result/6oim/6oim_D_MOV_model.pm --library examples/library --out result.csv --cpus 1
python screening.py -p ./result/6oim/6oim_D_MOV_model.pm --library examples/library --out result.csv --cpus 2 --hbd 5 --hba 5 --aromatic 8
```

#### Example python code for ligand evaluation

Also, it can be easily included in your custom script via the python code below. (\* Multiprocessing is allowed)

```python
from pmnet import PharmacophoreModel
model = PharmacophoreModel.load(<PHARMCOPHORE_MODEL_PATH>)

# NOTE: Scoring with ligand file with 1 or more conformers
score = model.scoring_file(<LIGAND_PATH>) # SDF, MOL2, PDB

# NOTE: Scoring with RDKit ETKDG Conformers
score = model.scoring_smiles(<SMILES>, <NUM_CONFORMERS>)
```

## Pharmacophore Feature Extraction

**_See: [`./developer/`](/developer/), [`./src/pmnet_appl/`](/src/pmnet_appl/)._**

For deep learning researcher who want to use PharmacoNet as pre-trained model for feature extraction, we provide the python API.

```python
from pmnet.api import PharmacoNet, get_pmnet_dev, ProteinParser
from pmnet.typing import PMNetAttr

module: PharmacoNet = get_pmnet_dev('cuda') # default: score_threshold=0.5 (less threshold: more features)

# End-to-End calculation
pmnet_attr: PMNetAttr
pmnet_attr = module.feature_extraction(<PROTEIN_PATH>, ref_ligand_path=<REF_LIGAND_PATH>)
pmnet_attr = module.feature_extraction(<PROTEIN_PATH>, center=(<CENTER_X>, <CENTER_Y>, <CENTER_Z>))

# Step-wise calculation
## In Dataset
parser = ProteinParser(center_noise=<CENTER_NOISE>) # center_noise: for data augmentation
pmnet_attr = module.run_extraction(protein_data)

"""
pmnet_attr: PMNetAttr
- multi_scale_features: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    - [96, 4, 4, 4], [96, 8, 8, 8], [96, 16, 16, 16], [96, 32, 32, 32], [96, 64, 64, 64]
- hotspots:
    hotspot_info: HotspotInfo
      - type: str (7 types)
          {'Hydrophobic', 'Aromatic', 'Cation', 'Anion', 'Halogen', 'HBond_donor', 'HBond_acceptor'}
      - features: Tensor [192,]
      - position: tuple[float, float, float] - (x, y, z)
      - score: float in [0, 1]
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
      - density_type: str (7 types)
          {'Hydrophobic', 'Aromatic', 'Cation', 'Anion', 'Halogen', 'HBond_donor', 'HBond_acceptor'}
"""
```

## Pretrained Docking Proxy

**_See: [`./src/pmnet_appl/`](/src/pmnet_appl/)._**

We provide pre-trained docking proxy models which predict docking score against arbitrary protein using PharmacoNet.
We hope this implementation prompts the molecule optimization.

If you use this implementation, please cite PharmacoNet with original papers.

Implementation List:

- TacoGFN: Target-conditioned GFlowNet for Structure-based Drug Design [[paper](https://arxiv.org/abs/2310.03223)]

Related Works:

- RxnFlow: Generative Flows on Synthetic Pathway for Drug Design [[paper](https://arxiv.org/abs/2410.04542)]
- CGFlow: Compositional Flows for 3D Molecule and Synthesis Pathway Co-design [[paper](https://arxiv.org/abs/2504.08051)]

## Citation

Paper on [Chemical Science](https://doi.org/10.1039/D4SC04854G), [arXiv](https://arxiv.org/abs/2310.00681).

```bibtex
@article{seo2024pharmaconet,
  title={PharmacoNet: deep learning-guided pharmacophore modeling for ultra-large-scale virtual screening},
  author={Seo, Seonghwan and Kim, Woo Youn},
  journal={Chemical Science},
  year={2024},
  publisher={Royal Society of Chemistry}
}
```
