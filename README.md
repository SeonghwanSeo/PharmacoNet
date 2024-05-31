# PharmacoNet: Open-source Protein-based Pharmacophore Modeling

**Before using PharmacoNet, also consider using PharmacoGUI - GUI powered by PharmacoNet.**

**[OpenPharmaco Github](https://github.com/SeonghwanSeo/OpenPharmaco) (Coming soon!)**

Accepted in ***NeurIPS Workshop 2023 (AI4D3 | New Frontiers of AI for Drug Discovery and Development)*** [[arxiv](https://arxiv.org/abs/2310.00681)]

Official Github for ***PharmacoNet: Accelerating Large-Scale Virtual Screening by Deep Pharmacophore Modeling*** by Seonghwan Seo* and Woo Youn Kim.

1. Fully automated protein-based pharmacophore modeling based on image instance segmentation modeling
2. Coarse-grained graph matching at the pharmacophore level for high throughput
3. Pharmacophore-aware scoring function with parameterized analytical function for robust generalization ability

PharmacoNet is an extremely rapid yet reasonably accurate ligand evaluation tool with high generation ability.

If you have any problems or need help with the code, please add an github issue or contact [shwan0106@kaist.ac.kr](mailto:shwan0106@kaist.ac.kr).

![x	](images/overview.png)



## Quick Start

```bash
# Pharmacophore Modeling
python modeling.py --pdb <PDB ID> 		# RCSB PDB importing
python modeling.py --protein <PROTEIN_PATH> --prefix <EXP_NAME> --cuda 	# CUDA acceleration
python modeling.py --protein <PROTEIN_PATH> --prefix <EXP_NAME> --ref_ligand <REF_LIGAND_PATH>

# Virtual Screening
python screening.py -p <MODEL_PATH> --library <LIBRARY_DIR> --out <RESULT_PATH> --cpus <NCPU>

# Feature Extraction for Deep Learning Researcher
python feature_extraction.py --protein <PROTEIN_PATH> --ref_ligand <REF_LIGAND_PATH> --out <SAVE_PKL_PATH>
python feature_extraction.py --protein <PROTEIN_PATH> --center <X> <Y> <Z> --out <SAVE_PKL_PATH> --cuda
```



## Environment

#### Installation with `environment.yml`

For various environment including Linux, MacOS and Window, the script installs **cpu-only version of PyTorch** by default.  You can install a cuda-available version by modifying `environment.yml` or installing PyTorch manually.

```bash
conda create -f environment.yml
conda activate pmnet
```

#### Manual Installation

```shell
# Required python>=3.9, Best Performance at higher version. (3.9, 3.10, 3.11, 3.12 - best)
conda create --name pmnet python=3.10 openbabel=3.1.1 pymol-open-source=3.0.0 numpy=1.26
conda activate pmnet

pip install torch # torch >= 1.13, CUDA acceleration is available. 1min for 1 cpu, 10s for 1 gpu
pip install rdkit biopython omegaconf numba # Numba is optional, but recommended.
pip install molvoxel # https://github.com/SeonghwanSeo/molvoxel.git
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
ligand number:3	# USER INPUT: Enter the ligand number for binding site detection
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
x: 2	# USER INPUT: Enter x
y: -8	# USER INPUT: Enter y
z: -1	# USER INPUT: Enter z
INFO:root:Using center (2.0, -8.0, -1.0)
INFO:root:Save Pharmacophore Model to result/6OIM/6OIM_2.0_-8.0_-1.0_model.pm
INFO:root:Save Pymol Visualization Session to result/6OIM/6OIM_2.0_-8.0_-1.0_model.pse
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
python screening.py -p ./result/6oim/6oim_D_MOV_model.pm --library examples/library --out result.csv --cpus 1 --hbd 5 --hba 5 --aromatic 8
```



#### Example python code for ligand evaluation

Also, it can be easily included in your custom script via the python code below. (\* Multiprocessing is allowed)

```python
from pmnet import PharmacophoreModel
model = PharmacophoreModel.load(<PHARMCOPHORE_MODEL_PATH>)

# NOTE: Scoring with ligand file with 1 or more conformers
score = model.scoring_file(<LIGAND_PATH>)	# SDF, MOL2, PDB

# NOTE: Scoring with RDKit ETKDG Conformers
score = model.scoring_smiles(<SMILES>, <NUM_CONFORMERS>)
```



## Pharmacophore Feature Extraction

For deep learning researcher who want to use PharmacoNet as pre-trained model for feature extraction, we provide the script `feature_extraction.py`.

```bash
python feature_extraction.py --protein <PROTEIN_PATH> --ref_ligand <REF_LIGAND_PATH> --out <SAVE_PKL_PATH>
python feature_extraction.py --protein <PROTEIN_PATH> --center <X> <Y> <Z> --out <SAVE_PKL_PATH>
```

```bash
PHARMACOPHORE NODE FEATURE LIST: list[dict[str, Any]]
    PHARMACOPHORE NODE FEATURE: dict[str, Any]
        - feature: NDArray[np.float32]
        - type: str (7 types)
            {'Hydrophobic', 'Aromatic', 'Cation', 'Anion',
             'Halogen', 'HBond_donor', 'HBond_acceptor'}
            *** `type` is obtained from `nci_type`.
        - nci_type: str (10 types)
            'Hydrophobic': Hydrophobic interaction
            'PiStacking_P': Pi-Pi Stacking (Parallel)
            'PiStacking_T': Pi-Pi Stacking (T-shaped)
            'PiCation_lring': Cation-Pi Interaction btw Protein Cation & Ligand Aromatic Ring
            'PiCation_pring': Cation-Pi Interaction btw Protein Aromatic Ring & Ligand Cation
            'SaltBridge_pneg': SaltBridge btw Protein Anion & Ligand Cation
            'SaltBridge_lneg': SaltBridge btw Protein Cation & Ligand Anion
            'HBond_pdon': Hydrogen Bond btw Protein Donor & Ligand Acceptor
            'HBond_ldon': Hydrogen Bond btw Protein Acceptor & Ligand Donor
            'XBond': Halogen Bond
        - hotspot_position: tuple[float, float, float] - (x, y, z)
        - priority_score: str in [0, 1]
        - center: tuple[float, float, float] - (x, y, z) 
        - radius: float
```



### Python Script

For feature extraction, it is recommended to use `score_threshold=0.5` instead of default setting used for pharmacophore modeling. If you want to extract more features, decrease the `score_threshold`.

```python
from pmnet.module import PharmacoNet

module = PharmacoNet(
    "cuda",
    score_threshold = 0.5		# <SCORE_THRESHOLD: float | dict[str, float], recommended=0.5>,
)
pharmacophore_node_feature_list = module.feature_extraction(<PROTEIN_PATH>, center=(<X>, <Y>, <Z>))
```



### Paper List

- TacoGFN [[paper](https://arxiv.org/abs/2310.03223)]



## Citation

Paper on [arxiv](https://arxiv.org/abs/2310.00681)

```
@article{seo2023pharmaconet,
  title = {PharmacoNet: Accelerating Large-Scale Virtual Screening by Deep Pharmacophore Modeling},
  author = {Seo, Seonghwan and Kim, Woo Youn},
  journal = {arXiv preprint arXiv:2310.00681},
  year = {2023},
  url = {https://arxiv.org/abs/2310.00681},
}
```

