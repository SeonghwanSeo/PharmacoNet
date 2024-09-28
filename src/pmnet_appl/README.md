# Applications of PharmacoNet

## Using PharmacoNet's protein representation

Example scripts to use PharmacoNet's protein pharmacophore representation, which depend on `torch-geometric`.

```bash
# construct conda environment; pymol-open-source is not required.
conda create -n pmnet-dev python=3.10 openbabel=3.1.1
conda activate pmnet-dev
# install PharmacoNet & torch_geometric & wandb & tensorboard
pip install -e '.[dev]' --find-links https://data.pyg.org/whl/torch-2.3.1+cu121.html
```

Example datasets (100 or 1,000 pockets) can be available at [Google Drive](https://drive.google.com/drive/folders/1o8tDCsjIqaPRoJhs5SKW4yi0geA9h_Nv?usp=sharing), which are constructed by CrossDocked2020 and QuickVina 2.1.


## Pre-trained Docking Proxy Models

Easy-to-use docking score prediction models.

- TacoGFN: Target-conditioned GFlowNet for Structure-based Drug Design [[paper](https://arxiv.org/abs/2310.03223)]

### Install

To use the pre-trained proxy model, you need to install torch geometric and associated libraries.
You can simply install them with following scheme at the root directory:

```bash
conda install openbabel=3.1.1
# Case 1. Install only PharmacoNet (already torch-geometric is installed)
pip install -e .
# Case 2. Install both PharmacoNet and torch-geometric
pip install -e '.[appl]' --find-links https://data.pyg.org/whl/torch-2.3.1+cu121.html
```

### Load Pretrained Model
```python
from pmnet_appl import get_docking_proxy
from pmnet_appl.tacogfn_reward import TacoGFN_Proxy

device: str | torch.device = "cuda" | "cpu"

# Cache for CrossDocked2020 Targets: 15,201 training pockets + 100 test pockets
cache_db = "train" | "test" | "all" | None

# TacoGFN Reward Function
train_dataset = "ZINCDock15M" | "CrossDocked2020"
proxy: TacoGFN_Proxy = get_docking_proxy("TacoGFN_Reward", "QVina", train_dataset, cache_db, device)
proxy = TacoGFN_Proxy.load("QVina", train_dataset, cache_db, device)

# if cache_db is 'test' | 'all'
print(proxy.scoring("14gs_A", "c1ccccc1"))
print(proxy.scoring_list("14gs_A", ["c1ccccc1", "C1CCCCC1"]))
```

### Use custom target cache

```python
proxy = get_docking_proxy("TacoGFN_Reward", "QVina", "ZINCDock15M", None, device)
save_cache_path = "<custom-cache-pt>"
protein_info_dict = {
    "<keyA>": ("<proteinA-pdb>", "<ref-ligand-path>"), # use center of reference ligand
    "<keyB>": ("<proteinB-pdb>", (1.0, 2.0, 3.0)),     # use center coordinates
}
proxy.get_cache_database(protein_info_dict, save_cache_path, verbose=False)

# Load Custom Target Cache
proxy = get_docking_proxy("TacoGFN_Reward", "QVina", "ZINCDock15M", save_cache_path, device)
proxy.scoring("key1", "c1ccccc1")
proxy.scoring_list("key2", ["c1ccccc1", "C1CCCCC1"])
