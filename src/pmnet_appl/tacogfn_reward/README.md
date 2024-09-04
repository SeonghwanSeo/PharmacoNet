
# QVina Proxy Model for TacoGFN

If you use these scripts, please cite:
```
@article{seo2023pharmaconet,
  title={PharmacoNet: Accelerating Large-Scale Virtual Screening by Deep Pharmacophore Modeling},
  author={Seo, Seonghwan and Kim, Woo Youn},
  journal={arXiv preprint arXiv:2310.00681},
  year={2023},
  url={https://arxiv.org/abs/2310.00681},
}
@article{shen2024tacogfn,
  title={Taco{GFN}: Target-conditioned {GF}lowNet for Structure-based Drug Design},
  author={Tony Shen and Seonghwan Seo and Grayson Lee and Mohit Pandey and Jason R Smith and Artem Cherkasov and Woo Youn Kim and Martin Ester},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=N8cPv95zOU},
}
```

## Install

To use the proxy model, you need to install torch geometric and associated libraries.
You can simply install them with following scheme at the root directory:

```bash
conda install python=3.10 openbabel=3.1.1

# Case 1. Install only PharmacoNet (already torch-geometric is installed)
pip install -e .

# Case 2. Install both PharmacoNet and torch-geometric
# If you want to use different version, install those libraries manually.
pip install -e '.[appl]' --find-links https://data.pyg.org/whl/torch-2.3.1+cu121.html
```

## Use pre-trained model

```python
from pmnet_appl import get_docking_proxy
from pmnet_appl.tacogfn_reward import TacoGFN_Proxy

device: str | torch.device = "cuda" | "cpu"

# Cache for CrossDocked Targets: 15,201 training pockets + 100 test pockets
cache_db = "train" | "test" | "all" | None

# TacoGFN Reward Function
train_dataset = "ZINCDock15M" | "CrossDocked2020"
proxy: TacoGFN_Proxy = get_docking_proxy("TacoGFN_Reward", "QVina", train_dataset, cache_db, device)
proxy: TacoGFN_Proxy = TacoGFN_Proxy.load("QVina", train_dataset, cache_db, device)

# if cache_db = 'test' | 'all'
print(proxy.scoring("14gs_A", "c1ccccc1"))
print(proxy.scoring_list("14gs_A", ["c1ccccc1", "C1CCCCC1"]))
```

### Use custom target cache

```python
proxy = TacoGFN_Proxy.load("QVina", "ZINCDock15M", None, device)
save_database_path = "./custom-cache-SBDDReward-UniDock_Vina_ZINC.pt"
protein_info_dict = {
    "key1": ("./protein1.pdb", "./ref_ligand1.sdf"),  # reference ligand path
    "key2": ("./protein2.pdb", (1.0, 2.0, 3.0)),  # pocket center
}

cache_dict = proxy.get_cache_database(protein_info_dict, save_database_path, verbose=False)
proxy.update_cache(cache_dict)
proxy.scoring("key1", "c1ccccc1")
proxy.scoring_list("key2", ["c1ccccc1", "C1CCCCC1"])

# Load Custom Target Cache
proxy = get_docking_proxy("TacoGFN_Proxy", "QVina", "ZINC", save_database_path, device)
