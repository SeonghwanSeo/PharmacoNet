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
  ```
  @article{
    shen2024tacogfn,
    title={Taco{GFN}: Target-conditioned {GF}lowNet for Structure-based Drug Design},
    author={Tony Shen and Seonghwan Seo and Grayson Lee and Mohit Pandey and Jason R Smith and Artem Cherkasov and Woo Youn Kim and Martin Ester},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2024},
    url={https://openreview.net/forum?id=N8cPv95zOU},
    note={}
  }
  ```
- SBDDReward: 

