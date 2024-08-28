# Application of PharmacoNet

Example scripts to use PharmacoNet's protein pharmacophore representation, which depend on `torch-geometric`.

```bash
# construct conda environment; pymol-open-source is not required.
conda create -n pmnet-dev python=3.10 openbabel=3.1.1
conda activate pmnet-dev

# install PharmacoNet&torch_geometric
pip install -e '.[dev]' --find-links https://data.pyg.org/whl/torch-2.3.1+cu121.html

# if you want to train model with example scripts
pip install wandb, tensorboard
```
## Dataset
The example dataset (100 or 1,000 pockets) can be available at [Google Drive](https://drive.google.com/drive/folders/1o8tDCsjIqaPRoJhs5SKW4yi0geA9h_Nv?usp=sharing).
The dataset was constructed by CrossDocked2020 and QuickVina 2.1.


## Comming Soon (For Archiving):
- TacoGFN: Target-conditioned GFlowNet for Structure-based Drug Design [[paper](https://arxiv.org/abs/2310.03223)]

<!-- Archives of applications which use PharmacoNet's protein pharmacophore representation. -->
<!---->
<!-- - TacoGFN: Target-conditioned GFlowNet for Structure-based Drug Design [[paper](https://arxiv.org/abs/2310.03223)] (`tacogfn_reward/`) -->
<!-- - LarGFN: Scaling GFlowNets into Large Action Spaces for Synthesizable Chemical Discovery [[paper]()] (`largfn_reward/`) -->
