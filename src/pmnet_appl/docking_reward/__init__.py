import os
import wandb

from pmnet_appl.docking_reward.config import Config
from pmnet_appl.docking_reward.trainer import Trainer


def run_config(config: Config, project: str, name: str):
    wandb.init(project=project, config=config.to_dict(), name=name)
    trainer = Trainer(config, device="cuda")
    trainer.fit()


if __name__ == "__main__":
    PROJECT = "pmnet-appl"
    NAME = "debug"

    config = Config()
    config.data.protein_dir = "/home/share/DATA/SBDDReward/protein/train/"
    config.data.ligand_dir = "/home/share/DATA/SBDDReward/lmdb/train"
    config.data.ligand_dir = "/home/shwan/GFLOWNET_PROJECT/DATA/"

    config.train.max_iterations = 1000
    config.train.batch_size = 8

    config.log_dir = f"./result/{NAME}"
    assert not os.path.exists(config.log_dir)
    os.mkdir(config.log_dir)
    run_config(config, PROJECT, NAME)
