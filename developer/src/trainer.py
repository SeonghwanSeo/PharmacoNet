import gc
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from pmnet.api import PharmacoNet

from .config import Config
from .dataset import BaseDataset
from .model import AffinityModel

torch.multiprocessing.set_sharing_strategy("file_system")


class Trainer:
    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True)
        self.save_dir = self.log_dir / "save"
        self.save_dir.mkdir(parents=True)

        self.dictconfig = OmegaConf.create(config.to_dict())
        OmegaConf.save(self.dictconfig, self.log_dir / "config.yaml")
        self.logger = create_logger(logfile=self.log_dir / "train.log")

        self.model = AffinityModel(config)
        self.model.to(device)
        self.pmnet: PharmacoNet = self.model.pmnet
        self.setup_data()
        self.setup_train()

    def fit(self):
        it = 1
        epoch = 0
        best_loss = float("inf")
        self.model.train()
        while it <= self.config.train.max_iterations:
            for batch in self.train_dataloader:
                if it > self.config.train.max_iterations:
                    break
                if it % 1024 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                tick = time.time()
                info = self.train_batch(batch)
                info["time"] = time.time() - tick

                if it % self.config.train.print_every == 0:
                    self.logger.info(
                        f"epoch {epoch} iteration {it} train : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items())
                    )
                if it % self.config.train.log_every == 0:
                    self.log(info, it, epoch, "train")
                if it % self.config.train.save_every == 0:
                    self.save_checkpoint(f"epoch-{epoch}-it-{it}.pth")
                if it % self.config.train.val_every == 0:
                    tick = time.time()
                    info = self.evaluate()
                    info["time"] = time.time() - tick
                    self.logger.info(
                        f"epoch {epoch} iteration {it} valid : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items())
                    )
                    self.log(info, it, epoch, "valid")
                    if info["loss"] < best_loss:
                        torch.save(self.model.state_dict(), self.save_dir / "best.pth")
                        best_loss = info["loss"]
                it += 1
            epoch += 1
        torch.save(self.model.state_dict(), self.save_dir / "last.pth")

    def log(self, info, index, epoch, key):
        info.update({"step": index, "epoch": epoch})
        if wandb.run is not None:
            wandb.log({f"{key}/{k}": v for k, v in info.items()}, step=index)

    def train_batch(self, batch) -> dict[str, float]:
        loss = self.model.forward_train(batch)
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.train.opt.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        logs = {"loss": []}
        for batch in self.val_dataloader:
            loss = self.model.forward_train(batch)
            logs["loss"].append(loss.item())
        self.model.train()
        return {k: float(np.mean(v)) for k, v in logs.items()}

    def setup_data(self):
        config = self.config
        protein_info = {}
        with open(config.data.protein_info_path) as f:
            lines = f.readlines()
        for line in lines:
            code, x, y, z = line.strip().split(",")
            protein_info[code] = (float(x), float(y), float(z))

        with open(config.data.train_protein_code_path) as f:
            codes = [ln.strip() for ln in f.readlines()]
        random.seed(0)
        random.shuffle(codes)
        split_offset = int(len(codes) * config.train.split_ratio)
        train_codes = codes[:split_offset]
        val_codes = codes[split_offset:]

        self.train_dataset = BaseDataset(
            train_codes,
            protein_info,
            config.data.protein_dir,
            config.data.ligand_path,
            config.train.center_noise,
        )

        self.val_dataset = BaseDataset(
            val_codes,
            protein_info,
            config.data.protein_dir,
            config.data.ligand_path,
        )

        self.train_dataloader: DataLoader = DataLoader(
            self.train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )

        self.val_dataloader: DataLoader = DataLoader(
            self.val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
            collate_fn=collate_fn,
        )

        self.logger.info(f"train set: {len(self.train_dataset)}")
        self.logger.info(f"valid set: {len(self.val_dataset)}")

    def setup_train(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.config.train.opt.lr,
            eps=self.config.train.opt.eps,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda steps: 2 ** (-steps / self.config.train.lr_scheduler.lr_decay),
        )

    def save_checkpoint(self, filename: str):
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "config": self.dictconfig,
        }
        torch.save(ckpt, self.save_dir / filename)


def collate_fn(batch):
    return batch


def create_logger(name="logger", loglevel=logging.INFO, logfile=None, streamHandle=True):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
    )
    handlers = []
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="a"))
    if streamHandle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in logger.handlers[:]:
        logging.root.removeHandler(handler)

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
