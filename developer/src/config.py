from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    ligand_num_convs: int = 4


@dataclass
class DataConfig:
    protein_info_path: str = MISSING
    train_protein_code_path: str = MISSING
    protein_dir: str = MISSING
    ligand_path: str = MISSING


@dataclass
class LrSchedulerConfig:
    scheduler: str = "lambdalr"
    lr_decay: int = 50_000


@dataclass
class OptimizerConfig:
    opt: str = "adam"
    lr: float = 1e-3
    eps: float = 1e-8
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.05
    clip_grad: float = 1.0


@dataclass
class TrainConfig:
    val_every: int = 2_000
    log_every: int = 10
    print_every: int = 100
    save_every: int = 1_000
    max_iterations: int = 300_000
    batch_size: int = 4
    num_workers: int = 4

    opt: OptimizerConfig = OptimizerConfig()
    lr_scheduler: LrSchedulerConfig = LrSchedulerConfig()

    # NOTE: HYPER PARAMETER
    split_ratio: float = 0.9
    center_noise: float = 3.0


@dataclass
class Config:
    log_dir: str = MISSING
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()

    def to_dict(self):
        return config_to_dict(self)


def config_to_dict(obj) -> dict:
    if not hasattr(obj, "__dataclass_fields__"):
        return obj
    result = {}
    for field in obj.__dataclass_fields__.values():
        value = getattr(obj, field.name)
        result[field.name] = config_to_dict(value)
    return {"config": result}
