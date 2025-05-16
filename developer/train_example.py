from src.config import Config
from src.trainer import Trainer

config = Config()
config.data.protein_dir = "./dataset/protein/"
config.data.train_protein_code_path = "./dataset/train_key.txt"
config.data.ligand_path = "./dataset/ligand.pkl"
config.train.max_iterations = 100
config.train.batch_size = 16
config.train.num_workers = 4
config.train.log_every = 1
config.train.print_every = 1
config.train.val_every = 10
config.log_dir = "./result/debug"
trainer = Trainer(config, device="cuda")
trainer.fit()
