from dataclasses import dataclass
from typing import Optional

@dataclass
class DataloaderConfig:
    base_path: str = "dataset/data"
    batch_size: int = 1
    num_workers: Optional[int] = 0
    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = True
    compute_normals: bool = False
    downsample_factor: int = 1
    subsample_dataset_ratio: float = 1.0

@dataclass
class TrainerConfig:
    log_dir: str
    exp_name: str
    n_epochs: int
    lr: float = 0.003
    weight_decay: float = 1e-4
    seed: int = 5201314
    key_metric: str = "loss"
    w_robot_loss: float = 10.0
    w_depth_loss: float = 100.0
    w_offset_loss: float = 100.0
    w_s_loss: float = 100.0
    save_ckpt_every_n_epoch: int = 1
    polydeg: int = 4

@dataclass
class Configs:
    trainer_cfg: TrainerConfig
    train_data_cfg: DataloaderConfig
    eval_data_cfg: DataloaderConfig

def register_configs(cfg) -> Configs:
    trainer_cfg = TrainerConfig(**cfg["trainer_cfg"])
    train_data_cfg = DataloaderConfig(**cfg["train_data_cfg"])
    eval_data_cfg = DataloaderConfig(**cfg["eval_data_cfg"])
    return Configs(trainer_cfg=trainer_cfg, train_data_cfg=train_data_cfg, eval_data_cfg=eval_data_cfg)