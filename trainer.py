import yaml
import os
from shutil import copyfile
from dataclasses import dataclass
import argparse

import numpy as np
import torch
import random
from tqdm import tqdm
from pathlib import Path

from torch import optim
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from model.mossnet import MossNet
from cfg.config import register_configs, TrainerConfig, Configs
from dataset.data_loader import BatchedInput, setup_dataloader, batched_input_to_device
from dataset.data_parser import SimDataset, RealDataset
from evaluator import Evaluator

@dataclass
class Progress:
    iters_per_epoch: int
    iter: int = 0
    epoch: int = 0
    

    def state_dict(self):
        state_dict_ = {
            "iter": self.iter,
            "epoch": self.epoch,
            "iters_per_epoch": self.iters_per_epoch,
        }
        return state_dict_

class Trainer:

    def __init__(
        self,
        config: TrainerConfig,
        train_loader: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler,
        exp_logger: SummaryWriter,
        exp_dirs: str,
        train_batch_size: int,
        evaluator: Evaluator,
        device: torch.device,
        pretrained_path: str,
        debug: bool = False,
        model_weights_path: str = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.exp_dirs = exp_dirs
        self.exp_logger = exp_logger
        self.train_batch_size = train_batch_size  # used for optimizer creation
        self.iteration = 0
        self.device = device
        self.pretrained_path = pretrained_path
        self.save_every_n_epoch = config.save_ckpt_every_n_epoch
        self.debug = debug

        self.progress = Progress(iters_per_epoch=len(train_loader))
        self.key_metric = config.key_metric
        self.best_results = float('inf')

        if self.pretrained_path is not None:
            if os.path.exists(self.pretrained_path):
                ckpt = torch.load(self.pretrained_path)
                self.progress = Progress(**ckpt["progress"])
                self.model.load_state_dict(ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.best_results = ckpt["best_results"]
                print(f"Loaded checkpoint from {self.pretrained_path}, resume training ...")
        else:
            last_available_ckpt = os.path.join(self.exp_dirs, "checkpoints", "model_last_epoch.pth.tar")
            if os.path.exists(last_available_ckpt):
                ckpt = torch.load(last_available_ckpt)
                self.progress = Progress(**ckpt["progress"])
                self.model.load_state_dict(ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.best_results = ckpt["best_results"]
                print(f"Loaded checkpoint from {last_available_ckpt}, resume training ...")
            elif model_weights_path is not None:
                if os.path.exists(model_weights_path):
                    ckpt = torch.load(model_weights_path)
                    self.model.load_state_dict(ckpt["model"])
                    print(f"Loaded model weights from {model_weights_path}, start training ...")

    def log(self, metas: dict, prefix="train_iter") -> None:
        for k, v in metas.items():
            if k.startswith("vis_"):
                for i, imgs in enumerate(v):
                    self.exp_logger.add_image(f"{prefix}/{k}_{i}", imgs, self.progress.iter, dataformats="HWC")
            else:
                self.exp_logger.add_scalar(f"{prefix}/{k}", v, self.progress.iter)
        self.exp_logger.flush()

    def train(self):
        while self.progress.epoch < self.config.n_epochs:

            self.train_epoch()
            self.progress.epoch += 1

            self.eval()
            self.save_checkpoint(suffix="last_epoch")
            if self.progress.epoch % self.save_every_n_epoch == 0:
                self.save_checkpoint()

    def train_epoch(self):

        # Initialize iterator and get workers working
        iterator = iter(self.train_loader)
        bar = tqdm(total=self.progress.iters_per_epoch, desc=f"Epoch {self.progress.epoch} training", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, batched_frames in enumerate(iterator):
            if self.debug and i > 100:
                break
            self.iteration = i
            batched_frames = batched_input_to_device(batched_frames, self.device)

            with torch.autograd.set_detect_anomaly(True):
                metas = self.train_step(batched_frames)

            self.log(metas, prefix="train_iter")
            self.progress.iter += 1
            bar.update(1)
            bar.set_postfix(metas)
        
        bar.close()


    def train_step(self, batched_frames: BatchedInput):
        total_loss, metas = self.model.train_iter(batched_frames)
        self.update_params(total_loss)
        return metas

    def update_params(self, total_loss):

        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def eval(self):
        metrics = self.evaluator.eval()
        self.evaluator.reset_intermediate_results()
        
        assert self.key_metric in metrics
        if metrics[self.key_metric] < self.best_results:
            self.best_results = metrics[self.key_metric]
            self.save_checkpoint(suffix="best")
        
        self.log(metrics, prefix="eval")

        return metrics

    def save_checkpoint(self, suffix=None):
        
        model_state_dict = self.model.state_dict()
        checkpoint_dict = {
            "model": model_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "best_results": self.best_results,
            "progress": self.progress.state_dict(),
        }

        if suffix is None:
            checkpoint_path = os.path.join(self.exp_dirs, "checkpoints", f"model_epoch{self.progress.epoch}.pth.tar")
        else:
            checkpoint_path = os.path.join(self.exp_dirs, "checkpoints", f"model_{suffix}.pth.tar")
        torch.save(checkpoint_dict, checkpoint_path)
        return checkpoint_path


def setup_trainer(
    configs: Configs, 
    config_path: str, 
    pretrained_path: str = None, 
    debug: bool = False,
    model_weights_path: str = None,
) -> Trainer:

    # setup data
    dataset_path = configs.train_data_cfg.base_path
    if not dataset_path.startswith("/"):
        dataset_path = f"{Path(__file__).parent}/{configs.train_data_cfg.base_path}"
    if "Real" in dataset_path:
        train_dataset = RealDataset(
            base_path=dataset_path,
            downsample_factor=configs.train_data_cfg.downsample_factor,
            subsample_dataset_ratio=configs.train_data_cfg.subsample_dataset_ratio,
            transform = True,
        )
    else:
        train_dataset = SimDataset(
            base_path=dataset_path, 
            compute_normals=configs.train_data_cfg.compute_normals,
            downsample_factor=configs.train_data_cfg.downsample_factor,
            subsample_dataset_ratio=configs.train_data_cfg.subsample_dataset_ratio,
            # transform = True,
        )
    train_loader = setup_dataloader(train_dataset, configs.train_data_cfg)

    # setup model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if configs.trainer_cfg.exp_name.startswith("mossnet_"):
        model = MossNet(
            w_depth_loss=configs.trainer_cfg.w_depth_loss, 
            w_offset_loss=configs.trainer_cfg.w_offset_loss,
            w_s_loss=configs.trainer_cfg.w_s_loss,
            polydeg=configs.trainer_cfg.polydeg,
        )
    ## example of adding your customized network
    # elif configs.trainer_cfg.exp_name.startswith("your_customized_net_"):
    #     model = YourCustomizedNet(
    #         ...your_network_arguments
    #     )
    else:
        print(f"ERROR: Could not find model that corresponds to {configs.trainer_cfg.exp_name}")
    model = model.to(device)

    # setup evaluator
    dataset_path = configs.eval_data_cfg.base_path
    if not dataset_path.startswith("/"):
        dataset_path = f"{Path(__file__).parent}/{configs.eval_data_cfg.base_path}"
    if "Real" in dataset_path:
        eval_dataset = RealDataset(
            base_path=dataset_path,
            downsample_factor=configs.eval_data_cfg.downsample_factor,
            subsample_dataset_ratio=configs.eval_data_cfg.subsample_dataset_ratio,
        )
    else:
        eval_dataset = SimDataset(
            base_path=dataset_path, 
            compute_normals=configs.eval_data_cfg.compute_normals,
            downsample_factor=configs.eval_data_cfg.downsample_factor,
            subsample_dataset_ratio=configs.eval_data_cfg.subsample_dataset_ratio,
        )
    eval_loader = setup_dataloader(eval_dataset, configs.eval_data_cfg)
    evaluator = Evaluator(eval_loader=eval_loader, model=model, device=device, key_metric=configs.trainer_cfg.key_metric, debug=debug)

    # setup output directories
    exp_dirs = os.path.join(configs.trainer_cfg.log_dir, configs.trainer_cfg.exp_name)
    if debug:
        exp_dirs = exp_dirs + "_debug"
    os.makedirs(os.path.join(exp_dirs, "checkpoints"), exist_ok=True)
    copyfile(config_path, os.path.join(exp_dirs, "cfg.yaml"))

    # setup logger
    exp_logger = SummaryWriter(log_dir=os.path.join(exp_dirs, "tb"))

    optimizer = optim.AdamW(model.parameters(), lr=configs.trainer_cfg.lr, weight_decay=configs.trainer_cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=configs.trainer_cfg.lr*2,
                                              steps_per_epoch=len(train_loader),
                                              epochs=configs.trainer_cfg.n_epochs)
    print("Using OneCycleLR scheduler.")

    trainer = Trainer(
        config=configs.trainer_cfg,
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=evaluator,
        exp_dirs=exp_dirs,
        exp_logger=exp_logger,
        train_batch_size=configs.train_data_cfg.batch_size,
        device=device,
        pretrained_path=pretrained_path,
        debug=debug,
        model_weights_path=model_weights_path,
    )

    return trainer

def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(
    config_path: str, 
    pretrained_path: str = None, 
    debug: bool = False, 
    model_weights_path: str = None,
):
    
    config = yaml.safe_load(open(config_path, "r"))
    configs = register_configs(config)
    set_seed_all(configs.trainer_cfg.seed)

    if "SLURM_JOB_ID" in os.environ:
        configs.trainer_cfg.log_dir = f"{configs.trainer_cfg.log_dir}/{os.environ['SLURM_JOB_ID']}"
        print(f"slurm job id found to be {os.environ['SLURM_JOB_ID']}")
    trainer = setup_trainer(configs, config_path, pretrained_path, debug, model_weights_path)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./trainer.py")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default="moss_sim",
        help='Name of the config. Default is ./cfg/moss_sim.yaml',
    )
    parser.add_argument(
        '--pretrained', '-p',
        type=str,
        required=False,
        default=None,
        help='Pretrained checkpoint path to resume training. Default None',
    )
    parser.add_argument(
        '--weights', '-w',
        type=str,
        required=False,
        default=None,
        help='Pretrained weights path to load. Default None',
    )
    parser.add_argument("--debug","-d",action="store_true",help="debug flag")

    FLAGS, unparsed = parser.parse_known_args()
    config_path = f"./cfg/{FLAGS.config}.yaml"

    if FLAGS.debug:
        print("DEBUG MODE ON!")
    print("Training on config: ", config_path)
    if FLAGS.pretrained is not None:
        assert FLAGS.pretrained.endswith(".pth.tar")
        assert os.path.exists(FLAGS.pretrained)
    if FLAGS.weights is not None:
        assert FLAGS.weights.endswith(".pth.tar")
        assert os.path.exists(FLAGS.weights)
    main(
        config_path=config_path, 
        pretrained_path=FLAGS.pretrained, 
        debug=FLAGS.debug, 
        model_weights_path=FLAGS.weights,
    )
