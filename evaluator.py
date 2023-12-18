import os
import yaml
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Any
from tqdm import tqdm
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.mossnet import MossNet
from cfg.config import register_configs, Configs
from dataset.data_loader import BatchedInput, setup_dataloader, batched_input_to_device
from dataset.data_parser import SimDataset, RealDataset

def update_dict_of_list(
    dict_of_list: Dict[str, list], kv: Dict[str, Any], keys_to_ignore: Optional[Sequence[str]] = None
):
    for k, v in kv.items():
        if not keys_to_ignore or k not in keys_to_ignore:
            dict_of_list[k] += v


class Evaluator:

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        key_metric: str = "loss",
        eval_loader: Optional[DataLoader] = None,
        debug: bool = False,
    ):
        self.eval_loader = eval_loader
        self.model = model
        self.device = device
        self.key_metric = key_metric
        self.debug = debug

        self.reset_intermediate_results()

    def reset_intermediate_results(self):
        self.model_outputs: Dict[str, List] = defaultdict(list)
        self.eval_metas: Dict[str, float] = defaultdict(float)

    def eval(self):
        self.reset_intermediate_results()

        with torch.no_grad():
            assert self.eval_loader is not None and len(self.eval_loader) > 0, "Calling eval with an empty dataset"
            bar = tqdm(total=len(self.eval_loader), desc=f"Running inference for eval", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

            for i, batched_frames in enumerate(self.eval_loader):
                if self.debug and i > 100:
                    break
                self.eval_step(batched_frames)
                bar.update(1)
            bar.close()

            self.eval_metas = self.model.compute_metrics(self.model_outputs)

            print(self.key_metric, self.eval_metas[self.key_metric])
            print("\n")

        return self.eval_metas

    def eval_step(self, batched_data: BatchedInput):
        batched_data = batched_input_to_device(batched_data, self.device)

        batch_output = self.model.eval_iter(
            batched_data=batched_data,
        )

        # Store model outputs and labels
        update_dict_of_list(self.model_outputs, batch_output)

def load_model(
    configs: Configs,
    ckpt_file: str,
) -> torch.nn.Module:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get the model and progress from the checkpoint
    if configs.trainer_cfg.exp_name.startswith("mossnet_"):
        model = MossNet(
            w_depth_loss=configs.trainer_cfg.w_depth_loss, 
            w_offset_loss=configs.trainer_cfg.w_offset_loss,
            w_s_loss=configs.trainer_cfg.w_s_loss,
            polydeg=configs.trainer_cfg.polydeg,
        )
    else:
        print(f"ERROR: Could not find model that corresponds to {configs.trainer_cfg.exp_name}")
    model = model.to(device)
    ckpt_dict = torch.load(ckpt_file)
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(ckpt_dict["model"], prefix="module.")
    model.load_state_dict(ckpt_dict["model"], strict=False)

    return model

def run_inference(
    model: nn.Module,
    dataloader: Optional[DataLoader] = None,
    debug: bool = False,
) -> Dict:

    # run evaluation
    evaluator = Evaluator(
        model=model,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        eval_loader=dataloader,
        debug=debug,
    )

    metrics = evaluator.eval()
    return metrics

def main(log_dir: str, ckpt_name: str, debug: bool = False) -> None:
    config_path = os.path.join(log_dir, "cfg.yaml")
    config = yaml.safe_load(open(config_path, "r"))
    configs = register_configs(config)

    ckpt_path = os.path.join(log_dir, "checkpoints", ckpt_name)
    model = load_model(configs, ckpt_path)

    dataset_path = configs.eval_data_cfg.base_path
    if not dataset_path.startswith("/"):
        dataset_path = f"{Path(__file__).parent}/{configs.eval_data_cfg.base_path}"
    if "/MoSS-Real/" in dataset_path or "/Disturbed-Real" in dataset_path:
        dataset = RealDataset(
            base_path=dataset_path,
            downsample_factor=configs.eval_data_cfg.downsample_factor,
            subsample_dataset_ratio=configs.eval_data_cfg.subsample_dataset_ratio,
        )
    else:
        dataset = SimDataset(
            base_path=dataset_path, 
            compute_normals=configs.eval_data_cfg.compute_normals,
            downsample_factor=configs.eval_data_cfg.downsample_factor,
            subsample_dataset_ratio=configs.eval_data_cfg.subsample_dataset_ratio,
        )
    dataloader = setup_dataloader(dataset, configs.eval_data_cfg)

    run_inference(
        model,
        dataloader,
        debug,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("./evaluator.py")
    parser.add_argument(
        '--log', '-l',
        type=str,
        required=True,
        help='Path to the log folder. No default.',
    )
    parser.add_argument(
        '--ckpt', '-c',
        type=str,
        required=False,
        default='model_best.pth.tar',
        help='Checkpoint name, default is model_best.pth.tar',
    )
    parser.add_argument("--debug","-d",action="store_true",help="debug flag")
    
    FLAGS, unparsed = parser.parse_known_args()
    log_dir = FLAGS.log
    ckpt_name = FLAGS.ckpt

    if FLAGS.debug:
        print("DEBUG MODE ON!")

    print("Evaluating log: ", log_dir)
    print("Checkpoint: ", ckpt_name)
    main(log_dir=log_dir, ckpt_name=ckpt_name, debug=FLAGS.debug)