# from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from typing import Dict, Optional, Sequence
from dataclasses import dataclass
from dataset.data_parser import SimOutput, SimDataset
from cfg.config import DataloaderConfig

def to_device(data, device: str):
    """Recursively copy underlying tensor data to the specified device, similar to `torch.Tensor.to`"""
    if data is None or isinstance(data, (int, float, str)):
        return data
    elif isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, Sequence):
        return [to_device(item, device) for item in data]
    elif isinstance(data, Dict):
        for k, v in data.items():
            data[k] = to_device(v, device)
        return data
    else:
        try:
            return data.to(device)
        except AttributeError as e:
            raise NotImplementedError(
                f"Cannot move {type(data).__name__} to {device}, since it does not implement `to`"
            ) from e

@dataclass(frozen=True)
class BatchedInput:
    """Batched VTK data after label preprocessing and collated by dataloader"""

    frame_id: Sequence[str]
    state: Tensor
    uv_rgb_img: Tensor
    mask: Optional[Tensor] = None
    xyz_img: Optional[Tensor] = None
    offsets_img: Optional[Tensor] = None
    img: Optional[Tensor] = None
    depth: Optional[Tensor] = None
    s_img: Optional[Tensor] = None
    meta: Optional[Sequence[Dict]] = None

def collate_batched_frames(batch: Sequence[SimOutput]) -> BatchedInput:
    """Collate a sequence of SimOutput into a BatchedInput

    Args:
        batch: Sequence of SimOutput

    Returns:
        Collated sequences
    """

    uv_rgb_img = torch.stack([frame.uv_rgb_img for frame in batch])
    state = torch.stack([frame.state for frame in batch])
    frame_id = [frame.frame_id for frame in batch]
    meta = None
    if batch[0].meta is not None:
        meta = [frame.meta for frame in batch]

    return BatchedInput(
        frame_id=frame_id,
        meta=meta,
        state=state,
        uv_rgb_img=uv_rgb_img,
    )

def setup_dataloader(
    dataset: SimDataset,
    dataloader_config: DataloaderConfig,
) -> DataLoader:
    """Setup dataloader"""

    num_workers = dataloader_config.num_workers if dataloader_config.num_workers else 0

    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_config.batch_size,
        drop_last=dataloader_config.drop_last,
        collate_fn=collate_batched_frames,
        num_workers=num_workers,
        pin_memory=dataloader_config.pin_memory,
        shuffle=dataloader_config.shuffle,
    )
    return dataloader

def batched_input_to_device(batched_input: BatchedInput, device: str) -> BatchedInput:
    data_dict = to_device(vars(batched_input), device)
    return BatchedInput(**data_dict)

if __name__ == "__main__":
    dataset = SimDataset()
    cfg = DataloaderConfig()
    cfg.batch_size = 2
    cfg.shuffle = False
    loader = setup_dataloader(dataset, cfg)

    for i, batched in enumerate(loader):
        print('-'*25)
        print(i, batched.img.shape, batched.depth.shape)
        print(batched.frame_id)
        print(batched.meta)