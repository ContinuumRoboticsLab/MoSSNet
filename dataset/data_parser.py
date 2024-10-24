import torch
from torch import Tensor
import torch.nn.functional as F
from pathlib import Path
import os
from dataclasses import dataclass

import numpy as np
from typing import Dict, Optional
from PIL import Image, ImageOps
import pandas as pd
import open3d as o3d
import random
from torchvision import transforms

def valid_img(file_path: str) -> bool:
    return file_path.endswith("_0.png")
    
def valid_depth(file_path: str) -> bool:
    return file_path.endswith("_0.bmp") or file_path.endswith("_0.png")

def valid_meta(file_path: str) -> bool:
    return file_path.endswith("_0.txt")

def valid_state(file_path: str) -> bool:
    return file_path.endswith(".csv")

def valid_real_state(file_path: str) -> bool:
    return file_path.endswith(".txt")

def valid_real_img_depth_mask(file_path: str) -> bool:
    return file_path.endswith(".png")

def subsample_dataset(imgs, ratio=0.1, depths=None, metas=None, states=None):
    total_num = len(imgs)
    idx = random.sample(range(0, total_num), int(ratio * total_num))
    subsampled_imgs = list(map(imgs.__getitem__, idx))
    subsampled_depths, subsampled_metas, subsampled_states = None, None, None
    if depths is not None:
        subsampled_depths = list(map(depths.__getitem__, idx))
    if metas is not None:
        subsampled_metas = list(map(metas.__getitem__, idx))
    if states is not None:
        subsampled_states = list(map(states.__getitem__, idx))
    return subsampled_imgs, subsampled_depths, subsampled_metas, subsampled_states

@dataclass
class SimOutput:
    frame_id: str
    state: Tensor
    uv_rgb_img: Tensor
    img: Optional[Tensor] = None
    depth: Optional[Tensor] = None
    mask: Optional[Tensor] = None
    xyz_img: Optional[Tensor] = None
    offsets_img: Optional[Tensor] = None
    s_img: Optional[Tensor] = None
    meta: Optional[Dict] = None

@dataclass
class SimDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=f"{Path(__file__).parent}/data",
        enable_aug=False,
        data_aug_cfg=None,
        compute_normals=False,
        downsample_factor=1,
        subsample_dataset_ratio=1.0,
        transform=False,
    ):
        self.imgs = [os.path.join(base_path, "img", p) for p in sorted(os.listdir(os.path.join(base_path, "img"))) if valid_img(p)]
        self.depths = [os.path.join(base_path, "depth", p) for p in sorted(os.listdir(os.path.join(base_path, "depth"))) if valid_depth(p)]
        self.metas = [os.path.join(base_path, "meta", p) for p in sorted(os.listdir(os.path.join(base_path, "meta"))) if valid_meta(p)]
        self.states = [os.path.join(base_path, "state", os.path.split(p)[-1].split("_")[0] + ".csv") 
            for p in self.imgs if os.path.exists(os.path.join(base_path, "state", os.path.split(p)[-1].split("_")[0] + ".csv"))]

        assert len(self.imgs) == len(self.depths)
        assert len(self.depths) == len(self.metas)
        assert len(self.states) == len(self.metas)
        if subsample_dataset_ratio < 1.0:
            self.imgs, self.depths, self.masks, self.states = subsample_dataset(self.imgs, subsample_dataset_ratio, self.depths, self.masks, self.states)

        self.aug_cfg = data_aug_cfg
        self.enable_aug = enable_aug
        self.compute_normals = compute_normals
        self.min_h = 68
        self.min_w = 397
        self.max_h = self.min_h + 512
        self.max_w = self.min_w + 512
        self.downsample_factor = downsample_factor
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.imgs)

    def parse_meta_txt(self, file_path: str) -> Dict:
        """Parse the meta data into a dict"""
        d = {}
        with open(file_path) as f:
            for line in f:
                key = line.split('\n')[0]
                val = f.readline().split('\n')[0]
                d[key] = np.array(val.split(','), dtype='f')
        return d
    
    def __getitem__(self, index) -> SimOutput:
        """Generates one sample of data"""
        img = Image.open(self.imgs[index])
        w, h = img.size
        img = np.asarray(img)
        if self.transform:
            img = Image.fromarray(img)
            s=.10
            transform = transforms.Compose([transforms.ColorJitter(brightness=s, contrast=s, saturation=s, hue=s),
                                            transforms.ToTensor()])
            img = transform(img).numpy()
            img = img.astype(np.float32)
        else:
            img = img.transpose(2, 0, 1)
            img = (img / 255.0).astype(np.float32)
        

        xx, yy = np.meshgrid(np.linspace(-1, 1, num=w), 
            np.linspace(-1, 1, num=h))
        frame_id = os.path.splitext(os.path.split(self.imgs[index])[-1])[0]

        state = pd.read_csv(self.states[index])
        state = np.asarray(state)[:, :3]

        uv_rgb_img = np.concatenate([xx[None, ...], -yy[None, ...], img], axis=0)

        if w > 512 and h > 512:
            # cropping to 512 x 512
            uv_rgb_img = uv_rgb_img[..., self.min_h:self.max_h, self.min_w:self.max_w]
        
        state=torch.from_numpy(state).float()
        uv_rgb_img=torch.from_numpy(uv_rgb_img).float()
        
        if self.downsample_factor > 1:
            uv_rgb_img = F.interpolate(uv_rgb_img.reshape(1, -1, 512, 512), scale_factor=1/self.downsample_factor).squeeze()

        return SimOutput(
            meta=None,
            frame_id=frame_id,
            state=state,
            uv_rgb_img=uv_rgb_img,
        )


@dataclass
class RealDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=f"{Path(__file__).parent}/data",
        enable_aug=False,
        data_aug_cfg=None,
        compute_normals=False,
        downsample_factor=1,
        subsample_dataset_ratio=1.0,
        transform=False,
    ):
        self.imgs = [os.path.join(base_path, "img", p) for p in sorted(os.listdir(os.path.join(base_path, "img"))) if valid_real_img_depth_mask(p)]
        self.depths = [os.path.join(base_path, "depth", p) for p in sorted(os.listdir(os.path.join(base_path, "depth"))) if valid_real_img_depth_mask(p)]
        self.masks = [os.path.join(base_path, "mask", p) for p in sorted(os.listdir(os.path.join(base_path, "mask"))) if valid_real_img_depth_mask(p)]
        self.states = [os.path.join(base_path, "state", p) for p in sorted(os.listdir(os.path.join(base_path, "state"))) if valid_real_state(p)]

        assert len(self.imgs) == len(self.depths)
        assert len(self.depths) == len(self.masks)
        assert len(self.states) == len(self.masks)

        if subsample_dataset_ratio < 1.0:
            self.imgs, self.depths, self.masks, self.states = subsample_dataset(self.imgs, subsample_dataset_ratio, self.depths, self.masks, self.states)

        self.aug_cfg = data_aug_cfg
        self.enable_aug = enable_aug

        self.extrinsic_mat = np.asarray([[-0.9733839,   0.02282613, -0.22804113,  0.2897211 ],
                                        [-0.05040388,  0.9493431,   0.3101728,  -0.13707037],
                                        [ 0.2235693,   0.31341138, -0.92292476,  0.53125316],
                                        [ 0. ,         0.,          0. ,         1.        ]])
        self.near = 0.3
        self.far = 2.0
        self.min_h = 68
        self.min_w = 397
        self.max_h = self.min_h + 512
        self.max_w = self.min_w + 512
        self.compute_normals = compute_normals
        self.downsample_factor = downsample_factor
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.imgs)

    
    def depth2xyz(self, depth_map, flatten=True, depth_scale=0.0010):

        fx, fy = 442.907562 * 2, 442.907562 * 2
        cx, cy = 317.624878 * 2, 178.311752 * 2
        h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
        z = depth_map * depth_scale
        x = (w - cx) * z / fx
        y = (h - cy) * z / fy

        xyz = np.dstack((x, y, z)).reshape(-1, 3) if flatten else np.dstack((x, y, z))

        xyz[:, [1, 2]] *= -1.

        return xyz  # [N,3]

    def transform_pts(self, pts, trans_mat) -> np.ndarray:
        ones = np.ones((len(pts), 1))
        pts_ones = np.concatenate((pts, ones), axis=1)
        after_trans = (trans_mat @ pts_ones.T).T
        return after_trans[:, :3]

    def parse_shape(self, filepath: str, unit='m') -> np.ndarray:

        if unit != 'm' and unit != 'cm' and unit != 'mm':
            raise NotImplementedError("The specified unit is supported!")

        with open(filepath,"r") as file:
            content = file.read().rstrip('\n').split('\t')

        # locate data
        index_x = content.index("Shape x [cm]") + 2
        index_y = content.index("Shape y [cm]") + 2
        index_z = content.index("Shape z [cm]") + 2
        n = int(content[index_x-1])

        # fill array
        shape = np.zeros((n, 3))
        shape[:,0] = content[index_x:index_x+n]
        shape[:,1] = content[index_y:index_y+n]
        shape[:,2] = content[index_z:index_z+n]

        if unit == 'm':
            shape = shape * 0.01
        elif unit == 'mm':
            shape = shape * 10

        return shape
    
    def __getitem__(self, index) -> SimOutput:
        """Generates one sample of data"""
        img = Image.open(self.imgs[index])
        w, h = img.size
        img = np.asarray(img)
        if self.transform:
            img = Image.fromarray(img)
            s=.10
            transform = transforms.Compose([transforms.ColorJitter(brightness=s, contrast=s, saturation=s, hue=s),
                                            transforms.ToTensor()])
            img = transform(img).numpy()
            img = img.astype(np.float32)
        else:
            img = img.transpose(2, 0, 1)
            img = (img / 255.0).astype(np.float32)

        frame_id = os.path.splitext(os.path.split(self.imgs[index])[-1])[0]

        state = self.parse_shape(self.states[index], unit='m')

        xx, yy = np.meshgrid(np.linspace(-1, 1, num=w), 
            np.linspace(-1, 1, num=h))
        uv_rgb_img = np.concatenate([xx[None, ...], -yy[None, ...], img], axis=0)

        # cropping to 512 x 512
        uv_rgb_img = uv_rgb_img[..., self.min_h:self.max_h, self.min_w:self.max_w]

        state=torch.from_numpy(state).float()
        uv_rgb_img=torch.from_numpy(uv_rgb_img).float()
        
        if self.downsample_factor > 1:
            uv_rgb_img = F.interpolate(uv_rgb_img.reshape(1, -1, 512, 512), scale_factor=1/self.downsample_factor).squeeze()

        return SimOutput(
            frame_id=frame_id,
            state=state,
            uv_rgb_img=uv_rgb_img,
        )