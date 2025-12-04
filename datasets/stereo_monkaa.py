"""
SceneFlow Monkaa Dataset for Stereo Depth Estimation.

SceneFlow dataset: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
Virtual image sensor size: 960 px x 540 px or 32mm x 18mm
Virtual focal length: 35mm
Baseline: 1 Blender unit
"""

from typing import Tuple
import os
import torch
import numpy as np
import imageio
from torch.utils.data import Dataset
from datasets.augmentation import RandomTransform
from kornia.augmentation import CenterCrop
from torchvision import transforms
import torch.nn.functional as F
from datasets.frame_utils import readPFM

DATA_ROOT = '/mnt/ssd1/datasets/SceneFlow/monkaa/'


class Monkaa(Dataset):
    """SceneFlow Monkaa dataset for stereo matching training."""

    def __init__(self, dataset: str, image_size: Tuple[int, int], is_training: bool = True, randcrop: bool = False,
                 augment: bool = False, padding: int = 0, singleplane: bool = False, n_depths: int = 16):
        super().__init__()
        self.flip = transforms.RandomHorizontalFlip(p=1)
        self.transform = RandomTransform(image_size, randcrop, augment)
        self.centercrop = CenterCrop(image_size)

        disp_dir = os.path.join(DATA_ROOT, 'disparity')
        image_dir = os.path.join(DATA_ROOT, 'frames_cleanpass')
        scenes = sorted(os.listdir(disp_dir))
        self.sample_ids = []
        for scene in scenes:
            right_image_dir = os.path.join(image_dir, scene, 'right')
            left_image_dir = os.path.join(image_dir, scene, 'left')
            disparity_dir = os.path.join(disp_dir, scene, 'left')
            disparity_2_dir = os.path.join(disp_dir, scene, 'right')
            for filename in sorted(os.listdir(left_image_dir)):
                index = os.path.splitext(filename)[0]
                sample_id = {
                    'right_image_dir': right_image_dir,
                    'left_image_dir': left_image_dir,
                    'disparity_dir': disparity_dir,
                    'disparity_2_dir': disparity_2_dir,
                    'id': index
                }
                self.sample_ids.append(sample_id)

        self.is_training = torch.tensor(is_training)
        self.padding = padding
        self.singleplane = torch.tensor(singleplane)
        self.n_depths = n_depths
        self.image_size = image_size

    def stretch_depth(self, depth, depth_range, min_depth):
        return depth_range * depth + min_depth

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        right_image_dir = sample_id['right_image_dir']
        left_image_dir = sample_id['left_image_dir']
        disparity_dir = sample_id['disparity_dir']
        disparity_2_dir = sample_id['disparity_2_dir']
        id = sample_id['id']

        disparity = np.flip(readPFM(os.path.join(disparity_dir, f'{id}.pfm')), axis=0).astype(np.float32)
        disparity_2 = np.flip(readPFM(os.path.join(disparity_2_dir, f'{id}.pfm')), axis=0).astype(np.float32)
        
        left_img = imageio.imread(os.path.join(left_image_dir, f'{id}.png')).astype(np.float32) / 255.
        right_img = imageio.imread(os.path.join(right_image_dir, f'{id}.png')).astype(np.float32) / 255.

        # Apply padding
        left_img = np.pad(left_img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        right_img = np.pad(right_img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        disparity = np.pad(disparity, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')
        disparity_2 = np.pad(disparity_2, ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')

        # Convert to tensors
        right_img = torch.from_numpy(right_img).permute(2, 0, 1)
        left_img = torch.from_numpy(left_img).permute(2, 0, 1)
        disparity = torch.from_numpy(disparity)[None, ...]
        disparity_2 = torch.from_numpy(disparity_2)[None, ...]

        # Create mirrored images
        left_img_m = self.flip(right_img)
        right_img_m = self.flip(left_img)

        # Process depth maps
        depthmap = disparity * 1.0
        unnorm_depthmap = torch.clamp(depthmap, min=0) / 255
        depthmap -= depthmap.min()
        depthmap /= depthmap.max()

        depthmap_2 = disparity_2 * 1.0
        unnorm_depthmap_2 = self.flip(torch.clamp(depthmap_2, min=0) / 255)
        depthmap_2 -= depthmap_2.min()
        depthmap_2 /= depthmap_2.max()
        depthmap_2 = self.flip(depthmap_2)

        # Center crop
        size = (192, 576)
        right_img = self.centercrop(right_img, size=size)
        left_img = self.centercrop(left_img, size=size)
        right_img_m = self.centercrop(right_img_m, size=size)
        left_img_m = self.centercrop(left_img_m, size=size)
        depthmap = self.centercrop(depthmap, size=size)
        depthmap_2 = self.centercrop(depthmap_2, size=size)
        disparity = self.centercrop(disparity, size=size)
        disparity_2 = self.centercrop(disparity_2, size=size)
        unnorm_depthmap = self.centercrop(unnorm_depthmap, size=size)
        unnorm_depthmap_2 = self.centercrop(unnorm_depthmap_2, size=size)

        # Store original resolution
        original_left = left_img
        original_right = right_img
        original_left_m = left_img_m
        original_right_m = right_img_m
        original_depth = depthmap
        original_depth2 = depthmap_2

        # Downsample for training
        c, d = size
        scale = 1.5
        left_img = F.interpolate(left_img, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        right_img = F.interpolate(right_img, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        left_img_m = F.interpolate(left_img_m, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        right_img_m = F.interpolate(right_img_m, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        depthmap = F.interpolate(depthmap, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        depthmap_2 = F.interpolate(depthmap_2, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        unnorm_depthmap = F.interpolate(unnorm_depthmap, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        unnorm_depthmap_2 = F.interpolate(unnorm_depthmap_2, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)

        # Remove batch dim
        original_left = original_left.squeeze(0)
        original_right = original_right.squeeze(0)
        original_left_m = original_left_m.squeeze(0)
        original_right_m = original_right_m.squeeze(0)
        original_depth = original_depth.squeeze(0)
        original_depth2 = original_depth2.squeeze(0)
        right_img = right_img.squeeze(0)
        left_img = left_img.squeeze(0)
        right_img_m = right_img_m.squeeze(0)
        left_img_m = left_img_m.squeeze(0)
        depthmap = depthmap.squeeze(0)
        depthmap_2 = depthmap_2.squeeze(0)
        disparity = (disparity * (-1.0)).squeeze(0)
        disparity_2 = disparity_2.squeeze(0)
        unnorm_depthmap = unnorm_depthmap.squeeze(0)
        unnorm_depthmap_2 = unnorm_depthmap_2.squeeze(0)

        if self.singleplane:
            if self.is_training:
                depthmap = torch.rand((1,), device=depthmap.device) * torch.ones_like(depthmap)
                depthmap_2 = torch.rand((1,), device=depthmap_2.device) * torch.ones_like(depthmap_2)
                original_depth = torch.rand((1,), device=original_depth.device) * torch.ones_like(original_depth)
                original_depth2 = torch.rand((1,), device=original_depth2.device) * torch.ones_like(original_depth2)
            else:
                depthmap = torch.linspace(0., 1., steps=self.n_depths)[idx % self.n_depths] * torch.ones_like(depthmap)
                depthmap_2 = torch.linspace(0., 1., steps=self.n_depths)[idx % self.n_depths] * torch.ones_like(depthmap_2)
                original_depth = torch.linspace(0., 1., steps=self.n_depths)[idx % self.n_depths] * torch.ones_like(original_depth)
                original_depth2 = torch.linspace(0., 1., steps=self.n_depths)[idx % self.n_depths] * torch.ones_like(original_depth2)

        sample = {
            'id': id,
            'right_image': right_img, 'left_image': left_img,
            'right_image_m': right_img_m, 'left_image_m': left_img_m,
            'original_left': original_left, 'original_right': original_right,
            'original_left_m': original_left_m, 'original_right_m': original_right_m,
            'original_depth': original_depth, 'original_depth2': original_depth2,
            'depthmap': depthmap, 'depthmap_2': depthmap_2,
            'disparity': disparity, 'disparity_2': disparity_2,
            'unnorm_depthmap': unnorm_depthmap, 'unnorm_depthmap_2': unnorm_depthmap_2,
            'depth_conf': torch.ones_like(depthmap)
        }

        return sample