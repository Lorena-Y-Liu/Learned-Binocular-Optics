"""
SceneFlow FlyingThings3D Dataset Loader.

This module provides a PyTorch Dataset for loading stereo image pairs
and disparity maps from the SceneFlow FlyingThings3D dataset.

Dataset source: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
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


# Dataset root path - modify this to match your data location
DATA_ROOT = '/mnt/ssd1/datasets/SceneFlow'

# Training data paths
TRAIN_LEFT_IMAGE_PATH = [
    os.path.join(DATA_ROOT, 'FlyingThings3D_subset', 'train', 'image_clean', 'left')
]
TRAIN_RIGHT_IMAGE_PATH = [
    os.path.join(DATA_ROOT, 'FlyingThings3D_subset', 'train', 'image_clean', 'right')
]
TRAIN_DISPARITY_PATH = [
    os.path.join(DATA_ROOT, 'FlyingThings3D_subset_disparity', 'train', 'disparity', 'left')
]
TRAIN_DISPARITY_2_PATH = [
    os.path.join(DATA_ROOT, 'FlyingThings3D_subset_disparity', 'train', 'disparity', 'right')
]

# Validation data paths
VALIDATION_LEFT_IMAGE_PATH = [
    os.path.join(DATA_ROOT, 'FlyingThings3D_subset', 'val', 'image_clean', 'left')
]
VALIDATION_RIGHT_IMAGE_PATH = [
    os.path.join(DATA_ROOT, 'FlyingThings3D_subset', 'val', 'image_clean', 'right')
]
VALIDATION_DISPARITY_PATH = [
    os.path.join(DATA_ROOT, 'FlyingThings3D_subset_disparity', 'val', 'disparity', 'left')
]
VALIDATION_DISPARITY_2_PATH = [
    os.path.join(DATA_ROOT, 'FlyingThings3D_subset_disparity', 'val', 'disparity', 'right')
]

# Example data paths (for quick testing)
EXAMPLE_LEFT_IMAGE_PATH = [
    os.path.join(DATA_ROOT, 'example', 'FlyingThings3D', 'RGB_cleanpass', 'left')
]
EXAMPLE_RIGHT_IMAGE_PATH = [
    os.path.join(DATA_ROOT, 'example', 'FlyingThings3D', 'RGB_cleanpass', 'right')
]
EXAMPLE_DISPARITY_PATH = [
    os.path.join(DATA_ROOT, 'example', 'FlyingThings3D', 'disparity')
]


class SceneFlow(Dataset):
    """
    SceneFlow FlyingThings3D dataset for stereo depth estimation.
    
    Virtual camera specifications:
    - Image sensor size: 960 x 540 px (32mm x 18mm)
    - Focal length: 35mm
    - Baseline: 1 Blender unit
    
    Args:
        dataset: Dataset split ('train', 'val', or 'example')
        image_size: Target image size as (height, width)
        is_training: Whether this is for training
        augment: Whether to apply data augmentation
        padding: Padding to add around images
        singleplane: If True, use single-plane depth (for testing)
        n_depths: Number of depth planes for single-plane mode
    """

    def __init__(
        self, 
        dataset: str, 
        image_size: Tuple[int, int], 
        is_training: bool = True,
        augment: bool = False, 
        padding: int = 0, 
        singleplane: bool = False, 
        n_depths: int = 16
    ):
        super().__init__()
        self.flip = transforms.RandomHorizontalFlip(p=1)
        
        # Select data paths based on dataset split
        if dataset == 'train':
            left_image_dirs = TRAIN_LEFT_IMAGE_PATH
            right_image_dirs = TRAIN_RIGHT_IMAGE_PATH
            disparity_dirs = TRAIN_DISPARITY_PATH
            disparity_2_dirs = TRAIN_DISPARITY_2_PATH
        elif dataset == 'val':
            left_image_dirs = VALIDATION_LEFT_IMAGE_PATH
            right_image_dirs = VALIDATION_RIGHT_IMAGE_PATH
            disparity_dirs = VALIDATION_DISPARITY_PATH
            disparity_2_dirs = VALIDATION_DISPARITY_2_PATH
        elif dataset == 'example':
            left_image_dirs = EXAMPLE_LEFT_IMAGE_PATH
            right_image_dirs = EXAMPLE_RIGHT_IMAGE_PATH
            disparity_dirs = EXAMPLE_DISPARITY_PATH
            disparity_2_dirs = EXAMPLE_DISPARITY_PATH
        else:
            raise ValueError(f'Invalid dataset: {dataset}. Must be "train", "val", or "example".')

        self.transform = RandomTransform(image_size, augment)
        self.centercrop = CenterCrop(image_size)

        # Build sample list
        self.sample_ids = []
        for right_dir, left_dir, disp_dir, disp_2_dir in zip(
            right_image_dirs, left_image_dirs, disparity_dirs, disparity_2_dirs
        ):
            for filename in sorted(os.listdir(left_dir)):
                if filename.endswith('.png'):
                    sample_id = os.path.splitext(filename)[0]
                    disparity_path = os.path.join(disp_dir, f'{sample_id}.pfm')
                    
                    if os.path.exists(disparity_path):
                        self.sample_ids.append({
                            'right_image_dir': right_dir,
                            'left_image_dir': left_dir,
                            'disparity_dir': disp_dir,
                            'disparity_2_dir': disp_2_dir,
                            'id': sample_id,
                        })
                    else:
                        print(f'Warning: Disparity not found: {disparity_path}')
        
        self.is_training = torch.tensor(is_training)
        self.padding = padding
        self.singleplane = torch.tensor(singleplane)
        self.n_depths = n_depths

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.sample_ids[idx]
        
        # Load images and disparity maps
        left_img = self._load_image(sample_id['left_image_dir'], sample_id['id'])
        right_img = self._load_image(sample_id['right_image_dir'], sample_id['id'])
        disparity = self._load_disparity(sample_id['disparity_dir'], sample_id['id'])
        disparity_2 = self._load_disparity(sample_id['disparity_2_dir'], sample_id['id'])
        
        # Apply padding
        left_img = self._pad_image(left_img)
        right_img = self._pad_image(right_img)
        disparity = self._pad_disparity(disparity)
        disparity_2 = self._pad_disparity(disparity_2)

        # Convert to tensors
        right_img = torch.from_numpy(right_img).permute(2, 0, 1)
        left_img = torch.from_numpy(left_img).permute(2, 0, 1)
        disparity = torch.from_numpy(disparity)[None, ...]
        disparity_2 = torch.from_numpy(disparity_2)[None, ...]
        
        # Create mirrored stereo pair (for data augmentation)
        left_img_m = self.flip(right_img)
        right_img_m = self.flip(left_img)
        
        # Process depth maps
        depthmap, unnorm_depthmap = self._process_disparity(disparity, negate=True)
        depthmap_2, unnorm_depthmap_2 = self._process_disparity(disparity_2, negate=False)
        depthmap_2 = self.flip(depthmap_2)
        unnorm_depthmap_2 = self.flip(unnorm_depthmap_2)

        # Center crop to target size
        crop_size = (320, 736)
        tensors_to_crop = [
            right_img, left_img, right_img_m, left_img_m,
            depthmap, depthmap_2, disparity, disparity_2,
            unnorm_depthmap, unnorm_depthmap_2
        ]
        (right_img, left_img, right_img_m, left_img_m,
         depthmap, depthmap_2, disparity, disparity_2,
         unnorm_depthmap, unnorm_depthmap_2) = [
            self.centercrop(t, size=crop_size) for t in tensors_to_crop
        ]

        # Store original resolution copies
        original_left = left_img.squeeze(0)
        original_right = right_img.squeeze(0)
        original_left_m = left_img_m.squeeze(0)
        original_right_m = right_img_m.squeeze(0)
        original_depth = depthmap.squeeze(0)
        original_depth2 = depthmap_2.squeeze(0)

        # Resize for network input (scale=1 means no resizing)
        scale = 1
        h, w = crop_size
        new_size = (int(h / scale), int(w / scale))
        
        left_img = F.interpolate(left_img, size=new_size, mode='bilinear', align_corners=True)
        right_img = F.interpolate(right_img, size=new_size, mode='bilinear', align_corners=True)
        left_img_m = F.interpolate(left_img_m, size=new_size, mode='bilinear', align_corners=True)
        right_img_m = F.interpolate(right_img_m, size=new_size, mode='bilinear', align_corners=True)
        depthmap = F.interpolate(depthmap, size=new_size, mode='bilinear', align_corners=True)
        depthmap_2 = F.interpolate(depthmap_2, size=new_size, mode='bilinear', align_corners=True)
        unnorm_depthmap = F.interpolate(unnorm_depthmap, size=new_size, mode='bilinear', align_corners=True)
        unnorm_depthmap_2 = F.interpolate(unnorm_depthmap_2, size=new_size, mode='bilinear', align_corners=True)

        # Remove batch dimension added by Kornia
        right_img = right_img.squeeze(0)
        left_img = left_img.squeeze(0)
        right_img_m = right_img_m.squeeze(0)
        left_img_m = left_img_m.squeeze(0)
        depthmap = depthmap.squeeze(0)
        depthmap_2 = depthmap_2.squeeze(0)
        disparity = disparity.squeeze(0)
        disparity_2 = disparity_2.squeeze(0)
        unnorm_depthmap = unnorm_depthmap.squeeze(0)
        unnorm_depthmap_2 = unnorm_depthmap_2.squeeze(0)

        # Handle single-plane mode (for controlled testing)
        if self.singleplane:
            depthmap, depthmap_2, original_depth, original_depth2 = self._create_singleplane_depth(
                depthmap, depthmap_2, original_depth, original_depth2, idx
            )

        return {
            'id': sample_id['id'],
            'left_image': left_img,
            'right_image': right_img,
            'left_image_m': left_img_m,
            'right_image_m': right_img_m,
            'original_left': original_left,
            'original_right': original_right,
            'original_left_m': original_left_m,
            'original_right_m': original_right_m,
            'original_depth': original_depth,
            'original_depth2': original_depth2,
            'depthmap': depthmap,
            'depthmap_2': depthmap_2,
            'disparity': disparity,
            'disparity_2': disparity_2,
            'unnorm_depthmap': unnorm_depthmap,
            'unnorm_depthmap_2': unnorm_depthmap_2,
            'depth_conf': torch.ones_like(depthmap),
        }

    def _load_image(self, directory: str, sample_id: str) -> np.ndarray:
        """Load and normalize image to [0, 1]."""
        path = os.path.join(directory, f'{sample_id}.png')
        img = imageio.imread(path).astype(np.float32) / 255.0
        return img

    def _load_disparity(self, directory: str, sample_id: str) -> np.ndarray:
        """Load disparity map from PFM file."""
        path = os.path.join(directory, f'{sample_id}.pfm')
        disp = imageio.imread(path).astype(np.float32)
        return np.flip(disp, axis=0).copy()

    def _pad_image(self, img: np.ndarray) -> np.ndarray:
        """Apply reflection padding to image."""
        if self.padding > 0:
            return np.pad(img, ((self.padding, self.padding), 
                               (self.padding, self.padding), (0, 0)), mode='reflect')
        return img

    def _pad_disparity(self, disp: np.ndarray) -> np.ndarray:
        """Apply reflection padding to disparity map."""
        if self.padding > 0:
            return np.pad(disp, ((self.padding, self.padding), 
                                (self.padding, self.padding)), mode='reflect')
        return disp

    def _process_disparity(self, disparity: torch.Tensor, negate: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process disparity to create normalized depth map.
        
        Args:
            disparity: Raw disparity tensor
            negate: Whether to negate disparity (for left vs right)
        
        Returns:
            Tuple of (normalized_depthmap, unnormalized_depthmap)
        """
        if negate:
            depthmap = disparity * (-1.0)
        else:
            depthmap = disparity * 1.0
        
        unnorm_depthmap = torch.clamp(depthmap if negate else depthmap, min=0) / 255
        
        # Normalize to [0, 1]
        depthmap = depthmap - depthmap.min()
        depthmap = depthmap / depthmap.max()
        
        return depthmap, unnorm_depthmap

    def _create_singleplane_depth(
        self, 
        depthmap: torch.Tensor, 
        depthmap_2: torch.Tensor,
        original_depth: torch.Tensor,
        original_depth2: torch.Tensor,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create uniform depth planes for controlled testing."""
        if self.is_training:
            depth_val = torch.rand((1,))
            depthmap = depth_val * torch.ones_like(depthmap)
            depthmap_2 = depth_val * torch.ones_like(depthmap_2)
            original_depth = depth_val * torch.ones_like(original_depth)
            original_depth2 = depth_val * torch.ones_like(original_depth2)
        else:
            depth_val = torch.linspace(0., 1., steps=self.n_depths)[idx % self.n_depths]
            depthmap = depth_val * torch.ones_like(depthmap)
            depthmap_2 = depth_val * torch.ones_like(depthmap_2)
            original_depth = depth_val * torch.ones_like(original_depth)
            original_depth2 = depth_val * torch.ones_like(original_depth2)
        
        return depthmap, depthmap_2, original_depth, original_depth2