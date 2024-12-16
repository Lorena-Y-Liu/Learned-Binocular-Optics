from typing import Tuple
import os
import torch
import numpy as np
import imageio
from torch.utils.data import Dataset
from datasets.augmentation import RandomTransform
from kornia.augmentation import CenterCrop
from kornia.filters import gaussian_blur2d
import torchvision.utils
import cv2 
from torchvision import transforms
import torch.nn.functional as F

# 'left' image has negative disparity.
#DATA_ROOT = os.path.join('sampledata', 'training_data', 'SceneFlow')



#DATA_ROOT = 'C:/Users/Admin/Desktop/Dataset'
DATA_ROOT = '/mnt/ssd1/datasets/Instereo2K'

TRAIN_IMAGE_LEFT_PATHS = []
TRAIN_IMAGE_RIGHT_PATHS = []
TRAIN_DISPARITY_LEFT_PATHS = []
TRAIN_DISPARITY_RIGHT_PATHS = []
TRAIN_ids = []

for part in range(1, 7):  
    part_path = os.path.join(DATA_ROOT, f'train/part{part}')
    for id_folder in os.listdir(part_path):
        folder_path = os.path.join(part_path, id_folder)
        TRAIN_ids.append(id_folder)
        TRAIN_IMAGE_LEFT_PATHS.append(os.path.join(folder_path, 'left.png'))
        TRAIN_IMAGE_RIGHT_PATHS.append(os.path.join(folder_path, 'right.png'))
        TRAIN_DISPARITY_LEFT_PATHS.append(os.path.join(folder_path, 'left_disp_filled.png'))
        TRAIN_DISPARITY_RIGHT_PATHS.append(os.path.join(folder_path, 'right_disp_filled.png'))


VALIDATION_IMAGE_LEFT_PATHS = []
VALIDATION_IMAGE_RIGHT_PATHS = []
VALIDATION_DISPARITY_LEFT_PATHS = []
VALIDATION_DISPARITY_RIGHT_PATHS = []
VALIDATION_ids = []
part_path = os.path.join(DATA_ROOT, 'test') 
for id_folder in os.listdir(part_path):
    folder_path = os.path.join(part_path, id_folder)
    VALIDATION_ids.append(id_folder)
    VALIDATION_IMAGE_LEFT_PATHS.append(os.path.join(folder_path, 'left.png'))
    VALIDATION_IMAGE_RIGHT_PATHS.append(os.path.join(folder_path, 'right.png'))
    VALIDATION_DISPARITY_LEFT_PATHS.append(os.path.join(folder_path, 'left_disp_filled.png'))
    VALIDATION_DISPARITY_RIGHT_PATHS.append(os.path.join(folder_path, 'right_disp_filled.png'))


class InStereo2k(Dataset):

    def __init__(self, dataset: str, image_size: Tuple[int, int], is_training: bool = True, randcrop: bool = False,
                 augment: bool = False, padding: int = 0, singleplane: bool = False, n_depths: int = 16):
        super().__init__()
        self.flip=transforms.RandomHorizontalFlip(p=1)
        if dataset == 'train':
            left_image_dirs = TRAIN_IMAGE_LEFT_PATHS
            disparity_dirs = TRAIN_DISPARITY_LEFT_PATHS
            disparity_2_dirs = TRAIN_DISPARITY_RIGHT_PATHS
            right_image_dirs = TRAIN_IMAGE_RIGHT_PATHS
            ids = TRAIN_ids
        
        elif dataset == 'val':
            left_image_dirs = VALIDATION_IMAGE_LEFT_PATHS
            disparity_dirs = VALIDATION_DISPARITY_LEFT_PATHS
            disparity_2_dirs = VALIDATION_DISPARITY_RIGHT_PATHS
            right_image_dirs = VALIDATION_IMAGE_RIGHT_PATHS
            ids = VALIDATION_ids
        else:
            raise ValueError(f'dataset ({dataset}) has to be "train_left/right," "val_left/right," or "example."')

        self.transform = RandomTransform(image_size, randcrop, augment)
        self.centercrop = CenterCrop(image_size)

        self.sample_ids = []
        for right_image_dir, left_image_dir, disparity_dir, disparity_2_dir,id in zip(right_image_dirs, left_image_dirs, disparity_dirs, disparity_2_dirs,ids):
            if os.path.exists(disparity_dir):
                sample_id = {
                    'right_image_dir': right_image_dir,
                    'left_image_dir': left_image_dir,
                    'disparity_dir': disparity_dir,
                    'disparity_2_dir': disparity_2_dir,
                    'id': id,
                }
                self.sample_ids.append(sample_id)
            else:
                print(f'Disparity image does not exist!: {disparity_dir}')
        self.is_training = torch.tensor(is_training)
        self.padding = padding
        self.singleplane = torch.tensor(singleplane)
        self.n_depths = n_depths

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

        disparity = imageio.imread(disparity_dir).astype(np.float32)
        disparity_2 = imageio.imread(disparity_2_dir).astype(np.float32)
        left_img = imageio.imread(left_image_dir).astype(np.float32)
        left_img /= 255.  # Scale to [0, 1] 
        right_img = imageio.imread(os.path.join(right_image_dir)).astype(np.float32)
        right_img /= 255.  # Scale to [0, 1]
        left_img = np.pad(left_img,
                    ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        right_img = np.pad(right_img,
                     ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='reflect')
        disparity = np.pad(disparity,
                           ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')     
        disparity_2 = np.pad(disparity_2,
                           ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')
        
        right_img = torch.from_numpy(right_img).permute(2, 0, 1)
        left_img = torch.from_numpy(left_img).permute(2, 0, 1)
        
        left_img_m=self.flip(right_img) #original right is flipped as mirror left
        right_img_m=self.flip(left_img)
        
        disparity = torch.from_numpy(disparity)[None, ...]
        #torchvision.utils.save_image(disparity,'dis.jpg',normalize=True)
        disparity_2 = torch.from_numpy(disparity_2)[None, ...]

        # A far object is 0.
        Disparity=disparity*(-1)
        depthmap = disparity
        depthmap -= depthmap.min()
        depthmap /= depthmap.max()
        
        #torchvision.utils.save_image(depthmap,'dep.jpg',normalize=True)

        depthmap_2 = disparity_2
        depthmap_2 -= depthmap_2.min()
        depthmap_2 /= depthmap_2.max()

        # Flip the value. A near object is 0.
        depthmap = 1. - depthmap
        depthmap_2 = 1. - depthmap_2
        depthmap_2=self.flip(depthmap_2)
        #print(f"depth map size:{depthmap.shape}")
        #if self.is_training:
            #right_img, left_img,depthmap,depthmap_2 ,disparity = self.transform(right_img,left_img, depthmap, depthmap_2,disparity)
        #else:
        
        size=(256,256)
        right_img = self.centercrop(right_img,size=(size))
        left_img = self.centercrop(left_img,size=(size))
        depthmap = self.centercrop(depthmap,size=(size))
        depthmap_2 = self.centercrop(depthmap_2,size=(size))
        right_img_m = self.centercrop(right_img_m,size=(size))
        left_img_m = self.centercrop(left_img_m,size=(size))

        
        

        # SceneFlow's depthmap has some aliasing artifact.
        depthmap = gaussian_blur2d(depthmap, sigma=(0.8, 0.8), kernel_size=(5, 5))
        depthmap_2 = gaussian_blur2d(depthmap_2, sigma=(0.8, 0.8), kernel_size=(5, 5))

        original_left=left_img
        original_right=right_img
        original_left_m=left_img_m
        original_right_m=right_img_m
        original_depth=depthmap
        original_depth2=depthmap_2


        c,d=size
        scale=1.6

        left_img =F.interpolate(left_img, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        right_img =F.interpolate(right_img, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        left_img_m =F.interpolate(left_img_m, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        right_img_m =F.interpolate(right_img_m, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        
        depthmap =F.interpolate(depthmap, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        depthmap_2 =F.interpolate(depthmap_2, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        
        # Remove batch dim (Kornia adds batch dimension automatically.)
        original_left=original_left.squeeze(0)
        original_right=original_right.squeeze(0)
        original_left_m=original_left_m.squeeze(0)
        original_right_m=original_right_m.squeeze(0)
        original_depth=original_depth.squeeze(0)
        original_depth2=original_depth2.squeeze(0)

        right_img = right_img.squeeze(0)
        left_img = left_img.squeeze(0)
        right_img_m = right_img_m.squeeze(0)
        left_img_m = left_img_m.squeeze(0)
        depthmap = depthmap.squeeze(0)
        depthmap_2 = depthmap_2.squeeze(0)
        Disparity=Disparity.squeeze(0)
        
        
        
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

        sample = {'id': id, 'right_image': right_img,'left_image': left_img, 'right_image_m': right_img_m,'left_image_m': left_img_m, 
                  'original_left': original_left, 'original_right': original_right, 'original_left_m': original_left_m, 'original_right_m': original_right_m, 
                  'original_depth': original_depth,'original_depth2': original_depth2,
                  'depthmap': depthmap,'depthmap_2': depthmap_2, 'depth_conf': torch.ones_like(depthmap)}


        return sample
    


  