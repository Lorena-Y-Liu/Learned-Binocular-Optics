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
from torchvision import transforms

# 'left' image has negative disparity.
#DATA_ROOT = os.path.join('sampledata', 'training_data', 'SceneFlow')



#DATA_ROOT = 'C:/Users/Admin/Desktop/Dataset'
DATA_ROOT = '/mnt/ssd1/datasets/Instereo2K'

class InStereo2k(Dataset):

    def __init__(self, dataset: str, image_size: Tuple[int, int], is_training: bool = True, randcrop: bool = False,
                 augment: bool = False, padding: int = 0, singleplane: bool = False, n_depths: int = 16, hparams = None):
        super().__init__()

        self.dirs = []
        DATA_ROOT = '/mnt/ssd1/datasets/Instereo2K'
            
        if dataset == 'train':
            self.dir = os.path.join(DATA_ROOT, 'train')
            
            for i in range(1,7):
                path = os.path.join(self.dir,f'part{i}')
                files = os.listdir(path)
                files.sort()
                for file in files:
                    file_path = f'/part{i}/{file}'
                    self.dirs.append(file_path)

        
        elif dataset == 'val':
            self.dir = os.path.join(DATA_ROOT, 'test')
            files = os.listdir(self.dir)
            files.sort()   
            for file in files:
                file_path = f'/{file}'
                self.dirs.append(file_path)


        self.transform = RandomTransform(image_size, randcrop, augment)
        self.centercrop = CenterCrop(image_size)
        self.flip=transforms.RandomHorizontalFlip(p=1)

        self.is_training = torch.tensor(is_training)
        self.padding = padding
        self.singleplane = torch.tensor(singleplane)
        self.n_depths = n_depths
        self.image_size = image_size
        self.maxdisp = 192#hparams.maxdisp



    def stretch_depth(self, depth, depth_range, min_depth):
        return depth_range * depth + min_depth

    def __len__(self):
        return len(self.dirs)

    def norm(self, x):
        return (x-x.min())/(x.max()-x.min())
   
    def __getitem__(self, idx):

        path = self.dirs[idx]
        id = os.path.basename(path)
        right_image_dir = self.dir + path + '/right.png'
        left_image_dir = self.dir + path + '/left.png'
        disparity_dir_filled = self.dir + path + '/left_disp_filled.png'
        disparity_2_dir_filled = self.dir + path + '/right_disp_filled.png'
        disparity_dir = self.dir + path + '/left_disp.png'
        disparity_2_dir = self.dir + path +  '/right_disp.png'

        
        disparity = imageio.imread(disparity_dir).astype(np.float32)
        disparity_2 = imageio.imread(disparity_2_dir).astype(np.float32)
        disparity_filled = imageio.imread(disparity_dir_filled).astype(np.float32)
        disparity_2_filled = imageio.imread(disparity_2_dir_filled).astype(np.float32)

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
        disparity_filled = np.pad(disparity_filled,
                           ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')
        disparity_2_filled = np.pad(disparity_2_filled,
                           ((self.padding, self.padding), (self.padding, self.padding)), mode='reflect')                   
        right_img = torch.from_numpy(right_img).permute(2, 0, 1)
        left_img = torch.from_numpy(left_img).permute(2, 0, 1)
        
        left_img_m=self.flip(right_img) #original right is flipped as mirror left
        right_img_m=self.flip(left_img)
        
        disparity = torch.from_numpy(disparity_filled)[None, ...]#/100
        #torchvision.utils.save_image(disparity,'dis.jpg',normalize=True)
        disparity_2 = torch.from_numpy(disparity_2_filled)[None, ...]#/100
        # A far object is 0.
        #depthmap=torch.clamp(depthmap, 0, 255)
        unnorm_depthmap = torch.clamp(disparity, min=0)/255
        depthmap=self.norm(disparity)
        
        #depthmap_2 = torch.clamp(depthmap_2, 0, 255)
        unnorm_depthmap_2 = self.flip(torch.clamp(disparity_2*(1.0), min=0)/255)
        depthmap_2=self.flip(self.norm(disparity_2*(1.0)))

        #size=(192, 576)
        size=(160,448)

        right_img = self.centercrop(right_img,size=(size))
        left_img = self.centercrop(left_img,size=(size))
        right_img_m = self.centercrop(right_img_m,size=(size))
        left_img_m = self.centercrop(left_img_m,size=(size))

        depthmap = self.centercrop(depthmap,size=(size))
        depthmap_2 = self.centercrop(depthmap_2,size=(size))
        disparity = self.centercrop(disparity,size=(size))*(-1)
        disparity_2 = self.centercrop(disparity_2,size=(size))

        unnorm_depthmap = self.centercrop(unnorm_depthmap,size=(size))
        unnorm_depthmap_2 = self.centercrop(unnorm_depthmap_2,size=(size))
        # SceneFlow's depthmap has some aliasing artifact.
        #depthmap = gaussian_blur2d(depthmap, sigma=(0.8, 0.8), kernel_size=(5, 5))
        #depthmap_2 = gaussian_blur2d(depthmap_2, sigma=(0.8, 0.8), kernel_size=(5, 5))

        original_left=left_img
        original_right=right_img
        original_left_m=left_img_m
        original_right_m=right_img_m
        original_depth=depthmap
        original_depth2=depthmap_2

        c,d=size
        scale=1
        #scale=1.5
        left_img =F.interpolate(left_img, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        right_img =F.interpolate(right_img, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        left_img_m =F.interpolate(left_img_m, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        right_img_m =F.interpolate(right_img_m, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        
        depthmap =F.interpolate(depthmap, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        depthmap_2 =F.interpolate(depthmap_2, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        
        #disparity =F.interpolate(disparity, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)/scale
        #disparity_2 =F.interpolate(disparity_2, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)/scale

        unnorm_depthmap =F.interpolate(unnorm_depthmap, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
        unnorm_depthmap_2 =F.interpolate(unnorm_depthmap_2, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)

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
        disparity = disparity.squeeze(0)
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


        sample = {'id': id, 'right_image': right_img,'left_image': left_img, 'right_image_m': right_img_m,'left_image_m': left_img_m, 
                  'original_left': original_left, 'original_right': original_right, 'original_left_m': original_left_m, 'original_right_m': original_right_m, 
                  'original_depth': original_depth,'original_depth2': original_depth2,
                  'depthmap': depthmap,'depthmap_2': depthmap_2, 'disparity': disparity, 'disparity_2': disparity_2, 
                  'unnorm_depthmap': unnorm_depthmap, 'unnorm_depthmap_2': unnorm_depthmap_2 , 'depth_conf': torch.ones_like(depthmap)}

        return sample