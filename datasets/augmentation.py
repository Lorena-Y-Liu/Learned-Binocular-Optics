from typing import Tuple
import torch
import torch.nn as nn
import kornia.augmentation as K


class RandomTransform(nn.Module):
    def __init__(self, size: Tuple[int, int], augment: bool):
        super().__init__()
        self.crop = K.CenterCrop(size)
        self.flip = nn.Sequential(K.RandomVerticalFlip(p=0.5),
                                  K.RandomHorizontalFlip(p=0.5))
        self.augment = augment

    def forward(self, right_img, left_img, disparity,disparity_2, conf=None):
        if conf is None:
            input = torch.cat([right_img,left_img, disparity,disparity_2], dim=0)
            #input_l = torch.cat([left_img, disparity], dim=0)
        else:
            input = torch.cat([right_img, left_img, disparity, disparity_2, conf], dim=0)
            #input_l = torch.cat([left_img, disparity, conf], dim=0)
        input = self.crop(input)
        #input_l = self.crop(input_l)
        if self.augment:
            input = self.flip(input)
            #input_l = self.flip(input_l)
        right_img = input[:, :3]
        left_img = input[:, 3:6]
        disparity = input[:, [6]]
        disparity_2 = input[:, [7]]
        if conf is None:
            return right_img, left_img, disparity,disparity_2
        else:
            conf = input[:, [7]]
            return right_img, left_img, disparity, disparity_2, conf
