
import skimage.io
import torch
import PIL
import os

import numpy as np
from debayer import Debayer3x3

import matplotlib.pyplot as plt
import torch.nn.functional as F
def fftshift(x, dims):
    shifts = [(x.size(dim)) // 2 for dim in dims]
    x = torch.roll(x, shifts=shifts, dims=dims)
    return x


def crop_boundary(x, w):
    if w == 0:
        return x
    else:
        return x[..., w:-w, int(1.6*w):-int(1.6*w)]

def normalize_psf(psfimg):

        return psfimg / psfimg.sum(dim=(-2, -1), keepdims=True)

def psf_captured(device):
    device = torch.device(device) #image.device
    path_left="psf/psf_left.pth"
    path_right="psf/psf_right.pth"
    psf_left=torch.load(path_left).to(device)
    psf_right=torch.load(path_right).to(device)

    _,_,_,h,w=psf_left.shape
    psf_left= F.interpolate(psf_left.squeeze(0), scale_factor=0.8, mode='bilinear', align_corners=False).unsqueeze(0)
    psf_right= F.interpolate(psf_right.squeeze(0), scale_factor=0.8, mode='bilinear', align_corners=False).unsqueeze(0)
    _,_,_,hh,ww=psf_left.shape
    h,w=320, 736
    psf_left = F.pad(psf_left, ((w-ww)//2, (w-ww)//2,(h-hh)//2 ,(h-hh)//2 ), mode='constant', value=0)
    psf_right = F.pad(psf_right, ((w-ww)//2, (w-ww)//2,(h-hh)//2 ,(h-hh)//2 ), mode='constant', value=0)

    
    #psf_left= normalize_psf(fftshift(psf_left, dims=(-1, -2)))
    #psf_right= normalize_psf(fftshift(psf_right, dims=(-1, -2)))
    #psf_left= normalize_psf(psf_left)
    #psf_right= normalize_psf(psf_right)
    psf_left_b=psf_left
    psf_right_b=psf_right
    
    psf_left_b[:, 0, ...] = normalize_psf(psf_left[:, 0, ...])/1.2
    psf_left_b[:, 1, ...] = normalize_psf(psf_left[:, 1, ...])/1.2
    psf_left_b[:, 2, ...] = normalize_psf(psf_left[:, 2, ...])/1.2
    psf_right_b[:, 0, ...] = normalize_psf(psf_right[:, 0, ...])/1.2
    psf_right_b[:, 1, ...] = normalize_psf(psf_right[:, 1, ...])/1.2
    psf_right_b[:, 2, ...] = normalize_psf(psf_right[:, 2, ...])/1.2
    
    '''ratio_r=psf_right[:, 0, ...].max()/psf_left[:, 0, ...].max()
    ratio_g=psf_right[:, 1, ...].max()/psf_left[:, 1, ...].max()
    ratio_b=psf_right[:, 2, ...].max()/psf_left[:, 2, ...].max()
    psf_left_b[:, 0, ...]=psf_left[:, 0, ...]*ratio_r
    psf_left_b[:, 1, ...]=psf_left[:, 1, ...]*ratio_r
    psf_left_b[:, 2, ...]=psf_left[:, 2, ...]*ratio_r'''

    #psf_left=torch.flip(psf_left,dims=[2])
    #psf_right=torch.flip(psf_right,dims=[2])
    psf_left_b=torch.flip(psf_left_b,dims=[2])
    psf_right_b=torch.flip(psf_right_b,dims=[2])

    return psf_left_b, psf_right_b
