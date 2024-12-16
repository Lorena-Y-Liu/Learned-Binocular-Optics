
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

def psf_experiment(image,stereo):
    gaussian_psf = False
    if gaussian_psf:
        dir = 'psf/GaussianPrediction'
        psf = torch.load(f'{dir}/{stereo}_psf_voluem.pth')
        size = image
        pad_height = (size[-2] - psf.shape[-2]) // 2
        pad_width = (size[-1] - psf.shape[-1]) // 2
        padded_psfs = F.pad(psf, (pad_width, pad_width, pad_height, pad_height), mode='constant', value=0)
        padded_psfs = padded_psfs.to(device)
        padded_psfs= fftshift(padded_psfs, dims=(-1, -2))
        return padded_psfs
    
    device = torch.device('cpu') #image.device
    h=image[0]
    w=image[1]
    debayer=Debayer3x3()
    PSF=[]
    BG=[]

    depth = [0.670,0.791, 0.965, 1.236, 1.722, 2.833, 5.000]
    # depth=[1.0, 1.056, 1.12, 1.19, 1.27, 1.36,1.47, 1.6, 1.74, 1.92, 2.14, 2.42, 2.78, 3.26, 3.94, 5.0]
    psf_file_name = "ring_psf"
    for n in range(7):
        d=depth[n]
        
        if stereo == 'left':
            path="C:/Users/liangxun/Desktop/deep_stereo/psf/"+psf_file_name+"/bg.tif"
            bg=skimage.io.imread(path).astype(np.float32)
            path="C:/Users/liangxun/Desktop/deep_stereo/psf/"+psf_file_name+"/"+str(d)+"m_1"+".tif"
            psf_d=skimage.io.imread(path).astype(np.float32)
        elif stereo == 'right':
            path="C:/Users/liangxun/Desktop/deep_stereo/psf/"+psf_file_name+"/bg.tif"#+"/"+str(d)+"m_2"+".tif"
            bg=skimage.io.imread(path).astype(np.float32)
            path="C:/Users/liangxun/Desktop/deep_stereo/psf/"+psf_file_name+"/"+str(d)+"m_2"+".tif"
            psf_d=skimage.io.imread(path).astype(np.float32)
            
        psf_clean=np.maximum(psf_d-bg,0)
        
        psf_clean=torch.from_numpy(psf_clean)
        a,b=torch.where(psf_clean==psf_clean.max())

        if len(a)==1:
                center=(a,b)
        if len(a)!=1:
                m=len(a)//2
                n=len(b)//2
                center=(a[m],b[n])

        delta_y=psf_clean.shape[-2]-2*center[0]
        delta_x=psf_clean.shape[-1]-2*center[1]

        '''trans_y=torch.abs(psf_clean.shape[-2]-2*center[0])//2
        trans_x=torch.abs(psf_clean.shape[-1]-2*center[1])//2
        if delta_x>0 and delta_y>0:
                psf_clean= F.pad(psf_clean, (int(trans_x), 0, int(trans_y), 0), mode='constant', value=0)[:-int(trans_y),:-int(trans_x)]
        if delta_x>0 and delta_y<0:
                psf_clean= F.pad(psf_clean, (int(trans_x), 0, 0, int(trans_y)), mode='constant', value=0)[int(trans_y):,:-int(trans_x)]
        if delta_x<0 and delta_y<0:
                psf_clean= F.pad(psf_clean, (0, int(trans_x), 0, int(trans_y)), mode='constant', value=0)[int(trans_y):,int(trans_x):]
        if delta_x<0 and delta_y>0:
                psf_clean= F.pad(psf_clean, (0, int(trans_x), int(trans_y), 0), mode='constant', value=0)[:-int(trans_y), int(trans_x):]'''

        
        psf_clean=debayer(psf_clean.unsqueeze(0).unsqueeze(0))
        #psf_clean =F.interpolate(psf_clean, size=(int(h), int(1.6*int(h))), mode='bilinear', align_corners=True)
        # psf_clean =F.interpolate(psf_clean, size=(int(h), int(w)), mode='bilinear', align_corners=True)

        #psf_clean= F.pad(psf_clean, ((int(w)-int(1.6*int(h)))//2, (int(w)-int(1.6*int(h)))//2, 0, 0), mode='constant', value=0)
        psf_r = normalize_psf(psf_clean[:, 0, ...])
        psf_g = normalize_psf(psf_clean[:, 1, ...])
        psf_b = normalize_psf(psf_clean[:, 2, ...])
        ratio_r_to_g = psf_r.max() / psf_g.max()
        ratio_b_to_g = psf_b.max() / psf_g.max()

        psf_r = psf_r / ratio_r_to_g
        psf_b = psf_b / ratio_b_to_g


        psf = torch.stack([psf_r, psf_g, psf_b], dim=1)
        psf= psf[...,center[0]-h//2:center[0]+h//2,center[1]-w//2:center[1]+w//2]
        import torchvision
        torchvision.utils.save_image(psf*255, 'psf_'+str(d)+'.png') 
        PSF.append(psf_clean)  

    psf_linear=torch.stack(PSF, dim=2)
        
    psf_linear = psf_linear.to(device)
    psf_linear= fftshift(psf_linear, dims=(-1, -2))
    psf_r = normalize_psf(psf_linear[:, 0, ...])
    psf_g = normalize_psf(psf_linear[:, 1, ...])
    psf_b = normalize_psf(psf_linear[:, 2, ...])
    psf = torch.stack([psf_r, psf_g, psf_b], dim=1)
    psf = normalize_psf(psf_linear)

    return psf

psf_experiment([50,50],'left')