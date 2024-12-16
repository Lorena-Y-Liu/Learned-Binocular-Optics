# from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import numpy as np
# from apex import amp
from argparse import ArgumentParser
from debayer import Debayer2x2
from StereoCapture import *


import sys
sys.path.append('C:\\Users\\liangxun\\Desktop\\deep_stereo')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from util.utils import InputPadder, flip
import math

from util.helper import crop_boundary, gray_to_rgb, imresize, linear_to_srgb, srgb_to_linear, to_bayer
from util.loss import Vgg16PerceptualLoss
from util.fft import crop_psf, fftshift
from util.helper import crop_boundary, gray_to_rgb, imresize, linear_to_srgb, srgb_to_linear, to_bayer
import cv2
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    
    def rfft(x, d):
        t=torch.fft.fft2(x, dim = (-d,-1))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        t = torch.fft.ifft2(torch.complex(x[...,0], x[...,1]), dim = (-d,-1))
        return t.real


sys.path.append('C:\\Users\\liangxun\\Desktop\\deepstereo')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from deepstereo import Stereo3D
from solvers.image_reconstruction import apply_tikhonov_inverse
import torchvision
from util.warp import Warp




def srgb_to_linear(x, eps=1e-8):
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_uint8(x: torch.Tensor):
    """
    x: B x C x H x W
    """
    return (255 * x.squeeze(0).clamp(0, 1)).permute(1, 2, 0).to(torch.uint8)

    # #return (65535 * x.squeeze(0).clamp(0, 1)).permute(1, 2, 0).to(torch.int16)

    # x_numpy = (x.detach().cpu().numpy() * 65535).astype(np.uint16)
    # return x_numpy


def strech_img(x):
    return (x - x.min()) / (x.max() - x.min())


def find_minmax(img, saturation=0.1):
    min_val = np.percentile(img, saturation)
    max_val = np.percentile(img, 100 - saturation)
    return min_val, max_val
def srgb_to_linear(x, eps=1e-8):
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x, eps=1e-8):
    a = 0.055
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.0031308, 12.92 * x, (1. + a) * x ** (1. / 2.4) - a)

def rescale_image(x):
    min, max = find_minmax(x)
    return (x - min) / (max - min)


def average_inference(x):
    x = torch.stack([
        x[0],
        torch.flip(x[1], dims=(-1,)),
        torch.flip(x[2], dims=(-2,)),
        torch.flip(x[3], dims=(-2, -1)),
    ], dim=0)
    return x.mean(dim=0, keepdim=True)

def overlapping_resolution(depth: float, 
                           pixel_size: float = 5.86e-6, 
                           focal_length: float = 0.035, 
                           resolution_width: int = 1920, 
                           camera_distance: float = 0.075) -> int:
    
    sensor_width = 1920 * pixel_size
    fov = 2 * math.atan(sensor_width / (2 * focal_length))
    scene_width = 2 * math.tan(fov / 2) * depth
    overlap_width = scene_width - camera_distance
    overlap_resolution = int((overlap_width / scene_width) * resolution_width)

    return overlap_resolution

@torch.no_grad()
def load_model(args)->list:

    device = torch.device('cuda')
    ckpt = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
    hparams = ckpt['hyper_parameters']
    model = torch.nn.DataParallel(Stereo3D(hparams=hparams), device_ids=[0])
    #model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(device)
    model.eval()
    return model


def run_deepstereo(display_list,args,model):
    device = torch.device('cuda')
    left_img_np = display_list[0].astype(np.float32)
    right_img_np = display_list[1].astype(np.float32)

    left = torch.from_numpy(left_img_np).unsqueeze(0).unsqueeze(0) #torch.Size([1, 1, 1200, 1920])
    right = torch.from_numpy(right_img_np).unsqueeze(0).unsqueeze(0)

    debayer = Debayer2x2()
    left_linear = debayer(left).squeeze() #torch.Size([3, 1200, 1920])
    right_linear = debayer(right).squeeze()

  
    left_linear = left_linear.unsqueeze(0)#[...,520:680,736:1184]#torch.Size([1,3, 1200, 1920])
    right_linear = right_linear.unsqueeze(0)#[...,520:680,736:1184]

    #left_linear = F.interpolate(left_linear, size=args.image_size , mode='bilinear', align_corners=False)
    #right_linear = F.interpolate(right_linear, size=args.image_size, mode='bilinear', align_corners=False)

    left_linear = left_linear.to(device)*255
    right_linear = right_linear.to(device)*255

    with torch.no_grad():
        #input_l = left_linear / 65536.0
        #input_r = right_linear / 65536.0 
        #input_l = linear_to_srgb(input_l)
        #input_r = linear_to_srgb(input_r)
        #left_linear = input_l#*255.0
        #right_linear = input_r#*255.0
        '''print(left_linear[:,0,...].max(),left_linear[:,0,...].min())
        print(left_linear[:,1,...].max(),left_linear[:,1,...].min())
        print(left_linear[:,2,...].max(),left_linear[:,2,...].min())'''
        for i in range(3):
            left_linear[:,i,...]=(left_linear[:,i,...]-left_linear[:,i,...].min())/(left_linear[:,i,...].max()-left_linear[:,i,...].min())
            right_linear[:,i,...]=(right_linear[:,i,...]-right_linear[:,i,...].min())/(right_linear[:,i,...].max()-right_linear[:,i,...].min())

        padder = InputPadder(left_linear.shape, divis_by=32)
        left_linear, right_linear = padder.pad(left_linear, right_linear)
        #left_linear, right_linear= padder.pad(input_l, input_r)
        image_sz=left_linear.shape[-2:]
        psf_left_0 = model.camera_left.normalize_psf(model.camera_left.psf_at_camera(size=image_sz,modulate_phase=False).unsqueeze(0))#.cpu()

        #psf_left = model.camera_left.normalize_psf(psf_experiment(image_sz)).to(device)
        psf_left = crop_psf(psf_left_0, image_sz)



        pinv_volumes_left = apply_tikhonov_inverse(left_linear, psf_left, model.hparams.reg_tikhonov,
                                            apply_edgetaper=True)
        
        psf_right_0 = model.camera_right.normalize_psf(model.camera_right.psf_at_camera(size=image_sz,modulate_phase=False).unsqueeze(0))#.cpu()
        psf_right = crop_psf(psf_right_0, image_sz)
        pinv_volumes_right = apply_tikhonov_inverse(right_linear, psf_right, model.hparams.reg_tikhonov,
                                            apply_edgetaper=True)
        left_linear_m, right_linear_m=flip(right_linear), flip(left_linear)
        pinv_volumes_left_m, pinv_volumes_right_m= flip(pinv_volumes_right), flip(pinv_volumes_left)
        _, est_sq= model.matching(model.hparams,linear_to_srgb(left_linear), linear_to_srgb(right_linear),iters=model.hparams.valid_iters)
        _, est_m_sq= model.matching(model.hparams, linear_to_srgb(left_linear_m), linear_to_srgb(right_linear_m), iters=model.hparams.valid_iters)
        est=est_sq[-1]
        est_m=est_m_sq[-1]


        warping = Warp()
        b, c, h, w = est.shape
        w_disparity=est
        w_disparity_2=flip(est_m)
        warped_right, mask=warping.warp_disp(right_linear, w_disparity, w_disparity_2)
        warped_right+=left_linear*(1-mask)
        warped_right_m, mask_m=warping.warp_disp(right_linear_m, flip(w_disparity_2), flip(w_disparity))
        warped_right_m+=left_linear_m*(1-mask_m)

        norm_max, norm_min=est.max()/255,est.min()/255
        norm_max_m, norm_min_m = est_m.max()/255, est_m.min()/255
        input_rough=(est/255-norm_min.reshape(-1,1,1,1))/(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))
        input_rough_m=(est_m/255-norm_min_m.reshape(-1,1,1,1))/(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))

        # Image recovery, detailed depth estimation and super resolution


        model_outputs = model.decoder(captimgs_left=left_linear.float(),captimgs_right=warped_right.float(),
                                    pinv_volumes_left=pinv_volumes_left.float(), rough_depth=input_rough.float(),hparams=model.hparams)
        model_outputs_m = model.decoder(captimgs_left=left_linear_m.float(),captimgs_right=warped_right_m.float(),
                                    pinv_volumes_left=pinv_volumes_left_m.float(), rough_depth=input_rough_m.float(), hparams=model.hparams)

        est_images_left = crop_boundary(model_outputs[0], model.crop_width)
        #est_images_left = average_inference(est_images_left)
        est_images_left_m = crop_boundary(model_outputs_m[0], model.crop_width)
        #est_images_left_m = average_inference(est_images_left_m)

        captimgs_left = crop_boundary(left_linear, model.crop_width)
        captimgs_right = crop_boundary(right_linear, model.crop_width)

        est_depthmaps = model_outputs[2]
        #est_dfd=est_dfd*(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))+norm_min.reshape(-1,1,1,1)
        #est_depthmaps=est_depthmaps*(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))+norm_min.reshape(-1,1,1,1)

        est_depthmaps_m = model_outputs_m[2]
        #est_dfd_m=est_dfd_m*(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))+norm_min_m.reshape(-1,1,1,1)
        #est_depthmaps_m=est_depthmaps_m*(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))+norm_min_m.reshape(-1,1,1,1)
        est_depthmaps=est_depthmaps*(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))+norm_min.reshape(-1,1,1,1)
        est_depthmaps_m=est_depthmaps_m*(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))+norm_min_m.reshape(-1,1,1,1)
        
        est_depthmaps=est_depthmaps#*255
        est_depthmaps_m=est_depthmaps_m#*255
        #est_depthmaps = crop_boundary(est_depthmaps, model.crop_width)
        #est_depthmaps = average_inference(est_depthmaps)
        #est_depthmaps_m = crop_boundary(est_depthmaps_m, model.crop_width)
        #est_depthmaps_m = average_inference(est_depthmaps_m)
        
        torchvision.utils.save_image(est_images_left[[0],...].cpu(), 'sampledata\\result\\left_recovered.png')
        depthmaps=gray_to_rgb(1-est_depthmaps).squeeze(0)#*255
        #torchvision.utils.save_image((est[[0],:,200:1000,320:1600].cpu()-est[[0],:,200:1000,320:1600].cpu().min())/(est[[0],:,200:1000,320:1600].cpu().max()-est[[0],:,200:1000,320:1600].cpu().min()), 'sampledata\\result\\est_depth.png')
        torchvision.utils.save_image((est.cpu()-est.cpu().min())/(est.cpu().max()-est.cpu().min()), 'sampledata\\result\\est_depth.png')
        
        est_depthmaps=est_depthmaps[:,:,40:600,100:900]
        est_depthmaps=(est_depthmaps-est_depthmaps.min())/(est_depthmaps.max()-est_depthmaps.min())
        est_depthmaps_m=est_depthmaps_m[:,:,40:600,100:900]
        est_depthmaps_m=(est_depthmaps_m-est_depthmaps_m.min())/(est_depthmaps_m.max()-est_depthmaps_m.min())
        
        for i in range(3):
            est_images_left[:,i,...]=(est_images_left[:,i,...]-est_images_left[:,i,...].min())/(est_images_left[:,i,...].max()-est_images_left[:,i,...].min())
            est_images_left_m[:,i,...]=(est_images_left_m[:,i,...]-est_images_left_m[:,i,...].min())/(est_images_left_m[:,i,...].max()-est_images_left_m[:,i,...].min())

    return captimgs_left,captimgs_left, est_images_left_m*255, est_images_left*255,est_depthmaps, est_depthmaps_m

def main(args):

    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()
    
    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()


    # cam_list.Clear()
    # system.ReleaseInstance()

    # system = PySpin.System.GetInstance()
    # cam_list = system.GetCameras()
    # version = system.GetLibraryVersion()
    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Run example on all cameras
    print('Running example for all cameras...')

    #result = run_multiple_cameras(cam_list)
    root = Tk()
    root.title("Depth Input")
    setup_input_window(root, cam_list,args)
    root.mainloop()


    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')
    return result

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--savepath', default='C:\\Users\\liangxun\\Desktop\\Capture\\Ring_test\\result')

    parser.add_argument('--interpolate', action='store_false')
    parser.add_argument('--image_size', default=(640,1024))

    parser.add_argument('--captimg_path_left', type=str, default='C:\\Users\\liangxun\\Desktop\\Capture\\Ring_test\\left_1.tif')
    parser.add_argument('--captimg_path_right', type=str, default='C:\\Users\\liangxun\\Desktop\\Capture\\Ring_test\\right_1.tif')           
    #parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./sceneflow.pth')


    parser.add_argument('--ckpt_path', type=str, default='C:/Users/liangxun/Desktop/deep_stereo/sampledata/logs/log/version_108/checkpoints/interrupted_model.ckpt')# C:/Users/Admin/Desktop/liuyuhui/workspace/hybrid/sampledata//logs/log/version_52/checkpoints/epoch=18-val_loss=6.2492.ckpt') # epoch=22-val_loss=10.2003.ckpt')
    #parser.add_argument('--ckpt_path', type=str, default='c:/Users/liangxun/Desktop/deep_stereo/sampledata/logs/log/version_146/checkpoints/epoch=0-val_loss=4.5036.ckpt')
    #parser.add_argument('--ckpt_path', type=str, default='C:\\Users\\Admin\\Desktop\\liuyuhui\\workspace\\jupyter_hybrid_rgbd\\sampledata\\logs\\log\\version_78\\checkpoints\\epoch=11-val_loss=125.9943.ckpt')

    parser = Stereo3D.add_model_specific_args(parser)
    #args = parser.parse_args()
    args = parser.parse_known_args()[0]
    main(args)

    args = parser.parse_args()
    main(args)
