# from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
#from tensorboardX import SummaryWriter
# from datasets import __datasets__
# from models import __models__, model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test
# from utils import *
import skimage.io
import gc
# from apex import amp
import cv2
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from debayer import Debayer2x2
#from calibration import stereo_rectify, white_balance
from StereoCapture import *
from util.fft import crop_psf, fftshift
from util.helper import crop_boundary, gray_to_rgb, imresize, linear_to_srgb, srgb_to_linear, to_bayer
from util.loss import Vgg16PerceptualLoss
import cv2

from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder

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

import torch.nn.parallel
import numpy as np
import sys
sys.path.append('C:\\Users\\liangxun\\Desktop\\deep_stereo')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from deepstereo import Stereo3D
#from core.utils.utils import InputPadder
import math
from solvers.image_reconstruction import apply_tikhonov_inverse
import torchvision
from util.warp import Warp

def flip(x):
    func=torchvision.transforms.RandomHorizontalFlip(p=1)
    return func(x)


def to_uint8(x: torch.Tensor):
    """
    x: B x C x H x W
    """
    return (255 * x.clamp(0, 1)).permute(1, 2, 0).to(torch.uint8)

    # #return (65535 * x.squeeze(0).clamp(0, 1)).permute(1, 2, 0).to(torch.int16)

    # x_numpy = (x.detach().cpu().numpy() * 65535).astype(np.uint16)
    # return x_numpy


def strech_img(x):
    return (x - x.min()) / (x.max() - x.min())


def find_minmax(img, saturation=0.1):
    min_val = np.percentile(img.cpu(), saturation)
    max_val = np.percentile(img.cpu(), 100 - saturation)
    return min_val, max_val


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
                           camera_distance: float = 0.07) -> int:
    
    sensor_width = 1920 * pixel_size
    fov = 2 * math.atan(sensor_width / (2 * focal_length))
    scene_width = 2 * math.tan(fov / 2) * depth
    overlap_width = scene_width - camera_distance
    overlap_resolution = int((overlap_width / scene_width) * resolution_width)

    return overlap_resolution

@torch.no_grad()
def load_model(args):
    device = torch.device('cuda')
    ckpt = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
    hparams = ckpt['hyper_parameters']
    
    model = Stereo3D(hparams=hparams)

    #model.camera_left.heightmap1d_.data = ckpt['state_dict']['camera_left.heightmap1d_']
    #model.camera_right.heightmap1d_.data = ckpt['state_dict']['camera_right.heightmap1d_']
    decoder_dict = {key[8:]: value for key, value in ckpt['state_dict'].items() if 'decoder' in key}
    model.decoder.load_state_dict(decoder_dict)
    #matching_dict = {key[9:]: value for key, value in ckpt['state_dict'].items() if 'matching' in key}
    #model.matching.load_state_dict(matching_dict)
    matching = IGEVStereo(args)
    matching = torch.nn.DataParallel(matching, device_ids=[0])
    matching.load_state_dict(torch.load(args.matching_ckpt))
    matching = matching.module
    matching.to(device)
    matching.eval()
    #matching = matching.module
    model.eval()
    return model, matching



def run_deepstereo(display_list,args, model, matching):
    device = torch.device('cuda')

    left_img_np = display_list[0].astype(np.float32)
    right_img_np = display_list[1].astype(np.float32)

    left_linear = torch.from_numpy(left_img_np).unsqueeze(0)
    right_linear = torch.from_numpy(right_img_np).unsqueeze(0)
    #left_linear = left_linear[..., 120:1080,160:1760]
    #right_linear = right_linear[...,120:1080,160:1760]
    #left_linear = rescale_image(left_linear)
    #right_linear = rescale_image(right_linear)
    left_linear = left_linear / 65535.0
    right_linear = right_linear / 65535.0



    # Remove the offset value of the camera
    #left_linear -= 64
    #right_linear -= 64
    # add batch dim
    left_linear = left_linear.unsqueeze(0).cuda()
    right_linear = right_linear.unsqueeze(0).cuda()

    #left_linear = model.debayer(left_linear)
    #right_linear = model.debayer(right_linear)s

    debayer = Debayer2x2()
    debayer = debayer.to(device)
    left_linear = debayer(left_linear) #torch.Size([3, 1200, 1920])
    right_linear = debayer(right_linear)

    size=(int(1200),int(1920))
    left_linear=F.interpolate(left_linear, size, mode='bilinear', align_corners=False)
    right_linear=F.interpolate(right_linear, size, mode='bilinear', align_corners=False)
    #left_linear = left_linear[:, :, :384, :608]
    #right_linear = right_linear[:, :, :384, :608]
    
    #left_linear = left_linear.squeeze(1)  # Removes the second dimension of size 1
    #left_linear = left_linear.permute( 0, 1, 2, 3)
    #right_linear = right_linear.squeeze(1)  # Removes the second dimension of size 1
    #right_linear = right_linear.permute( 0, 1, 2, 3)
    # Debayer with the bilinear interpolation
    

    '''
    # Adjust white balance (The values are estimated from a white paper and manually tuned.)
    if 'indoor1' in save_name:
        captimg_linear[:, 0] *= (40293.078 - 64) / (34013.722 - 64) * 1.03
        captimg_linear[:, 2] *= (40293.078 - 64) / (13823.391 - 64) * 0.97
    elif 'indoor2' in save_name:
        captimg_linear[:, 0] *= (38563. - 64) / (28537. - 64) * 0.94
        captimg_linear[:, 2] *= (38563. - 64) / (15134. - 64) * 1.13
    elif 'outdoor' in save_name:
        captimg_linear[:, 0] *= (61528.274 - 64) / (46357.955 - 64) * 0.9
        captimg_linear[:, 2] *= (61528.274 - 64) / (36019.744 - 64) * 1.4
    else:
        raise ValueError('white balance is not set.')
    '''

    # Inference-time augmentation

    left_linear = torch.cat([left_linear.float() ], dim=0)

    right_linear = torch.cat([right_linear.float()], dim=0)
    '''left_linear = torch.cat([
        left_linear.float(),
        torch.flip(left_linear, dims=(-1,)),
        torch.flip(left_linear, dims=(-2,)),
        torch.flip(left_linear, dims=(-1, -2)),
    ], dim=0)

    right_linear = torch.cat([
        right_linear.float(),
        torch.flip(right_linear, dims=(-1,)),
        torch.flip(right_linear, dims=(-2,)),
        torch.flip(right_linear, dims=(-1, -2)),
    ], dim=0)'''

    image_sz = left_linear.shape[-2:]
    left_linear = left_linear.to(device)
    right_linear = right_linear.to(device)
    model = model.to(device)
    #resize=(int(image_sz[0]/1.5),int(image_sz[1]/1.5))
    with torch.no_grad():
        psf_left = model.camera_left.normalize_psf(model.camera_left.psf_at_camera(size=image_sz,modulate_phase=False).unsqueeze(0))
        #psf_left = model.camera_left.normalize_psf(psf_experiment(image_sz)).to(device)
        #psf_cropped_left = crop_psf(psf_left, image_sz)

        pinv_volumes_left = apply_tikhonov_inverse(left_linear, psf_left, model.hparams.reg_tikhonov,
                                            apply_edgetaper=True)
        
        psf_right = model.camera_right.normalize_psf(model.camera_right.psf_at_camera(size=image_sz,modulate_phase=False).unsqueeze(0))
        #psf_cropped_right = crop_psf(psf_right, image_sz)
        pinv_volumes_right = apply_tikhonov_inverse(right_linear, psf_right, model.hparams.reg_tikhonov,
                                            apply_edgetaper=True)
        left_linear_m, right_linear_m=flip(right_linear), flip(left_linear)
        pinv_volumes_left_m, pinv_volumes_right_m= flip(pinv_volumes_right), flip(pinv_volumes_left)

        #est= model.matching(hparams=model.hparams, image1=linear_to_srgb(left_linear)*255, image2=linear_to_srgb(right_linear)*255,iters=22, test_mode=True)[-1].unsqueeze(0)
        #est_m= model.matching(hparams=model.hparams, image1=linear_to_srgb(left_linear_m)*255, image2=linear_to_srgb(right_linear_m)*255,iters=22, test_mode=True)[-1].unsqueeze(0)
        padder = InputPadder(left_linear.shape, divis_by=32)
        left_mat, right_mat = padder.pad(left_linear, right_linear)
        left_mat_m, right_mat_m = padder.pad(left_linear_m, right_linear_m)
        est= matching(image1=linear_to_srgb(left_mat)*255, image2=linear_to_srgb(right_mat)*255,iters=32, test_mode=True)[-1].unsqueeze(0)
        est_m= matching(image1=linear_to_srgb(left_mat_m)*255, image2=linear_to_srgb(right_mat_m)*255,iters=32, test_mode=True)[-1].unsqueeze(0)
        
        est=padder.unpad(est)
        est_m=padder.unpad(est_m)
        
        #rough=(est-est.min())/(est.max()-est.min())
        #rough_m=(est_m-est_m.min())/(est_m.max()-est_m.min())
        rough=est.clamp(0,args.max_disp)
        rough_m=est_m.clamp(0,args.max_disp)
        rough=est/args.max_disp
        rough_m=est_m/args.max_disp

        
        warping = Warp()
        w_disparity=est
        w_disparity_2=flip(est_m)
        warped_right, mask=warping.warp_disp(right_linear, w_disparity, w_disparity_2)
        warped_right+=left_linear*(1-mask)
        warped_right_m, mask_m=warping.warp_disp(right_linear_m, flip(w_disparity_2), flip(w_disparity))
        warped_right_m+=left_linear_m*(1-mask_m)


        '''model_outputs = model.decoder(hparams=model.hparams, captimgs_left=left_linear.float(),captimgs_right=right_linear.float(),
                                    pinv_volumes_left=pinv_volumes_left.float(), rough_depth=rough.float())
        model_outputs_m = model.decoder(hparams=model.hparams, captimgs_left=left_linear_m.float(),captimgs_right=right_linear_m.float(),
                                    pinv_volumes_left=pinv_volumes_left_m.float(), rough_depth=rough_m.float())'''
        rough_norm=(rough-rough.min())/(rough.max()-rough.min())
        rough_m_norm=(rough_m-rough_m.min())/(rough_m.max()-rough_m.min())
        model_outputs = model.decoder(hparams=model.hparams, captimgs_left=left_linear.float(),captimgs_right=warped_right.float(),
                                    pinv_volumes_left=pinv_volumes_left.float(), rough_depth=rough_norm.float())
        model_outputs_m = model.decoder(hparams=model.hparams, captimgs_left=left_linear_m.float(),captimgs_right=warped_right_m.float(),
                                    pinv_volumes_left=pinv_volumes_left_m.float(), rough_depth=rough_m_norm.float())


        est_images_left = crop_boundary(model_outputs[0], model.crop_width)
        #est_images_left = average_inference(est_images_left)
        est_images_left_m = crop_boundary(model_outputs_m[0], model.crop_width)
        #est_images_left_m = average_inference(est_images_left_m)

        captimgs_left = crop_boundary(left_linear[[0]], model.crop_width)
        captimgs_right = crop_boundary(right_linear[[0]], model.crop_width)

        #est_depthmaps_1 = crop_boundary(model_outputs[-1], model.crop_width)
        #est_depthmaps_1 = average_inference(est_depthmaps_1)
        est_images_left = srgb_to_linear(est_images_left)
        est_images_left_m = srgb_to_linear(est_images_left_m)

        
        
        est_depthmaps=model_outputs[-1]
        est_depthmaps_m=model_outputs_m[-1]
        est_depthmaps = crop_boundary(est_depthmaps, model.crop_width)
        #est_depthmaps = average_inference(est_depthmaps)
        est_depthmaps_m = crop_boundary(est_depthmaps_m, model.crop_width)
        #est_depthmaps_m = average_inference(est_depthmaps_m)
        
        

    return captimgs_left,captimgs_right,est_images_left,flip(est_images_left_m),rough.squeeze(0),flip(rough_m.squeeze(0))


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
    parser.add_argument('--savepath', default='C:/Users/liangxun/Desktop/Result/comp/dlr1')

    parser.add_argument('--interpolate', action='store_false')
    parser.add_argument('--image_size', default=(1200,1920))

    #parser.add_argument('--captimg_path_left', type=str, default='C:\\Users\\liangxun\\Desktop\\Capture\\Ring_test\\left_1.tif')
    #parser.add_argument('--captimg_path_right', type=str, default='C:\\Users\\liangxun\\Desktop\\Capture\\Ring_test\\right_1.tif')           
    #parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./sceneflow.pth')


    #parser.add_argument('--ckpt_path', type=str, default='c:/Users/liangxun/Desktop/deep_stereo/sampledata/logs/log/version_108/checkpoints/interrupted_model.ckpt')# C:/Users/Admin/Desktop/liuyuhui/workspace/hybrid/sampledata//logs/log/version_52/checkpoints/epoch=18-val_loss=6.2492.ckpt') # epoch=22-val_loss=10.2003.ckpt')
    parser.add_argument('--ckpt_path', type=str, default='c:/Users/liangxun/Desktop/deep_stereo/sampledata/logs/log/version_108/checkpoints/interrupted_model.ckpt')
    #parser.add_argument('--ckpt_path', type=str, default='C:\\Users\\Admin\\Desktop\\liuyuhui\\workspace\\jupyter_hybrid_rgbd\\sampledata\\logs\\log\\version_78\\checkpoints\\epoch=11-val_loss=125.9943.ckpt')
    parser.add_argument('--matching_ckpt', type=str, default='./sceneflow.pth')

    parser = Stereo3D.add_model_specific_args(parser)
    #args = parser.parse_args()
    args = parser.parse_known_args()[0]
    main(args)

    args = parser.parse_args()
    main(args)
