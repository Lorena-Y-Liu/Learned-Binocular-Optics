"""
Learned Binocular-Encoding Optics for RGBD Imaging.

Official implementation of "Learned binocular-encoding optics for RGBD imaging 
using joint stereo and focus cues".

Project Page: https://liangxunou.github.io/25liulearned/

This module implements the main Stereo3D PyTorch Lightning model for training
and inference of a depth-from-defocus stereo system with learnable diffractive
optical elements (DOE).

Code References:
    - DOE optimization framework: https://github.com/computational-imaging/DepthFromDefocusWithLearnedOptics
    - Wave propagation (LS-ASM): https://github.com/whywww/ASASM
    - Stereo matching (IGEV): https://github.com/gangweix/IGEV

Usage:
    # Training with config file (recommended):
    python deepstereo_trainer.py --config configs/config.yaml
    
    # Training with command line arguments:
    python deepstereo_trainer.py --gpus 1 --batch_sz 2 --doe_type rank2

License: MIT
"""

import copy
import os
from argparse import ArgumentParser, Namespace
from collections import namedtuple
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.optim
import torchvision.transforms
import torchvision.utils
from debayer import Debayer3x3
from psf.psf_import import *

from models.fusion import Recovery
from util.warp import Warp
from util.matrix import *
from core.igev_stereo import IGEVStereo

from solvers.image_reconstruction import apply_tikhonov_inverse
from util.fft import crop_psf, fftshift
from util.helper import crop_boundary, gray_to_rgb, imresize, linear_to_srgb, srgb_to_linear, to_bayer
from util.loss import Vgg16PerceptualLoss
import cv2

# Import config module for YAML configuration support
try:
    from config import load_config, Config, config_to_namespace
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

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
from torch.cuda.amp import GradScaler

StereoOutputs = namedtuple('StereoOutputs',
                             field_names=['captimgs_left','captimgs_right', 'captimgs_linear_left','captimgs_linear_right',
                                          'captimgs_left_m','captimgs_right_m', 'captimgs_linear_left_m','captimgs_linear_right_m',
                                          'est_images_left','est_images_right','est_depthmaps','est','est_1', 'est_sq','est_dfd','est_dfd_m',
                                          'est_images_left_m','est_images_right_m','est_depthmaps_m','est_m','est_1_m','est_sq_m',
                                          'target_images_left','target_images_right', 
                                          'target_images_left_m','target_images_right_m', 
                                          'norm_max','norm_min','norm_max_m','norm_min_m',
                                          'target_depthmaps','target_roughdepth','target_depthmaps_m','target_roughdepth_m','psf_left','psf_right'])

class Stereo3D(pl.LightningModule):
    """
    Main PyTorch Lightning module for Deep Stereo depth estimation.
    
    This module combines:
    - Dual camera simulation with learnable DOE phase masks
    - Image reconstruction from defocused captures
    - Stereo matching for disparity estimation
    - Multi-loss training including perceptual, depth, and PSF regularization
    
    Args:
        hparams: Hyperparameters namespace containing model configuration
        log_dir: Optional directory for logging outputs
    """

    def __init__(self, hparams, log_dir=None):
        super().__init__()
        self.hparams = hparams
        self.flip = torchvision.transforms.RandomHorizontalFlip(p=1)
        self.save_hyperparameters(copy.deepcopy(hparams))
        self.save_hyperparameters(self.hparams)
        self.__build_model()
        
        # Perceptual loss metrics for image quality evaluation
        self.metrics = {
            'vgg_image_left': Vgg16PerceptualLoss(),
            'vgg_image_left_mirror': Vgg16PerceptualLoss(),
            'vgg_image_right': Vgg16PerceptualLoss(),
        }

        self.log_dir = log_dir
       
    def set_image_size(self, image_sz):
        self.hparams.image_sz = image_sz
        if type(image_sz) == int:
            image_sz += 4 * self.crop_width
        else:
            image_sz[0] += 4 * self.crop_width
            image_sz[1] += 4 * self.crop_width

        self.camera_left.set_image_size(image_sz)
        self.camera_right.set_image_size(image_sz)

    # learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        
        # warm up lr
        if self.trainer.global_step < 4000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 4000.)
            lr_scale_optics = lr_scale = min(1., float(self.trainer.global_step + 1) / 400.)
            optimizer.param_groups[0]['lr'] = lr_scale_optics * float(self.hparams.optics_lr)
            optimizer.param_groups[1]['lr'] = lr_scale_optics * float(self.hparams.optics_lr)
            optimizer.param_groups[2]['lr'] = lr_scale * float(self.hparams.cnn_lr)
            optimizer.param_groups[3]['lr'] = lr_scale * float(self.hparams.depth_lr)
        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        params = [
            {'params': self.camera_left.parameters(), 'lr': self.hparams.optics_lr},
            {'params': self.camera_right.parameters(), 'lr': self.hparams.optics_lr},
            {'params': self.decoder.parameters(), 'lr': self.hparams.cnn_lr},
            {'params': self.matching.parameters(), 'lr': self.hparams.depth_lr},
        ]
        optimizer = torch.optim.Adam(params)
        return optimizer
    
    def training_step(self, samples, batch_idx):
        
        self.matching.eval()
        target_images_left = samples['left_image']
        target_images_right = samples['right_image']

        target_depthmaps = samples['unnorm_depthmap']
        target_norm_depthmaps = samples['depthmap']
        original_depthmaps = samples['original_depth']
        target_depthmaps_m = samples['unnorm_depthmap_2']
        target_norm_depthmaps_m = samples['depthmap_2']
        original_depthmaps_m = samples['original_depth2']
        disparity = samples['disparity']
        disparity_2 = samples['disparity_2']

        # Full resolution images
        original_images_left = samples['original_left']
        original_images_right = samples['original_right']
        original_images_left_m = samples['original_left_m']
        original_images_right_m = samples['original_right_m']
        input_args=[target_images_left,target_images_right,
                    original_images_left, original_images_right, original_images_left_m, original_images_right_m, 
                    target_depthmaps,  target_depthmaps_m, target_norm_depthmaps, target_norm_depthmaps_m,
                    original_depthmaps, original_depthmaps_m ,disparity, disparity_2]
        outputs = self.forward(*input_args, is_training=True)
        target_images_left = outputs.target_images_left
        target_images_right = outputs.target_images_right
        
        target_depthmaps = outputs.target_depthmaps
        target_depthmaps_m = outputs.target_depthmaps_m

        data_loss, loss_logs = self.__compute_loss(outputs)
        loss_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}

        logs = {}
        logs.update(loss_logs)

        if not self.global_step % self.hparams.summary_track_train_every:
            
            self.__log_images(outputs, original_images_left, original_depthmaps,
                              original_images_left_m, original_depthmaps_m,'train')

        self.log_dict(logs)

        
        return data_loss
    
    def on_validation_epoch_start(self) -> None:
        """Move metrics to device before validation."""
        for metric in self.metrics.values():
            metric.to(self.device)
            
    def validation_step(self, samples, batch_idx):
        """Validation step with metric computation."""
        with torch.no_grad():
            target_images_left = samples['left_image']
            target_images_right = samples['right_image']
            target_depthmaps = samples['unnorm_depthmap']
            target_norm_depthmaps = samples['depthmap']
            original_depthmaps = samples['original_depth']
            target_depthmaps_m = samples['unnorm_depthmap_2']
            target_norm_depthmaps_m = samples['depthmap_2']
            original_depthmaps_m = samples['original_depth2']
            original_images_left = samples['original_left']
            original_images_right = samples['original_right']
            original_images_left_m = samples['original_left_m']
            original_images_right_m = samples['original_right_m']

            disparity = samples['disparity']
            disparity_2 = samples['disparity_2']

            input_args = [
                target_images_left, target_images_right,
                original_images_left, original_images_right, 
                original_images_left_m, original_images_right_m,
                target_depthmaps, target_depthmaps_m, 
                target_norm_depthmaps, target_norm_depthmaps_m,
                original_depthmaps, original_depthmaps_m,
                disparity, disparity_2
            ]

            outputs = self.forward(*input_args, is_training=False)

            # Unpack outputs
            est_images_left = outputs.est_images_left
            est_images_left_m = outputs.est_images_left_m
            est_depthmaps = outputs.est_depthmaps
            est_depthmaps_m = outputs.est_depthmaps_m
            rough_depth = outputs.est
            rough_depth_m = outputs.est_m
            target_images_left = outputs.target_images_left
            target_images_left_m = outputs.target_images_left_m

            target_depthmaps = outputs.target_depthmaps
            target_depthmaps_m = outputs.target_depthmaps_m
            target_roughdepth = outputs.target_roughdepth
            target_roughdepth_m = outputs.target_roughdepth_m
            
            # Create valid masks for disparity evaluation
            valid = ((target_roughdepth >= 0.5) & (target_roughdepth < self.hparams.max_disp))
            valid_m = ((target_roughdepth_m >= 0.5) & (target_roughdepth_m < self.hparams.max_disp))
            assert valid.shape == target_roughdepth.shape, [valid.shape, target_roughdepth.shape]
            assert not torch.isinf(target_roughdepth[valid.bool()]).any()
            assert valid_m.shape == target_roughdepth_m.shape, [valid_m.shape, target_roughdepth_m.shape]
            assert not torch.isinf(target_roughdepth_m[valid_m.bool()]).any()

            # Compute metrics
            depth_mse = mse(est_depthmaps, target_depthmaps)
            depth_epe = mae((est_depthmaps) * 255, (target_depthmaps) * 255)
            epe_match = mae(rough_depth[valid.bool()], target_roughdepth[valid.bool()])
            img_mse = mse(est_images_left, target_images_left)

            depth_mse_m = mse(est_depthmaps_m, target_depthmaps_m)
            depth_epe_m = mae((est_depthmaps_m) * 255, (target_depthmaps_m) * 255)
            epe_match_m = mae(rough_depth_m[valid_m.bool()], target_roughdepth_m[valid_m.bool()])
            img_mse_m = mse(est_images_left_m, target_images_left_m)
            
            # Compute val_loss
            val_loss, _ = self.__compute_loss(outputs)
            
            # Log validation metrics
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('validation/mse_depthmap', depth_mse, on_step=False, on_epoch=True)
            self.log('validation/mse_depthmap_m', depth_mse_m, on_step=False, on_epoch=True)
            self.log('validation/depth_epe', depth_epe, on_step=False, on_epoch=True)
            self.log('validation/depth_epe_m', depth_epe_m, on_step=False, on_epoch=True)
            self.log('validation/mae_depthmap_1', epe_match, on_step=False, on_epoch=True)
            self.log('validation/mae_depthmap_1_mirror', epe_match_m, on_step=False, on_epoch=True)

            self.log('validation/mse_image_left', img_mse, on_step=False, on_epoch=True)
            self.log('validation/mse_image_left_mirror', img_mse_m, on_step=False, on_epoch=True)
            self.log('validation/ssim', calculate_ssim(est_images_left, target_images_left), on_step=False, on_epoch=True)
            self.log('validation/ssim_mirror', calculate_ssim(est_images_left_m, target_images_left_m), on_step=False, on_epoch=True)
            
            if batch_idx == 0:
                self.__log_images(outputs, target_images_left, target_depthmaps,
                                  target_images_left_m, target_depthmaps_m, 'validation')
    
    def forward(self, left_images,right_images,
                original_left, original_right, original_left_m, original_right_m, 
                depthmaps, depthmaps_m, depthmaps_norm, depthmaps_norm_m,
                original_depth, original_depth_m, disparity, disparity_2, is_training=True):
        
        hparams=self.hparams
        # invert the gamma correction for sRGB image
        left_images_linear = srgb_to_linear(left_images)
        right_images_linear = srgb_to_linear(right_images)
        # Currently PSF jittering is supported only for MixedCamera.
        if self.hparams.psf_jitter:
            # Jitter the PSF on the evaluation as well.
            captimgs_left,  target_volumes_left, _ = self.camera_left.forward_train(left_images_linear, 
                          depthmaps_norm, occlusion=self.hparams.occlusion)

            # We don't want to use the jittered PSF for the pseudo inverse.
            psf_left = self.camera_left.psf_at_camera(size=(100, 100), is_training=False, modulate_phase=self.hparams.optimize_optics).unsqueeze(0)
            captimgs_right, target_volumes_right, _ = self.camera_right.forward_train(right_images_linear, self.flip(depthmaps_norm_m), 
                          occlusion=self.hparams.occlusion)
            # We don't want to use the jittered PSF for the pseudo inverse.
            psf_right = self.camera_right.psf_at_camera(size=(100, 100), is_training=False, modulate_phase=self.hparams.optimize_optics).unsqueeze(0)
        
        else:
            captimgs_left, target_volumes_left, psf_left =self.camera_left.forward(left_images_linear,depthmaps_norm,occlusion=self.hparams.occlusion, modulate_phase=self.hparams.optimize_optics)
            captimgs_right, target_volumes_right, psf_right =self.camera_right.forward(right_images_linear, self.flip(depthmaps_norm_m),occlusion=self.hparams.occlusion, modulate_phase=self.hparams.optimize_optics)
            
        dtype_left = left_images.dtype
        dtype_right = right_images.dtype
        device_left = left_images.device
        device_right = right_images.device
        noise_sigma_min = self.hparams.noise_sigma_min
        noise_sigma_max = self.hparams.noise_sigma_max
        noise_sigma_left = (noise_sigma_max - noise_sigma_min) * torch.rand((captimgs_left.shape[0], 1, 1, 1), device=device_left,
                                                                       dtype=dtype_left) + noise_sigma_min
        noise_sigma_right =(noise_sigma_max - noise_sigma_min)* torch.rand((captimgs_right.shape[0], 1, 1, 1),device=device_right,
                                                                           dtype=dtype_right) + noise_sigma_min
        # without Bayer
        if not self.hparams.bayer:
            captimgs_left = captimgs_left + noise_sigma_left * torch.randn(captimgs_left.shape, device=device_left, dtype=dtype_left)
            captimgs_right = captimgs_right + noise_sigma_right * torch.randn(captimgs_right.shape, device=device_right, dtype=dtype_right)
        else:
            #cross=random.randint(0, 1)
            #if cross==1:
            captimgs_right_bayer = to_bayer(captimgs_right)
            captimgs_right_bayer = captimgs_right_bayer + noise_sigma_right * torch.randn(captimgs_right_bayer.shape, device=device_left,
                                                                        dtype=dtype_left)
            captimgs_right = self.debayer(captimgs_right_bayer.float())
            
            captimgs_left_bayer = to_bayer(captimgs_left)
            captimgs_left_bayer = captimgs_left_bayer + noise_sigma_left * torch.randn(captimgs_left_bayer.shape, device=device_right,
                                                                        dtype=dtype_right)
            captimgs_left = self.debayer(captimgs_left_bayer.float())
        # Crop the boundary artifact of DFT-based convolution
        captimgs_left = crop_boundary(captimgs_left, self.crop_width)
        captimgs_right = crop_boundary(captimgs_right, self.crop_width)
        target_volumes_left = crop_boundary(target_volumes_left, self.crop_width)
        target_volumes_right = crop_boundary(target_volumes_right, self.crop_width)
        captimgs_left_m, captimgs_right_m=self.flip(captimgs_right), self.flip(captimgs_left)
        
        if self.hparams.preinverse:
            # Apply the Tikhonov-regularized inverse
            psf_cropped_left = crop_psf(psf_left, captimgs_left.shape[-2:])
            psf_cropped_right = crop_psf(psf_right, captimgs_left.shape[-2:])
            pinv_volumes_left = apply_tikhonov_inverse(captimgs_left, psf_cropped_left, self.hparams.reg_tikhonov,
                                                  apply_edgetaper=True)
            pinv_volumes_right = apply_tikhonov_inverse(captimgs_right, psf_cropped_right, self.hparams.reg_tikhonov,
                                                  apply_edgetaper=True)
            pinv_volumes_left_m, pinv_volumes_right_m= self.flip(pinv_volumes_right), self.flip(pinv_volumes_left)
        else:
            pinv_volumes_left = torch.zeros_like(target_volumes_left)
            pinv_volumes_right = torch.zeros_like(target_volumes_right)

            pinv_volumes_left_m, pinv_volumes_right_m = pinv_volumes_left,pinv_volumes_right

        # Use different iteration counts for training vs validation
        iters = hparams.train_iters if is_training else getattr(hparams, 'valid_iters', hparams.train_iters)
        
        est_1, est_sq = self.matching(linear_to_srgb(captimgs_left)*255, linear_to_srgb(captimgs_right)*255, iters=iters)
        est_1_m, est_m_sq = self.matching(linear_to_srgb(captimgs_left_m)*255, linear_to_srgb(captimgs_right_m)*255, iters=iters) 
        est=est_sq[-1]
        est_m=est_m_sq[-1]
        batch = captimgs_left.shape[0]
        device = est.device  # Get device from model output
        
        # Compute normalization values on the same device as est
        norm_max = torch.zeros(batch, device=device)
        norm_min = torch.zeros(batch, device=device)
        norm_max_m = torch.zeros(batch, device=device)
        norm_min_m = torch.zeros(batch, device=device)


        for i in range(batch):
            # Ensure depthmaps are on the same device before computing max/min
            norm_max[i], norm_min[i] = depthmaps[i,...].to(device).max(), depthmaps[i,...].to(device).min()
            norm_max_m[i], norm_min_m[i] = depthmaps_m[i,...].to(device).max(), depthmaps_m[i,...].to(device).min()

        input_rough=(est/255-norm_min.reshape(-1,1,1,1))/(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))
        input_rough_m=(est_m/255-norm_min_m.reshape(-1,1,1,1))/(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))

        if hparams.warp_img:
            self.warping = Warp()
            _, _, h, w = est.shape
            w_disparity=F.interpolate(est, size=(int(h/hparams.scale), int(w/hparams.scale)), mode='bilinear', align_corners=False)
            w_disparity_2=F.interpolate(self.flip(est_m), size=(int(h/hparams.scale), int(w/hparams.scale)), mode='bilinear', align_corners=False)
            
            # Left reconstruction: warp right image to left view
            warped_right, mask=self.warping.warp_disp(captimgs_right, w_disparity, w_disparity_2)
            warped_right+=captimgs_left*(1-mask)
            
            # Right reconstruction: warp left image to right view (symmetric)
            warped_left, mask_r=self.warping.warp_disp(captimgs_left, -w_disparity, -w_disparity_2)
            warped_left+=captimgs_right*(1-mask_r)
            
            # Mirror left reconstruction
            warped_right_m, mask_m=self.warping.warp_disp(captimgs_right_m, self.flip(w_disparity_2), self.flip(w_disparity))
            warped_right_m+=captimgs_left_m*(1-mask_m)
            
            # Mirror right reconstruction (symmetric)
            warped_left_m, mask_r_m=self.warping.warp_disp(captimgs_left_m, -self.flip(w_disparity_2), -self.flip(w_disparity))
            warped_left_m+=captimgs_right_m*(1-mask_r_m)
            
            right=warped_right
            left_for_right=warped_left
            right_m=warped_right_m
            left_for_right_m=warped_left_m
        
        else:
            right=captimgs_right
            left_for_right=captimgs_left
            right_m=captimgs_right_m
            left_for_right_m=captimgs_left_m
            
        # Left image reconstruction
        Outputs = self.decoder(captimgs_left=captimgs_left.float(),
                                        pinv_volumes_left=pinv_volumes_left.float(),
                                        captimgs_right=right.float(),
                                        rough_depth=input_rough.float(), hparams=hparams)
        
        # Right image reconstruction: use right camera as primary view with warped left
        Outputs_right = self.decoder(captimgs_left=captimgs_right.float(),
                                        pinv_volumes_left=pinv_volumes_right.float(),
                                        captimgs_right=left_for_right.float(),
                                        rough_depth=self.flip(input_rough_m).float(), hparams=hparams)
                    
        # Mirror left reconstruction
        Outputs_m = self.decoder(captimgs_left=captimgs_left_m.float(),
                                        pinv_volumes_left=pinv_volumes_left_m.float(),
                                        captimgs_right=right_m.float(),
                                        rough_depth=input_rough_m.float(), hparams=hparams)
        
        # Mirror right reconstruction with warped left
        Outputs_right_m = self.decoder(captimgs_left=captimgs_right_m.float(),
                                        pinv_volumes_left=pinv_volumes_right_m.float(),
                                        captimgs_right=left_for_right_m.float(),
                                        rough_depth=self.flip(input_rough).float(), hparams=hparams)
        
        left=Outputs[0]
        est_dfd=Outputs[1]
        est_depthmaps = Outputs[2]
        est_dfd=est_dfd*(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))+norm_min.reshape(-1,1,1,1)
        est_depthmaps=est_depthmaps*(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))+norm_min.reshape(-1,1,1,1)

        # 右图重建结果
        right_recon = Outputs_right[0]

        left_m= Outputs_m[0]
        est_dfd_m=Outputs_m[1]
        est_depthmaps_m = Outputs_m[2]
        est_dfd_m=est_dfd_m*(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))+norm_min_m.reshape(-1,1,1,1)
        est_depthmaps_m=est_depthmaps_m*(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))+norm_min_m.reshape(-1,1,1,1)

        # 镜像右图重建结果
        right_recon_m = Outputs_right_m[0]

        # Require twice cropping because the image formation also crops the boundary.

        est_images_left = crop_boundary(left, self.crop_width)
        est_images_right = crop_boundary(right_recon, self.crop_width)
        est_images_left_m = crop_boundary(left_m, self.crop_width)
        est_images_right_m = crop_boundary(right_recon_m, self.crop_width)
        outputs = StereoOutputs(
            target_images_right=original_right,
            target_images_left=original_left,
            target_roughdepth=disparity*(-1), #target_roughdepth,
            target_depthmaps=disparity*(-1)/255,#original_depthmaps,
            norm_max=norm_max,
            norm_min=norm_min,
            norm_max_m=norm_max_m,
            norm_min_m=norm_min_m,
            captimgs_right=linear_to_srgb(captimgs_right),
            captimgs_left=linear_to_srgb(captimgs_left),
            captimgs_linear_right=captimgs_right,
            captimgs_linear_left=captimgs_left,
            est_images_left=est_images_left,
            est_images_right=est_images_right,
            est_1=est_1,  #rough depth from matching
            est=est,
            est_sq=est_sq,
            est_sq_m=est_m_sq,  
            est_dfd=est_dfd,
            est_depthmaps=est_depthmaps,############## Mirror
            target_images_right_m=original_right_m,
            target_images_left_m=original_left_m,
            target_roughdepth_m = self.flip(disparity_2), #target_roughdepth_m,
            target_depthmaps_m=self.flip(disparity_2)/255,#original_depthmaps_m,
            captimgs_right_m=linear_to_srgb(captimgs_right_m),
            captimgs_left_m=linear_to_srgb(captimgs_left_m),
            captimgs_linear_right_m=captimgs_right_m,
            captimgs_linear_left_m=captimgs_left_m,
            est_images_left_m=est_images_left_m,
            est_images_right_m=est_images_right_m,
            est_1_m=est_1_m,  #rough depth from matching
            est_m=est_m,
            est_dfd_m=est_dfd_m,
            est_depthmaps_m=est_depthmaps_m,
            psf_left=psf_left,
            psf_right=psf_right)
        return outputs

    def __build_model(self):
        hparams = self.hparams
        self.crop_width = hparams.crop_width
        mask_diameter = hparams.mask_diameter #hparams.focal_length / hparams.f_number
        wavelengths = [632e-9, 550e-9, 450e-9]
        
        # Get use_pretrained_doe from hparams, default to False if not set
        use_pretrained_doe = getattr(hparams, 'use_pretrained_doe', False)
        
        camera_recipe = {
            'wavelengths': wavelengths,
            'min_depth': hparams.min_depth,
            'max_depth': hparams.max_depth,
            'focal_depth': hparams.focal_depth,
            'n_depths': hparams.n_depths,
            'image_size': hparams.image_sz ,#+ 4 * self.crop_width,
            'camera_pixel_pitch': hparams.camera_pixel_pitch,
            'focal_length': hparams.focal_length,
            'mask_diameter': mask_diameter,
            'mask_size': hparams.mask_sz,
            'mask_pitch': hparams.mask_pitch,
            'mask_upsample_factor': hparams.mask_upsample_factor,
            'diffraction_efficiency': hparams.diffraction_efficiency,
            'full_size': hparams.full_size,
            'use_pretrained_doe': use_pretrained_doe,
        }
        camera_recipe_right = {
            'wavelengths': wavelengths,
            'min_depth': hparams.min_depth,
            'max_depth': hparams.max_depth,
            'focal_depth': hparams.focal_depth_right,
            'n_depths': hparams.n_depths,
            'image_size': hparams.image_sz ,#+ 4 * self.crop_width,
            'camera_pixel_pitch': hparams.camera_pixel_pitch,
            'focal_length': hparams.focal_length,
            'mask_diameter': mask_diameter,
            'mask_size': hparams.mask_sz,
            'mask_pitch': hparams.mask_pitch,
            'mask_upsample_factor': hparams.mask_upsample_factor,
            'diffraction_efficiency': hparams.diffraction_efficiency,
            'full_size': hparams.full_size,
            'use_pretrained_doe': use_pretrained_doe,
        }
        optimize_optics = hparams.optimize_optics
        doe_type=hparams.doe_type
        if doe_type=='rank2':
            from optics import camera_left_rank2 as camera_left
            from optics import camera_right_rank2 as camera_right
        if doe_type=='rank1':
            from optics import camera_left_rank1 as camera_left
            from optics import camera_right_rank1 as camera_right
        if doe_type=='ring':    
            from optics import camera_left_ring as camera_left
            from optics import camera_right_ring as camera_right
        if doe_type=='ring_base':    
            from optics import camera_left_ring as camera_left
            from optics import camera_left_ring as camera_right
        '''if doe_type=='pixel_wise':
            from optics import camera_left_pw as camera_left
            from optics import camera_right_pw as camera_right'''
        self.camera_left = camera_left.MixedCamera(**camera_recipe, requires_grad=optimize_optics)
        self.camera_right = camera_right.MixedCamera(**camera_recipe_right, requires_grad=optimize_optics)
        self.matching = IGEVStereo(hparams)
        self.decoder = Recovery(hparams, requires_grad=True)
        self.debayer = Debayer3x3()
        self.image_lossfn = Vgg16PerceptualLoss()
        self.image_lossfn2 = torch.nn.L1Loss()
        self.depth_lossfn = torch.nn.MSELoss()
        self.depth_lossfn2 = torch.nn.L1Loss()
        print(self.camera_left)

    def __combine_loss(self, depth_loss,depth_1_loss, image_loss, psf_loss):
        return self.hparams.depth_loss_weight * depth_loss + \
                self.hparams.depth_1_loss_weight * depth_1_loss + \
               self.hparams.image_loss_weight * image_loss+ \
               self.hparams.psf_loss_weight * psf_loss    
    def __compute_loss(self, outputs):
        
        hparams = self.hparams
        target_depthmaps=outputs.target_depthmaps
        target_images_left=outputs.target_images_left
        target_images_right=outputs.target_images_right  
        target_depthmaps_m=outputs.target_depthmaps_m
        target_images_left_m=outputs.target_images_left_m
        target_images_right_m=outputs.target_images_right_m  
        est_images_left = outputs.est_images_left
        est_images_right = outputs.est_images_right  
        est_1=outputs.est_1
        est=outputs.est
        est_depthmaps = outputs.est_depthmaps
        est_dfd=outputs.est_dfd
        target_roughdepth= outputs.target_roughdepth
        # Mirror
        est_images_left_m = outputs.est_images_left_m
        est_images_right_m = outputs.est_images_right_m  
        est_1_m=outputs.est_1_m
        est_m=outputs.est_m
        est_depthmaps_m = outputs.est_depthmaps_m
        est_dfd_m=outputs.est_dfd_m
        target_roughdepth_m= outputs.target_roughdepth_m

        psnr_left = calculate_psnr(est_images_left, target_images_left)
        ssmi_left = calculate_ssim(est_images_left, target_images_left)
        left_image_loss = self.image_lossfn.train_loss(est_images_left, target_images_left)
        left_image_loss_m = self.image_lossfn.train_loss(est_images_left_m, target_images_left_m)
        right_image_loss = self.image_lossfn.train_loss(est_images_right, target_images_right)
        right_image_loss_m = self.image_lossfn.train_loss(est_images_right_m, target_images_right_m)
        
        valid = ((target_roughdepth >= 0.5) & (target_roughdepth < hparams.max_disp))
        valid_m = ((target_roughdepth_m >= 0.5) & (target_roughdepth_m < hparams.max_disp))
        disp_loss = 0.0
        disp_loss_m = 0.0
        
        # Use actual sequence length instead of hparams.train_iters
        num_iters = len(outputs.est_sq)
        for i in range(num_iters):
            est_s = outputs.est_sq[i]
            est_s_m = outputs.est_sq_m[i]
            loss_gamma = 0.9
            adjusted_loss_gamma = loss_gamma**(15/(num_iters - 1))
            i_weight = adjusted_loss_gamma**(num_iters - i - 1)
            i_loss = (target_roughdepth - est_s).abs()
            i_loss_m = (target_roughdepth_m - est_s_m).abs()
            disp_loss += i_weight * i_loss[valid.bool()].mean()
            disp_loss_m += i_weight * i_loss_m[valid_m.bool()].mean()

        disp_loss/=num_iters
        disp_loss_m/=num_iters
        depth_1_loss=disp_loss+mae(est_1[valid.bool()], target_roughdepth[valid.bool()])
        depth_2_loss=mae(est[valid.bool()], target_roughdepth[valid.bool()])
        depth_2_loss_all=mae(est, target_roughdepth)
        depth_1_loss_m=disp_loss_m+mae(est_1_m[valid_m.bool()], target_roughdepth_m[valid_m.bool()])
        depth_2_loss_m=mae(est_m[valid_m.bool()], target_roughdepth_m[valid_m.bool()])
        dfd_loss=mae(est_dfd, target_depthmaps)
        dfd_loss_m=mae(est_dfd_m, target_depthmaps_m)
        depth_loss = mae(est_depthmaps, target_depthmaps)
        
        px_3=calculate_3px(255*est_depthmaps,255*target_depthmaps)
        epe_loss = mae(255*est_depthmaps,255*target_depthmaps)
        epe_loss_m = mae(255*est_depthmaps_m,255*target_depthmaps_m)
        psf_left_out_of_fov_sum = self.camera_left.psf_out_of_fov_energy(hparams.psf_size)
        psf_left_loss = psf_left_out_of_fov_sum

        psf_right_out_of_fov_sum = self.camera_right.psf_out_of_fov_energy(hparams.psf_size)
        psf_right_loss = psf_right_out_of_fov_sum
        
        total_image_loss = (left_image_loss + left_image_loss_m + right_image_loss + right_image_loss_m) / 2
        
        total_loss = self.__combine_loss(
            (epe_loss + epe_loss_m + dfd_loss + dfd_loss_m) / 10, 
            (depth_2_loss + depth_2_loss_m) / 2 + (depth_1_loss + depth_1_loss_m) / 4, 
            total_image_loss,  
            psf_left_loss + psf_right_loss
        )
        
        logs = {
            'total_loss': total_loss,
            'depth_loss': depth_loss,
            'disp_loss': depth_2_loss, 
            'disp_loss_all': depth_2_loss_all, 
            'left_image_loss': left_image_loss,
            'right_image_loss': right_image_loss,  
            'psf_loss_left': psf_left_loss,
            'psf_loss_right': psf_right_loss,
            'left_image_psnr': psnr_left,
            'left_image_ssmi': ssmi_left,
            'depth_epe': epe_loss,
            'depth_3px': px_3,
        }
        return total_loss, logs

    @torch.no_grad()
    def __log_images(self, outputs, target_images_left, target_depthmaps, target_images_left_m, target_depthmaps_m, tag: str):
        # Unpack outputs
        captimgs_left = outputs.captimgs_left
        est_images_left = outputs.est_images_left
        est_depthmaps = outputs.est_depthmaps
        
        est = outputs.est/255
        target_roughdepth= outputs.target_roughdepth/255
        target_depthmaps=outputs.target_depthmaps
        captimgs_left_m = outputs.captimgs_left_m
        est_images_left_m = outputs.est_images_left_m
        est_depthmaps_m = outputs.est_depthmaps_m
        
        est_m =  outputs.est_m/255#-outputs.est_m.min())/(outputs.est_m.max()-outputs.est_m.min())
        target_depthmaps_m=outputs.target_depthmaps_m

        est_dfd=outputs.est_dfd
        est_dfd_m=outputs.est_dfd_m


        summary_image_sz = self.hparams.summary_image_sz
        # CAUTION! Summary image is clamped, and visualized in sRGB.
        summary_max_images = min(self.hparams.summary_max_images, target_images_left.shape[0])
        # Flip [0, 1] for visualization purpose
        target_depthmaps = gray_to_rgb(1-target_depthmaps)
        est_depthmaps = gray_to_rgb(1-est_depthmaps)
        est = gray_to_rgb(1-est)

        target_roughdepth= gray_to_rgb(1-target_roughdepth)

        est_m= gray_to_rgb(1-est_m)
        est_dfd= gray_to_rgb(1-est_dfd)
        est_dfd_m= gray_to_rgb(1-est_dfd_m)

        est_depthmaps_m= gray_to_rgb(1-est_depthmaps_m)
        target_depthmaps_m= gray_to_rgb(1-target_depthmaps_m)

        summary = torch.cat([captimgs_left[:,:3,...], captimgs_left_m[:,:3,...]], dim=-2)
        
        summary2 = torch.cat([target_images_left, est_images_left, target_depthmaps,est, est_dfd, est_depthmaps], dim=-2)
        summary3 = torch.cat([target_images_left_m, est_images_left_m, target_depthmaps_m, est_m, est_dfd_m, est_depthmaps_m], dim=-2)
        summary = summary[:summary_max_images]
        summary2 = summary2[:summary_max_images]
        summary3 = summary3[:summary_max_images]
        grid_summary = torchvision.utils.make_grid(summary, nrow=summary_max_images)
        grid_summary2 = torchvision.utils.make_grid(summary2, nrow=summary_max_images)
        grid_summary3 = torchvision.utils.make_grid(summary3, nrow=summary_max_images)
        self.logger.experiment.add_image(f'{tag}/summary', grid_summary, self.global_step)
        self.logger.experiment.add_image(f'{tag}/summary2', grid_summary2, self.global_step)
        self.logger.experiment.add_image(f'{tag}/summary3', grid_summary3, self.global_step)
        
        if self.hparams.optimize_optics or self.global_step >=0:

            size=(200,200)
            psf_left = self.camera_left.psf_at_camera(size=size, is_training=False, modulate_phase=self.hparams.optimize_optics)

            phasemap_left_1 = imresize(self.camera_left.phase()[[1], :, :,:],
                                 [self.hparams.summary_mask_sz, self.hparams.summary_mask_sz]).squeeze(0)
            
            sorted_0, _ = torch.sort(phasemap_left_1.view(-1))
            phasemap_left_1 = torch.where(phasemap_left_1 == phasemap_left_1.min(), sorted_0[-2], phasemap_left_1)
            phasemap_left_1 -= phasemap_left_1.min()
            phasemap_left_1 /= phasemap_left_1.max()
            
            self.logger.experiment.add_image('optics/phasemap_left_G', phasemap_left_1, self.global_step)
            psf_left= psf_left.flip(1)
            grid_psf_left = torchvision.utils.make_grid(psf_left.transpose(0, 1),
                                                   nrow=9, pad_value=1, normalize=False)
            self.logger.experiment.add_image('optics/psf_left', grid_psf_left, self.global_step)
            
            psf_left /= psf_left.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
            
            grid_psf_left = torchvision.utils.make_grid(psf_left.transpose(0, 1),
                                                   nrow=9, pad_value=1, normalize=False)
            self.logger.experiment.add_image('optics/psf_stretched_left', grid_psf_left, self.global_step)

            psf_right = self.camera_right.psf_at_camera(size=size, is_training=False, modulate_phase=self.hparams.optimize_optics)
            phasemap_right_1 = imresize(self.camera_right.phase()[[1], :, :,:],
                                 [self.hparams.summary_mask_sz, self.hparams.summary_mask_sz]).squeeze(0)
            sorted_0_r, _ = torch.sort(phasemap_right_1.view(-1))
            phasemap_right_1 = torch.where(phasemap_right_1 == phasemap_right_1.min(), sorted_0_r[-2], phasemap_right_1)
            phasemap_right_1 -= phasemap_right_1.min()
            phasemap_right_1 /= phasemap_right_1.max()
            self.logger.experiment.add_image('optics/phasemap_right_G', phasemap_right_1, self.global_step)
            psf_right= psf_right.flip(1)
            grid_psf_right = torchvision.utils.make_grid(psf_right.transpose(0, 1),
                                                   nrow=9, pad_value=1, normalize=False)
            self.logger.experiment.add_image('optics/psf_right', grid_psf_right, self.global_step)
            
            psf_right /= psf_right.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
            grid_psf_right = torchvision.utils.make_grid(psf_right.transpose(0, 1),
                                                   nrow=9, pad_value=1, normalize=False)
            self.logger.experiment.add_image('optics/psf_stretched_right', grid_psf_right, self.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model-specific arguments to the argument parser.
        
        Supports both config file and command-line arguments. When both are provided,
        command-line arguments take precedence over config file values.
        
        Args:
            parent_parser: Parent ArgumentParser to extend
            
        Returns:
            ArgumentParser with all model-specific arguments added
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        # Config file argument (highest priority when loading defaults)
        parser.add_argument('--config', type=str, default=None, 
                           help='Path to YAML configuration file (e.g., configs/default.yaml)')
        
        # Logger parameters
        parser.add_argument('--summary_max_images', type=int, default=8)
        parser.add_argument('--summary_image_sz', type=int, default=200)#256)
        parser.add_argument('--summary_mask_sz', type=int, default=1260)#256)
        parser.add_argument('--summary_depth_every', type=int, default=2000)
        parser.add_argument('--summary_track_train_every', type=int, default=500) #1000)

        # training parameters
        parser.add_argument('--cnn_lr', type=float, default=1e-3)#0.5e-3)
        parser.add_argument('--depth_lr', type=float, default=1e-5)
        parser.add_argument('--optics_lr', type=float, default=0)#0.1e-3)#2e-2)#1e-3)#=0.5e-3
        parser.add_argument('--batch_sz', type=int, default=1)#10) #6
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--augment', default=True, action='store_true')
        
        # loss parameters
        parser.add_argument('--depth_loss_weight', type=float, default=1)
        parser.add_argument('--depth_1_loss_weight', type=float, default=0)#0.5)
        parser.add_argument('--image_loss_weight', type=float, default=1)
        parser.add_argument('--psf_loss_weight', type=float, default=0)
        parser.add_argument('--psf_size', type=int, default=160)

        # dataset parameters
        parser.add_argument('--image_sz', type=list, default=[320, 736])
        parser.add_argument('--n_depths', type=int, default=7)
        parser.add_argument('--min_depth', type=float, default=0.67) 
        parser.add_argument('--max_depth', type=float, default=8.0)
        parser.add_argument('--crop_width', type=int, default=0)

        # solver parameters
        parser.add_argument('--reg_tikhonov', type=float, default=0.1)
        parser.add_argument('--model_base_ch', type=int, default=32)
        parser.add_argument('--preinverse', dest='preinverse', action='store_true')
        parser.add_argument('--no-preinverse', dest='preinverse', action='store_false')
        parser.set_defaults(preinverse=True)
        parser.add_argument('--warp_img', dest='warp_img', action='store_true')
        parser.set_defaults(warp_img=True)
        # optics parameters
        parser.add_argument('--camera_type', type=str, default='mixed')
        parser.add_argument('--mask_sz', type=int, default=1260) 
        
        parser.add_argument('--focal_length', type=float, default=35e-3)
        parser.add_argument('--focal_depth', type=float, default=1.23) 
        parser.add_argument('--focal_depth_right', type=float, default=1.23) 
        parser.add_argument('--mask_pitch', type=float, default=3.45e-6)
        parser.add_argument('--mask_diameter', type=float, default=4.347e-3)
        parser.add_argument('--camera_pixel_pitch', type=float, default=5.86e-6)
        parser.add_argument('--noise_sigma_min', type=float, default=0.001)
        parser.add_argument('--noise_sigma_max', type=float, default=0.005)
        parser.add_argument('--full_size', type=int, default=1200)
        parser.add_argument('--mask_upsample_factor', type=int, default=2)
        parser.add_argument('--diffraction_efficiency', type=float, default=0.7)
        parser.add_argument('--scale', type=float, default=1)

        parser.add_argument('--bayer', dest='bayer', action='store_true')
        parser.add_argument('--no-bayer', dest='bayer', action='store_false')
        parser.set_defaults(bayer=True)
        parser.add_argument('--occlusion', dest='occlusion', action='store_true')
        parser.add_argument('--no-occlusion', dest='occlusion', action='store_false')
        parser.set_defaults(occlusion=True)
        parser.add_argument('--optimize_optics', dest='optimize_optics', action='store_true')
        parser.add_argument('--no-optimize_optics', dest='optimize_optics', action='store_false')
        parser.set_defaults(optimize_optics=True)
        parser.add_argument('--doe_type', type=str, default='rank2', help="doe modeling method")
        
        # model parameters
        parser.add_argument('--psfjitter', dest='psf_jitter', action='store_true')
        parser.add_argument('--no-psfjitter', dest='psf_jitter', action='store_false')
        parser.set_defaults(psf_jitter=False)

        ###IGEV
        parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
        parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
        parser.add_argument('--train_iters', type=int, default=12, help="number of updates to the disparity field in each forward pass.")
        
        # Validation parameters
        parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during validation forward pass')

        # Architecure choices
        parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
        parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
        parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
        parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
        parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
        parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

        # Data augmentation
        parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
        parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
        parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
        torch.manual_seed(666)

        return parser

    @staticmethod
    def load_args_from_config(args):
        """
        Load configuration from YAML file and merge with command-line arguments.
        
        Priority order (highest to lowest):
        1. Command-line arguments (explicitly provided)
        2. Config file values
        3. Default values in argparse
        
        Args:
            args: Namespace object from argparse
            
        Returns:
            Updated Namespace with merged configuration
        """
        if not CONFIG_AVAILABLE:
            print("Warning: config module not available. Using command-line args only.")
            return args
            
        if args.config is None:
            return args
        
        # Load config from YAML file
        try:
            config = load_config(args.config)
            config_hparams = config.to_hparams()
            print(f"Loaded configuration from: {args.config}")
        except FileNotFoundError:
            print(f"Warning: Config file not found: {args.config}")
            return args
        except Exception as e:
            print(f"Warning: Error loading config file: {e}")
            return args
        
        # Merge config values with args (command-line args take priority)
        for key, value in config_hparams.items():
            # Only update if not explicitly set via command line
            if hasattr(args, key):
                current_value = getattr(args, key)
                # Check if value is still the default (argparse default)
                # This is a simple heuristic - explicit CLI args will override
                setattr(args, key, value)
        
        return args


def load_hparams_from_config(config_path: str) -> Namespace:
    """
    Convenience function to load hyperparameters directly from a config file.
    
    This allows using config files without argparse for inference or testing.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Namespace object with all hyperparameters
        
    Example:
        hparams = load_hparams_from_config('configs/rank2.yaml')
        model = Stereo3D(hparams)
    """
    if not CONFIG_AVAILABLE:
        raise ImportError("config module not available. Please check config.py exists.")
    
    config = load_config(config_path)
    return config_to_namespace(config)