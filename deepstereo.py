"""
python deepstereo_trainer.py --gpus 4 --experiment_name 'fabrication_mixed_camera' --occlusion --augment --batch_sz 3 --preinverse --camera_type mixed --optimize_optics --bayer --focal_depth 1.7 --distributed_backend ddp  --max_epochs 1000 --psf_loss_weight 1.00

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['MASTER_ADDR'] = 'localhost'
import copy
from argparse import ArgumentParser
from collections import namedtuple
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.optim
import torchvision.transforms
import torchvision.utils
from debayer import Debayer3x3
from psf.psf_import import *

from models.recovery2_m import Recovery
from util.warp import Warp
from util.matrix import *
from core.igev_stereo import IGEVStereo

from solvers.image_reconstruction import apply_tikhonov_inverse
from util.fft import crop_psf, fftshift
from util.helper import crop_boundary, gray_to_rgb, imresize, linear_to_srgb, srgb_to_linear, to_bayer
from util.loss import Vgg16PerceptualLoss
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
from torch.cuda.amp import GradScaler

StereoOutputs = namedtuple('StereoOutputs',
                             field_names=['captimgs_left','captimgs_right', 'captimgs_linear_left','captimgs_linear_right',
                                          'captimgs_left_m','captimgs_right_m', 'captimgs_linear_left_m','captimgs_linear_right_m',
                                          'est_images_left','est_depthmaps','est','est_1', 'est_sq','est_dfd','est_dfd_m',
                                          'est_images_left_m','est_depthmaps_m','est_m','est_1_m','est_sq_m',
                                          'target_images_left','target_images_right', 
                                          'target_images_left_m','target_images_right_m', 
                                          'norm_max','norm_min','norm_max_m','norm_min_m',
                                          'target_depthmaps','target_roughdepth','target_depthmaps_m','target_roughdepth_m','psf_left','psf_right'])

class Stereo3D(pl.LightningModule):

    def __init__(self, hparams, log_dir=None):
        super().__init__()
        # self.hparams = hparams
        self.flip=torchvision.transforms.RandomHorizontalFlip(p=1)
        #self.psf_experiment=psf_experiment(self.hparams.image_sz)
        self.save_hyperparameters(copy.deepcopy(hparams))
        #for key in hparams.keys():
            #self.hparams[key]=copy.deepcopy(hparams[key])
        
        self.save_hyperparameters(self.hparams)
        self.__build_model()
        
        self.metrics = {
            'vgg_image_left': Vgg16PerceptualLoss(),#MeanSquaredError(),
            'vgg_image_left_mirror': Vgg16PerceptualLoss(),
            'vgg_image_right': Vgg16PerceptualLoss(),
        }

        self.log_dir = log_dir


    def train_dataloader(self):
        if self.train_dataloader_obj is None:
            raise ValueError("train_dataloader_obj has not been set.")
        return self.train_dataloader_obj

    def val_dataloader(self):
        if self.val_dataloader_obj is None:
            raise ValueError("val_dataloader_obj has not been set.")
        return self.val_dataloader_obj
       
    def set_image_size(self, image_sz):
        self.hparams.image_sz = image_sz
        if type(image_sz) == int:
            image_sz += 4 * self.crop_width
        else:
            image_sz[0] += 4 * self.crop_width
            image_sz[1] += 4 * self.crop_width

        self.camera_left.set_image_size(image_sz)
        self.camera_right.set_image_size(image_sz)

    # # learning rate warm-up
    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False,
    #                    using_native_amp=False, using_lbfgs=False):
        
    #     # warm up lr
    #     if self.trainer.global_step < 4000:
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / 4000.)
    #         lr_scale_optics = lr_scale 
    #         optimizer.param_groups[0]['lr'] = lr_scale_optics * self.hparams.optics_lr
    #         optimizer.param_groups[1]['lr'] = lr_scale_optics * self.hparams.optics_lr
    #         optimizer.param_groups[2]['lr'] = lr_scale * self.hparams.cnn_lr
    #         optimizer.param_groups[3]['lr'] = lr_scale * self.hparams.depth_lr
    #     # update params
       
    #     # if optimizer_closure is not None:
    #     #     optimizer_closure()
    #     optimizer.step(optimizer_closure)
    #     optimizer.zero_grad()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.camera_left.parameters(), 'lr': self.hparams.optics_lr},
            {'params': self.camera_right.parameters(), 'lr': self.hparams.optics_lr},
            {'params': self.decoder.parameters(), 'lr': self.hparams.cnn_lr},
            {'params': self.matching.parameters(), 'lr': self.hparams.depth_lr},
        ])
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1., float(step + 1) / 4000.)  # Warm-up 4000 steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'  
        }
    
    def training_step(self, samples, batch_idx):
        
        self.matching.eval()
        target_images_left = samples['left_image']
        target_images_right = samples['right_image']

        target_depthmaps = samples['unnorm_depthmap']#samples['depthmap']
        target_norm_depthmaps = samples['depthmap']
        original_depthmaps=samples['original_depth']
        target_depthmaps_m = samples['unnorm_depthmap_2']#samples['depthmap_2']
        target_norm_depthmaps_m = samples['depthmap_2']
        original_depthmaps_m=samples['original_depth2']
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
        outputs = self.forward(*input_args)
        target_images_left = outputs.target_images_left
        target_images_right = outputs.target_images_right
        
        target_depthmaps = outputs.target_depthmaps
        target_depthmaps_m = outputs.target_depthmaps_m

        data_loss, loss_logs = self.__compute_loss(outputs)
        loss_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}

        logs = {}
        logs.update(loss_logs)
        #logs.update(misc_logs)

        if not self.global_step % self.hparams.summary_track_train_every:
            
            self.__log_images(outputs, original_images_left, original_depthmaps,
                              original_images_left_m, original_depthmaps_m,'train')

        self.log_dict(logs)

        
        return data_loss
    
    def on_validation_epoch_start(self) -> None:
        
        for metric in self.metrics.values():
            #metric.reset() 
            metric.to(self.device)
            
    def validation_step(self, samples, batch_idx):
        
        with torch.no_grad():
            target_images_left = samples['left_image']
            target_images_right = samples['right_image']
            target_depthmaps = samples['unnorm_depthmap']
            target_norm_depthmaps = samples['depthmap']
            original_depthmaps=samples['original_depth']
            target_depthmaps_m = samples['unnorm_depthmap_2']
            target_norm_depthmaps_m = samples['depthmap_2']
            original_depthmaps_m=samples['original_depth2']
            original_images_left = samples['original_left']
            original_images_right = samples['original_right']
            original_images_left_m = samples['original_left_m']
            original_images_right_m = samples['original_right_m']

            disparity = samples['disparity']
            disparity_2 = samples['disparity_2']

            input_args=[target_images_left,target_images_right,
                    original_images_left, original_images_right, original_images_left_m, original_images_right_m, 
                    target_depthmaps,  target_depthmaps_m, target_norm_depthmaps, target_norm_depthmaps_m,
                    original_depthmaps, original_depthmaps_m ,disparity, disparity_2]

            outputs = self.forward(*input_args)

            # Unpack outputs
            est_images_left = outputs.est_images_left
            est_images_left_m = outputs.est_images_left_m
            est_depthmaps = outputs.est_depthmaps
            est_depthmaps_m = outputs.est_depthmaps_m
            rough_depth = outputs.est
            rough_depth_m = outputs.est_m
            #est_depthmaps = outputs.est_depthmaps
            target_images_left = outputs.target_images_left
            target_images_left_m = outputs.target_images_left_m

            target_depthmaps = outputs.target_depthmaps
            target_depthmaps_m = outputs.target_depthmaps_m
            #c,d= target_depthmaps.shape[-2:]
            #scale=self.hparams.scale #1.6
            target_roughdepth= outputs.target_roughdepth
            target_roughdepth_m= outputs.target_roughdepth_m
            #target_roughdepth= F.interpolate(target_depthmaps, size=(int(c/scale), int(d/scale)), mode='bilinear', align_corners=True)
            self.outputs=outputs
            #mag = torch.sum(target_roughdepth**2, dim=1).sqrt().unsqueeze(1)
            valid = ((target_roughdepth >= 0.5) & (target_roughdepth < self.hparams.max_disp))
            valid_m = ((target_roughdepth_m >= 0.5) & (target_roughdepth_m < self.hparams.max_disp))
            assert valid.shape == target_roughdepth.shape, [valid.shape, target_roughdepth.shape]
            assert not torch.isinf(target_roughdepth[valid.bool()]).any()

            depth_mse=mse(est_depthmaps, target_depthmaps)
            depth_epe=mae((est_depthmaps)*255, (target_depthmaps)*255)
            depth_3px=calculate_3px((est_depthmaps)*255, (target_depthmaps)*255)
            epe_match=mae(rough_depth[valid.bool()], target_roughdepth[valid.bool()])
            img_mse=mse(est_images_left, target_images_left)
            #self.metrics['vgg_image_left'].train_loss(est_images_left, target_images_left)

            depth_mse_m=mse(est_depthmaps_m, target_depthmaps_m)
            depth_epe_m=mae((est_depthmaps_m)*255, (target_depthmaps_m)*255)
            depth_3px_m=calculate_3px((est_depthmaps_m)*255, (target_depthmaps_m)*255)
            epe_match_m=mae(rough_depth_m[valid.bool()], target_roughdepth_m[valid.bool()])
            img_mse_m=mse(est_images_left_m, target_images_left_m)
            #self.metrics['vgg_image_left_mirror'].train_loss(est_images_left_m, target_images_left_m)
            
            
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
            #     (((est_depthmaps-target_depthmaps)*255).abs().mean())
            
            if batch_idx ==0:
                self.__log_images(outputs, target_images_left, target_depthmaps,
                                  target_images_left_m, target_depthmaps_m, 'validation')
            if batch_idx ==1:
                self.__log_images(outputs, target_images_left, target_depthmaps, 
                                  target_images_left_m, target_depthmaps_m, 'validation2')
            if batch_idx ==2:
                self.__log_images(outputs, target_images_left, target_depthmaps, 
                                  target_images_left_m, target_depthmaps_m, 'validation3')
            if batch_idx ==3:
                self.__log_images(outputs, target_images_left, target_depthmaps, 
                                  target_images_left_m, target_depthmaps_m, 'validation4')
           
    def on_validation_epoch_end(self):
        
        outputs=self.outputs
        with torch.no_grad():
            val_loss, _= self.__compute_loss(outputs)
        self.log('val_loss', val_loss)
    
    def forward(self, left_images,right_images,
                original_left, original_right, original_left_m, original_right_m, 
                depthmaps, depthmaps_m, depthmaps_norm, depthmaps_norm_m,
                original_depth, original_depth_m, disparity, disparity_2):
        
        hparams=self.hparams
        # invert the gamma correction for sRGB image
        left_images_linear = srgb_to_linear(left_images)
        right_images_linear = srgb_to_linear(right_images)
        # Currently PSF jittering is supported only for MixedCamera.
        if torch.tensor(self.hparams.psf_jitter):
            # Jitter the PSF on the evaluation as well.
            captimgs_left,  target_volumes_left, _ = self.camera_left.forward_train(left_images_linear, 
                          depthmaps_norm, occlusion=self.hparams.occlusion)

            # We don't want to use the jittered PSF for the pseudo inverse.
            psf_left = self.camera_left.psf_at_camera(size=(100, 100), is_training=torch.tensor(False), modulate_phase=self.hparams.optimize_optics).unsqueeze(0)
            captimgs_right, target_volumes_right, _ = self.camera_right.forward_train(right_images_linear, self.flip(depthmaps_norm_m), 
                          occlusion=self.hparams.occlusion)
            # We don't want to use the jittered PSF for the pseudo inverse.
            psf_right = self.camera_right.psf_at_camera(size=(100, 100), is_training=torch.tensor(False),modulate_phase=self.hparams.optimize_optics).unsqueeze(0)
        
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
        if not torch.tensor(self.hparams.bayer):
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

        est_1, est_sq = self.matching(linear_to_srgb(captimgs_left)*255, linear_to_srgb(captimgs_right)*255, iters=hparams.train_iters)
        est_1_m, est_m_sq = self.matching(linear_to_srgb(captimgs_left_m)*255, linear_to_srgb(captimgs_right_m)*255, iters=hparams.train_iters) 
        est=est_sq[-1]
        est_m=est_m_sq[-1]
        batch = captimgs_left.shape[0]
        norm_max=torch.zeros(batch).cuda()
        norm_min=torch.zeros(batch).cuda()
        norm_max_m=torch.zeros(batch).cuda()
        norm_min_m=torch.zeros(batch).cuda()

        dataNew = 'DOE_left.mat'

        for i in range(batch):
            norm_max[i], norm_min[i]=depthmaps[i,...].max(),depthmaps[i,...].min()
            norm_max_m[i], norm_min_m[i] = depthmaps_m[i,...].max(), depthmaps_m[i,...].min()

        input_rough=(est/255-norm_min.reshape(-1,1,1,1))/(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))
        input_rough_m=(est_m/255-norm_min_m.reshape(-1,1,1,1))/(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))

        if hparams.warp_img:
            self.warping = Warp()
            b, c, h, w = est.shape
            w_disparity=F.interpolate(est, size=(int(h/hparams.scale), int(w/hparams.scale)), mode='bilinear', align_corners=False)
            w_disparity_2=F.interpolate(self.flip(est_m), size=(int(h/hparams.scale), int(w/hparams.scale)), mode='bilinear', align_corners=False)
            warped_right, mask=self.warping.warp_disp(captimgs_right, w_disparity, w_disparity_2)
            warped_right+=captimgs_left*(1-mask)
            warped_right_m, mask_m=self.warping.warp_disp(captimgs_right_m, self.flip(w_disparity_2), self.flip(w_disparity))
            warped_right_m+=captimgs_left_m*(1-mask_m)
            right=warped_right
            right_m=warped_right_m
        
        else:
            right=captimgs_right
            right_m=captimgs_right_m
            
        Outputs = self.decoder(captimgs_left=captimgs_left.float(),
                                        pinv_volumes_left=pinv_volumes_left.float(),
                                        captimgs_right=right.float(),
                                        rough_depth=input_rough.float(), hparams=hparams)
                    
        Outputs_m = self.decoder(captimgs_left=captimgs_left_m.float(),
                                        pinv_volumes_left=pinv_volumes_left_m.float(),
                                        captimgs_right=right_m.float(),
                                        rough_depth=input_rough_m.float(), hparams=hparams)
        left=Outputs[0]
        est_dfd=Outputs[1]
        est_depthmaps = Outputs[2]
        est_dfd=est_dfd*(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))+norm_min.reshape(-1,1,1,1)
        est_depthmaps=est_depthmaps*(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))+norm_min.reshape(-1,1,1,1)

        left_m= Outputs_m[0]
        est_dfd_m=Outputs_m[1]
        est_depthmaps_m = Outputs_m[2]
        est_dfd_m=est_dfd_m*(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))+norm_min_m.reshape(-1,1,1,1)
        est_depthmaps_m=est_depthmaps_m*(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))+norm_min_m.reshape(-1,1,1,1)

        # Require twice cropping because the image formation also crops the boundary.

        target_roughdepth = crop_boundary(depthmaps*255, 2 * self.crop_width)
        target_roughdepth_m = crop_boundary(depthmaps_m*255, 2 * self.crop_width)
        original_depthmaps = crop_boundary(original_depth, 2 * self.crop_width)
        original_depthmaps_m = crop_boundary(original_depth_m, 2 * self.crop_width)
        est_images_left = crop_boundary(left, self.crop_width)
        est_images_left_m = crop_boundary(left_m, self.crop_width)
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
        '''if doe_type=='pixel_wise':
            from optics import camera_left_pw as camera_left
            from optics import camera_right_pw as camera_right'''
        self.camera_left = camera_left.MixedCamera(**camera_recipe, requires_grad=optimize_optics)
        self.camera_right = camera_right.MixedCamera(**camera_recipe_right, requires_grad=optimize_optics)
        self.matching = IGEVStereo(hparams) #CVNet(hparams, requires_grad=True)
        self.decoder = Recovery(hparams, requires_grad=True)
        self.debayer = Debayer3x3()
        self.image_lossfn = Vgg16PerceptualLoss()
        self.image_lossfn2 = torch.nn.L1Loss()  #torch.nn.MSELoss()
        self.depth_lossfn = torch.nn.MSELoss()
        self.depth_lossfn2= torch.nn.L1Loss()#torch.nn.SmoothL1Loss() #torch.nn.MSELoss() #torch.nn.L1Loss() 
        print(self.camera_left)
        
        '''decoder_ckpt = '/mnt/ssd2/liuyuhui/checkpoint/version_108/checkpoints/interrupted_model.ckpt'
        decoder_ckpt = torch.load(decoder_ckpt, map_location=lambda storage, loc: storage)
        decoder_state_dict = {
        key.replace('decoder.', '', 1): value 
        for key, value in decoder_ckpt['state_dict'].items() 
        if key.startswith('decoder.')}
        self.decoder.load_state_dict(decoder_state_dict)

        self.matching = torch.nn.DataParallel(self.matching, device_ids=[0])
        self.matching.load_state_dict(torch.load('/mnt/ssd2/liuyuhui/checkpoint/sceneflow.pth'))
        self.matching = self.matching.module
        
        for param in self.matching.parameters():
            param.requires_grad = False
        for param in self.camera_left.parameters():
            param.requires_grad = False
        for param in self.camera_right.parameters():
            param.requires_grad = False'''
        

    def __combine_loss(self, depth_loss,depth_1_loss, image_loss, psf_loss):
        return self.hparams.depth_loss_weight * depth_loss + \
                self.hparams.depth_1_loss_weight * depth_1_loss + \
               self.hparams.image_loss_weight * image_loss+ \
               self.hparams.psf_loss_weight * psf_loss    
    def __compute_loss(self, outputs):
        
        hparams = self.hparams
        target_depthmaps=outputs.target_depthmaps
        target_images_left=outputs.target_images_left
        target_depthmaps_m=outputs.target_depthmaps_m
        target_images_left_m=outputs.target_images_left_m
        est_images_left = outputs.est_images_left
        est_1=outputs.est_1
        est=outputs.est
        est_depthmaps = outputs.est_depthmaps
        est_dfd=outputs.est_dfd
        target_roughdepth= outputs.target_roughdepth
        # Mirror
        est_images_left_m = outputs.est_images_left_m
        est_1_m=outputs.est_1_m
        est_m=outputs.est_m
        est_depthmaps_m = outputs.est_depthmaps_m
        est_dfd_m=outputs.est_dfd_m
        target_roughdepth_m= outputs.target_roughdepth_m

        psnr_left = calculate_psnr(est_images_left, target_images_left)
        ssmi_left = calculate_ssim(est_images_left, target_images_left)
        norm_max, norm_min = outputs.norm_max.reshape(-1,1,1,1), outputs.norm_min.reshape(-1,1,1,1)
        norm_max_m, norm_min_m = outputs.norm_max_m.reshape(-1,1,1,1), outputs.norm_min_m.reshape(-1,1,1,1)
        left_image_loss = self.image_lossfn.train_loss(est_images_left, target_images_left)#+self.image_lossfn2(est_images_left, target_images_left)
        left_image_loss_m = self.image_lossfn.train_loss(est_images_left_m, target_images_left_m)#+self.image_lossfn2(est_images_left_m, target_images_left_m)
        #left_image_loss = self.image_lossfn(est_images_left, target_images_left)
        
        #mag = torch.sum(target_roughdepth**2, dim=1).sqrt().unsqueeze(1)
        valid = ((target_roughdepth >= 0.5) & (target_roughdepth < hparams.max_disp))
        valid_m = ((target_roughdepth_m >= 0.5) & (target_roughdepth_m < hparams.max_disp))
        disp_loss = 0.0
        disp_loss_m = 0.0
        for i in range(hparams.train_iters):
            est_s = outputs.est_sq[i]
            est_s_m = outputs.est_sq_m[i]
            loss_gamma = 0.9
            adjusted_loss_gamma = loss_gamma**(15/(hparams.train_iters - 1))
            i_weight = adjusted_loss_gamma**(hparams.train_iters - i - 1)
            i_loss = (target_roughdepth - est_s).abs()
            i_loss_m = (target_roughdepth_m - est_s_m).abs()
            disp_loss += i_weight * i_loss[valid.bool()].mean()
            disp_loss_m += i_weight * i_loss_m[valid_m.bool()].mean()

        disp_loss/=hparams.train_iters
        disp_loss_m/=hparams.train_iters
        est_norm=est/255
        target_norm=target_roughdepth/255
        depth_1_loss=disp_loss+mae(est_1[valid.bool()], target_roughdepth[valid.bool()])#+self.depth_lossfn2(est_1, target_roughdepth)
        depth_2_loss=mae(est[valid.bool()], target_roughdepth[valid.bool()])#+self.depth_lossfn2(est, target_roughdepth)
        depth_2_loss_all=mae(est, target_roughdepth)
        #depth_2_loss=self.depth_lossfn2(est, target_roughdepth)
        depth_1_loss_m=disp_loss_m+mae(est_1_m[valid_m.bool()], target_roughdepth_m[valid_m.bool()])#+self.depth_lossfn2(est_1_m, target_roughdepth_m)
        #depth_2_loss_m=mae(est_m[valid_m.bool()], target_roughdepth_m[valid_m.bool()])#+self.depth_lossfn2(est_m, target_roughdepth_m)
        depth_2_loss_m=mae(est_m, target_roughdepth_m)
        dfd_loss=mae(est_dfd, target_depthmaps)#+ self.depth_lossfn(est_dfd, target_depthmaps)
        dfd_loss_m=mae(est_dfd_m, target_depthmaps_m)#+ self.depth_lossfn(est_dfd_m, target_depthmaps_m)
        depth_loss = mae(est_depthmaps, target_depthmaps)#+ self.depth_lossfn(est_depthmaps, target_depthmaps) #0.5*self.depth_lossfn.train_loss(est_depthmaps, target_depthmaps)+self.depth_lossfn2(est_depthmaps, target_depthmaps)
        
        px_3=calculate_3px(255*est_depthmaps,255*target_depthmaps)
        epe_loss = mae(255*est_depthmaps,255*target_depthmaps)#+ self.depth_lossfn(est_depthmaps, target_depthmaps) #0.5*self.depth_lossfn.train_loss(est_depthmaps, target_depthmaps)+self.depth_lossfn2(est_depthmaps, target_depthmaps)
        epe_loss_m = mae(255*est_depthmaps_m,255*target_depthmaps_m)
        #print(epe_loss)
        
        depth_loss_m= mae(est_depthmaps_m, target_depthmaps_m)#+ self.depth_lossfn(est_depthmaps_m, target_depthmaps_m) #0.5*self.depth_lossfn.train_loss(est_depthmaps_m, target_depthmaps_m)+self.depth_lossfn2(est_depthmaps_m, target_depthmaps_m)
        psf_left_out_of_fov_sum = self.camera_left.psf_out_of_fov_energy(hparams.psf_size)
        psf_left_loss = psf_left_out_of_fov_sum

        psf_right_out_of_fov_sum = self.camera_right.psf_out_of_fov_energy(hparams.psf_size)
        psf_right_loss = psf_right_out_of_fov_sum
        total_loss = self.__combine_loss((epe_loss+epe_loss_m+dfd_loss+dfd_loss_m)/2, (depth_2_loss+depth_2_loss_m)/2+(depth_1_loss+depth_1_loss_m)/4, left_image_loss+left_image_loss_m, psf_left_loss+psf_right_loss)
        logs = {
            'total_loss': total_loss,#'delta': delta,
            'depth_loss': depth_loss,#'dfd_loss': dfd_loss,
            'disp_loss': depth_2_loss, 
            'disp_loss_all': depth_2_loss_all, 
            'left image_loss': left_image_loss,
            'psf_loss_left':psf_left_loss,
            'psf_loss_right':psf_right_loss,
            'left_image_psnr': psnr_left, #'left_image_psnr_mirror': psnr_left_m,
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
        #captimgs_right = outputs.captimgs_right
        est_depthmaps = outputs.est_depthmaps
        
        est = outputs.est/255#-outputs.est.min())/(outputs.est.max()-outputs.est.min())
        target_roughdepth= outputs.target_roughdepth/255
        target_depthmaps=outputs.target_depthmaps#*(outputs.norm_max.reshape(-1,1,1,1)-outputs.norm_min.reshape(-1,1,1,1))+outputs.norm_min.reshape(-1,1,1,1)

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

        #torchvision.utils.save_image(outputs.target_roughdepth[[0],...]/255,'dis.jpg')
        #torchvision.utils.save_image(outputs.target_roughdepth_m[[0],...]/255,'dis2.jpg')
        #torchvision.utils.save_image(target_images_left[[0],...],'cap.jpg')
        #torchvision.utils.save_image(outputs.target_depthmaps[[0],...],'dep.jpg')

        #pinv_volumes_left=outputs.pinv_volumes_left
        #pinv_volumes_left=(pinv_volumes_left-pinv_volumes_left.min())/(pinv_volumes_left.max()-pinv_volumes_left.min())
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

            size=(50,50)
            #size=(256,256)#self.hparams.image_sz
            # PSF and heightmap is not visualized at computed size.
            psf_left = self.camera_left.psf_at_camera(size=size, is_training=torch.tensor(False), modulate_phase=self.hparams.optimize_optics)
            #psf_left = self.camera_left.normalize_psf(psf_left)
            #psf_left=crop_psf(psf_left, 100)
            #psf_left = fftshift(psf_left, dims=(-1, -2))
            #psf_left /= psf_left.max()
            

            '''c,d,h,w=psf_left.shape
            kernel=torch.tensor(cv2.getGaussianKernel(3, 1)).cuda()
            kernel_2d=(kernel*kernel.T)#.reshape(1,1,3,3)
            #psf=psf.view(b*c,d,h,w)
            for i in range(c):
                for j in range(d):
                    psf_left[[i],[j],...]=F.conv2d(psf_left[[i],[j],...], kernel_2d.expand(1,1,3,3), padding=(1,1))
                #psf_left[i,...]=F.conv2d(psf_left[i,...], kernel_2d.expand(5,5,3,3), padding=(1,1))
            #psf=psf.view(b,c,d,h,w)
            '''
            #phasemap_left_1 = imresize(self.camera_left.height()[None, None, ...],
                                 #[self.hparams.summary_mask_sz, self.hparams.summary_mask_sz]).squeeze(0)
            
            phasemap_left_1 = imresize(self.camera_left.phase()[[1], :, :,:],
                                 [self.hparams.summary_mask_sz, self.hparams.summary_mask_sz]).squeeze(0)
            
            #sorted, _ = torch.sort(heightmap_left.view(-1))
            #heightmap_left = torch.where(heightmap_left == heightmap_left.min(), sorted[-2], heightmap_left)
            #heightmap_left -= heightmap_left.min()
            #heightmap_left /= heightmap_left.max()

            sorted_0, _ = torch.sort(phasemap_left_1.view(-1))
            phasemap_left_1 = torch.where(phasemap_left_1 == phasemap_left_1.min(), sorted_0[-2], phasemap_left_1)
            phasemap_left_1 -= phasemap_left_1.min()
            phasemap_left_1 /= phasemap_left_1.max()

            

            #self.logger.experiment.add_image('optics/heightmap_left', heightmap_left, self.global_step)
            self.logger.experiment.add_image('optics/phasemap_left_G', phasemap_left_1, self.global_step)
            psf_left= psf_left.flip(1)
            grid_psf_left = torchvision.utils.make_grid(psf_left.transpose(0, 1),
                                                   nrow=9, pad_value=1, normalize=False)
            self.logger.experiment.add_image('optics/psf_left', grid_psf_left, self.global_step)
            
            psf_left /= psf_left.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=0, keepdim=True)[0]
            
            grid_psf_left = torchvision.utils.make_grid(psf_left.transpose(0, 1),
                                                   nrow=9, pad_value=1, normalize=False)
            self.logger.experiment.add_image('optics/psf_stretched_left', grid_psf_left, self.global_step)

           

            #psf_right=outputs.psf_right[[0],:,:,:,:].squeeze(0)
            psf_right = self.camera_right.psf_at_camera(size=size , is_training=torch.tensor(False),modulate_phase=self.hparams.optimize_optics)
            #psf_right = self.camera_right.normalize_psf(psf_right)
            #psf_right=crop_psf(psf_right, 100)
            #psf_right = fftshift(psf_right, dims=(-1, -2))
            #psf_right /= psf_right.max()

            #phasemap_right_1 = imresize(self.camera_right.height()[None, None, ...],
                                 #[self.hparams.summary_mask_sz, self.hparams.summary_mask_sz]).squeeze(0)
            
            phasemap_right_1 = imresize(self.camera_right.phase()[[1], :, :,:],
                                 [self.hparams.summary_mask_sz, self.hparams.summary_mask_sz]).squeeze(0)
            
            #sorted_r, _ = torch.sort(heightmap_right.view(-1))
            #eightmap_right = torch.where(heightmap_right == heightmap_right.min(), sorted_r[-2], heightmap_right)
            #heightmap_right -= heightmap_right.min()
            #heightmap_right /= heightmap_right.max()

            sorted_0_r, _ = torch.sort(phasemap_right_1.view(-1))
            phasemap_right_1 = torch.where(phasemap_right_1 == phasemap_right_1.min(), sorted_0_r[-2], phasemap_right_1)
            phasemap_right_1 -= phasemap_right_1.min()
            phasemap_right_1 /= phasemap_right_1.max()
            '''save_left=F.interpolate(phasemap_left_1.unsqueeze(0), size=(630, 630), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
            save_right=F.interpolate(phasemap_right_1.unsqueeze(0), size=(630, 630), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
            torchvision.utils.save_image(save_left,'doe_left.png')
            torchvision.utils.save_image(save_right,'doe_right.png')
            import scipy.io as scio
            dataNew1 = 'doe_left.mat'
            save_left= save_left*2 * torch.pi
            save_left= save_left.cpu().numpy()
            scio.savemat(dataNew1,{'phase':save_left,'scale':7,'wavelength': '550nm, N=4', 'aperture_type': 'rectangular'})
            save_right= save_right*2 * torch.pi
            save_right= save_right.cpu().numpy()
            print(save_right.shape, save_right)
            dataNew2 = 'doe_right.mat'
            scio.savemat(dataNew2,{'phase':save_right,'scale':7,'wavelength': '550nm, N=4', 'aperture_type': 'rectangular'})

            print(scio.loadmat('doe_left.mat'))'''
            #self.logger.experiment.add_image('optics/heightmap_right', heightmap_right, self.global_step)
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
        Specify the hyperparams for this LightningModule
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # logger parameters
        parser.add_argument('--summary_max_images', type=int, default=8)
        parser.add_argument('--summary_image_sz', type=int, default=200)#256)
        parser.add_argument('--summary_mask_sz', type=int, default=1260)#256)
        parser.add_argument('--summary_depth_every', type=int, default=2000)
        parser.add_argument('--summary_track_train_every', type=int, default=500) #1000)

        # training parameters
        parser.add_argument('--cnn_lr', type=float, default=0.5e-3)#0.5e-3)
        parser.add_argument('--depth_lr', type=float, default=1e-5)
        parser.add_argument('--optics_lr', type=float, default=0)#0.1e-3)#2e-2)#1e-3)#=0.5e-3
        parser.add_argument('--batch_sz', type=int, default=1)#10) #6
        #parser.add_argument('--control_num', type=int, default=5)z
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--augment', default=True, action='store_true')
        
        # loss parameters
        parser.add_argument('--depth_loss_weight', type=float, default=1)
        parser.add_argument('--depth_1_loss_weight', type=float, default=1)#0.5)
        parser.add_argument('--image_loss_weight', type=float, default=5)
        parser.add_argument('--psf_loss_weight', type=float, default=0)
        parser.add_argument('--psf_size', type=int, default=160)

        # dataset parameters
        parser.add_argument('--image_sz', type=list, default= [320, 736])#[128,384])##[128,256]) #[160,160])[160,448]
        parser.add_argument('--n_depths', type=int, default=7)
        parser.add_argument('--min_depth', type=float, default=0.67) #0.67)
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
        parser.add_argument('--mask_sz', type=int, default=1260) #1260) 
        
        parser.add_argument('--focal_length', type=float, default=35e-3)
        parser.add_argument('--focal_depth', type=float, default=1.23) #1.78) #1.7) 1.23
        parser.add_argument('--focal_depth_right', type=float, default=1.23) # 3.7) #1.7) 1.23
        parser.add_argument('--mask_pitch', type=float, default=3.45e-6)#3.5e-6) #3.6e-6)#4.5)
        parser.add_argument('--mask_diameter', type=float, default=4.347e-3)#4.41e-3) # 4.76e-3) #4.86e-3) #0.0036) 4.221
        parser.add_argument('--camera_pixel_pitch', type=float, default=5.86e-6)#5.86e-6) #6.45e-6)
        parser.add_argument('--noise_sigma_min', type=float, default=0.001)
        parser.add_argument('--noise_sigma_max', type=float, default=0.005)
        parser.add_argument('--full_size', type=int, default=1200)
        parser.add_argument('--mask_upsample_factor', type=int, default=2)
        parser.add_argument('--diffraction_efficiency', type=float, default=0.7)
        parser.add_argument('--scale', type=float, default=1)#1.5)

        parser.add_argument('--bayer', dest='bayer', action='store_true')
        parser.add_argument('--no-bayer', dest='bayer', action='store_false')
        parser.set_defaults(bayer=True)
        parser.add_argument('--occlusion', dest='occlusion', action='store_true')
        parser.add_argument('--no-occlusion', dest='occlusion', action='store_false')
        parser.set_defaults(occlusion=True)
        parser.add_argument('--optimize_optics', dest='optimize_optics', action='store_true')
        parser.add_argument('--no-optimize_optics', dest='optimize_optics', action='store_false')
        parser.set_defaults(optimize_optics=False)
        parser.add_argument('--doe_type', type=str, default='rank2', help="doe modeling method")
        
        # model parameters
        parser.add_argument('--psfjitter', dest='psf_jitter', action='store_true')
        parser.add_argument('--no-psfjitter', dest='psf_jitter', action='store_false')
        parser.set_defaults(psf_jitter=False)

        ###IGEV
        parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
        parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
        parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
        parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

        # Validation parameters
        parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

        # Architecure choices
        parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
        parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
        parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
        parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
        parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
        parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
        parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
        parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

        # Data augmentation
        parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
        parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
        parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
        parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
        parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
        torch.manual_seed(666)

        return parser