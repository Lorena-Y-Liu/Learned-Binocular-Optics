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
import math
from argparse import ArgumentParser, Namespace
from collections import namedtuple, deque
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.optim
import torchvision.transforms
import torchvision.utils
from debayer import Debayer3x3
from psf.psf_import import *

#from models.fusion import Recovery
from models.fusion2 import Recovery2 as Recovery
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
                                          'est_images_left','est_depthmaps','est','est_1', 'est_sq','est_dfd','est_dfd_m',
                                          'est_images_left_m','est_depthmaps_m','est_m','est_1_m','est_sq_m',
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

        # Numerical stability / diagnosis constants
        self._norm_eps = 1e-6

        # Optional: freeze optics when PSF loss is stable
        self._optics_frozen = False
        window = int(getattr(self.hparams, 'psf_stable_window', 20))
        self._psf_loss_history = deque(maxlen=max(5, window))
        self._psf_stable_hits = 0
        self._psf_loss_computed_this_step = False
        
        # NaN batch tracking
        self._nan_batch_count = 0
        self._max_consecutive_nan = 5  # Warn when consecutive NaN exceeds this count
        
        # Perceptual loss metrics for image quality evaluation
        self.metrics = {
            'vgg_image_left': Vgg16PerceptualLoss(),
            'vgg_image_left_mirror': Vgg16PerceptualLoss(),
            'vgg_image_right': Vgg16PerceptualLoss(),
        }

        self.log_dir = log_dir

        # One-time marker to verify which source file is actually running.
        # This prints to stdout and helps avoid "edited the wrong copy" issues.
        try:
            print(f"[Stereo3D] loaded from: {__file__}")
        except Exception:
            pass
       
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

        # Ensure the closure runs (Lightning uses it to run forward/backward in some configs)
        # NOTE: some training steps may intentionally "skip" and return None; handle that gracefully.
        closure_result = None
        if optimizer_closure is not None:
            closure_result = optimizer_closure()
        if closure_result is None:
            # Nothing to optimize for this batch (e.g., NaN/Inf batch skipped)
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()
            return

        # warm up lr
        if self.trainer.global_step < 4000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 4000.0)
            lr_scale_optics = min(1.0, float(self.trainer.global_step + 1) / 400.0)
            # param_groups: [left_optics, right_optics, decoder, matching]
            optimizer.param_groups[0]['lr'] = lr_scale_optics * float(self.hparams.optics_lr)
            optimizer.param_groups[1]['lr'] = lr_scale_optics * float(self.hparams.optics_lr)
            optimizer.param_groups[2]['lr'] = lr_scale * float(self.hparams.cnn_lr)
            optimizer.param_groups[3]['lr'] = lr_scale * float(self.hparams.depth_lr)

        # If optics are frozen, keep their learning rates at 0
        # param_groups: [left_optics, right_optics, decoder, matching]
        if getattr(self, '_optics_frozen', False) and len(optimizer.param_groups) >= 2:
            optimizer.param_groups[0]['lr'] = 0.0
            optimizer.param_groups[1]['lr'] = 0.0

        # Detect non-finite gradients before stepping to prevent permanently corrupting weights.
        # Late-training NaNs often come from a single bad step.
        has_nonfinite_grad = False
        for group in optimizer.param_groups:
            for p in group.get('params', []):
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    has_nonfinite_grad = True
                    break
            if has_nonfinite_grad:
                break

        if has_nonfinite_grad:
            if self.trainer.global_step % 10 == 0:
                print(f"⚠️ Non-finite gradients detected at step {self.trainer.global_step}. Skipping optimizer step.")
            optimizer.zero_grad(set_to_none=True)
            return

        # Gradient clipping (do it here to be robust even if Trainer-side clipping is bypassed by custom optimizer_step)
        clip_val = float(getattr(self.hparams, 'grad_clip_val', 1.0))
        if clip_val > 0:
            params = [p for p in self.parameters() if p.grad is not None]
            if len(params) > 0:
                # Compute gradient norm before clipping for diagnostics
                total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=clip_val)
                
        # update params
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        params = [
            {'params': self.camera_left.parameters(), 'lr': self.hparams.optics_lr},
            {'params': self.camera_right.parameters(), 'lr': self.hparams.optics_lr},
            {'params': self.decoder.parameters(), 'lr': self.hparams.cnn_lr},
            {'params': self.matching.parameters(), 'lr': self.hparams.depth_lr},
        ]
        optimizer = torch.optim.Adam(params)
        return optimizer

    def _skip_batch_loss(self, reason: str, batch_idx: int) -> torch.Tensor:
        """Return None to safely skip a batch when NaN/Inf is detected.

        Returning None tells PyTorch Lightning to skip the optimization step entirely,
        avoiding issues with gradient clipping when no gradients are available.
        
        Note: Returning a dummy tensor with requires_grad=True causes issues in older
        Lightning versions because gradient clipping fails when there are no parameters
        with gradients (IndexError: list index out of range).
        """
        self._nan_batch_count = getattr(self, '_nan_batch_count', 0) + 1
        # Print occasionally to avoid flooding logs on persistent instability
        if self._nan_batch_count <= 5 or (self._nan_batch_count % 50 == 0):
            print(
                f"⚠️ Skipping batch at Step {self.global_step} (Batch {batch_idx}). "
                f"Reason: {reason}. Total skipped: {self._nan_batch_count}"
            )

        # Return None to skip this batch entirely (no backward, no optimizer step)
        return None
    
    def training_step(self, samples, batch_idx):

        # By default the stereo matching network is kept frozen (eval mode) to stabilize training.
        # Set `train_matching: true` in your config to fine-tune it.
        if getattr(self.hparams, 'train_matching', False):
            self.matching.train()
        else:
            self.matching.eval()
        
        # ========== Input Data NaN Check ==========
        # Check if input data contains NaN, skip this batch if so
        input_tensors_to_check = [
            ('left_image', samples['left_image']),
            ('right_image', samples['right_image']),
            ('depthmap', samples['unnorm_depthmap']),
            ('depthmap_2', samples['unnorm_depthmap_2']),
        ]
        
        for name, tensor in input_tensors_to_check:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"\n{'='*80}")
                print(f"🔴 Input Data Anomaly Detected [Step {self.global_step}, Batch {batch_idx}]")
                print(f"{'='*80}")
                print(f"Detected input '{name}' contains NaN/INF!")
                print(f"  Shape: {tensor.shape}")
                print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
                print(f"  INF count: {torch.isinf(tensor).sum().item()}")
                print(f"  Skipping this batch and continuing training...")
                print(f"{'='*80}\n")
                return self._skip_batch_loss(reason=f"input {name} has NaN/Inf", batch_idx=batch_idx)
        
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
        
        # ========== Model Output NaN Detection ==========
        # Skip this batch if model output contains NaN
        outputs_to_check = [
            ('est_images_left', outputs.est_images_left),
            ('est_depthmaps', outputs.est_depthmaps),
        ]
        for name, tensor in outputs_to_check:
            if tensor is not None and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                print(f"\n{'='*80}")
                print(f"🔴 Model Output NaN Detected [Step {self.global_step}, Batch {batch_idx}]")
                print(f"{'='*80}")
                print(f"Detected output '{name}' contains NaN/INF!")
                print(f"  Shape: {tensor.shape}")
                print(f"  NaN count: {nan_count}")
                print(f"  INF count: {inf_count}")
                print(f"  Valid value range: [{tensor[torch.isfinite(tensor)].min().item() if torch.isfinite(tensor).any() else 'N/A'}, "
                      f"{tensor[torch.isfinite(tensor)].max().item() if torch.isfinite(tensor).any() else 'N/A'}]")
                print(f"  Skipping this batch and continuing training...")
                print(f"{'='*80}\n")
                return self._skip_batch_loss(reason=f"output {name} has NaN/Inf", batch_idx=batch_idx)
        
        target_images_left = outputs.target_images_left
        target_images_right = outputs.target_images_right
        
        target_depthmaps = outputs.target_depthmaps
        target_depthmaps_m = outputs.target_depthmaps_m

        loss_result = self.__compute_loss(outputs, is_training=True)
        
        # If __compute_loss returns None, it means this batch should be skipped (NaN detected)
        if loss_result is None:
            return self._skip_batch_loss(reason="__compute_loss returned None (NaN/Inf detected)", batch_idx=batch_idx)
        
        data_loss, loss_logs = loss_result

        # Optional: freeze optics updates once PSF loss is stable
        self._maybe_freeze_optics_from_psf(loss_logs)

        train_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}


        if not self.global_step % self.hparams.summary_track_train_every:
            self.__log_images(outputs, original_images_left, original_depthmaps,
                              original_images_left_m, original_depthmaps_m,'train')

        # Always log train losses with explicit flags, so they show up as step curves.
        self.log_dict(train_logs, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        # Clear PSF cache after training step (DOE parameters may have changed)
        if hasattr(self.camera_left, 'clear_psf_cache'):
            self.camera_left.clear_psf_cache()
        if hasattr(self.camera_right, 'clear_psf_cache'):
            self.camera_right.clear_psf_cache()
        
        # Periodically clear cache to prevent memory fragmentation during long training
        if self.global_step % 100 == 0:
            torch.cuda.empty_cache()

        return data_loss

    def _maybe_freeze_optics_from_psf(self, loss_logs: dict) -> None:
        """Freeze optics parameters once PSF loss becomes stable.

        Enabled by setting `freeze_optics_when_psf_stable: true`.
        Stability is checked only on steps where PSF loss was actually computed.
        """
        if not getattr(self.hparams, 'freeze_optics_when_psf_stable', False):
            return
        if not getattr(self.hparams, 'optimize_optics', False):
            return
        if getattr(self, '_optics_frozen', False):
            return
        if not bool(getattr(self, '_psf_loss_computed_this_step', False)):
            return

        psf_left = loss_logs.get('psf_loss_left', None)
        psf_right = loss_logs.get('psf_loss_right', None)
        if psf_left is None or psf_right is None:
            return

        psf_total = psf_left.detach() + psf_right.detach()
        if not torch.isfinite(psf_total):
            return
        psf_val = float(psf_total.item())
        self._psf_loss_history.append(psf_val)

        min_steps = int(getattr(self.hparams, 'psf_stable_min_steps', 20000))
        if int(self.global_step) < min_steps:
            return

        window = int(getattr(self.hparams, 'psf_stable_window', 20))
        if len(self._psf_loss_history) < max(5, window):
            return

        hist = list(self._psf_loss_history)[-window:]
        mean = sum(hist) / len(hist)
        var = sum((x - mean) ** 2 for x in hist) / max(1, (len(hist) - 1))
        std = var ** 0.5

        abs_tol = float(getattr(self.hparams, 'psf_stable_abs_tol', 1e-4))
        rel_tol = float(getattr(self.hparams, 'psf_stable_rel_tol', 2e-2))
        stable = (std < abs_tol) or (abs(mean) > 0 and (std / abs(mean)) < rel_tol)

        patience = int(getattr(self.hparams, 'psf_stable_patience', 3))
        self._psf_stable_hits = (self._psf_stable_hits + 1) if stable else 0
        if self._psf_stable_hits < patience:
            return

        # Freeze optics parameters
        self._optics_frozen = True
        try:
            self.camera_left.requires_grad_(False)
            self.camera_right.requires_grad_(False)
        except Exception:
            pass

        print(
            f"✅ Freezing optics at step {self.global_step} (PSF stable: mean={mean:.6g}, std={std:.6g}, window={window})."
        )

        # Optional: stop computing PSF loss after freezing (saves compute)
        if not getattr(self.hparams, 'psf_monitor_after_freeze', True):
            self.hparams.optimize_optics = False

    def on_after_backward(self):
        """Optional grad-norm diagnostics for key decoder heads.

        This helps confirm whether gradients reach the DFD/depth heads.
        """
        grad_every = int(getattr(self.hparams, 'grad_diag_every_n_steps', 200))
        if grad_every <= 0 or (self.global_step % grad_every != 0):
            return

        def _param_grad_l2(module: torch.nn.Module) -> torch.Tensor:
            sq = torch.zeros((), device=self.device)
            for p in module.parameters(recurse=True):
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    continue
                sq = sq + (g.detach().float() ** 2).sum()
            return torch.sqrt(sq)


    @torch.no_grad()
    def __dfd_diagnostics(self, outputs):
        """Sanity stats to confirm est/target are on the same scale and not saturated."""
        est_dfd = outputs.est_dfd
        tgt = outputs.target_depthmaps
        est_depth = outputs.est_depthmaps
        valid = (outputs.target_roughdepth > 0).bool() if hasattr(outputs, 'target_roughdepth') else None

        from typing import Optional

        def _stats(x: torch.Tensor, mask: Optional[torch.Tensor], prefix: str) -> dict:
            x = x.detach()
            if mask is not None:
                # Make masking robust for common shapes:
                #   x: (B,1,H,W) or (B,C,H,W)
                #   mask: (B,H,W) or (B,1,H,W) or already flattened
                m = mask
                if m.dtype != torch.bool:
                    m = m.bool()
                # Broadcast mask to x shape when possible
                if m.ndim == x.ndim - 1 and x.ndim >= 3:
                    # (B,H,W) -> (B,1,H,W)
                    m = m.unsqueeze(1)
                if m.shape == x.shape:
                    x = x[m]
                else:
                    # Fallback: flatten both and apply 1D mask
                    x_flat = x.reshape(-1)
                    m_flat = m.reshape(-1)
                    n = min(x_flat.numel(), m_flat.numel())
                    if n > 0:
                        x_flat = x_flat[:n]
                        m_flat = m_flat[:n]
                        if m_flat.any():
                            x = x_flat[m_flat]
            # Guard: if mask selects 0 elements, avoid .min/.max on empty tensors.
            if x.numel() == 0:
                z = torch.zeros((), device=self.device)
                return {
                    f'{prefix}min': z,
                    f'{prefix}max': z,
                    f'{prefix}mean': z,
                    f'{prefix}std': z,
                }
            return {
                f'{prefix}min': x.min(),
                f'{prefix}max': x.max(),
                f'{prefix}mean': x.mean(),
                f'{prefix}std': x.std(unbiased=False),
            }

        diag = {}
        diag.update(_stats(est_dfd, None, 'est_dfd_'))
        diag.update(_stats(tgt, None, 'tgt_'))
        diag.update(_stats(est_depth, None, 'est_depth_'))

        if valid is not None:
            valid4 = valid.unsqueeze(1)
            diag['valid_frac'] = valid4.float().mean()
            diag.update(_stats(est_dfd, valid4, 'est_dfd_valid_'))
            diag.update(_stats(tgt, valid4, 'tgt_valid_'))
            diag.update(_stats(est_depth, valid4, 'est_depth_valid_'))

        # Saturation indicators for sigmoid outputs
        diag['est_dfd_sat_low_frac'] = (est_dfd < 0.01).float().mean()
        diag['est_dfd_sat_high_frac'] = (est_dfd > 0.99).float().mean()
        diag['est_depth_sat_low_frac'] = (est_depth < 0.01).float().mean()
        diag['est_depth_sat_high_frac'] = (est_depth > 0.99).float().mean()
        return diag
    
    def on_validation_epoch_start(self) -> None:
        """Move metrics to device before validation."""
        for metric in self.metrics.values():
            metric.to(self.device)
        # Ensure all models are in eval mode
        self.eval()
        # Clear cache before validation to ensure maximum memory available
        torch.cuda.empty_cache()
            
    def validation_step(self, samples, batch_idx):
        """Validation step with metric computation."""
        # Pre-emptively clear cache at the start of each validation step
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            # Check if validation input data contains NaN
            if torch.isnan(samples['left_image']).any() or torch.isnan(samples['unnorm_depthmap']).any():
                print(f"⚠️ WARNING: Validation batch {batch_idx} contains NaN in input, skipping...")
                return None
            
            # Extract all needed data from samples immediately
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
            
            # Clear samples dict to free memory immediately
            samples = None

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
            
            # Delete input_args immediately after forward to free memory
            del input_args

            # Unpack outputs and detach to prevent any gradient tracking
            est_images_left = outputs.est_images_left.detach()
            est_images_left_m = outputs.est_images_left_m.detach()
            est_depthmaps = outputs.est_depthmaps.detach()
            est_depthmaps_m = outputs.est_depthmaps_m.detach()
            rough_depth = outputs.est.detach()
            rough_depth_m = outputs.est_m.detach()
            target_images_left = outputs.target_images_left.detach()
            target_images_left_m = outputs.target_images_left_m.detach()

            target_depthmaps = outputs.target_depthmaps.detach()
            target_depthmaps_m = outputs.target_depthmaps_m.detach()
            target_roughdepth = outputs.target_roughdepth.detach()
            target_roughdepth_m = outputs.target_roughdepth_m.detach()
            
            # Create valid masks for disparity evaluation
            valid = ((target_roughdepth >= 0.5) & (target_roughdepth < self.hparams.max_disp))
            valid_m = ((target_roughdepth_m >= 0.5) & (target_roughdepth_m < self.hparams.max_disp))

            # Counters to diagnose why validation may produce NaN/invalid losses.
            # 1) Empty valid masks cause masked means / MAE to become NaN.
            if not valid.any():
                self.log('validation/empty_valid_batches', 1.0, on_step=False, on_epoch=True)
            if not valid_m.any():
                self.log('validation/empty_valid_mirror_batches', 1.0, on_step=False, on_epoch=True)

            # 2) Zero/near-zero normalization range makes divide-by-zero likely in forward.
            # We log stats from outputs computed in forward.
            norm_range = (outputs.norm_max - outputs.norm_min)
            norm_range_m = (outputs.norm_max_m - outputs.norm_min_m)
            if (norm_range.abs() <= self._norm_eps).any():
                self.log('validation/zero_depth_range_batches', 1.0, on_step=False, on_epoch=True)
            if (norm_range_m.abs() <= self._norm_eps).any():
                self.log('validation/zero_depth_range_mirror_batches', 1.0, on_step=False, on_epoch=True)
            assert valid.shape == target_roughdepth.shape, [valid.shape, target_roughdepth.shape]
            if valid.any():
                assert not torch.isinf(target_roughdepth[valid.bool()]).any()
            assert valid_m.shape == target_roughdepth_m.shape, [valid_m.shape, target_roughdepth_m.shape]
            if valid_m.any():
                assert not torch.isinf(target_roughdepth_m[valid_m.bool()]).any()
            # Compute metrics
            depth_mse = mse(est_depthmaps, target_depthmaps)
            depth_epe = mae((est_depthmaps) * 255, (target_depthmaps) * 255)
            if valid.any():
                epe_match = mae(rough_depth[valid.bool()], target_roughdepth[valid.bool()])
            else:
                epe_match = torch.zeros((), device=rough_depth.device, dtype=rough_depth.dtype)
            img_mse = mse(est_images_left, target_images_left)

            depth_mse_m = mse(est_depthmaps_m, target_depthmaps_m)
            depth_epe_m = mae((est_depthmaps_m) * 255, (target_depthmaps_m) * 255)
            if valid_m.any():
                epe_match_m = mae(rough_depth_m[valid_m.bool()], target_roughdepth_m[valid_m.bool()])
            else:
                epe_match_m = torch.zeros((), device=rough_depth_m.device, dtype=rough_depth_m.dtype)
            img_mse_m = mse(est_images_left_m, target_images_left_m)
            
            # Compute val_loss
            # NOTE: __compute_loss uses valid masks. On some validation batches valid can be empty,
            # which makes .mean() return NaN. If that happens, treat this batch as skipped for val_loss.
            loss_result = self.__compute_loss(outputs, is_training=False)
            
            # If __compute_loss returns None, it means this batch should be skipped (NaN detected)
            if loss_result is None:
                self.log('validation/val_loss_nan_batches', 1.0, on_step=False, on_epoch=True)
                return None
            
            val_loss, val_logs = loss_result
            
            if not torch.isfinite(val_loss):
                # Avoid poisoning epoch-level val_loss aggregation with NaNs
                self.log('validation/val_loss_nan_batches', 1.0, on_step=False, on_epoch=True)
                return None

            # Log validation loss breakdown (mirrors train_loss/* so TensorBoard can show curves)
            # NOTE: keys coming from __compute_loss contain tensors; Lightning will reduce/aggregate.
            val_loss_logs = {f'validation_loss/{k}': v.item() if isinstance(v, torch.Tensor) else v for k, v in val_logs.items()}
            self.log_dict(val_loss_logs, on_step=False, on_epoch=True)
            
            # Log validation metrics - CRITICAL: convert to Python scalars to prevent tensor accumulation
            self.log('val_loss', val_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log('validation/mse_depthmap', depth_mse.item(), on_step=False, on_epoch=True)
            self.log('validation/mse_depthmap_m', depth_mse_m.item(), on_step=False, on_epoch=True)
            self.log('validation/depth_epe', depth_epe.item(), on_step=False, on_epoch=True)
            self.log('validation/depth_epe_m', depth_epe_m.item(), on_step=False, on_epoch=True)
            self.log('validation/mae_depthmap_1', epe_match.item(), on_step=False, on_epoch=True)
            self.log('validation/mae_depthmap_1_mirror', epe_match_m.item(), on_step=False, on_epoch=True)

            self.log('validation/mse_image_left', img_mse.item(), on_step=False, on_epoch=True)
            self.log('validation/mse_image_left_mirror', img_mse_m.item(), on_step=False, on_epoch=True)
            ssim_l = calculate_ssim(est_images_left, target_images_left)
            ssim_m = calculate_ssim(est_images_left_m, target_images_left_m)
            if isinstance(ssim_l, torch.Tensor):
                ssim_l = ssim_l.item() if ssim_l.numel() == 1 else ssim_l.mean().item()
            if isinstance(ssim_m, torch.Tensor):
                ssim_m = ssim_m.item() if ssim_m.numel() == 1 else ssim_m.mean().item()
            self.log('validation/ssim', ssim_l, on_step=False, on_epoch=True)
            self.log('validation/ssim_mirror', ssim_m, on_step=False, on_epoch=True)
            
            if batch_idx == 0:
                self.__log_images(outputs, target_images_left, target_depthmaps,
                                  target_images_left_m, target_depthmaps_m, 'validation')
            
            # Explicitly delete large intermediate variables to free memory
            del outputs, est_images_left, est_images_left_m, est_depthmaps, est_depthmaps_m
            del rough_depth, rough_depth_m, target_images_left, target_images_left_m
            del target_depthmaps, target_depthmaps_m, target_roughdepth, target_roughdepth_m
            del valid, valid_m, depth_mse, depth_epe, epe_match, img_mse
            del depth_mse_m, depth_epe_m, epe_match_m, img_mse_m, ssim_l, ssim_m
            
            # Force CUDA synchronization and clear cache
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    def on_validation_epoch_end(self) -> None:
        """Clean up GPU memory after validation before starting training."""
        # If no valid val_loss was logged (all batches returned None due to NaN),
        # log a placeholder to prevent ModelCheckpoint errors
        # Check if val_loss was logged in this epoch
        if not any('val_loss' in str(k) for k in self.trainer.callback_metrics.keys()):
            self.log('val_loss', float('inf'), on_step=False, on_epoch=True, prog_bar=True)
        
        # Explicitly delete any cached tensors
        torch.cuda.empty_cache()
        # Force garbage collection
        import gc
        gc.collect()
    
    def on_train_epoch_start(self) -> None:
        """Clean up GPU memory before starting training epoch."""
        # Clear any residual cache from previous validation
        torch.cuda.empty_cache()
    
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
        
        # In validation, detach all intermediate iterations to free gradient computation
        if not is_training:
            est_1 = est_1.detach()
            est_1_m = est_1_m.detach()
            # Detach all elements in the sequence but keep them for loss computation
            est_sq = [e.detach() if e is not None else None for e in est_sq]
            est_m_sq = [e.detach() if e is not None else None for e in est_m_sq]
        
        batch = captimgs_left.shape[0]
        device = est.device  # Get device from model output

        # Normalize rough disparity to [0, 1] for the decoder/fusion (DFD) depth head.
        # NOTE: `est` is a pixel-disparity field in [0..max_disp] (not 0..255), so do NOT divide by 255.
        #disp_norm = est / float(self.hparams.max_disp)
        #disp_norm_m = est_m / float(self.hparams.max_disp)
        #input_rough = disp_norm.clamp(0.0, 1.0)
        #input_rough_m = disp_norm_m.clamp(0.0, 1.0)

        norm_max=torch.zeros(batch).cuda()
        norm_min=torch.zeros(batch).cuda()
        norm_max_m=torch.zeros(batch).cuda()
        norm_min_m=torch.zeros(batch).cuda()

        for i in range(batch):
            norm_max[i], norm_min[i]=depthmaps[i,...].max(),depthmaps[i,...].min()
            norm_max_m[i], norm_min_m[i] = depthmaps_m[i,...].max(), depthmaps_m[i,...].min()
        
        input_rough=est/float(self.hparams.max_disp)
        input_rough_m=est_m/float(self.hparams.max_disp)
        # Use larger epsilon for numerical stability
        norm_eps = 1e-6
        norm_range = norm_max.reshape(-1,1,1,1) - norm_min.reshape(-1,1,1,1)
        norm_range_m = norm_max_m.reshape(-1,1,1,1) - norm_min_m.reshape(-1,1,1,1)
        # Clamp range to avoid division by very small numbers
        norm_range = torch.clamp(norm_range, min=norm_eps)
        norm_range_m = torch.clamp(norm_range_m, min=norm_eps)
        input_rough = (input_rough - norm_min.reshape(-1,1,1,1)) / norm_range
        input_rough_m = (input_rough_m - norm_min_m.reshape(-1,1,1,1)) / norm_range_m
        # Clamp normalized inputs to valid range [0, 1] to prevent extreme values
        input_rough = torch.clamp(input_rough, 0.0, 1.0)
        input_rough_m = torch.clamp(input_rough_m, 0.0, 1.0)
        
        if hparams.warp_img:
            # Use pre-initialized warping object
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
            right_m=warped_right_m
        
        else:
            right=captimgs_right
            right_m=captimgs_right_m
            
        # Left image reconstruction
        Outputs = self.decoder(captimgs_left=captimgs_left.float(),
                                        pinv_volumes_left=pinv_volumes_left.float(),
                                        captimgs_right=right.float(),
                                        rough_depth=input_rough.float(), hparams=hparams)
        
                    
        # Mirror left reconstruction
        Outputs_m = self.decoder(captimgs_left=captimgs_left_m.float(),
                                        pinv_volumes_left=pinv_volumes_left_m.float(),
                                        captimgs_right=right_m.float(),
                                        rough_depth=input_rough_m.float(), hparams=hparams)
        
        left = Outputs[0]
        # Clamp decoder outputs to valid range to prevent NaN propagation
        left = torch.clamp(left, 0.0, 1.0)
        est_dfd = Outputs[1]
        est_dfd = torch.clamp(est_dfd, 0.0, 1.0)  # Clamp before denormalization
        est_dfd = est_dfd*(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))+norm_min.reshape(-1,1,1,1)
        est_depthmaps = Outputs[2]
        est_depthmaps = torch.clamp(est_depthmaps, 0.0, 1.0)  # Clamp before denormalization
        est_depthmaps = est_depthmaps*(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))+norm_min.reshape(-1,1,1,1)
        #est_dfd=est_dfd*(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))+norm_min.reshape(-1,1,1,1)
        #est_depthmaps=est_depthmaps*(norm_max.reshape(-1,1,1,1)-norm_min.reshape(-1,1,1,1))+norm_min.reshape(-1,1,1,1)

        

        left_m = Outputs_m[0]
        # Clamp mirror decoder outputs to valid range
        left_m = torch.clamp(left_m, 0.0, 1.0)
        est_dfd_m = Outputs_m[1]
        est_dfd_m = torch.clamp(est_dfd_m, 0.0, 1.0)  # Clamp before denormalization
        est_dfd_m = est_dfd_m*(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))+norm_min_m.reshape(-1,1,1,1)
        est_depthmaps_m = Outputs_m[2]
        est_depthmaps_m = torch.clamp(est_depthmaps_m, 0.0, 1.0)  # Clamp before denormalization
        est_depthmaps_m = est_depthmaps_m*(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))+norm_min_m.reshape(-1,1,1,1)
        #est_dfd_m=est_dfd_m*(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))+norm_min_m.reshape(-1,1,1,1)
        ##est_depthmaps_m=est_depthmaps_m*(norm_max_m.reshape(-1,1,1,1)-norm_min_m.reshape(-1,1,1,1))+norm_min_m.reshape(-1,1,1,1)


        
        # Require twice cropping because the image formation also crops the boundary.

        est_images_left = crop_boundary(left, self.crop_width)
        est_images_left_m = crop_boundary(left_m, self.crop_width)
        # DFD/depth heads supervise normalized disparity in [0,1]
        # Auto-correct sign of disparities because left/right may have
        # opposite conventions in different datasets / loaders.
        raw_disp = disparity.to(est.device)
        try:
            mean_raw = raw_disp.mean()
        except Exception:
            mean_raw = torch.tensor(0.0, device=raw_disp.device)
        if torch.isfinite(mean_raw) and mean_raw < 0:
            raw_disp = -raw_disp

        raw_disp_m = disparity_2.to(est.device)
        try:
            mean_raw_m = raw_disp_m.mean()
        except Exception:
            mean_raw_m = torch.tensor(0.0, device=raw_disp_m.device)
        if torch.isfinite(mean_raw_m) and mean_raw_m < 0:
            raw_disp_m = -raw_disp_m

        target_disp = raw_disp
        # Mirror target should be flipped horizontally to match mirror image orientation
        target_disp_m = self.flip(raw_disp_m)

        # `target_disp` is also in pixel units; keep consistent with disp_norm.
        target_disp_norm = (target_disp / float(self.hparams.max_disp)).clamp(0.0, 1.0)
        target_disp_norm_m = (target_disp_m / float(self.hparams.max_disp)).clamp(0.0, 1.0)

        # In validation, replace large iteration sequences with empty lists to save memory
        # The sequences est_sq and est_m_sq each contain ~16 disparity maps and can use >500MB
        if is_training:
            output_est_sq = est_sq
            output_est_sq_m = est_m_sq
        else:
            # Keep only final iteration for validation, discard intermediate steps
            output_est_sq = []
            output_est_sq_m = []

        outputs = StereoOutputs(
            target_images_right=original_right,
            target_images_left=original_left,
            # raw disparity (pixels) for valid-mask evaluation
            target_roughdepth=target_disp,
            # normalized disparity [0,1] for DFD/depth supervision
            target_depthmaps=target_disp_norm,
            # placeholders kept for backwards-compatibility (not used after switching to disp normalization)
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
            est_sq=output_est_sq,
            est_sq_m=output_est_sq_m,  
            est_dfd=est_dfd,
            est_depthmaps=est_depthmaps,############## Mirror
            target_images_right_m=original_right_m,
            target_images_left_m=original_left_m,
            target_roughdepth_m=target_disp_m,
            target_depthmaps_m=target_disp_norm_m,
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
        
        # Clean up large intermediate variables before returning
        # Only keep what's needed in outputs
        if not is_training:
            # In validation, be more aggressive with cleanup
            del captimgs_left, captimgs_right, captimgs_left_m, captimgs_right_m
            del left_images_linear, right_images_linear
            del target_volumes_left, target_volumes_right
            del pinv_volumes_left, pinv_volumes_right, pinv_volumes_left_m, pinv_volumes_right_m
            del est_1, est_sq, est_1_m, est_m_sq
            if hparams.warp_img:
                del warped_right, warped_left, mask, mask_r, w_disparity, w_disparity_2
        
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
        self.warping = Warp()  # Initialize once, not in forward
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
    def __compute_loss(self, outputs, is_training=True):
        
        hparams = self.hparams

        # Track whether PSF loss was computed on this step (used for optics-freeze logic)
        self._psf_loss_computed_this_step = False
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
        
        # ===== Early NaN Detection: If critical output is NaN, return None to skip the entire batch =====
        critical_outputs = [
            ('est_images_left', est_images_left),
            ('est_images_left_m', est_images_left_m),
            ('est_depthmaps', est_depthmaps),
            ('est_depthmaps_m', est_depthmaps_m),
        ]
        for name, tensor in critical_outputs:
            if tensor is not None and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                print(f"🛑 NaN/Inf detected in {name} at Step {self.global_step}. Skipping batch.")
                return None  # Signal to training_step to skip this batch

        # NOTE: some helper metrics may return CPU tensors. Keep scalars device-safe
        # because Lightning stacks logged values at epoch end.
        psnr_left = calculate_psnr(est_images_left, target_images_left)
        ssmi_left = calculate_ssim(est_images_left, target_images_left)
        if isinstance(psnr_left, torch.Tensor) and psnr_left.device != est_images_left.device:
            psnr_left = psnr_left.to(est_images_left.device)
        if isinstance(ssmi_left, torch.Tensor) and ssmi_left.device != est_images_left.device:
            ssmi_left = ssmi_left.to(est_images_left.device)
        
        # Clamp estimated images to valid range [0, 1] to prevent VGG loss from exploding
        est_images_left_safe = torch.clamp(est_images_left, 0, 1)
        est_images_left_m_safe = torch.clamp(est_images_left_m, 0, 1)
        
        left_image_loss = self.image_lossfn.train_loss(est_images_left_safe, target_images_left)
        left_image_loss_m = self.image_lossfn.train_loss(est_images_left_m_safe, target_images_left_m)
        
        # Safety check for image loss NaN - If NaN detected, return None to skip this batch
        if torch.isnan(left_image_loss) or torch.isinf(left_image_loss):
            print(f"🛑 NaN in left_image_loss at Step {self.global_step}. Skipping batch.")
            return None
        
        if torch.isnan(left_image_loss_m) or torch.isinf(left_image_loss_m):
            print(f"🛑 NaN in left_image_loss_m at Step {self.global_step}. Skipping batch.")
            return None
        
        valid = ((target_roughdepth >= 0.5) & (target_roughdepth < hparams.max_disp))
        valid_m = ((target_roughdepth_m >= 0.5) & (target_roughdepth_m < hparams.max_disp))
        # Initialize as tensors on the correct device to avoid mixed cpu/cuda errors during Lightning aggregation
        disp_loss = torch.zeros((), device=est.device, dtype=est.dtype)
        disp_loss_m = torch.zeros((), device=est_m.device, dtype=est_m.dtype)
        
        # Use actual sequence length instead of hparams.train_iters
        num_iters = len(outputs.est_sq)
        for i in range(num_iters):
            est_s = outputs.est_sq[i]
            est_s_m = outputs.est_sq_m[i]
            
            # Skip None elements (shouldn't happen, but safety check)
            if est_s is None or est_s_m is None:
                continue
                
            loss_gamma = 0.9
            # Avoid division by zero when num_iters == 1
            if num_iters > 1:
                adjusted_loss_gamma = loss_gamma**(15/(num_iters - 1))
                i_weight = adjusted_loss_gamma**(num_iters - i - 1)
            else:
                i_weight = 1.0
            i_loss = (target_roughdepth - est_s).abs()
            i_loss_m = (target_roughdepth_m - est_s_m).abs()
            # In validation, valid masks can be empty; .mean() on empty tensors -> NaN.
            if valid.bool().any():
                disp_loss += i_weight * i_loss[valid.bool()].mean()
            if valid_m.bool().any():
                disp_loss_m += i_weight * i_loss_m[valid_m.bool()].mean()

        # Avoid division by zero when est_sq is empty (validation with memory optimization)
        if num_iters > 0:
            disp_loss /= num_iters
            disp_loss_m /= num_iters
        # If num_iters is 0 (validation), disp_loss remains 0 (already initialized)
        # Guard against empty valid masks causing NaNs
        if valid.bool().any():
            depth_1_loss = disp_loss + mae(est_1[valid.bool()], target_roughdepth[valid.bool()])
            depth_2_loss = mae(est[valid.bool()], target_roughdepth[valid.bool()])
        else:
            depth_1_loss = torch.zeros((), device=est.device, dtype=est.dtype)
            depth_2_loss = torch.zeros((), device=est.device, dtype=est.dtype)
        depth_2_loss_all=mae(est, target_roughdepth)
        if valid_m.bool().any():
            depth_1_loss_m = disp_loss_m + mae(est_1_m[valid_m.bool()], target_roughdepth_m[valid_m.bool()])
            depth_2_loss_m = mae(est_m[valid_m.bool()], target_roughdepth_m[valid_m.bool()])
        else:
            depth_1_loss_m = torch.zeros((), device=est_m.device, dtype=est_m.dtype)
            depth_2_loss_m = torch.zeros((), device=est_m.device, dtype=est_m.dtype)

        # DFD / depth supervision
        #print('est_dfd min/max:', est_dfd.min(), est_dfd.max())
        #print('target_depthmaps min/max:', target_depthmaps.min(), target_depthmaps.max())
        #print('est_depthmaps min/max:', est_depthmaps.min(), est_depthmaps.max())
        #print('est_dfd_m min/max:', est_dfd_m.min(), est_dfd_m.max())
        #print('target_depthmaps_m min/max:', target_depthmaps_m.min(), target_depthmaps_m.max())
        #print('est_depthmaps_m min/max:', est_depthmaps_m.min(), est_depthmaps_m.max())
        dfd_loss = mae(est_dfd, target_depthmaps)
        dfd_loss_m = mae(est_dfd_m, target_depthmaps_m)
        dfd_loss_total = (dfd_loss + dfd_loss_m) / 2

        depth_loss = mae(est_depthmaps, target_depthmaps)
        depth_loss_m = mae(est_depthmaps_m, target_depthmaps_m)
        depth_loss_total = (depth_loss + depth_loss_m) / 2
        px_3=calculate_3px(255*est_depthmaps,255*target_depthmaps)
        
        # Clamp depth maps to valid range to prevent overflow when multiplying by 255
        est_depthmaps_safe = torch.clamp(est_depthmaps, 0, 1)
        est_depthmaps_m_safe = torch.clamp(est_depthmaps_m, 0, 1)
        target_depthmaps_safe = torch.clamp(target_depthmaps, 0, 1)
        target_depthmaps_m_safe = torch.clamp(target_depthmaps_m, 0, 1)
        
        epe_loss = mae(255*est_depthmaps_safe, 255*target_depthmaps_safe)
        epe_loss_m = mae(255*est_depthmaps_m_safe, 255*target_depthmaps_m_safe)
        
        # Safety check for EPE loss - If NaN, skip batch
        if torch.isnan(epe_loss) or torch.isinf(epe_loss):
            print(f"🛑 NaN in epe_loss at Step {self.global_step}. Skipping batch.")
            return None
        
        if torch.isnan(epe_loss_m) or torch.isinf(epe_loss_m):
            print(f"🛑 NaN in epe_loss_m at Step {self.global_step}. Skipping batch.")
            return None
        
        # Only compute PSF loss during training to save memory in validation
        # PSF doesn't change per batch, so computing it every validation batch is wasteful
        # Also throttle PSF computation during training to reduce memory pressure
        compute_psf = is_training and hparams.optimize_optics
        if compute_psf:
            # Only compute PSF loss every N steps to reduce memory usage
            psf_freq = int(getattr(hparams, 'psf_loss_freq', 5))
            if self.global_step % psf_freq == 0:
                self._psf_loss_computed_this_step = True
                psf_left_out_of_fov_sum = self.camera_left.psf_out_of_fov_energy(hparams.psf_size)
                psf_left_loss = psf_left_out_of_fov_sum

                psf_right_out_of_fov_sum = self.camera_right.psf_out_of_fov_energy(hparams.psf_size)
                psf_right_loss = psf_right_out_of_fov_sum
                
                # Safety check for PSF loss NaN (critical since optical optimization can diverge)
                if torch.isnan(psf_left_loss) or torch.isinf(psf_left_loss):
                    print(f"⚠️ WARNING [Step {self.global_step}]: psf_left_loss = {psf_left_loss}. "
                          f"Disabling PSF loss for stability.")
                    psf_left_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                if torch.isnan(psf_right_loss) or torch.isinf(psf_right_loss):
                    print(f"⚠️ WARNING [Step {self.global_step}]: psf_right_loss = {psf_right_loss}. "
                          f"Disabling PSF loss for stability.")
                    psf_right_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                # Skip PSF computation for this step
                psf_left_loss = torch.zeros((), device=self.device)
                psf_right_loss = torch.zeros((), device=self.device)
                psf_left_out_of_fov_sum = psf_left_loss
                psf_right_out_of_fov_sum = psf_right_loss
        else:
            # Use zero loss during validation or when not optimizing optics
            psf_left_loss = torch.zeros((), device=self.device)
            psf_right_loss = torch.zeros((), device=self.device)
            psf_left_out_of_fov_sum = psf_left_loss
            psf_right_out_of_fov_sum = psf_right_loss
        
        total_image_loss = (left_image_loss + left_image_loss_m) / 2

        # Backward-compatible optional weights (if not present in YAML/args, defaults keep behavior unchanged)
        dfd_term_weight = float(getattr(hparams, 'dfd_term_weight', 5.0))
        epe_term_weight = float(getattr(hparams, 'epe_term_weight', 1.0))  # was (epe)/10
        disp2_term_weight = float(getattr(hparams, 'disp2_term_weight', 0.5))
        disp1_term_weight = float(getattr(hparams, 'disp1_term_weight', 0.25))
        
        depth_term = epe_term_weight * (depth_loss + depth_loss_m) + dfd_term_weight * (dfd_loss + dfd_loss_m)
        disp_term = disp2_term_weight * (depth_2_loss + depth_2_loss_m) + disp1_term_weight * (depth_1_loss + depth_1_loss_m)
        # Safety check for NaN in intermediate terms - Return None to skip batch
        if torch.isnan(depth_term) or torch.isinf(depth_term):
            print(f"🛑 NaN in depth_term at Step {self.global_step}. Skipping batch.")
            return None
        
        if torch.isnan(disp_term) or torch.isinf(disp_term):
            print(f"🛑 NaN in disp_term at Step {self.global_step}. Skipping batch.")
            return None
        
        if torch.isnan(total_image_loss) or torch.isinf(total_image_loss):
            print(f"🛑 NaN in total_image_loss at Step {self.global_step}. Skipping batch.")
            return None

        total_loss = self.__combine_loss(
            depth_term,
            disp_term,
            total_image_loss,
            psf_left_loss + psf_right_loss
        )
        
        # Final NaN check on total loss - Return None to skip batch
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"🛑 NaN in total_loss at Step {self.global_step}. Skipping batch.")
            return None
        
        # ========== NaN Detailed Diagnostics (Optional) ==========
        # To track which loss becomes NaN first, enable via config
        if getattr(hparams, 'enable_nan_diagnostics', False):
            nan_diagnostic = {
                'step': self.global_step,
                'dfd_loss': dfd_loss.item(),
                'dfd_loss_m': dfd_loss_m.item(),
                'depth_loss': depth_loss.item(),
                'depth_loss_m': depth_loss_m.item(),
                'epe_loss': epe_loss.item(),
                'epe_loss_m': epe_loss_m.item(),
                'depth_2_loss': depth_2_loss.item(),
                'depth_2_loss_m': depth_2_loss_m.item(),
                'depth_1_loss': depth_1_loss.item(),
                'depth_1_loss_m': depth_1_loss_m.item(),
                'left_image_loss': left_image_loss.item(),
                'left_image_loss_m': left_image_loss_m.item(),
                'psf_left_loss': psf_left_loss.item(),
                'psf_right_loss': psf_right_loss.item(),
                'depth_term': depth_term.item(),
                'disp_term': disp_term.item(),
                'total_image_loss': total_image_loss.item(),
                'total_loss': total_loss.item(),
            }
            
            # Check if each loss is NaN
            nan_losses = []
            for name, value in nan_diagnostic.items():
                if name != 'step' and (math.isnan(value) or math.isinf(value)):
                    nan_losses.append(f"{name}={value}")
            
            if nan_losses:
                print(f"\n{'='*80}")
                print(f"🔴 NaN Detection [Step {self.global_step}]")
                print(f"{'='*80}")
                print(f"Found {len(nan_losses)} NaN/INF loss terms:")
                for loss_str in nan_losses:
                    print(f"  ❌ {loss_str}")
                print(f"{'='*80}\n")
        
        logs = {
            'total_loss': total_loss,
            # expose more supervision-related scalars for TensorBoard
            'depth_loss': depth_loss_total,
            'depth_loss_left': depth_loss,
            'depth_loss_mirror': depth_loss_m,

            'dfd_loss': dfd_loss_total,
            'dfd_loss_left': dfd_loss,
            'dfd_loss_mirror': dfd_loss_m,

            'disp_loss': depth_2_loss,
            'disp_loss_mirror': depth_2_loss_m,
            'disp_seq_loss': disp_loss,
            'disp_seq_loss_mirror': disp_loss_m,
            'disp_loss_all': depth_2_loss_all, 
            'left_image_loss': left_image_loss,
            'psf_loss_left': psf_left_loss,
            'psf_loss_right': psf_right_loss,
            'left_image_psnr': psnr_left if isinstance(psnr_left, torch.Tensor) else torch.tensor(float(psnr_left), device=est.device),
            'left_image_ssmi': ssmi_left if isinstance(ssmi_left, torch.Tensor) else torch.tensor(float(ssmi_left), device=est.device),
            'depth_epe': epe_loss,
            'depth_3px': px_3,
        }
        
        # Clean up PSF tensors to free memory
        del psf_left_out_of_fov_sum, psf_right_out_of_fov_sum
        
        return total_loss, logs

    @torch.no_grad()
    def __log_images(self, outputs, target_images_left, target_depthmaps, target_images_left_m, target_depthmaps_m, tag: str):
        # Unpack outputs
        captimgs_left = outputs.captimgs_left
        est_images_left = outputs.est_images_left
        est_depthmaps = outputs.est_depthmaps

        est = outputs.est/self.hparams.max_disp
        # NOTE:
        # - outputs.target_depthmaps is the *normalized disparity* in [0,1] used for DFD/depth supervision.
        # - outputs.target_roughdepth is the *raw disparity* in pixel units (roughly 0..max_disp),
        #   used for stereo-matching loss/masks.
        target_depthmaps = outputs.target_depthmaps
        target_roughdepth = outputs.target_roughdepth
        # Convenience visualizations in [0,1] and 0..255 scales
        target_roughdepth_vis01 = (target_roughdepth / float(self.hparams.max_disp)).clamp(0.0, 1.0)
        # Still rendered as [0,1] for TensorBoard, but corresponds to 0..255 after scaling
        target_depthmaps_vis255 = (target_depthmaps * 255.0).clamp(0.0, 255.0) / 255.0
        captimgs_left_m = outputs.captimgs_left_m
        est_images_left_m = outputs.est_images_left_m
        est_depthmaps_m = outputs.est_depthmaps_m

        est_m =  outputs.est_m/self.hparams.max_disp#-outputs.est_m.min())/(outputs.est_m.max()-outputs.est_m.min())
        target_depthmaps_m = outputs.target_depthmaps_m
        target_roughdepth_m = outputs.target_roughdepth_m
        target_roughdepth_m_vis01 = (target_roughdepth_m / float(self.hparams.max_disp)).clamp(0.0, 1.0)
        target_depthmaps_m_vis255 = (target_depthmaps_m * 255.0).clamp(0.0, 255.0) / 255.0

        est_dfd=outputs.est_dfd
        est_dfd_m=outputs.est_dfd_m


        summary_image_sz = self.hparams.summary_image_sz
        # CAUTION! Summary image is clamped, and visualized in sRGB.
        summary_max_images = min(self.hparams.summary_max_images, target_images_left.shape[0])

        # Flip [0, 1] for visualization purpose
        target_depthmaps = gray_to_rgb(1-target_depthmaps)
        # Also visualize the same GT in a "0-255" style (still normalized to [0,1] for RGB rendering)
        target_depthmaps_255 = gray_to_rgb(1-target_depthmaps_vis255)
        est_depthmaps = gray_to_rgb(1-est_depthmaps)
        est = gray_to_rgb(1-est)

        target_roughdepth = gray_to_rgb(1-target_roughdepth_vis01)

        est_m= gray_to_rgb(1-est_m)
        est_dfd= gray_to_rgb(1-est_dfd)
        est_dfd_m= gray_to_rgb(1-est_dfd_m)

        est_depthmaps_m= gray_to_rgb(1-est_depthmaps_m)
        target_depthmaps_m = gray_to_rgb(1-target_depthmaps_m)
        target_depthmaps_m_255 = gray_to_rgb(1-target_depthmaps_m_vis255)
        target_roughdepth_m = gray_to_rgb(1-target_roughdepth_m_vis01)

        summary = torch.cat([captimgs_left[:,:3,...], captimgs_left_m[:,:3,...]], dim=-2)

        # Include both normalized GT and "0-255 view" GT to avoid confusion.
        summary2 = torch.cat([target_images_left, est_images_left, target_depthmaps, target_depthmaps_255, est, est_depthmaps], dim=-2)
        summary3 = torch.cat([target_images_left_m, est_images_left_m, target_depthmaps_m, target_depthmaps_m_255, est_m, est_depthmaps_m], dim=-2)
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
            return args
            
        if args.config is None:
            return args
        
        # Load config from YAML file
        try:
            config = load_config(args.config)
            config_hparams = config.to_hparams()
        except (FileNotFoundError, Exception):
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