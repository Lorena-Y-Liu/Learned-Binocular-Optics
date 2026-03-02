"""
Improved Fusion Module for Deep Stereo (Version 2).

Key improvements over fusion.py:
1. Separate prediction heads for RGB and Depth (no gradient competition)
2. Proper gradient flow for DFD loss (est_dfd returns prediction, not input)
3. Residual learning for depth refinement (learn correction, not absolute depth)
4. Utilizes DFD features from pinv_volumes (Tikhonov inverse)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.outputs_container import OutputsContainer
from models.unet import UNet
from models.feature_fusion import SCAM


class Recovery2(nn.Module):
    """
    Improved image and depth recovery module.
    
    Changes from Recovery:
    - Separate output heads for RGB and depth prediction
    - Residual depth learning (predicts correction to rough_depth)
    - Incorporates pinv_volumes for DFD cues
    - Returns actual DFD prediction for proper loss computation
    """
    
    def __init__(self, hparams, *args, **kargs):
        super().__init__()
        self.preinverse = hparams.preinverse
        self.scale = hparams.scale
        
        depth_ch = 1
        color_ch = 3
        color = 3
        n_layers = 4
        n_depths = hparams.n_depths
        base_ch = 32
        
        # Calculate input channels:
        # - captimgs_left: 3 channels (RGB)
        # - captimgs_right: 3 channels (RGB)  
        # - rough_depth: 1 channel
        # - pinv_volumes_left: n_depths * 3 channels (DFD features)
        preinv_input_ch = color * n_depths + color_ch
        
        # [Improvement 4] Input layer now includes pinv_volumes for DFD cues
        # Input: left_img(3) + right_img(3) + rough_depth(1) + pinv_volumes(n_depths*3)
        total_input_ch = 2 * color + depth_ch + color * n_depths
        
        self.input_layers = nn.Sequential(
            nn.Conv2d(total_input_ch, preinv_input_ch + color, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(preinv_input_ch + color),
            nn.ReLU(),
            nn.Conv2d(preinv_input_ch + color, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )
        
        # Fallback input layer without pinv_volumes (for backward compatibility)
        self.input_layers_no_pinv = nn.Sequential(
            nn.Conv2d(2 * color + depth_ch, preinv_input_ch + color, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(preinv_input_ch + color),
            nn.ReLU(),
            nn.Conv2d(preinv_input_ch + color, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )

        # Main decoder (shared feature extraction)
        self.decoder = nn.Sequential(
            UNet(
                channels=[base_ch, base_ch, 2 * base_ch, 2 * base_ch, 4 * base_ch, 4 * base_ch, 4 * base_ch, 4 * base_ch],
                n_layers=n_layers,
            )
        )
        
        # [Improvement 1] Separate prediction heads for RGB and Depth
        # RGB prediction head
        self.output_layers_rgb = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch // 2),
            nn.ReLU(),
            nn.Conv2d(base_ch // 2, color, kernel_size=1, bias=True)
        )
        
        # [Improvement 3] Depth prediction head - outputs RESIDUAL (correction to rough_depth)
        # Using tanh activation to allow positive and negative corrections
        self.output_layers_depth = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch // 2),
            nn.ReLU(),
            nn.Conv2d(base_ch // 2, depth_ch, kernel_size=1, bias=True),
            # No activation here - we'll handle it in forward
        )
        
        # Scale factor for residual (FIXED, not learnable, to prevent scale drift)
        # Using a fixed value ensures the network cannot grow the residual magnitude
        # unboundedly during training, which was causing global scale shift.
        self.max_residual = 0.1
        
        # Optional: Additional depth refinement branch using SCAM
        base_ch2 = 16
        self.use_depth_refinement = True
        
        if self.use_depth_refinement:
            # Feature extraction for depth refinement
            self.depth_feature_conv = nn.Sequential(
                nn.Conv2d(depth_ch, base_ch2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch2),
                nn.ReLU()
            )
            self.dfd_feature_conv = nn.Sequential(
                nn.Conv2d(depth_ch, base_ch2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch2),
                nn.ReLU()
            )
            # SCAM module for feature fusion
            self.scam_depth = SCAM(base_ch2)
            
            # Final depth refinement
            self.depth_refine = nn.Sequential(
                nn.Conv2d(base_ch2, base_ch2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_ch2),
                nn.ReLU(),
                nn.Conv2d(base_ch2, depth_ch, kernel_size=1, bias=True)
            )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, captimgs_left, pinv_volumes_left, captimgs_right, rough_depth, *args, **kargs):
        """
        Forward pass with improved architecture.
        
        Args:
            captimgs_left: Left captured image [B, 3, H, W]
            pinv_volumes_left: Tikhonov inverse volumes [B, n_depths*3, H, W] - contains DFD cues
            captimgs_right: Right captured image [B, 3, H, W]
            rough_depth: Initial depth from stereo matching [B, 1, H, W], normalized to [0, 1]
        
        Returns:
            OutputsContainer with:
                - est_images: Reconstructed RGB image
                - est_dfd: DFD-based depth prediction (with residual learning)
                - est_depthmaps: Final refined depth map
        """
        b_sz, _, h_sz, w_sz = captimgs_left.shape
        target_h, target_w = int(h_sz * self.scale), int(w_sz * self.scale)
        
        # Resize inputs to target scale
        captimgs_left = F.interpolate(captimgs_left, size=(target_h, target_w),
                                      mode='bilinear', align_corners=False)
        captimgs_right = F.interpolate(captimgs_right, size=(target_h, target_w),
                                       mode='bilinear', align_corners=False)
        rough_depth = rough_depth.reshape(b_sz, 1, target_h, target_w)
        
        # [Improvement 4] Include pinv_volumes for DFD cues
        # Check if pinv_volumes has meaningful content (not all zeros)
        use_pinv = pinv_volumes_left is not None and pinv_volumes_left.abs().sum() > 1e-6
        
        if use_pinv:
            # Resize pinv_volumes
            pinv_volumes_left = F.interpolate(
                pinv_volumes_left, 
                size=[pinv_volumes_left.shape[2], target_h, target_w],
                mode='trilinear', align_corners=False
            )
            # Flatten depth dimension into channels: [B, n_depths, C, H, W] -> [B, n_depths*C, H, W]
            if len(pinv_volumes_left.shape) == 5:
                b, d, c, h, w = pinv_volumes_left.shape
                pinv_flat = pinv_volumes_left.reshape(b, d * c, h, w)
            else:
                pinv_flat = pinv_volumes_left
            
            # Concatenate all inputs including DFD features
            inputs = torch.cat([captimgs_left, captimgs_right, rough_depth, pinv_flat], dim=1)
            features = self.input_layers(inputs)
        else:
            # Fallback without pinv_volumes
            inputs = torch.cat([captimgs_left, captimgs_right, rough_depth], dim=1)
            features = self.input_layers_no_pinv(inputs)
        
        # Main decoder for shared feature extraction
        decoded_features = self.decoder(features)
        
        # [Improvement 1] Separate RGB prediction
        est_images = torch.sigmoid(self.output_layers_rgb(decoded_features))
        
        # [Improvement 2 & 3] Depth prediction with residual learning
        # Predict a residual/correction to the rough depth
        depth_residual = self.output_layers_depth(decoded_features)
        # Use tanh to bound residual, then scale with FIXED max_residual
        depth_residual = torch.tanh(depth_residual) * self.max_residual
        # [KEY FIX] Zero-mean constraint: subtract spatial mean so the residual
        # can only add high-frequency details (edges, texture) without shifting
        # the global depth level. This preserves rough_depth's good global scale
        # while allowing local refinement.
        depth_residual = depth_residual - depth_residual.mean(dim=[2, 3], keepdim=True)
        # Add residual to rough depth, clamp to valid range [0, 1]
        est_depthmaps_dfd = (rough_depth + depth_residual).clamp(0.0, 1.0)
        
        # Optional: Further refinement using SCAM fusion
        if self.use_depth_refinement:
            # Extract features from rough depth and DFD prediction
            rough_features = self.depth_feature_conv(rough_depth)
            dfd_features = self.dfd_feature_conv(est_depthmaps_dfd)
            
            # Fuse features using SCAM
            fused_features = self.scam_depth(rough_features, dfd_features)
            
            # Predict final refined depth (also as residual)
            refine_residual = torch.tanh(self.depth_refine(fused_features)) * 0.05
            # [KEY FIX] Zero-mean constraint on refinement residual too
            refine_residual = refine_residual - refine_residual.mean(dim=[2, 3], keepdim=True)
            est_depthmaps_refined = (est_depthmaps_dfd + refine_residual).clamp(0.0, 1.0)
        else:
            est_depthmaps_refined = est_depthmaps_dfd
        
        # [Improvement 2] Return actual predictions for proper gradient flow
        outputs = OutputsContainer(
            est_images=est_images,
            est_dfd=est_depthmaps_dfd,  # Now returns actual DFD prediction, not input!
            est_depthmaps=est_depthmaps_refined  # Final refined depth
        )
        return outputs


class Recovery2Light(nn.Module):
    """
    Lightweight version of Recovery2 without SCAM refinement.
    
    Use this if you want faster training or have memory constraints.
    """
    
    def __init__(self, hparams, *args, **kargs):
        super().__init__()
        self.preinverse = hparams.preinverse
        self.scale = hparams.scale
        
        depth_ch = 1
        color = 3
        n_layers = 4
        n_depths = hparams.n_depths
        base_ch = 32
        preinv_input_ch = color * n_depths + color
        
        # Input layer (without pinv for simplicity)
        self.input_layers = nn.Sequential(
            nn.Conv2d(2 * color + depth_ch, preinv_input_ch + color, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(preinv_input_ch + color),
            nn.ReLU(),
            nn.Conv2d(preinv_input_ch + color, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )

        # Main decoder
        self.decoder = nn.Sequential(
            UNet(
                channels=[base_ch, base_ch, 2 * base_ch, 2 * base_ch, 4 * base_ch, 4 * base_ch, 4 * base_ch, 4 * base_ch],
                n_layers=n_layers,
            )
        )
        
        # [Improvement 1] Separate RGB head
        self.output_layers_rgb = nn.Sequential(
            nn.Conv2d(base_ch, color, kernel_size=1, bias=True)
        )
        
        # [Improvement 1 & 3] Separate Depth head with residual learning
        self.output_layers_depth = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch // 2),
            nn.ReLU(),
            nn.Conv2d(base_ch // 2, depth_ch, kernel_size=1, bias=True),
        )
        
        # Fixed residual scale (not learnable) to prevent scale drift
        self.max_residual = 0.1
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, captimgs_left, pinv_volumes_left, captimgs_right, rough_depth, *args, **kargs):
        b_sz, _, h_sz, w_sz = captimgs_left.shape
        target_h, target_w = int(h_sz * self.scale), int(w_sz * self.scale)
        
        captimgs_left = F.interpolate(captimgs_left, size=(target_h, target_w),
                                      mode='bilinear', align_corners=False)
        captimgs_right = F.interpolate(captimgs_right, size=(target_h, target_w),
                                       mode='bilinear', align_corners=False)
        rough_depth = rough_depth.reshape(b_sz, 1, target_h, target_w)
        
        inputs = torch.cat([captimgs_left, captimgs_right, rough_depth], dim=1)
        features = self.input_layers(inputs)
        decoded_features = self.decoder(features)
        
        # Separate predictions
        est_images = torch.sigmoid(self.output_layers_rgb(decoded_features))
        
        # Residual depth prediction with zero-mean constraint
        depth_residual = torch.tanh(self.output_layers_depth(decoded_features)) * self.max_residual
        # Zero-mean: preserve rough_depth's global scale, only add details
        depth_residual = depth_residual - depth_residual.mean(dim=[2, 3], keepdim=True)
        est_depthmaps_dfd = (rough_depth + depth_residual).clamp(0.0, 1.0)
        
        # [Improvement 2] Return actual DFD prediction
        outputs = OutputsContainer(
            est_images=est_images,
            est_dfd=est_depthmaps_dfd,  # Actual prediction for gradient flow
            est_depthmaps=est_depthmaps_dfd  # Same as est_dfd in light version
        )
        return outputs
