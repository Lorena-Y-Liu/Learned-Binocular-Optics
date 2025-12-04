"""
Loss functions for Deep Stereo training.

This module provides perceptual loss based on VGG16 features
for image reconstruction quality evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import VGG16_Weights
from torchmetrics import Metric


class Vgg16PerceptualLoss(Metric):
    """
    VGG16-based perceptual loss for image quality assessment.
    
    Uses features from early VGG16 layers to compute perceptual
    similarity between images, which correlates better with human
    perception than pixel-wise metrics.
    
    The loss combines:
    - L1 pixel loss (25% weight)
    - VGG feature losses from conv1_2, conv2_2, conv3_3 (75% weight)
    """

    def __init__(self):
        super().__init__()
        # Load pretrained VGG16 with new weights API
        vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Extract feature blocks (frozen for inference)
        self.vgg_blocks = nn.ModuleList([
            vgg16.features[:4].eval(),   # conv1_2
            vgg16.features[4:9].eval(),  # conv2_2
            vgg16.features[9:16].eval(), # conv3_3
        ])
        self.vgg_blocks.requires_grad_(False)
        
        # ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        
        # Feature weights (normalized)
        self.weight = [11.17 / 35.04 / 4, 35.04 / 35.04 / 4, 29.09 / 35.04 / 4]

        # Metric state for distributed training
        self.add_state('diff', default=torch.tensor([0., 0., 0., 0.]), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor([0, 0, 0., 0.]), dist_reduce_fx='sum')

    def train_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss for training.
        
        Args:
            input: Predicted image tensor (B, 3, H, W), range [0, 1]
            target: Ground truth image tensor (B, 3, H, W), range [0, 1]
        
        Returns:
            Combined perceptual loss value
        """
        # Normalize to ImageNet statistics
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Pixel-wise L1 loss
        loss = F.l1_loss(input, target) / 4
        
        # Pad for valid convolution at boundaries
        input = F.pad(input, mode='reflect', pad=(4, 4, 4, 4))
        target = F.pad(target, mode='reflect', pad=(4, 4, 4, 4))
        
        # Feature-based losses
        for i, block in enumerate(self.vgg_blocks):
            input = block(input)
            target = block(target)
            loss += self.weight[i] * F.l1_loss(input[..., 4:-4, 4:-4], target[..., 4:-4, 4:-4])
        
        return loss

    def update(self, input: torch.Tensor, target: torch.Tensor):
        """Update metric state for distributed evaluation."""
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        self.diff[0] += (input - target).sum() / 4
        self.total[0] += input.numel()

        input = F.pad(input, mode='reflect', pad=(4, 4, 4, 4))
        target = F.pad(target, mode='reflect', pad=(4, 4, 4, 4))
        
        for i, block in enumerate(self.vgg_blocks):
            input = block(input)
            target = block(target)
            self.diff[i + 1] += self.weight[i] * (input[..., 4:-4, 4:-4] - target[..., 4:-4, 4:-4]).sum()
            self.total[i + 1] += input[..., 4:-4, 4:-4].numel()

    def compute(self) -> torch.Tensor:
        """Compute final metric value from accumulated state."""
        return (self.diff[0] / self.total[0] + 
                self.diff[1] / self.total[1] + 
                self.diff[2] / self.total[2] + 
                self.diff[3] / self.total[3])
