"""
Evaluation metrics for image and depth estimation.

This module provides common metrics for evaluating:
- Image quality (PSNR, SSIM, MSE, MAE)
- Depth estimation accuracy (3px error rate)
"""

import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim


def mae(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """Compute Mean Absolute Error between two tensors."""
    return F.l1_loss(image1, image2).item()


def mse(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """Compute Mean Squared Error between two tensors."""
    return F.mse_loss(image1, image2).item()


def rmse(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """Compute Root Mean Squared Error between two tensors."""
    return torch.sqrt(F.mse_loss(image1, image2)).item()


def calculate_psnr(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Assumes images are normalized to [0, 1] range.
    """
    mse_val = F.mse_loss(image1, image2)
    psnr = 10 * torch.log10(1 / mse_val)
    return psnr.item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Compute Structural Similarity Index between two images.
    
    Args:
        img1: First image tensor, normalized to [0, 1]
        img2: Second image tensor, normalized to [0, 1]
    
    Returns:
        SSIM value as float
    """
    # Scale to [0, 255] for SSIM computation
    img1 = img1 * 255
    img2 = img2 * 255
    return ssim(img1, img2).item()


def calculate_3px(depth1: torch.Tensor, depth2: torch.Tensor, threshold: float = 3.0) -> float:
    """
    Compute 3-pixel error rate for disparity/depth maps.
    
    Args:
        depth1: Estimated depth/disparity map
        depth2: Ground truth depth/disparity map
        threshold: Error threshold in pixels (default: 3.0)
    
    Returns:
        Percentage of pixels with error > threshold
    """
    abs_diff = torch.abs(depth1 - depth2)
    error_mask = abs_diff > threshold
    return error_mask.float().mean().item() * 100