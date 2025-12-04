"""
PSF Import Utilities.

This module provides functions for loading and processing captured PSF data
for stereo camera systems with DOE.
"""

import torch
import torch.nn.functional as F


def fftshift(x, dims):
    """Shift zero-frequency component to center of spectrum."""
    shifts = [(x.size(dim)) // 2 for dim in dims]
    x = torch.roll(x, shifts=shifts, dims=dims)
    return x


def crop_boundary(x, w):
    """Crop boundary pixels from image."""
    if w == 0:
        return x
    else:
        return x[..., w:-w, int(1.6*w):-int(1.6*w)]


def normalize_psf(psfimg):
    """Normalize PSF to sum to 1."""
    return psfimg / psfimg.sum(dim=(-2, -1), keepdims=True)


def psf_captured(device):
    """
    Load captured PSF data for left and right cameras.
    
    Args:
        device: Target device for tensors
        
    Returns:
        Tuple of (psf_left, psf_right) tensors
    """
    device = torch.device(device)
    path_left = "psf/psf_left.pth"
    path_right = "psf/psf_right.pth"
    
    psf_left = torch.load(path_left).to(device)
    psf_right = torch.load(path_right).to(device)

    # Resize PSF
    psf_left = F.interpolate(
        psf_left.squeeze(0), scale_factor=0.8, mode='bilinear', align_corners=False
    ).unsqueeze(0)
    psf_right = F.interpolate(
        psf_right.squeeze(0), scale_factor=0.8, mode='bilinear', align_corners=False
    ).unsqueeze(0)
    
    # Pad to target size
    _, _, _, hh, ww = psf_left.shape
    h, w = 320, 736
    psf_left = F.pad(psf_left, ((w-ww)//2, (w-ww)//2, (h-hh)//2, (h-hh)//2), mode='constant', value=0)
    psf_right = F.pad(psf_right, ((w-ww)//2, (w-ww)//2, (h-hh)//2, (h-hh)//2), mode='constant', value=0)

    # Normalize each channel
    psf_left_b = psf_left.clone()
    psf_right_b = psf_right.clone()
    
    for c in range(3):
        psf_left_b[:, c, ...] = normalize_psf(psf_left[:, c, ...]) / 1.2
        psf_right_b[:, c, ...] = normalize_psf(psf_right[:, c, ...]) / 1.2

    # Flip depth dimension
    psf_left_b = torch.flip(psf_left_b, dims=[2])
    psf_right_b = torch.flip(psf_right_b, dims=[2])

    return psf_left_b, psf_right_b
