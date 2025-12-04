"""
FFT utilities for optical simulation.

This module provides FFT-related functions for PSF manipulation
and frequency-domain image processing.
"""

from typing import List, Tuple, Union

import torch


def fftshift(x: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    """
    Shift zero-frequency component to center of spectrum.
    
    Args:
        x: Input tensor
        dims: Dimensions along which to shift
    
    Returns:
        Shifted tensor
    """
    shifts = [(x.size(dim)) // 2 for dim in dims]
    return torch.roll(x, shifts=shifts, dims=dims)


def ifftshift(x: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
    """
    Inverse of fftshift - shift center back to corner.
    
    Args:
        x: Input tensor (centered spectrum)
        dims: Dimensions along which to shift
    
    Returns:
        Unshifted tensor
    """
    shifts = [(x.size(dim) + 1) // 2 for dim in dims]
    return torch.roll(x, shifts=shifts, dims=dims)


def crop_psf(x: torch.Tensor, sz: Union[int, Tuple[int, int], List[int]]) -> torch.Tensor:
    """
    Crop PSF to specified size while preserving FFT convention.
    
    The PSF is assumed to be in FFT convention (zero-frequency at corner).
    This function crops the PSF while maintaining the correct frequency
    ordering for FFT operations.
    
    Args:
        x: PSF tensor without fftshift applied (center at upper-left corner)
           Shape: (S, D, H, W) or (C, D, H, W)
        sz: Target size as int (square) or (height, width) tuple
    
    Returns:
        Cropped PSF tensor, shape (S, D, sz[0], sz[1])
    """
    device = x.device
    if isinstance(sz, int):
        sz = (sz, sz)
    
    # Calculate crop indices for both halves of the spectrum
    p0 = (sz[0] - 1) // 2 + 1  # positive frequencies
    p1 = (sz[1] - 1) // 2 + 1
    q0 = sz[0] - p0            # negative frequencies
    q1 = sz[1] - p1
    
    # Crop height dimension
    x_0 = torch.index_select(
        x, dim=-2,
        index=torch.cat([
            torch.arange(p0, device=device),
            torch.arange(x.shape[-2] - q0, x.shape[-2], device=device)
        ], dim=0)
    )
    
    # Crop width dimension
    x_1 = torch.index_select(
        x_0, dim=-1,
        index=torch.cat([
            torch.arange(p1, device=device),
            torch.arange(x.shape[-1] - q1, x.shape[-1], device=device)
        ], dim=0)
    )
    
    return x_1
