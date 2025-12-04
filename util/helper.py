"""
Helper functions for image processing and optics simulation.

This module provides utilities for:
- Depth map processing and layered representation
- Color space conversion (sRGB <-> linear)
- Optical calculations (refractive index, phase)
- Bayer pattern simulation
"""

import math

import torch
import torch.nn.functional as F


def matting(depthmap: torch.Tensor, n_depths: int, binary: bool, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert continuous depth map to soft layered representation.
    
    Args:
        depthmap: Normalized depth map [0, 1], shape (B, 1, H, W)
        n_depths: Number of depth layers
        binary: If True, use hard assignment; if False, use soft blending
        eps: Small value for numerical stability
    
    Returns:
        Alpha weights for each depth layer, shape (B, 1, D, H, W)
    """
    depthmap = depthmap.clamp(eps, 1.0)
    d = torch.arange(0, n_depths, dtype=depthmap.dtype, device=depthmap.device).reshape(1, 1, -1, 1, 1) + 1
    depthmap = depthmap * n_depths
    diff = d - depthmap
    alpha = torch.zeros_like(diff)
    if binary:
        alpha[torch.logical_and(diff >= 0., diff < 1.)] = 1.
    else:
        mask = torch.logical_and(diff > -1., diff <= 0.)
        alpha[mask] = diff[mask] + 1.
        alpha[torch.logical_and(diff > 0., diff <= 1.)] = 1.
    return alpha


def depthmap_to_layereddepth(depthmap: torch.Tensor, n_depths: int, binary: bool = False) -> torch.Tensor:
    """Convert depth map to layered depth representation."""
    depthmap = depthmap[:, None, ...]  # add color dim
    return matting(depthmap, n_depths, binary=binary)


def over_op(alpha: torch.Tensor) -> torch.Tensor:
    """Compute over compositing weights from alpha values."""
    bs, cs, ds, hs, ws = alpha.shape
    out = torch.cumprod(1. - alpha, dim=-3)
    return torch.cat([torch.ones((bs, cs, 1, hs, ws), dtype=out.dtype, device=out.device), out[:, :, :-1]], dim=-3)


def crop_boundary(x: torch.Tensor, w: int) -> torch.Tensor:
    """Crop w pixels from each boundary of the spatial dimensions."""
    if w == 0:
        return x
    return x[..., w:-w, w:-w]


def refractive_index(wavelength: torch.Tensor, a: float = 1.5375, b: float = 0.00829045, c: float = -0.000211046) -> torch.Tensor:
    """
    Compute refractive index using Cauchy's dispersion formula.
    
    Default coefficients are for NOA61 optical adhesive.
    Reference: https://refractiveindex.info/?shelf=other&book=Optical_adhesives&page=Norland_NOA61
    
    Args:
        wavelength: Wavelength in meters
        a, b, c: Cauchy coefficients
    
    Returns:
        Refractive index at the given wavelength
    """
    return a + b / (wavelength * 1e6) ** 2 + c / (wavelength * 1e6) ** 4


def gray_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Convert single-channel grayscale to 3-channel RGB by replication."""
    return x.repeat(1, 3, 1, 1)


def linear_to_srgb(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert linear RGB to sRGB color space."""
    a = 0.055
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.0031308, 12.92 * x, (1. + a) * x ** (1. / 2.4) - a)


def srgb_to_linear(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert sRGB to linear RGB color space."""
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def heightmap_to_phase(height: torch.Tensor, wavelength: float, refractive_idx: float) -> torch.Tensor:
    """Convert DOE height map to phase delay."""
    return height * (2 * math.pi / wavelength) * (refractive_idx - 1)


def phase_to_heightmap(phase: torch.Tensor, wavelength: float, refractive_idx: float) -> torch.Tensor:
    """Convert phase delay to DOE height map."""
    return phase / (2 * math.pi / wavelength) / (refractive_idx - 1)


def imresize(img: torch.Tensor, size) -> torch.Tensor:
    """Resize image using bilinear interpolation."""
    return F.interpolate(img, size=size)


def ips_to_metric(d: torch.Tensor, min_depth: float, max_depth: float) -> torch.Tensor:
    """
    Convert inverse perspective sampling to metric depth.
    
    Args:
        d: Normalized depth in [0, 1] (inverse perspective sampling)
        min_depth: Minimum depth in meters
        max_depth: Maximum depth in meters
    
    Returns:
        Metric depth in meters
    """
    return (max_depth * min_depth) / (max_depth - (max_depth - min_depth) * d)


def metric_to_ips(d: torch.Tensor, min_depth: float, max_depth: float) -> torch.Tensor:
    """
    Convert metric depth to inverse perspective sampling.
    
    Args:
        d: Metric depth in [min_depth, max_depth] meters
        min_depth: Minimum depth in meters
        max_depth: Maximum depth in meters
    
    Returns:
        Normalized depth in [0, 1]
    """
    return (max_depth * d - max_depth * min_depth) / ((max_depth - min_depth) * d)


def copy_quadruple(x_rd: torch.Tensor) -> torch.Tensor:
    """
    Create a symmetric pattern by mirroring a quadrant.
    
    Takes the lower-right quadrant and mirrors it to create
    a full symmetric 2D pattern (used for DOE height maps).
    """
    x_ld = torch.flip(x_rd, dims=(-2,))
    x_d = torch.cat([x_ld, x_rd], dim=-2)
    x_u = torch.flip(x_d, dims=(-1,))
    return torch.cat([x_u, x_d], dim=-1)


def to_bayer(x: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image to Bayer pattern mosaic.
    
    Uses RGGB Bayer pattern:
    - Red at even rows, even columns
    - Blue at odd rows, odd columns  
    - Green at remaining positions
    
    Args:
        x: RGB image tensor (B, 3, H, W)
    
    Returns:
        Bayer mosaic tensor (B, 1, H, W)
    """
    mask = torch.zeros_like(x)
    mask[:, 0, ::2, ::2] = 1      # R
    mask[:, 2, 1::2, 1::2] = 1    # B
    mask[:, 1, 1::2, ::2] = 1     # G
    mask[:, 1, ::2, 1::2] = 1     # G
    y = x * mask
    return y.sum(dim=1, keepdim=True)
