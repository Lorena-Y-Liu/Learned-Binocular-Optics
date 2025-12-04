"""
Image reconstruction solvers using Tikhonov regularization.

This module provides functions for reconstructing images from defocused
captures using Tikhonov-regularized inverse filtering.
"""

import torch
import torch.nn.functional as F


def tikhonov_inverse_fast(Y, G, v=None, beta=0, gamma=1e-1, dataformats='SDHW'):
    """
    Compute Tikhonov-regularized inverse.
    
    This function solves:
        argmin_x || y - sum_{k} G_k x_k ||^2 + beta sum_{k} || S x_k ||^2 + gamma || x - v ||^2
    
    Args:
        Y: Captured image in frequency domain
        G: PSF in frequency domain
        v: Prior estimate (optional)
        beta: Laplacian regularization weight
        gamma: Tikhonov regularization weight (required for numerical stability)
        dataformats: Data format string ('DHW', 'SDHW', 'CSDHW', 'BCDHW', 'BCSDHW')
    
    Returns:
        Reconstructed volume in frequency domain
    """
    if dataformats == 'DHW':
        Y = Y[None, None, None, ...]
        G = G[None, None, None, ...]
        if v is not None:
            v = v[None, None, None, ...]
    elif dataformats == 'SDHW':
        Y = Y[None, None, ...]
        G = G[None, None, ...]
        if v is not None:
            v = v[None, None, ...]
    elif dataformats == 'CSDHW':
        Y = Y[None, ...]
        G = G[None, ...]
        if v is not None:
            v = v[None, ...]
    elif dataformats == 'BCDHW':
        Y = Y.unsqueeze(2)
        G = G.unsqueeze(2)
        if v is not None:
            v = v.unsqueeze(2)
    elif dataformats == 'BCSDHW':
        pass
    else:
        raise NotImplementedError(f'Data format not supported: {dataformats}')

    device = Y.device
    dtype = Y.dtype
    num_colors, num_shots, depth, height, width = G.shape[1:6]
    batch_sz = Y.shape[0]

    Y_real = Y[..., 0].reshape([batch_sz, num_colors, num_shots, 1, -1]).transpose(2, 4)
    Y_imag = Y[..., 1].reshape([batch_sz, num_colors, num_shots, 1, -1]).transpose(2, 4)
    G_real = (G[..., 0]).reshape([1, num_colors, num_shots, depth, -1]).transpose(2, 4)
    G_imag = (G[..., 1]).reshape([1, num_colors, num_shots, depth, -1]).transpose(2, 4)
    Gc_real = G_real
    Gc_imag = -G_imag

    GcY_real = (Gc_real * Y_real - Gc_imag * Y_imag).sum(dim=-1, keepdims=True)
    GcY_imag = (Gc_imag * Y_real + Gc_real * Y_imag).sum(dim=-1, keepdims=True)

    if v is not None:
        V = gamma * torch.rfft(v, 2)
        V_real = (V[..., 0]).reshape([batch_sz, num_colors, 1, depth, -1]).transpose(2, 4)
        V_imag = (V[..., 1]).reshape([batch_sz, num_colors, 1, depth, -1]).transpose(2, 4)
        GcY_real += V_real
        GcY_imag += V_imag

    if not isinstance(gamma, torch.Tensor):
        reg = torch.tensor(gamma, device=device, dtype=dtype)
    else:
        reg = gamma

    Gc_real_t = Gc_real.transpose(3, 4)
    Gc_imag_t = Gc_imag.transpose(3, 4)
    
    if num_shots == 1:
        innerprod = torch.matmul(Gc_real_t, G_real) - torch.matmul(Gc_imag_t, G_imag)
        outerprod_real = torch.matmul(G_real, Gc_real_t) - torch.matmul(G_imag, Gc_imag_t)
        outerprod_imag = torch.matmul(G_imag, Gc_real_t) + torch.matmul(G_real, Gc_imag_t)
        invM_real = 1. / reg * (
                torch.eye(depth, device=device, dtype=dtype) - outerprod_real / (reg + innerprod))
        invM_imag = -1. / reg * outerprod_imag / (reg + innerprod)
    else:
        eye_plus_inner = torch.eye(num_shots, device=device, dtype=dtype) + 1 / reg * (
                torch.matmul(Gc_real_t, G_real) - torch.matmul(Gc_imag_t, G_imag))
        eye_plus_inner_inv = torch.inverse(eye_plus_inner)
        inner_Gc_real = torch.matmul(eye_plus_inner_inv, Gc_real_t)
        inner_Gc_imag = torch.matmul(eye_plus_inner_inv, Gc_imag_t)
        prod_real = 1 / reg * (torch.matmul(G_real, inner_Gc_real) - torch.matmul(G_imag, inner_Gc_imag))
        prod_imag = 1 / reg * (torch.matmul(G_imag, inner_Gc_real) + torch.matmul(G_real, inner_Gc_imag))
        invM_real = 1 / reg * (torch.eye(depth, device=device, dtype=dtype).unsqueeze(0) - prod_real)
        invM_imag = - 1 / reg * prod_imag

    X_real = (torch.matmul(invM_real, GcY_real) - torch.matmul(invM_imag, GcY_imag))
    X_imag = (torch.matmul(invM_imag, GcY_real) + torch.matmul(invM_real, GcY_imag))
    X = torch.stack(
        [X_real.transpose(2, 3).reshape(batch_sz, num_colors, depth, height, width),
         X_imag.transpose(2, 3).reshape(batch_sz, num_colors, depth, height, width)],
        dim=-1)

    if dataformats == 'SDHW':
        X = X.reshape(depth, height, width, 2)
    elif dataformats == 'CSDHW':
        X = X.reshape(num_colors, depth, height, width, 2)
    elif dataformats == 'BCSDHW' or dataformats == 'BCDHW':
        X = X.reshape(batch_sz, num_colors, depth, height, width, 2)

    return X


# FFT compatibility layer for different PyTorch versions
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft2(x, dim=(-d, -1))
        return torch.stack((t.real, t.imag), -1)
    
    def irfft(x, d, signal_sizes):
        t = torch.fft.ifft2(torch.complex(x[..., 0], x[..., 1]), dim=(-d, -1))
        return t.real
    


def apply_tikhonov_inverse(captimg, psf, reg_tikhonov, apply_edgetaper=True):
    """
    Apply Tikhonov-regularized inverse filtering to reconstruct depth volumes.

    Args:
        captimg: Captured image tensor (B x C x H x W)
        psf: Point spread function, lateral size should equal captimg (1 x C x D x H x W)
        reg_tikhonov: Tikhonov regularization parameter (float)
        apply_edgetaper: Whether to apply edge tapering (currently disabled)

    Returns:
        Reconstructed volume tensor (B x C x D x H x W)
    """
    Fpsf = rfft(psf, 2)
    Fcaptimgs = rfft(captimg, 2)
    Fpsf = Fpsf.unsqueeze(2)  # add shot dimension
    Fcaptimgs = Fcaptimgs.unsqueeze(2)  # add shot dimension
    
    est_X = tikhonov_inverse_fast(
        Fcaptimgs, Fpsf, v=None, beta=0, gamma=reg_tikhonov, dataformats='BCSDHW'
    )
    
    est_volumes = irfft(est_X, 2, signal_sizes=captimg.shape[-2:])
    
    return est_volumes
