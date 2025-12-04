"""
Least-Sampling Angular Spectrum Method (LS-ASM) Implementation.

This code is based on the paper:
Haoyu Wei, Xin Liu, Xiang Hao, Edmund Y. Lam, and Yifan Peng, 
"Modeling off-axis diffraction with the least-sampling angular spectrum method," 
Optica 10, 959-962 (2023)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 
International license (CC BY-NC.)
"""

import torch
import math


def mdft(in_matrix, x, y, fx, fy):
    """Matrix DFT for non-uniform sampling."""
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-2)
    fx = fx.unsqueeze(-2)
    fy = fy.unsqueeze(-1)
    mx = torch.exp(-2 * torch.pi * 1j * torch.matmul(x, fx))
    my = torch.exp(-2 * torch.pi * 1j * torch.matmul(fy, y))
    out_matrix = torch.matmul(torch.matmul(my, in_matrix), mx)

    lx = torch.numel(x)
    ly = torch.numel(y)
    dx = 1 if lx == 1 else (torch.squeeze(x)[-1] - torch.squeeze(x)[0]) / (lx - 1)
    dy = 1 if ly == 1 else (torch.squeeze(y)[-1] - torch.squeeze(y)[0]) / (ly - 1)

    out_matrix = out_matrix * dx * dy
    return out_matrix


def midft(in_matrix, x, y, fx, fy):
    """Matrix inverse DFT for non-uniform sampling."""
    x = x.unsqueeze(-2)
    y = y.unsqueeze(-1)
    fx = fx.unsqueeze(-1)
    fy = fy.unsqueeze(-2)
    mx = torch.exp(2 * torch.pi * 1j * torch.matmul(fx, x))
    my = torch.exp(2 * torch.pi * 1j * torch.matmul(y, fy))
    out_matrix = torch.matmul(torch.matmul(my, in_matrix), mx)

    lfx = torch.numel(fx)
    lfy = torch.numel(fy)
    dfx = 1 if lfx == 1 else (torch.squeeze(fx)[-1] - torch.squeeze(fx)[0]) / (lfx - 1)
    dfy = 1 if lfy == 1 else (torch.squeeze(fy)[-1] - torch.squeeze(fy)[0]) / (lfy - 1)

    out_matrix = out_matrix * dfx * dfy
    return out_matrix


class LeastSamplingASM():
    """
    Least-Sampling Angular Spectrum Method for wave propagation.
    
    This class implements efficient wave propagation using the LS-ASM algorithm,
    which reduces computational cost while maintaining accuracy for off-axis diffraction.
    """

    def __init__(self, Uin, xvec, yvec, z, device):
        """
        Initialize LS-ASM propagator.
        
        Args:
            Uin: Input field object containing aperture and wavelength info
            xvec, yvec: Vectors of destination coordinates
            z: Propagation distance
            device: Computation device (cuda/cpu)
        """
        super().__init__()

        dtype = torch.double
        complex_dtype = torch.complex64

        xivec = torch.as_tensor(Uin.xi, device=device)
        etavec = torch.as_tensor(Uin.eta, device=device)
        xvec = torch.as_tensor(xvec, device=device)
        yvec = torch.as_tensor(yvec, device=device)
        z = torch.as_tensor(z, device=device)
        wavelength = torch.as_tensor(Uin.wvls, device=device)

        n = 1
        k = 2 * math.pi / wavelength * n

        # Bandwidth of aperture
        Lfx = Uin.fbX
        Lfy = Uin.fbY

        # Off-axis offset
        offx = torch.as_tensor(Uin.fcX, device=device)
        offy = torch.as_tensor(Uin.fcY, device=device)

        deltax = torch.tensor(0, device=device)
        deltay = torch.tensor(0, device=device)

        # Frequency sampling resolution
        LRfx = 1520
        LRfy = 1520

        dfx2 = Lfx / LRfx
        dfy2 = Lfy / LRfy

        # Spatial frequency coordinates
        fx = torch.linspace(-Lfx / 2, Lfx / 2 - dfx2, LRfx, device=device, dtype=complex_dtype)
        fy = torch.linspace(-Lfy / 2, Lfy / 2 - dfy2, LRfy, device=device, dtype=complex_dtype)
        fx_shift, fy_shift = fx + offx, fy + offy

        fxx, fyy = torch.meshgrid(fx_shift, fy_shift, indexing='xy')

        # Setup for multi-wavelength
        wave_num = wavelength.shape[0]
        wavelength = wavelength.reshape(wave_num, -1, 1, 1)
        k = k.reshape(wave_num, -1, 1, 1)
        fxx = fxx.unsqueeze(0).unsqueeze(0).repeat(wave_num, 1, 1, 1)
        fyy = fyy.unsqueeze(0).unsqueeze(0).repeat(wave_num, 1, 1, 1)

        # Transfer function H
        self.H = torch.exp(1j * k * (wavelength * fxx * deltax + wavelength * fyy * deltay
                          + z * torch.sqrt(1 - (fxx * wavelength)**2 - (fyy * wavelength)**2)))

        self.xi = xivec.to(dtype=complex_dtype)
        self.eta = etavec.to(dtype=complex_dtype)
        self.x = xvec.to(dtype=complex_dtype) - deltax
        self.y = yvec.to(dtype=complex_dtype) - deltay
        self.offx, self.offy = offx, offy
        self.device = device
        self.fx = fx_shift
        self.fy = fy_shift
        self.fbX = Uin.fbX
        self.fbY = Uin.fbY

    def __call__(self, E0):
        """
        Propagate input field E0.
        
        Args:
            E0: Input electric field
            
        Returns:
            Propagated electric field at destination plane
        """
        E0 = torch.as_tensor(E0, dtype=torch.complex64, device=self.device)

        fx = self.fx.unsqueeze(0)
        fy = self.fy.unsqueeze(0)

        Fu = mdft(E0, self.xi, self.eta, fx - self.offx, fy - self.offy)
        Eout = midft(Fu * self.H, self.x, self.y, fx, fy)

        return Eout

    def grad_H(self, lam, z, fx, fy):
        """Compute gradient of transfer function H."""
        eps = torch.tensor(1e-9, device=fx.device)
        denom = torch.max(1 - (lam * fx)**2 - (lam * fy)**2, eps)
        gradx = -z * 2 * torch.pi * lam * fx / torch.sqrt(denom)
        grady = -z * 2 * torch.pi * lam * fy / torch.sqrt(denom)
        return gradx, grady

    def compute_shift_of_H(self, C1, C2, pc, w):
        """Compute optimal shift for transfer function."""
        C1 = C1.mean()
        C2 = C2.mean()
        if (w > -2 * C1 - 2 * pc + C2) and (w < 2 * C1 + 2 * pc + C2):
            delta = pc / 2 + w / 4 - C1 / 2 - C2 / 4
        elif (w > 2 * C1 + 2 * pc + C2) and (w < -2 * C1 - 2 * pc + C2):
            delta = pc / 2 - w / 4 - C1 / 2 + C2 / 4
        elif (w > 2 * C1 + 2 * pc + C2) and (w > -2 * C1 - 2 * pc + C2):
            delta = pc
        elif (w < 2 * C1 + 2 * pc + C2) and (w < -2 * C1 - 2 * pc + C2):
            delta = -C1
        return delta