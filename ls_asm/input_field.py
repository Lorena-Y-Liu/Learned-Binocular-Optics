"""
Input Field Module for LS-ASM.

This module prepares the input field for wave propagation simulation.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 
International license (CC BY-NC.)

Technical Paper:
Haoyu Wei, Xin Liu, Xiang Hao, Edmund Y. Lam, and Yifan Peng, 
"Modeling off-axis diffraction with the least-sampling angular spectrum method," 
Optica 10, 959-962 (2023)
"""

import numpy as np
import torch
from ls_asm.phase_plates import SphericalWave, ThinLens


class InputField():
    """Prepare compensated input field and spatial sampling."""

    def __init__(self, type, wvls, r, z0, f, zf, mask_size, modulate_phase, heightmap):
        """
        Initialize input field.
        
        Args:
            type: Field type string (e.g., "12" for spherical wave + thin lens)
            wvls: Wavelengths of light
            r: Aperture radius
            z0: Source distance
            f: Focal length
            zf: Focus distance
            mask_size: Size of the DOE mask
            modulate_phase: Whether to modulate phase
            heightmap: DOE height map (optional)
        """
        self.wvls = wvls
        self.k = 2 * torch.pi / self.wvls
        thetaX = torch.tensor(0)
        thetaY = torch.tensor(0)
        self.modulate_phase = modulate_phase
        self.heightmap = heightmap
        s = 1

        # Define incident wave
        r0 = z0 / torch.sqrt(1 - torch.sin(thetaX / 180 * torch.pi)**2 - torch.sin(thetaY / 180 * torch.pi)**2)
        x0 = r0 * torch.sin(thetaX / 180 * torch.pi)
        y0 = r0 * torch.sin(thetaY / 180 * torch.pi)

        # Prepare wave components
        typelist = [*type]
        wavelist = []
        fcX = 0
        fcY = 0

        if "1" in typelist:
            field = SphericalWave(self.k, x0, y0, z0, (0, 0), zf)
            fcX += field.fcX
            fcY += field.fcY
            wavelist.append(field)

        if "2" in typelist:
            lens = ThinLens(self.k, f)
            fcX += lens.fcX
            fcY += lens.fcY
            wavelist.append(lens)

        # Compute spatial sampling
        Nx, Ny, fbX, fbY = self.spatial_sampling(r, s, wavelist, mask_size)

        # Prepare input field
        self.set_input_plane(r, Nx, Ny)
        E0 = self.pupil
        for wave in wavelist:
            E0 = wave.forward(E0, self.xi_, self.eta_)

        self.fcX = fcX
        self.fcY = fcY
        self.fbX = fbX
        self.fbY = fbY
        self.E0 = E0
        self.s = s
        self.zf = zf
        self.D = 2 * r
        self.type = type
        self.mask_size = mask_size

    def spatial_sampling(self, r, s, wavelist, mask_size):
        """
        Compute spatial sampling parameters.
        
        Args:
            r: Aperture radius
            s: Oversampling factor
            wavelist: List of input wave components
            mask_size: Size of the mask
            
        Returns:
            Tuple of (Nx, Ny, fbX, fbY) - sample counts and bandwidths
        """
        Nx = torch.tensor(mask_size)
        Ny = torch.tensor(mask_size)
        return Nx, Ny, (Nx - 1) / (2 * r), (Ny - 1) / (2 * r)

    def set_input_plane(self, r, Nx, Ny):
        """
        Set up the input plane coordinates and aperture.
        
        Args:
            r: Aperture radius
            Nx, Ny: Number of samples in each dimension
        """
        xi = np.linspace(-r, r, Nx, endpoint=True)
        eta = np.linspace(-r, r, Ny, endpoint=True)
        xi_, eta_ = np.meshgrid(xi, eta, indexing='xy')

        # Circular aperture
        pupil = np.where(xi_**2 + eta_**2 <= r**2, 1, 0)

        self.pupil = pupil
        self.xi, self.eta = xi, eta
        self.xi_, self.eta_ = xi_, eta_
