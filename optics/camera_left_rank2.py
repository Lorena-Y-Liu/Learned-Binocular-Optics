"""
Rank-2 DOE (Diffractive Optical Element) camera module for left camera.

This module implements a camera with Rank-2 low-rank DOE parameterization,
enabling efficient optimization of the phase mask with reduced parameters.
The Rank-2 structure represents the height map as a sum of two outer products.
"""

from typing import List, Union

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from ls_asm.LSASM import LeastSamplingASM
from optics.base_camera import BaseCamera
from psf.psf_import import psf_captured
from util.helper import copy_quadruple, ips_to_metric, refractive_index


class MixedCamera(BaseCamera):
    """
    Left camera with Rank-2 parameterized diffractive optical element.
    
    The DOE height map is parameterized as:
        h(x,y) = y_0 * x_0^T + y_1 * x_1^T
    
    This low-rank representation reduces the number of parameters while
    maintaining sufficient expressiveness for depth-from-defocus applications.
    
    Args:
        focal_depth: Focal depth of the camera in meters
        min_depth: Minimum scene depth in meters
        max_depth: Maximum scene depth in meters
        n_depths: Number of discrete depth planes
        image_size: Output image size (int or [H, W])
        mask_size: Size of the DOE mask in pixels
        mask_pitch: Physical pitch of mask pixels in meters
        focal_length: Camera focal length in meters
        mask_diameter: Physical diameter of the DOE mask in meters
        camera_pixel_pitch: Physical size of camera sensor pixels in meters
        wavelengths: List of wavelengths to simulate in meters
        full_size: Full PSF computation size
        mask_upsample_factor: Upsampling factor for mask computation
        diffraction_efficiency: Efficiency of the diffractive element (0-1)
        requires_grad: Whether to enable gradient computation for DOE optimization
    """
    
    # Maximum height of the DOE in meters
    H_MAX = 0.55 / 0.5625 * 1e-6
    
    # PSF observation grid size
    PSF_OBS_SIZE = 256
    
    # Camera sensor pixel pitch in meters
    SENSOR_PIXEL_PITCH = 5.86e-6
    
    def __init__(
        self,
        focal_depth: float,
        min_depth: float,
        max_depth: float,
        n_depths: int,
        image_size: Union[int, List[int]],
        mask_size: int,
        mask_pitch: float,
        focal_length: float,
        mask_diameter: float,
        camera_pixel_pitch: float,
        wavelengths: torch.Tensor = torch.tensor([632e-9, 550e-9, 450e-9]),
        full_size: int = 100,
        mask_upsample_factor: int = 1,
        diffraction_efficiency: float = 0.7,
        requires_grad: bool = True,
        use_pretrained_doe: bool = True
    ):
        self.diffraction_efficiency = diffraction_efficiency
        super().__init__(
            focal_depth, min_depth, max_depth, n_depths,
            image_size, mask_size, mask_pitch, focal_length,
            mask_diameter, camera_pixel_pitch, wavelengths,
            full_size, mask_upsample_factor, requires_grad
        )
        
        self.full_size = full_size
        self.mask_upsample_factor = mask_upsample_factor
        self.modulate_phase = requires_grad
        
        # Initialize DOE parameters (load from pretrained vectors or use analytical init)
        self._init_doe_parameters(requires_grad, use_pretrained_doe)
        
        # Initialize propagation model
        self.Uin, self.E0 = self.propagation(self.scene_distances, self.modulate_phase)
        
        # Store propagation parameters
        self.wvls = self.wavelengths
        self.k = 2 * torch.pi / self.wavelengths
        self.s = 1.0
        self.zf = 1 / (1 / self.focal_length - 1 / self.focal_depth)
        self.D = self.focal_length / self.f_number
        self.pupil = self.Uin.pupil
        
        # Store frequency coordinates from input field
        self.fcX, self.fcY = self.Uin.fcX, self.Uin.fcY
        self.fbX, self.fbY = self.Uin.fbX, self.Uin.fbY
        self.xi, self.eta = self.Uin.xi, self.Uin.eta
        self.xi_, self.eta_ = self.Uin.xi_, self.Uin.eta_

    def _init_doe_parameters(self, requires_grad: bool, use_pretrained_doe: bool):
        """Initialize the Rank-2 DOE parameters from pretrained vectors or analytically."""
        if use_pretrained_doe:
            # Load pretrained DOE vectors for left camera
            init_x_0 = torch.load('doe_vector/x_0_l.pth')
            init_y_0 = torch.load('doe_vector/y_0_l.pth')
            init_x_1 = torch.load('doe_vector/x_1_l.pth')
            init_y_1 = torch.load('doe_vector/y_1_l.pth')
        else:
            # Initialize DOE vectors analytically for left camera
            n_param = self.mask_size // (2 * self.mask_upsample_factor)
            n = torch.tensor(n_param, dtype=torch.float32)
            y = torch.arange(n_param, dtype=torch.float32)
            
            init_x_0 = 2 * torch.ones(n_param).float()
            init_x_1 = 4 * (1 - (torch.square(y)) / torch.square(n)).float()
            init_y_0 = 2 * (1 - (torch.square(y)) / torch.square(n)).float()
            init_y_1 = 4 * torch.ones(n_param).float()
        
        # Register as learnable parameters
        self.heightmap1d_x_0 = nn.Parameter(init_x_0, requires_grad=requires_grad)
        self.heightmap1d_y_0 = nn.Parameter(init_y_0, requires_grad=requires_grad)
        self.heightmap1d_x_1 = nn.Parameter(init_x_1, requires_grad=requires_grad)
        self.heightmap1d_y_1 = nn.Parameter(init_y_1, requires_grad=requires_grad)
        
    def _upsample_1d(self, vec: torch.Tensor) -> torch.Tensor:
        """Upsample a 1D vector by the mask upsample factor."""
        return F.interpolate(
            vec.reshape(1, 1, -1),
            scale_factor=self.mask_upsample_factor,
            mode='nearest'
        ).reshape(-1)

    def heightmap1d_x(self) -> torch.Tensor:
        """Get upsampled x-component of first rank term."""
        return self._upsample_1d(self.heightmap1d_x_0)

    def heightmap1d_y(self) -> torch.Tensor:
        """Get upsampled y-component of first rank term."""
        return self._upsample_1d(self.heightmap1d_y_0)

    def heightmap1d_x1(self) -> torch.Tensor:
        """Get upsampled x-component of second rank term."""
        return self._upsample_1d(self.heightmap1d_x_1)

    def heightmap1d_y1(self) -> torch.Tensor:
        """Get upsampled y-component of second rank term."""
        return self._upsample_1d(self.heightmap1d_y_1)

    def heightmap2d(self) -> torch.Tensor:
        """
        Compute the 2D height map as sum of two outer products.
        
        Returns:
            2D height map tensor of shape [H/2, W/2] (before quadruple copy)
        """
        term1 = torch.matmul(
            self.heightmap1d_y().reshape(-1, 1),
            self.heightmap1d_x().reshape(1, -1)
        )
        term2 = torch.matmul(
            self.heightmap1d_y1().reshape(-1, 1),
            self.heightmap1d_x1().reshape(1, -1)
        )
        return term1 + term2

    def _rotation_matrix(self, theta: float, device: torch.device) -> torch.Tensor:
        """Create 2D rotation matrix for affine transformation."""
        cos_t = torch.cos(torch.tensor(theta, device=device))
        sin_t = torch.sin(torch.tensor(theta, device=device))
        return torch.tensor([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0]
        ], device=device)

    def _rotate_height(self, x: torch.Tensor, theta: float) -> torch.Tensor:
        """Apply rotation to height map using grid sampling."""
        device = x.device
        rot_mat = self._rotation_matrix(theta, device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(rot_mat, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    def aperture(self) -> torch.Tensor:
        """Create circular aperture mask."""
        x = torch.arange(self.mask_size, device=self.device).float()
        y = torch.arange(self.mask_size, device=self.device).float()
        X, Y = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt((X + 0.5 - self.mask_size // 2) ** 2 + 
                       (Y + 0.5 - self.mask_size // 2) ** 2)
        return torch.where(r < self.mask_size // 2, 1, 0).float()

    def height(self) -> torch.Tensor:
        """
        Compute the full DOE height map.
        
        The height map is:
        1. Computed from rank-2 representation
        2. Copied to all four quadrants (symmetry)
        3. Rotated by 45 degrees (for left camera)
        4. Wrapped to [0, 1] range
        5. Masked by circular aperture
        
        Returns:
            Height map tensor [mask_size, mask_size] in [0, 1] range
        """
        heightmap2d = self.heightmap2d()
        # Get the device from the computed heightmap (follows parameter device)
        device = heightmap2d.device
        heightmap2d = copy_quadruple(heightmap2d.unsqueeze(0).unsqueeze(0))
        heightmap2d = self._rotate_height(heightmap2d, torch.pi / 4)  # +45° for left
        height_map = heightmap2d.squeeze(0).squeeze(0)
        height_map = torch.remainder(height_map, 1)
        # Ensure aperture is on the same device as height_map
        aperture = self.aperture().to(device)
        height_map = height_map * aperture
        return height_map

    def phase(self) -> torch.Tensor:
        """
        Compute the phase map from height map.
        
        Returns:
            Phase map tensor [n_wl, 1, mask_size, mask_size] in [0, 2π] range
        """
        heightmap = torch.remainder(self.height(), 1)
        device = heightmap.device
        wavelengths = self.wavelengths.to(device).reshape(-1, 1, 1, 1)
        k = 2 * torch.pi / wavelengths
        n = refractive_index(wavelengths)
        phase = heightmap * k * self.H_MAX * (n - 1)
        return torch.remainder(phase, 2 * torch.pi)

    def through_plate(self, Ein: torch.Tensor, heightmap: torch.Tensor) -> torch.Tensor:
        """
        Propagate field through the DOE.
        
        Args:
            Ein: Input electric field
            heightmap: Height map of the DOE
            
        Returns:
            Output electric field after phase modulation
        """
        device = heightmap.device
        heightmap = heightmap.unsqueeze(0).unsqueeze(0)
        wavelengths = self.wavelengths.to(device).reshape(-1, 1, 1, 1)
        k = 2 * torch.pi / wavelengths
        n = refractive_index(wavelengths)
        phase = heightmap * k * self.H_MAX * (n - 1)
        Ein = Ein.to(device)
        return Ein * torch.exp(1j * phase)

    def psf_obs(self, Ein: torch.Tensor) -> torch.Tensor:
        """
        Compute PSF at observation plane using LS-ASM.
        
        Args:
            Ein: Input electric field
            
        Returns:
            PSF intensity [n_wl, n_depths, H, W]
        """
        device = Ein.device
        Mx, My = self.PSF_OBS_SIZE, self.PSF_OBS_SIZE
        l = self.SENSOR_PIXEL_PITCH * Mx
        z = 1 / (1 / self.focal_length - 1 / self.focal_depth)
        
        x = torch.linspace(-l / 2, l / 2, Mx, device=device)
        y = torch.linspace(-l / 2, l / 2, My, device=device)
        
        prop = LeastSamplingASM(self, x, y, z, device)
        U2 = prop(Ein)
        self.psf_phase = torch.remainder(torch.angle(U2), 2 * torch.pi)
        return torch.abs(U2) ** 2

    def psf_ph(self) -> torch.Tensor:
        """Get the phase of the PSF (for analysis)."""
        return self.psf_phase

    def psf_obs_full(self, Ein: torch.Tensor) -> torch.Tensor:
        """Compute full-resolution PSF for regularization."""
        device = Ein.device
        Mx, My = self.full_size, self.full_size
        l = self.SENSOR_PIXEL_PITCH * Mx
        z = 1 / (1 / self.focal_length - 1 / self.focal_depth)
        
        x = torch.linspace(-l / 2, l / 2, Mx, device=device)
        y = torch.linspace(-l / 2, l / 2, My, device=device)
        
        prop = LeastSamplingASM(self, x, y, z, device)
        U2 = prop(Ein)
        return torch.abs(U2) ** 2

    def psf_full(self, modulate_phase: bool) -> torch.Tensor:
        """Compute full PSF with or without phase modulation."""
        heightmap = self.height()
        device = heightmap.device
        E0 = self.E0.to(device)
        if modulate_phase:
            Ein = self.through_plate(E0, heightmap)
        else:
            Ein = E0
        return F.relu(self.psf_obs_full(Ein))

    def psf_at_camera(
        self,
        size: tuple,
        modulate_phase: bool,
        is_training: bool = False
    ) -> torch.Tensor:
        """
        Compute PSF at camera sensor with optional augmentation.
        
        Args:
            size: Output size (H, W)
            modulate_phase: Whether to apply DOE phase modulation
            is_training: Whether to apply training augmentations
            
        Returns:
            PSF tensor [C, D, H, W]
        """
        if not self.experiment:
            heightmap = self.height()
            device = heightmap.device
            
            # Optionally randomize depth sampling during training
            if is_training:
                scene_distances = ips_to_metric(
                    torch.linspace(0, 1, steps=self.n_depths, device=device) +
                    1 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5),
                    self.min_depth, self.max_depth
                )
                scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)
            
            # Compute PSF with phase modulation
            E0 = self.E0.to(device)
            Ein = self.through_plate(E0, heightmap)
            diffracted_psf = F.relu(self.psf_obs(Ein))
            undiffracted_psf = F.relu(self.psf_obs(E0))
            
            # Normalize PSFs
            self.diff_normalization_scaler = diffracted_psf.sum(dim=(-1, -2), keepdim=True)
            self.undiff_normalization_scaler = undiffracted_psf.sum(dim=(-1, -2), keepdim=True)
            
            diffracted_psf = diffracted_psf / self.diff_normalization_scaler
            undiffracted_psf = undiffracted_psf / self.undiff_normalization_scaler
            
            # Combine diffracted and undiffracted PSFs
            psf = (self.diffraction_efficiency * diffracted_psf + 
                   (1 - self.diffraction_efficiency) * undiffracted_psf)
        else:
            # Use captured PSF for experiments
            device = self.heightmap1d_x_0.device
            psf = psf_captured(device)[0].squeeze(0).double()

        # Apply training augmentations
        if is_training:
            psf = transforms.RandomRotation(1)(psf)
            max_shift = 2
            r_shift = tuple(torch.randint(-max_shift, max_shift, (2,)))
            b_shift = tuple(torch.randint(-max_shift, max_shift, (2,)))
            psf_r = torch.roll(psf[0], shifts=r_shift, dims=(-1, -2))
            psf_g = psf[1]
            psf_b = torch.roll(psf[2], shifts=b_shift, dims=(-1, -2))
            psf = torch.stack([psf_r, psf_g, psf_b], dim=0)

        # Apply Gaussian blur to PSF
        psf = self._blur_psf(psf, kernel_size=5, sigma=0.5)
        
        # Crop to target size
        psf = transforms.CenterCrop(size)(psf)
        return psf.squeeze(0)

    def _blur_psf(
        self,
        psf: torch.Tensor,
        kernel_size: int = 5,
        sigma: float = 0.5
    ) -> torch.Tensor:
        """Apply Gaussian blur to PSF for realistic simulation."""
        device = psf.device
        dtype = psf.dtype
        c, d, h, w = psf.shape
        kernel = torch.tensor(cv2.getGaussianKernel(kernel_size, sigma), device=device, dtype=dtype)
        kernel_2d = kernel * kernel.T
        
        for i in range(c):
            for j in range(d):
                psf[[i], [j], ...] = F.conv2d(
                    psf[[i], [j], ...],
                    kernel_2d.expand(1, 1, kernel_size, kernel_size),
                    padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2)
                )
        return psf

    def psf_out_of_fov_energy(self, psf_size: int) -> torch.Tensor:
        """
        Compute PSF energy outside the field of view for regularization.
        
        This encourages the PSF to be compact and within the valid region.
        
        Args:
            psf_size: Size parameter (unused, kept for API compatibility)
            
        Returns:
            Scalar loss value for out-of-FOV energy
        """
        psf_diffracted = self.psf_full(self.modulate_phase)
        device = psf_diffracted.device
        
        try:
            psf_diffracted = psf_diffracted / self.diff_normalization_scaler.to(device)
        except AttributeError:
            pass
        
        # Create circular mask to exclude center region
        mask = torch.ones_like(psf_diffracted)
        center = mask.shape[-1] // 2
        
        x = torch.arange(2 * center, device=device).float()
        y = torch.arange(2 * center, device=device).float()
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        dist = torch.sqrt((X + 0.5 - center) ** 2 + (Y + 0.5 - center) ** 2)
        outer_mask = torch.where(dist > 10, 1, 0)
        mask[..., :, :] = outer_mask
        
        # Compute out-of-FOV energy
        psf_out_of_fov = (psf_diffracted * mask).float()
        return psf_out_of_fov.sum() / 10

    def forward_train(
        self,
        img: torch.Tensor,
        depthmap: torch.Tensor,
        occlusion: bool
    ) -> tuple:
        """Forward pass with training augmentations enabled."""
        return self.forward(
            img, depthmap, occlusion,
            is_training=True,
            modulate_phase=self.modulate_phase
        )

    def set_diffraction_efficiency(self, de: float):
        """Set the diffraction efficiency of the DOE."""
        self.diffraction_efficiency = de
