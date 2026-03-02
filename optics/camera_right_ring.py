"""
Ring-coded DOE (Diffractive Optical Element) camera module for right camera.

This module implements a camera with a radially symmetric ring-coded DOE.
The DOE height map is parameterized by a 1D radial profile that is expanded
to 2D via radial interpolation.

When use_pretrained_doe=True, the DOE height map is loaded from a .mat file.
When use_pretrained_doe=False, the DOE is parameterized by a learnable 1D
radial profile vector.
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
    Right camera with ring-coded diffractive optical element.

    The DOE height map is parameterized by a 1D radial profile that is
    radially symmetric. The profile is expanded to 2D via linear
    interpolation in the radial direction.

    When use_pretrained_doe is True, the height map is loaded directly from
    a .mat file (does/Ring/Ring_R.mat) and is not optimizable.
    When False, a learnable 1D radial profile is used.
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
        self.use_pretrained_doe = use_pretrained_doe

        # PSF cache for validation
        self._psf_cache = None
        self._psf_cache_size = None

        # Initialize DOE parameters (always created; used when use_pretrained_doe=False)
        self._init_doe_parameters(requires_grad)

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

    def _init_doe_parameters(self, requires_grad: bool):
        """Initialize the ring DOE parameter (1D radial profile)."""
        n_param = self.mask_size // (2 * self.mask_upsample_factor)
        init_x = torch.zeros(n_param).float()
        self.heightmap_x_0 = nn.Parameter(init_x, requires_grad=requires_grad)

    def _upsample_1d(self, vec: torch.Tensor) -> torch.Tensor:
        """Upsample a 1D vector by the mask upsample factor."""
        return F.interpolate(
            vec.reshape(1, 1, -1),
            scale_factor=self.mask_upsample_factor,
            mode='nearest'
        ).reshape(-1)

    def heightmap_x(self) -> torch.Tensor:
        """Get upsampled radial profile."""
        return self._upsample_1d(self.heightmap_x_0)

    def heightmap2d(self) -> torch.Tensor:
        """
        Convert 1D radial profile to 2D height map via linear interpolation.

        The 1D profile is sampled at radial distances for each pixel in a
        quarter-plane, then copy_quadruple is used to fill all 4 quadrants.
        """
        profile = self.heightmap_x()  # length: mask_size // 2
        device = profile.device

        # Pad profile with zeros to cover full diagonal range
        profile_full = torch.cat([
            profile,
            torch.zeros(self.mask_size // 2, device=device, dtype=profile.dtype)
        ], dim=0)

        # Quarter-plane coordinates
        half = self.mask_size // 2
        y_coord = torch.arange(half, device=device, dtype=torch.float32) + 0.5
        x_coord = torch.arange(half, device=device, dtype=torch.float32) + 0.5
        Y, X = torch.meshgrid(y_coord, x_coord, indexing='ij')
        r_coord = torch.sqrt(X ** 2 + Y ** 2)

        # Linear interpolation of 1D profile at radial distances
        r_floor = r_coord.long().clamp(0, len(profile_full) - 2)
        r_frac = (r_coord - r_floor.float()).clamp(0, 1)
        heightmap_quarter = (
            profile_full[r_floor] * (1 - r_frac) +
            profile_full[r_floor + 1] * r_frac
        )

        # Copy to all 4 quadrants
        return copy_quadruple(heightmap_quarter.unsqueeze(0).unsqueeze(0)).squeeze()

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

        When use_pretrained_doe=True: loads from does/Ring/Ring_R.mat
        When use_pretrained_doe=False: uses learnable 1D radial profile
            1. Radial interpolation to 2D
            2. Wrap to [0, 1] range
            3. Mask by circular aperture
        """
        if self.use_pretrained_doe:
            import scipy.io as scio
            height_map = scio.loadmat('does/Ring/Ring_R.mat')['phase']
            height_map = (torch.from_numpy(height_map) / (2 * torch.pi)).to(self.device).float()
            if self.mask_upsample_factor != 1:
                height_map = F.interpolate(
                    height_map.unsqueeze(0).unsqueeze(0),
                    scale_factor=self.mask_upsample_factor, mode='nearest'
                ).squeeze(0).squeeze(0)
            height_map = height_map * self.aperture()
            return height_map
        else:
            heightmap2d = self.heightmap2d()
            device = heightmap2d.device
            height_map = torch.remainder(heightmap2d, 1)
            height_map = height_map * self.aperture().to(device)
            return height_map

    def phase(self) -> torch.Tensor:
        """Compute the phase map from height map."""
        heightmap = torch.remainder(self.height(), 1)
        device = heightmap.device
        wavelengths = self.wavelengths.to(device).reshape(-1, 1, 1, 1)
        k = 2 * torch.pi / wavelengths
        n = refractive_index(wavelengths)
        phase = heightmap * k * self.H_MAX * (n - 1)
        return torch.remainder(phase, 2 * torch.pi)

    def through_plate(self, Ein: torch.Tensor, heightmap: torch.Tensor) -> torch.Tensor:
        """Propagate field through the DOE."""
        device = heightmap.device
        heightmap = heightmap.unsqueeze(0).unsqueeze(0)
        wavelengths = self.wavelengths.to(device).reshape(-1, 1, 1, 1)
        k = 2 * torch.pi / wavelengths
        n = refractive_index(wavelengths)
        phase = heightmap * k * self.H_MAX * (n - 1)
        Ein = Ein.to(device)
        return Ein * torch.exp(1j * phase)

    def psf_obs(self, Ein: torch.Tensor) -> torch.Tensor:
        """Compute PSF at observation plane using LS-ASM."""
        device = Ein.device
        Mx, My = self.PSF_OBS_SIZE, self.PSF_OBS_SIZE
        l = self.SENSOR_PIXEL_PITCH * Mx
        z = 1 / (1 / self.focal_length - 1 / self.focal_depth)
        x = torch.linspace(-l / 2, l / 2, Mx, device=device)
        y = torch.linspace(-l / 2, l / 2, My, device=device)
        prop = LeastSamplingASM(self, x, y, z, device)
        U2 = prop(Ein)
        self.psf_phase = torch.remainder(torch.angle(U2), 2 * torch.pi)
        result = torch.abs(U2) ** 2
        del prop, U2, x, y
        return result

    def psf_ph(self) -> torch.Tensor:
        """Get the phase of the PSF."""
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
        result = torch.abs(U2) ** 2
        del prop, U2, x, y
        torch.cuda.empty_cache()
        return result

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

    def clear_psf_cache(self):
        """Clear the cached PSF."""
        self._psf_cache = None
        self._psf_cache_size = None

    def psf_at_camera(self, size: tuple, modulate_phase: bool, is_training: bool = False) -> torch.Tensor:
        """Compute PSF at camera sensor with optional augmentation."""
        if not is_training and self._psf_cache is not None and self._psf_cache_size == size:
            return self._psf_cache.clone()

        if not self.experiment:
            heightmap = self.height()
            device = heightmap.device

            if is_training:
                scene_distances = ips_to_metric(
                    torch.linspace(0, 1, steps=self.n_depths, device=device) +
                    1 / self.n_depths * (torch.rand(self.n_depths, device=device) - 0.5),
                    self.min_depth, self.max_depth
                )
                scene_distances[-1] += torch.rand(1, device=device)[0] * (100.0 - self.max_depth)

            E0 = self.E0.to(device)
            Ein = self.through_plate(E0, heightmap)
            diffracted_psf = F.relu(self.psf_obs(Ein))
            undiffracted_psf = F.relu(self.psf_obs(E0))

            self.diff_normalization_scaler = diffracted_psf.sum(dim=(-1, -2), keepdim=True)
            self.undiff_normalization_scaler = undiffracted_psf.sum(dim=(-1, -2), keepdim=True)

            diffracted_psf = diffracted_psf / self.diff_normalization_scaler
            undiffracted_psf = undiffracted_psf / self.undiff_normalization_scaler

            psf = (self.diffraction_efficiency * diffracted_psf +
                   (1 - self.diffraction_efficiency) * undiffracted_psf)
        else:
            device = self.heightmap_x_0.device
            psf = psf_captured(device)[1].squeeze(0).double()

        # Training augmentations
        if is_training:
            psf = transforms.RandomRotation(3)(psf)
            max_shift = 2
            r_shift = tuple(torch.randint(-max_shift, max_shift, (2,)))
            b_shift = tuple(torch.randint(-max_shift, max_shift, (2,)))
            psf_r = torch.roll(psf[0], shifts=r_shift, dims=(-1, -2))
            psf_g = psf[1]
            psf_b = torch.roll(psf[2], shifts=b_shift, dims=(-1, -2))
            psf = torch.stack([psf_r, psf_g, psf_b], dim=0)

        # Gaussian blur
        psf = self._blur_psf(psf, kernel_size=5, sigma=0.5)

        # Crop to target size
        psf = transforms.CenterCrop(size)(psf)
        result = psf.squeeze(0)

        if not is_training:
            self._psf_cache = result.clone().detach()
            self._psf_cache_size = size

        return result

    def _blur_psf(self, psf: torch.Tensor, kernel_size: int = 5, sigma: float = 0.5) -> torch.Tensor:
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
        """Compute PSF energy outside the field of view for regularization."""
        psf_diffracted = self.psf_full(self.modulate_phase)
        device = psf_diffracted.device
        try:
            psf_diffracted = psf_diffracted / self.diff_normalization_scaler.to(device)
        except AttributeError:
            pass

        mask = torch.ones_like(psf_diffracted)
        center = mask.shape[-1] // 2
        x = torch.arange(2 * center, device=device).float()
        y = torch.arange(2 * center, device=device).float()
        X, Y = torch.meshgrid(x, y, indexing='ij')
        dist = torch.sqrt((X + 0.5 - center) ** 2 + (Y + 0.5 - center) ** 2)
        outer_mask = torch.where(dist > 10, 1, 0)
        mask[..., :, :] = outer_mask

        psf_out_of_fov = (psf_diffracted * mask).float()
        result = psf_out_of_fov.sum() / 10
        del psf_diffracted, mask, X, Y, dist, outer_mask, psf_out_of_fov
        return result

    def forward_train(self, img: torch.Tensor, depthmap: torch.Tensor, occlusion: bool) -> tuple:
        """Forward pass with training augmentations enabled."""
        return self.forward(
            img, depthmap, occlusion,
            is_training=True,
            modulate_phase=self.modulate_phase
        )

    def set_diffraction_efficiency(self, de: float):
        """Set the diffraction efficiency of the DOE."""
        self.diffraction_efficiency = de
