"""
Base camera module for deep stereo optical system simulation.

This module provides the BaseCamera abstract class that handles:
- Wavelength-dependent light propagation
- Point Spread Function (PSF) computation
- Image capture simulation with occlusion handling
- Depth-dependent blur simulation
"""

import abc
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ls_asm.input_field import InputField
from util import complex
from util.fft import fftshift
from util.helper import depthmap_to_layereddepth, ips_to_metric, over_op, refractive_index


# Handle PyTorch version compatibility for FFT functions
try:
    from torch import irfft, rfft
except ImportError:
    def rfft(x, d):
        """Real FFT with backward compatibility."""
        t = torch.fft.fft2(x, dim=(-d, -1))
        return torch.stack((t.real, t.imag), -1)

    def irfft(x, d, signal_sizes):
        """Inverse real FFT with backward compatibility."""
        t = torch.fft.ifft2(torch.complex(x[..., 0], x[..., 1]), dim=(-d, -1))
        return t.real


class BaseCamera(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for camera optical system simulation.
    
    This class models a camera with diffractive optical elements (DOE) and handles:
    - Multi-wavelength light propagation
    - Depth-dependent PSF computation
    - Image formation with optional occlusion handling
    
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
        require_grads: Whether to enable gradient computation for DOE optimization
    """

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
        wavelengths: Union[List[float], torch.Tensor],
        full_size: int = 1920,
        mask_upsample_factor: int = 1,
        diffraction_efficiency: float = 0.7,
        require_grads: bool = False,
        **kwargs
    ):
        super().__init__()
        
        assert min_depth > 1e-6, f'Minimum depth is too small. min_depth: {min_depth}'
        
        # Store device first
        self.device = 'cuda'
        
        # Compute scene distances using inverse perspective sampling
        scene_distances = ips_to_metric(
            torch.linspace(0, 1, steps=n_depths, device=self.device), min_depth, max_depth
        )
        scene_distances = scene_distances.flip(-1)
        
        # Initialize wavelengths
        self.modulate_phase = require_grads
        self._register_wavelength(wavelengths)
        
        # Store camera parameters
        self.n_depths = n_depths
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.focal_depth = focal_depth
        self.mask_diameter = mask_diameter
        self.camera_pixel_pitch = camera_pixel_pitch
        self.focal_length = focal_length
        self.f_number = self.focal_length / self.mask_diameter
        self.image_size = self._normalize_image_size(image_size)
        self.mask_pitch = mask_pitch
        self.mask_size = mask_size
        self.full_size = full_size
        self.experiment = False
        
        self.register_buffer('scene_distances', scene_distances)

    def _register_wavelength(self, wavelengths: Union[List[float], float, torch.Tensor]):
        """
        Register wavelengths as a buffer for the model.
        
        Args:
            wavelengths: Wavelengths in meters (float, list, or tensor)
        """
        if isinstance(wavelengths, list):
            wavelengths = torch.tensor(wavelengths, device=self.device)
        elif isinstance(wavelengths, float):
            wavelengths = torch.tensor([wavelengths], device=self.device)
        elif isinstance(wavelengths, torch.Tensor):
            wavelengths = wavelengths.to(self.device)
        else:
            raise ValueError('wavelengths must be a float, list of floats, or tensor.')

        if len(wavelengths) % 3 != 0:
            raise ValueError('Number of wavelengths must be a multiple of 3 (for RGB channels).')

        self.n_wl = len(wavelengths)
        if not hasattr(self, 'wavelengths'):
            self.register_buffer('wavelengths', wavelengths)
        else:
            self.wavelengths = wavelengths.to(self.wavelengths.device)

    def sensor_distance(self) -> float:
        """Calculate the sensor distance from the lens using the thin lens equation."""
        return 1.0 / (1.0 / self.focal_length - 1.0 / self.focal_depth)

    def normalize_psf(self, psfimg: torch.Tensor) -> torch.Tensor:
        """
        Normalize PSF so that total energy equals 1.
        
        Args:
            psfimg: PSF tensor of shape [..., H, W]
            
        Returns:
            Normalized PSF tensor
        """
        return psfimg / psfimg.sum(dim=(-2, -1), keepdims=True)

    def _capture_impl(
        self,
        volume: torch.Tensor,
        layered_depth: torch.Tensor,
        psf: torch.Tensor,
        occlusion: bool,
        eps: float = 1e-3
    ) -> tuple:
        """
        Core implementation of image capture simulation.
        
        Args:
            volume: Layered volume tensor [B, C, D, H, W]
            layered_depth: Depth layer masks [B, 1, D, H, W]
            psf: Point spread function [B, C, D, H, W]
            occlusion: Whether to simulate occlusion effects
            eps: Small constant for numerical stability
            
        Returns:
            Tuple of (captured_image, volume)
        """
        scale = volume.max()
        volume = volume / scale
        Fpsf = rfft(psf, 2)
        
        if occlusion:
            Fvolume = rfft(volume, 2)
            Flayered_depth = rfft(layered_depth, 2)
            
            # Blur the volume and depth with PSF
            blurred_alpha_rgb = irfft(
                complex.multiply(Flayered_depth, Fpsf), 2, signal_sizes=volume.shape[-2:]
            )
            blurred_volume = irfft(
                complex.multiply(Fvolume, Fpsf), 2, signal_sizes=volume.shape[-2:]
            )
            
            # Normalize the blurred intensity for proper occlusion handling
            cumsum_alpha = torch.flip(
                torch.cumsum(torch.flip(layered_depth, dims=(-3,)), dim=-3), dims=(-3,)
            )
            Fcumsum_alpha = rfft(cumsum_alpha, 2)
            blurred_cumsum_alpha = irfft(
                complex.multiply(Fcumsum_alpha, Fpsf), 2, signal_sizes=volume.shape[-2:]
            )
            
            blurred_volume = blurred_volume / (blurred_cumsum_alpha + eps)
            blurred_alpha_rgb = blurred_alpha_rgb / (blurred_cumsum_alpha + eps)
            over_alpha = over_op(blurred_alpha_rgb)
            captimg = torch.sum(over_alpha * blurred_volume, dim=2)
        else:
            Fvolume = rfft(volume, 2)
            Fcaptimg = complex.multiply(Fvolume, Fpsf).sum(dim=2)
            captimg = irfft(Fcaptimg, 2, signal_sizes=volume.shape[-2:])
            
        captimg = scale * captimg
        volume = scale * volume
        return captimg, volume

    def _capture_from_rgbd_with_psf_impl(
        self,
        img: torch.Tensor,
        depthmap: torch.Tensor,
        psf: torch.Tensor,
        occlusion: bool
    ) -> tuple:
        """
        Capture image from RGB-D input with given PSF.
        
        Args:
            img: Input RGB image [B, C, H, W]
            depthmap: Depth map [B, 1, H, W]
            psf: Point spread function
            occlusion: Whether to simulate occlusion
            
        Returns:
            Tuple of (captured_image, volume)
        """
        layered_depth = depthmap_to_layereddepth(depthmap, self.n_depths, binary=True)
        volume = layered_depth * img[:, :, None, ...]
        return self._capture_impl(volume, layered_depth, psf, occlusion)

    def capture_from_rgbd(
        self,
        img: torch.Tensor,
        depthmap: torch.Tensor,
        occlusion: bool
    ) -> torch.Tensor:
        """
        Capture image from RGB-D input.
        
        Args:
            img: Input RGB image [B, C, H, W]
            depthmap: Depth map [B, 1, H, W]
            occlusion: Whether to simulate occlusion
            
        Returns:
            Captured image tensor
        """
        psf = self.psf_at_camera(img.shape[-2:], self.modulate_phase)
        psf = fftshift(self.normalize_psf(psf), dims=(-1, -2))
        return self.capture_from_rgbd_with_psf(img, depthmap, psf, occlusion)

    def capture_from_rgbd_with_psf(
        self,
        img: torch.Tensor,
        depthmap: torch.Tensor,
        psf: torch.Tensor,
        occlusion: bool
    ) -> torch.Tensor:
        """Capture image with explicit PSF."""
        return self._capture_from_rgbd_with_psf_impl(img, depthmap, psf, occlusion)[0]

    @abc.abstractmethod
    def psf_at_camera(
        self,
        size: tuple,
        modulate_phase: bool,
        is_training: bool = False
    ) -> torch.Tensor:
        """
        Compute PSF at camera sensor.
        
        Args:
            size: Output PSF size (H, W)
            modulate_phase: Whether to apply phase modulation
            is_training: Whether in training mode (enables augmentation)
            
        Returns:
            PSF tensor [C, D, H, W]
        """
        pass

    @abc.abstractmethod
    def height(self) -> torch.Tensor:
        """
        Get the height map of the diffractive optical element.
        
        Returns:
            Height map tensor [H, W]
        """
        pass

    def forward(
        self,
        img: torch.Tensor,
        depthmap: torch.Tensor,
        occlusion: bool,
        modulate_phase: bool,
        is_training: bool = False
    ) -> tuple:
        """
        Forward pass: simulate image capture.
        
        Args:
            img: Input RGB image [B, C, H, W]
            depthmap: Depth map [B, 1, H, W]
            occlusion: Whether to simulate occlusion
            modulate_phase: Whether to apply DOE phase modulation
            is_training: Whether in training mode
            
        Returns:
            Tuple of (captured_image, volume, psf)
        """
        psf = self.psf_at_camera(
            img.shape[-2:], modulate_phase, is_training=is_training
        ).unsqueeze(0)
        
        psf = fftshift(self.normalize_psf(psf), dims=(-1, -2))
        captimg, volume = self._capture_from_rgbd_with_psf_impl(img, depthmap, psf, occlusion)
        return captimg, volume, psf

    def _normalize_image_size(self, image_size: Union[int, List[int]]) -> List[int]:
        """Normalize image size to [H, W] format."""
        if isinstance(image_size, int):
            image_size = [image_size, image_size]
        elif isinstance(image_size, list):
            if image_size[0] % 2 == 1 or image_size[1] % 2 == 1:
                raise ValueError('Image size must be even.')
        else:
            raise ValueError('image_size must be int or list of int.')
        return image_size

    def set_image_size(self, image_size: Union[int, List[int]]):
        """Set the output image size."""
        self.image_size = self._normalize_image_size(image_size)

    def set_wavelengths(self, wavelengths: Union[List[float], torch.Tensor]):
        """Set the wavelengths for simulation."""
        self._register_wavelength(wavelengths)

    def set_n_depths(self, n_depths: int):
        """Set the number of depth planes."""
        self.n_depths = n_depths

    def propagation(
        self,
        scene_distances: torch.Tensor,
        modulate_phase: bool
    ) -> tuple:
        """
        Initialize light propagation model.
        
        Args:
            scene_distances: Scene depth values
            modulate_phase: Whether to apply phase modulation
            
        Returns:
            Tuple of (InputField, E0 field)
        """
        r = self.focal_length / self.f_number / 2
        self.k = 2 * torch.pi / self.wavelengths
        self.modulate_phase = modulate_phase
        self.s_LSASM = 1
        self.thetaX = 0
        self.thetaY = 0
        
        wavelength = self.wavelengths.reshape(-1).double()
        zf = 1 / (1 / self.focal_length - 1 / self.focal_depth)
        
        Uin = InputField(
            "12", wavelength, r, scene_distances,
            self.focal_length, zf, self.mask_size,
            False, None
        )
        E0 = Uin.E0
        return Uin, E0

    def extra_repr(self) -> str:
        """String representation of camera parameters."""
        return (
            f'Camera module\n'
            f'  Refractive index (center wavelength): '
            f'{refractive_index(self.wavelengths[self.n_wl // 2]):.4f}\n'
            f'  Mask pitch: {self.mask_pitch * 1e6:.2f} [um]\n'
            f'  f-number: {self.f_number:.2f}\n'
            f'  Focal depth: {self.focal_depth:.2f} [m]\n'
            f'  Depth range: {self.min_depth:.2f} - {self.max_depth:.2f} [m]\n'
            f'  Input image size: {self.image_size}'
        )
