"""
Optics module for deep stereo camera simulation.

This module provides camera models with different diffractive optical elements (DOE)
parameterizations for depth-from-defocus stereo vision.

Available camera types:
- Rank1: Single outer product DOE parameterization  
- Rank2: Sum of two outer products DOE parameterization
- Ring: Radially symmetric ring-coded DOE

Each camera type has left and right variants for stereo setup.
"""

from optics.base_camera import BaseCamera

__all__ = ['BaseCamera']
