"""
Output containers for model predictions.

This module defines named tuples for organizing model outputs
in a structured and readable way.
"""

from collections import namedtuple

OutputsContainer = namedtuple(
    'OutputsContainer',
    field_names=['est_images', 'est_dfd', 'est_depthmaps']
)
"""
Container for decoder network outputs.

Fields:
    est_images: Reconstructed RGB images (B, 3, H, W)
    est_dfd: Depth-from-defocus estimates (B, 1, H, W)
    est_depthmaps: Final depth/disparity maps (B, 1, H, W)
"""
