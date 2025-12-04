"""
Example: Visualize DOE Phase Masks and PSFs

This script demonstrates how to:
1. Load camera models with different DOE types
2. Visualize the phase masks
3. Compute and display PSFs at different depths
"""

import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from optics.camera_left_rank2 import MixedCamera


def main():
    # Camera configuration
    config = {
        'wavelengths': [632e-9, 550e-9, 450e-9],  # RGB wavelengths
        'min_depth': 0.67,
        'max_depth': 8.0,
        'focal_depth': 1.23,
        'n_depths': 7,
        'image_size': [256, 512],
        'camera_pixel_pitch': 5.86e-6,
        'focal_length': 35e-3,
        'mask_diameter': 4.347e-3,
        'mask_size': 1260,
        'mask_pitch': 3.45e-6,
        'mask_upsample_factor': 2,
        'diffraction_efficiency': 0.7,
        'full_size': 1200,
        'use_pretrained_doe': False,
    }
    
    # Create camera model
    print("Creating camera model...")
    camera = MixedCamera(**config, requires_grad=False)
    camera.eval()
    
    # Get phase mask
    print("Extracting phase mask...")
    with torch.no_grad():
        phase = camera.phase()  # Shape: (C, 1, H, W)
    
    # Get PSFs at different depths
    print("Computing PSFs...")
    with torch.no_grad():
        psf = camera.psf_at_camera(size=(64, 64), is_training=False, modulate_phase=False)
        # Shape: (C, D, H, W) - Color, Depth, Height, Width
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Plot phase mask (green channel)
    ax = axes[0, 0]
    phase_img = phase[1, 0].cpu().numpy()  # Green channel
    im = ax.imshow(phase_img, cmap='twilight')
    ax.set_title('DOE Phase Mask (Green)')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Plot PSFs at different depths
    depth_indices = [0, 2, 4, 6]  # Near to far
    for i, d_idx in enumerate(depth_indices):
        ax = axes[0, i] if i == 0 else axes[0 if i < 3 else 1, i if i < 3 else i-3]
        if i > 0:
            ax = axes[0, i] if i <= 3 else axes[1, i-4]
        
    # Plot PSFs for green channel at different depths
    for i, d_idx in enumerate(depth_indices):
        row = 1 if i >= 0 else 0
        col = i
        ax = axes[row, col]
        psf_img = psf[1, d_idx].cpu().numpy()  # Green channel
        ax.imshow(psf_img, cmap='hot')
        ax.set_title(f'PSF at Depth {d_idx+1}/{config["n_depths"]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('doe_visualization.png', dpi=150)
    print("Saved visualization to 'doe_visualization.png'")
    plt.show()


if __name__ == '__main__':
    main()
