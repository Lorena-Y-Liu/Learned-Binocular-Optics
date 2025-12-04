"""
Example: Run Inference on a Stereo Image Pair

This script demonstrates how to:
1. Load a pretrained Deep Stereo model
2. Run inference on stereo images
3. Visualize depth estimation results
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append('..')

from deepstereo import Stereo3D, load_hparams_from_config
from util.helper import srgb_to_linear, linear_to_srgb


def load_image(path: str) -> torch.Tensor:
    """Load image and convert to tensor."""
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
    return img.unsqueeze(0)  # Add batch dimension


def main():
    # Configuration
    config_path = '../configs/config.yaml'
    checkpoint_path = None  # Set to your checkpoint path
    left_image_path = 'path/to/left.png'
    right_image_path = 'path/to/right.png'
    
    # Check if example images exist
    if not os.path.exists(left_image_path):
        print("Please provide stereo image paths.")
        print("Usage: Edit left_image_path and right_image_path in this script.")
        return
    
    # Load configuration
    print("Loading configuration...")
    hparams = load_hparams_from_config(config_path)
    
    # Create model
    print("Creating model...")
    model = Stereo3D(hparams=hparams)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Load images
    print("Loading images...")
    left_img = load_image(left_image_path).to(device)
    right_img = load_image(right_image_path).to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # Convert to linear space for optical simulation
        left_linear = srgb_to_linear(left_img)
        right_linear = srgb_to_linear(right_img)
        
        # Get stereo matching result
        disparity, _ = model.matching(
            left_img * 255, 
            right_img * 255, 
            iters=hparams.valid_iters
        )
    
    # Convert results to numpy
    disparity_np = disparity[0, 0].cpu().numpy()
    left_np = left_img[0].permute(1, 2, 0).cpu().numpy()
    right_np = right_img[0].permute(1, 2, 0).cpu().numpy()
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(left_np)
    axes[0].set_title('Left Image')
    axes[0].axis('off')
    
    axes[1].imshow(right_np)
    axes[1].set_title('Right Image')
    axes[1].axis('off')
    
    im = axes[2].imshow(disparity_np, cmap='magma')
    axes[2].set_title('Estimated Disparity')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('inference_result.png', dpi=150)
    print("Saved result to 'inference_result.png'")
    plt.show()


if __name__ == '__main__':
    main()
