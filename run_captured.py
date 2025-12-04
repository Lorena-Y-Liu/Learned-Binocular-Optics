"""
Run inference on captured stereo images.

This script demonstrates how to run the Deep Stereo model on real captured images
from a stereo camera setup with DOE.

Usage:
    python run_captured.py --ckpt_path <checkpoint_path> --left_img <left_image> --right_img <right_image>
"""

import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch
import torch.nn.functional as F
import torchvision

from deepstereo import Stereo3D
from solvers.image_reconstruction import apply_tikhonov_inverse
from util.fft import crop_psf
from util.helper import crop_boundary, linear_to_srgb, srgb_to_linear
from util.warp import Warp
from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from debayer import Debayer2x2


def flip(x):
    """Horizontal flip for mirror augmentation."""
    flip_transform = torchvision.transforms.RandomHorizontalFlip(p=1)
    return flip_transform(x)


def to_uint8(x: torch.Tensor):
    """Convert tensor to uint8 image format."""
    if x.dim() == 3:
        return (255 * (x.clamp(0, 1))).permute(1, 2, 0).to(torch.uint8)
    if x.dim() == 4:
        return (255 * (x.squeeze(0).clamp(0, 1))).permute(1, 2, 0).to(torch.uint8)


def strech_img(x):
    """Normalize image to [0, 1] range."""
    return (x - x.min()) / (x.max() - x.min())


def find_minmax(img, saturation=0.1):
    """Find min/max values with saturation clipping."""
    min_val = np.percentile(img, saturation)
    max_val = np.percentile(img, 100 - saturation)
    return min_val, max_val


def rescale_image(x):
    """Rescale image using percentile-based normalization."""
    min_val, max_val = find_minmax(x.cpu().numpy())
    min_val = torch.from_numpy(np.array(min_val))
    max_val = torch.from_numpy(np.array(max_val))
    return ((x - min_val) / (max_val - min_val)).cuda()


@torch.no_grad()
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    hparams = ckpt['hyper_parameters']
    model = Stereo3D(hparams=hparams)

    # Load decoder weights
    decoder_dict = {key[8:]: value for key, value in ckpt['state_dict'].items() if 'decoder' in key}
    model.decoder.load_state_dict(decoder_dict)

    # Load stereo matching network
    matching = IGEVStereo(args)
    matching = torch.nn.DataParallel(matching, device_ids=[0])
    matching.load_state_dict(torch.load(args.matching_ckpt))
    matching = matching.module
    matching.to(device)
    matching.eval()
    model.eval()
    model.to(device)

    # Load images
    left_img_np = skimage.io.imread(args.left_img).astype(np.float32)
    right_img_np = skimage.io.imread(args.right_img).astype(np.float32)

    left_linear = torch.from_numpy(left_img_np).unsqueeze(0) / 65535
    right_linear = torch.from_numpy(right_img_np).unsqueeze(0) / 65535

    left_linear = left_linear.unsqueeze(0).to(device)
    right_linear = right_linear.unsqueeze(0).to(device)

    # Debayer
    debayer = Debayer2x2().to(device)
    left_linear = debayer(left_linear)
    right_linear = debayer(right_linear)

    # Resize if needed
    size = (int(1200), int(1920))
    left_linear = F.interpolate(left_linear, size, mode='bilinear', align_corners=False)
    right_linear = F.interpolate(right_linear, size, mode='bilinear', align_corners=False)

    image_sz = left_linear.shape[-2:]

    # Compute PSF-based pseudo-inverse
    psf_left = model.camera_left.normalize_psf(
        model.camera_left.psf_at_camera(size=image_sz, modulate_phase=False).unsqueeze(0)
    )
    pinv_volumes_left = apply_tikhonov_inverse(
        left_linear, psf_left, model.hparams.reg_tikhonov, apply_edgetaper=True
    )

    psf_right = model.camera_right.normalize_psf(
        model.camera_right.psf_at_camera(size=image_sz, modulate_phase=False).unsqueeze(0)
    )
    pinv_volumes_right = apply_tikhonov_inverse(
        right_linear, psf_right, model.hparams.reg_tikhonov, apply_edgetaper=True
    )

    # Mirror augmentation
    left_linear_m, right_linear_m = flip(right_linear), flip(left_linear)
    pinv_volumes_left_m, pinv_volumes_right_m = flip(pinv_volumes_right), flip(pinv_volumes_left)

    # Stereo matching
    padder = InputPadder(left_linear.shape, divis_by=32)
    left_mat, right_mat = padder.pad(left_linear, right_linear)
    left_mat_m, right_mat_m = padder.pad(left_linear_m, right_linear_m)

    est = matching(
        image1=linear_to_srgb(left_mat) * 255,
        image2=linear_to_srgb(right_mat) * 255,
        iters=32, test_mode=True
    )[-1].unsqueeze(0)
    est_m = matching(
        image1=linear_to_srgb(left_mat_m) * 255,
        image2=linear_to_srgb(right_mat_m) * 255,
        iters=32, test_mode=True
    )[-1].unsqueeze(0)

    est = padder.unpad(est)
    est_m = padder.unpad(est_m)

    # Normalize disparity
    rough = est.clamp(0, args.max_disp) / args.max_disp
    rough_m = est_m.clamp(0, args.max_disp) / args.max_disp

    # Warp right image using disparity
    warping = Warp()
    w_disparity = est
    w_disparity_2 = flip(est_m)
    warped_right, mask = warping.warp_disp(right_linear, w_disparity, w_disparity_2)
    warped_right += left_linear * (1 - mask)
    warped_right_m, mask_m = warping.warp_disp(right_linear_m, flip(w_disparity_2), flip(w_disparity))
    warped_right_m += left_linear_m * (1 - mask_m)

    # Run decoder for image recovery and depth refinement
    model_outputs = model.decoder(
        hparams=model.hparams,
        captimgs_left=left_linear.float(),
        captimgs_right=warped_right.float(),
        pinv_volumes_left=pinv_volumes_left.float(),
        rough_depth=rough.float()
    )
    model_outputs_m = model.decoder(
        hparams=model.hparams,
        captimgs_left=left_linear_m.float(),
        captimgs_right=warped_right_m.float(),
        pinv_volumes_left=pinv_volumes_left_m.float(),
        rough_depth=rough_m.float()
    )

    # Process outputs
    est_images_left = crop_boundary(model_outputs[0], model.crop_width)
    est_images_left_m = crop_boundary(model_outputs_m[0], model.crop_width)
    captimgs_left = crop_boundary(left_linear[[0]], model.crop_width)
    captimgs_right = crop_boundary(right_linear[[0]], model.crop_width)

    est_depthmaps = crop_boundary(model_outputs[-1], model.crop_width)
    est_depthmaps_m = crop_boundary(model_outputs_m[-1], model.crop_width)

    est_images_left = linear_to_srgb(est_images_left)
    est_images_left_m = linear_to_srgb(est_images_left_m)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    skimage.io.imsave(
        os.path.join(args.output_dir, 'left_captimg.png'),
        to_uint8(rescale_image(captimgs_left.cpu())).cpu().numpy()
    )
    skimage.io.imsave(
        os.path.join(args.output_dir, 'right_captimg.png'),
        to_uint8(rescale_image(captimgs_right.cpu())).cpu().numpy()
    )
    skimage.io.imsave(
        os.path.join(args.output_dir, 'est_img_left.png'),
        to_uint8(rescale_image(est_images_left.cpu())).cpu().numpy()
    )
    skimage.io.imsave(
        os.path.join(args.output_dir, 'est_img_right.png'),
        to_uint8(rescale_image(flip(est_images_left_m.cpu()))).cpu().numpy()
    )

    plt.imsave(
        os.path.join(args.output_dir, 'disparity_left.png'),
        (255 * (1 - rough.cpu()).squeeze().clamp(0, 1)).to(torch.uint8).numpy(),
        cmap='inferno'
    )
    plt.imsave(
        os.path.join(args.output_dir, 'depth_refined_left.png'),
        (255 * (1 - est_depthmaps.cpu()).squeeze().clamp(0, 1)).to(torch.uint8).numpy(),
        cmap='inferno'
    )

    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--left_img', type=str, required=True, help='Path to left image')
    parser.add_argument('--right_img', type=str, required=True, help='Path to right image')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--matching_ckpt', type=str, default='./sceneflow.pth', help='Path to stereo matching checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')

    parser = Stereo3D.add_model_specific_args(parser)
    args = parser.parse_known_args()[0]

    main(args)