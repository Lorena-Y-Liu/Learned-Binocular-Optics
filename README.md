# Learned Binocular-Encoding Optics for RGBD Imaging

Official PyTorch implementation of **IEEE CVPR 2025, Oral Paper:** 

**"Learned binocular-encoding optics for RGBD imaging using joint stereo and focus cues"**.

🌐 **[Project Page](https://liangxunou.github.io/25liulearned/)** | 📄 **[Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Learned_Binocular-Encoding_Optics_for_RGBD_Imaging_Using_Joint_Stereo_and_CVPR_2025_paper.pdf)** | 💻 **[Code](https://github.com/Lorena-Y-Liu/Learned-Binocular-Optics)** | 📒 **[Supplementary](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Liu_Learned_Binocular-Encoding_Optics_CVPR_2025_supplemental.pdf)**

## Overview

This project implements a novel RGBD imaging system that combines:
- **Diffractive Optical Elements (DOE)**: Learnable phase masks for depth-dependent Point Spread Functions (PSF)
- **Stereo Vision**: Learned binocular camera setup for RGB imaging and disparity-based depth estimation
- **Joint Optimization**: End-to-end training of both optical elements and neural network components

![Teaser](docs/teaser.png)

## News

- **[2025.06]** Paper accepted to CVPR 2025 as **Oral Paper**
- **[2025.11]** Code released

## Project Structure

```
deep_stereo/
├── config.py               # Configuration system with dataclasses
├── deepstereo.py           # Main PyTorch Lightning training module
├── deepstereo_trainer.py   # Training script with data loading
├── configs/                # YAML configuration files
│   └── config.yaml         # Default configuration
├── optics/                 # DOE camera simulation modules
│   ├── base_camera.py      # Base camera class with shared functionality
│   ├── camera_left_rank1.py   # Left camera with Rank-1 DOE
│   ├── camera_right_rank1.py  # Right camera with Rank-1 DOE
│   ├── camera_left_rank2.py   # Left camera with Rank-2 DOE
│   ├── camera_right_rank2.py  # Right camera with Rank-2 DOE
│   ├── camera_left_ring.py    # Left camera with Ring DOE
│   └── camera_right_ring.py   # Right camera with Ring DOE
├── core/                   # Core stereo matching components (IGEV-based)
│   ├── igev_stereo.py      # IGEV stereo matching network
│   ├── extractor.py        # Feature extraction
│   ├── geometry.py         # Geometric operations
│   ├── submodule.py        # Common submodules
│   └── update.py           # Iterative update modules
├── models/                 # Neural network models
│   ├── fusion.py           # Image recovery network
│   ├── feature_fusion.py   # Feature fusion modules
│   ├── outputs_container.py # Output data container
│   ├── submodule.py        # Common submodules
│   └── unet.py             # U-Net architecture
├── ls_asm/                 # Least-Sampling Angular Spectrum Method
│   ├── LSASM.py            # Light propagation simulation
│   ├── input_field.py      # Input field generation
│   ├── phase_plates.py     # Phase plate definitions
│   └── utils.py            # LS-ASM utilities
├── util/                   # Utility functions
├── datasets/               # Dataset loaders
├── solvers/                # Image reconstruction solvers
├── psf/                    # PSF data and utilities
├── doe_vector/             # Pre-trained DOE parameters
└── examples/               # Usage examples
```

## DOE Types

The project supports three types of Diffractive Optical Elements:

| Type | Parameterization | Description |
|------|------------------|-------------|
| **Rank-1** | `h(x,y) = y * x^T` | Minimal parameters, simple depth-varying PSFs |
| **Rank-2** | `h(x,y) = y₀ * x₀^T + y₁ * x₁^T` | More expressive, good balance |
| **Ring** | Radially symmetric | Rotationally invariant PSFs |

## Installation

```bash
git clone https://github.com/Lorena-Y-Liu/Learned-Binocular-Optics.git
cd Learned-Binocular-Optics
```

# Create conda environment
conda create -n deep_stereo python=3.8
conda activate deep_stereo

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/cheind/pytorch-debayer@v1.0
```


### Dependencies
- PyTorch >= 1.9
- PyTorch Lightning
- torchvision
- OpenCV
- numpy
- scipy
- PyYAML
- kornia

## Quick Start

### Training

```bash
# Run training with default config
python deepstereo_trainer.py

# Or specify a config file
python deepstereo_trainer.py --config configs/config.yaml
```

## Pretrained Checkpoint

The pretrained SceneFlow checkpoint is hosted on Hugging Face:

- Hugging Face: [https://huggingface.co/LorenaLiu6/deepstereo-checkpoint-sceneflow](https://huggingface.co/LorenaLiu6/deepstereo-checkpoint-sceneflow)

Checkpoint file:

- `ds-model.ckpt`

After downloading, place the checkpoint at:

```bash
checkpoints/ds-model.ckpt
```

This repository contains the official source code, while the model weights are distributed separately through Hugging Face.

### Inference

```bash
# Run inference on captured images
python run_captured.py --checkpoint path/to/checkpoint.ckpt
```

## Configuration

Edit `configs/config.yaml` to modify parameters:

```yaml
# Experiment settings
experiment_name: deep_stereo_training

# Optical system parameters
optics:
  doe_type: rank2           # Options: rank1, rank2, ring
  use_pretrained_doe: false
  focal_depth: 1.23         # meters

# Training parameters
training:
  batch_sz: 2
  cnn_lr: 0.001
  max_epochs: 100
```

## Camera Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Focal length | 35mm | Camera focal length |
| Sensor pitch | 5.86μm | Camera sensor pixel size |
| Mask size | 1260 | DOE mask resolution |
| Wavelengths | 632nm, 550nm, 450nm | RGB wavelengths |

## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{liu2025learned,
  title={Learned Binocular-Encoding Optics for RGBD Imaging Using Joint Stereo and Focus Cues},
  author={Liu, Yuhui and Ou, Liangxun and Fu, Qiang and Amata, Hadi and Heidrich, Wolfgang and Peng, Yifan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={15833--15842},
  year={2025}
}
```

## Acknowledgments

This project builds upon several excellent open-source works:

- **[Depth from Defocus with Learned Optics](https://github.com/computational-imaging/DepthFromDefocusWithLearnedOptics)** - DOE optimization and depth-from-defocus framework
- **[ASASM (Adaptive Sampling Angular Spectrum Method)](https://github.com/whywww/ASASM)** - Wave propagation simulation
- **[IGEV-Stereo](https://github.com/gangweix/IGEV)** - Iterative geometry encoding stereo matching network

We thank the authors for releasing their code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact us via the [project page](https://liangxunou.github.io/25liulearned/).
