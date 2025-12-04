"""
Configuration file for Deep Stereo training and inference.

This module provides a centralized configuration system using YAML files
and dataclasses for type-safe parameter management.

Usage:
    from config import Config, load_config
    
    # Load from YAML file
    config = load_config('configs/default.yaml')
    
    # Or create with defaults
    config = Config()
    
    # Access parameters
    print(config.training.learning_rate)
    print(config.optics.focal_depth)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union
import yaml


@dataclass
class LoggerConfig:
    """Logging and visualization parameters."""
    summary_max_images: int = 8
    summary_image_sz: int = 200
    summary_mask_sz: int = 1260
    summary_depth_every: int = 2000
    summary_track_train_every: int = 500


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Learning rates
    cnn_lr: float = 1e-3
    depth_lr: float = 1e-5
    optics_lr: float = 0.0
    
    # Batch and workers
    batch_sz: int = 1
    num_workers: int = 8
    
    # Training settings
    max_epochs: int = 100
    augment: bool = True
    mixed_precision: bool = True
    num_steps: int = 200000
    wdecay: float = 0.00001
    
    # Random seed
    seed: int = 666


@dataclass
class LossConfig:
    """Loss function weights."""
    depth_loss_weight: float = 1.0
    depth_1_loss_weight: float = 0.0
    image_loss_weight: float = 1.0
    psf_loss_weight: float = 0.0
    psf_size: int = 160


@dataclass
class DatasetConfig:
    """Dataset and image parameters."""
    image_sz: List[int] = field(default_factory=lambda: [320, 736])
    n_depths: int = 7
    min_depth: float = 0.67
    max_depth: float = 8.0
    crop_width: int = 0
    
    # Data augmentation
    img_gamma: Optional[List[float]] = None
    saturation_range: List[float] = field(default_factory=lambda: [0.0, 1.4])
    do_flip: Union[bool, str] = False
    spatial_scale: List[float] = field(default_factory=lambda: [-0.2, 0.4])
    noyjitter: bool = False


@dataclass 
class SolverConfig:
    """Solver and reconstruction parameters."""
    reg_tikhonov: float = 0.1
    model_base_ch: int = 32
    preinverse: bool = True
    warp_img: bool = True


@dataclass
class OpticsConfig:
    """Optical system parameters."""
    # Camera type
    camera_type: str = 'mixed'
    doe_type: str = 'rank2'  # Options: rank1, rank2, ring
    
    # DOE initialization
    use_pretrained_doe: bool = False  # If True, load from pretrained vectors; if False, use analytical initialization
    
    # DOE mask parameters
    mask_sz: int = 1260
    mask_pitch: float = 3.45e-6  # meters
    mask_diameter: float = 4.347e-3  # meters
    mask_upsample_factor: int = 2
    
    # Camera parameters
    focal_length: float = 35e-3  # meters
    focal_depth: float = 1.23  # meters (left camera)
    focal_depth_right: float = 1.23  # meters (right camera)
    camera_pixel_pitch: float = 5.86e-6  # meters
    
    # Simulation parameters
    diffraction_efficiency: float = 0.7
    full_size: int = 1200
    
    # Noise parameters
    noise_sigma_min: float = 0.001
    noise_sigma_max: float = 0.005
    
    # Simulation options
    bayer: bool = True
    occlusion: bool = True
    optimize_optics: bool = False
    psf_jitter: bool = False
    scale: float = 1.0


@dataclass
class IGEVConfig:
    """IGEV stereo matching network parameters."""
    train_iters: int = 6
    valid_iters: int = 12
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 128])
    corr_implementation: str = 'reg'  # Options: reg, alt, reg_cuda, alt_cuda
    shared_backbone: bool = False
    corr_levels: int = 2
    corr_radius: int = 4
    n_downsample: int = 2
    slow_fast_gru: bool = False
    n_gru_layers: int = 3
    max_disp: int = 192


@dataclass
class Config:
    """
    Main configuration class containing all sub-configurations.
    
    This class aggregates all configuration categories and provides
    methods for loading from and saving to YAML files.
    """
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    igev: IGEVConfig = field(default_factory=IGEVConfig)
    
    # Experiment settings
    experiment_name: str = 'default_experiment'
    checkpoint_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert config to nested dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'checkpoint_path': self.checkpoint_path,
            'logger': self.logger.__dict__,
            'training': self.training.__dict__,
            'loss': self.loss.__dict__,
            'dataset': {
                **self.dataset.__dict__,
                'image_sz': list(self.dataset.image_sz),
                'saturation_range': list(self.dataset.saturation_range),
                'spatial_scale': list(self.dataset.spatial_scale),
            },
            'solver': self.solver.__dict__,
            'optics': self.optics.__dict__,
            'igev': {
                **self.igev.__dict__,
                'hidden_dims': list(self.igev.hidden_dims),
            },
        }
    
    def to_hparams(self) -> dict:
        """
        Convert config to flat dictionary for compatibility with existing code.
        This maintains backward compatibility with the argparse-based system.
        """
        hparams = {}
        
        # Logger params
        hparams.update({
            'summary_max_images': self.logger.summary_max_images,
            'summary_image_sz': self.logger.summary_image_sz,
            'summary_mask_sz': self.logger.summary_mask_sz,
            'summary_depth_every': self.logger.summary_depth_every,
            'summary_track_train_every': self.logger.summary_track_train_every,
        })
        
        # Training params
        hparams.update({
            'cnn_lr': self.training.cnn_lr,
            'depth_lr': self.training.depth_lr,
            'optics_lr': self.training.optics_lr,
            'batch_sz': self.training.batch_sz,
            'num_workers': self.training.num_workers,
            'augment': self.training.augment,
            'mixed_precision': self.training.mixed_precision,
            'num_steps': self.training.num_steps,
            'wdecay': self.training.wdecay,
        })
        
        # Loss params
        hparams.update({
            'depth_loss_weight': self.loss.depth_loss_weight,
            'depth_1_loss_weight': self.loss.depth_1_loss_weight,
            'image_loss_weight': self.loss.image_loss_weight,
            'psf_loss_weight': self.loss.psf_loss_weight,
            'psf_size': self.loss.psf_size,
        })
        
        # Dataset params
        hparams.update({
            'image_sz': self.dataset.image_sz,
            'n_depths': self.dataset.n_depths,
            'min_depth': self.dataset.min_depth,
            'max_depth': self.dataset.max_depth,
            'crop_width': self.dataset.crop_width,
            'img_gamma': self.dataset.img_gamma,
            'saturation_range': self.dataset.saturation_range,
            'do_flip': self.dataset.do_flip,
            'spatial_scale': self.dataset.spatial_scale,
            'noyjitter': self.dataset.noyjitter,
        })
        
        # Solver params
        hparams.update({
            'reg_tikhonov': self.solver.reg_tikhonov,
            'model_base_ch': self.solver.model_base_ch,
            'preinverse': self.solver.preinverse,
            'warp_img': self.solver.warp_img,
        })
        
        # Optics params
        hparams.update({
            'camera_type': self.optics.camera_type,
            'doe_type': self.optics.doe_type,
            'mask_sz': self.optics.mask_sz,
            'mask_pitch': self.optics.mask_pitch,
            'mask_diameter': self.optics.mask_diameter,
            'mask_upsample_factor': self.optics.mask_upsample_factor,
            'focal_length': self.optics.focal_length,
            'focal_depth': self.optics.focal_depth,
            'focal_depth_right': self.optics.focal_depth_right,
            'camera_pixel_pitch': self.optics.camera_pixel_pitch,
            'diffraction_efficiency': self.optics.diffraction_efficiency,
            'full_size': self.optics.full_size,
            'noise_sigma_min': self.optics.noise_sigma_min,
            'noise_sigma_max': self.optics.noise_sigma_max,
            'bayer': self.optics.bayer,
            'occlusion': self.optics.occlusion,
            'optimize_optics': self.optics.optimize_optics,
            'psf_jitter': self.optics.psf_jitter,
            'scale': self.optics.scale,
            'use_pretrained_doe': self.optics.use_pretrained_doe,
        })
        
        # IGEV params
        hparams.update({
            'train_iters': self.igev.train_iters,
            'valid_iters': self.igev.valid_iters,
            'hidden_dims': self.igev.hidden_dims,
            'corr_implementation': self.igev.corr_implementation,
            'shared_backbone': self.igev.shared_backbone,
            'corr_levels': self.igev.corr_levels,
            'corr_radius': self.igev.corr_radius,
            'n_downsample': self.igev.n_downsample,
            'slow_fast_gru': self.igev.slow_fast_gru,
            'n_gru_layers': self.igev.n_gru_layers,
            'max_disp': self.igev.max_disp,
        })
        
        return hparams
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to {path}")
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create Config from dictionary."""
        config = cls()
        
        if 'experiment_name' in config_dict:
            config.experiment_name = config_dict['experiment_name']
        if 'checkpoint_path' in config_dict:
            config.checkpoint_path = config_dict['checkpoint_path']
        
        if 'logger' in config_dict:
            config.logger = LoggerConfig(**config_dict['logger'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'loss' in config_dict:
            config.loss = LossConfig(**config_dict['loss'])
        if 'dataset' in config_dict:
            config.dataset = DatasetConfig(**config_dict['dataset'])
        if 'solver' in config_dict:
            config.solver = SolverConfig(**config_dict['solver'])
        if 'optics' in config_dict:
            config.optics = OpticsConfig(**config_dict['optics'])
        if 'igev' in config_dict:
            config.igev = IGEVConfig(**config_dict['igev'])
            
        return config


def load_config(path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        Config object with loaded parameters
    """
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config.from_dict(config_dict)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


# Convenience function to create hparams namespace from config
def config_to_namespace(config: Config, args=None):
    """
    Convert Config to argparse.Namespace for backward compatibility.
    Merge with existing args if provided.
    
    Args:
        config: Config object
        args: Optional existing argparse.Namespace to merge with
        
    Returns:
        argparse.Namespace with all parameters
    """
    from argparse import Namespace
    
    config_hparams = config.to_hparams()
    
    if args is not None:
        # Merge config values into existing args
        # Config file values will override argparse defaults
        for key, value in config_hparams.items():
            setattr(args, key, value)
        # Also set experiment_name from config
        if config.experiment_name:
            args.experiment_name = config.experiment_name
        return args
    else:
        return Namespace(**config_hparams)


if __name__ == '__main__':
    # Generate default configuration file
    config = Config()
    config.save('configs/config.yaml')
    print("Configuration generated at configs/config.yaml")
