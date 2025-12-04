"""
Deep Stereo Training Script.

This script handles the training pipeline for the Deep Stereo model,
including data loading, model initialization, and training loop management.

Usage:
    python deepstereo_trainer.py
    
Configuration is loaded from configs/config.yaml by default.
"""

import os
import warnings
from argparse import ArgumentParser

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Default grid_sample and affine_grid behavior has changed")
warnings.filterwarnings("ignore", message="Importing `spectral_angle_mapper` from `torchmetrics.functional` was deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchmetrics")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

import torch
import imageio
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader

from datasets.stereo_flyingthings import SceneFlow
from deepstereo import Stereo3D
from util.log_manager import LogManager

imageio.plugins.freeimage.download()

# Import config module for YAML configuration support
try:
    from config import load_config, config_to_namespace
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

seed_everything(123)

# GPU configuration
GPU_ID = '4'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID


def prepare_data(hparams):
    """
    Prepare training and validation dataloaders.
    
    Args:
        hparams: Hyperparameters namespace containing dataset configuration
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    image_sz = hparams.image_sz
    crop_width = hparams.crop_width
    augment = hparams.augment
    padding = 0

    # Create SceneFlow datasets
    sf_train_dataset = SceneFlow(
        'train',
        (image_sz[0] + 4 * crop_width, image_sz[1] + 4 * crop_width),
        is_training=True,
        augment=augment,
        padding=padding,
        singleplane=False
    )
    
    sf_val_dataset = SceneFlow(
        'val',
        (image_sz[0] + 4 * crop_width, image_sz[1] + 4 * crop_width),
        is_training=False,
        augment=augment,
        padding=padding,
        singleplane=False
    )
    
    if hparams.mix_instereo_dataset:
        # Mix with InStereo2k dataset
        train_datasets = []
        val_datasets = []
        sample_weights_list = []
        
        is_train_dataset = InStereo2k(
            'train',
            (image_sz[0] + 4 * crop_width, image_sz[1] + 4 * crop_width),
            is_training=True,
            augment=augment,
            padding=padding,
            singleplane=False
        )
        n_is = len(is_train_dataset)
        
        is_val_dataset = InStereo2k(
            'val',
            (image_sz[0] + 4 * crop_width, image_sz[1] + 4 * crop_width),
            is_training=False,
            augment=augment,
            padding=padding,
            singleplane=False,
            hparams=hparams
        )

        train_datasets.append(is_train_dataset)
        val_datasets.append(is_val_dataset)
        sample_weights_list.append(1.0 / n_is * torch.ones(n_is, dtype=torch.double))
        
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        sample_weights = torch.cat(sample_weights_list, dim=0)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=hparams.batch_sz,
            sampler=sampler,
            num_workers=hparams.num_workers,
            shuffle=False,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=hparams.batch_sz,
            num_workers=hparams.num_workers,
            shuffle=False,
            pin_memory=True
        )
    else:
        # Use SceneFlow only
        train_dataloader = DataLoader(
            sf_train_dataset,
            batch_size=hparams.batch_sz,
            num_workers=hparams.num_workers,
            shuffle=True,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            sf_val_dataset,
            batch_size=hparams.batch_sz,
            num_workers=hparams.num_workers,
            shuffle=False,
            pin_memory=True
        )

    return train_dataloader, val_dataloader


def main(args):
    """
    Main training function.
    
    Args:
        args: Parsed command-line arguments
    """
    # Load configuration from YAML file
    if CONFIG_AVAILABLE:
        config_path = 'configs/config.yaml'
        if os.path.exists(config_path):
            config = load_config(config_path)
            args = config_to_namespace(config, args)
            print(f"Loaded configuration from: {config_path}")
        else:
            print(f"Warning: Config file not found at {config_path}, using command-line args only.")
    
    # Setup logger and callbacks
    logger = TensorBoardLogger(args.default_root_dir, name=args.experiment_name)
    logmanager_callback = LogManager()
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_loss',
        filepath=os.path.join(logger.log_dir, 'checkpoints', '{epoch}-{val_loss:.4f}'),
        save_top_k=1,
        period=1,
        mode='min'
    )
    
    # Initialize model and data
    model = Stereo3D(hparams=args, log_dir=logger.log_dir)
    model.eval()
    train_dataloader, val_dataloader = prepare_data(hparams=args)

    # Create trainer and start training
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[logmanager_callback],
        checkpoint_callback=checkpoint_callback,
        sync_batchnorm=True,
        benchmark=True,
        val_check_interval=0.5
    )
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    # Basic arguments
    parser.add_argument('--experiment_name', type=str, default='log')
    parser.add_argument('--mix_instereo_dataset', dest='mix_instereo_dataset', action='store_true')
    parser.set_defaults(mix_instereo_dataset=False)
    parser.add_argument("--local_rank", type=int)
    
    # Add trainer and model arguments
    parser = Trainer.add_argparse_args(parser)
    parser = Stereo3D.add_model_specific_args(parser)

    # Set defaults
    parser.set_defaults(
        gpus=1,
        default_root_dir='sampledata/logs',
        max_epochs=200
    )

    args = parser.parse_known_args()[0]
    main(args)
