import pytorch_lightning
import torchmetrics
import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader
from datasets.stereo_flyingthings import SceneFlow
from datasets.stereo_driving import Driving
from datasets.stereo_monkaa import Monkaa
from datasets.stereo_instereo2k import InStereo2k
#from datasets.instereo2k2 import InStereo2k
from deepstereo import Stereo3D
from util.log_manager import LogManager
import imageio
imageio.plugins.freeimage.download()

seed_everything(123)


def prepare_data(hparams):
    image_sz = hparams.image_sz
    crop_width = hparams.crop_width
    augment = hparams.augment
    padding = 0
    #val_idx = 50
    #val_idx = 3994

    val_idx = 3994
    sf_train_dataset = SceneFlow('train',
                                 (image_sz[0] + 4 * crop_width,
                                  image_sz[1] + 4 * crop_width),
                                 is_training=True,
                                 augment=augment, padding=padding,
                                 singleplane=False)
    
    
    sf_val_dataset = SceneFlow('val',
                               (image_sz[0] + 4 * crop_width,
                                image_sz[1] + 4 * crop_width),
                               is_training=False,
                               augment=augment, padding=padding,
                               singleplane=False)
    
    if hparams.mix_instereo_dataset:

        train_datasets = []
        val_datasets = []
        sample_weights_list = []

        '''sf_train_dataset = SceneFlow('train',
                                    (image_sz[0] + 4 * crop_width,
                                    image_sz[1] + 4 * crop_width),
                                    is_training=True,
                                    augment=augment, padding=padding,
                                    singleplane=False)
        # sf_train_dataset = torch.utils.data.Subset(sf_train_dataset,
        #                                            range(val_idx, len(sf_train_dataset)))
        n_sf =len(ft_train_dataset)
        sf_val_dataset = SceneFlow('val',
                                (image_sz[0] + 4 * crop_width,
                                    image_sz[1] + 4 * crop_width),
                                is_training=False,
                                augment=augment, padding=padding,
                                singleplane=False)
        train_datasets.append(sf_train_dataset)

        val_datasets.append(sf_val_dataset)
        sample_weights_list.append(1. / n_sf * torch.ones(n_sf, dtype=torch.double))'''

        '''mk_train_dataset = Monkaa('train',
                                    (image_sz[0] + 4 * crop_width,
                                    image_sz[1] + 4 * crop_width),
                                    is_training=True,
                                    augment=augment, padding=padding,
                                    singleplane=False)
        n_mk = len(mk_train_dataset)

        mk_val_dataset = Monkaa('val',
                                (image_sz[0] + 4 * crop_width,
                                    image_sz[1] + 4 * crop_width),
                                is_training=False,
                                augment=augment, padding=padding,
                                singleplane=False)

        val_datasets.append(mk_val_dataset)
        train_datasets.append(mk_train_dataset)
        sample_weights_list.append(1. / n_mk * torch.ones(n_mk, dtype=torch.double))'''

        '''dr_train_dataset = Driving('train',
                                    (image_sz[0] + 4 * crop_width,
                                    image_sz[1] + 4 * crop_width),
                                    is_training=True,
                                    augment=augment, padding=padding,
                                    singleplane=False)
        n_dr = len(dr_train_dataset)

        train_datasets.append(dr_train_dataset)
        sample_weights_list.append(1. / n_dr * torch.ones(n_dr, dtype=torch.double))'''
        
        Is_train_dataset = InStereo2k('train',
                                    (image_sz[0] + 4 * crop_width,
                                    image_sz[1] + 4 * crop_width),
                                    is_training=True,
                                    augment=augment, padding=padding,
                                    singleplane=False)
        n_is = len(Is_train_dataset)
        Is_val_dataset = InStereo2k('val',
                                (image_sz[0] + 4 * crop_width,
                                    image_sz[1] + 4 * crop_width),
                                is_training=False,
                                augment=augment, padding=padding,
                                singleplane=False,
                                hparams=hparams,)

        train_datasets.append(Is_train_dataset)
        val_datasets.append(Is_val_dataset)
        sample_weights_list.append(1. / n_is * torch.ones(n_is, dtype=torch.double))
        
    
        
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        
        sample_weights = torch.cat(sample_weights_list, dim=0)

        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

        train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_sz, sampler=sampler, #hparams.batch_sz
                                      num_workers=hparams.num_workers, shuffle=False, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_sz,#hparams.batch_sz,
                                    num_workers=hparams.num_workers, shuffle=False, pin_memory=True)
    
       
        
    else:
        train_dataset = sf_train_dataset
        val_dataset = sf_val_dataset
        train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_sz,
                                      num_workers=hparams.num_workers, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_sz,
                                    num_workers=hparams.num_workers, shuffle=False, pin_memory=True)

   
   
    return train_dataloader, val_dataloader



def main(args):

    
    logger = TensorBoardLogger(args.default_root_dir,
                               name=args.experiment_name)
    logmanager_callback = LogManager()
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_loss',
        filepath=os.path.join(logger.log_dir, 'checkpoints', '{epoch}-{val_loss:.4f}'),
        save_top_k=1,
        period=1,
        mode='min')
    
    model = Stereo3D(hparams=args, log_dir=logger.log_dir)
    
    model.eval()
    
    train_dataloader, val_dataloader = prepare_data(hparams=args)

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[logmanager_callback],
        checkpoint_callback=checkpoint_callback,
        sync_batchnorm=True,
        benchmark=True,
        val_check_interval= 0.5,
        #gpus=[0,1,2]
    )
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser.add_argument('--experiment_name', type=str, default='log')
    parser.add_argument('--mix_instereo_dataset', dest='mix_instereo_dataset', action='store_true')
    parser.set_defaults(mix_instereo_dataset=False)
    parser.add_argument("--local_rank", type=int)
    parser = Trainer.add_argparse_args(parser)
    parser = Stereo3D.add_model_specific_args(parser)
    #parser.add_argument('--local_rank', default=-1, type=int)

    parser.set_defaults(
        gpus=1,
        default_root_dir='sampledata/logs',
        max_epochs=200,
    )

    args = parser.parse_known_args()[0]
    

    main(args)
