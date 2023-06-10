import os
import sys

sys.path.insert(1, os.path.abspath(os.path.curdir))

import json

import numpy as np
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, SubsetRandomSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.make_dataset import DatasetVesuvius
from src.data.make_dataset_compressed import DatasetVesuviusCompressed
from src.models.lightning import LightningVesuvius
from src.utils import get_device

import src.constant as cst

import wandb


def main():
    wandb.init(
        project='vesuvius-challenge-ink-detection',
        entity='rosia-lab',
        group='efficientnet-b5'
    )
    
    splits = KFold(n_splits=wandb.config.n_splits, shuffle=True, random_state=42)
    dataset_vesuvius = get_dataset()
    
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset_vesuvius)))):
        wandb.config.num_split = fold + 1
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_dataloader = DataLoader(dataset_vesuvius, batch_size=wandb.config.batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(dataset_vesuvius, batch_size=wandb.config.batch_size, sampler=val_sampler)
    
        torch.cuda.empty_cache()
        model = get_model()
        trainer = get_trainer()
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


def main_manual():
    if sys.argv[2] == 'EfficientUNetV2':
        wandb_parameters_path = os.path.join('src', 'models', 'wandb_efficientunetv2.json')
    elif sys.argv[2] == 'efficientnet-b5':
        wandb_parameters_path = os.path.join('src', 'models', 'wandb_efficientnet-b5.json')
    
    with open(wandb_parameters_path, mode='r') as f:
        wandb_parameters = json.load(f)
        
    splits = KFold(n_splits=wandb_parameters['config']['n_splits'], shuffle=True, random_state=42)
    dataset_vesuvius = get_dataset_manual(wandb_parameters)
    
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset_vesuvius)))):
        wandb_parameters['config']['num_split'] = fold + 1
        
        wandb.init(**wandb_parameters)
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_dataloader = DataLoader(dataset_vesuvius, batch_size=wandb.config.batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(dataset_vesuvius, batch_size=wandb.config.batch_size, sampler=val_sampler)
    
        torch.cuda.empty_cache()
        model = get_model()
        trainer = get_trainer()
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        wandb.finish()


def get_model():
    model_params = {}

    if wandb.config.model_name == 'UNet3D':
        model_params = {
            'num_blocks': wandb.config.num_blocks,
            'inputs_size': wandb.config.tile_size,
        }
    elif 'EfficientUNetV2' in wandb.config.model_name:
        model_params = {
            'in_channels': wandb.config.num_slices,
        }
    elif 'efficientnet' in wandb.config.model_name:
        model_params = {
            "in_channels": wandb.config.num_slices,
            "encoder_weights": wandb.config.encoder_weights,
            "classes": 1
        }

    lightning_model = LightningVesuvius(
        model_name=wandb.config.model_name,
        model_params=model_params,
        learning_rate=wandb.config.learning_rate,
        bce_weight=wandb.config.bce_weight,
        dice_threshold=wandb.config.dice_threshold
    )

    return lightning_model


def get_dataloaders():
    device = get_device()

    train_dataset = DatasetVesuviusCompressed(
        fragments=wandb.config.train_fragments,
        tile_size=wandb.config.tile_size,
        num_slices=wandb.config.num_slices,
        slices_list=None,
        start_slice=min(wandb.config.start_slice, 64 - wandb.config.num_slices),
        reverse_slices=wandb.config.reverse_slices,
        selection_thr=wandb.config.selection_thr,
        augmentation=wandb.config.augmentation,
        device=device
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=True,
        drop_last=True
    )

    wandb.config['slices_list'] = train_dataset.slices

    val_dataset = DatasetVesuviusCompressed(
        fragments=[wandb.config.val_fragment],
        tile_size=wandb.config.tile_size,
        num_slices=wandb.config.num_slices,
        slices_list=wandb.config.slices_list,
        start_slice=None,
        reverse_slices=wandb.config.reverse_slices,
        selection_thr=wandb.config.selection_thr,
        augmentation=wandb.config.augmentation,
        device=device
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=wandb.config.batch_size,
        drop_last=True
    )
    print('\n')

    return train_dataloader, val_dataloader


def get_dataset():
    device = get_device()
    
    dataset_vesuvius = DatasetVesuvius(
        fragments=['1', '2', '3'],
        tile_size=wandb.config.tile_size,
        num_slices=wandb.config.num_slices,
        slices_list=None,
        start_slice=min(wandb.config.start_slice, 64 - wandb.config.num_slices),
        reverse_slices=wandb.config.reverse_slices,
        selection_thr=wandb.config.selection_thr,
        augmentation=wandb.config.augmentation,
        device=device,
        overlap=wandb.config.overlap,
    )
    
    return dataset_vesuvius


def get_dataset_manual(wandb_parameters):
    device = get_device()
    
    dataset_vesuvius = DatasetVesuvius(
        fragments=['1', '2', '3'],
        tile_size=wandb_parameters['config']['tile_size'],
        num_slices=wandb_parameters['config']['num_slices'],
        slices_list=None,
        start_slice=min(wandb_parameters['config']['start_slice'], 64 - wandb_parameters['config']['num_slices']),
        reverse_slices=wandb_parameters['config']['reverse_slices'],
        selection_thr=wandb_parameters['config']['selection_thr'],
        augmentation=wandb_parameters['config']['augmentation'],
        device=device,
        overlap=wandb_parameters['config']['overlap'],
    )
    
    return dataset_vesuvius


def get_trainer():
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val/sub_f05_score',
        mode='max',
        dirpath=cst.MODELS_DIR,
        filename=f'{wandb.run.name}-{wandb.run.id}-{wandb.config.start_slice}-{wandb.config.num_slices}-{wandb.config.num_split}',
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=wandb.config.epochs,
        callbacks=[checkpoint_callback],
        logger=WandbLogger(),
        log_every_n_steps=1,
    )

    return trainer


if __name__ == '__main__':

    if sys.argv[1] == '--manual' or sys.argv[1] == '-m':
        main_manual()
    else:
        main()
