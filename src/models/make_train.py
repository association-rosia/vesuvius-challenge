import os
import sys
sys.path.insert(1, os.path.abspath(os.path.curdir))

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data.make_dataset import DatasetVesuvius
from src.models.lightning import LightningVesuvius
from src.utils import get_fragments_shape, get_device

from constant import TRAIN_FRAGMENTS, VAL_FRAGMENTS, MODELS_DIR, TILE_SIZE, Z_DIM

import wandb


def main():
    torch.cuda.empty_cache()
    model = get_model()
    train_dataloader, val_dataloader = get_dataloaders(wandb.config.num_slices)
    trainer = get_trainer()
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


def get_model():
    model_params = {}

    if wandb.config.model_name == 'UNet3D':
        model_params = {
            'nb_blocks': wandb.config.nb_blocks,
            'inputs_size': wandb.config.tile_size,
        }

    lightning_model = LightningVesuvius(
        model_name=wandb.config.model_name,
        model_params=model_params,
        learning_rate=wandb.config.learning_rate,
        scheduler_patience=wandb.config.scheduler_patience,
        bce_weight=wandb.config.bce_weight,
        dice_threshold=wandb.config.dice_threshold,
        val_fragments_shape=get_fragments_shape(
            wandb.config.val_fragments, 
            wandb.config.tile_size),
    )

    return lightning_model


def get_dataloaders(num_slices):
    device = get_device()

    train_dataset = DatasetVesuvius(
        fragments=wandb.config.train_fragments,
        tile_size=wandb.config.tile_size,
        num_slices=num_slices,
        random_slices=False,
        selection_thr=0.01,
        augmentation=True,
        device=device
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=True,
        drop_last=True
    )

    val_dataset = DatasetVesuvius(
        fragments=wandb.config.val_fragments,
        tile_size=wandb.config.tile_size,
        num_slices=num_slices,
        random_slices=False,
        selection_thr=0.01,
        augmentation=True,
        device=device
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=wandb.config.batch_size,
        drop_last=True
    )
    print('\n')

    return train_dataloader, val_dataloader


def get_trainer():
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        dirpath=MODELS_DIR,
        filename=f'{wandb.run.name}-{wandb.run.id}',
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator='gpu',
        val_check_interval=0.25,
        max_epochs=wandb.config.epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        logger=WandbLogger(),
    )

    return trainer


if __name__ == '__main__':

    if sys.argv[1] == '--manual' or sys.argv[1] == '-m':
        wandb.init(
            project='vesuvius-challenge-ink-detection',
            entity='rosia-lab',
            group='test',
            config={
                'batch_size': 6,
                'model_name': 'UNet3D',
                'nb_blocks': 3,
                'bce_weight': 0.5,
                'dice_threshold': 0.5,
                'scheduler_patience': 5,
                'learning_rate': 0.00001,
                'epochs': 20,
                'tile_size': TILE_SIZE,
                'num_slices': Z_DIM,
                'train_fragments': TRAIN_FRAGMENTS,
                'val_fragments': VAL_FRAGMENTS,
            },
        )
    else:
        wandb.init(project='vesuvius-challenge-ink-detection', entity='rosia-lab')

    main()
