import os, sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

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
    model = get_model()
    train_dataloader, val_dataloader = get_dataloaders()
    print('\n')
    trainer = get_trainer()
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def get_model():
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
        val_fragments_shape=get_fragments_shape(VAL_FRAGMENTS, wandb.config.tile_size),
    )

    return lightning_model


def get_dataloaders():
    device = get_device()
    train_dataloader = DataLoader(
        dataset=DatasetVesuvius(fragments=TRAIN_FRAGMENTS,
                                tile_size=wandb.config.tile_size,
                                num_slices=Z_DIM,
                                random_slices=False,
                                selection_thr=0.01,
                                augmentation=True,
                                test=False,
                                device=device),
        batch_size=wandb.config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        dataset=DatasetVesuvius(fragments=VAL_FRAGMENTS,
                                tile_size=wandb.config.tile_size,
                                num_slices=Z_DIM,
                                random_slices=False,
                                selection_thr=0.01,
                                augmentation=True,
                                test=False,
                                device=device),
        batch_size=wandb.config.batch_size,
        shuffle=False,
        drop_last=True,
    )
    
    return train_dataloader, val_dataloader


def get_trainer():
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        dirpath=MODELS_DIR,
        filename='{val/loss:.5f}-' + f'{wandb.run.name}-{wandb.run.id}',
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # init the trainer
    trainer = pl.Trainer(
        accelerator='cpu',
        num_sanity_val_steps=0,
        # devices=1,
        max_epochs=wandb.config.epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        logger=WandbLogger(),
        precision=16,
    )

    return trainer


if __name__ == '__main__':

    if sys.argv[1] == '--manual' or sys.argv[1] == '-m':
        wandb.init(
            project='vesuvius-challenge-ink-detection',
            entity='rosia-lab',
            group='test',
            config=dict(
                batch_size=8,
                model_name='UNet3D',
                nb_blocks=2,
                bce_weight=0.5,
                scheduler_patience=5,
                learning_rate=0.0001,
                epochs=20,
                tile_size=TILE_SIZE,
            ),
        )
    else:
        wandb.init(project='vesuvius-challenge-ink-detection', entity='rosia-lab')

    main()
