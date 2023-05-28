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
    # empty the GPU cache
    torch.cuda.empty_cache()
    device = get_device()
    model = get_model()

    train_dataloader = DataLoader(
        dataset=DatasetVesuvius(fragments=TRAIN_FRAGMENTS,
                                tile_size=TILE_SIZE,
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
                                tile_size=TILE_SIZE,
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

    print('\n')

    trainer = get_trainer()

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def get_model():
    model_params = dict()

    if wandb.config.model_name == 'UNet3D':
        num_block = wandb.config.num_block
        model_params = dict(
            list_channels=[1] + [32 * 2 ** i for i in range(num_block)],
            inputs_size=TILE_SIZE,
        )

    lightning_model = LightningVesuvius(
        model_name=wandb.config.model_name,
        model_params=model_params,
        learning_rate=wandb.config.learning_rate,
        scheduler_patience=wandb.config.scheduler_patience,
        bce_weight=wandb.config.bce_weight,
        f05score_threshold=wandb.config.f05score_threshold,
        val_fragments_shape=get_fragments_shape(VAL_FRAGMENTS, TILE_SIZE),
    )

    return lightning_model


def get_trainer():
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val/F05Score',
        mode='max',
        dirpath=MODELS_DIR,
        filename='{val/F05Score:.5f}-' + f'{wandb.run.name}-{wandb.run.id}',
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # init the trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=wandb.config.epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        logger=WandbLogger(),
        precision='16-mixed',
    )

    return trainer


if __name__ == '__main__':

    if sys.argv[1] == '--manual' or sys.argv[1] == '-m':
        wandb.init(
            project='vesuvius-challenge-ink-detection',
            entity='rosia-lab',
            group='test',
            config=dict(
                batch_size=16,
                model_name='UNet3D',
                num_block=2,
                bce_weight=1,
                scheduler_patience=3,
                learning_rate=0.0001,
                epochs=3,
                f05score_threshold=0.5,
            ),
        )
    else:
        wandb.init(project='vesuvius-challenge-ink-detection', entity='rosia-lab')

    main()
