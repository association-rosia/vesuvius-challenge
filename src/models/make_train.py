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

import src.constant as cst

import wandb


def main():
    torch.cuda.empty_cache()
    model = get_model()
    train_dataloader, val_dataloader = get_dataloaders()
    trainer = get_trainer()
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


def get_model():
    model_params = {}

    if wandb.config.model_name == 'UNet3D':
        model_params = {
            'num_blocks': wandb.config.num_blocks,
            'inputs_size': wandb.config.tile_size,
        }
    elif wandb.config.model_name == 'EfficientUNetV2':
        model_params = {
            'in_channels': wandb.config.num_slices,
        }

    lightning_model = LightningVesuvius(
        model_name=wandb.config.model_name,
        model_params=model_params,
        learning_rate=wandb.config.learning_rate,
        scheduler_patience=wandb.config.scheduler_patience,
        bce_weight=wandb.config.bce_weight,
        dice_threshold=wandb.config.dice_threshold,
        val_fragments_shape=get_fragments_shape(wandb.config.val_fragments, wandb.config.tile_size),
    )

    return lightning_model


def get_dataloaders():
    device = get_device()

    train_dataset = DatasetVesuvius(
        fragments=wandb.config.train_fragments,
        tile_size=wandb.config.tile_size,
        num_slices=wandb.config.num_slices,
        slices_list=[],
        start_slice=min(wandb.config.start_slice, 61),
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

    val_dataset = DatasetVesuvius(
        fragments=wandb.config.val_fragments,
        tile_size=wandb.config.tile_size,
        num_slices=wandb.config.num_slices,
        slices_list=wandb.config.slices_list,
        start_slice=min(wandb.config.start_slice, 61),
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


def get_trainer():
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        dirpath=cst.MODELS_DIR,
        filename=f'{wandb.run.name}-{wandb.run.id}',
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator='gpu',
        val_check_interval=0.25,
        max_epochs=wandb.config.epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        logger=WandbLogger(),
        precision=32
    )

    return trainer


if __name__ == '__main__':

    if sys.argv[1] == '--manual' or sys.argv[1] == '-m':
        wandb.init(
            project='vesuvius-challenge-ink-detection',
            entity='rosia-lab',
            group='EfficientUNetV2',
            config={
                'model_name': cst.MODEL_NAME,
                # 'num_blocks': cst.NUM_BLOCKS,
                'epochs': cst.EPOCHS,
                'batch_size': cst.BATCH_SIZE,
                'learning_rate': cst.LEARNING_RATE,
                'scheduler_patience': cst.SCHEDULER_PATIENCE,
                'bce_weight': cst.BCE_WEIGHT,
                'dice_threshold': cst.DICE_THRESHOLD,
                'tile_size': cst.TILE_SIZE,
                'num_slices': cst.NUM_SLICES,
                'reverse_slices': cst.REVERSE_SLICES,
                'start_slice': cst.START_SLICE,
                'selection_thr': cst.SELECTION_THR,
                'augmentation': cst.AUGMENTATION,
                'train_fragments': cst.TRAIN_FRAGMENTS,
                'val_fragments': cst.VAL_FRAGMENTS,
            },
        )
    else:
        wandb.init(project='vesuvius-challenge-ink-detection', entity='rosia-lab')

    main()
