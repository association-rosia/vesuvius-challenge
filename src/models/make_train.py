import os, sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.models.losses import CombinedLoss
from src.models.lightning import LightningVesuvius
from src.models.unet3d import UNet3d
from src.data.make_dataset import CustomDataset
from src.utils import get_dict_mask_shapes

from constant import TRAIN_FRAGMENTS, VAL_FRAGMENTS, MODELS_DIR, TILE_SIZE

import wandb


def main():
    wandb.init(
        project='vesuvius-challenge-ink-detection',
        entity='winged-bull'
    )
    
    # empty the GPU cache
    torch.cuda.empty_cache()
    
    model = get_model()
    
    train_dataloader = DataLoader(
        dataset=CustomDataset(TRAIN_FRAGMENTS, augmentation=True, test=False),
        batch_size=wandb.config.batch_size,
        shuffle=True,
        drop_last=True
        )
    
    val_dataloader = DataLoader(
        dataset=CustomDataset(VAL_FRAGMENTS, augmentation=False, test=False),
        batch_size=wandb.config.batch_size,
        shuffle=False,
        drop_last=True
        )
    
    trainer = get_trainer()
    
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        )


def get_model():
    if wandb.config.model_name == 'UNet3d':
        num_block = wandb.config.num_block
        model_parameters = dict(
            list_channels=[1] + [32 * 2**(i) for i in range(num_block)],
            inputs_size=TILE_SIZE,
        )
        
        # list_channels = [1] + [32 * 2**(i) for i in range(num_block)]
        # pytorch_model = UNet3d(list_channels, inputs_size=TILE_SIZE)
    
    # get the device
    # device = get_device()
    # pytorch_model#.to(device)
    
    # learning_rate = wandb.config.learning_rate
    
    # bce_weight = wandb.config.bce_weight
    # criterion = CombinedLoss(bce_weight=bce_weight)
    
    # scheduler_patience = wandb.config.scheduler_patience

    val_mask_shapes = get_dict_mask_shapes(VAL_FRAGMENTS)
    
    lightning_model = LightningVesuvius(
        model_name=wandb.config.model_name,
        model_parameters=model_parameters,
        learning_rate=wandb.config.learning_rate,
        scheduler_patience=wandb.config.scheduler_patience,
        bce_weight=wandb.config.bce_weight,
        f05score_threshold=wandb.config.f05score_threshold, 
        val_mask_shapes=get_dict_mask_shapes(VAL_FRAGMENTS),
        )#.to(device)
    
    return lightning_model


def get_trainer():
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/F05Score",
        mode="max",
        dirpath=MODELS_DIR,
        filename="{val/F05Score:.5f}-" + f"{wandb.run.name}-{wandb.run.id}"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # init the trainer
    trainer = pl.Trainer(
        max_epochs=wandb.config.epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        logger=WandbLogger(),
    )
    
    return trainer


if __name__ == '__main__':
    wandb.init(
        project='vesuvius-challenge-ink-detection',
        entity='winged-bull',
        group='test',
        config=dict(
            batch_size=16,
            model_name='UNet3d',
            num_block=2,
            bce_weight=1,
            scheduler_patience=3,
            learning_rate=0.0001,
            epochs=3,
            f05score_threshold=.5
        )
    )
    
    main()