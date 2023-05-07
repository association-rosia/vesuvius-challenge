import os, sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.models.losses import CombinedLoss
from src.models.models import LightningVesuvius, UNet3d
from src.data.make_dataset import CustomDataset, get_image_sizes
from src.utils import get_device

from constant import TRAIN_FRAGMENTS, VAL_FRAGMENTS, MODELS_PATH

import wandb

def main():
    wandb.init(
        project='vesuvius-challenge-ink-detection'
    )
    
    # empty the GPU cache
    torch.cuda.empty_cache()
    
    model = get_model()
    
    train_dataloader = DataLoader(
        dataset=CustomDataset(TRAIN_FRAGMENTS),
        batch_size=wandb.config.batch_size,
        shuffle=True,
        drop_last=True
        )
    
    val_dataloader = DataLoader(
        dataset=CustomDataset(VAL_FRAGMENTS),
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
    if wandb.config.model == 'UNet3d':
        num_block = wandb.config.num_block
        list_channels = [32 * i**2 for i in range(0, num_block + 1)]
        pytorch_model = UNet3d(list_channels)
    
    # get the device
    device = get_device()
    pytorch_model.to(device)
    
    learning_rate = wandb.config.learning_rate
    
    dice_weight = wandb.config.dice_weight
    criterion = CombinedLoss(dice_weight=dice_weight)
    
    scheduler_patience = wandb.config.scheduler_patience

    val_image_sizes = get_image_sizes(VAL_FRAGMENTS)
    
    lightning_model = LightningVesuvius(
        pytorch_model, 
        learning_rate,
        scheduler_patience,
        criterion, 
        val_image_sizes,
        )
    
    return lightning_model


def get_trainer():
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_F05Score",
        mode="max",
        dirpath= MODELS_PATH,
        filename="{val_F05Score}-{wandb.name}-{wandb.id}",
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
    main()