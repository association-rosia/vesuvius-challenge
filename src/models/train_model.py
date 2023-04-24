from math import sqrt
import torch

from src.models.trainer import Trainer
from src.models.losses import CombinedLoss
from src.models.models import UNet3d

import wandb

def main():
    wandb.init(
        project='visuvius-ink-detection',
        group='Unet3d',
    )
    
    # empty the GPU cache
    torch.cuda.empty_cache()
    
    # get the device
    device = get_device()
    
    model = get_model(device)
    train_dataloader, val_dataloader = get_trainer_dataloaders(device)
    trainer = get_trainer(model, train_dataloader, val_dataloader)
    
    trainer.train()


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        raise Exception("None accelerator available")

    return device


def get_model(device):
    num_block = wandb.config.num_block
    list_channels = [32 * i**2 for i in range(0, num_block + 1)]
    model = UNet3d(list_channels)
    model.to(device)
    
    return model


def get_trainer_dataloaders(device):
    batch_size = wandb.config.batch_size
    return None


def get_trainer(model, train_dataloader, val_dataloader):
    dice_weight = wandb.config.dice_weight
    bce_weight = 1 - dice_weight
    criterion = CombinedLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           patience=wandb.config.scheduler_patience,
                                                           verbose=True)
    
    trainer_config = {
        'model': model,
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'epochs': wandb.config.epochs,
        'criterion': criterion,
        'scheduler': scheduler,
    }
    
    # init the trainer
    trainer = Trainer(**trainer_config)
    
    return trainer


if __name__ == '__main__':
    main()