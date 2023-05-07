import os, sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.models.losses import CombinedLoss
from src.data.make_dataset import CustomDataset
from src.models.metrics import F05Score
from constant import TRAIN_FRAGMENTS, VAL_FRAGMENTS, Z_DIM


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        
        x = self.bn1(x)
        
        x = self.relu(x)
        
        x = self.conv2(x)
        
        x = self.bn2(x)
        
        x = self.relu(x)

        return x


class EncoderBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock3d(in_channels, out_channels)
        self.pool = nn.MaxPool3d((2, 2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class UNetEncoder3d(nn.Module):
    def __init__(self, list_channels):
        super().__init__()
        
        self.nb_block = len(list_channels) - 1 
        for i in range(self.nb_block):
            self.__setattr__(f'block_{i}', EncoderBlock3d(list_channels[i], list_channels[i+1]))
        
    def forward(self, inputs):
        list_skips = []
        x = inputs
        for i in range(self.nb_block):
            outputs, x = self.__getattr__(f'block_{i}')(x)
            list_skips.append(outputs)
        
        return x, list_skips


class DecoderBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock3d(out_channels+out_channels, out_channels)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x
    

class UNetDecoder3d(nn.Module):
    def __init__(self, list_channels):
        super().__init__()
        
        self.nb_block = len(list_channels) - 1
        for i in range(self.nb_block, -1, -1):
            self.__setattr__(f'block_{i - 1}', DecoderBlock3d(list_channels[i], list_channels[i-1]))
        
    def forward(self, inputs, list_skips):
        x = inputs
        
        for i in range(self.nb_block - 1, -1, -1):
            x = self.__getattr__(f'block_{i}')(x, list_skips[i])
            
        return x


class UNet3d(nn.Module):
    def __init__(self, list_channels, depth=64):
        super().__init__()
        
        # Architecture
        self.encoder = UNetEncoder3d(list_channels[:-1])

        self.bottleneck = ConvBlock3d(*list_channels[-2:])
        
        self.decoder = UNetDecoder3d(list_channels[1:])
        
        self.outputs3d = nn.Conv3d(list_channels[1], list_channels[0], kernel_size=1, padding=0)

        self.outputs2d = nn.MaxPool3d((depth, 1, 1))
        
    
    def forward(self, inputs):
        # Encoder
        x, list_skips = self.encoder(inputs)
        
        # Botteneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.decoder(x, list_skips)
        
        # Classifier 3D
        x = self.outputs3d(x)

        # Classifier 2D
        x = self.outputs2d(x)
        
        return x
    
    
class LightningVesuvius(pl.LightningModule):
    def __init__(
        self, model,
        learning_rate=0.0001,
        scheduler_patience=6,
        criterion=CombinedLoss(),
        val_image_sizes=[],
        ):
        super().__init__()
        
        # Model
        self.model = model
        
        # Training parameters
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.criterion = criterion
        self.metric = F05Score(val_image_sizes)
    
    
    def forward(self, inputs):
        x = self.model(inputs)
        return x
    
    
    def training_step(self, batch, batch_idx):
        inputs, masks, _, _ = batch
        
        # Forward pass
        outputs = self(inputs)

        loss = self.criterion(outputs, masks)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        inputs, masks, coords, indexes = batch
        
        # Forward pass
        outputs = self(inputs)
        loss = self.criterion(outputs, masks)
        self.log('val_loss', loss, prog_bar=True)
        
        # Update the evaluation metric
        self.metric.update(outputs, masks, coords, indexes)
        
        return loss
    
    
    def on_validation_end(self) -> None:
        # evaluate model on the validation dataset
        f05_score, sub_f05_score = self.metric.compute()
        self.best_f05_score = f05_score if self.best_f05_score is None else max(f05_score, self.best_f05_score)
        
        self.log('val_best_F05Score', self.best_f05_score, prog_bar=True)
        self.log('val_F05Score', f05_score)
        self.log('val_SubF05Score', sub_f05_score)
        return None
    
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=self.scheduler_patience, verbose=True)
        return [optimizer], [scheduler]
    

if __name__ == '__main__':
    trainer = pl.Trainer(fast_dev_run=True)
    
    train_dataloader = DataLoader(
        dataset=CustomDataset(TRAIN_FRAGMENTS),
        batch_size=8,
        shuffle=False,
        drop_last=True
        )
    
    val_dataloader = DataLoader(
        dataset=CustomDataset(VAL_FRAGMENTS),
        batch_size=8,
        shuffle=False,
        drop_last=True
        )
    
    model = UNet3d(list_channels=[1, 32, 64, 128], depth=Z_DIM)
    
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        )