import os, sys
# src_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
# sys.path.append(src_path)

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.models.losses import CombinedLoss
from src.data.make_dataset import CustomDataset
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


class UNet3d(pl.LightningModule):
    def __init__(
        self, list_channels, depth=64,
        learning_rate=0.1, 
        criterion=CombinedLoss(),
        batch_size=16,
        ):
        super().__init__()
        
        # Architecture
        self.encoder = UNetEncoder3d(list_channels[:-1])

        self.bottleneck = ConvBlock3d(*list_channels[-2:])
        
        self.decoder = UNetDecoder3d(list_channels[1:])
        
        self.outputs3d = nn.Conv3d(list_channels[1], list_channels[0], kernel_size=1, padding=0)

        self.outputs2d = nn.MaxPool3d((depth, 1, 1))
        
        # Training parameters
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.batch_size = batch_size
        
    
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
    
    
    def training_step(self, batch, batch_idx):
        inputs, masks = batch
        
        # Forward pass
        outputs = self(inputs)

        loss = self.criterion(outputs, masks)
        self.log('train_loss', loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        inputs, masks = batch
        
        # Forward pass
        outputs = self(inputs)
        loss = self.criterion(outputs, masks)
        self.log('val_loss', loss)
        return loss
    
    def on_validation_end(self) -> None:
        # evaluate model on the validation dataset
        # score = f05Score()
        # self.log('val_f05score': score)
        return None
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    

if __name__ == '__main__':
    trainer = pl.Trainer(fast_dev_run=True, accelerator='gpu', devices=2)
    
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
    
    model = UNet3d(list_channels=[1, 32, 64, 128], depth=Z_DIM).half()
    
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        )