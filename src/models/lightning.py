import os
import sys
parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl

from src.models.losses import BCEDiceWithLogitsLoss
from src.models.metrics import F05Score
from src.models.unet3d import Unet3d
from src.models.efficienunetv2 import EfficientUNetV2_L, EfficientUNetV2_M, EfficientUNetV2_S

import wandb


class LightningVesuvius(pl.LightningModule):
    def __init__(self, model_name, model_params, learning_rate, bce_weight, dice_threshold, val_fragment_shape):
        super().__init__()

        # Model
        if model_name == 'UNet3D':
            self.model = Unet3d(**model_params)
        elif model_name == 'EfficientUNetV2_L':
            self.model = EfficientUNetV2_L(**model_params)
        elif model_name == 'EfficientUNetV2_M':
            self.model = EfficientUNetV2_M(**model_params)
        elif model_name == 'EfficientUNetV2_S':
                self.model = EfficientUNetV2_S(**model_params)

        self.learning_rate = learning_rate
        self.criterion = BCEDiceWithLogitsLoss(bce_weight=bce_weight, dice_threshold=dice_threshold)
        self.metric = F05Score(val_fragment_shape)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.model(inputs)

        return x

    def training_step(self, batch, batch_idx):
        _, _, masks, images = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, masks)
        self.log('train/loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        fragments, bboxes, masks, images = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, masks)
        self.log('val/loss', loss, on_epoch=True)
        outputs = self.sigmoid(outputs)
        self.metric.update(fragments, bboxes, masks, outputs)

        return loss

    def on_validation_epoch_end(self) -> None:
        f05_threshold, f05_score, sub_f05_threshold, sub_f05_score, reconstructed_mask, predicted_mask = self.metric.compute()

        metrics = {
            'val/F05Threshold': f05_threshold,
            'val/F05Score': f05_score,
            'val/SubF05Threshold': sub_f05_threshold,
            'val/SubF05Score': sub_f05_score
        }

        self.log_dict(metrics, on_epoch=True)
        
        self.logger.log_image(key="samples", images=[reconstructed_mask, predicted_mask])

        self.metric.reset()

        return metrics

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        return optimizer
