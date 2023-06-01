import os
import sys
parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

from src.models.losses import BCEDiceWithLogitsLoss
from src.models.metrics import F05Score
from src.models.unet3d import Unet3d


class LightningVesuvius(pl.LightningModule):
    def __init__(self,
                 model_name,
                 model_params,
                 learning_rate,
                 scheduler_patience,
                 bce_weight,
                 dice_threshold,
                 val_fragments_shape):
        super().__init__()

        # Model
        if model_name == 'UNet3D':
            self.pytorch_model = Unet3d(**model_params)

        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.criterion = BCEDiceWithLogitsLoss(bce_weight=bce_weight, dice_threshold=dice_threshold)
        self.metric = F05Score(val_fragments_shape)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.pytorch_model(inputs)
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
        self.log('val/loss', loss, on_step=True, on_epoch=True)
        outputs = self.sigmoid(outputs)
        self.metric.update(fragments, bboxes, masks, outputs)

        return loss

    def on_validation_epoch_end(self) -> None:
        # evaluate model on the validation dataset
        f05_threshold, f05_score, sub_f05_threshold, sub_f05_score = self.metric.compute()

        # self.best_f05_score = f05_score if self.best_f05_score is None else max(f05_score, self.best_f05_score)
        metrics = {
            'val/F05Threshold': f05_threshold,
            'val/F05Score': f05_score,
            'val/SubF05Threshold': sub_f05_threshold,
            'val/SubF05Score': sub_f05_score
        }

        # self.log('val/best_F05Score', self.best_f05_score, prog_bar=True)
        self.log_dict(metrics, on_step=True, on_epoch=True)
        self.metric.reset()

        return metrics

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=self.scheduler_patience, verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val/loss'}
