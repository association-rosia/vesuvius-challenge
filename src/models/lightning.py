import os, sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

from src.models.losses import CombinedLoss
from src.models.metrics import F05Score
from src.models.unet3d import UNet3d


class LightningVesuvius(pl.LightningModule):
    def __init__(
        self,
        model_name,
        model_parameters,
        learning_rate=0.0001,
        scheduler_patience=6,
        bce_weight=1,
        f05score_threshold=0.5,
        val_mask_shapes=None,
    ):
        super().__init__()

        # Model
        if model_name == "UNet3d":
            self.pytorch_model = UNet3d(**model_parameters)

        # Training parameters
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.criterion = CombinedLoss(bce_weight=bce_weight)
        self.metric = F05Score(val_mask_shapes, f05score_threshold)
        # self.submission = Submission(val_image_sizes)

    def forward(self, inputs):
        x = self.pytorch_model(inputs)
        return x

    def training_step(self, batch, batch_idx):
        _, inputs, masks, _ = batch
        
        # Forward pass
        outputs = self(inputs)

        loss = self.criterion(outputs, masks)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        fragments_ids, inputs, masks, coords = batch

        # Forward pass
        outputs = self(inputs)
        loss = self.criterion(outputs, masks)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Update the evaluation metric
        self.metric.update(outputs, masks, coords, fragments_ids)

        return loss

    def on_validation_epoch_end(self) -> None:
        # evaluate model on the validation dataset
        f05_score, sub_f05_score = self.metric.compute()
        # self.best_f05_score = f05_score if self.best_f05_score is None else max(f05_score, self.best_f05_score)

        metrics = {"val/F05Score": f05_score, "val/SubF05Score": sub_f05_score}

        # self.log('val/best_F05Score', self.best_f05_score, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True)

        return metrics

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer, patience=self.scheduler_patience, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }


if __name__ == "__main__":
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from src.data.make_dataset import CustomDataset
    from torch.utils.data import DataLoader
    from constant import MODELS_DIR, TILE_SIZE, TRAIN_FRAGMENTS, VAL_FRAGMENTS
    from src.utils import get_dict_mask_shapes

    import wandb

    wandb.init(
        project="vesuvius-challenge-ink-detection", group="test", entity="winged-bull"
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/F05Score",
        mode="max",
        dirpath=MODELS_DIR,
        filename="{val/F05Score:.5f}-test",
        auto_insert_metric_name=False,
    )

    logger = WandbLogger()

    # use 3 batches of train, 2 batches of val and test
    trainer = pl.Trainer(
        limit_train_batches=3,
        limit_val_batches=2,
        max_epochs=1,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    train_dataloader = DataLoader(
        dataset=CustomDataset(
            TRAIN_FRAGMENTS, test=False, augmentation=False
        ),
        batch_size=8,
        shuffle=False,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        dataset=CustomDataset(
            VAL_FRAGMENTS, test=False, augmentation=False
        ),
        batch_size=8,
        shuffle=False,
        drop_last=True,
    )

    val_mask_shapes = get_dict_mask_shapes(VAL_FRAGMENTS)

    model = LightningVesuvius(
        model_name="UNet3d",
        model_parameters=dict(list_channels=[1, 32, 64], inputs_size=TILE_SIZE),
        val_mask_shapes=val_mask_shapes,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
