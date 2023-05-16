import os, sys

parent = os.path.abspath(os.path.curdir)
sys.path.insert(1, parent)

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.models.losses import CombinedLoss
from src.data.make_dataset import CustomDataset
from src.models.metrics import F05Score
from src.models.submission import Submission
from src.models.unet3d import UNet3d
from constant import TRAIN_FRAGMENTS, VAL_FRAGMENTS, Z_DIM

 
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
        self.submission = Submission(val_image_sizes)
    
    
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
    
    
    def on_validation_epoch_end(self) -> None:
        # evaluate model on the validation dataset
        f05_score, sub_f05_score = self.metric.compute()
        self.best_f05_score = f05_score if self.best_f05_score is None else max(f05_score, self.best_f05_score)
        
        self.log('val_best_F05Score', self.best_f05_score, prog_bar=True)
        self.log('val_F05Score', f05_score)
        self.log('val_SubF05Score', sub_f05_score)
        return None
    
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=self.scheduler_patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    

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
    
    unet3D = UNet3d(list_channels=[1, 32, 64, 128], depth=Z_DIM)
    model = LightningVesuvius(
        model=unet3D
    )
    
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        )